# eka_eval/benchmarks/tasks/reasoning/arc_challenge.py

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
import argparse
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate
import gc

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_ARC = "allenai/ai2_arc"
DEFAULT_CONFIG_ARC_CHALLENGE = "ARC-Challenge" 
DEFAULT_SPLIT_ARC = "validation"
DEFAULT_MAX_NEW_TOKENS_ARC = 5
DEFAULT_GENERATION_BATCH_SIZE_ARC = 8

try:
    arc_ch_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for ARC-Challenge loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for ARC-Challenge: {e}. ARC-Challenge may not run correctly.", exc_info=True)
    arc_ch_accuracy_metric = None

def _format_arc_prompt(item: Dict) -> str:
    question = item.get('question', '')
    choices_dict = item.get('choices', {})
    choice_texts = choices_dict.get('text', [])
    choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))])
    prompt = (
        "Given the following question and multiple choices, select the best answer.\n"
        "Respond with ONLY the letter (e.g., A, B, C, D, or E) corresponding to the correct choice.\n\n"
        f"Question: {question}\n\nChoices:\n"
    )
    for label, text in zip(choice_labels, choice_texts):
        prompt += f"{label}. {text}\n"
    prompt += "\nAnswer:"
    return prompt

def _extract_arc_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    match = re.search(r'(?:[Aa]nswer[:\s]*)?\b([A-E])\b', completion_part)
    if match: return match.group(1).upper()
    simple_match = re.search(r'^[A-E]', completion_part, re.IGNORECASE)
    if simple_match: return simple_match.group(0).upper()
    logger.debug(f"ARC-Challenge: Could not extract A-E from: '{completion_part[:20]}'")
    return "X"

def _map_arc_answerkey_to_int(answer_key_str: str, choice_labels: List[str]) -> int:
    ak_str = str(answer_key_str).strip().upper()
    if ak_str in choice_labels: return choice_labels.index(ak_str)
    if ak_str.isdigit() and choice_labels and choice_labels[0].isalpha():
        num_val = int(ak_str)
        if 1 <= num_val <= len(choice_labels): return num_val - 1
    logger.warning(f"ARC-Challenge: Could not map answerKey '{ak_str}' with labels {choice_labels} to int.")
    return -1


def evaluate_arc_challenge(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_ARC,
    dataset_config_name: str = DEFAULT_CONFIG_ARC_CHALLENGE,
    dataset_split: str = DEFAULT_SPLIT_ARC,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_ARC,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_ARC,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if arc_ch_accuracy_metric is None:
        return {"ARC-Challenge": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting ARC-Challenge: {model_name_for_logging} on {dataset_name}/{dataset_config_name}")


    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"ARC-Challenge": 0.0, "error_message": f"DatasetLoadFailed ARC-Challenge: {e}"}
    logger.info(f"P{process_id}: Loaded ARC-Challenge '{dataset_name}/{dataset_config_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    if num_gpus > 1:
        total = len(full_data); per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    if len(subset_to_process) == 0: return {"ARC-Challenge": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} ARC-Challenge examples.")

    predictions_numeric, true_labels_numeric = [], []
    prompts_for_batch, original_items_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - ARC-Challenge Eval")):
        choice_labels_for_item = item_data.get('choices',{}).get('label',[])
        true_answer_key_str = str(item_data.get('answerKey','')).strip()
        true_label_idx = _map_arc_answerkey_to_int(true_answer_key_str, choice_labels_for_item)
        if true_label_idx == -1 :
            logger.warning(f"P{process_id}: Skipping ARC-Challenge item with unmappable answerKey '{true_answer_key_str}'. ID: {item_data.get('id')}")
            continue
        prompt_text = _format_arc_prompt(item_data)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {"do_sample": False, "max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id, "return_full_text": True}
            try:
                with torch.no_grad(): batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item = original_items_for_batch[k]
                    raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompts_for_batch[k] + "X"
                    pred_letter = _extract_arc_answer(raw_gen, prompts_for_batch[k])
                    pred_item_choice_labels = original_item.get('choices',{}).get('label',[])
                    pred_numeric = _map_arc_answerkey_to_int(pred_letter, pred_item_choice_labels)
                    true_item_choice_labels = original_item.get('choices',{}).get('label',[])
                    true_numeric_from_item = _map_arc_answerkey_to_int(str(original_item['answerKey']).strip(), true_item_choice_labels)
                    if pred_numeric == -1 and true_numeric_from_item != -1:
                        pred_numeric = (true_numeric_from_item + 1) % len(pred_item_choice_labels if pred_item_choice_labels else ["A","B","C","D"])
                    predictions_numeric.append(pred_numeric)
                    true_labels_numeric.append(true_numeric_from_item)
            except Exception as e_batch_arc_c:
                logger.error(f"P{process_id}: Error in ARC-Challenge gen batch: {e_batch_arc_c}", exc_info=True)
                for item_err_info in original_items_for_batch:
                     true_num_err = _map_arc_answerkey_to_int(str(item_err_info['answerKey']).strip(), item_err_info.get('choices',{}).get('label',[]))
                     if true_num_err == -1: true_num_err = 0
                     predictions_numeric.append((true_num_err + 1) % len(item_err_info.get('choices',{}).get('label',["A","B","C","D"])))
                     true_labels_numeric.append(true_num_err)
            prompts_for_batch, original_items_for_batch = [], []
    
    if not true_labels_numeric: return {"ARC-Challenge": 0.0}
    acc_score = 0.0
    try:
        valid_indices = [i for i, ref in enumerate(true_labels_numeric) if ref != -1]
        if not valid_indices: logger.warning(f"P{process_id}: No valid reference labels for ARC-Challenge.")
        else:
            valid_preds = [predictions_numeric[i] for i in valid_indices]
            valid_refs = [true_labels_numeric[i] for i in valid_indices]
            if valid_preds and valid_refs:
                acc_results = arc_ch_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
                acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric: logger.error(f"P{process_id}: Error computing ARC-Challenge accuracy: {e_metric}")
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final ARC-Challenge Acc: {acc_score:.2f}% on {len(valid_refs) if 'valid_refs' in locals() and valid_refs else 0} examples.")
    return {"ARC-Challenge": acc_score}

# Standalone test block
if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser_arc_c = argparse.ArgumentParser(description="Standalone Test ARC-Challenge")
    test_parser_arc_c.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser_arc_c.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser_arc_c.add_argument("--gen_batch_size_test", type=int, default=2)
    arc_c_args = test_parser_arc_c.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="ARCChFileTest")
    logger.info(f"--- Standalone ARC-Challenge Test: {arc_c_args.model_name_test} ---")
    arc_c_pipe, _ = initialize_model_pipeline(arc_c_args.model_name_test, target_device_id=0)
    if arc_c_pipe:
        arc_c_eval_args = {
            "pipe": arc_c_pipe, "tokenizer": arc_c_pipe.tokenizer, "model_name_for_logging": arc_c_args.model_name_test,
            "device": arc_c_pipe.device, "dataset_split": arc_c_args.dataset_split_test,
            "generation_batch_size": arc_c_args.gen_batch_size_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_arc_challenge(**arc_c_eval_args), indent=2))
        finally: cleanup_model_resources(arc_c_pipe, getattr(arc_c_pipe, 'model', None))