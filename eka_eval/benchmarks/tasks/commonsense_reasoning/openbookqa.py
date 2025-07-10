import torch
import re
import sys
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate
import gc

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_OBQA = "allenai/openbookqa"
DEFAULT_CONFIG_OBQA = "main"
DEFAULT_SPLIT_OBQA = "validation"
DEFAULT_MAX_NEW_TOKENS_OBQA = 5

try:
    obqa_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for OpenBookQA loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for OBQA: {e}. OBQA may not run correctly.", exc_info=True)
    obqa_accuracy_metric = None

def _format_openbookqa_prompt(item: Dict) -> str:
    question_stem = item.get('question_stem', '')
    choices_dict = item.get('choices', {})
    choice_texts = choices_dict.get('text', [])
    choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))])
    prompt = (
        "Answer the following multiple-choice question based on general knowledge and the provided context if any. "
        "Respond with ONLY the letter (e.g., A, B, C, D) of the correct choice.\n\n"
        f"Question: {question_stem}\n"
        "Choices:\n"
    )
    for label, text in zip(choice_labels, choice_texts):
        prompt += f"{label}. {text}\n"
    prompt += "Answer:"
    return prompt

def _extract_openbookqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    match = re.search(r'(?:[Aa]nswer[:\s]*)?\b([A-D])\b', completion_part)
    if match: return match.group(1).upper()
    simple_match = re.search(r'^[A-D]', completion_part, re.IGNORECASE)
    if simple_match: return simple_match.group(0).upper()
    logger.debug(f"OBQA: Could not extract A-D from: '{completion_part[:20]}'")
    return "X"

def evaluate_openbookqa(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_OBQA,
    dataset_config_name: str = DEFAULT_CONFIG_OBQA,
    dataset_split: str = DEFAULT_SPLIT_OBQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_OBQA,
    generation_batch_size: int = 8,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if obqa_accuracy_metric is None:
        return {"OpenBookQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting OpenBookQA: {model_name_for_logging} on {dataset_name}/{dataset_config_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Split='{dataset_split}', GenBatchSize={generation_batch_size}")

    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"OpenBookQA": 0.0, "error_message": f"DatasetLoadFailed OBQA: {e}"}
    logger.info(f"P{process_id}: Loaded OBQA '{dataset_name}/{dataset_config_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    if num_gpus > 1:
        total = len(full_data)
        per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1:
            end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data

    if len(subset_to_process) == 0:
        return {"OpenBookQA": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} OBQA examples.")

    predictions_numeric, true_labels_numeric = [], []
    label_to_int = {chr(65 + i): i for i in range(4)}

    prompts_for_batch, original_items_for_batch = [], []
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - OBQA Eval")):
        true_answer_letter = item_data.get('answerKey', '').strip()
        if not true_answer_letter or true_answer_letter not in label_to_int:
            logger.warning(f"P{process_id}: Skipping OBQA item with invalid answerKey '{true_answer_letter}'. ID: {item_data.get('id')}")
            continue

        prompt_text = _format_openbookqa_prompt(item_data)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config_obqa = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config_obqa)
                    for k, raw_out_list in enumerate(batch_raw_outputs):
                        original_item = original_items_for_batch[k]
                        raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompts_for_batch[k] + "X"
                        pred_letter = _extract_openbookqa_answer(raw_gen, prompts_for_batch[k])
                        pred_num = label_to_int.get(pred_letter, -1)
                        true_num = label_to_int.get(original_item['answerKey'].strip(), -1)
                        if pred_num == -1 and true_num != -1:
                            pred_num = (true_num + 1) % 4
                        predictions_numeric.append(pred_num)
                        true_labels_numeric.append(true_num)
            except Exception as e_batch_obqa:
                logger.error(f"P{process_id}: Error in OBQA gen batch: {e_batch_obqa}", exc_info=True)
                for item_err_info in original_items_for_batch:
                    true_num_err = label_to_int.get(item_err_info['answerKey'].strip(), 0)
                    predictions_numeric.append((true_num_err + 1) % 4)
                    true_labels_numeric.append(true_num_err)
            prompts_for_batch, original_items_for_batch = [], []

    if not true_labels_numeric:
        return {"OpenBookQA": 0.0}

    acc_score = 0.0
    try:
        valid_preds = [p for i, p in enumerate(predictions_numeric) if true_labels_numeric[i] != -1]
        valid_refs = [r for r in true_labels_numeric if r != -1]
        if valid_preds and valid_refs:
            acc_results = obqa_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
            acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric:
        logger.error(f"P{process_id}: Error computing OBQA accuracy: {e_metric}")
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final OpenBookQA Acc: {acc_score:.2f}% on {len(valid_refs)} examples.")
    return {"OpenBookQA": acc_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser = argparse.ArgumentParser(description="Standalone Test OpenBookQA")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    ob_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="OBQAFileTest")
    logger.info(f"--- Standalone OBQA Test: {ob_args.model_name_test} ---")
    ob_pipe, _ = initialize_model_pipeline(ob_args.model_name_test, target_device_id=0)
    if ob_pipe:
        ob_eval_args = {
            "pipe": ob_pipe,
            "tokenizer": ob_pipe.tokenizer,
            "model_name_for_logging": ob_args.model_name_test,
            "device": ob_pipe.device,
            "dataset_split": ob_args.dataset_split_test,
            "generation_batch_size": ob_args.gen_batch_size_test,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1
        }
        try:
            print(json.dumps(evaluate_openbookqa(**ob_eval_args), indent=2))
        finally:
            cleanup_model_resources(ob_pipe, getattr(ob_pipe, 'model', None))
