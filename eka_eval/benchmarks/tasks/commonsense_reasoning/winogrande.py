# eka_eval/benchmarks/tasks/reasoning/winogrande.py

import torch
import sys
import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_WINO = "winogrande"
DEFAULT_CONFIG_WINO = "winogrande_xl" 
DEFAULT_SPLIT_WINO = "validation"
DEFAULT_MAX_NEW_TOKENS_WINO = 5 
DEFAULT_CHECKPOINT_DIR_WINO = "checkpoints/winogrande_checkpoints"

try:
    wino_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for Winogrande loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for Winogrande: {e}. Winogrande will not run correctly.", exc_info=True)
    wino_accuracy_metric = None

def _format_winogrande_prompt(item: Dict) -> str:
    """Formats a Winogrande problem into a prompt."""
    sentence = item.get('sentence', '').replace('_', '_____') 
    option1 = item.get('option1', '')
    option2 = item.get('option2', '')
    
    prompt = (
        "Given the sentence with a blank, choose the correct option (1 or 2) to fill in the blank.\n\n"
        f"Sentence: {sentence}\n\n"
        f"Options:\n1) {option1}\n2) {option2}\n\n"
        "Your answer must be exactly 1 or 2.\nAnswer:"
    )
    return prompt

def _extract_winogrande_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    """Extracts the predicted answer (1 or 2) from the generated text."""
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    match = re.search(r'^\s*\b(1|2)\b', completion_part)
    if match:
        return match.group(1)
    logger.debug(f"Wino: Could not extract 1 or 2 from start of completion: '{completion_part[:20]}'")
    return "X"

def evaluate_winogrande(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_WINO,
    dataset_config_name: str = DEFAULT_CONFIG_WINO,
    dataset_split: str = DEFAULT_SPLIT_WINO,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_WINO,
    generation_batch_size: int = 8,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if wino_accuracy_metric is None:
        return {"WinoGrande": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting Winogrande: {model_name_for_logging} on {dataset_name}/{dataset_config_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Params: split='{dataset_split}', gen_batch_size={generation_batch_size}")

    try:
        full_data_for_split = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"WinoGrande": 0.0, "error_message": f"DatasetLoadFailed Wino: {e}"}
    logger.info(f"P{process_id}: Loaded Winogrande '{dataset_name}/{dataset_config_name}' (split: '{dataset_split}') with {len(full_data_for_split)} examples.")

    if num_gpus > 1:
        total_examples = len(full_data_for_split)
        examples_per_instance = total_examples // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1: end_idx = total_examples
        dataset_subset_to_process = full_data_for_split.select(range(start_idx, end_idx))
    else:
        dataset_subset_to_process = full_data_for_split
        
    if len(dataset_subset_to_process) == 0: return {"WinoGrande": 0.0}

    predictions_numeric, true_labels_numeric = [], []
    prompts_for_batch, infos_for_batch = [], []
    for item_idx, item_data in enumerate(tqdm(dataset_subset_to_process, desc=f"P{process_id} - Wino Eval")):
        true_label_str = item_data.get('answer')
        if true_label_str not in ['1', '2']:
            logger.warning(f"P{process_id}: Skipping Winogrande item with invalid answer/label '{true_label_str}'. Item: {item_data.get('sentence','')[:50]}")
            continue

        prompt_text = _format_winogrande_prompt(item_data)
        prompts_for_batch.append(prompt_text)
        infos_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(dataset_subset_to_process) - 1:
            generation_config_wino = {
                "do_sample": False, "temperature": 0.0, "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id, "return_full_text": True
            }
            try:
                with torch.no_grad():
                     batch_raw_outputs = pipe(prompts_for_batch, **generation_config_wino)
                for k, raw_output_list in enumerate(batch_raw_outputs):
                    original_item = infos_for_batch[k]
                    raw_gen = raw_output_list[0]['generated_text'] if raw_output_list and raw_output_list[0] else prompts_for_batch[k] + "X"
                    pred_str = _extract_winogrande_answer(raw_gen, prompts_for_batch[k])
                    pred_numeric = int(pred_str) if pred_str in ["1", "2"] else -1 
                    true_numeric = int(original_item['answer']) 

                    if pred_numeric == -1: pred_numeric = 3 - true_numeric 
                    
                    predictions_numeric.append(pred_numeric)
                    true_labels_numeric.append(true_numeric)
            except Exception as e_b_wino:
                logger.error(f"P{process_id}: Error in Wino gen batch: {e_b_wino}", exc_info=True)
                for _ in range(len(prompts_for_batch)): predictions_numeric.append(0); true_labels_numeric.append(1) # Mark as wrong
            prompts_for_batch, infos_for_batch = [], []

    if not true_labels_numeric: return {"WinoGrande": 0.0}
    accuracy_score = 0.0
    try:
        accuracy_results = wino_accuracy_metric.compute(predictions=predictions_numeric, references=true_labels_numeric)
        accuracy_score = accuracy_results.get("accuracy", 0.0) * 100
    except Exception as e_metric: logger.error(f"P{process_id}: Error computing Wino accuracy: {e_metric}", exc_info=True)

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final Winogrande Accuracy: {accuracy_score:.2f}% on {len(true_labels_numeric)} examples.")
    return {"WinoGrande": accuracy_score} 
if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser_wino = argparse.ArgumentParser(description="Standalone Test Winogrande")
    test_parser_wino.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser_wino.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser_wino.add_argument("--dataset_config_test", type=str, default="winogrande_xs") # Use xs for quick test
    test_parser_wino.add_argument("--gen_batch_size_test", type=int, default=2)
    w_args = test_parser_wino.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="WinoFileTest")
    logger.info(f"--- Standalone Winogrande Test: {w_args.model_name_test} ---")
    w_pipe, _ = initialize_model_pipeline(w_args.model_name_test, target_device_id=0)
    if w_pipe:
        w_eval_args = {
            "pipe": w_pipe, "tokenizer": w_pipe.tokenizer, "model_name_for_logging": w_args.model_name_test,
            "device": w_pipe.device, "dataset_config_name": w_args.dataset_config_test,
            "dataset_split": w_args.dataset_split_test, "generation_batch_size": w_args.gen_batch_size_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_winogrande(**w_eval_args), indent=2))
        finally: cleanup_model_resources(w_pipe, getattr(w_pipe, 'model', None))
    else: logger.error(f"Failed to init model {w_args.model_name_test} for Winogrande test.")