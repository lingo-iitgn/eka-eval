# eka_eval/benchmarks/tasks/reasoning/siqa.py

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

# --- Constants and Helper Functions for SIQA ---
DEFAULT_DATASET_NAME_SIQA = "allenai/social_i_qa"
DEFAULT_SPLIT_SIQA = "validation"
DEFAULT_MAX_NEW_TOKENS_SIQA = 5 # For "1", "2", or "3"
DEFAULT_GENERATION_BATCH_SIZE_SIQA = 8

try:
    siqa_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for SIQA loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for SIQA: {e}. SIQA may not run correctly.", exc_info=True)
    siqa_accuracy_metric = None

def _format_siqa_prompt(item: Dict) -> str:
    context = item.get('context', '')
    question = item.get('question', '')
    # SIQA has answerA, answerB, answerC
    ans_a = item.get('answerA', '')
    ans_b = item.get('answerB', '')
    ans_c = item.get('answerC', '')
    
    prompt = (
        "Given the context and question, choose the most appropriate answer (1, 2, or 3).\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        f"Options:\n"
        f"1. {ans_a}\n"
        f"2. {ans_b}\n"
        f"3. {ans_c}\n\n"
        "Your answer must be exactly 1, 2, or 3.\nAnswer:"
    )
    return prompt

def _extract_siqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    match = re.search(r'^\s*\b([1-3])\b', completion_part) # Look for 1, 2, or 3 at the start
    if match:
        return match.group(1)
    logger.debug(f"SIQA: Could not extract 1-3 from completion: '{completion_part[:20]}'")
    return "X"

# --- Main Evaluation Function ---
def evaluate_siqa(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_SIQA,
    dataset_split: str = DEFAULT_SPLIT_SIQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_SIQA,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_SIQA,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if siqa_accuracy_metric is None:
        return {"SIQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting SIQA: {model_name_for_logging} on {dataset_name}")
    # ... (logging and dataset loading similar to PIQA, using SIQA constants) ...
    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"SIQA": 0.0, "error_message": f"DatasetLoadFailed SIQA: {e}"}
    logger.info(f"P{process_id}: Loaded SIQA '{dataset_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    if num_gpus > 1: # Data splitting
        total = len(full_data); per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    if len(subset_to_process) == 0: return {"SIQA": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} SIQA examples.")

    predictions_numeric, true_labels_numeric = [], []
    prompts_for_batch, original_items_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - SIQA Eval")):
        true_label_str = str(item_data.get('label','')).strip() # SIQA label is '1', '2', '3' (strings)
        if true_label_str not in ['1', '2', '3']:
            logger.warning(f"P{process_id}: Skipping SIQA item with invalid label '{true_label_str}'. Context: {item_data.get('context','')[:50]}")
            continue
        
        prompt_text = _format_siqa_prompt(item_data)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {"do_sample": False, "max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id, "return_full_text": True}
            try:
                with torch.no_grad(): batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item = original_items_for_batch[k]
                    raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompts_for_batch[k] + "X"
                    pred_str = _extract_siqa_answer(raw_gen, prompts_for_batch[k])
                    pred_num = int(pred_str) if pred_str in ["1", "2", "3"] else -1
                    true_num = int(original_item['label']) # Already 1, 2, or 3
                    if pred_num == -1: pred_num = (true_num % 3) + 1 # Assign a different wrong answer (1->2, 2->3, 3->1)
                    
                    predictions_numeric.append(pred_num)
                    true_labels_numeric.append(true_num)
            except Exception as e_batch_siqa:
                logger.error(f"P{process_id}: Error in SIQA gen batch: {e_batch_siqa}", exc_info=True)
                for item_err_info in original_items_for_batch:
                     true_num_err = int(item_err_info['label'])
                     predictions_numeric.append((true_num_err % 3) + 1); true_labels_numeric.append(true_num_err)
            prompts_for_batch, original_items_for_batch = [], []

    if not true_labels_numeric: return {"SIQA": 0.0}
    acc_score = 0.0
    try:
        # Filter out any invalid labels if necessary, though SIQA labels should be clean
        valid_indices = [i for i, ref in enumerate(true_labels_numeric) if ref in [1,2,3]]
        if not valid_indices: logger.warning(f"P{process_id}: No valid reference labels for SIQA.")
        else:
            valid_preds = [predictions_numeric[i] for i in valid_indices]
            valid_refs = [true_labels_numeric[i] for i in valid_indices]
            if valid_preds and valid_refs:
                acc_results = siqa_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
                acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric: logger.error(f"P{process_id}: Error computing SIQA accuracy: {e_metric}")
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final SIQA Accuracy: {acc_score:.2f}% on {len(valid_refs) if 'valid_refs' in locals() and valid_refs else 0} examples.")
    return {"SIQA": acc_score}

# Standalone test block
if __name__ == '__main__':
    # ... (Similar standalone test block as PIQA/Winogrande, adjusting names and dataset_name)
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser = argparse.ArgumentParser(description="Standalone Test SIQA")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    si_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="SIQAFileTest")
    logger.info(f"--- Standalone SIQA Test: {si_args.model_name_test} ---")
    si_pipe, _ = initialize_model_pipeline(si_args.model_name_test, target_device_id=0)
    if si_pipe:
        si_eval_args = {
            "pipe": si_pipe, "tokenizer": si_pipe.tokenizer, "model_name_for_logging": si_args.model_name_test,
            "device": si_pipe.device, "dataset_split": si_args.dataset_split_test,
            "generation_batch_size": si_args.gen_batch_size_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_siqa(**si_eval_args), indent=2))
        finally: cleanup_model_resources(si_pipe, getattr(si_pipe, 'model', None))