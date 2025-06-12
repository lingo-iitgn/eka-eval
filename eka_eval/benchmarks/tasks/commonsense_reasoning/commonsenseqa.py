# eka_eval/benchmarks/tasks/reasoning/commonsenseqa.py

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
import gc

logger = logging.getLogger(__name__)

# --- Constants and Helper Functions for CommonsenseQA ---
DEFAULT_DATASET_NAME_CSQA = "commonsense_qa" # HF dataset name is "commonsense_qa"
DEFAULT_SPLIT_CSQA = "validation"
DEFAULT_MAX_NEW_TOKENS_CSQA = 5 # For "A", "B", etc.
DEFAULT_FEW_SHOT_COUNT_CSQA = 7 # As per your original config "7-shot"

try:
    csqa_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for CommonsenseQA loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for CSQA: {e}. CSQA may not run correctly.", exc_info=True)
    csqa_accuracy_metric = None

# Hardcoded few-shot examples for CommonsenseQA (can be expanded or loaded from a file)
# These should ideally be diverse and high-quality.
DEFAULT_FEW_SHOT_EXAMPLES_CSQA = [
    {"question": "What do people use to absorb body odor?", "choices": {"label": ["A", "B", "C"], "text": ["deodorant", "perfume", "cologne"]}, "answerKey": "A"},
    {"question": "Where would you find a towel in a house?", "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["kitchen", "bedroom", "bathroom", "living room", "garage"]}, "answerKey": "C"},
    {"question": "What is the primary purpose of a refrigerator?", "choices": {"label": ["A", "B", "C"], "text": ["heating food", "storing clothes", "keeping food cold"]}, "answerKey": "C"},
    {"question": "If you want to write a letter, what tool would you primarily use?", "choices": {"label": ["A", "B", "C", "D"], "text": ["hammer", "pen", "saw", "wrench"]}, "answerKey": "B"},
    {"question": "What object is typically used for drinking water?", "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["plate", "bowl", "fork", "cup", "knife"]}, "answerKey": "D"},
    {"question": "To see in the dark, you would most likely use a?", "choices": {"label": ["A", "B", "C"], "text": ["sunglasses", "radio", "flashlight"]}, "answerKey": "C"},
    {"question": "What do you use to clean your teeth?", "choices": {"label": ["A", "B", "C", "D"], "text": ["soap", "shampoo", "toothbrush", "towel"]}, "answerKey": "C"}
]


def _format_commonsenseqa_prompt(item: Dict, few_shot_examples: List[Dict]) -> str:
    question = item.get('question', '')
    choices_dict = item.get('choices', {})
    choice_texts = choices_dict.get('text', [])
    choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))]) # Default A, B, C..

    prompt = "Answer the following multiple-choice question by providing ONLY the letter (e.g., A, B, C, D, or E) of the correct answer. Here are some examples:\n\n"
    
    for ex_item in few_shot_examples:
        ex_q = ex_item.get('question', '')
        ex_choices_dict = ex_item.get('choices', {})
        ex_choice_texts = ex_choices_dict.get('text', [])
        ex_choice_labels = ex_choices_dict.get('label', [chr(65+i) for i in range(len(ex_choice_texts))])
        ex_ans_key = ex_item.get('answerKey', '')

        prompt += f"Q: {ex_q}\n"
        for label, text in zip(ex_choice_labels, ex_choice_texts):
            prompt += f"{label}. {text}\n"
        prompt += f"A: {ex_ans_key}\n\n"
    
    prompt += "Now, answer this question:\n"
    prompt += f"Q: {question}\n"
    for label, text in zip(choice_labels, choice_texts):
        prompt += f"{label}. {text}\n"
    prompt += "A:" # Model should output A, B, C, D, or E
    return prompt

def _extract_commonsenseqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    # Robustly find a single uppercase letter A-E, possibly after "Answer: " or similar
    match = re.search(r'(?:[Aa]nswer[:\s]*)?\b([A-E])\b', completion_part)
    if match:
        return match.group(1).upper()
    # Fallback: first letter if it's A-E
    simple_match = re.search(r'^[A-E]', completion_part, re.IGNORECASE)
    if simple_match:
        return simple_match.group(0).upper()
    logger.debug(f"CSQA: Could not extract A-E from completion: '{completion_part[:20]}'")
    return "X"

# --- Main Evaluation Function ---
def evaluate_commonsenseqa(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_CSQA,
    dataset_split: str = DEFAULT_SPLIT_CSQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_CSQA,
    generation_batch_size: int = 8,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_CSQA, # Number of few-shot examples to use
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if csqa_accuracy_metric is None:
        return {"CommonSenseQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting CommonsenseQA ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Split='{dataset_split}', GenBatchSize={generation_batch_size}")

    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"CommonSenseQA": 0.0, "error_message": f"DatasetLoadFailed CSQA: {e}"}
    logger.info(f"P{process_id}: Loaded CSQA '{dataset_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    if num_gpus > 1:
        # ... (Data splitting logic) ...
        total = len(full_data); per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    if len(subset_to_process) == 0: return {"CommonSenseQA": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} CSQA examples.")

    # Prepare few-shot examples (use a fixed set for consistency across runs/shards)
    # Or, if you want to sample from the 'train' split:
    # few_shot_data = load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(num_few_shot))
    # few_shot_examples_list = list(few_shot_data)
    few_shot_examples_list = DEFAULT_FEW_SHOT_EXAMPLES_CSQA[:num_few_shot] if num_few_shot > 0 else []


    predictions_numeric, true_labels_numeric = [], []
    label_to_int = {chr(65 + i): i for i in range(5)} # A=0, B=1, ...

    prompts_for_batch, original_items_for_batch = [], []
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - CSQA Eval")):
        true_answer_letter = item_data.get('answerKey')
        if not true_answer_letter or true_answer_letter not in label_to_int:
            logger.warning(f"P{process_id}: Skipping CSQA item with invalid answerKey '{true_answer_letter}'. Item ID: {item_data.get('id')}")
            continue
        
        prompt_text = _format_commonsenseqa_prompt(item_data, few_shot_examples_list)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config_csqa = {"do_sample": False, "max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id, "return_full_text": True}
            try:
                with torch.no_grad(): batch_raw_outputs = pipe(prompts_for_batch, **gen_config_csqa)
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item = original_items_for_batch[k]
                    raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompts_for_batch[k] + "X"
                    pred_letter = _extract_commonsenseqa_answer(raw_gen, prompts_for_batch[k])
                    pred_num = label_to_int.get(pred_letter, -1) # Convert 'A'->0, 'B'->1 etc.
                    true_num = label_to_int.get(original_item['answerKey'], -1)
                    if pred_num == -1 and true_num != -1: pred_num = (true_num + 1) % 5 # Mark unparseable as wrong, pick a different valid index
                    
                    predictions_numeric.append(pred_num)
                    true_labels_numeric.append(true_num)
            except Exception as e_batch_csqa:
                logger.error(f"P{process_id}: Error in CSQA gen batch: {e_batch_csqa}", exc_info=True)
                for item_err_info in original_items_for_batch:
                     true_num_err = label_to_int.get(item_err_info['answerKey'],0) # Default true to 0 for error pred
                     predictions_numeric.append((true_num_err + 1) % 5) # Mark as wrong
                     true_labels_numeric.append(true_num_err)
            prompts_for_batch, original_items_for_batch = [], []

    if not true_labels_numeric: return {"CommonSenseQA": 0.0}
    acc_score = 0.0
    try:
        # Filter out any -1 labels if they occurred (shouldn't for CSQA if data is clean)
        valid_preds = [p for i, p in enumerate(predictions_numeric) if true_labels_numeric[i] != -1]
        valid_refs = [r for r in true_labels_numeric if r != -1]
        if valid_preds and valid_refs:
            acc_results = csqa_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
            acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric: logger.error(f"P{process_id}: Error computing CSQA accuracy: {e_metric}")
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final CommonSenseQA Acc: {acc_score:.2f}% on {len(valid_refs)} examples.")
    return {"CommonSenseQA": acc_score} # Main key


# Standalone test block
if __name__ == '__main__':
    # ... (Similar standalone test block as PIQA/Winogrande, adjusting dataset_name and task-specific args)
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser = argparse.ArgumentParser(description="Standalone Test CommonsenseQA")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser.add_argument("--num_few_shot_test", type=int, default=3) # Test with fewer few-shot
    cs_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="CSQAFileTest")
    logger.info(f"--- Standalone CSQA Test: {cs_args.model_name_test} ({cs_args.num_few_shot_test}-shot) ---")
    cs_pipe, _ = initialize_model_pipeline(cs_args.model_name_test, target_device_id=0)
    if cs_pipe:
        cs_eval_args = {
            "pipe": cs_pipe, "tokenizer": cs_pipe.tokenizer, "model_name_for_logging": cs_args.model_name_test,
            "device": cs_pipe.device, "dataset_split": cs_args.dataset_split_test,
            "generation_batch_size": cs_args.gen_batch_size_test,
            "num_few_shot": cs_args.num_few_shot_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_commonsenseqa(**cs_eval_args), indent=2))
        finally: cleanup_model_resources(cs_pipe, getattr(cs_pipe, 'model', None))