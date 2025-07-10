# eka_eval/benchmarks/tasks/reasoning/hellaswag.py

import torch
import sys
import argparse 
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate
import gc
from datetime import datetime

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME_HELLASWAG = "hellaswag"
DEFAULT_SPLIT_HELLASWAG = "validation"
DEFAULT_MAX_NEW_TOKENS_HELLASWAG = 5 
DEFAULT_FEW_SHOT_COUNT_HELLASWAG = 10 
DEFAULT_GENERATION_BATCH_SIZE_HELLASWAG = 8
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "hellaswag_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "hellaswag_10shot"
PROMPT_FILE_BENCHMARK_KEY = "hellaswag"
PROMPT_FILE_CATEGORY = "reasoning"

try:
    hellaswag_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for HellaSwag loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for HellaSwag: {e}. HellaSwag may not run correctly.", exc_info=True)
    hellaswag_accuracy_metric = None

def save_detailed_hellaswag_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    num_few_shot: int,
    accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed HellaSwag results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hellaswag_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_few_shot": num_few_shot,
        "total_questions": len(results_data),
        "correct_answers": sum(1 for r in results_data if r["is_correct"]),
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id,
        "generation_failures": sum(1 for r in results_data if not r.get("extraction_successful", True))
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed HellaSwag results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed HellaSwag results: {e}")
        return ""

def _get_hellaswag_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_hellaswag",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for HellaSwag.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_hellaswag from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def preprocess(text):
    """Preprocess text following official HellaSwag format"""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs_hellaswag(item: Dict) -> Dict:
    """Process raw HellaSwag item following official format"""
    ctx = item["ctx_a"] + " " + item["ctx_b"].capitalize()
    out_doc = {
        "query": preprocess(item["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in item["endings"]],
        "gold": int(item["label"]),
    }
    return out_doc

def _prepare_hellaswag_choices_data(choices: List[str]) -> Dict[str, str]:
    """Prepare choices data for template formatting."""
    while len(choices) < 4:
        choices.append("")
    
    return {
        "choiceA": choices[0],
        "choiceB": choices[1],
        "choiceC": choices[2],
        "choiceD": choices[3]
    }

def _extract_hellaswag_answer(generated_text: str, prompt_text_sent_to_llm: Optional[str] = None) -> Optional[str]:
    """Extract answer from generated text, looking for A, B, C, or D with enhanced debugging"""
    completion_part = generated_text
    if prompt_text_sent_to_llm and generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    
    logger.debug(f"HellaSwag extraction input: '{completion_part[:100]}'")
    
    if len(completion_part) == 0:
        logger.debug("HellaSwag: Empty completion, extraction failed")
        return None
    
    patterns = [
        r'^([A-D])$',                          # Exact single letter
        r'^([A-D])\b',                         # Letter at start
        r'^([A-D])[\.\)\:]',                   # Letter with punctuation
        r'(?:Answer|answer)[:\s]*([A-D])\b',   # After "Answer:"
        r'\b([A-D])\b',                        # Any isolated letter
        r'([A-D])'                             # Letter anywhere
    ]
    
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, completion_part, re.IGNORECASE)
        if matches:
            letter = matches[0].upper()
            logger.debug(f"HellaSwag: Extracted '{letter}' using pattern {i+1}: {pattern}")
            return letter
    
    logger.debug(f"HellaSwag: No pattern matched for: '{completion_part[:50]}'")
    return None

# --- Main Evaluation Function ---
def evaluate_hellaswag(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_HELLASWAG,
    dataset_split: str = DEFAULT_SPLIT_HELLASWAG,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_HELLASWAG,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_HELLASWAG,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_HELLASWAG,
    choices_format: str = "letter",
    few_shot_examples_key: Optional[str] = None,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output", 
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:

    if hellaswag_accuracy_metric is None:
        return {"HellaSwag": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting HellaSwag ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Split='{dataset_split}', GenBatchSize={generation_batch_size}")
    logger.info(f"Generation config: max_new_tokens={max_new_tokens}, batch_size={generation_batch_size}")

    # Get prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"HellaSwag": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_hellaswag_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("HellaSwag: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"HellaSwag": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"HellaSwag": 0.0, "error_message": f"DatasetLoadFailed HellaSwag: {e}"}
    logger.info(f"P{process_id}: Loaded HellaSwag '{dataset_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    # Handle multi-GPU processing
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
        return {"HellaSwag": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} HellaSwag examples.")

    predictions_numeric, true_labels_numeric = [], []
    label_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    detailed_results = []
    prompts_for_batch, original_items_for_batch, processed_items_for_batch = [], [], []
    
    for item_idx, raw_item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - HellaSwag Eval")):
        try:
            processed_item = process_docs_hellaswag(raw_item_data)
        except Exception as e:
            logger.warning(f"P{process_id}: Error processing HellaSwag item: {e}")
            continue
            
        true_label_int = processed_item.get('gold', -1)
        if true_label_int < 0 or true_label_int > 3:
            logger.warning(f"P{process_id}: Skipping HellaSwag item with invalid gold label '{true_label_int}'.")
            continue
        
        query = processed_item.get('query', '')
        choices = processed_item.get('choices', [])
        choices_data = _prepare_hellaswag_choices_data(choices)
        main_q_data = {"query": query, **choices_data}

        if num_few_shot > 0 and few_shot_examples_to_use:
            prep_fs_data = []
            for ex in few_shot_examples_to_use:
                ex_choices_data = {
                    "choiceA": ex.get('choiceA', ''),
                    "choiceB": ex.get('choiceB', ''),
                    "choiceC": ex.get('choiceC', ''),
                    "choiceD": ex.get('choiceD', '')
                }
                prep_fs_data.append({
                    "query": ex.get('query', ''),
                    "answer_letter": ex.get('answer_letter', 'A'),
                    **ex_choices_data
                })
            llm_prompt = format_few_shot_prompt(prompt_template_dict, prep_fs_data, main_q_data)
        else:
            llm_prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_for_batch.append(llm_prompt)
        original_items_for_batch.append(raw_item_data)
        processed_items_for_batch.append({
            'processed_item': processed_item,
            'llm_prompt': llm_prompt,
            'true_label_int': true_label_int,
            'item_idx': item_idx
        })

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config_hellaswag = {
                "do_sample": False, 
                "max_new_tokens": max_new_tokens, 
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            
            try:
                with torch.no_grad(): 
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config_hellaswag)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    processed_item_info = processed_items_for_batch[k]
                    processed_item = processed_item_info['processed_item']
                    original_item = original_items_for_batch[k]
                    prompt = processed_item_info['llm_prompt']
                    
                    generated_full_text = ""
                    generated_part = ""
                    
                    if raw_out_list and len(raw_out_list) > 0 and 'generated_text' in raw_out_list[0]:
                        generated_full_text = raw_out_list[0]['generated_text']
                        if generated_full_text.startswith(prompt):
                            generated_part = generated_full_text[len(prompt):]
                        else:
                            generated_part = generated_full_text
                    else:
                        generated_full_text = "[NO_GENERATION]"
                        generated_part = "[NO_GENERATION]"
                    
                    pred_letter = _extract_hellaswag_answer(generated_full_text, prompt)
                    true_letter = chr(65 + processed_item_info['true_label_int']) 
                    
                    extraction_successful = pred_letter is not None
                    if pred_letter is None:
                        import random
                        pred_letter = random.choice(['A', 'B', 'C', 'D'])
                        logger.debug(f"Extraction failed, using random fallback: {pred_letter}")
                    
                    pred_num = label_to_int.get(pred_letter, -1)  
                    true_num = processed_item_info['true_label_int']
                    
                    if pred_num == -1 and true_num != -1: 
                        pred_num = (true_num + 1) % 4
                    
                    is_correct = pred_letter == true_letter
                    predictions_numeric.append(pred_num)
                    true_labels_numeric.append(true_num)
                    
    
                    if save_detailed:
                        choices = processed_item.get('choices', [])
                        while len(choices) < 4:
                            choices.append("")
                            
                        formatted_choices = {
                            'A': choices[0],
                            'B': choices[1], 
                            'C': choices[2],
                            'D': choices[3]
                        }
                        
                        detailed_result = {
                            "question_id": original_item.get('ind', f"item_{processed_item_info['item_idx']}"),
                            "activity_label": original_item.get('activity_label', ''),
                            "ctx_a": original_item.get('ctx_a', ''),
                            "ctx_b": original_item.get('ctx_b', ''),
                            "query": processed_item.get('query', ''),
                            "choices": formatted_choices,
                            "correct_answer_letter": true_letter,
                            "correct_answer_text": choices[true_num] if 0 <= true_num < len(choices) else "Unknown",
                            "predicted_answer_letter": pred_letter,
                            "predicted_answer_text": choices[pred_num] if pred_num != -1 and 0 <= pred_num < len(choices) else "Unknown",
                            "generated_text_full": generated_full_text,
                            "generated_text_part": generated_part.strip(),
                            "is_correct": is_correct,
                            "prompt_used": prompt,
                            "extraction_successful": extraction_successful
                        }
                        detailed_results.append(detailed_result)
                        
            except Exception as e_batch_hellaswag:
                logger.error(f"P{process_id}: Error in HellaSwag gen batch: {e_batch_hellaswag}", exc_info=True)
                for processed_item_err_info in processed_items_for_batch:
                     true_num_err = processed_item_err_info['true_label_int']
                     pred_num_err = (true_num_err + 1) % 4 
                     predictions_numeric.append(pred_num_err)
                     true_labels_numeric.append(true_num_err)
                     
                     if save_detailed:
                         detailed_result = {
                             "question_id": f"error_item_{processed_item_err_info['item_idx']}",
                             "activity_label": original_items_for_batch[processed_items_for_batch.index(processed_item_err_info)].get('activity_label', ''),
                             "query": processed_item_err_info['processed_item'].get('query', ''),
                             "choices": processed_item_err_info['processed_item'].get('choices', []),
                             "correct_answer_letter": chr(65 + true_num_err),
                             "correct_answer_text": "ERROR - Generation failed",
                             "predicted_answer_letter": "ERROR",
                             "predicted_answer_text": "ERROR - Generation failed",
                             "generated_text_full": "[GENERATION_ERROR]",
                             "generated_text_part": "[GENERATION_ERROR]",
                             "is_correct": False,
                             "prompt_used": processed_item_err_info['llm_prompt'],
                             "extraction_successful": False
                         }
                         detailed_results.append(detailed_result)
                         
            prompts_for_batch, original_items_for_batch, processed_items_for_batch = [], [], []

    if not true_labels_numeric: 
        return {"HellaSwag": 0.0}
    
    acc_score = 0.0
    try:
        valid_preds = [p for i, p in enumerate(predictions_numeric) if true_labels_numeric[i] != -1]
        valid_refs = [r for r in true_labels_numeric if r != -1]
        if valid_preds and valid_refs:
            acc_results = hellaswag_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
            acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric: 
        logger.error(f"P{process_id}: Error computing HellaSwag accuracy: {e_metric}")
    
    extraction_success_rate = sum(1 for r in detailed_results if r.get("extraction_successful", False)) / len(detailed_results) * 100 if detailed_results else 0
    
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final HellaSwag Acc: {acc_score:.2f}% on {len(valid_refs)} examples.")
    logger.info(f"P{process_id} - Extraction Success Rate: {extraction_success_rate:.2f}%")
    
    if save_detailed and detailed_results:
        saved_path = save_detailed_hellaswag_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            num_few_shot,
            acc_score,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed HellaSwag results with {len(detailed_results)} examples saved to: {saved_path}")
    
    return {"HellaSwag": acc_score}


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: 
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test HellaSwag")
    test_parser.add_argument("--model_name_test", type=str, default="google/gemma-2b")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:1000]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser.add_argument("--num_few_shot_test", type=int, default=5) 
    test_parser.add_argument("--save_detailed", action="store_true", help="Save detailed outputs to JSON file")
    
    hs_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="HellaSwagFileTest")
    logger.info(f"--- Standalone HellaSwag Test: {hs_args.model_name_test} ({hs_args.num_few_shot_test}-shot) ---")
    
    hs_pipe, _ = initialize_model_pipeline(hs_args.model_name_test, target_device_id=0)
    if hs_pipe:
        hs_eval_args = {
            "pipe": hs_pipe, "tokenizer": hs_pipe.tokenizer, "model_name_for_logging": hs_args.model_name_test,
            "device": hs_pipe.device, "dataset_split": hs_args.dataset_split_test,
            "generation_batch_size": hs_args.gen_batch_size_test,
            "num_few_shot": hs_args.num_few_shot_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1,
            "save_detailed": hs_args.save_detailed
        }
        try: 
            print(json.dumps(evaluate_hellaswag(**hs_eval_args), indent=2))
        finally: 
            cleanup_model_resources(hs_pipe, getattr(hs_pipe, 'model', None))