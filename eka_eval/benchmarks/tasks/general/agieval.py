# eka_eval/benchmarks/tasks/general/agieval.py - Fixed version with better debug

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import evaluate as hf_evaluate
from datetime import datetime

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME_AGIEVAL = "hails/agieval-lsat-ar"
DEFAULT_SPLIT_AGIEVAL = "test"
DEFAULT_MAX_NEW_TOKENS_AGIEVAL = 10  
DEFAULT_GENERATION_BATCH_SIZE_AGIEVAL = 8
DEFAULT_NUM_FEWSHOT_AGIEVAL = 5
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "agieval_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "agieval_5shot"
PROMPT_FILE_BENCHMARK_KEY = "agieval"
PROMPT_FILE_CATEGORY = "general"

try:
    agieval_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for AGIEval loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for AGIEval: {e}. AGIEval will not run correctly.", exc_info=True)
    agieval_accuracy_metric = None

def save_detailed_agieval_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    num_few_shot: int,
    accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed AGIEval results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agieval_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
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
        logger.info(f"Detailed AGIEval results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed AGIEval results: {e}")
        return ""

def _get_agieval_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_agieval",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for AGIEval.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_agieval from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def _extract_agieval_answer_letter(generated_text: str, prompt_text_sent_to_llm: Optional[str] = None) -> Optional[str]:
    """Extract the answer letter (A-E) from generated text with enhanced debugging."""
    completion_part = generated_text
    if prompt_text_sent_to_llm and generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    
    completion_part = completion_part.strip()
    
    logger.debug(f"AGIEval extraction input: '{completion_part[:100]}'")
    
    if len(completion_part) == 0:
        logger.debug("AGIEval: Empty completion, extraction failed")
        return None
    

    patterns = [
        r'^([A-E])$',                          
        r'^([A-E])\b',                         # Letter at start
        r'^([A-E])[\.\)\:]',                   # Letter with punctuation
        r'(?:Answer|answer)[:\s]*([A-E])\b',   # After "Answer:"
        r'\b([A-E])\b',                        # Any isolated letter
        r'([A-E])'                             # Letter anywhere
    ]
    
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, completion_part, re.IGNORECASE)
        if matches:
            letter = matches[0].upper()
            logger.debug(f"AGIEval: Extracted '{letter}' using pattern {i+1}: {pattern}")
            return letter
    
    logger.debug(f"AGIEval: No pattern matched for: '{completion_part[:50]}'")
    return None

def _prepare_agieval_choices_string(choices: List[str], format_style: str = "letter") -> str:
    """Format choices into a string."""
    if format_style == "letter":
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    elif format_style == "number":
        return "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    elif format_style == "simple":
        return "\n".join(choices)
    else:
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

def evaluate_agieval(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_AGIEVAL,
    dataset_split: str = DEFAULT_SPLIT_AGIEVAL,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_AGIEVAL,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_AGIEVAL,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_AGIEVAL,
    choices_format: str = "letter",
    few_shot_examples_key: Optional[str] = None,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:

    if agieval_accuracy_metric is None:
        return {"AGIEval": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting AGIEval ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")
    logger.info(f"Generation config: max_new_tokens={max_new_tokens}, batch_size={generation_batch_size}")

    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"AGIEval": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_agieval_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("AGIEval: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"AGIEval": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    # Load dataset
    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"AGIEval": 0.0, "error_message": f"DatasetLoadFailed: {e}"}
    
    logger.info(f"P{process_id}: Loaded AGIEval ({len(full_data)} examples).")


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
        return {"AGIEval": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} AGIEval examples.")
    predictions, true_labels = [], []
    prompts_for_batch, original_items_for_batch_info = [], []
    detailed_results = []

    # Main evaluation loop
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - AGIEval Eval")):
        question = item_data.get("query")
        choices = item_data.get("choices")
        gold = item_data.get("gold")
        
       
        if isinstance(gold, list) and len(gold) > 0:
            true_answer_idx = gold[0]
            true_answer_letter = chr(65 + true_answer_idx)
        else:
            logger.warning(f"Invalid gold format for item {item_idx}: {gold}")
            continue
        
        if not all([question, choices]):
            logger.warning(f"Missing data for item {item_idx}")
            continue

        choices_str_fmt = _prepare_agieval_choices_string(choices, choices_format)
        main_q_data = {"question": question, "choices_str": choices_str_fmt}

        if num_few_shot > 0 and few_shot_examples_to_use:
            prep_fs_data = []
            for ex in few_shot_examples_to_use:
                ex_choices_str_fs = _prepare_agieval_choices_string(ex.get('choices', []), choices_format)
                prep_fs_data.append({
                    "question": ex.get('question'),
                    "choices_str": ex_choices_str_fs,
                    "answer_letter": ex.get('answer_letter')
                })
            llm_prompt = format_few_shot_prompt(prompt_template_dict, prep_fs_data, main_q_data)
        else:
            llm_prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_for_batch.append(llm_prompt)
        original_items_for_batch_info.append({
            'true_answer_idx': true_answer_idx,
            'true_answer_letter': true_answer_letter,
            'llm_prompt': llm_prompt,
            'question': question,
            'choices': choices,
            'item_idx': item_idx
        })

        # Process batch
        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True  
            }
            
            if item_idx < generation_batch_size:
                logger.info(f"Generation config: {gen_config}")
            
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    orig_info = original_items_for_batch_info[k]
                    
                    generated_full_text = ""
                    generated_part = ""
                    
                    if raw_out_list and len(raw_out_list) > 0 and 'generated_text' in raw_out_list[0]:
                        generated_full_text = raw_out_list[0]['generated_text']
                        if generated_full_text.startswith(orig_info['llm_prompt']):
                            generated_part = generated_full_text[len(orig_info['llm_prompt']):]
                        else:
                            generated_part = generated_full_text
                    else:
                        generated_full_text = "[NO_GENERATION]"
                        generated_part = "[NO_GENERATION]"
                    
                    pred_letter = _extract_agieval_answer_letter(generated_full_text, orig_info['llm_prompt'])
                    true_letter = orig_info['true_answer_letter']
                    
                    extraction_successful = pred_letter is not None
                    if pred_letter is None:
                        import random
                        pred_letter = random.choice(['A', 'B', 'C', 'D', 'E'])
                        logger.debug(f"Extraction failed, using random fallback: {pred_letter}")
                    
                    is_correct = pred_letter == true_letter
                    predictions.append(pred_letter)
                    true_labels.append(true_letter)
                    
 #debugging
                    if save_detailed:
                        detailed_result = {
                            "question_id": orig_info['item_idx'],
                            "question": orig_info['question'],
                            "choices": orig_info['choices'],
                            "correct_answer_letter": true_letter,
                            "correct_answer_text": orig_info['choices'][orig_info['true_answer_idx']] if 0 <= orig_info['true_answer_idx'] < len(orig_info['choices']) else "Unknown",
                            "predicted_answer_letter": pred_letter,
                            "predicted_answer_text": orig_info['choices'][ord(pred_letter) - 65] if pred_letter in 'ABCDE' and 0 <= ord(pred_letter) - 65 < len(orig_info['choices']) else "Unknown",
                            "generated_text_full": generated_full_text,
                            "generated_text_part": generated_part.strip(),
                            "is_correct": is_correct,
                            "prompt_used": orig_info['llm_prompt'],
                            "extraction_successful": extraction_successful
                        }
                        detailed_results.append(detailed_result)
                        
            except Exception as e_batch:
                logger.error(f"P{process_id}: AGIEval generation batch error: {e_batch}", exc_info=True)
               
                for info in original_items_for_batch_info:
                    predictions.append("A")
                    true_labels.append(info['true_answer_letter'])
                    
                    if save_detailed:
                        detailed_result = {
                            "question_id": info['item_idx'],
                            "question": info['question'],
                            "choices": info['choices'],
                            "correct_answer_letter": info['true_answer_letter'],
                            "correct_answer_text": "ERROR - Generation failed",
                            "predicted_answer_letter": "A",
                            "predicted_answer_text": "ERROR - Generation failed",
                            "generated_text_full": "[GENERATION_ERROR]",
                            "generated_text_part": "[GENERATION_ERROR]",
                            "is_correct": False,
                            "prompt_used": info['llm_prompt'],
                            "extraction_successful": False
                        }
                        detailed_results.append(detailed_result)
            
            prompts_for_batch, original_items_for_batch_info = [], []
    
    if not true_labels:
        return {"AGIEval": 0.0}
    
    correct_count = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = (correct_count / len(true_labels)) * 100
    
    extraction_success_rate = sum(1 for r in detailed_results if r.get("extraction_successful", False)) / len(detailed_results) * 100 if detailed_results else 0
    
    logger.info(f"P{process_id} - Final AGIEval Accuracy: {accuracy:.2f}% on {len(true_labels)} examples.")
    logger.info(f"P{process_id} - Extraction Success Rate: {extraction_success_rate:.2f}%")
    
    if save_detailed and detailed_results:
        saved_path = save_detailed_agieval_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            num_few_shot,
            accuracy,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed AGIEval results with {len(detailed_results)} examples saved to: {saved_path}")
    
    return {"AGIEval": accuracy}
