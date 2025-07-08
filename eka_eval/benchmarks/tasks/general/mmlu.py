# eka_eval/benchmarks/tasks/general/mmlu.py - Enhanced with detailed results saving

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

# Your original defaults
DEFAULT_DATASET_NAME_MMLU = "cais/mmlu"
DEFAULT_SUBSET_MMLU = "all"
DEFAULT_SPLIT_MMLU = "test"
DEFAULT_MAX_NEW_TOKENS_MMLU = 1  # Changed to 1 for better performance
DEFAULT_GENERATION_BATCH_SIZE_MMLU = 8
DEFAULT_NUM_FEWSHOT_MMLU = 5
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "mmlu_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "mmlu_5shot"
PROMPT_FILE_BENCHMARK_KEY = "mmlu"
PROMPT_FILE_CATEGORY = "general"

try:
    mmlu_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for MMLU loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for MMLU: {e}. MMLU will not run correctly.", exc_info=True)
    mmlu_accuracy_metric = None

def save_detailed_results(
    results_data: List[Dict],
    model_name: str,
    dataset_config: str,
    num_few_shot: int,
    accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:

    # Create detailed results directory
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    # Generate filename
    model_clean = model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mmlu_{model_clean}_{dataset_config}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    # Prepare summary data
    summary = {
        "model_name": model_name,
        "dataset_config": dataset_config,
        "num_few_shot": num_few_shot,
        "total_questions": len(results_data),
        "correct_answers": sum(1 for r in results_data if r["is_correct"]),
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id
    }
    
    # Calculate subject-wise breakdown
    subject_stats = {}
    for result in results_data:
        subject = result["subject"]
        if subject not in subject_stats:
            subject_stats[subject] = {"total": 0, "correct": 0}
        subject_stats[subject]["total"] += 1
        if result["is_correct"]:
            subject_stats[subject]["correct"] += 1
    
    # Add accuracy for each subject
    for subject in subject_stats:
        total = subject_stats[subject]["total"]
        correct = subject_stats[subject]["correct"]
        subject_stats[subject]["accuracy"] = (correct / total * 100) if total > 0 else 0.0
    
    summary["subject_breakdown"] = subject_stats
    
    # Prepare full data structure
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    # Save to JSON
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed MMLU results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def _get_mmlu_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_mmlu",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for MMLU.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_mmlu from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json or it's not a list.")
    return []

def _extract_mmlu_answer_index(generated_text: str, prompt_text_sent_to_llm: Optional[str] = None) -> Optional[int]:
    """Enhanced answer extraction for MMLU"""
    completion_part = generated_text
    if prompt_text_sent_to_llm and generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    
    completion_part = completion_part.strip()
    
    if len(completion_part) == 0:
        return None
    
    # Try multiple extraction patterns
    patterns = [
        r'^([A-D])$',                          # Just the letter
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
            answer_idx = ord(letter) - ord('A')
            logger.debug(f"MMLU: Extracted '{letter}' (index {answer_idx}) using pattern {i+1}")
            return answer_idx
    
    # Fallback word mappings
    completion_lower = completion_part.lower()
    word_mappings = {
        'first': 0, 'second': 1, 'third': 2, 'fourth': 3,
        'one': 0, 'two': 1, 'three': 2, 'four': 3,
        'option a': 0, 'option b': 1, 'option c': 2, 'option d': 3,
        'choice a': 0, 'choice b': 1, 'choice c': 2, 'choice d': 3
    }
    
    for phrase, idx in word_mappings.items():
        if phrase in completion_lower:
            logger.debug(f"MMLU: Extracted answer {idx} from phrase '{phrase}'")
            return idx
    
    logger.debug(f"MMLU: Failed to extract answer from: '{completion_part[:50]}'")
    return None

def evaluate_mmlu(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_MMLU,
    dataset_config_name: str = DEFAULT_SUBSET_MMLU,
    dataset_split: str = DEFAULT_SPLIT_MMLU,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_MMLU,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_MMLU,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", 
    save_detailed: bool = True,  # New parameter to enable detailed saving
    **kwargs
) -> Dict[str, float]:

    if mmlu_accuracy_metric is None:
        return {"MMLU": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    benchmark_display_name = f"MMLU_{dataset_config_name}" if dataset_config_name.lower() != "all" else "MMLU"
    logger.info(f"Starting {benchmark_display_name} ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}/{dataset_config_name}")

    # Load prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    if not prompt_template_dict:
        return {"MMLU": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_mmlu_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("MMLU: Failed to load few-shot from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict: 
                return {"MMLU": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    # Load dataset
    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"MMLU": 0.0, "error_message": f"DatasetLoadFailed MMLU: {e}"}
    logger.info(f"P{process_id}: Loaded MMLU ({len(full_data)} examples).")

    # Multi-GPU data splitting
    if num_gpus > 1:
        total = len(full_data); per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    if len(subset_to_process) == 0: return {"MMLU": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} MMLU examples.")

    # Initialize tracking
    predictions_idx_list, true_labels_idx_list = [], []
    prompts_for_batch, original_items_for_batch_info = [], []
    detailed_results = []  # Store detailed results

    # Main evaluation loop
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - MMLU Eval")):
        question = item_data.get("question")
        subject = item_data.get("subject", "general") 
        choices = item_data.get("choices")
        true_answer_idx = item_data.get("answer")
        
        if not all([question, choices, isinstance(true_answer_idx, int)]): 
            continue

        choices_str_fmt = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        main_q_data = {"subject": subject.replace('_', ' ').title(), "question": question, "choices_str": choices_str_fmt}
        
        # Generate prompt
        if num_few_shot > 0 and few_shot_examples_to_use:
            prep_fs_data = []
            for ex in few_shot_examples_to_use:
                ex_choices_str_fs = "\n".join([f"{choice_text.strip()}" for choice_text in ex.get('choices', [])])
                prep_fs_data.append({"question": ex.get('question'), "choices_str": ex_choices_str_fs, "answer_letter": ex.get('answer_letter')})
            llm_prompt = format_few_shot_prompt(prompt_template_dict, prep_fs_data, main_q_data)
        else:
            llm_prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_for_batch.append(llm_prompt)
        original_items_for_batch_info.append({
            'true_answer_idx': true_answer_idx, 
            'llm_prompt': llm_prompt,
            'question': question,
            'subject': subject,
            'choices': choices,
            'item_idx': item_idx
        })

        # Process batch
        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            # Optimized generation config
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": False,  # Only return new tokens
                "clean_up_tokenization_spaces": True
            }
            
            try:
                with torch.no_grad(): 
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    orig_info = original_items_for_batch_info[k]
                    
                    # Extract generated text
                    if raw_out_list and len(raw_out_list) > 0 and 'generated_text' in raw_out_list[0]:
                        generated_part = raw_out_list[0]['generated_text']
                        raw_gen = orig_info['llm_prompt'] + generated_part
                    else:
                        generated_part = "[NO_GENERATION]"
                        raw_gen = orig_info['llm_prompt'] + generated_part
                    
                    # Extract answer
                    pred_idx = _extract_mmlu_answer_index(raw_gen, orig_info['llm_prompt'])
                    true_idx = orig_info['true_answer_idx']
                    
                    # Handle failed extraction
                    if pred_idx is None: 
                        pred_idx = (true_idx + 1) % 4  # Wrong answer as fallback
                    
                    is_correct = pred_idx == true_idx
                    predictions_idx_list.append(pred_idx)
                    true_labels_idx_list.append(true_idx)
                    
                    # Store detailed result
                    if save_detailed:
                        detailed_result = {
                            "question_id": orig_info['item_idx'],
                            "subject": orig_info['subject'],
                            "question": orig_info['question'],
                            "choices": orig_info['choices'],
                            "correct_answer_index": true_idx,
                            "correct_answer_letter": chr(65 + true_idx),
                            "correct_answer_text": orig_info['choices'][true_idx] if 0 <= true_idx < len(orig_info['choices']) else "Unknown",
                            "predicted_answer_index": pred_idx,
                            "predicted_answer_letter": chr(65 + pred_idx) if 0 <= pred_idx <= 3 else "Unknown",
                            "predicted_answer_text": orig_info['choices'][pred_idx] if 0 <= pred_idx < len(orig_info['choices']) else "Unknown",
                            "generated_text": generated_part.strip(),
                            "is_correct": is_correct,
                            "prompt_used": orig_info['llm_prompt'],
                            "extraction_successful": _extract_mmlu_answer_index(raw_gen, orig_info['llm_prompt']) is not None
                        }
                        detailed_results.append(detailed_result)
                    
                    # Debug first few examples
                    if orig_info['item_idx'] < 3:
                        print(f"\n=== DEBUG Example {orig_info['item_idx']} ===")
                        print(f"Subject: {orig_info['subject']}")
                        print(f"Question: {orig_info['question'][:100]}...")
                        print(f"Generated: '{generated_part.strip()}'")
                        print(f"Extracted: {chr(65 + pred_idx)} (index {pred_idx})")
                        print(f"Expected: {chr(65 + true_idx)} (index {true_idx})")
                        print(f"Correct: {'✅' if is_correct else '❌'}")
                        
            except Exception as e_batch: 
                logger.error(f"P{process_id}: MMLU generation batch error: {e_batch}", exc_info=True)
                # Add fallback predictions for failed batch
                for info in original_items_for_batch_info:
                    pred_idx = (info['true_answer_idx'] + 1) % 4
                    predictions_idx_list.append(pred_idx)
                    true_labels_idx_list.append(info['true_answer_idx'])
                    
                    if save_detailed:
                        detailed_result = {
                            "question_id": info['item_idx'],
                            "subject": info['subject'],
                            "question": info['question'],
                            "choices": info['choices'],
                            "correct_answer_index": info['true_answer_idx'],
                            "correct_answer_letter": chr(65 + info['true_answer_idx']),
                            "correct_answer_text": info['choices'][info['true_answer_idx']] if 0 <= info['true_answer_idx'] < len(info['choices']) else "Unknown",
                            "predicted_answer_index": pred_idx,
                            "predicted_answer_letter": chr(65 + pred_idx),
                            "predicted_answer_text": "ERROR - Generation failed",
                            "generated_text": "[GENERATION_ERROR]",
                            "is_correct": False,
                            "prompt_used": info['llm_prompt'],
                            "extraction_successful": False
                        }
                        detailed_results.append(detailed_result)
            
            # Reset batch
            prompts_for_batch, original_items_for_batch_info = [], []
    
    # Calculate final accuracy
    if not true_labels_idx_list: 
        return {"MMLU": 0.0}
    
    acc = 0.0
    try:
        valid_preds = [p for p in predictions_idx_list if isinstance(p, int)]
        valid_refs = [r for i,r in enumerate(true_labels_idx_list) if isinstance(predictions_idx_list[i], int)]
        if valid_preds and valid_refs: 
            acc = mmlu_accuracy_metric.compute(predictions=valid_preds, references=valid_refs).get("accuracy",0.0) * 100
    except Exception as e_metric: 
        logger.error(f"P{process_id}: MMLU metric error: {e_metric}")
    
    logger.info(f"P{process_id} - Final MMLU Acc: {acc:.2f}% on {len(valid_refs) if 'valid_refs' in locals() else 0} examples.")
    
    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_results(
            detailed_results, 
            model_name_for_logging, 
            dataset_config_name, 
            num_few_shot, 
            acc, 
            results_dir, 
            process_id
        )
        if saved_path:
            logger.info(f"Detailed results with {len(detailed_results)} examples saved to: {saved_path}")
    
    return {"MMLU": acc}
