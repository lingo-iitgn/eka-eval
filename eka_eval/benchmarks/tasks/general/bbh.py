# eka_eval/benchmarks/tasks/general/bbh.py

import torch
import re
import json
import os
import sys
import logging
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import evaluate as hf_evaluate
from datetime import datetime

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DATASET_NAME_BBH = "lukaemon/bbh"
DEFAULT_SPLIT_BBH = "test"
DEFAULT_MAX_NEW_TOKENS_BBH = 512
DEFAULT_GENERATION_BATCH_SIZE_BBH = 4
DEFAULT_NUM_FEWSHOT_BBH = 3
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "bbh_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "bbh_3shot"
PROMPT_FILE_BENCHMARK_KEY = "bbh"
PROMPT_FILE_CATEGORY = "general"

# Complete list of 27 BBH tasks
BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement", 
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects", 
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting"
]

try:
    bbh_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for BBH loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for BBH: {e}. BBH may not run correctly.", exc_info=True)
    bbh_accuracy_metric = None

def save_detailed_bbh_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    num_few_shot: int,
    avg_accuracy: float,
    task_accuracies: Dict[str, float],
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed BBH results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bbh_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    # Calculate task-wise statistics
    task_stats = {}
    for task_name, accuracy in task_accuracies.items():
        task_examples = [r for r in results_data if r.get('task_name') == task_name]
        task_stats[task_name] = {
            'accuracy': accuracy,
            'total_examples': len(task_examples),
            'correct_examples': sum(1 for r in task_examples if r.get('is_correct', False))
        }
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_few_shot": num_few_shot,
        "total_tasks": len(task_accuracies),
        "average_accuracy": avg_accuracy,
        "task_accuracies": task_accuracies,
        "task_statistics": task_stats,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed BBH results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed BBH results: {e}")
        return ""

def _get_bbh_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str, task_name: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration for specific task."""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_dict = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_bbh",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_dict and isinstance(loaded_examples_dict, dict):
        task_examples = loaded_examples_dict.get(task_name, [])
        if task_examples:
            logger.info(f"Successfully loaded {len(task_examples)} few-shot examples for BBH task {task_name}.")
            return task_examples[:num_few_shot]
    
    logger.warning(f"Could not load few-shot examples for BBH task {task_name}")
    return []

def _extract_bbh_answer(generated_text: str, prompt_text: str, task_name: str) -> Optional[str]:
    """Extract answer from BBH generation, handling different task formats."""
    
    # Remove prompt from generation
    completion = generated_text
    if generated_text.startswith(prompt_text):
        completion = generated_text[len(prompt_text):].strip()
    
    # Task-specific answer extraction
    if task_name in ["boolean_expressions"]:
        # Look for True/False
        match = re.search(r'\b(True|False)\b', completion, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    elif task_name in ["causal_judgement"]:
        # Look for Yes/No
        match = re.search(r'\b(Yes|No)\b', completion, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    elif task_name.startswith("logical_deduction"):
        # Look for letters A, B, C, D, E or object names
        match = re.search(r'\b([A-E])\b', completion)
        if match:
            return match.group(1).upper()
        # Also look for common object references
        objects = re.findall(r'\b(red|blue|green|yellow|purple|orange|black|white|brown|gray)\s+\w+\b', completion, re.IGNORECASE)
        if objects:
            return objects[-1].lower()  # Take the last mentioned object
    
    elif task_name in ["date_understanding"]:
        # Look for date format MM/DD/YYYY
        match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', completion)
        if match:
            return match.group(1)
        # Also look for day names
        days = re.search(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', completion, re.IGNORECASE)
        if days:
            return days.group(1).capitalize()
    
    elif task_name in ["multistep_arithmetic_two"]:
        # Look for numbers, prefer the last number mentioned
        numbers = re.findall(r'\b(\d+)\b', completion)
        if numbers:
            return numbers[-1]  # Take the last number
    
    elif task_name in ["object_counting"]:
        # Look for numbers
        match = re.search(r'\b(\d+)\b', completion)
        if match:
            return match.group(1)
    
    else:
        # Generic extraction - look for the last meaningful word/phrase
        lines = completion.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("Let's think") and not line.startswith("Step"):
                # Remove common prefixes
                line = re.sub(r'^(The answer is|Answer:|So,?|Therefore,?)\s*', '', line, flags=re.IGNORECASE)
                if line:
                    # Extract the final answer after common conclusion words
                    final_answer = re.search(r'(?:is|answer is|equals?)\s+(.+?)\.?$', line, re.IGNORECASE)
                    if final_answer:
                        return final_answer.group(1).strip()
                    return line.strip()
    
    logger.debug(f"BBH: Could not extract answer from completion for {task_name}: '{completion[:100]}'")
    return None

def _normalize_answer(answer: str, target: str, task_name: str) -> Tuple[str, str]:
    """Normalize answers for comparison."""
    if not answer or not target:
        return str(answer or "").strip(), str(target or "").strip()
    
    answer = str(answer).strip()
    target = str(target).strip()
    
    # Task-specific normalization
    if task_name in ["boolean_expressions"]:
        answer = answer.lower().capitalize()
        target = target.lower().capitalize()
    elif task_name in ["causal_judgement"]:
        answer = answer.lower().capitalize()
        target = target.lower().capitalize()
    elif task_name in ["date_understanding"]:
        # Normalize date formats
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', answer):
            parts = answer.split('/')
            answer = f"{int(parts[0]):02d}/{int(parts[1]):02d}/{parts[2]}"
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', target):
            parts = target.split('/')
            target = f"{int(parts[0]):02d}/{int(parts[1]):02d}/{parts[2]}"
    
    return answer.lower(), target.lower()

def evaluate_bbh_task(
    pipe: Any, tokenizer: Any, task_name: str, task_data: List[Dict],
    num_few_shot: int, prompt_template_dict: Dict[str, Any], few_shot_examples: List[Dict],
    max_new_tokens: int, generation_batch_size: int, process_id: int
) -> Tuple[List[str], List[str], List[Dict]]:
    """Evaluate a single BBH task."""
    
    predictions, targets = [], []
    prompts_batch, targets_batch, items_batch = [], [], []
    detailed_results = []
    
    for item_idx, item in enumerate(tqdm(task_data, desc=f"P{process_id} - BBH {task_name}")):
        question = item.get("input", "")
        target = item.get("target", "")
        
        if not question or not target:
            logger.warning(f"P{process_id}: Skipping BBH {task_name} item due to missing data")
            continue
        
        # Prepare data for prompt formatting
        main_q_data = {
            "question": question,
            "task_type": task_name.replace('_', ' ')
        }
        
        # Generate prompt using templates
        if num_few_shot > 0 and few_shot_examples:
            # Format few-shot examples
            prep_fs_data = []
            for ex in few_shot_examples:
                ex_data = {
                    "question": ex.get('question', ''),
                    "reasoning": ex.get('reasoning', ''),
                    "target": ex.get('target', '')
                }
                prep_fs_data.append(ex_data)
            llm_prompt = format_few_shot_prompt(prompt_template_dict, prep_fs_data, main_q_data)
        else:
            llm_prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_batch.append(llm_prompt)
        targets_batch.append(target)
        items_batch.append(item)
        
        # Process batch
        if len(prompts_batch) == generation_batch_size or item_idx == len(task_data) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            
            try:
                with torch.no_grad():
                    batch_outputs = pipe(prompts_batch, **gen_config)
                
                for k, output_list in enumerate(batch_outputs):
                    if output_list and len(output_list) > 0 and 'generated_text' in output_list[0]:
                        raw_gen = output_list[0]['generated_text']
                    else:
                        raw_gen = prompts_batch[k] + " [NO_GENERATION]"
                    
                    pred = _extract_bbh_answer(raw_gen, prompts_batch[k], task_name)
                    target_val = targets_batch[k]
                    
                    if pred is None:
                        pred = "UNKNOWN"
                    
                    # Normalize for comparison
                    norm_pred, norm_target = _normalize_answer(pred, target_val, task_name)
                    is_correct = norm_pred == norm_target
                    
                    predictions.append(pred)
                    targets.append(target_val)
                    
                    # Store detailed result
                    detailed_result = {
                        "task_name": task_name,
                        "question_id": item_idx,
                        "question": question,
                        "target_answer": target_val,
                        "predicted_answer": pred,
                        "is_correct": is_correct,
                        "prompt_used": prompts_batch[k],
                        "generated_text": raw_gen,
                        "extraction_successful": pred != "UNKNOWN"
                    }
                    detailed_results.append(detailed_result)
                    
                    # Debug first few examples
                    if item_idx < 2:
                        completion = raw_gen[len(prompts_batch[k]):] if raw_gen.startswith(prompts_batch[k]) else raw_gen
                        print(f"\n=== DEBUG BBH {task_name} Example {item_idx} ===")
                        print(f"Question: {question[:100]}...")
                        print(f"Generated: {completion[:200]}...")
                        print(f"Extracted: {pred}")
                        print(f"Expected: {target_val}")
                        print(f"Correct: {'✅' if is_correct else '❌'}")
                        print("=" * 60)
                    
            except Exception as e:
                logger.error(f"P{process_id}: Error in BBH {task_name} generation: {e}", exc_info=True)
                # Add fallback predictions
                for j, target_val in enumerate(targets_batch):
                    predictions.append("UNKNOWN")
                    targets.append(target_val)
                    detailed_result = {
                        "task_name": task_name,
                        "question_id": item_idx - len(targets_batch) + j + 1,
                        "question": "ERROR",
                        "target_answer": target_val,
                        "predicted_answer": "UNKNOWN",
                        "is_correct": False,
                        "prompt_used": prompts_batch[j] if j < len(prompts_batch) else "ERROR",
                        "generated_text": "[GENERATION_ERROR]",
                        "extraction_successful": False
                    }
                    detailed_results.append(detailed_result)
            
            prompts_batch, targets_batch, items_batch = [], [], []
    
    return predictions, targets, detailed_results

def evaluate_bbh(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_BBH,
    dataset_split: str = DEFAULT_SPLIT_BBH,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_BBH,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_BBH,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_BBH,
    tasks_to_eval: Optional[List[str]] = None,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:

    if bbh_accuracy_metric is None:
        return {"BBH": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    # Determine tasks to evaluate
    tasks_to_run = [t for t in (tasks_to_eval or BBH_TASKS) if t in BBH_TASKS]
    if not tasks_to_run:
        return {"BBH": 0.0, "error_message": "NoValidTasksSpecified"}

    logger.info(f"Starting BBH ({num_few_shot}-shot): {model_name_for_logging} on {len(tasks_to_run)} tasks")

    # Load prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"BBH": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Split tasks across GPUs
    if num_gpus > 1:
        tasks_per_gpu = len(tasks_to_run) // num_gpus
        start_idx = process_id * tasks_per_gpu
        end_idx = (process_id + 1) * tasks_per_gpu if process_id < num_gpus - 1 else len(tasks_to_run)
        tasks_for_gpu = tasks_to_run[start_idx:end_idx]
    else:
        tasks_for_gpu = tasks_to_run
    
    logger.info(f"P{process_id}: Processing {len(tasks_for_gpu)} BBH tasks: {tasks_for_gpu}")
    
    # Evaluate each task
    task_scores = {}
    all_detailed_results = []
    
    for task_name in tasks_for_gpu:
        logger.info(f"P{process_id}: Loading dataset for task {task_name}")
        
        # Load dataset for this task
        try:
            dataset = load_dataset(
                dataset_name,
                task_name,
                split=dataset_split,
                trust_remote_code=True
            )
            task_data = list(dataset)
            logger.info(f"P{process_id}: Loaded {len(task_data)} examples for {task_name}")
        except Exception as e:
            logger.error(f"P{process_id}: Failed to load task {task_name}: {e}")
            task_scores[task_name] = 0.0
            continue
        
        if not task_data:
            logger.warning(f"P{process_id}: No data available for task {task_name}")
            task_scores[task_name] = 0.0
            continue
        
        # Load few-shot examples for this task
        few_shot_examples = []
        if num_few_shot > 0:
            few_shot_examples = _get_bbh_fewshot_examples_from_config(num_few_shot, prompt_file_category, task_name)
            if not few_shot_examples:
                logger.warning(f"BBH: No few-shot examples found for {task_name}, falling back to 0-shot")
        
        # Evaluate this task
        logger.info(f"P{process_id}: Evaluating {task_name} with {len(task_data)} examples")
        
        predictions, targets, detailed_results = evaluate_bbh_task(
            pipe, tokenizer, task_name, task_data,
            num_few_shot, prompt_template_dict, few_shot_examples,
            max_new_tokens, generation_batch_size, process_id
        )
        
        # Add to all detailed results
        all_detailed_results.extend(detailed_results)
        
        # Calculate accuracy for this task
        if predictions and targets:
            try:
                correct = 0
                for pred, target in zip(predictions, targets):
                    norm_pred, norm_target = _normalize_answer(pred, target, task_name)
                    if norm_pred == norm_target:
                        correct += 1
                
                accuracy = (correct / len(targets)) * 100
                task_scores[task_name] = accuracy
                logger.info(f"P{process_id}: {task_name} accuracy: {accuracy:.2f}% ({correct}/{len(targets)})")
            except Exception as e:
                logger.error(f"P{process_id}: Error computing accuracy for {task_name}: {e}")
                task_scores[task_name] = 0.0
        else:
            task_scores[task_name] = 0.0
    
    # Compute average score
    if task_scores:
        avg_score = sum(task_scores.values()) / len(task_scores)
        results = {"BBH": avg_score}
        
        # Add individual task scores
        for task_name, score in task_scores.items():
            results[f"BBH_{task_name}"] = score
        
        logger.info(f"P{process_id}: BBH Average: {avg_score:.2f}% across {len(task_scores)} tasks")
    else:
        avg_score = 0.0
        results = {"BBH": avg_score}
        logger.warning(f"P{process_id}: No BBH tasks completed successfully")
    
    # Save detailed results
    if save_detailed and all_detailed_results:
        saved_path = save_detailed_bbh_results(
            all_detailed_results,
            model_name_for_logging,
            dataset_name,
            num_few_shot,
            avg_score,
            task_scores,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed BBH results with {len(all_detailed_results)} examples saved to: {saved_path}")
    
    return results
