
# eka_eval/benchmarks/tasks/knowledge/triviaqa.py

import torch
import sys
import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import string
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate
from datetime import datetime

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_TRIVIAQA = "trivia_qa"
DEFAULT_DATASET_CONFIG_TRIVIAQA = "rc"  # 'rc' for reading comprehension
DEFAULT_SPLIT_TRIVIAQA = "validation"
DEFAULT_MAX_NEW_TOKENS_TRIVIAQA = 32
DEFAULT_FEW_SHOT_COUNT_TRIVIAQA = 5
DEFAULT_GENERATION_BATCH_SIZE_TRIVIAQA = 8
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "triviaqa_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "triviaqa_5shot"
PROMPT_FILE_BENCHMARK_KEY = "triviaqa"
PROMPT_FILE_CATEGORY = "knowledge"

try:
    triviaqa_exact_match_metric = hf_evaluate.load("exact_match")
    logger.info("Exact match metric for TriviaQA loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'exact_match' metric for TriviaQA: {e}. TriviaQA may not run correctly.", exc_info=True)
    triviaqa_exact_match_metric = None

def save_detailed_triviaqa_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    num_few_shot: int,
    exact_match: float,
    f1_score: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed TriviaQA results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"triviaqa_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_few_shot": num_few_shot,
        "total_questions": len(results_data),
        "correct_answers": sum(1 for r in results_data if r["is_correct"]),
        "exact_match": exact_match,
        "f1_score": f1_score,
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
        logger.info(f"Detailed TriviaQA results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed TriviaQA results: {e}")
        return ""

def _get_triviaqa_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_triviaqa",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for TriviaQA.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_triviaqa from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def normalize_answer_triviaqa(s: str) -> str:
    """Normalize answer following TriviaQA evaluation protocol with aggressive cleaning."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    def remove_common_prefixes(text):
        # Remove common response prefixes that might interfere
        prefixes = [
            r'^the\s+', r'^a\s+', r'^an\s+',
            r'^it\s+is\s+', r'^that\s+is\s+', r'^this\s+is\s+',
            r'^the\s+answer\s+is\s+', r'^answer\s*:\s*'
        ]
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        return text
    
    if not isinstance(s, str):
        return ""
    
    # Apply all normalizations
    result = s
    result = remove_common_prefixes(result)
    result = lower(result)
    result = remove_punc(result)
    result = remove_articles(result)
    result = white_space_fix(result)
    
    return result

def remove_prefixes(aliases: List[str]) -> List[str]:
    """Remove any alias that has a strict prefix elsewhere in the list."""
    aliases_sorted = sorted(aliases)
    ret = [aliases_sorted[0]] if aliases_sorted else []
    for alias in aliases_sorted[1:]:
        if not alias.startswith(ret[-1]):
            ret.append(alias)
    return ret

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer_triviaqa(prediction) == normalize_answer_triviaqa(ground_truth)

def f1_score_triviaqa(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer_triviaqa(prediction).split()
    truth_tokens = normalize_answer_triviaqa(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(common_tokens) / len(truth_tokens) if len(truth_tokens) > 0 else 0.0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def evaluate_answer_triviaqa(prediction: str, answer_aliases: List[str]) -> Dict[str, float]:
    """Evaluate prediction against all possible answer aliases."""
    if not answer_aliases:
        return {'exact_match': 0.0, 'f1': 0.0}
    
    exact_matches = [exact_match_score(prediction, alias) for alias in answer_aliases]
    f1_scores = [f1_score_triviaqa(prediction, alias) for alias in answer_aliases]
    
    return {
        'exact_match': max(exact_matches) if exact_matches else 0.0,
        'f1': max(f1_scores) if f1_scores else 0.0
    }

def _extract_answer_triviaqa(generated_text: str, prompt: str) -> str:
    """Extract answer from generated response, handling various formats."""
    # Remove the prompt part
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    # Remove common prefixes more aggressively
    response = re.sub(r'^[Aa]nswer\s*:?\s*', '', response)
    response = re.sub(r'^(The answer is|It is|That would be|The correct answer is|According to|Based on)\s*', '', response, flags=re.IGNORECASE)
    
    # Take only the first line/sentence as answer (before any explanation)
    lines = response.split('\n')
    answer = lines[0].strip()
    
    # Handle common formats like "A: Paris" or "Answer: Paris"
    answer = re.sub(r'^[A-Z]\s*:\s*', '', answer)
    answer = re.sub(r'^[Aa]nswer\s*:?\s*', '', answer)
    
    # Split by common delimiters and take first part
    for delimiter in ['.', '!', '?', ',', ';', '(', '\n']:
        if delimiter in answer:
            answer = answer.split(delimiter)[0].strip()
            break
    
    # Remove quotes if they wrap the entire answer
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Remove trailing periods and spaces
    answer = answer.rstrip('.!? ')
    
    return answer.strip()

def evaluate_triviaqa(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_TRIVIAQA,
    dataset_config: str = DEFAULT_DATASET_CONFIG_TRIVIAQA,
    dataset_split: str = DEFAULT_SPLIT_TRIVIAQA,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_TRIVIAQA,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_TRIVIAQA,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_TRIVIAQA,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:
    
    if triviaqa_exact_match_metric is None:
        return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0, "error_message": "MetricLoadFailed"}

    logger.info(f"Starting TriviaQA ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}/{dataset_config}")
    logger.info(f"P{process_id}(GPU{gpu_id}): split='{dataset_split}', batch_size={generation_batch_size}")

    # Get prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_triviaqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("TriviaQA: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    # Load dataset
    try:
        full_data_for_split = load_dataset(dataset_name, dataset_config, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}
    
    logger.info(f"P{process_id}: Loaded TriviaQA '{dataset_name}/{dataset_config}' ({len(full_data_for_split)} examples) for split '{dataset_split}'.")

    # Handle multi-GPU processing
    if num_gpus > 1:
        total_examples = len(full_data_for_split)
        examples_per_instance = total_examples // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1: 
            end_idx = total_examples
        dataset_subset_to_process = full_data_for_split.select(range(start_idx, end_idx))
        logger.info(f"P{process_id}: Processing {len(dataset_subset_to_process)} examples (from {start_idx} to {end_idx-1}).")
    else:
        dataset_subset_to_process = full_data_for_split

    if len(dataset_subset_to_process) == 0:
        return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0, "error_message": "NoSamplesAfterSplit"}

    prompts_to_generate, current_batch_info_for_processing = [], []
    detailed_results = []
    
    for example_data in tqdm(dataset_subset_to_process, desc=f"P{process_id} - Preparing TriviaQA"):
        question_id = example_data.get('question_id', 'unknown')
        question = example_data.get('question', "")
        answer_data = example_data.get('answer', {})
        
        # Process aliases following official format with better normalization
        raw_aliases = []
        if 'aliases' in answer_data:
            raw_aliases = answer_data['aliases']
        elif 'value' in answer_data:
            raw_aliases = [answer_data['value']]
        
        if not raw_aliases:
            logger.warning(f"TriviaQA: No answer aliases found for question ID {question_id}. Skipping.")
            continue
        
        # Create both original and normalized aliases for better matching
        answer_aliases = raw_aliases
        normalized_aliases = [
            alias.lower().translate(str.maketrans("", "", string.punctuation))
            for alias in remove_prefixes(raw_aliases)
        ]
        
        # Also create additional variants for common formats
        extended_aliases = list(raw_aliases)
        for alias in raw_aliases:
            # Add version without "The" prefix
            if alias.lower().startswith('the '):
                extended_aliases.append(alias[4:])
            # Add version with first letter capitalized
            extended_aliases.append(alias.capitalize())
            # Add version all lowercase
            extended_aliases.append(alias.lower())

        if not question or not answer_aliases:
            logger.warning(f"TriviaQA: Skipping question ID {question_id} due to missing question or aliases.")
            continue
        
        # Format prompt using templates
        main_q_data = {"question": question}

        if num_few_shot > 0 and few_shot_examples_to_use:
            prompt = format_few_shot_prompt(prompt_template_dict, few_shot_examples_to_use, main_q_data)
        else:
            prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_to_generate.append(prompt)
        current_batch_info_for_processing.append({
            'question_id': question_id,
            'question': question,
            'answer_aliases': answer_aliases,
            'extended_aliases': extended_aliases,
            'normalized_aliases': normalized_aliases,
            'prompt': prompt
        })

    if not prompts_to_generate:
        logger.info(f"P{process_id}: No TriviaQA examples to process.")
        return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0}

    logger.info(f"P{process_id}: Starting TriviaQA batch inference for {len(prompts_to_generate)} prompts (batch_size={generation_batch_size}).")

    # Improved generation config for factual answers
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,  # Changed to True for better diversity
        "temperature": 0.3,  # Low but not zero for slight variation
        "top_p": 0.9,  # Nucleus sampling for better quality
        "repetition_penalty": 1.1,  # Reduce repetition
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }

    predictions_and_scores = []

    for i in tqdm(range(0, len(prompts_to_generate), generation_batch_size), desc=f"P{process_id} - Generating TriviaQA", unit="batch"):
        batch_prompts_slice = prompts_to_generate[i : i + generation_batch_size]
        batch_info_slice = current_batch_info_for_processing[i : i + generation_batch_size]
        
        try:
            with torch.no_grad():
                batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)

            for j, output_list_item in enumerate(batch_outputs_raw):
                info_item = batch_info_slice[j]
                question_id = info_item['question_id']
                answer_aliases = info_item['answer_aliases']
                normalized_aliases = info_item['normalized_aliases']
                prompt = info_item['prompt']
                
                # Extract and clean prediction
                if output_list_item and output_list_item[0] and 'generated_text' in output_list_item[0]:
                    raw_generated = output_list_item[0]['generated_text']
                    pred_text = _extract_answer_triviaqa(raw_generated, prompt)
                else:
                    raw_generated = "#GenFail"
                    pred_text = "#GenFail"
                
                # Evaluate against all aliases (including extended variants)
                all_aliases_to_check = info_item['extended_aliases']
                scores = evaluate_answer_triviaqa(pred_text, all_aliases_to_check)
                
                predictions_and_scores.append({
                    'question_id': question_id,
                    'question': info_item['question'],
                    'prediction': pred_text,
                    'answer_aliases': answer_aliases,
                    'normalized_aliases': normalized_aliases,
                    'exact_match': scores['exact_match'],
                    'f1': scores['f1'],
                    'raw_generated': raw_generated
                })
                
                # Save detailed output if requested
                if save_detailed:
                    detailed_results.append({
                        "question_id": question_id,
                        "question": info_item['question'],
                        "answer_aliases": answer_aliases,
                        "predicted_answer": pred_text,
                        "is_correct": scores['exact_match'] > 0.0,
                        "exact_match_score": scores['exact_match'],
                        "f1_score": scores['f1'],
                        "prompt": prompt,
                        "raw_response": raw_generated
                    })

        except Exception as e_batch_gen:
            logger.error(f"P{process_id}: Error during TriviaQA generation batch {i//generation_batch_size}: {e_batch_gen}", exc_info=True)
            for info_item_err in batch_info_slice:
                question_id = info_item_err['question_id']
                predictions_and_scores.append({
                    'question_id': question_id,
                    'question': info_item_err['question'],
                    'prediction': "#PipelineError",
                    'answer_aliases': info_item_err['answer_aliases'],
                    'exact_match': 0.0,
                    'f1': 0.0,
                    'raw_generated': "#PipelineError"
                })
                
                if save_detailed:
                    detailed_results.append({
                        "question_id": question_id,
                        "question": info_item_err['question'],
                        "answer_aliases": info_item_err['answer_aliases'],
                        "predicted_answer": "#PipelineError",
                        "is_correct": False,
                        "exact_match_score": 0.0,
                        "f1_score": 0.0,
                        "prompt": info_item_err['prompt'],
                        "raw_response": "#PipelineError"
                    })

    logger.info(f"P{process_id}: TriviaQA inference complete. Total items for metric: {len(predictions_and_scores)}.")

    if not predictions_and_scores:
        return {"TriviaQA": 0.0, "TriviaQA_exact_match": 0.0, "TriviaQA_f1": 0.0, "error_message": "NoPredsForMetric"}

    # Calculate overall metrics
    total_examples = len(predictions_and_scores)
    exact_match_sum = sum(item['exact_match'] for item in predictions_and_scores)
    f1_sum = sum(item['f1'] for item in predictions_and_scores)
    
    overall_exact_match = exact_match_sum / total_examples if total_examples > 0 else 0.0
    overall_f1 = f1_sum / total_examples if total_examples > 0 else 0.0

    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_triviaqa_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            num_few_shot,
            overall_exact_match * 100,
            overall_f1 * 100,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed TriviaQA results with {len(detailed_results)} examples saved to: {saved_path}")

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final TriviaQA: EM={overall_exact_match*100:.2f}%, F1={overall_f1*100:.2f}% on {len(predictions_and_scores)} examples.")
    
    return {
        "TriviaQA": overall_f1 * 100,  # Main score (F1)
        "TriviaQA_exact_match": overall_exact_match * 100,
        "TriviaQA_f1": overall_f1 * 100
    }
