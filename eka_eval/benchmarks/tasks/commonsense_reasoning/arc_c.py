# eka_eval/benchmarks/tasks/reasoning/arc_c.py - Updated with prompt system integration
import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate
from datetime import datetime

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default configuration
DEFAULT_DATASET_NAME_ARC = "allenai/ai2_arc"
DEFAULT_CONFIG_ARC_CHALLENGE = "ARC-Challenge"
DEFAULT_SPLIT_ARC = "test"  # Changed to test for evaluation
DEFAULT_MAX_NEW_TOKENS_ARC = 5
DEFAULT_GENERATION_BATCH_SIZE_ARC = 8
DEFAULT_NUM_FEWSHOT_ARC = 5
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "arc_c_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "arc_c_5shot"
PROMPT_FILE_BENCHMARK_KEY = "arc_c"
PROMPT_FILE_CATEGORY = "commonsense"

def save_detailed_arc_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    num_few_shot: int,
    accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed ARC-Challenge results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arc_c_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
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
        "extraction_failures": sum(1 for r in results_data if not r.get("extraction_successful", True))
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed ARC-Challenge results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed ARC-Challenge results: {e}")
        return ""

def _get_arc_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_arc_c",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for ARC-Challenge.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_arc_c from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def _prepare_arc_choices_string(choices_dict: Dict, format_style: str = "letter") -> str:
    """Format choices into a string."""
    choice_texts = choices_dict.get('text', [])
    choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))])
    
    if format_style == "letter":
        return "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
    elif format_style == "number":
        return "\n".join([f"{i+1}. {text}" for i, text in enumerate(choice_texts)])
    elif format_style == "simple":
        return "\n".join(choice_texts)
    else:
        return "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])

def _extract_arc_answer_letter(generated_text: str, prompt_text_sent_to_llm: Optional[str] = None) -> Optional[str]:
    """Extract the answer letter (A-E) from generated text with enhanced debugging."""
    completion_part = generated_text
    if prompt_text_sent_to_llm and generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    
    completion_part = completion_part.strip()
    
    # Log what we're trying to extract from
    logger.debug(f"ARC-Challenge extraction input: '{completion_part[:100]}'")
    
    if len(completion_part) == 0:
        logger.debug("ARC-Challenge: Empty completion, extraction failed")
        return None
    
    # Try multiple extraction patterns
    patterns = [
        r'^([A-E])$',                          # Just the letter
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
            logger.debug(f"ARC-Challenge: Extracted '{letter}' using pattern {i+1}: {pattern}")
            return letter
    
    logger.debug(f"ARC-Challenge: No pattern matched for: '{completion_part[:50]}'")
    return None

def _map_arc_answerkey_to_int(answer_key_str: str, choice_labels: List[str]) -> int:
    """Map answer key to index, handling both letter and number formats."""
    ak_str = str(answer_key_str).strip().upper()
    
    # Handle letter labels (A, B, C...)
    if ak_str in choice_labels:
        return choice_labels.index(ak_str)
    
    # Handle numeric labels (1, 2, 3...)
    if ak_str.isdigit():
        num_val = int(ak_str)
        if 1 <= num_val <= len(choice_labels):
            return num_val - 1
    
    # Handle cases where labels are numbers but answer is letter
    if ak_str in ['A', 'B', 'C', 'D', 'E'] and choice_labels and all(l.isdigit() for l in choice_labels):
        letter_idx = ord(ak_str) - ord('A')
        if 0 <= letter_idx < len(choice_labels):
            return letter_idx
    
    logger.warning(f"Could not map answerKey '{ak_str}' with labels {choice_labels}")
    return -1

def evaluate_arc_challenge(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_ARC,
    dataset_config_name: str = DEFAULT_CONFIG_ARC_CHALLENGE,
    dataset_split: str = DEFAULT_SPLIT_ARC,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_ARC,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_ARC,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_ARC,
    choices_format: str = "letter",
    few_shot_examples_key: Optional[str] = None,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:

    try:
        arc_accuracy_metric = hf_evaluate.load("accuracy")
        logger.info("Accuracy metric for ARC-Challenge loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load 'accuracy' metric for ARC-Challenge: {e}")
        return {"ARC-Challenge": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting ARC-Challenge ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}/{dataset_config_name}")
    logger.info(f"Generation config: max_new_tokens={max_new_tokens}, batch_size={generation_batch_size}")

    # Load prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"ARC-Challenge": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_arc_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("ARC-Challenge: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"ARC-Challenge": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    # Load dataset
    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"ARC-Challenge": 0.0, "error_message": f"DatasetLoadFailed: {e}"}
    
    logger.info(f"P{process_id}: Loaded ARC-Challenge ({len(full_data)} examples).")

    # Handle multi-GPU data splitting
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
        return {"ARC-Challenge": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} ARC-Challenge examples.")

    # Initialize tracking
    predictions, true_labels = [], []
    prompts_for_batch, original_items_for_batch_info = [], []
    detailed_results = []

    # Main evaluation loop
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - ARC-Challenge Eval")):
        question = item_data.get("question")
        choices_dict = item_data.get("choices", {})
        answer_key = item_data.get("answerKey")
        
        choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choices_dict.get('text', [])))])
        true_answer_idx = _map_arc_answerkey_to_int(str(answer_key).strip(), choice_labels)
        
        if true_answer_idx == -1:
            logger.warning(f"Invalid answer key format for item {item_idx}: {answer_key}")
            continue
        
        true_answer_letter = choice_labels[true_answer_idx] if 0 <= true_answer_idx < len(choice_labels) else "A"
        
        if not all([question, choices_dict.get('text')]):
            logger.warning(f"Missing data for item {item_idx}")
            continue

        # Format choices
        choices_str_fmt = _prepare_arc_choices_string(choices_dict, choices_format)
        main_q_data = {"question": question, "choices_str": choices_str_fmt}
        
        # Generate prompt
        if num_few_shot > 0 and few_shot_examples_to_use:
            llm_prompt = format_few_shot_prompt(prompt_template_dict, few_shot_examples_to_use, main_q_data)
        else:
            llm_prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_for_batch.append(llm_prompt)
        original_items_for_batch_info.append({
            'true_answer_idx': true_answer_idx,
            'true_answer_letter': true_answer_letter,
            'llm_prompt': llm_prompt,
            'question': question,
            'choices_dict': choices_dict,
            'choice_labels': choice_labels,
            'item_idx': item_idx
        })

        # Process batch
        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True,
                "temperature": None,
                "top_p": None,
            }
            
            # Log generation config for first batch
            if item_idx < generation_batch_size:
                logger.info(f"Generation config: {gen_config}")
            
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    orig_info = original_items_for_batch_info[k]
                    
                    # Extract generated text with better handling
                    generated_full_text = ""
                    generated_part = ""
                    
                    if raw_out_list and len(raw_out_list) > 0 and 'generated_text' in raw_out_list[0]:
                        generated_full_text = raw_out_list[0]['generated_text']
                        # Extract only the new part
                        if generated_full_text.startswith(orig_info['llm_prompt']):
                            generated_part = generated_full_text[len(orig_info['llm_prompt']):]
                        else:
                            generated_part = generated_full_text
                    else:
                        generated_full_text = "[NO_GENERATION]"
                        generated_part = "[NO_GENERATION]"
                    
                    # Extract answer with debugging
                    pred_letter = _extract_arc_answer_letter(generated_full_text, orig_info['llm_prompt'])
                    true_letter = orig_info['true_answer_letter']
                    
                    # Handle failed extraction - use random fallback
                    extraction_successful = pred_letter is not None
                    if pred_letter is None:
                        import random
                        pred_letter = random.choice(orig_info['choice_labels'])
                        logger.debug(f"Extraction failed, using random fallback: {pred_letter}")
                    
                    is_correct = pred_letter == true_letter
                    predictions.append(pred_letter)
                    true_labels.append(true_letter)
                    
                    # Enhanced debug output for first few examples
                    if orig_info['item_idx'] < 3:
                        print(f"\n=== ENHANCED DEBUG ARC-Challenge Example {orig_info['item_idx']} ===")
                        print(f"Question: {orig_info['question'][:100]}...")
                        print(f"Choices: {orig_info['choices_dict']['text']}")
                        print(f"--- PROMPT (first 200 chars) ---")
                        print(f"'{orig_info['llm_prompt'][:200]}...'")
                        print(f"--- FULL GENERATION ---")
                        print(f"'{generated_full_text}'")
                        print(f"--- EXTRACTED PART ---")
                        print(f"'{generated_part.strip()}'")
                        print(f"--- RESULTS ---")
                        print(f"Extracted: {pred_letter} (Success: {extraction_successful})")
                        print(f"Expected: {true_letter}")
                        print(f"Correct: {'✅' if is_correct else '❌'}")
                        print("=" * 60)
                    
                    # Store detailed result
                    if save_detailed:
                        choice_texts = orig_info['choices_dict'].get('text', [])
                        pred_idx = orig_info['choice_labels'].index(pred_letter) if pred_letter in orig_info['choice_labels'] else -1
                        
                        detailed_result = {
                            "question_id": orig_info['item_idx'],
                            "question": orig_info['question'],
                            "choices": choice_texts,
                            "choice_labels": orig_info['choice_labels'],
                            "correct_answer_letter": true_letter,
                            "correct_answer_text": choice_texts[orig_info['true_answer_idx']] if 0 <= orig_info['true_answer_idx'] < len(choice_texts) else "Unknown",
                            "predicted_answer_letter": pred_letter,
                            "predicted_answer_text": choice_texts[pred_idx] if 0 <= pred_idx < len(choice_texts) else "Unknown",
                            "generated_text_full": generated_full_text,
                            "generated_text_part": generated_part.strip(),
                            "is_correct": is_correct,
                            "prompt_used": orig_info['llm_prompt'],
                            "extraction_successful": extraction_successful
                        }
                        detailed_results.append(detailed_result)
                        
            except Exception as e_batch:
                logger.error(f"P{process_id}: ARC-Challenge generation batch error: {e_batch}", exc_info=True)
                # Add fallback predictions for failed batch
                for info in original_items_for_batch_info:
                    predictions.append("A")
                    true_labels.append(info['true_answer_letter'])
                    
                    if save_detailed:
                        detailed_result = {
                            "question_id": info['item_idx'],
                            "question": info['question'],
                            "choices": info['choices_dict'].get('text', []),
                            "choice_labels": info['choice_labels'],
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
            
            # Reset batch
            prompts_for_batch, original_items_for_batch_info = [], []
    
    # Calculate accuracy
    if not true_labels:
        return {"ARC-Challenge": 0.0}
    
    # Convert letters to indices for accuracy calculation
    predictions_numeric = []
    true_labels_numeric = []
    
    for pred, true in zip(predictions, true_labels):
        # Use first choice labels as reference
        reference_labels = ['A', 'B', 'C', 'D', 'E']
        pred_idx = reference_labels.index(pred) if pred in reference_labels else 0
        true_idx = reference_labels.index(true) if true in reference_labels else 0
        predictions_numeric.append(pred_idx)
        true_labels_numeric.append(true_idx)
    
    try:
        acc_results = arc_accuracy_metric.compute(predictions=predictions_numeric, references=true_labels_numeric)
        accuracy = acc_results.get("accuracy", 0.0) * 100
    except Exception as e:
        logger.error(f"Accuracy computation failed: {e}")
        accuracy = 0.0
    
    # Calculate extraction success rate
    extraction_success_rate = sum(1 for r in detailed_results if r.get("extraction_successful", False)) / len(detailed_results) * 100 if detailed_results else 0
    
    logger.info(f"P{process_id} - Final ARC-Challenge Accuracy: {accuracy:.2f}% on {len(true_labels)} examples.")
    logger.info(f"P{process_id} - Extraction Success Rate: {extraction_success_rate:.2f}%")
    
    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_arc_results(
            detailed_results,
            model_name_for_logging,
            f"{dataset_name}_{dataset_config_name}",
            num_few_shot,
            accuracy,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed ARC-Challenge results with {len(detailed_results)} examples saved to: {saved_path}")
    
    return {"ARC-Challenge": accuracy}
