# eka_eval/benchmarks/tasks/knowledge/naturalqa.py

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

DEFAULT_DATASET_NAME_NQ = "natural_questions"
DEFAULT_SPLIT_NQ = "validation"
DEFAULT_MAX_NEW_TOKENS_NQ = 32
DEFAULT_FEW_SHOT_COUNT_NQ = 5
DEFAULT_GENERATION_BATCH_SIZE_NQ = 8
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "naturalqa_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "naturalqa_5shot"
PROMPT_FILE_BENCHMARK_KEY = "naturalqa"
PROMPT_FILE_CATEGORY = "knowledge"

try:
    nq_exact_match_metric = hf_evaluate.load("exact_match")
    logger.info("Exact match metric for Natural Questions loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'exact_match' metric for Natural Questions: {e}. NQ may not run correctly.", exc_info=True)
    nq_exact_match_metric = None

def save_detailed_naturalqa_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    num_few_shot: int,
    exact_match: float,
    f1_score: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed Natural Questions results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"naturalqa_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
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
        "process_id": process_id,
        "answerable_questions": sum(1 for r in results_data if not r["is_unanswerable"]),
        "unanswerable_questions": sum(1 for r in results_data if r["is_unanswerable"])
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed Natural Questions results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed Natural Questions results: {e}")
        return ""

def _get_naturalqa_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_naturalqa",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for Natural Questions.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_naturalqa from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def normalize_answer_nq(s: str) -> str:
    """Normalize answer following Natural Questions evaluation protocol."""
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

def exact_match_score_nq(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer_nq(prediction) == normalize_answer_nq(ground_truth)

def f1_score_nq(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer_nq(prediction).split()
    truth_tokens = normalize_answer_nq(ground_truth).split()
    
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

def evaluate_answer_nq(prediction: str, ground_truth_answers: List[str], is_unanswerable: bool = False) -> Dict[str, float]:
    """
    Evaluate prediction against all possible ground truth answers for Natural Questions.
    Handles unanswerable cases.
    """
    normalized_prediction = normalize_answer_nq(prediction)
    
    # Handle unanswerable questions
    if is_unanswerable:
        # For unanswerable questions, empty or "no answer" type responses are correct
        if not normalized_prediction or normalized_prediction in ['no', 'none', 'no answer', 'unanswerable']:
            return {'exact_match': 1.0, 'f1': 1.0}
        else:
            return {'exact_match': 0.0, 'f1': 0.0}
    
    # Handle answerable questions
    if not ground_truth_answers:
        return {'exact_match': 0.0, 'f1': 0.0}

    em_scores = []
    f1_scores = []

    for gt_answer in ground_truth_answers:
        if not isinstance(gt_answer, str):
            continue
        
        # Exact Match
        em_scores.append(1.0 if exact_match_score_nq(prediction, gt_answer) else 0.0)
        
        # F1 Score
        f1_scores.append(f1_score_nq(prediction, gt_answer))

    return {
        'exact_match': max(em_scores) if em_scores else 0.0,
        'f1': max(f1_scores) if f1_scores else 0.0
    }

def _extract_answer_nq(generated_text: str, prompt: str) -> str:
    """Extract answer from generated response for Natural Questions."""
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

def extract_ground_truth_from_annotations(annotations) -> Tuple[List[str], bool]:
    """Extract ground truth answers from Natural Questions annotations."""
    if not annotations or not isinstance(annotations, list):
        return [], True
    
    all_short_answers = set()
    all_yes_no_answers = set()
    
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
            
        # Extract short answers
        short_answers = annotation.get('short_answers', [])
        if isinstance(short_answers, list):
            for sa in short_answers:
                if isinstance(sa, dict) and 'text' in sa:
                    if isinstance(sa['text'], list):
                        for text in sa['text']:
                            if isinstance(text, str) and text.strip():
                                all_short_answers.add(text.strip())
                    elif isinstance(sa['text'], str) and sa['text'].strip():
                        all_short_answers.add(sa['text'].strip())
        
        # Extract yes/no answers
        yes_no_answer = annotation.get('yes_no_answer', -1)
        if isinstance(yes_no_answer, int):
            if yes_no_answer == 0:
                all_yes_no_answers.add("no")
            elif yes_no_answer == 1:
                all_yes_no_answers.add("yes")
    
    # Determine final ground truth
    if all_short_answers:
        return sorted(list(all_short_answers)), False
    elif all_yes_no_answers:
        return sorted(list(all_yes_no_answers)), False
    else:
        return [], True

def evaluate_naturalqa(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_NQ,
    dataset_split: str = DEFAULT_SPLIT_NQ,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_NQ,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_NQ,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_NQ,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:
    
    if nq_exact_match_metric is None:
        return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0, "error_message": "MetricLoadFailed"}

    logger.info(f"Starting Natural Questions ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")
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
        return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_naturalqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("Natural Questions: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    # Load dataset
    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}
    
    logger.info(f"P{process_id}: Loaded Natural Questions '{dataset_name}' ({len(full_data_for_split)} examples) for split '{dataset_split}'.")

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
        return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0, "error_message": "NoSamplesAfterSplit"}

    prompts_to_generate, current_batch_info_for_processing = [], []
    detailed_results = []
    
    for example_data in tqdm(dataset_subset_to_process, desc=f"P{process_id} - Preparing Natural Questions"):
        example_id = example_data.get('example_id', 'unknown')
        
        # Extract question text
        question_data = example_data.get('question', {})
        if isinstance(question_data, dict):
            question_text = question_data.get('text', '')
        else:
            question_text = str(question_data) if question_data else ''
        
        if not question_text:
            logger.warning(f"Natural Questions: No question text found for example ID {example_id}. Skipping.")
            continue

        # Extract ground truth from annotations
        annotations = example_data.get('annotations', [])
        ground_truth_answers, is_unanswerable = extract_ground_truth_from_annotations(annotations)
        
        # Format prompt using templates
        main_q_data = {"question": question_text}

        if num_few_shot > 0 and few_shot_examples_to_use:
            prompt = format_few_shot_prompt(prompt_template_dict, few_shot_examples_to_use, main_q_data)
        else:
            prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_to_generate.append(prompt)
        current_batch_info_for_processing.append({
            'example_id': example_id,
            'question': question_text,
            'ground_truth_answers': ground_truth_answers,
            'is_unanswerable': is_unanswerable,
            'prompt': prompt
        })

    if not prompts_to_generate:
        logger.info(f"P{process_id}: No Natural Questions examples to process.")
        return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0}

    logger.info(f"P{process_id}: Starting Natural Questions batch inference for {len(prompts_to_generate)} prompts (batch_size={generation_batch_size}).")

    # Generation config optimized for factual answers
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.3,  # Low but not zero for slight variation
        "top_p": 0.9,  # Nucleus sampling for better quality
        "repetition_penalty": 1.1,  # Reduce repetition
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }

    predictions_and_scores = []

    for i in tqdm(range(0, len(prompts_to_generate), generation_batch_size), desc=f"P{process_id} - Generating Natural Questions", unit="batch"):
        batch_prompts_slice = prompts_to_generate[i : i + generation_batch_size]
        batch_info_slice = current_batch_info_for_processing[i : i + generation_batch_size]
        
        try:
            with torch.no_grad():
                batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)

            for j, output_list_item in enumerate(batch_outputs_raw):
                info_item = batch_info_slice[j]
                example_id = info_item['example_id']
                ground_truth_answers = info_item['ground_truth_answers']
                is_unanswerable = info_item['is_unanswerable']
                prompt = info_item['prompt']
                
                # Extract and clean prediction
                if output_list_item and output_list_item[0] and 'generated_text' in output_list_item[0]:
                    raw_generated = output_list_item[0]['generated_text']
                    pred_text = _extract_answer_nq(raw_generated, prompt)
                else:
                    raw_generated = "#GenFail"
                    pred_text = "#GenFail"
                
                # Evaluate against ground truth
                scores = evaluate_answer_nq(pred_text, ground_truth_answers, is_unanswerable)
                
                predictions_and_scores.append({
                    'example_id': example_id,
                    'question': info_item['question'],
                    'prediction': pred_text,
                    'ground_truth_answers': ground_truth_answers,
                    'is_unanswerable': is_unanswerable,
                    'exact_match': scores['exact_match'],
                    'f1': scores['f1'],
                    'raw_generated': raw_generated
                })
                
                # Save detailed output if requested
                if save_detailed:
                    detailed_results.append({
                        "example_id": example_id,
                        "question": info_item['question'],
                        "ground_truth_answers": ground_truth_answers,
                        "is_unanswerable": is_unanswerable,
                        "predicted_answer": pred_text,
                        "is_correct": scores['exact_match'] > 0.0,
                        "exact_match_score": scores['exact_match'],
                        "f1_score": scores['f1'],
                        "prompt": prompt,
                        "raw_response": raw_generated
                    })

        except Exception as e_batch_gen:
            logger.error(f"P{process_id}: Error during Natural Questions generation batch {i//generation_batch_size}: {e_batch_gen}", exc_info=True)
            for info_item_err in batch_info_slice:
                example_id = info_item_err['example_id']
                predictions_and_scores.append({
                    'example_id': example_id,
                    'question': info_item_err['question'],
                    'prediction': "#PipelineError",
                    'ground_truth_answers': info_item_err['ground_truth_answers'],
                    'is_unanswerable': info_item_err['is_unanswerable'],
                    'exact_match': 0.0,
                    'f1': 0.0,
                    'raw_generated': "#PipelineError"
                })
                
                if save_detailed:
                    detailed_results.append({
                        "example_id": example_id,
                        "question": info_item_err['question'],
                        "ground_truth_answers": info_item_err['ground_truth_answers'],
                        "is_unanswerable": info_item_err['is_unanswerable'],
                        "predicted_answer": "#PipelineError",
                        "is_correct": False,
                        "exact_match_score": 0.0,
                        "f1_score": 0.0,
                        "prompt": info_item_err['prompt'],
                        "raw_response": "#PipelineError"
                    })

    logger.info(f"P{process_id}: Natural Questions inference complete. Total items for metric: {len(predictions_and_scores)}.")

    if not predictions_and_scores:
        return {"NaturalQuestions": 0.0, "NQ_exact_match": 0.0, "NQ_f1": 0.0, "error_message": "NoPredsForMetric"}

    # Calculate overall metrics
    total_examples = len(predictions_and_scores)
    exact_match_sum = sum(item['exact_match'] for item in predictions_and_scores)
    f1_sum = sum(item['f1'] for item in predictions_and_scores)
    
    overall_exact_match = exact_match_sum / total_examples if total_examples > 0 else 0.0
    overall_f1 = f1_sum / total_examples if total_examples > 0 else 0.0

    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_naturalqa_results(
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
            logger.info(f"Detailed Natural Questions results with {len(detailed_results)} examples saved to: {saved_path}")

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final Natural Questions: EM={overall_exact_match*100:.2f}%, F1={overall_f1*100:.2f}% on {len(predictions_and_scores)} examples.")
    
    return {
        "NaturalQuestions": overall_f1 * 100,  # Main score (F1)
        "NQ_exact_match": overall_exact_match * 100,
        "NQ_f1": overall_f1 * 100
    }

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test Natural Questions")
    test_parser.add_argument("--model_name_test", type=str, default="meta-llama/Meta-Llama-3-8B")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:100]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=4)
    test_parser.add_argument("--num_few_shot_test", type=int, default=3)
    test_parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum new tokens to generate")
    test_parser.add_argument("--save_detailed", action="store_true", help="Save detailed outputs to JSON file")
    
    nq_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="NaturalQuestionsFileTest")
    logger.info(f"--- Standalone Natural Questions Test: {nq_args.model_name_test} ({nq_args.num_few_shot_test}-shot) ---")
    
    nq_pipe, _ = initialize_model_pipeline(nq_args.model_name_test, target_device_id=0)
    if nq_pipe:
        nq_eval_args = {
            "pipe": nq_pipe,
            "tokenizer": nq_pipe.tokenizer,
            "model_name_for_logging": nq_args.model_name_test,
            "device": nq_pipe.device,
            "dataset_split": nq_args.dataset_split_test,
            "generation_batch_size": nq_args.gen_batch_size_test,
            "num_few_shot": nq_args.num_few_shot_test,
            "max_new_tokens": nq_args.max_new_tokens,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_detailed": nq_args.save_detailed
        }
        try:
            print(json.dumps(evaluate_naturalqa(**nq_eval_args), indent=2))
        finally:
            cleanup_model_resources(nq_pipe, getattr(nq_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {nq_args.model_name_test} for Natural Questions test.")