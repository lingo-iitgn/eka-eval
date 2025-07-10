# eka_eval/benchmarks/tasks/long_context/infinitebench.py

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

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_IB = "xinrongzhang2022/InfiniteBench"
DEFAULT_MAX_NEW_TOKENS_IB = 100
DEFAULT_FEW_SHOT_COUNT_IB = 3

try:
    infinitebench_squad_metric = hf_evaluate.load("squad")  # For EN.QA F1 scores
    infinitebench_accuracy_metric = hf_evaluate.load("accuracy")  # For EN.MC accuracy
    logger.info("Metrics for InfiniteBench loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load metrics for InfiniteBench: {e}. InfiniteBench may not run correctly.", exc_info=True)
    infinitebench_squad_metric = None
    infinitebench_accuracy_metric = None

# Few-shot examples for InfiniteBench tasks
DEFAULT_FEW_SHOT_EXAMPLES_EN_QA = [
    {
        "context": "The old mansion stood at the end of the winding road, its gothic architecture casting long shadows in the moonlight. Sarah approached the heavy wooden door, her heart pounding with anticipation.",
        "question": "What type of architecture did the mansion have?",
        "answer": "gothic architecture"
    },
    {
        "context": "Professor Martinez had spent years researching renewable energy sources. His latest breakthrough involved a new type of solar panel that could generate electricity even on cloudy days.",
        "question": "What was Professor Martinez's area of research?",
        "answer": "renewable energy sources"
    },
    {
        "context": "The small café on the corner was famous for its homemade pastries. Every morning, the baker would arrive at 4 AM to prepare fresh croissants and Danish pastries for the early customers.",
        "question": "What time did the baker arrive at work?",
        "answer": "4 AM"
    }
]

DEFAULT_FEW_SHOT_EXAMPLES_EN_MC = [
    {
        "context": "The expedition team discovered an ancient temple hidden deep in the Amazon rainforest. The temple contained mysterious hieroglyphs that had never been seen before.",
        "question": "Where was the ancient temple discovered?",
        "choices": ["A. In the mountains", "B. In the Amazon rainforest", "C. In the desert", "D. On an island"],
        "answer": "B"
    },
    {
        "context": "Dr. Chen's laboratory was equipped with the latest scientific instruments. She was conducting experiments on plant genetics to develop drought-resistant crops.",
        "question": "What was Dr. Chen studying?",
        "choices": ["A. Animal behavior", "B. Astronomy", "C. Plant genetics", "D. Chemistry"],
        "answer": "C"
    },
    {
        "context": "The vintage motorcycle had been restored to its original condition. Its chrome parts gleamed in the sunlight, and the engine purred like a contented cat.",
        "question": "How did the engine sound?",
        "choices": ["A. Like thunder", "B. Like a contented cat", "C. Like wind", "D. Like music"],
        "answer": "B"
    }
]

def normalize_answer_infinitebench(s: str) -> str:
    """Normalize answer following InfiniteBench evaluation protocol."""
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
        prefixes = [
            r'^the\s+answer\s+is\s+', r'^answer\s*:\s*',
            r'^the\s+correct\s+answer\s+is\s+', r'^option\s+',
            r'^choice\s+', r'^letter\s+'
        ]
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        return text
    
    if not isinstance(s, str):
        return ""
    
    result = s
    result = remove_common_prefixes(result)
    result = lower(result)
    result = remove_punc(result)
    result = remove_articles(result)
    result = white_space_fix(result)
    
    return result

def exact_match_score_ib(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer_infinitebench(prediction) == normalize_answer_infinitebench(ground_truth)

def f1_score_ib(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer_infinitebench(prediction).split()
    truth_tokens = normalize_answer_infinitebench(ground_truth).split()
    
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

def _format_infinitebench_en_qa_prompt(context: str, question: str, few_shot_examples: List[Dict]) -> str:
    """Format prompt for InfiniteBench EN.QA task."""
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        prompt += "Read the text carefully and answer the questions based on the information provided.\n\n"
        for ex_item in few_shot_examples:
            ex_context = ex_item.get('context', '').strip()
            ex_question = ex_item.get('question', '').strip()
            ex_answer = ex_item.get('answer', '').strip()
            
            prompt += f"Text: {ex_context}\nQuestion: {ex_question}\nAnswer: {ex_answer}\n\n"
    
    # Add the target question
    prompt += f"Text: {context}\nQuestion: {question}\nAnswer:"
    
    return prompt

def _format_infinitebench_en_mc_prompt(context: str, question: str, choices: List[str], few_shot_examples: List[Dict]) -> str:
    """Format prompt for InfiniteBench EN.MC task."""
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        prompt += "Read the text and choose the correct answer from the given options.\n\n"
        for ex_item in few_shot_examples:
            ex_context = ex_item.get('context', '').strip()
            ex_question = ex_item.get('question', '').strip()
            ex_choices = ex_item.get('choices', [])
            ex_answer = ex_item.get('answer', '').strip()
            
            prompt += f"Text: {ex_context}\nQuestion: {ex_question}\n"
            for choice in ex_choices:
                prompt += f"{choice}\n"
            prompt += f"Answer: {ex_answer}\n\n"
    
    # Add the target question
    prompt += f"Text: {context}\nQuestion: {question}\n"
    for choice in choices:
        prompt += f"{choice}\n"
    prompt += "Answer:"
    
    return prompt

def _extract_answer_en_qa(generated_text: str, prompt: str) -> str:
    """Extract answer for EN.QA task."""
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    # Remove common prefixes
    response = re.sub(r'^[Aa]nswer\s*:?\s*', '', response)
    response = re.sub(r'^(The answer is|It is|That would be|The correct answer is)\s*', '', response, flags=re.IGNORECASE)
    
    # Take only the first sentence/line as answer
    lines = response.split('\n')
    answer = lines[0].strip()
    
    # Split by sentence and take first
    sentences = re.split(r'[.!?]', answer)
    answer = sentences[0].strip() if sentences else answer
    
    # Remove quotes if they wrap the entire answer
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    return answer.strip()

def _extract_answer_en_mc(generated_text: str, prompt: str) -> str:
    """Extract answer for EN.MC task (looking for A, B, C, D)."""
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    # Remove common prefixes
    response = re.sub(r'^[Aa]nswer\s*:?\s*', '', response)
    response = re.sub(r'^(The answer is|The correct answer is|Option|Choice)\s*', '', response, flags=re.IGNORECASE)
    
    # Look for A, B, C, D pattern
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1).upper()
    
    # Fallback: take first character if it's A-D
    if response and response[0].upper() in 'ABCD':
        return response[0].upper()
    
    return "A"  # Default fallback

def parse_infinitebench_input(input_text: str, task_type: str) -> Tuple[str, str, List[str]]:
    """Parse InfiniteBench input format to extract context, question, and choices."""
    context = ""
    question = ""
    choices = []
    
    if task_type == "longbook_qa_eng":
        # Format: Context + Question
        if "\nQuestion:" in input_text:
            parts = input_text.split("\nQuestion:")
            context = parts[0].strip()
            question_part = parts[1].strip()
            if "\nAnswer:" in question_part:
                question = question_part.split("\nAnswer:")[0].strip()
            else:
                question = question_part
        else:
            # Fallback
            context = input_text
            question = "What is the main topic discussed?"
    
    elif task_type == "longbook_choice_eng":
        # Format: Context + Question + Multiple choices
        if "\nQuestion:" in input_text:
            parts = input_text.split("\nQuestion:")
            context = parts[0].strip()
            question_and_choices = parts[1].strip()
            
            # Extract question (before first choice)
            choice_pattern = r'\n[A-D]\.'
            choice_match = re.search(choice_pattern, question_and_choices)
            if choice_match:
                question = question_and_choices[:choice_match.start()].strip()
                choices_text = question_and_choices[choice_match.start():].strip()
                
                # Extract choices
                choice_lines = choices_text.split('\n')
                for line in choice_lines:
                    line = line.strip()
                    if re.match(r'^[A-D]\.', line):
                        choices.append(line)
            else:
                question = question_and_choices
        else:
            # Fallback
            context = input_text
            question = "Choose the correct option:"
            choices = ["A. Option A", "B. Option B", "C. Option C", "D. Option D"]
    
    return context, question, choices

def evaluate_infinitebench(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_IB,
    dataset_split: str = "longbook_qa_eng",  # or "longbook_choice_eng"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_IB,
    generation_batch_size: int = 1,  # Keep small for long context
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_IB,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:
    
    if infinitebench_squad_metric is None or infinitebench_accuracy_metric is None:
        return {"InfiniteBench": 0.0, "error_message": "MetricLoadFailed"}

    task_type = dataset_split
    logger.info(f"Starting InfiniteBench ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}/{task_type}")
    logger.info(f"P{process_id}(GPU{gpu_id}): split='{dataset_split}', batch_size={generation_batch_size}")

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"InfiniteBench": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}/{dataset_split}"}
    
    logger.info(f"P{process_id}: Loaded InfiniteBench '{dataset_name}/{task_type}' ({len(full_data_for_split)} examples).")

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
        return {"InfiniteBench": 0.0, "error_message": "NoSamplesAfterSplit"}

    # Select appropriate few-shot examples
    if "qa" in task_type:
        few_shot_examples_list = DEFAULT_FEW_SHOT_EXAMPLES_EN_QA[:num_few_shot] if num_few_shot > 0 else []
    else:
        few_shot_examples_list = DEFAULT_FEW_SHOT_EXAMPLES_EN_MC[:num_few_shot] if num_few_shot > 0 else []

    prompts_to_generate, current_batch_info_for_processing = [], []
    outputs_dump = []
    
    for example_data in tqdm(dataset_subset_to_process, desc=f"P{process_id} - Preparing InfiniteBench"):
        input_text = example_data.get('input', '')
        output_text = example_data.get('output', '').strip()
        
        if not input_text or not output_text:
            logger.warning(f"InfiniteBench: Skipping example due to missing input or output.")
            continue

        # Parse input based on task type
        context, question, choices = parse_infinitebench_input(input_text, task_type)
        
        # Create appropriate prompt
        if "qa" in task_type:
            prompt = _format_infinitebench_en_qa_prompt(context, question, few_shot_examples_list)
        else:  # choice task
            prompt = _format_infinitebench_en_mc_prompt(context, question, choices, few_shot_examples_list)
        
        prompts_to_generate.append(prompt)
        current_batch_info_for_processing.append({
            'context': context,
            'question': question,
            'choices': choices,
            'reference_answer': output_text,
            'task_type': task_type,
            'prompt': prompt
        })

    if not prompts_to_generate:
        logger.info(f"P{process_id}: No InfiniteBench examples to process.")
        return {"InfiniteBench": 0.0}

    logger.info(f"P{process_id}: Starting InfiniteBench batch inference for {len(prompts_to_generate)} prompts (batch_size={generation_batch_size}).")

    # Generation config optimized for long context
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }

    predictions_and_scores = []

    for i in tqdm(range(0, len(prompts_to_generate), generation_batch_size), desc=f"P{process_id} - Generating InfiniteBench", unit="batch"):
        batch_prompts_slice = prompts_to_generate[i : i + generation_batch_size]
        batch_info_slice = current_batch_info_for_processing[i : i + generation_batch_size]
        
        try:
            with torch.no_grad():
                batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)

            for j, output_list_item in enumerate(batch_outputs_raw):
                info_item = batch_info_slice[j]
                reference_answer = info_item['reference_answer']
                task_type = info_item['task_type']
                prompt = info_item['prompt']
                
                # Extract and clean prediction
                if output_list_item and output_list_item[0] and 'generated_text' in output_list_item[0]:
                    raw_generated = output_list_item[0]['generated_text']
                    if "qa" in task_type:
                        pred_text = _extract_answer_en_qa(raw_generated, prompt)
                    else:
                        pred_text = _extract_answer_en_mc(raw_generated, prompt)
                else:
                    raw_generated = "#GenFail"
                    pred_text = "#GenFail"
                
                # Evaluate based on task type
                if "qa" in task_type:
                    # Use F1 score for QA task
                    f1 = f1_score_ib(pred_text, reference_answer)
                    em = 1.0 if exact_match_score_ib(pred_text, reference_answer) else 0.0
                    score = f1  # Primary metric for QA
                else:
                    # Use accuracy for MC task
                    em = 1.0 if pred_text.upper() == reference_answer.upper() else 0.0
                    f1 = em  # For MC, F1 = EM
                    score = em  # Primary metric for MC
                
                predictions_and_scores.append({
                    'question': info_item['question'],
                    'prediction': pred_text,
                    'reference': reference_answer,
                    'exact_match': em,
                    'f1': f1,
                    'score': score,
                    'task_type': task_type,
                    'raw_generated': raw_generated
                })
                
                # Save detailed output if requested
                if save_outputs:
                    outputs_dump.append({
                        "context": info_item['context'][:500] + "..." if len(info_item['context']) > 500 else info_item['context'],
                        "question": info_item['question'],
                        "choices": info_item['choices'],
                        "reference_answer": reference_answer,
                        "predicted_answer": pred_text,
                        "is_correct": em > 0.0,
                        "exact_match_score": em,
                        "f1_score": f1,
                        "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        "raw_response": raw_generated
                    })

        except Exception as e_batch_gen:
            logger.error(f"P{process_id}: Error during InfiniteBench generation batch {i//generation_batch_size}: {e_batch_gen}", exc_info=True)
            for info_item_err in batch_info_slice:
                predictions_and_scores.append({
                    'question': info_item_err['question'],
                    'prediction': "#PipelineError",
                    'reference': info_item_err['reference_answer'],
                    'exact_match': 0.0,
                    'f1': 0.0,
                    'score': 0.0,
                    'task_type': info_item_err['task_type'],
                    'raw_generated': "#PipelineError"
                })

    logger.info(f"P{process_id}: InfiniteBench inference complete. Total items: {len(predictions_and_scores)}.")

    if not predictions_and_scores:
        return {"InfiniteBench": 0.0, "error_message": "NoPredsForMetric"}

    # Calculate overall metrics
    total_examples = len(predictions_and_scores)
    avg_score = sum(item['score'] for item in predictions_and_scores) / total_examples if total_examples > 0 else 0.0
    avg_em = sum(item['exact_match'] for item in predictions_and_scores) / total_examples if total_examples > 0 else 0.0
    avg_f1 = sum(item['f1'] for item in predictions_and_scores) / total_examples if total_examples > 0 else 0.0

    # Save outputs to JSON file if requested
    if save_outputs and outputs_dump:
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"infinitebench_{task_type}_outputs_{model_name_for_logging.replace('/', '_')}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        summary_data = {
            "model_name": model_name_for_logging,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "task_type": task_type,
            "num_few_shot": num_few_shot,
            "total_examples": len(outputs_dump),
            "primary_score": avg_score * 100,
            "exact_match": avg_em * 100,
            "f1_score": avg_f1 * 100,
            "correct_predictions": sum(1 for item in outputs_dump if item["is_correct"]),
            "process_id": process_id,
            "gpu_id": gpu_id,
            "examples": outputs_dump
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"P{process_id}: Saved {len(outputs_dump)} InfiniteBench outputs to {output_path}")
        except Exception as e_save:
            logger.error(f"P{process_id}: Error saving InfiniteBench outputs: {e_save}")

    # Determine primary metric name based on task
    if "qa" in task_type:
        metric_name = "F1"
        primary_score = avg_f1 * 100
    else:
        metric_name = "Accuracy"
        primary_score = avg_em * 100

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final InfiniteBench {task_type}: {metric_name}={primary_score:.2f}% on {len(predictions_and_scores)} examples.")
    
    return {
        "InfiniteBench": primary_score,
        f"InfiniteBench_{task_type}": primary_score,
        "InfiniteBench_exact_match": avg_em * 100,
        "InfiniteBench_f1": avg_f1 * 100
    }

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test InfiniteBench")
    test_parser.add_argument("--model_name_test", type=str, default="meta-llama/Meta-Llama-3-8B")
    test_parser.add_argument("--dataset_split_test", type=str, default="longbook_qa_eng", 
                           choices=["longbook_qa_eng", "longbook_choice_eng"])
    test_parser.add_argument("--gen_batch_size_test", type=int, default=1)
    test_parser.add_argument("--num_few_shot_test", type=int, default=2)
    test_parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    test_parser.add_argument("--save_outputs", action="store_true", help="Save detailed outputs to JSON file")
    
    ib_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="InfiniteBenchFileTest")
    logger.info(f"--- Standalone InfiniteBench Test: {ib_args.model_name_test} ({ib_args.dataset_split_test}) ---")
    
    ib_pipe, _ = initialize_model_pipeline(ib_args.model_name_test, target_device_id=0)
    if ib_pipe:
        ib_eval_args = {
            "pipe": ib_pipe,
            "tokenizer": ib_pipe.tokenizer,
            "model_name_for_logging": ib_args.model_name_test,
            "device": ib_pipe.device,
            "dataset_split": ib_args.dataset_split_test,
            "generation_batch_size": ib_args.gen_batch_size_test,
            "num_few_shot": ib_args.num_few_shot_test,
            "max_new_tokens": ib_args.max_new_tokens,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": ib_args.save_outputs
        }
        try:
            print(json.dumps(evaluate_infinitebench(**ib_eval_args), indent=2))
        finally:
            cleanup_model_resources(ib_pipe, getattr(ib_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {ib_args.model_name_test} for InfiniteBench test.")