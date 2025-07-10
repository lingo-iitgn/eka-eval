# eka_eval/benchmarks/tasks/code/humaneval.py

import torch
import re
import sys
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import evaluate as hf_evaluate
import logging

logger = logging.getLogger(__name__)

@dataclass
class HumanEvalResultDetail:
    task_id: str
    problem_prompt: str
    full_llm_prompt: str
    entry_point: str
    raw_generation: str
    extracted_completion: str
    full_code_for_eval: str
    reference_test_script: str
    passed: Optional[bool] = None
    error_message: str = ""

# Global Setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
try:
    humaneval_pass_at_k_metric = hf_evaluate.load("code_eval")
    logger.info("code_eval metric for HumanEval loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'code_eval' metric for HumanEval: {e}")
    humaneval_pass_at_k_metric = None

def get_fewshot_examples() -> List[Dict[str, str]]:
    """Returns canonical HumanEval examples for few-shot prompting."""
    return [
        {
            "task_id": "HumanEval/0",
            "prompt_example_for_llm": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        }
    ]

def format_prompt(problem_prompt: str, few_shot_examples: List[Dict[str, str]], use_fewshot: bool) -> str:
    """Formats the prompt for HumanEval, optionally with few-shot examples."""
    if use_fewshot and few_shot_examples:
        few_shot_text = "Complete the following Python functions based on their docstrings. Here are some examples:\n\n"
        for ex in few_shot_examples:
            few_shot_text += ex["prompt_example_for_llm"].strip() + "\n\n"
        few_shot_text += "Now, complete the following function:\n"
        return few_shot_text + problem_prompt
    else:
        return "Complete the following Python function based on its docstring:\n" + problem_prompt

def extract_completion(full_generated_text: str, prompt_sent_to_llm: str) -> str:
    """Extracts the model's code completion from the full generated text."""
    completion_part = ""
    if full_generated_text.startswith(prompt_sent_to_llm):
        completion_part = full_generated_text[len(prompt_sent_to_llm):]
    else:
        logger.warning("Prompt not found at start of generation, using full text")
        completion_part = full_generated_text

    # Stop at common sequences
    stop_sequences = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\nassert ", "</s>", "<|EOT|>"]
    min_stop_index = len(completion_part)
    for seq in stop_sequences:
        found_idx = completion_part.find(seq)
        if found_idx != -1:
            min_stop_index = min(min_stop_index, found_idx)
    
    cleaned_completion = completion_part[:min_stop_index].rstrip()

    # Remove markdown code blocks
    if cleaned_completion.startswith("```python"):
        cleaned_completion = cleaned_completion[len("```python"):].lstrip()
    if cleaned_completion.startswith("```"):
        cleaned_completion = cleaned_completion[len("```"):].lstrip()
    if cleaned_completion.endswith("```"):
        cleaned_completion = cleaned_completion[:-len("```")].rstrip()

    return cleaned_completion

def evaluate_humaneval(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = "openai_humaneval",
    dataset_split: str = "test",
    num_samples_per_task: int = 1,
    k_values: List[int] = [1],
    use_fewshot: bool = False,
    max_new_tokens_completion: int = 384,
    generation_batch_size: int = 1,
    **kwargs
) -> Dict[str, float]:
    """Evaluates the model on the HumanEval benchmark for code generation."""

    if humaneval_pass_at_k_metric is None:
        logger.error("HumanEval: code_eval metric not available")
        return {"HumanEval": 0.0, "error_message": "CodeEvalMetricLoadFailed"}

    logger.info(f"Starting HumanEval evaluation for model: {model_name_for_logging}")

    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        logger.error("HumanEval: Tokenizer missing eos_token_id")
        return {"HumanEval": 0.0, "error_message": "TokenizerMissingEOS"}

    try:
        humaneval_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"Loaded HumanEval dataset with {len(humaneval_dataset)} problems")
    except Exception as e:
        logger.critical(f"Failed to load dataset '{dataset_name}': {e}")
        return {"HumanEval": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}

    if len(humaneval_dataset) == 0:
        return {"HumanEval": 0.0}

    few_shot_examples = get_fewshot_examples() if use_fewshot else []
    generation_inputs = []
    problem_references = {}

    for problem in tqdm(humaneval_dataset, desc="Preparing HumanEval Prompts"):
        task_id = problem.get("task_id")
        problem_prompt = problem.get("prompt")
        test_script = problem.get("test")
        entry_point = problem.get("entry_point")

        if not all([task_id, problem_prompt, test_script, entry_point]):
            logger.warning(f"Skipping problem {task_id} due to missing data")
            continue

        full_prompt = format_prompt(problem_prompt, few_shot_examples, use_fewshot)
        for _ in range(num_samples_per_task):
            generation_inputs.append({
                "llm_prompt": full_prompt,
                "problem_prompt": problem_prompt,
                "task_id": task_id,
                "entry_point": entry_point,
                "test_script": test_script
            })
        problem_references[task_id] = test_script

    if not generation_inputs:
        logger.error("No valid prompts prepared")
        return {"HumanEval": 0.0, "error_message": "NoValidPrompts"}

    predictions_by_task_id = defaultdict(list)
    detailed_results_log = []

    logger.info(f"Starting code generation for {len(generation_inputs)} samples")

    generation_params = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": max_new_tokens_completion,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": True
    }

    for i in tqdm(range(0, len(generation_inputs), generation_batch_size), desc="Generating Completions"):
        batch_inputs = generation_inputs[i:i + generation_batch_size]
        batch_prompts = [info['llm_prompt'] for info in batch_inputs]

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            raw_outputs = pipe(batch_prompts, **generation_params)

            for batch_idx, input_info in enumerate(batch_inputs):
                output = raw_outputs[batch_idx]
                raw_text = "#ERROR: No output generated"
                
                if output and isinstance(output, list) and output[0]:
                    raw_text = output[0].get('generated_text', raw_text)
                elif isinstance(output, dict):
                    raw_text = output.get('generated_text', raw_text)

                extracted = extract_completion(raw_text, input_info['llm_prompt'])
                full_code = input_info['problem_prompt'] + extracted
                predictions_by_task_id[input_info['task_id']].append(full_code)

                detailed_results_log.append(HumanEvalResultDetail(
                    task_id=input_info['task_id'],
                    problem_prompt=input_info['problem_prompt'],
                    full_llm_prompt=input_info['llm_prompt'],
                    entry_point=input_info['entry_point'],
                    raw_generation=raw_text,
                    extracted_completion=extracted,
                    full_code_for_eval=full_code,
                    reference_test_script=input_info['test_script']
                ))

        except Exception as e:
            logger.error(f"Error during generation batch {i}: {e}")
            for input_info in batch_inputs:
                error_completion = f"# GENERATION ERROR: {e}"
                predictions_by_task_id[input_info['task_id']].append(input_info['problem_prompt'] + error_completion)
                detailed_results_log.append(HumanEvalResultDetail(
                    task_id=input_info['task_id'],
                    problem_prompt=input_info['problem_prompt'],
                    full_llm_prompt=input_info['llm_prompt'],
                    entry_point=input_info['entry_point'],
                    raw_generation=error_completion,
                    extracted_completion=error_completion,
                    full_code_for_eval=input_info['problem_prompt'] + error_completion,
                    reference_test_script=input_info['test_script'],
                    passed=False,
                    error_message=str(e)
                ))

    # Prepare for evaluation
    final_predictions = []
    final_references = []
    sorted_task_ids = sorted(problem_references.keys(), key=lambda tid: int(tid.split('/')[-1]))

    for task_id in sorted_task_ids:
        if task_id in predictions_by_task_id and problem_references[task_id]:
            final_predictions.append(predictions_by_task_id[task_id])
            final_references.append(problem_references[task_id])

    if not final_predictions or not final_references:
        logger.error("No valid predictions or references for evaluation")
        return {"HumanEval": 0.0, "error_message": "NoSamplesForCodeEval"}

    logger.info(f"Running code evaluation for {len(final_references)} problems")
    final_scores = {}
    
    try:
        eval_output = humaneval_pass_at_k_metric.compute(
            references=final_references,
            predictions=final_predictions,
            k=k_values
        )
        
        if isinstance(eval_output, tuple):
            scores = eval_output[0]
            detailed_results = eval_output[1] if len(eval_output) > 1 else None
        else:
            scores = eval_output
            detailed_results = None
        
        if scores:
            logger.info(f"HumanEval Pass@k scores: {scores}")
            for k_val in k_values:
                metric_key = f"pass@{k_val}"
                score_value = scores.get(metric_key, 0.0) * 100
                
                if k_val == k_values[0]:
                    final_scores["HumanEval"] = score_value
                final_scores[f"HumanEval_pass@{k_val}"] = score_value
        else:
            logger.error("code_eval did not return valid scores")
            final_scores["HumanEval"] = 0.0

        # Update detailed results with pass/fail status
        if detailed_results and isinstance(detailed_results, list):
            for task_idx, task_results in enumerate(detailed_results):
                if task_idx < len(sorted_task_ids):
                    task_id = sorted_task_ids[task_idx]
                    log_entry = next((entry for entry in detailed_results_log if entry.task_id == task_id), None)
                    
                    if log_entry and task_results and isinstance(task_results, list):
                        first_result = task_results[0]
                        if isinstance(first_result, tuple) and len(first_result) == 2:
                            result_dict = first_result[1]
                            if isinstance(result_dict, dict):
                                log_entry.passed = result_dict.get('passed', False)
                                log_entry.error_message = result_dict.get('result', '') if not log_entry.passed else ""

    except Exception as e:
        logger.error(f"Error during code evaluation: {e}")
        final_scores["HumanEval"] = 0.0
        final_scores["error_message"] = "CodeEvalComputationError"

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name_for_logging.replace("/", "_").replace("-", "_")
    filename = f"humaneval_results_{safe_model_name}_{dataset_split}_{timestamp}.jsonl"
    
    results_dir = os.path.join(kwargs.get("results_dir", "results_output"), "humaneval_detailed")
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in detailed_results_log:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(detailed_results_log)} detailed results to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")

    if "HumanEval" not in final_scores:
        final_scores["HumanEval"] = 0.0

    logger.info(f"HumanEval evaluation finished. Final scores: {final_scores}")
    return final_scores


# Test function for standalone execution
if __name__ == '__main__':
    # Set GPU constraint for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Add project root to path for imports
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    # Import required modules for testing
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    # Parse command line arguments for testing
    test_parser = argparse.ArgumentParser(description="Standalone Test HumanEval")
    test_parser.add_argument("--model_name_test", type=str, default="google/gemma-2-2b", help="Model to test")
    test_parser.add_argument("--dataset_split_test", type=str, default="test[:10]", help="Dataset split to test")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=1, help="Generation batch size")
    test_parser.add_argument("--num_samples_test", type=int, default=1, help="Number of samples per task")
    test_parser.add_argument("--k_values_test", type=int, nargs='+', default=[1], help="k values for Pass@k")
    test_parser.add_argument("--use_fewshot", action="store_true", help="Use few-shot prompting")
    test_parser.add_argument("--max_new_tokens", type=int, default=384, help="Maximum new tokens to generate")
    
    humaneval_args = test_parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG, worker_id="HumanEvalTest")
    logger.info(f"--- Standalone HumanEval Test: {humaneval_args.model_name_test} ---")
    
    # Initialize model pipeline
    humaneval_pipe, _ = initialize_model_pipeline(humaneval_args.model_name_test, target_device_id=0)
    
    if humaneval_pipe:
        # Prepare evaluation arguments
        humaneval_eval_args = {
            "pipe": humaneval_pipe,
            "tokenizer": humaneval_pipe.tokenizer,
            "model_name_for_logging": humaneval_args.model_name_test,
            "device": humaneval_pipe.device,
            "dataset_split": humaneval_args.dataset_split_test,
            "num_samples_per_task": humaneval_args.num_samples_test,
            "k_values": humaneval_args.k_values_test,
            "use_fewshot": humaneval_args.use_fewshot,
            "max_new_tokens_completion": humaneval_args.max_new_tokens,
            "generation_batch_size": humaneval_args.gen_batch_size_test,
            "results_dir": "test_results"
        }
        
        try:
            # Run evaluation and print results
            results = evaluate_humaneval(**humaneval_eval_args)
            print("\n" + "="*50)
            print("HUMANEVAL TEST RESULTS:")
            print("="*50)
            print(json.dumps(results, indent=2))
        except Exception as e:
            logger.error(f"Error during HumanEval evaluation: {e}")
            print(f"Evaluation failed: {e}")
        finally:
            # Clean up resources
            cleanup_model_resources(humaneval_pipe, getattr(humaneval_pipe, 'model', None))
    else:
        logger.error(f"Failed to initialize model {humaneval_args.model_name_test} for HumanEval test.")
        print("Model initialization failed!")