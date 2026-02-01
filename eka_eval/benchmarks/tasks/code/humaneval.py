# eka_eval/benchmarks/tasks/code/humaneval.py
"""
HumanEval benchmark evaluation - COMPLETE FIXED VERSION
Aligned with MBPP best practices for maximum accuracy
"""

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
    """
    Extracts the model's code completion from the full generated text.
    IMPROVED: More conservative stop sequences to avoid cutting valid code.
    """
    completion_part = ""
    if full_generated_text.startswith(prompt_sent_to_llm):
        completion_part = full_generated_text[len(prompt_sent_to_llm):]
    else:
        logger.warning("Prompt not found at start of generation, using full text")
        completion_part = full_generated_text

    # IMPROVED: More conservative stop sequences
    # Remove "\nassert " from stop sequences - valid code can have assertions!
    stop_sequences = [
        "\ndef ",        # Next function definition
        "\nclass ",     # Next class definition
        "\nif __name__", # Main block
        "\n# Test",     # Test section
        "\n# Example",  # Example section
        "</s>",         # End of sequence token
        "<|EOT|>",      # End of turn token
    ]
    
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

def fix_test_script(test_script: str, entry_point: str) -> str:
    """
    CRITICAL FIX: Add function call to test script if it doesn't have one.
    HumanEval test scripts define check() but don't call it!
    """
    # Check if the script already calls check()
    if f"check({entry_point})" in test_script:
        return test_script
    
    # Add the function call at the end
    fixed_script = test_script.rstrip()
    if not fixed_script.endswith('\n'):
        fixed_script += '\n'
    
    # Add the call to check() function with the entry_point
    fixed_script += f"\ncheck({entry_point})\n"
    
    return fixed_script

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
    process_id: int = 0,
    gpu_id: int = 0,
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluates the model on the HumanEval benchmark for code generation.
    IMPROVED: Aligned with MBPP best practices for better accuracy.
    """

    if humaneval_pass_at_k_metric is None:
        logger.error("HumanEval: code_eval metric not available")
        return {"HumanEval": 0.0, "error_message": "CodeEvalMetricLoadFailed"}

    logger.info(f"P{process_id}: Starting HumanEval evaluation for model: {model_name_for_logging}")

    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        logger.error("HumanEval: Tokenizer missing eos_token_id")
        return {"HumanEval": 0.0, "error_message": "TokenizerMissingEOS"}

    try:
        humaneval_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"P{process_id}: Loaded HumanEval dataset with {len(humaneval_dataset)} problems")
    except Exception as e:
        logger.critical(f"P{process_id}: Failed to load dataset '{dataset_name}': {e}")
        return {"HumanEval": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}

    if len(humaneval_dataset) == 0:
        return {"HumanEval": 0.0}

    few_shot_examples = get_fewshot_examples() if use_fewshot else []
    generation_inputs = []
    problem_references = {}

    for problem in tqdm(humaneval_dataset, desc=f"P{process_id} - Preparing HumanEval"):
        task_id = problem.get("task_id")
        problem_prompt = problem.get("prompt")
        test_script = problem.get("test")
        entry_point = problem.get("entry_point")

        if not all([task_id, problem_prompt, test_script, entry_point]):
            logger.warning(f"P{process_id}: Skipping problem {task_id} due to missing data")
            continue

        full_prompt = format_prompt(problem_prompt, few_shot_examples, use_fewshot)
        
        # CRITICAL FIX: Fix the test script to actually call check()
        fixed_test_script = fix_test_script(test_script, entry_point)
        
        for _ in range(num_samples_per_task):
            generation_inputs.append({
                "llm_prompt": full_prompt,
                "problem_prompt": problem_prompt,
                "task_id": task_id,
                "entry_point": entry_point,
                "test_script": fixed_test_script
            })
        problem_references[task_id] = fixed_test_script

    if not generation_inputs:
        logger.error(f"P{process_id}: No valid prompts prepared")
        return {"HumanEval": 0.0, "error_message": "NoValidPrompts"}

    predictions_by_task_id = defaultdict(list)
    detailed_results_log = []

    logger.info(f"P{process_id}: Generating code for {len(generation_inputs)} samples")

    # CRITICAL FIX: Use greedy decoding like MBPP for better pass@1 performance
    generation_params = {
        "do_sample": False,  # Greedy decoding (deterministic)
        "max_new_tokens": max_new_tokens_completion,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": True
    }

    for i in tqdm(range(0, len(generation_inputs), generation_batch_size), desc=f"P{process_id} - HumanEval Generation"):
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
                    full_llm_prompt=input_info['llm_prompt'][:500] + "..." if len(input_info['llm_prompt']) > 500 else input_info['llm_prompt'],
                    entry_point=input_info['entry_point'],
                    raw_generation=raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text,
                    extracted_completion=extracted,
                    full_code_for_eval=full_code,
                    reference_test_script=input_info['test_script']
                ))
                
                # Debug logging for first few samples
                if i < 3:
                    logger.info(
                        f"\n=== HumanEval {input_info['task_id']} ===\n"
                        f"Entry point: {input_info['entry_point']}\n"
                        f"Extracted completion (first 200 chars):\n{extracted[:200]}...\n"
                    )

        except Exception as e:
            logger.error(f"P{process_id}: Error during generation batch {i}: {e}", exc_info=True)
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
        logger.error(f"P{process_id}: No valid predictions or references for evaluation")
        return {"HumanEval": 0.0, "error_message": "NoSamplesForCodeEval"}

    logger.info(f"P{process_id}: Evaluating {len(final_references)} problems with code_eval")
    
    # Log samples for verification
    logger.info(f"\n=== Sample Test Script ===\n{final_references[0]}\n")
    logger.info(f"\n=== Sample Prediction ===\n{final_predictions[0][0][:500]}...\n")
    
    # ========== CRITICAL: CODE_EVAL SECTION - COMPLETELY FIXED ==========
    logger.info(f"P{process_id}: Calling code_eval with {len(final_predictions)} predictions and k={k_values}")
    
    final_scores = {}
    
    try:
        eval_output = humaneval_pass_at_k_metric.compute(
            references=final_references,
            predictions=final_predictions,
            k=k_values
        )
        
        # EXTENSIVE DEBUG LOGGING
        logger.info(f"P{process_id}: ===== CODE_EVAL RAW RESULT =====")
        logger.info(f"P{process_id}: Result type: {type(eval_output)}")
        logger.info(f"P{process_id}: Result value: {eval_output}")
        
        # Validate and parse result
        if eval_output is None:
            logger.error(f"P{process_id}: code_eval returned None!")
            return {"HumanEval": 0.0, "error_message": "code_eval_returned_none"}
        
        # Handle both tuple and dict return formats
        if isinstance(eval_output, tuple):
            if len(eval_output) < 1:
                logger.error(f"P{process_id}: code_eval tuple is empty!")
                return {"HumanEval": 0.0, "error_message": "empty_tuple"}
            
            scores = eval_output[0]
            detailed_results = eval_output[1] if len(eval_output) > 1 else None
            
            logger.info(f"P{process_id}: Tuple unpacked - scores: {scores}, detailed_results: {detailed_results is not None}")
        elif isinstance(eval_output, dict):
            scores = eval_output
            detailed_results = None
            logger.info(f"P{process_id}: Direct dict format: {scores}")
        else:
            scores = eval_output
            detailed_results = None
        
        # Validate scores is a dict
        if not isinstance(scores, dict):
            logger.error(f"P{process_id}: Scores is not a dict! Type: {type(scores)}, Value: {scores}")
            return {"HumanEval": 0.0, "error_message": f"invalid_scores_type_{type(scores).__name__}"}
        
        # Check if scores dict is empty
        if not scores:
            logger.error(f"P{process_id}: Scores dict is empty!")
            return {"HumanEval": 0.0, "error_message": "empty_scores_dict"}
        
        logger.info(f"P{process_id}: Available score keys: {list(scores.keys())}")
        
        # Extract pass@k scores
        for k_val in k_values:
            metric_key = f"pass@{k_val}"
            
            if metric_key not in scores:
                logger.error(f"P{process_id}: {metric_key} not found in scores!")
                logger.error(f"P{process_id}: Available keys: {list(scores.keys())}")
                continue
            
            raw_score = scores[metric_key]
            logger.info(f"P{process_id}: {metric_key} raw value: {raw_score} (type: {type(raw_score)})")
            
            # Validate score is a number
            if not isinstance(raw_score, (int, float)):
                logger.error(f"P{process_id}: {metric_key} is not a number! Type: {type(raw_score)}, Value: {raw_score}")
                continue
            
            # Convert to percentage (raw_score is 0.0 to 1.0)
            score_percentage = float(raw_score) * 100.0
            
            if k_val == k_values[0]:
                final_scores["HumanEval"] = score_percentage
            
            final_scores[f"HumanEval_pass@{k_val}"] = score_percentage
            
            logger.info(f"P{process_id}: {metric_key} = {score_percentage:.2f}%")
        
        # Verify we got at least one score
        if not final_scores:
            logger.error(f"P{process_id}: No valid scores extracted from code_eval!")
            return {"HumanEval": 0.0, "error_message": "no_valid_scores_extracted"}
        
        # Process detailed results
        if detailed_results:
            # Convert dict/defaultdict to list for uniform processing
            if isinstance(detailed_results, dict):
                detailed_results_list = sorted(detailed_results.items())
                logger.info(f"P{process_id}: Processing {len(detailed_results_list)} detailed test results (from dict)")
            elif isinstance(detailed_results, list):
                detailed_results_list = list(enumerate(detailed_results))
                logger.info(f"P{process_id}: Processing {len(detailed_results_list)} detailed test results (from list)")
            else:
                logger.warning(f"P{process_id}: Unexpected detailed_results type: {type(detailed_results)}")
                detailed_results_list = []
            
            if detailed_results_list:
                # Create a mapping of task_id to list indices
                task_log_indices = defaultdict(list)
                for idx, log_entry in enumerate(detailed_results_log):
                    task_log_indices[log_entry.task_id].append(idx)
                
                # Update each result entry with pass/fail status
                actual_passes = 0
                actual_fails = 0
                
                for task_idx, task_results in detailed_results_list:
                    if task_idx < len(sorted_task_ids):
                        task_id = sorted_task_ids[task_idx]
                        log_entry_indices = task_log_indices.get(task_id, [])
                        
                        if task_results and isinstance(task_results, list):
                            for sample_idx, result in enumerate(task_results):
                                if isinstance(result, tuple) and len(result) == 2:
                                    completion_id = result[0]
                                    result_dict = result[1]
                                    
                                    if isinstance(result_dict, dict):
                                        passed = result_dict.get('passed', False)
                                        error_msg = result_dict.get('result', '')
                                        
                                        if passed:
                                            actual_passes += 1
                                        else:
                                            actual_fails += 1
                                        
                                        # Update log entry
                                        if sample_idx < len(log_entry_indices):
                                            log_idx = log_entry_indices[sample_idx]
                                            detailed_results_log[log_idx].passed = passed
                                            detailed_results_log[log_idx].error_message = error_msg if not passed else ""
                                            
                                            # Log first few for debugging
                                            if task_idx < 5:
                                                logger.info(
                                                    f"P{process_id}: Task {task_id} sample {sample_idx}: "
                                                    f"passed={passed}, "
                                                    f"result={error_msg[:100] if error_msg else 'OK'}"
                                                )
                
                total_tests = actual_passes + actual_fails
                if total_tests > 0:
                    actual_pass_rate = (actual_passes / total_tests) * 100
                    logger.info(
                        f"P{process_id}: Actual test results: "
                        f"{actual_passes} passed, {actual_fails} failed "
                        f"({actual_pass_rate:.2f}%)"
                    )
                    
                    # Sanity check
                    computed_score = final_scores.get("HumanEval", 0.0)
                    if abs(computed_score - actual_pass_rate) > 0.1:
                        logger.warning(
                            f"P{process_id}: Score mismatch! "
                            f"Computed: {computed_score:.2f}%, "
                            f"Actual: {actual_pass_rate:.2f}%"
                        )
        else:
            logger.warning(f"P{process_id}: No detailed results returned from code_eval")

    except KeyError as e:
        logger.error(f"P{process_id}: KeyError in code evaluation: {e}", exc_info=True)
        final_scores = {"HumanEval": 0.0, "error_message": f"KeyError: {str(e)}"}

    except ValueError as e:
        logger.error(f"P{process_id}: ValueError in code evaluation: {e}", exc_info=True)
        final_scores = {"HumanEval": 0.0, "error_message": f"ValueError: {str(e)}"}

    except Exception as e:
        logger.error(f"P{process_id}: Unexpected error in code evaluation: {e}", exc_info=True)
        final_scores = {"HumanEval": 0.0, "error_message": f"Exception: {type(e).__name__}: {str(e)}"}
    
    # ========== END CODE_EVAL SECTION ==========
    
    # CRITICAL FIX: Final validation
    if "HumanEval" not in final_scores:
        logger.error(f"P{process_id}: HumanEval score missing from final_scores! Setting to 0.0")
        final_scores["HumanEval"] = 0.0  # Default to 0.0, NOT 100.0!

    # Save detailed results
    if save_detailed and detailed_results_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name_for_logging.replace("/", "_").replace("-", "_")
        filename = f"humaneval_{safe_model_name}_{dataset_split.replace(':', '_')}_{timestamp}.jsonl"
        
        results_dir = kwargs.get("results_dir", "results_output")
        detailed_dir = os.path.join(results_dir, "detailed_results")
        os.makedirs(detailed_dir, exist_ok=True)
        filepath = os.path.join(detailed_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for result in detailed_results_log:
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            logger.info(f"P{process_id}: Saved {len(detailed_results_log)} detailed results to {filepath}")
        except Exception as e:
            logger.error(f"P{process_id}: Failed to save detailed results: {e}")

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final HumanEval Pass@1: {final_scores.get('HumanEval', 0.0):.2f}%")
    
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