# eka_eval/benchmarks/tasks/code/mbpp.py
"""
MBPP (Mostly Basic Python Problems) benchmark evaluation - COMPLETE FIXED VERSION

All bugs fixed:
1. Test scripts properly execute check() function
2. No placeholder comments that break code_eval
3. Proper error handling and validation
4. Correct score calculation (defaults to 0.0, not 100.0)
5. Extensive logging for debugging
"""

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_MBPP = "google-research-datasets/mbpp"
DEFAULT_DATASET_CONFIG_MBPP = "full"
DEFAULT_SPLIT_MBPP = "test"
DEFAULT_MAX_NEW_TOKENS_MBPP = 1024
DEFAULT_GENERATION_BATCH_SIZE_MBPP = 4
DEFAULT_NUM_FEWSHOT_MBPP = 3
DEFAULT_K_VALUES = [1]

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Few-shot examples from lm-evaluation-harness
FEWSHOT_EXAMPLES = [
    {
        "task_id": 2,
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
    },
    {
        "task_id": 3,
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
    },
    {
        "task_id": 4,
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
    },
]


def _format_mbpp_prompt(
    text: str,
    test_list: List[str],
    num_fewshot: int = 3
) -> str:
    """Format MBPP prompt following lm-harness specification"""
    prompt = ""
    
    # Add few-shot examples
    examples_to_use = FEWSHOT_EXAMPLES[:num_fewshot] if num_fewshot > 0 else []
    
    for example in examples_to_use:
        ex_text = example['text']
        ex_tests = example['test_list']
        ex_code = example['code']
        
        prompt += f"You are an expert Python programmer, and here is your task: {ex_text} Your code should pass these tests:\n\n"
        for i in range(min(3, len(ex_tests))):
            prompt += f"{ex_tests[i]}\n"
        prompt += "[BEGIN]\n"
        prompt += ex_code
        prompt += "\n[DONE]\n\n"
    
    # Add current problem
    prompt += f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n"
    for i in range(min(3, len(test_list))):
        prompt += f"{test_list[i]}\n"
    prompt += "[BEGIN]\n"
    
    return prompt


def _extract_function_name(code: str) -> Optional[str]:
    """
    Extract the main function name from generated code.
    
    Example:
        "def similar_elements(a, b):\n    ..." -> "similar_elements"
    """
    # Look for function definition
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if match:
        return match.group(1)
    return None


def _extract_code_blocks(text: str) -> str:
    """Extract code from generated text"""
    # First, try to extract between [BEGIN] and [DONE]
    begin_done_pattern = r'\[BEGIN\](.*?)(?:\[DONE\]|$)'
    match = re.search(begin_done_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        if code and 'def ' in code:
            return code
    
    # Pattern to match ```...``` blocks
    pattern = r"```(?:python)?\s*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
        if code and 'def ' in code:
            return code
    
    # Fallback: try to extract first function definition
    lines = text.split('\n')
    code_lines = []
    in_function = False
    function_indent = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Skip common noise
        if any(skip in stripped.lower() for skip in [
            "you are", "here is", "this function", "note that",
            "this code", "the function", "this will", "here's",
            "explanation", "example usage"
        ]):
            continue
        
        # Start of function
        if stripped.startswith("def ") and "(" in stripped and ":" in stripped:
            in_function = True
            function_indent = len(line) - len(line.lstrip())
            code_lines.append(line)
            continue
        
        if in_function:
            current_indent = len(line) - len(line.lstrip())
            
            # Empty line or properly indented line
            if not stripped or current_indent > function_indent:
                code_lines.append(line)
            # Next function at same or lower indent
            elif current_indent <= function_indent and stripped.startswith("def "):
                code_lines.append(line)
                function_indent = current_indent
            # End of function
            elif current_indent <= function_indent and stripped:
                break
        # Imports or class definitions before function
        elif stripped.startswith(("import ", "from ", "class ")):
            code_lines.append(line)
    
    if code_lines:
        code = '\n'.join(code_lines).strip()
        if 'def ' in code:
            return code
    
    return ""


def _clean_code(code: str) -> str:
    """Clean extracted code"""
    if not code.strip():
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip test assertions and examples
        if any(bad in line for bad in ['>>>', 'assert ', '# Example:', '# Test:', '[DONE]', '[BEGIN]']):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def _build_test_script(test_setup: str, test_list: List[str], function_name: str) -> str:
    """
    Build executable test script for MBPP.
    
    CRITICAL: MBPP tests contain the actual function name in assertions.
    We need to execute these tests directly after defining the function.
    
    Args:
        test_setup: Setup code (imports, etc.)
        test_list: List of assertion strings like "assert func(...) == ..."
        function_name: Name of the function being tested
    
    Returns:
        Executable test script
    """
    script = ""
    
    # Add setup code
    if test_setup:
        script += test_setup + "\n\n"
    
    # CRITICAL FIX: Do NOT add placeholder comments!
    # The function will be defined by code_eval when it concatenates:
    # exec(prediction + "\n" + reference)
    
    # Add test execution wrapped in a check function
    script += "def check():\n"
    for test_line in test_list:
        # Each test is like: "assert similar_elements((3,4),(5,4)) == (4,)"
        # We need to indent it and add to check function
        script += f"    {test_line}\n"
    
    # CRITICAL: Call the check function!
    script += "\ncheck()\n"
    
    return script


def save_detailed_mbpp_results(
    results_data: List[Dict],
    model_name: str,
    num_few_shot: int,
    pass_at_k: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed MBPP results to JSON"""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mbpp_{model_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "num_few_shot": num_few_shot,
        "total_problems": len(results_data),
        "passed_problems": sum(1 for r in results_data if r.get("passed", False)),
        "pass_at_1": pass_at_k,
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
        logger.info(f"Detailed MBPP results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed MBPP results: {e}")
        return ""


def evaluate_mbpp(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_MBPP,
    dataset_config: str = DEFAULT_DATASET_CONFIG_MBPP,
    dataset_split: str = DEFAULT_SPLIT_MBPP,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_MBPP,
    num_samples_per_task: int = 1,
    k_values: List[int] = DEFAULT_K_VALUES,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MBPP,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_MBPP,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:
    """Evaluate model on MBPP - COMPLETE FIXED VERSION"""
    
    logger.info(f"Starting MBPP ({num_few_shot}-shot): {model_name_for_logging}")
    
    # Load code_eval metric
    try:
        code_eval_metric = hf_evaluate.load("code_eval")
        logger.info("Successfully loaded code_eval metric")
    except Exception as e:
        logger.error(f"Failed to load code_eval metric: {e}")
        return {"MBPP": 0.0, "error_message": "CodeEvalMetricLoadFailed"}
    
    # Load dataset
    try:
        full_data = load_dataset(dataset_name, dataset_config, split=dataset_split)
        logger.info(f"Loaded MBPP dataset: {len(full_data)} problems")
    except Exception as e:
        logger.error(f"Failed to load MBPP dataset: {e}")
        try:
            full_data = load_dataset("mbpp", "sanitized", split=dataset_split)
            logger.info("Loaded MBPP sanitized version as fallback")
        except Exception as e2:
            logger.error(f"Failed to load MBPP fallback: {e2}")
            return {"MBPP": 0.0, "error_message": f"DatasetLoadFailed: {e2}"}
    
    # Multi-GPU data sharding
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
        return {"MBPP": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} MBPP problems")
    
    # Prepare prompts and references
    generation_inputs = []
    problem_references = {}
    detailed_results = []
    
    for problem_idx, problem in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - Preparing MBPP")):
        task_id = str(problem.get('task_id', f"task_{problem_idx}"))
        
        # Support both 'text' (full config) and 'prompt' (sanitized config)
        text = problem.get('text', problem.get('prompt', ''))
        
        code = problem.get('code', '')
        
        # Support both 'test_setup_code' (full) and 'test_imports' (sanitized)
        test_setup = problem.get('test_setup_code', '')
        if not test_setup and 'test_imports' in problem:
            # Convert test_imports list to import statements
            test_imports = problem.get('test_imports', [])
            if test_imports:
                test_setup = '\n'.join(test_imports)
        
        test_list = problem.get('test_list', [])
        
        if not text or not test_list:
            logger.warning(f"P{process_id}: Skipping problem {task_id} due to missing data")
            continue
        
        # Extract function name from reference code
        function_name = _extract_function_name(code)
        if not function_name:
            logger.warning(f"P{process_id}: Could not extract function name from {task_id}")
            continue
        
        # Format prompt
        prompt = _format_mbpp_prompt(text, test_list, num_few_shot)
        
        # Generate multiple samples for pass@k
        for sample_idx in range(num_samples_per_task):
            generation_inputs.append({
                'task_id': task_id,
                'prompt': prompt,
                'text': text,
                'code': code,
                'test_setup': test_setup,
                'test_list': test_list,
                'function_name': function_name,
                'sample_idx': sample_idx
            })
        
        # Build test script - FIXED VERSION (no placeholder comments!)
        test_script = _build_test_script(test_setup, test_list, function_name)
        problem_references[task_id] = test_script
    
    if not generation_inputs:
        logger.error(f"P{process_id}: No valid prompts prepared")
        return {"MBPP": 0.0, "error_message": "NoValidPrompts"}
    
    logger.info(f"P{process_id}: Generating code for {len(generation_inputs)} samples")
    
    predictions_by_task = defaultdict(list)
    prompts_for_batch = []
    batch_info = []
    
    for input_idx, input_info in enumerate(tqdm(generation_inputs, desc=f"P{process_id} - MBPP Generation")):
        prompts_for_batch.append(input_info['prompt'])
        batch_info.append(input_info)
        
        # Process batch
        if len(prompts_for_batch) == generation_batch_size or input_idx == len(generation_inputs) - 1:
            
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            
            try:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    batch_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, output in enumerate(batch_outputs):
                    info = batch_info[k]
                    
                    # Extract generated text
                    if output and len(output) > 0 and 'generated_text' in output[0]:
                        full_generated = output[0]['generated_text']
                    else:
                        full_generated = ""
                    
                    # Remove prompt to get completion
                    completion = full_generated
                    if full_generated.startswith(info['prompt']):
                        completion = full_generated[len(info['prompt']):].strip()
                    
                    # Stop at [DONE]
                    if '[DONE]' in completion:
                        completion = completion.split('[DONE]')[0].strip()
                    
                    # Extract and clean code
                    extracted_code = _extract_code_blocks(completion)
                    extracted_code = _clean_code(extracted_code)
                    
                    # If no code extracted, use placeholder
                    if not extracted_code or 'def ' not in extracted_code:
                        extracted_code = f"def {info['function_name']}():\n    pass"
                    
                    # CRITICAL: Build full code for evaluation
                    # The code_eval metric will:
                    # 1. Execute the generated function
                    # 2. Then run the test script (which calls check())
                    full_code_for_eval = ""
                    if info['test_setup']:
                        full_code_for_eval += info['test_setup'] + "\n\n"
                    full_code_for_eval += extracted_code
                    
                    predictions_by_task[info['task_id']].append(full_code_for_eval)
                    
                    # Save detailed results
                    if info['sample_idx'] == 0 and save_detailed:
                        detailed_result = {
                            "task_id": info['task_id'],
                            "problem_text": info['text'],
                            "function_name": info['function_name'],
                            "prompt": info['prompt'][:500] + "..." if len(info['prompt']) > 500 else info['prompt'],
                            "completion": completion[:500] + "..." if len(completion) > 500 else completion,
                            "extracted_code": extracted_code,
                            "full_code_for_eval": full_code_for_eval,
                            "reference_code": info['code'],
                            "test_script": problem_references[info['task_id']]
                        }
                        detailed_results.append(detailed_result)
                    
                    # Debug logging
                    if input_idx < 3 and info['sample_idx'] == 0:
                        logger.info(
                            f"\n=== MBPP Task {info['task_id']} ===\n"
                            f"Problem: {info['text'][:100]}...\n"
                            f"Function: {info['function_name']}\n"
                            f"Extracted code:\n{extracted_code[:300]}...\n"
                        )
            
            except Exception as e:
                logger.error(f"P{process_id}: Error in MBPP generation: {e}", exc_info=True)
                # Placeholder for failures
                for info in batch_info:
                    fallback_code = f"def {info['function_name']}():\n    pass"
                    if info['test_setup']:
                        fallback_code = info['test_setup'] + "\n\n" + fallback_code
                    predictions_by_task[info['task_id']].append(fallback_code)
                    
                    if info['sample_idx'] == 0 and save_detailed:
                        detailed_result = {
                            "task_id": info['task_id'],
                            "problem_text": info['text'],
                            "error": str(e),
                            "full_code_for_eval": fallback_code
                        }
                        detailed_results.append(detailed_result)
            
            prompts_for_batch = []
            batch_info = []
    
    # Prepare for code_eval
    final_predictions = []
    final_references = []
    
    sorted_task_ids = sorted(problem_references.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    
    for task_id in sorted_task_ids:
        if task_id in predictions_by_task and problem_references[task_id]:
            final_predictions.append(predictions_by_task[task_id])
            final_references.append(problem_references[task_id])
    
    if not final_predictions or not final_references:
        logger.error(f"P{process_id}: No valid predictions for evaluation")
        return {"MBPP": 0.0, "error_message": "NoValidPredictions"}
    
    logger.info(f"P{process_id}: Evaluating {len(final_predictions)} problems")
    
    # Log samples for verification
    logger.info(f"\n=== Sample Test Script ===\n{final_references[0]}\n")
    logger.info(f"\n=== Sample Prediction ===\n{final_predictions[0][0][:500]}...\n")
    
    # ========== CRITICAL: CODE_EVAL SECTION - COMPLETELY FIXED ==========
    logger.info(f"P{process_id}: Calling code_eval with {len(final_predictions)} predictions and k={k_values}")
    
    try:
        # Call code_eval metric
        eval_result = code_eval_metric.compute(
            references=final_references,
            predictions=final_predictions,
            k=k_values
        )
        
        # EXTENSIVE DEBUG LOGGING
        logger.info(f"P{process_id}: ===== CODE_EVAL RAW RESULT =====")
        logger.info(f"P{process_id}: Result type: {type(eval_result)}")
        logger.info(f"P{process_id}: Result value: {eval_result}")
        
        # Validate and parse result
        if eval_result is None:
            logger.error(f"P{process_id}: code_eval returned None!")
            return {"MBPP": 0.0, "error_message": "code_eval_returned_none"}
        
        # Handle tuple format (scores, test_results)
        if isinstance(eval_result, tuple):
            if len(eval_result) < 1:
                logger.error(f"P{process_id}: code_eval tuple is empty!")
                return {"MBPP": 0.0, "error_message": "empty_tuple"}
            
            scores = eval_result[0]
            test_results = eval_result[1] if len(eval_result) > 1 else None
            
            logger.info(f"P{process_id}: Tuple unpacked - scores: {scores}, test_results: {test_results is not None}")
        else:
            # Direct dict format
            scores = eval_result
            test_results = None
            logger.info(f"P{process_id}: Direct dict format: {scores}")
        
        # Validate scores is a dict
        if not isinstance(scores, dict):
            logger.error(f"P{process_id}: Scores is not a dict! Type: {type(scores)}, Value: {scores}")
            return {"MBPP": 0.0, "error_message": f"invalid_scores_type_{type(scores).__name__}"}
        
        # Check if scores dict is empty
        if not scores:
            logger.error(f"P{process_id}: Scores dict is empty!")
            return {"MBPP": 0.0, "error_message": "empty_scores_dict"}
        
        logger.info(f"P{process_id}: Available score keys: {list(scores.keys())}")
        
        # Extract pass@k scores
        final_scores = {}
        
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
            
            if k_val == k_values[0]:  # First k value is the primary metric
                final_scores["MBPP"] = score_percentage
            
            final_scores[f"MBPP_pass@{k_val}"] = score_percentage
            
            logger.info(f"P{process_id}: {metric_key} = {score_percentage:.2f}%")
        
        # Verify we got at least one score
        if not final_scores:
            logger.error(f"P{process_id}: No valid scores extracted from code_eval!")
            return {"MBPP": 0.0, "error_message": "no_valid_scores_extracted"}
        
        # Count actual passes from test_results if available
        if test_results and isinstance(test_results, (list, tuple)):
            logger.info(f"P{process_id}: Processing {len(test_results)} test results")
            
            actual_passes = 0
            actual_fails = 0
            
            for task_idx, task_test_results in enumerate(test_results):
                if not task_test_results:
                    continue
                
                if isinstance(task_test_results, list):
                    for sample_result in task_test_results:
                        if isinstance(sample_result, tuple) and len(sample_result) >= 2:
                            result_dict = sample_result[1]
                            if isinstance(result_dict, dict):
                                passed = result_dict.get('passed', False)
                                if passed:
                                    actual_passes += 1
                                else:
                                    actual_fails += 1
                                
                                # Log first few for debugging
                                if task_idx < 5:
                                    logger.info(
                                        f"P{process_id}: Task {task_idx}: "
                                        f"passed={passed}, "
                                        f"result={result_dict.get('result', '')[:100]}"
                                    )
                                
                                # Update detailed results if available
                                if save_detailed and task_idx < len(detailed_results):
                                    detailed_results[task_idx]["passed"] = passed
                                    detailed_results[task_idx]["result"] = result_dict.get('result', '')
                                    if not passed:
                                        detailed_results[task_idx]["error_message"] = result_dict.get('result', '')
            
            total_tests = actual_passes + actual_fails
            if total_tests > 0:
                actual_pass_rate = (actual_passes / total_tests) * 100
                logger.info(
                    f"P{process_id}: Actual test results: "
                    f"{actual_passes} passed, {actual_fails} failed "
                    f"({actual_pass_rate:.2f}%)"
                )
                
                # Sanity check: compare with computed score
                computed_score = final_scores.get("MBPP", 0.0)
                if abs(computed_score - actual_pass_rate) > 0.1:
                    logger.warning(
                        f"P{process_id}: Score mismatch! "
                        f"Computed: {computed_score:.2f}%, "
                        f"Actual: {actual_pass_rate:.2f}%"
                    )
        else:
            logger.warning(f"P{process_id}: No test_results available for verification")
        
    except KeyError as e:
        logger.error(f"P{process_id}: KeyError in code evaluation: {e}", exc_info=True)
        final_scores = {"MBPP": 0.0, "error_message": f"KeyError: {str(e)}"}

    except ValueError as e:
        logger.error(f"P{process_id}: ValueError in code evaluation: {e}", exc_info=True)
        final_scores = {"MBPP": 0.0, "error_message": f"ValueError: {str(e)}"}

    except Exception as e:
        logger.error(f"P{process_id}: Unexpected error in code evaluation: {e}", exc_info=True)
        final_scores = {"MBPP": 0.0, "error_message": f"Exception: {type(e).__name__}: {str(e)}"}
    
    # ========== END CODE_EVAL SECTION ==========
    
    # CRITICAL FIX: Final validation - ensure MBPP score exists and defaults to 0.0
    if "MBPP" not in final_scores:
        logger.error(f"P{process_id}: MBPP score missing from final_scores! Setting to 0.0")
        final_scores["MBPP"] = 0.0  # ← CRITICAL: Default to 0.0, NOT 100.0!
    
    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_mbpp_results(
            detailed_results,
            model_name_for_logging,
            num_few_shot,
            final_scores.get("MBPP", 0.0),
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"P{process_id}: Detailed results saved to: {saved_path}")
    
    # Log final result
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final MBPP Pass@1: {final_scores.get('MBPP', 0.0):.2f}%")
    
    return final_scores