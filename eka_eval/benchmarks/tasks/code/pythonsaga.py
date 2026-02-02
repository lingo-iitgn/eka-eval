# eka_eval/benchmarks/tasks/code/pythonsaga.py
"""
PythonSaga benchmark evaluation
Dataset: LingoIITGN/PythonSaga
Similar structure to HumanEval with Indian context
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
class PythonSagaResultDetail:
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
    pythonsaga_code_eval_metric = hf_evaluate.load("code_eval")
    logger.info("code_eval metric for PythonSaga loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'code_eval' metric for PythonSaga: {e}")
    pythonsaga_code_eval_metric = None

def get_fewshot_examples() -> List[Dict[str, str]]:
    """Returns PythonSaga examples for 3-shot prompting."""
    return [
        {
            "task_id": "PythonSaga/0",
            "prompt_example_for_llm": """from typing import List

def extra_marks(marks:List[float])-> float:
    \"\"\"Return the extra marks given to students based on the following criteria:
    - If a student's mark is greater than 100, give them (mark - 100) extra marks
    - If a student's mark is less than 0, give them the absolute value of their mark as extra marks
    - Otherwise, give them 0 extra marks
    
    >>> extra_marks([101, -5, 50])
    6.0
    \"\"\"
    extra = 0
    for mark in marks:
        if mark > 100:
            extra += mark - 100
        elif mark < 0:
            extra += mark
    return extra
""",
        },
        {
            "task_id": "PythonSaga/1",
            "prompt_example_for_llm": """from typing import List

def split_big_bag(big_bag: List[int])->bool:
    \"\"\"Check if the total weight of items in the bag is even.
    
    >>> split_big_bag([1, 2, 3, 4])
    True
    >>> split_big_bag([1, 2, 3])
    False
    \"\"\"
    total_weight = sum(big_bag)
    if total_weight % 2 != 0:
        return False
    return True
""",
        },
        {
            "task_id": "PythonSaga/2",
            "prompt_example_for_llm": """from typing import List

def is_path_crossing(distances: List[int]) -> bool:
    \"\"\"Check if a path crosses itself given a list of distances.
    Start at origin (0,0) and move in directions: North, West, South, East repeatedly.
    
    >>> is_path_crossing([1, 1, 1, 1])
    True
    >>> is_path_crossing([1, 2, 3, 4])
    False
    \"\"\"
    if len(distances) < 4:
        return False
    
    x, y = 0, 0
    visited = {(0, 0)}
    directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # N, W, S, E
    
    for i, dist in enumerate(distances):
        dx, dy = directions[i % 4]
        for _ in range(dist):
            x += dx
            y += dy
            if (x, y) in visited:
                return True
            visited.add((x, y))
    
    return False
""",
        }
    ]

def format_prompt(problem_prompt: str, few_shot_examples: List[Dict[str, str]], use_fewshot: bool) -> str:
    """Formats the prompt for PythonSaga, optionally with few-shot examples."""
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

    # Conservative stop sequences
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
    Add function call to test script if it doesn't have one.
    Similar to HumanEval fix.
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

def evaluate_pythonsaga(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = "LingoIITGN/PythonSaga",
    dataset_split: str = "train",  # PythonSaga uses "train" split
    num_samples_per_task: int = 1,
    k_values: List[int] = [1],
    use_fewshot: bool = True,  # Default to 3-shot
    max_new_tokens_completion: int = 512,
    generation_batch_size: int = 8,
    process_id: int = 0,
    gpu_id: int = 0,
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluates the model on the PythonSaga benchmark.
    """

    if pythonsaga_code_eval_metric is None:
        logger.error("PythonSaga: code_eval metric not available")
        return {"PythonSaga": 0.0, "error_message": "CodeEvalMetricLoadFailed"}

    logger.info(f"P{process_id}: Starting PythonSaga evaluation for model: {model_name_for_logging}")

    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        logger.error("PythonSaga: Tokenizer missing eos_token_id")
        return {"PythonSaga": 0.0, "error_message": "TokenizerMissingEOS"}

    try:
        pythonsaga_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"P{process_id}: Loaded PythonSaga dataset with {len(pythonsaga_dataset)} problems")
    except Exception as e:
        logger.critical(f"P{process_id}: Failed to load dataset '{dataset_name}': {e}")
        return {"PythonSaga": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}

    if len(pythonsaga_dataset) == 0:
        return {"PythonSaga": 0.0}

    few_shot_examples = get_fewshot_examples() if use_fewshot else []
    generation_inputs = []
    problem_references = {}

    for problem in tqdm(pythonsaga_dataset, desc=f"P{process_id} - Preparing PythonSaga"):
        task_id = problem.get("task_id")
        problem_prompt = problem.get("prompt")
        test_script = problem.get("test")
        entry_point = problem.get("entry_point")

        if not all([task_id, problem_prompt, test_script, entry_point]):
            logger.warning(f"P{process_id}: Skipping problem {task_id} due to missing data")
            continue

        full_prompt = format_prompt(problem_prompt, few_shot_examples, use_fewshot)
        
        # Fix the test script to actually call check()
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
        return {"PythonSaga": 0.0, "error_message": "NoValidPrompts"}

    predictions_by_task_id = defaultdict(list)
    detailed_results_log = []

    logger.info(f"P{process_id}: Generating code for {len(generation_inputs)} samples")

    # Use greedy decoding for best pass@1 performance
    generation_params = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens_completion,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": True
    }

    for i in tqdm(range(0, len(generation_inputs), generation_batch_size), desc=f"P{process_id} - PythonSaga Generation"):
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

                detailed_results_log.append(PythonSagaResultDetail(
                    task_id=input_info['task_id'],
                    problem_prompt=input_info['problem_prompt'],
                    full_llm_prompt=input_info['llm_prompt'][:500] + "..." if len(input_info['llm_prompt']) > 500 else input_info['llm_prompt'],
                    entry_point=input_info['entry_point'],
                    raw_generation=raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text,
                    extracted_completion=extracted,
                    full_code_for_eval=full_code,
                    reference_test_script=input_info['test_script']
                ))

        except Exception as e:
            logger.error(f"P{process_id}: Error during generation batch {i}: {e}", exc_info=True)
            for input_info in batch_inputs:
                error_completion = f"# GENERATION ERROR: {e}"
                predictions_by_task_id[input_info['task_id']].append(input_info['problem_prompt'] + error_completion)
                detailed_results_log.append(PythonSagaResultDetail(
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
        return {"PythonSaga": 0.0, "error_message": "NoSamplesForCodeEval"}

    logger.info(f"P{process_id}: Evaluating {len(final_references)} problems with code_eval")
    
    final_scores = {}
    
    try:
        eval_output = pythonsaga_code_eval_metric.compute(
            references=final_references,
            predictions=final_predictions,
            k=k_values
        )
        
        logger.info(f"P{process_id}: code_eval result type: {type(eval_output)}")
        
        # Validate and parse result
        if eval_output is None:
            logger.error(f"P{process_id}: code_eval returned None!")
            return {"PythonSaga": 0.0, "error_message": "code_eval_returned_none"}
        
        # Handle both tuple and dict return formats
        if isinstance(eval_output, tuple):
            if len(eval_output) < 1:
                logger.error(f"P{process_id}: code_eval tuple is empty!")
                return {"PythonSaga": 0.0, "error_message": "empty_tuple"}
            
            scores = eval_output[0]
            detailed_results = eval_output[1] if len(eval_output) > 1 else None
        elif isinstance(eval_output, dict):
            scores = eval_output
            detailed_results = None
        else:
            scores = eval_output
            detailed_results = None
        
        # Validate scores is a dict
        if not isinstance(scores, dict):
            logger.error(f"P{process_id}: Scores is not a dict! Type: {type(scores)}")
            return {"PythonSaga": 0.0, "error_message": f"invalid_scores_type_{type(scores).__name__}"}
        
        if not scores:
            logger.error(f"P{process_id}: Scores dict is empty!")
            return {"PythonSaga": 0.0, "error_message": "empty_scores_dict"}
        
        logger.info(f"P{process_id}: Available score keys: {list(scores.keys())}")
        
        # Extract pass@k scores
        for k_val in k_values:
            metric_key = f"pass@{k_val}"
            
            if metric_key not in scores:
                logger.error(f"P{process_id}: {metric_key} not found in scores!")
                continue
            
            raw_score = scores[metric_key]
            
            if not isinstance(raw_score, (int, float)):
                logger.error(f"P{process_id}: {metric_key} is not a number!")
                continue
            
            score_percentage = float(raw_score) * 100.0
            
            if k_val == k_values[0]:
                final_scores["PythonSaga"] = score_percentage
            
            final_scores[f"PythonSaga_pass@{k_val}"] = score_percentage
            
            logger.info(f"P{process_id}: {metric_key} = {score_percentage:.2f}%")
        
        if not final_scores:
            logger.error(f"P{process_id}: No valid scores extracted from code_eval!")
            return {"PythonSaga": 0.0, "error_message": "no_valid_scores_extracted"}
        
        # Process detailed results if available
        if detailed_results:
            logger.info(f"P{process_id}: Processing detailed test results")
            # Similar processing as HumanEval/MBPP
            if isinstance(detailed_results, (dict, list)):
                logger.info(f"P{process_id}: Detailed results available")
        else:
            logger.warning(f"P{process_id}: No detailed results returned from code_eval")

    except Exception as e:
        logger.error(f"P{process_id}: Error during code evaluation: {e}", exc_info=True)
        final_scores = {"PythonSaga": 0.0, "error_message": f"Exception: {type(e).__name__}: {str(e)}"}
    
    # Final validation
    if "PythonSaga" not in final_scores:
        logger.error(f"P{process_id}: PythonSaga score missing from final_scores! Setting to 0.0")
        final_scores["PythonSaga"] = 0.0

    # Save detailed results
    if save_detailed and detailed_results_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name_for_logging.replace("/", "_").replace("-", "_")
        filename = f"pythonsaga_{safe_model_name}_{dataset_split.replace(':', '_')}_{timestamp}.jsonl"
        
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

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final PythonSaga Pass@1: {final_scores.get('PythonSaga', 0.0):.2f}%")
    
    return final_scores


# Test function for standalone execution
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test PythonSaga")
    test_parser.add_argument("--model_name_test", type=str, default="google/gemma-2-2b-it", help="Model to test")
    test_parser.add_argument("--dataset_split_test", type=str, default="train[:10]", help="Dataset split to test")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=8, help="Generation batch size")
    test_parser.add_argument("--num_samples_test", type=int, default=1, help="Number of samples per task")
    test_parser.add_argument("--k_values_test", type=int, nargs='+', default=[1], help="k values for Pass@k")
    test_parser.add_argument("--use_fewshot", action="store_true", default=True, help="Use few-shot prompting")
    test_parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    
    args = test_parser.parse_args()
    
    setup_logging(level=logging.DEBUG, worker_id="PythonSagaTest")
    logger.info(f"--- Standalone PythonSaga Test: {args.model_name_test} ---")
    
    pipe, _ = initialize_model_pipeline(args.model_name_test, target_device_id=0)
    
    if pipe:
        eval_args = {
            "pipe": pipe,
            "tokenizer": pipe.tokenizer,
            "model_name_for_logging": args.model_name_test,
            "device": pipe.device,
            "dataset_split": args.dataset_split_test,
            "num_samples_per_task": args.num_samples_test,
            "k_values": args.k_values_test,
            "use_fewshot": args.use_fewshot,
            "max_new_tokens_completion": args.max_new_tokens,
            "generation_batch_size": args.gen_batch_size_test,
            "results_dir": "test_results"
        }
        
        try:
            results = evaluate_pythonsaga(**eval_args)
            print("\n" + "="*50)
            print("PYTHONSAGA TEST RESULTS:")
            print("="*50)
            print(json.dumps(results, indent=2))
        except Exception as e:
            logger.error(f"Error during PythonSaga evaluation: {e}")
            print(f"Evaluation failed: {e}")
        finally:
            cleanup_model_resources(pipe, getattr(pipe, 'model', None))
    else:
        logger.error(f"Failed to initialize model {args.model_name_test} for PythonSaga test.")
        print("Model initialization failed!")