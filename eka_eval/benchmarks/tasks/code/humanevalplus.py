# eka_eval/benchmarks/tasks/code/humaneval.py

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict 
from typing import List, Dict, Any, Tuple, Optional
import evaluate as hf_evaluate
import gc
import sys
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
    pass_at_k_metric_details: Optional[Any] = None 
    error_message: str = ""

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
try:
    humaneval_pass_at_k_metric = hf_evaluate.load("code_eval")
    logger.info("code_eval metric for HumanEval loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'code_eval' metric for HumanEval: {e}. HumanEval will not run.", exc_info=True)
    humaneval_pass_at_k_metric = None

DEFAULT_DATASET_NAME = "evalplus/humanevalplus"
DEFAULT_SPLIT = "test" 
DEFAULT_MAX_NEW_TOKENS_COMPLETION = 384 
DEFAULT_GENERATION_BATCH_SIZE = 1 

def _get_humaneval_fewshot_examples() -> List[Dict[str, str]]:
    """Returns a few canonical HumanEval examples for few-shot prompting."""
    samples = [
        {
            "task_id": "HumanEval/0",
            "prompt_example_for_llm": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        },
        {
            "task_id": "HumanEval/2",
            "prompt_example_for_llm": "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than the given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n    return number % 1.0\n",
        }
    ]
    return samples

def _format_humaneval_prompt(
    problem_prompt_to_complete: str, 
    few_shot_examples: List[Dict[str, str]],
    use_fewshot: bool
) -> str:
    """Formats the prompt for HumanEval, optionally with few-shot examples."""
    if use_fewshot and few_shot_examples:
        few_shot_prompt_text = "Complete the following Python functions based on their docstrings. Here are some examples (function signature, docstring, and solution):\n\n"
        for ex in few_shot_examples:
            few_shot_prompt_text += ex["prompt_example_for_llm"].strip() + "\n\n"
        few_shot_prompt_text += "Now, complete the following function (provide only the function body starting after the docstring):\n"
        final_prompt_to_llm = few_shot_prompt_text + problem_prompt_to_complete
    else: # Zero-shot
        final_prompt_to_llm = "Complete the following Python function based on its docstring (provide only the function body starting after the docstring):\n" + problem_prompt_to_complete
    return final_prompt_to_llm

def _extract_humaneval_completion(
    full_generated_text: str, 
    prompt_sent_to_llm: str  
) -> str:
    """
    Extracts the model's code completion part from the full generated text.
    It assumes the model was asked to complete the `original_problem_prompt`,
    which was part of `prompt_sent_to_llm`.
    """
    completion_part = ""
    if full_generated_text.startswith(prompt_sent_to_llm):
        completion_part = full_generated_text[len(prompt_sent_to_llm):]
    else:
        logger.warning(f"Prompt sent to LLM not found at the start of LLM generation. This might indicate an issue or use of return_full_text=False. Raw generation used as completion. Prompt preview: ...{prompt_sent_to_llm[-100:]}, Gen preview: {full_generated_text[:100]}...")
        completion_part = full_generated_text 
    stop_sequences = [
        "\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\nassert ", # Python specific
        "</s>", "<|EOT|>", "\n\n#", "\n\n\"\"\"" # Common model stop tokens or comment starts
    ]
    min_stop_index = len(completion_part)
    for seq in stop_sequences:
        found_idx = completion_part.find(seq)
        if found_idx != -1:
            min_stop_index = min(min_stop_index, found_idx)
    
    cleaned_completion = completion_part[:min_stop_index].rstrip() # Remove trailing whitespace
    if cleaned_completion.startswith("```python"):
        cleaned_completion = cleaned_completion[len("```python"):].lstrip()
    if cleaned_completion.startswith("```"):
        cleaned_completion = cleaned_completion[len("```"):].lstrip()
    if cleaned_completion.endswith("```"):
        cleaned_completion = cleaned_completion[:-len("```")].rstrip()

    return cleaned_completion


# --- Main Evaluation Function ---
def evaluate_humanevalplus(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_SPLIT,
    num_samples_per_task: int = 1,
    k_values: List[int] = [1],    
    use_fewshot: bool = False,
    max_new_tokens_completion: int = DEFAULT_MAX_NEW_TOKENS_COMPLETION,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE, 
    **kwargs
) -> Dict[str, float]:
    """Evaluates the model on the HumanEval benchmark for code generation."""

    if humaneval_pass_at_k_metric is None:
        logger.error("HumanEval: code_eval metric not available. Skipping evaluation.")
        return {"HumanEval": 0.0, "error_message": "CodeEvalMetricLoadFailed"}

    logger.info(f"Starting HumanEval evaluation for model: {model_name_for_logging}")
    logger.info(f"Params: dataset='{dataset_name}', split='{dataset_split}', samples_per_task={num_samples_per_task}, "
                f"k_values={k_values}, use_fewshot={use_fewshot}, max_new_tokens_completion={max_new_tokens_completion}, "
                f"gen_batch_size={generation_batch_size}")

    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        logger.error("HumanEval: Tokenizer does not have a valid eos_token_id. This is crucial for generation. Skipping.")
        return {"HumanEval": 0.0, "error_message": "TokenizerMissingEOS"}

    try:
        humaneval_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"Loaded HumanEval dataset '{dataset_name}' (split: '{dataset_split}') with {len(humaneval_dataset)} problems.")
    except Exception as e:
        logger.critical(f"HumanEval: Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        return {"HumanEval": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}

    if len(humaneval_dataset) == 0:
        logger.warning(f"HumanEval: No problems found in dataset for split '{dataset_split}'. Returning 0.")
        return {"HumanEval": 0.0}

    few_shot_prompt_examples = _get_humaneval_fewshot_examples() if use_fewshot else []
    generation_inputs = []
    problem_references = {} 

    for problem in tqdm(humaneval_dataset, desc="Preparing HumanEval Prompts"):
        task_id = problem.get("task_id")
        problem_prompt_to_complete = problem.get("prompt") 
        test_script = problem.get("test")
        entry_point = problem.get("entry_point")

        if not all([task_id, problem_prompt_to_complete, test_script, entry_point]):
            logger.warning(f"Skipping HumanEval problem (task_id: {task_id or 'Unknown'}) due to missing critical data.")
            continue

        full_prompt_for_llm = _format_humaneval_prompt(problem_prompt_to_complete, few_shot_prompt_examples, use_fewshot)
        for _ in range(num_samples_per_task):
            generation_inputs.append({
                "llm_prompt": full_prompt_for_llm,
                "problem_prompt": problem_prompt_to_complete,
                "task_id": task_id,
                "entry_point": entry_point,
                "test_script": test_script
            })
        problem_references[task_id] = test_script 

    if not generation_inputs:
        logger.error("HumanEval: No valid prompts were prepared for generation. Aborting.")
        return {"HumanEval": 0.0, "error_message": "NoValidPrompts"}
    predictions_by_task_id = defaultdict(list)
    detailed_results_log: List[HumanEvalResultDetail] = []

    logger.info(f"Starting HumanEval code generation for {len(generation_inputs)} total samples "
                f"(num_problems={len(problem_references)}, samples_per_task={num_samples_per_task}).")

    generation_params = {
        "do_sample": True, "temperature": 0.2, "top_p": 0.95,
        "max_new_tokens": max_new_tokens_completion,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": True 
    }
    if not generation_params["do_sample"]: 
        generation_params["temperature"] = None
        generation_params["top_p"] = None

    for i in tqdm(range(0, len(generation_inputs), generation_batch_size), desc="Generating HumanEval Completions", unit="batch"):
        batch_input_infos = generation_inputs[i : i + generation_batch_size]
        batch_llm_prompts = [info['llm_prompt'] for info in batch_input_infos]

        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache() 
            raw_llm_outputs = pipe(batch_llm_prompts, **generation_params)

            for batch_idx, input_info in enumerate(batch_input_infos):
                output_for_this_prompt = raw_llm_outputs[batch_idx]
                raw_generated_text = "#ERROR: No output generated" 
                if output_for_this_prompt and isinstance(output_for_this_prompt, list) and output_for_this_prompt[0]:
                    raw_generated_text = output_for_this_prompt[0].get('generated_text', raw_generated_text)
                elif isinstance(output_for_this_prompt, dict) : 
                    raw_generated_text = output_for_this_prompt.get('generated_text', raw_generated_text)


                extracted_completion = _extract_humaneval_completion(raw_generated_text, input_info['llm_prompt'])
                full_code_for_eval = input_info['problem_prompt'] + extracted_completion
                predictions_by_task_id[input_info['task_id']].append(full_code_for_eval)

                detailed_results_log.append(HumanEvalResultDetail(
                    task_id=input_info['task_id'],
                    problem_prompt=input_info['problem_prompt'],
                    full_llm_prompt=input_info['llm_prompt'],
                    entry_point=input_info['entry_point'],
                    raw_generation=raw_generated_text,
                    extracted_completion=extracted_completion,
                    full_code_for_eval=full_code_for_eval,
                    reference_test_script=input_info['test_script'],
                    passed=None # To be updated later
                ))

        except Exception as e_batch_gen:
            logger.error(f"Error during HumanEval generation batch starting at index {i}: {e_batch_gen}", exc_info=True)
            for input_info in batch_input_infos:
                error_completion = f"# GENERATION ERROR: {e_batch_gen}"
                predictions_by_task_id[input_info['task_id']].append(input_info['problem_prompt'] + error_completion)
                detailed_results_log.append(HumanEvalResultDetail(
                    task_id=input_info['task_id'], problem_prompt=input_info['problem_prompt'],
                    full_llm_prompt=input_info['llm_prompt'], entry_point=input_info['entry_point'],
                    raw_generation=error_completion, extracted_completion=error_completion,
                    full_code_for_eval=input_info['problem_prompt'] + error_completion,
                    reference_test_script=input_info['test_script'], passed=False, error_message=str(e_batch_gen)
                ))

    final_predictions_list = []
    final_references_list = []  
    sorted_task_ids = sorted(problem_references.keys(), key=lambda tid: int(tid.split('/')[-1]))

    for task_id in sorted_task_ids:
        if task_id in predictions_by_task_id and problem_references[task_id]:
            final_predictions_list.append(predictions_by_task_id[task_id])
            final_references_list.append(problem_references[task_id])
        else:
            logger.warning(f"Missing predictions or reference for task_id {task_id}. It will be excluded from code_eval.")

    if not final_predictions_list or not final_references_list:
        logger.error("HumanEval: No valid predictions or references to evaluate with code_eval. Returning 0.")
        return {"HumanEval": 0.0, "error_message": "NoSamplesForCodeEval"}

    logger.info(f"Running HumanEval functional correctness (code_eval) for {len(final_references_list)} problems with k={k_values}...")
    final_scores_dict = {}
    try:
        code_eval_output = humaneval_pass_at_k_metric.compute(
            references=final_references_list,
            predictions=final_predictions_list, 
            k=k_values
        )
        
        pass_at_k_scores: Optional[Dict[str, float]] = None
        detailed_code_eval_results: Optional[List[List[Tuple[int, Dict]]]] = None

        if isinstance(code_eval_output, tuple) and len(code_eval_output) > 0:
            pass_at_k_scores = code_eval_output[0] if isinstance(code_eval_output[0], dict) else None
            if len(code_eval_output) > 1:
                detailed_code_eval_results = code_eval_output[1]
        elif isinstance(code_eval_output, dict): 
            pass_at_k_scores = code_eval_output
        
        if pass_at_k_scores:
            logger.info(f"HumanEval Pass@k scores: {pass_at_k_scores}")
            for k_val in k_values:
                metric_key_in_scores = f"pass@{k_val}" # Standard key from code_eval
                score_value = pass_at_k_scores.get(metric_key_in_scores, 0.0) * 100 
                
                if k_val == k_values[0]: 
                    final_scores_dict["HumanEval"] = score_value
                final_scores_dict[f"HumanEval_pass@{k_val}"] = score_value 
        else:
            logger.error("HumanEval: code_eval did not return valid pass@k scores.")
            final_scores_dict["HumanEval"] = 0.0
        if detailed_code_eval_results and isinstance(detailed_code_eval_results, list):
            logger.debug(f"Processing detailed_code_eval_results (length {len(detailed_code_eval_results)}). First item type: {type(detailed_code_eval_results[0]) if detailed_code_eval_results else 'N/A'}")
            for task_idx_from_code_eval, per_task_sample_results_list in enumerate(detailed_code_eval_results):
                if task_idx_from_code_eval < len(sorted_task_ids): # Ensure we don't go out of bounds
                    original_task_id_str = sorted_task_ids[task_idx_from_code_eval]
 
                    log_entry_to_update = next((entry for entry in detailed_results_log if entry.task_id == original_task_id_str), None)

                    if log_entry_to_update:
                        if isinstance(per_task_sample_results_list, list) and per_task_sample_results_list:
                            first_sample_eval_result_tuple = per_task_sample_results_list[0] 
                            
                            if isinstance(first_sample_eval_result_tuple, tuple) and len(first_sample_eval_result_tuple) == 2:
                                result_data_dict = first_sample_eval_result_tuple[1]
                                if isinstance(result_data_dict, dict):
                                    log_entry_to_update.passed = result_data_dict.get('passed', False)
                                    log_entry_to_update.pass_at_k_metric_details = result_data_dict 
                                    log_entry_to_update.error_message = result_data_dict.get('result', '') if not log_entry_to_update.passed else ""
                                    logger.debug(f"Updated log for {original_task_id_str}: Passed={log_entry_to_update.passed}")
                                else:
                                    logger.warning(f"Detailed result data for task {original_task_id_str} sample 0 is not a dict: {result_data_dict}")
                            else:
                                logger.warning(f"Unexpected structure for first sample result tuple for task {original_task_id_str}: {first_sample_eval_result_tuple}")
                        else:
                            logger.warning(f"Empty or invalid per_task_sample_results_list for task {original_task_id_str}: {per_task_sample_results_list}")
                    else:
                        logger.warning(f"Could not find matching log entry for task_id '{original_task_id_str}' from code_eval results.")
                else:
                    logger.warning(f"task_idx_from_code_eval ({task_idx_from_code_eval}) out of bounds for sorted_task_ids (len {len(sorted_task_ids)}).")
        elif detailed_code_eval_results is not None:
            logger.warning(f"detailed_code_eval_results is not a list as expected. Type: {type(detailed_code_eval_results)}. Cannot update detailed logs.")

    except Exception as e_code_eval:
        logger.error(f"HumanEval: Error during code_eval computation: {e_code_eval}", exc_info=True)
        final_scores_dict["HumanEval"] = 0.0 
        final_scores_dict["error_message"] = "CodeEvalComputationError"

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name_for_logging.replace("/", "_").replace("-", "_") # Make filename safe
    jsonl_filename = f"humaneval_results_{safe_model_name}_{dataset_split.replace('[:','_').replace(']','')}_{timestamp_str}.jsonl"
    
    detailed_results_dir = os.path.join(kwargs.get("results_dir", "results_output"), "humaneval_detailed")
    if not os.path.exists(detailed_results_dir):
        try:
            os.makedirs(detailed_results_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create detailed results directory {detailed_results_dir}: {e}")
            detailed_results_dir = "." 

    full_jsonl_path = os.path.join(detailed_results_dir, jsonl_filename)
    
    logger.info(f"Saving detailed HumanEval results to: {full_jsonl_path}")
    try:
        with open(full_jsonl_path, 'w', encoding='utf-8') as f_jsonl:
            for result_entry in detailed_results_log:
                f_jsonl.write(json.dumps(asdict(result_entry), ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(detailed_results_log)} detailed HumanEval result entries.")
    except Exception as e_json_save:
        logger.error(f"HumanEval: Failed to save detailed results to {full_jsonl_path}: {e_json_save}", exc_info=True)

    if "HumanEval" not in final_scores_dict: 
        final_scores_dict["HumanEval"] = 0.0

    logger.info(f"HumanEval evaluation finished for {model_name_for_logging}. Final scores: {final_scores_dict}")
    return final_scores_dict
