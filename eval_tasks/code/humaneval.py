import torch
import re
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import json
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import evaluate as hf_evaluate 
import gc
import sys 

@dataclass
class HumanEvalResultDetail:
    """Contains detailed results for a single HumanEval task."""
    task_id: str
    prompt: str
    entry_point: str
    raw_generation: str
    extracted_code: str
    full_code_for_eval: str
    reference_test_script: str
    passed: bool
    pass_at_k_details: Any
    error_message: str = ""

os.environ["HF_ALLOW_CODE_EVAL"] = "1" 
try:
    pass_at_k_metric_humaneval = hf_evaluate.load("code_eval")
    print("DEBUG (humaneval.py): code_eval metric loaded successfully for HumanEval.")
except Exception as e:
    print(f"CRITICAL ERROR (humaneval.py): Failed to load code_eval metric: {e}")
    pass_at_k_metric_humaneval = None

def get_humaneval_fewshot_examples() -> List[Dict]:
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

def format_humaneval_prompt(problem_prompt: str, few_shot_examples: List[Dict], use_fewshot: bool) -> str:
    """Formats the prompt for HumanEval, optionally with few-shot examples."""
    if use_fewshot and few_shot_examples:
        few_shot_prompt_text = "Complete the following Python functions based on their docstrings. Here are some examples:\n\n"
        for ex in few_shot_examples:
            few_shot_prompt_text += ex["prompt_example_for_llm"] + "\n\n"
        few_shot_prompt_text += "Now, complete the following function:\n"
        final_prompt = few_shot_prompt_text + problem_prompt
    else:
        final_prompt = "Complete the following Python function based on its docstring:\n" + problem_prompt
    
    return final_prompt

def extract_humaneval_completion(generated_text_full: str, original_prompt_for_completion: str) -> str:
    """Extracts the model's completion part for HumanEval."""
    if generated_text_full.startswith(original_prompt_for_completion):
        completion = generated_text_full[len(original_prompt_for_completion):]
    else:
        completion = generated_text_full

    stop_lines = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\nassert ", "\n\n#", "\n\n\"\"\"", "</s>"]
    
    min_stop_idx = len(completion)
    for stop_seq in stop_lines:
        stop_idx = completion.find(stop_seq)
        if stop_idx != -1:
            min_stop_idx = min(min_stop_idx, stop_idx)
    
    completion = completion[:min_stop_idx]
    
    if completion.strip().endswith("```"):
        completion = completion.strip()[:-3].strip()

    return completion

def safe_generate_for_humaneval(pipe, prompts: List[str], max_retries=3, generation_params_override=None):
    """Safely generate text for HumanEval prompts."""
    base_generation_params = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 384,
        "num_return_sequences": 1,
        "pad_token_id": pipe.tokenizer.eos_token_id,
        "eos_token_id": pipe.tokenizer.eos_token_id,
        "return_full_text": True,
    }
    current_gen_params = base_generation_params.copy()
    if generation_params_override:
        current_gen_params.update(generation_params_override)
    
    for attempt in range(max_retries):
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            outputs = pipe(prompts, **current_gen_params)
            if prompts and len(prompts) == 1 and isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                outputs = [outputs]

            results = []
            for i, output_for_prompt_list in enumerate(outputs):
                if output_for_prompt_list and isinstance(output_for_prompt_list, list) and \
                   len(output_for_prompt_list) > 0 and isinstance(output_for_prompt_list[0], dict):
                    generated_text = output_for_prompt_list[0].get('generated_text', f'#GenFail prompt {i}: No text')
                    results.append([{"generated_text": generated_text}])
                else:
                    print(f"Warn (humaneval.py): Bad output struct prompt {i}. Got: {output_for_prompt_list}")
                    results.append([{"generated_text": f"#GenFail prompt {i}: Bad output struct"}])
            return results
        except Exception as e:
            print(f"Error (humaneval.py): Generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()
                gc.collect(); print("DEBUG (humaneval.py): Retrying generation...")
            else:
                print(f"Error (humaneval.py): Max retries reached for generation.")
                return [[{"generated_text": f"#GenFail max retries: {e}"}] for _ in prompts]
    return [[{"generated_text": "#GenFail all retries"}] for _ in prompts]

def evaluate_humaneval(
    model_name: str,
    pipe: pipeline, 
    model_size_gb: float,
    batch_size: int = 1,
    num_samples_per_task: int = 1,
    dataset_split: str = "test[:5]",
    k_values: List[int] = [1],
    use_fewshot: bool = False
) -> Dict[str, float]:
    """Evaluates model performance on HumanEval benchmark."""

    if pass_at_k_metric_humaneval is None:
        print("ERROR (humaneval.py): code_eval metric not available. Skipping HumanEval.")
        return {"HumanEval": 0.0}

    print(f"\n--- Starting HumanEval evaluation for {model_name} ---")
    print(f"Parameters: batch_size(for generation loop)={batch_size}, samples_per_task={num_samples_per_task}, split='{dataset_split}', k_values={k_values}, use_fewshot={use_fewshot}")

    if not hasattr(pipe, 'tokenizer') or pipe.tokenizer is None:
        print("ERROR (humaneval.py): Pipeline object does not have a valid tokenizer. Skipping.")
        return {"HumanEval": 0.0}

    try:
        dataset_loaded = load_dataset("openai_humaneval", split=dataset_split, trust_remote_code=True)
        print(f"Loaded HumanEval dataset (openai_humaneval) split: '{dataset_split}', size: {len(dataset_loaded)}")
    except Exception as e:
        print(f"FATAL (humaneval.py): Failed to load HumanEval dataset: {e}. Cannot proceed.")
        return {"HumanEval": 0.0}

    humaneval_problems = list(dataset_loaded)
    if not humaneval_problems:
        print(f"No problems found in HumanEval dataset for split '{dataset_split}'. Returning 0.")
        return {"HumanEval": 0.0}
    print(f"Evaluating on {len(humaneval_problems)} HumanEval problems from split '{dataset_split}'.")

    few_shot_examples = get_humaneval_fewshot_examples() if use_fewshot else []
    
    all_prompts_info_for_generation = []
    references_for_eval = {}
    detailed_task_results_log: List[HumanEvalResultDetail] = []

    for problem_idx, problem_data in enumerate(tqdm(humaneval_problems, desc="Preparing HumanEval Prompts", file=sys.stdout)):
        task_id = problem_data.get("task_id")
        original_prompt = problem_data.get("prompt")
        test_script = problem_data.get("test")
        entry_point = problem_data.get("entry_point")

        if not all([task_id, original_prompt, test_script, entry_point]):
            print(f"Warn (humaneval.py): Skipping problem {problem_idx} due to missing data: {problem_data}")
            continue

        full_formatted_prompt = format_humaneval_prompt(original_prompt, few_shot_examples, use_fewshot)
        
        for _ in range(num_samples_per_task):
            all_prompts_info_for_generation.append({
                'full_formatted_prompt': full_formatted_prompt,
                'original_problem_prompt': original_prompt,
                'task_id': task_id,
                'entry_point': entry_point
            })
        references_for_eval[task_id] = test_script

    if not all_prompts_info_for_generation:
        print("No valid prompts prepared for HumanEval. Aborting."); return {"HumanEval": 0.0}

    prompts_to_feed_llm = [item['full_formatted_prompt'] for item in all_prompts_info_for_generation]
    predictions_for_code_eval_dict = defaultdict(list) 

    print(f"Starting HumanEval code generation for {len(prompts_to_feed_llm)} total samples with generation batch size {batch_size}...")
    humaneval_gen_params_override = {"temperature": 0.2, "top_p": 0.95, "max_new_tokens": 384}

    num_batches = (len(prompts_to_feed_llm) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Generating HumanEval completions", file=sys.stdout, mininterval=1.0, dynamic_ncols=True, unit="batch") as pbar:
        for i in range(0, len(prompts_to_feed_llm), batch_size):
            current_batch_prompts_llm = prompts_to_feed_llm[i : i + batch_size]
            current_batch_info = all_prompts_info_for_generation[i : i + batch_size]

            raw_outputs_batch = safe_generate_for_humaneval(
                pipe, current_batch_prompts_llm, generation_params_override=humaneval_gen_params_override
            )

            for j, raw_output_list_for_one_sample in enumerate(raw_outputs_batch):
                original_info = current_batch_info[j]
                task_id = original_info['task_id']
                original_problem_prompt = original_info['original_problem_prompt']
                entry_point = original_info['entry_point']
                
                raw_generated_text_from_model = "#GenFail: No output"
                if raw_output_list_for_one_sample and isinstance(raw_output_list_for_one_sample, list) and \
                   len(raw_output_list_for_one_sample) > 0 and isinstance(raw_output_list_for_one_sample[0], dict) and \
                   'generated_text' in raw_output_list_for_one_sample[0]:
                    raw_generated_text_from_model = raw_output_list_for_one_sample[0]['generated_text']

                extracted_completion_part = extract_humaneval_completion(raw_generated_text_from_model, original_info['full_formatted_prompt'])
                
                full_code_for_eval = original_problem_prompt + extracted_completion_part

                print(f"\n--- DEBUG HumanEval Task ID: {task_id} ---")
                print(f"Original Problem Prompt (sent to LLM as part of a larger prompt):\n{original_problem_prompt[:300]}...")
                print(f"Raw LLM Generation (len {len(raw_generated_text_from_model)}, first 500 of raw):\n{raw_generated_text_from_model[:500]}...")
                print(f"Extracted Completion Part (len {len(extracted_completion_part)}):\n{extracted_completion_part[:300]}...")
                print(f"Full Code for Eval (Original Prompt + Extracted, len {len(full_code_for_eval)}):\n{full_code_for_eval[:500]}...")

                predictions_for_code_eval_dict[task_id].append(full_code_for_eval)

                if num_samples_per_task == 1:
                    detailed_task_results_log.append(HumanEvalResultDetail(
                        task_id=task_id, prompt=original_info['full_formatted_prompt'], entry_point=entry_point,
                        raw_generation=raw_generated_text_from_model,
                        extracted_code=extracted_completion_part,
                        full_code_for_eval=full_code_for_eval,
                        reference_test_script=references_for_eval.get(task_id, "REF TEST NOT FOUND"),
                        passed=False, pass_at_k_details=None, error_message=""
                    ))
            pbar.update(1)

    final_predictions_for_code_eval = []
    final_references_for_code_eval = []
    
    sorted_task_ids_for_final_eval = sorted(
        references_for_eval.keys(), 
        key=lambda tid: int(tid.split('/')[-1])
    )

    for task_id in sorted_task_ids_for_final_eval:
        if task_id in predictions_for_code_eval_dict:
            final_references_for_code_eval.append(references_for_eval[task_id])
            final_predictions_for_code_eval.append(predictions_for_code_eval_dict[task_id])
        else:
            print(f"Warn (humaneval.py): No predictions found for task_id {task_id}, though reference exists. Skipping for final eval.")

    if not final_references_for_code_eval or not final_predictions_for_code_eval:
        print("No valid references or predictions for HumanEval code_eval. Aborting."); return {"HumanEval": 0.0}

    print(f"Running HumanEval functional correctness evaluation for {len(final_references_for_code_eval)} problems using k={k_values}...")
    results_summary_dict = {}
    try:
        evaluation_output_tuple_he = pass_at_k_metric_humaneval.compute(
            references=final_references_for_code_eval,
            predictions=final_predictions_for_code_eval,
            k=k_values,
        )
        pass_at_k_scores_he = None
        detailed_code_eval_results_he = None

        if isinstance(evaluation_output_tuple_he, tuple) and len(evaluation_output_tuple_he) > 0 and isinstance(evaluation_output_tuple_he[0], dict):
            pass_at_k_scores_he = evaluation_output_tuple_he[0]
            if len(evaluation_output_tuple_he) > 1: detailed_code_eval_results_he = evaluation_output_tuple_he[1]
        elif isinstance(evaluation_output_tuple_he, dict):
            pass_at_k_scores_he = evaluation_output_tuple_he
        else:
            print(f"ERROR HumanEval: Unexpected result format: {evaluation_output_tuple_he}")
            raise ValueError(f"Unexpected result format from code_eval for HumanEval: {evaluation_output_tuple_he}")

        if pass_at_k_scores_he is None: results_summary_dict["HumanEval"] = 0.0
        else:
            for k_val in k_values:
                metric_key = f"pass@{k_val}"
                if metric_key in pass_at_k_scores_he:
                    score = pass_at_k_scores_he[metric_key] * 100 
                    if k_val == k_values[0]: results_summary_dict["HumanEval"] = score
                    results_summary_dict[f"HumanEval_pass@{k_val}"] = score 
                    print(f"HumanEval {metric_key}: {score:.2f}%")
                else:
                    if k_val == k_values[0]: results_summary_dict["HumanEval"] = 0.0
            if "HumanEval" not in results_summary_dict: results_summary_dict["HumanEval"] = 0.0
        
        if detailed_code_eval_results_he and isinstance(detailed_code_eval_results_he, defaultdict):
            for task_idx_eval, results_for_task_list in detailed_code_eval_results_he.items():
                if task_idx_eval < len(sorted_task_ids_for_final_eval):
                    original_task_id_str = sorted_task_ids_for_final_eval[task_idx_eval]
                    for res_item in detailed_task_results_log:
                        if res_item.task_id == original_task_id_str:
                            if results_for_task_list: 
                                first_sample_res_tuple = results_for_task_list[0]
                                if isinstance(first_sample_res_tuple, tuple) and len(first_sample_res_tuple) == 2:
                                    result_data = first_sample_res_tuple[1]
                                    res_item.passed = result_data.get('passed', False)
                                    res_item.pass_at_k_details = result_data 
                                    res_item.error_message = result_data.get('result', '') if not res_item.passed else ""
                                break 
    except Exception as e:
        print(f"\033[91mError computing/processing Pass@k for HumanEval: {e}\033[0m")
        import traceback; traceback.print_exc()
        results_summary_dict["HumanEval"] = 0.0
    
    timestamp_str_he = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name_he = model_name.replace("/", "_").replace("-", "_")
    json_output_filename_he = f"humaneval_detailed_results_{safe_model_name_he}_{timestamp_str_he}.jsonl"
    results_dir_he = "humaneval_results"
    if not os.path.exists(results_dir_he): os.makedirs(results_dir_he)
    full_json_path_he = os.path.join(results_dir_he, json_output_filename_he)
    
    print(f"Saving detailed HumanEval results to: {full_json_path_he}")
    try:
        with open(full_json_path_he, 'w', encoding='utf-8') as f_json_he:
            for result_entry in detailed_task_results_log:
                f_json_he.write(json.dumps(asdict(result_entry)) + "\n")
        print(f"Successfully saved detailed HumanEval results to {full_json_path_he}")
    except Exception as e_json_he: print(f"ERROR HumanEval: Failed to save detailed results: {e_json_he}")

    return results_summary_dict