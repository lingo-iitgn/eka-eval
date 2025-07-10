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
class MBPPEvalResult:
    task_id: str; prompt: str; raw_generation: str; extracted_code: str
    reference_code: str; passed: bool; pass_at_k_details: Any; error_message: str = ""

os.environ["HF_ALLOW_CODE_EVAL"] = "1" 
pass_at_k_metric = hf_evaluate.load("code_eval")
print("DEBUG (mbpp.py): code_eval metric loaded successfully.")

def format_mbpp_prompt(example: Dict, include_tests_in_prompt: bool = False) -> str:
    cur_text = example.get("text", example.get("prompt", "No problem description provided."))
    if cur_text is None: cur_text = "No problem description provided."
    example_tests_str = ""
    if include_tests_in_prompt:
        test_assertions = example.get("test_list", [])
        if test_assertions:
            example_tests_str = "\nYour code should pass these example tests:\n" + "\n".join(test_assertions[:min(len(test_assertions), 2)])
            example_tests_str += "\n"
    prompt = (
        f"Problem: {cur_text}\n"
        f"{example_tests_str}"
        f"Write a Python function to solve the above problem. "
        f"Output ONLY the Python code for the function, including necessary imports. "
        f"Your Python code should start with 'def' or 'import'.\n"
        f"Format your entire solution as a single Python code block starting with ```python and ending with ```.\n"
        f"End your code block with the special token [END] just before the final ```.\n"
        f"Begin solution now:\n"
        f"```python\n"
    )
    return prompt

def extract_function_mbpp(generated_text_full: str, prompt_text: str) -> str:
    completion = generated_text_full
    if prompt_text and completion.startswith(prompt_text):
        completion = completion[len(prompt_text):]
    completion = completion.strip()
    match_markdown = re.search(r"```python\n(.*?)(?:\n```|\Z)", completion, re.DOTALL)
    extracted_code_from_markdown = ""
    if match_markdown:
        extracted_code_from_markdown = match_markdown.group(1).strip()
        end_marker_in_markdown = extracted_code_from_markdown.find("[END]")
        if end_marker_in_markdown != -1:
            extracted_code_from_markdown = extracted_code_from_markdown[:end_marker_in_markdown].strip()
        if "def " in extracted_code_from_markdown or "import " in extracted_code_from_markdown:
            return extracted_code_from_markdown
    end_marker_index = completion.find("[END]")
    code_part_before_end = ""
    if end_marker_index != -1:
        code_part_before_end = completion[:end_marker_index].strip()
        match_markdown_before_end = re.search(r"```python\n(.*?)(?:\n```|\Z)", code_part_before_end, re.DOTALL)
        if match_markdown_before_end:
             code_from_md_before_end = match_markdown_before_end.group(1).strip()
             if "def " in code_from_md_before_end or "import " in code_from_md_before_end:
                 return code_from_md_before_end
        lines = code_part_before_end.splitlines()
        code_lines = []
        found_code_start = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "import ", "from ", "class ")): found_code_start = True
            if found_code_start: code_lines.append(line)
        if code_lines: return "\n".join(code_lines).strip()
        if extracted_code_from_markdown: return extracted_code_from_markdown
        return code_part_before_end 
    if extracted_code_from_markdown: return extracted_code_from_markdown
    lines = completion.splitlines()
    potential_code_lines = []
    in_code_block = False
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not in_code_block and stripped.startswith(("def ", "import ", "from ", "class ")):
            in_code_block = True
            potential_code_lines.append(line) 
        elif in_code_block:
            if stripped.lower().startswith(("example:", "note:", "explanation:", "this function will", "test cases:")): break
            potential_code_lines.append(line)
    if potential_code_lines: return "\n".join(potential_code_lines).strip()
    return completion

# --- SAFE_GENERATE_FOR_MBPP DEFINITION RESTORED HERE ---
def safe_generate_for_mbpp(pipe, prompts: List[str], max_retries=3, generation_config_override=None):
    """Generates text from the model with retry logic and GPU memory cleanup for MBPP."""
    end_token_id = pipe.tokenizer.convert_tokens_to_ids("[END]")
    eos_ids_to_use = []
    if end_token_id != pipe.tokenizer.unk_token_id:
        eos_ids_to_use.append(end_token_id)
    if pipe.tokenizer.eos_token_id is not None and pipe.tokenizer.eos_token_id not in eos_ids_to_use:
        eos_ids_to_use.append(pipe.tokenizer.eos_token_id)
    if not eos_ids_to_use:
        eos_ids_to_use = pipe.tokenizer.eos_token_id # Fallback if [END] fails and no default EOS
        if eos_ids_to_use is None: # Absolute fallback if tokenizer has no eos_token_id either
            print("CRITICAL WARNING (MBPP safe_generate): No EOS or [END] token ID found. Generation will likely not stop correctly.")
            # Forcing a common token ID like newline if everything else fails, though this is not ideal.
            # Or, let the model run to max_new_tokens. For now, let transformers pipeline handle it if eos_ids_to_use is None.
            pass # eos_ids_to_use remains None
        else:
            print("Warning (MBPP safe_generate): No valid [END] token ID found, using default eos_token_id.")


    base_generation_config = {
        "do_sample": True, "temperature": 0.1, "top_p": 0.95, "top_k": 50,
        "max_new_tokens": 400, "num_return_sequences": 1,
        "pad_token_id": pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id is not None else pipe.tokenizer.eos_token_id,
        "eos_token_id": eos_ids_to_use, "return_full_text": True,
        "repetition_penalty": 1.05, "length_penalty": 1.0,
    }
    current_gen_config = base_generation_config.copy()
    if generation_config_override: current_gen_config.update(generation_config_override)

    # print(f"DEBUG MBPP safe_generate: Using generation config: {current_gen_config}")

    for attempt in range(max_retries):
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # The pipeline should handle batching if its internal batch_size > 1 and multiple prompts are given.
            # The batch_size argument to evaluate_mbpp controls how many prompts we feed to pipe() at once.
            outputs = pipe(prompts, **current_gen_config) 
            
            # Ensure outputs is List[List[Dict]]
            if prompts and len(prompts) == 1 and isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                outputs = [outputs] # Wrap if pipe returns List[Dict] for single prompt

            results = []
            for i, output_for_prompt in enumerate(outputs): # output_for_prompt should be List[Dict]
                if output_for_prompt and isinstance(output_for_prompt, list) and \
                   len(output_for_prompt) > 0 and isinstance(output_for_prompt[0], dict):
                    generated_text = output_for_prompt[0].get('generated_text', f'#GenFail prompt {i}: No text')
                    results.append([{"generated_text": generated_text}]) 
                else:
                    print(f"Warn: Bad output struct prompt {i} MBPP. Expected List[Dict], got: {type(output_for_prompt)}. Full output: {output_for_prompt}")
                    results.append([{"generated_text": f"#GenFail prompt {i}: Bad output struct"}])
            return results # Expected: List of (List of Dicts), one inner list per prompt
        except Exception as e:
            print(f"Err MBPP gen attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()
                gc.collect()
            else: return [[{"generated_text": f"#GenFail max retries: {e}"}] for _ in prompts] # Match expected structure
    return [[{"generated_text": "#GenFail all retries"}] for _ in prompts] # Match expected structure
# --- END SAFE_GENERATE_FOR_MBPP DEFINITION ---


def evaluate_mbpp(
    model_name: str,
    pipe: pipeline, 
    model_size_gb: float,
    batch_size: int = 8, 
    num_samples_per_task: int = 1, 
    dataset_split: str = "test[:5]",
    k_values: List[int] = [1],
    include_tests_in_prompt: bool = False
) -> Dict[str, float]:

    print(f"\n--- Starting MBPP evaluation for {model_name} ---")
    print(f"Parameters: batch_size(for generation)={batch_size}, samples_per_task={num_samples_per_task}, split='{dataset_split}', k_values={k_values}, include_tests_in_prompt={include_tests_in_prompt}")

    if "[END]" not in pipe.tokenizer.get_vocab():
         print("Warning (MBPP): '[END]' token not found in tokenizer vocab during MBPP eval.")

    dataset_loaded = None
    # ... (Dataset loading logic - kept same as your last version) ...
    try:
        dataset_loaded = load_dataset("google-research-datasets/mbpp", "full", split=dataset_split, trust_remote_code=True)
        print(f"Loaded MBPP 'full' (from google-research-datasets) dataset split: '{dataset_split}', size: {len(dataset_loaded)}")
    except Exception as e:
        print(f"Failed to load 'google-research-datasets/mbpp' ('full'): {e}. Trying 'mbpp' ('sanitized')...")
        try:
            dataset_loaded = load_dataset("mbpp", "sanitized", split=dataset_split, trust_remote_code=True)
            print(f"Loaded MBPP 'sanitized' (from HF Hub) dataset split: '{dataset_split}', size: {len(dataset_loaded)}")
        except Exception as e2:
            print(f"FATAL: Failed to load any MBPP dataset (split '{dataset_split}'): {e2}. Cannot proceed.")
            return {"MBPP": 0.0}
    if not dataset_loaded or len(dataset_loaded) == 0:
        print(f"No problems found in MBPP dataset for split '{dataset_split}'. Returning 0.")
        return {"MBPP": 0.0}
    mbpp_problems = list(dataset_loaded)
    print(f"Evaluating on {len(mbpp_problems)} MBPP problems from split '{dataset_split}'.")


    all_prompts_info = [] 
    references_for_eval = {} 
    detailed_task_results: List[MBPPEvalResult] = [] 

    for example_idx, example in enumerate(mbpp_problems):
        prompt_text_formatted = format_mbpp_prompt(example, include_tests_in_prompt=include_tests_in_prompt)
        task_id_val = example.get('task_id')
        if task_id_val is None: print(f"Warn: Ex {example_idx} missing 'task_id'. Skipping."); continue
        task_id_str = str(task_id_val)
        original_problem_description = example.get("text", example.get("prompt", "N/A"))
        test_setup_code = example.get('test_setup_code', example.get('prompt', ''))
        test_list = example.get('test_list')
        if test_list is None: print(f"Warn: Ex {example_idx} (task_id: {task_id_str}) missing 'test_list'. Skipping."); continue

        for _ in range(num_samples_per_task):
            all_prompts_info.append({
                'prompt': prompt_text_formatted, 'task_id': task_id_str,
                'original_problem_text': original_problem_description,
                'test_setup_code': test_setup_code,
                'ground_truth_code': example.get('code', 'N/A')
            })
        full_test_script = (test_setup_code if test_setup_code else "") + "\n\n" + "\n".join(test_list)
        references_for_eval[task_id_str] = full_test_script
    
    if not all_prompts_info: print("No valid prompts for MBPP. Aborting."); return {"MBPP": 0.0}
        
    prompts_to_generate = [item['prompt'] for item in all_prompts_info]
    predictions_for_eval_dict = defaultdict(list)
    
    print(f"Starting code generation for {len(prompts_to_generate)} total samples with pipeline batch size {batch_size}...")
    generation_config_mbpp = { "temperature": 0.1, "top_p": 0.95, "max_new_tokens": 400 }

    num_batches = (len(prompts_to_generate) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Generating MBPP completions", disable=False, 
              file=sys.stdout, mininterval=0.5, dynamic_ncols=True, unit="batch") as pbar:
        for i in range(0, len(prompts_to_generate), batch_size): # This batch_size is from evaluate_mbpp's args
            batch_prompt_texts = prompts_to_generate[i : i + batch_size]
            batch_original_info_list = all_prompts_info[i : i + batch_size]
            
            batch_generated_outputs_raw = safe_generate_for_mbpp(
                pipe, batch_prompt_texts, generation_config_override=generation_config_mbpp
            )

            for j, raw_output_list_for_one_prompt in enumerate(batch_generated_outputs_raw):
                original_info = batch_original_info_list[j]
                task_id = original_info['task_id']
                current_prompt_text = original_info['prompt']
                raw_generated_text_from_model = "#GenFail: No valid output"
                if raw_output_list_for_one_prompt and isinstance(raw_output_list_for_one_prompt, list) and \
                   len(raw_output_list_for_one_prompt) > 0 and isinstance(raw_output_list_for_one_prompt[0], dict) and \
                   'generated_text' in raw_output_list_for_one_prompt[0]:
                    raw_generated_text_from_model = raw_output_list_for_one_prompt[0]['generated_text']
                
                extracted_code = extract_function_mbpp(raw_generated_text_from_model, current_prompt_text)
                print(f"\n--- DEBUG Task ID: {task_id} ---")
                print(f"Prompt:\n{current_prompt_text}")
                print(f"Raw Generation (len {len(raw_generated_text_from_model)}):\n{raw_generated_text_from_model[:500]}...")
                print(f"Extracted Code for Eval (len {len(extracted_code)}):\n{extracted_code[:500]}...")
                
                final_code_for_code_eval = ""
                setup_code = original_info.get('test_setup_code', '')
                if setup_code: final_code_for_code_eval += setup_code + "\n\n"
                final_code_for_code_eval += extracted_code
                predictions_for_eval_dict[task_id].append(final_code_for_code_eval)

                if num_samples_per_task == 1:
                    detailed_task_results.append(MBPPEvalResult(
                        task_id=task_id, prompt=current_prompt_text,
                        raw_generation=raw_generated_text_from_model,
                        extracted_code=extracted_code, 
                        reference_code=references_for_eval.get(task_id, "REF NOT FOUND"),
                        passed=False, pass_at_k_details=None, error_message=""
                    ))
            pbar.update(1)

    # --- (Rest of the code: preparing lists for code_eval, calling compute, processing results, saving JSONL) ---
    # This part should be the same as the version that correctly handled the tuple output from code_eval
    final_predictions_list_of_lists = []
    final_references_list = []
    valid_task_ids_in_preds = set(predictions_for_eval_dict.keys())
    sorted_task_ids_for_eval = sorted(
        [tid for tid in references_for_eval.keys() if tid in valid_task_ids_in_preds], 
        key=lambda x: int(x)
    )
    if not sorted_task_ids_for_eval: print("No tasks with refs & preds. MBPP eval cannot proceed."); return {"MBPP": 0.0}

    for task_id in sorted_task_ids_for_eval:
        final_references_list.append(references_for_eval[task_id])
        samples_for_task = predictions_for_eval_dict.get(task_id, ["#No pred for task"])
        if not isinstance(samples_for_task, list): samples_for_task = [str(samples_for_task)]
        final_predictions_list_of_lists.append(samples_for_task)

    if not final_references_list or not final_predictions_list_of_lists:
        print("No refs or preds to eval for MBPP. Return 0."); return {"MBPP": 0.0}

    print(f"Running functional correctness evaluation for {len(final_references_list)} MBPP problems using k={k_values}...")
    results_summary_dict = {}
    try:
        evaluation_output_tuple = pass_at_k_metric.compute(
            references=final_references_list, predictions=final_predictions_list_of_lists, k=k_values,
        )
        pass_at_k_scores_dict = None; detailed_code_eval_results = None
        if isinstance(evaluation_output_tuple, tuple) and len(evaluation_output_tuple) > 0 and isinstance(evaluation_output_tuple[0], dict):
            pass_at_k_scores_dict = evaluation_output_tuple[0]
            if len(evaluation_output_tuple) > 1: detailed_code_eval_results = evaluation_output_tuple[1]
            print(f"DEBUG MBPP: code_eval tuple. Scores: {pass_at_k_scores_dict}. Details type: {type(detailed_code_eval_results)}")
        elif isinstance(evaluation_output_tuple, dict):
            pass_at_k_scores_dict = evaluation_output_tuple
            print(f"DEBUG MBPP: code_eval dict: {pass_at_k_scores_dict}")
        else:
            print(f"ERROR MBPP: Unexpected result format: {evaluation_output_tuple}")
            raise ValueError(f"Unexpected result format from pass_at_k_metric.compute: {evaluation_output_tuple}")

        if pass_at_k_scores_dict is None: results_summary_dict["MBPP"] = 0.0
        else:
            for k_val in k_values:
                metric_key = f"pass@{k_val}"
                if metric_key in pass_at_k_scores_dict:
                    score = pass_at_k_scores_dict[metric_key] * 100 
                    if k_val == k_values[0]: results_summary_dict["MBPP"] = score
                    results_summary_dict[f"MBPP_pass@{k_val}"] = score 
                    print(f"MBPP {metric_key}: {score:.2f}%")
                else:
                    if k_val == k_values[0]: results_summary_dict["MBPP"] = 0.0
            if "MBPP" not in results_summary_dict: results_summary_dict["MBPP"] = 0.0
        
        if detailed_code_eval_results and isinstance(detailed_code_eval_results, defaultdict):
            for task_idx_eval, results_for_task_list in detailed_code_eval_results.items():
                if task_idx_eval < len(sorted_task_ids_for_eval):
                    original_task_id_str = sorted_task_ids_for_eval[task_idx_eval]
                    for res_item in detailed_task_results:
                        if res_item.task_id == original_task_id_str:
                            if results_for_task_list: 
                                first_sample_result_info = results_for_task_list[0]
                                if isinstance(first_sample_result_info, tuple) and len(first_sample_result_info) == 2:
                                    result_data = first_sample_result_info[1]
                                    res_item.passed = result_data.get('passed', False)
                                    res_item.pass_at_k_details = result_data 
                                    res_item.error_message = result_data.get('result', '') if not res_item.passed else ""
                                break 
    except Exception as e:
        print(f"\033[91mError computing/processing Pass@k for MBPP: {e}\033[0m")
        import traceback; traceback.print_exc()
        results_summary_dict["MBPP"] = 0.0
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name_part = model_name.replace("/", "_")
    json_output_filename = f"mbpp_detailed_results_{safe_model_name_part}_{timestamp_str}.jsonl"
    print(f"Saving detailed MBPP results to: {json_output_filename}")
    try:
        with open(json_output_filename, 'w', encoding='utf-8') as f_json:
            for result_entry in detailed_task_results:
                f_json.write(json.dumps(asdict(result_entry)) + "\n")
        print(f"Successfully saved detailed results to {json_output_filename}")
    except Exception as e_json: print(f"ERROR MBPP: Failed to save detailed results to JSON: {e_json}")

    return results_summary_dict