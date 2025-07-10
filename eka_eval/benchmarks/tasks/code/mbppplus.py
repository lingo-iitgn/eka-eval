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
from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)

@dataclass
class MBPPEvalResultDetail:
    task_id: str
    problem_description: str
    full_llm_prompt: str
    raw_generation: str
    extracted_code: str
    code_for_eval: str
    reference_test_script: str
    ground_truth_code: str
    passed: Optional[bool] = None
    error_message: str = ""

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

try:
    mbpp_pass_at_k_metric = hf_evaluate.load("code_eval")
except Exception as e:
    logger.critical(f"Failed to load 'code_eval' metric for MBPP: {e}")
    mbpp_pass_at_k_metric = None

DEFAULT_DATASET_NAME_PRIMARY = "evalplus/mbppplus"
DEFAULT_DATASET_CONFIG_PRIMARY = None
DEFAULT_DATASET_NAME_FALLBACK = "mbpp"
DEFAULT_DATASET_CONFIG_FALLBACK = "sanitized"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_NEW_TOKENS_COMPLETION = 512
DEFAULT_GENERATION_BATCH_SIZE = 1
DEFAULT_NUM_FEWSHOT_MBPP = 0
DEFAULT_NUM_SAMPLES_PER_TASK = 1
DEFAULT_K_VALUES = [1]
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "mbpp_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "mbpp_3shot"
PROMPT_FILE_BENCHMARK_KEY = "mbpp"
PROMPT_FILE_CATEGORY = "code"

def _get_mbpp_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    if num_few_shot <= 0:
        return []
    loaded_examples_list = get_prompt_data(PROMPT_FILE_BENCHMARK_KEY, data_key="default_few_shot_examples_mbpp", specific_task_group=prompt_file_category)
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        return loaded_examples_list[:num_few_shot]
    return []

def _format_mbpp_prompt_with_template(
    example: Dict, 
    num_few_shot: int = 0, 
    include_tests_in_prompt: bool = True,
    prompt_file_category: str = PROMPT_FILE_CATEGORY
) -> str:
    problem_description = example.get("text", example.get("prompt", "No problem description provided."))
    if not isinstance(problem_description, str):
        problem_description = str(problem_description)

    test_examples_str = ""
    if include_tests_in_prompt:
        test_assertions = example.get("test_list", [])
        if test_assertions and isinstance(test_assertions, list):
            num_to_show = min(len(test_assertions), 2)
            if num_to_show > 0:
                test_examples_str = "Example tests:\n" + "\n".join(test_assertions[:num_to_show]) + "\n\n"

    if num_few_shot > 0:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT
        few_shot_examples = _get_mbpp_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        
        try:
            formatted_prompt = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=few_shot_examples,
                problem_description=problem_description,
                test_examples=test_examples_str,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"
    else:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT
        try:
            formatted_prompt = format_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                problem_description=problem_description,
                test_examples=test_examples_str,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"

def _extract_mbpp_completion(generated_text: str, prompt_text_sent_to_llm: Optional[str] = None) -> str:
    completion_part = generated_text
    if prompt_text_sent_to_llm and generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    
    if len(completion_part) == 0:
        return ""
    
    if completion_part.startswith("Error: Prompt formatting failed"):
        return "# Error: Prompt formatting failed\ndef placeholder():\n    pass"
    
    patterns = [
        r"```python\s*\n(.*?)\s*\[END\]\s*\n?\s*```",
        r"```python\s*\n(.*?)\s*\n?\s*```",
        r"```\s*\n(.*?)\s*\n?\s*```",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion_part, re.DOTALL)
        if match:
            extracted = match.group(1).strip().replace("[END]", "").strip()
            return _clean_extracted_code(extracted)
    
    end_marker_idx = completion_part.find("[END]")
    if end_marker_idx != -1:
        extracted = completion_part[:end_marker_idx].strip()
        if extracted and (extracted.startswith(("def ", "import ", "class ", "from "))):
            return _clean_extracted_code(extracted)
    
    lines = completion_part.split('\n')
    code_lines = []
    in_function = False
    function_indent = 0
    
    for line in lines:
        stripped = line.strip()
        if any(skip in stripped.lower() for skip in ["the following", "this command", "note that", "you can"]):
            continue
        
        if stripped.startswith("def ") and "(" in stripped and "):" in stripped:
            in_function = True
            function_indent = len(line) - len(line.lstrip())
            code_lines.append(line)
            continue
        
        if in_function:
            current_indent = len(line) - len(line.lstrip())
            if not stripped or current_indent > function_indent:
                code_lines.append(line)
            elif current_indent <= function_indent and stripped:
                break
        elif stripped.startswith(("import ", "from ", "class ")):
            code_lines.append(line)
    
    if code_lines:
        extracted = '\n'.join(code_lines).strip()
        if 'def ' in extracted:
            return _clean_extracted_code(extracted)
    
    return "# Could not extract valid code from generation\ndef placeholder():\n    pass"

def _clean_extracted_code(code: str) -> str:
    if not code.strip():
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.strip().startswith('return ') and not line.startswith('    '):
            line = '    ' + line.strip()
        if any(bad in line for bad in ['>>>', '# prints', '# 1, 2, 3']):
            continue
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    if 'def ' in result and 'return ' in result:
        lines = result.split('\n')
        fixed_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                fixed_lines.append(line)
            elif in_function and line.strip() and not line.startswith('    '):
                fixed_lines.append('    ' + line.strip())
            else:
                fixed_lines.append(line)
        
        result = '\n'.join(fixed_lines)
    
    return result

def _safe_generate_for_mbpp(
    pipe: Any,
    prompts: List[str],
    tokenizer: Any,
    max_new_tokens: int,
    num_retries: int = 2,
    generation_config_override: Optional[Dict] = None
) -> List[List[Dict[str, str]]]:
    end_token_id = tokenizer.convert_tokens_to_ids("[END]")
    effective_eos_token_id = []
    
    if end_token_id != tokenizer.unk_token_id:
        effective_eos_token_id.append(end_token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in effective_eos_token_id:
        effective_eos_token_id.append(tokenizer.eos_token_id)
    
    if not effective_eos_token_id and tokenizer.eos_token_id:
        effective_eos_token_id = [tokenizer.eos_token_id]
    
    base_gen_config = {
        "do_sample": True, 
        "temperature": 0.1, 
        "top_p": 0.95,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": effective_eos_token_id if effective_eos_token_id else None,
        "return_full_text": True,
        "repetition_penalty": 1.03,
    }
    
    current_gen_config = base_gen_config.copy()
    if generation_config_override:
        current_gen_config.update(generation_config_override)

    for attempt in range(num_retries):
        try:
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
            
            outputs_raw = pipe(prompts, **current_gen_config)
            
            processed_outputs = []
            if not isinstance(outputs_raw, list):
                outputs_raw = [{"generated_text": "#ERROR: Unexpected pipe output type"}] * len(prompts)

            for i, item_output in enumerate(outputs_raw):
                if isinstance(item_output, list) and item_output and isinstance(item_output[0], dict):
                    processed_outputs.append([item_output[0]])
                elif isinstance(item_output, dict):
                    processed_outputs.append([item_output])
                else:
                    processed_outputs.append([{"generated_text": f"#GenFail prompt {i}: Bad item structure"}])
            
            return processed_outputs
            
        except Exception as e:
            if attempt < num_retries - 1:
                if torch.cuda.is_available(): 
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                return [[{"generated_text": f"#GenFail max_retries: {str(e)[:100]}"}] for _ in prompts]
    
    return [[{"generated_text": "#GenFail: AllRetriesExhausted"}] for _ in prompts]

def save_detailed_mbpp_results(
    results_data: List[Dict], 
    model_name: str, 
    dataset_name: str, 
    num_few_shot: int, 
    pass_at_1: float, 
    results_dir: str, 
    process_id: int = 0
) -> str:
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mbpp_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_few_shot": num_few_shot,
        "total_problems": len(results_data),
        "passed_problems": sum(1 for r in results_data if r.get("passed", False)),
        "pass_at_1": pass_at_1,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id,
        "generation_failures": sum(1 for r in results_data if not r.get("generation_successful", True))
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    
    return filepath

def evaluate_mbpp_plus(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name_primary: str = DEFAULT_DATASET_NAME_PRIMARY,
    dataset_config_primary: str = DEFAULT_DATASET_CONFIG_PRIMARY,
    dataset_name_fallback: str = DEFAULT_DATASET_NAME_FALLBACK,
    dataset_config_fallback: str = DEFAULT_DATASET_CONFIG_FALLBACK,
    dataset_split: str = DEFAULT_SPLIT,
    num_samples_per_task: int = DEFAULT_NUM_SAMPLES_PER_TASK,
    k_values: List[int] = DEFAULT_K_VALUES,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_MBPP,
    include_tests_in_prompt: bool = True,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    max_new_tokens_completion: int = DEFAULT_MAX_NEW_TOKENS_COMPLETION,
    results_dir: str = "results_output",
    **kwargs
) -> Dict[str, float]:

    if mbpp_pass_at_k_metric is None:
        return {"MBPP": 0.0, "error_message": "CodeEvalMetricLoadFailed"}

    mbpp_dataset = None
    try:
        mbpp_dataset = load_dataset(dataset_name_primary, dataset_config_primary, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        try:
            mbpp_dataset = load_dataset(dataset_name_fallback, dataset_config_fallback, split=dataset_split, trust_remote_code=True)
        except Exception as e2:
            return {"MBPP": 0.0, "error_message": f"DatasetLoadFailed MBPP: {e2}"}
    
    if not mbpp_dataset or len(mbpp_dataset) == 0:
        return {"MBPP": 0.0, "error_message": "NoDatasetSamples MBPP"}

    generation_inputs_info, problem_references, detailed_results_log = [], {}, []
    
    for problem_idx, problem_data in enumerate(tqdm(mbpp_dataset, desc="Preparing MBPP Prompts")):
        task_id = str(problem_data.get('task_id', f"unknown_task_{problem_idx}"))
        desc = problem_data.get("text", problem_data.get("prompt", ""))
        setup = problem_data.get('test_setup_code', "")
        tests = problem_data.get('test_list', [])
        gt_code = problem_data.get("code", "#NO_GT_CODE")
        
        if not desc or not tests: 
            continue

        llm_prompt = _format_mbpp_prompt_with_template(
            problem_data, 
            num_few_shot=num_few_shot, 
            include_tests_in_prompt=include_tests_in_prompt
        )
        
        for _ in range(num_samples_per_task):
            generation_inputs_info.append({
                'llm_prompt': llm_prompt, 
                'task_id': task_id, 
                'desc': desc, 
                'setup': setup, 
                'gt_code': gt_code
            })
        
        problem_references[task_id] = (setup + "\n\n" if setup else "") + "\n".join(tests)

    if not generation_inputs_info: 
        return {"MBPP": 0.0, "error_message": "NoValidPrompts MBPP"}

    predictions_by_task_id = defaultdict(list)
    gen_config_override = {"temperature": 0.1, "top_p": 0.95}

    for i in tqdm(range(0, len(generation_inputs_info), generation_batch_size), desc="MBPP Completions", unit="batch"):
        batch_infos = generation_inputs_info[i : i + generation_batch_size]
        batch_prompts = [info['llm_prompt'] for info in batch_infos]
        
        try:
            batch_outputs = _safe_generate_for_mbpp(
                pipe, batch_prompts, tokenizer, max_new_tokens_completion, 
                generation_config_override=gen_config_override
            )
            
            for idx_in_batch, info in enumerate(batch_infos):
                task_id = info['task_id']
                llm_prompt = info['llm_prompt']
                desc = info['desc']
                setup = info['setup']
                gt_code = info['gt_code']
                
                raw_gen = batch_outputs[idx_in_batch][0]['generated_text'] if batch_outputs[idx_in_batch] and batch_outputs[idx_in_batch][0] else "#ERROR_NO_GEN"
                extracted = _extract_mbpp_completion(raw_gen, llm_prompt)
                code_for_eval = (setup + "\n\n" if setup else "") + extracted
                
                predictions_by_task_id[task_id].append(code_for_eval)
                
                if len(predictions_by_task_id[task_id]) == 1:
                    detailed_results_log.append(MBPPEvalResultDetail(
                        task_id, desc, llm_prompt, raw_gen, extracted, code_for_eval, 
                        problem_references.get(task_id, ""), gt_code
                    ))
                    
        except Exception as e_gen_batch:
            for info in batch_infos:
                predictions_by_task_id[info['task_id']].append(f"#BATCH_GEN_ERROR: {e_gen_batch}")

    preds_list, refs_list, sorted_tids = [], [], sorted(problem_references.keys(), key=lambda x: int(x))
    
    for tid in sorted_tids:
        if tid in predictions_by_task_id:
            preds_list.append(predictions_by_task_id[tid])
            refs_list.append(problem_references[tid])
    
    if not preds_list: 
        return {"MBPP": 0.0, "error_message": "NoPredictionsForCodeEval MBPP"}

    final_scores = {}
    try:
        eval_output = mbpp_pass_at_k_metric.compute(references=refs_list, predictions=preds_list, k=k_values)
        scores_dict, detailed_results = (eval_output[0], eval_output[1]) if isinstance(eval_output, tuple) else (eval_output, None)
        
        if scores_dict:
            for k in k_values:
                key = f"pass@{k}"
                score = scores_dict.get(key, 0.0) * 100
                if k == k_values[0]: 
                    final_scores["MBPP"] = score
                final_scores[f"MBPP_{key}"] = score
        else: 
            final_scores["MBPP"] = 0.0

        if detailed_results and isinstance(detailed_results, dict):
            for task_idx_from_code_eval, per_task_sample_results_list in detailed_results.items():
                if not isinstance(task_idx_from_code_eval, int):
                    continue

                if task_idx_from_code_eval < len(sorted_tids):
                    original_task_id_str = sorted_tids[task_idx_from_code_eval]
                    log_entry_to_update = next((entry for entry in detailed_results_log if entry.task_id == original_task_id_str), None)

                    if log_entry_to_update:
                        if isinstance(per_task_sample_results_list, list) and per_task_sample_results_list:
                            first_sample_eval_result_tuple = per_task_sample_results_list[0]
                            
                            if isinstance(first_sample_eval_result_tuple, tuple) and len(first_sample_eval_result_tuple) == 2:
                                result_data_dict = first_sample_eval_result_tuple[1]
                                if isinstance(result_data_dict, dict):
                                    log_entry_to_update.passed = result_data_dict.get('passed', False)
                                    log_entry_to_update.error_message = result_data_dict.get('result', '') if not log_entry_to_update.passed else ""
                            
    except Exception as e_eval:
        final_scores["MBPP"] = 0.0
        final_scores["error_message"] = "CodeEvalComputeError"
    
    if "MBPP" not in final_scores: 
        final_scores["MBPP"] = 0.0

    # Save detailed results
    detailed_results_dicts = [asdict(entry) for entry in detailed_results_log]
    save_detailed_mbpp_results(
        detailed_results_dicts, 
        model_name_for_logging, 
        dataset_name_primary, 
        num_few_shot, 
        final_scores.get("MBPP", 0.0), 
        results_dir
    )

    return final_scores