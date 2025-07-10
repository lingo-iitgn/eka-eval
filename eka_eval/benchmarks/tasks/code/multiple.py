# eka_eval/benchmarks/tasks/code/multiple.py - MultiPL-E evaluation system

import torch
import re
import subprocess
import tempfile
import shutil
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import evaluate as hf_evaluate
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
import concurrent.futures
from pathlib import Path

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)

@dataclass
class MultiPLEResultDetail:
    name: str
    language: str
    problem_prompt: str
    full_llm_prompt: str
    entry_point: Optional[str]
    raw_generation: str
    extracted_completion: str
    full_code_for_eval: str
    reference_test_script: str
    stop_tokens: List[str]
    passed: Optional[bool] = None
    error_message: str = ""
    compilation_successful: Optional[bool] = None

# Default configuration
DEFAULT_DATASET_BASE = "nuprl/MultiPL-E"
DEFAULT_MAX_NEW_TOKENS_MULTIPLE = 512
DEFAULT_GENERATION_BATCH_SIZE_MULTIPLE = 1
DEFAULT_NUM_FEWSHOT_MULTIPLE = 0
DEFAULT_NUM_SAMPLES_PER_TASK = 1
DEFAULT_K_VALUES = [1, 5, 10]
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "multiple_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "multiple_3shot"
PROMPT_FILE_BENCHMARK_KEY = "multiple"
PROMPT_FILE_CATEGORY = "code"

# Language configurations
LANGUAGE_CONFIGS = {
    "cpp": {
        "file_ext": "cpp",
        "compile_cmd": ["g++", "-std=c++17", "-o", "{output}", "{source}"],
        "run_cmd": ["./{output}"],
        "stop_tokens": ["#include", "using namespace", "int main", "return 0;"],
        "compilation_required": True,
        "entry_point_pattern": r"(\w+)\s*\(",
    },
    "java": {
        "file_ext": "java", 
        "compile_cmd": ["javac", "{source}"],
        "run_cmd": ["java", "{classname}"],
        "stop_tokens": ["class ", "public static void main", "import ", "package "],
        "compilation_required": True,
        "entry_point_pattern": r"(\w+)\s*\(",
    },
    "js": {
        "file_ext": "js",
        "compile_cmd": None,
        "run_cmd": ["node", "{source}"],
        "stop_tokens": ["const ", "function ", "module.exports", "require("],
        "compilation_required": False,
        "entry_point_pattern": r"(?:const|function|let|var)\s+(\w+)",
    },
    "ts": {
        "file_ext": "ts",
        "compile_cmd": ["tsc", "{source}"],
        "run_cmd": ["node", "{output}.js"],
        "stop_tokens": ["interface ", "type ", "export ", "import "],
        "compilation_required": True,
        "entry_point_pattern": r"function\s+(\w+)",
    },
    "go": {
        "file_ext": "go",
        "compile_cmd": ["go", "build", "-o", "{output}", "{source}"],
        "run_cmd": ["./{output}"],
        "stop_tokens": ["package ", "func main", "import "],
        "compilation_required": True,
        "entry_point_pattern": r"func\s+(\w+)",
    },
    "cs": {
        "file_ext": "cs",
        "compile_cmd": ["csc", "/out:{output}.exe", "{source}"],
        "run_cmd": ["./{output}.exe"],
        "stop_tokens": ["using ", "namespace ", "class ", "static void Main"],
        "compilation_required": True,
        "entry_point_pattern": r"(\w+)\s*\(",
    },
    "php": {
        "file_ext": "php",
        "compile_cmd": None,
        "run_cmd": ["php", "{source}"],
        "stop_tokens": ["<?php", "class ", "function ", "require "],
        "compilation_required": False,
        "entry_point_pattern": r"function\s+(\w+)",
    },
    "rs": {
        "file_ext": "rs",
        "compile_cmd": ["rustc", "-o", "{output}", "{source}"],
        "run_cmd": ["./{output}"],
        "stop_tokens": ["fn main", "use ", "mod ", "extern crate"],
        "compilation_required": True,
        "entry_point_pattern": r"fn\s+(\w+)",
    }
}

# Global Setup for code evaluation
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

try:
    code_eval_metric = hf_evaluate.load("code_eval")
    logger.info("code_eval metric for MultiPL-E loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'code_eval' metric for MultiPL-E: {e}. MultiPL-E will not run correctly.", exc_info=True)
    code_eval_metric = None

def save_detailed_multiple_results(
    results_data: List[Dict],
    model_name: str,
    language: str,
    dataset_name: str,
    num_few_shot: int,
    pass_at_1: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed MultiPL-E results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multiple_{language}_{model_clean}_{dataset_clean}_{num_few_shot}shot_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "language": language,
        "dataset_name": dataset_name,
        "num_few_shot": num_few_shot,
        "total_problems": len(results_data),
        "passed_problems": sum(1 for r in results_data if r.get("passed", False)),
        "pass_at_1": pass_at_1,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id,
        "generation_failures": sum(1 for r in results_data if not r.get("generation_successful", True)),
        "compilation_failures": sum(1 for r in results_data if r.get("compilation_successful") == False)
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed MultiPL-E results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed MultiPL-E results: {e}")
        return ""

def _get_multiple_fewshot_examples_from_config(num_few_shot: int, language: str, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration for specific language"""
    if num_few_shot <= 0:
        return []
    
    # Try language-specific examples first
    examples_key = f"default_few_shot_examples_{language}"
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key=examples_key,
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples for {language} from JSON.")
        return loaded_examples_list[:num_few_shot]
    
    # Fallback to generic examples if language-specific not found
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_cpp",  # Use C++ as fallback
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.warning(f"Using fallback examples for {language}. Consider adding language-specific examples.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load few-shot examples for {language} from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def _extract_entry_point(prompt: str, language: str) -> Optional[str]:
    """Extract the function/method name to be tested from the prompt."""
    lang_config = LANGUAGE_CONFIGS.get(language, {})
    pattern = lang_config.get("entry_point_pattern", r"(\w+)\s*\(")
    
    match = re.search(pattern, prompt)
    if match:
        return match.group(1)
    
    # Fallback patterns for different languages
    fallback_patterns = [
        r"def\s+(\w+)\s*\(",  # Python style
        r"function\s+(\w+)\s*\(",  # JavaScript/TypeScript
        r"fn\s+(\w+)\s*\(",  # Rust
        r"func\s+(\w+)\s*\(",  # Go
        r"(\w+)\s*\([^)]*\)\s*{",  # General function pattern
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, prompt)
        if match:
            return match.group(1)
    
    return None
def _extract_multiple_completion(
    generated_text: str, 
    prompt_text_sent_to_llm: Optional[str] = None,
    language: str = "cpp",
    stop_tokens: List[str] = None
) -> str:
    """Extract the code completion from generated text for MultiPL-E."""
    completion_part = generated_text
    
    # Remove the original prompt from the generation
    if prompt_text_sent_to_llm and generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    
    completion_part = completion_part.strip()
    
    logger.debug(f"MultiPL-E {language} extraction input: '{completion_part[:200]}'")
    
    if len(completion_part) == 0:
        logger.debug(f"MultiPL-E {language}: Empty completion, extraction failed")
        return ""
    
    # Remove markdown code blocks
    if completion_part.startswith(f"```{language}"):
        completion_part = completion_part[len(f"```{language}"):].lstrip()
    elif completion_part.startswith("```"):
        completion_part = completion_part[len("```"):].lstrip()
    
    if completion_part.endswith("```"):
        completion_part = completion_part[:-len("```")].rstrip()
    
    # More selective stop tokens - avoid cutting off good code
    critical_stops = [
        "</s>", "<|EOT|>", "<|end|>", "<eos>", 
        "\n\nint main(", "\n\nvoid main(", "\n\npublic static void main",
        "\n\nif __name__", "\n\ndef test_", "\n\nassert ",
        "METADATA", "\n\n#include", "\n\nclass Test"
    ]
    
    # Find the earliest critical stop token
    min_stop_index = len(completion_part)
    for seq in critical_stops:
        found_idx = completion_part.find(seq)
        if found_idx != -1:
            min_stop_index = min(min_stop_index, found_idx)
    
    completion_part = completion_part[:min_stop_index].rstrip()
    
    # Clean up the completion
    lines = completion_part.split('\n')
    cleaned_lines = []
    brace_count = 0
    
    for line in lines:
        # Skip obvious test/main function lines
        if any(skip in line.lower() for skip in [
            "int main(", "void main(", "public static void main", 
            "if __name__", "def test_", "assert(", "#include"
        ]):
            break
            
        # Track braces for proper function ending
        brace_count += line.count('{') - line.count('}')
        cleaned_lines.append(line)
        
        # If we've closed all braces and have a return or closing brace, stop
        if brace_count <= 0 and line.strip().endswith(('}', ';')):
            break
    
    completion_part = '\n'.join(cleaned_lines)
    
    # Language-specific post-processing
    if language in ["cpp", "java", "cs"]:
        # Ensure proper indentation
        if completion_part and not completion_part.startswith('    '):
            lines = completion_part.split('\n')
            indented_lines = []
            for line in lines:
                if line.strip():
                    if not line.startswith('    '):
                        indented_lines.append('    ' + line.lstrip())
                    else:
                        indented_lines.append(line)
                else:
                    indented_lines.append(line)
            completion_part = '\n'.join(indented_lines)
    
    logger.debug(f"MultiPL-E {language}: Extracted completion: '{completion_part[:100]}...'")
    return completion_part

def evaluate_multiple(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    language: str = "cpp",
    dataset_name: str = None,
    dataset_split: str = "test",
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_MULTIPLE,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MULTIPLE,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_MULTIPLE,
    num_samples_per_task: int = DEFAULT_NUM_SAMPLES_PER_TASK,
    k_values: List[int] = None,
    few_shot_examples_key: Optional[str] = None,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    use_compilation_check: bool = False,  # Disabled by default for simplicity
    **kwargs
) -> Dict[str, float]:

    if code_eval_metric is None:
        return {"MultiPL-E": 0.0, "error_message": "CodeEvalMetricLoadFailed"}

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # Construct dataset name if not provided
    if dataset_name is None:
        dataset_name = f"humaneval-{language}"
    
    logger.info(f"Starting MultiPL-E ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")
    logger.info(f"Language: {language}")
    logger.info(f"Generation config: max_new_tokens={max_new_tokens}, batch_size={generation_batch_size}")
    logger.info(f"Evaluation config: samples_per_task={num_samples_per_task}, k_values={k_values}")

    # Load prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"MultiPL-E": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_multiple_fewshot_examples_from_config(num_few_shot, language, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("MultiPL-E: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"MultiPL-E": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    # Load dataset
    try:
        # MultiPL-E dataset loading - the dataset_name contains the configuration like "humaneval-cpp"
        full_data = load_dataset(DEFAULT_DATASET_BASE, dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return {"MultiPL-E": 0.0, "error_message": f"DatasetLoadFailed: {e}"}
    
    logger.info(f"P{process_id}: Loaded MultiPL-E {language} ({len(full_data)} problems).")

    # Handle multi-GPU data splitting
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
        return {"MultiPL-E": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} MultiPL-E {language} problems.")

    # Initialize tracking
    predictions_by_name = defaultdict(list)
    problem_references = {}
    prompts_for_batch, original_items_for_batch_info = [], []
    detailed_results = []

    # Main evaluation loop
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - MultiPL-E {language} Eval")):
        name = item_data.get("name")
        problem_prompt = item_data.get("prompt")
        test_script = item_data.get("tests")
        stop_tokens = item_data.get("stop_tokens", [])
        
        if not all([name, problem_prompt, test_script]):
            logger.warning(f"Missing data for problem {name}")
            continue

        # Extract entry point
        entry_point = _extract_entry_point(problem_prompt, language)
        
        # Store reference test for this problem
        problem_references[name] = test_script

        # Generate multiple samples for this problem
        for sample_idx in range(num_samples_per_task):
            # Format prompt for this problem
            main_problem_data = {
                "problem_prompt": problem_prompt,
                "language": language.upper()
            }
            
            # Generate prompt
            if num_few_shot > 0 and few_shot_examples_to_use:
                llm_prompt = format_few_shot_prompt(prompt_template_dict, few_shot_examples_to_use, main_problem_data)
            else:
                llm_prompt = format_prompt(prompt_template_dict, **main_problem_data)
            
            prompts_for_batch.append(llm_prompt)
            original_items_for_batch_info.append({
                'name': name,
                'sample_idx': sample_idx,
                'problem_prompt': problem_prompt,
                'llm_prompt': llm_prompt,
                'test_script': test_script,
                'entry_point': entry_point,
                'stop_tokens': stop_tokens,
                'item_idx': item_idx,
                'language': language
            })

            # Process batch
            if len(prompts_for_batch) == generation_batch_size or (item_idx == len(subset_to_process) - 1 and sample_idx == num_samples_per_task - 1):
                # Generation config optimized for code - FIXED: removed stop_strings
                gen_config = {
                    "do_sample": True,
                    "temperature": 0.1,   # Lower temperature for more focused code
                    "top_p": 0.9,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "return_full_text": False,
                    "repetition_penalty": 1.1,  # Prevent repetitive text
                    # Note: stop_strings removed to avoid tokenizer issues with pipelines
                    # Stop sequences will be handled in post-processing instead
                }
                
                # Log generation config for first batch
                if item_idx == 0 and sample_idx == 0:
                    logger.info(f"Generation config for {language}: {gen_config}")
                
                try:
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                    
                    for k, raw_out_list in enumerate(batch_raw_outputs):
                        orig_info = original_items_for_batch_info[k]
                        
                        # Extract generated text
                        generated_full_text = ""
                        generated_part = ""
                        
                        if raw_out_list and len(raw_out_list) > 0 and 'generated_text' in raw_out_list[0]:
                            generated_full_text = raw_out_list[0]['generated_text']
                            
                            if gen_config.get("return_full_text", True):
                                if generated_full_text.startswith(orig_info['llm_prompt']):
                                    generated_part = generated_full_text[len(orig_info['llm_prompt']):]
                                else:
                                    generated_part = generated_full_text
                            else:
                                generated_part = generated_full_text
                        else:
                            generated_full_text = "[NO_GENERATION]"
                            generated_part = "[NO_GENERATION]"
                        
                        # Extract code completion
                        extracted_completion = _extract_multiple_completion(
                            generated_part, 
                            orig_info['llm_prompt'],
                            language,
                            orig_info['stop_tokens']
                        )
                        
                        # Ensure we have some kind of completion
                        if not extracted_completion.strip():
                            extracted_completion = _extract_multiple_completion(generated_full_text, orig_info['llm_prompt'], language)
                        
                        # Language-specific fallback
                        if not extracted_completion.strip():
                            if language in ["cpp", "java", "cs"]:
                                extracted_completion = "    // TODO: Implement function\n    return {};"
                            elif language in ["js", "ts"]:
                                extracted_completion = "    // TODO: Implement function\n    return null;"
                            elif language == "go":
                                extracted_completion = "    // TODO: Implement function\n    return nil"
                            elif language == "rs":
                                extracted_completion = "    // TODO: Implement function\n    todo!()"
                            elif language == "php":
                                extracted_completion = "    // TODO: Implement function\n    return null;"
                            else:
                                extracted_completion = "    // TODO: Implement function"
                        
                        full_code_for_eval = orig_info['problem_prompt'] + extracted_completion
                        
                        # Store prediction for this problem
                        predictions_by_name[orig_info['name']].append(full_code_for_eval)
                        
                        generation_successful = extracted_completion != "[NO_GENERATION]" and len(extracted_completion.strip()) > 0
                        
                        # Enhanced debug output for first few examples
                        if orig_info['item_idx'] < 3 and orig_info['sample_idx'] == 0:
                            print(f"\n{'='*80}")
                            print(f"🔍 DEBUG MultiPL-E {language.upper()} Problem {orig_info['item_idx']}")
                            print(f"{'='*80}")
                            print(f"📝 Name: {orig_info['name']}")
                            print(f"🎯 Entry Point: {orig_info['entry_point']}")
                            print(f"🔧 Generation Success: {'✅' if generation_successful else '❌'}")
                            
                            print(f"\n📋 PROBLEM PROMPT ({language}):")
                            print("-" * 50)
                            print(f"{orig_info['problem_prompt'][:500]}...")
                            
                            print(f"\n✂️ EXTRACTED COMPLETION:")
                            print("-" * 50)
                            print(f"'{extracted_completion}'")
                            
                            print(f"\n🔗 FULL CODE FOR EVALUATION:")
                            print("-" * 50)
                            print(f"{full_code_for_eval[:500]}...")
                            
                            print(f"\n{'='*80}")
                            print(f"END DEBUG Problem {orig_info['item_idx']}")
                            print(f"{'='*80}\n")
                        
                        # Store detailed result
                        if save_detailed:
                            detailed_result = {
                                "name": orig_info['name'],
                                "language": language,
                                "sample_index": orig_info['sample_idx'],
                                "problem_prompt": orig_info['problem_prompt'],
                                "entry_point": orig_info['entry_point'],
                                "full_llm_prompt": orig_info['llm_prompt'],
                                "raw_generation": generated_full_text,
                                "extracted_completion": extracted_completion,
                                "full_code_for_eval": full_code_for_eval,
                                "test_script": orig_info['test_script'],
                                "stop_tokens": orig_info['stop_tokens'],
                                "generation_successful": generation_successful,
                                "compilation_successful": None,
                                "passed": None,
                                "error_message": ""
                            }
                            detailed_results.append(detailed_result)
                            
                except Exception as e_batch:
                    logger.error(f"P{process_id}: MultiPL-E {language} generation batch error: {e_batch}", exc_info=True)
                    # Add fallback predictions for failed batch
                    for info in original_items_for_batch_info:
                        error_code = info['problem_prompt'] + f"\n    // GENERATION ERROR: {e_batch}"
                        predictions_by_name[info['name']].append(error_code)
                        
                        if save_detailed:
                            detailed_result = {
                                "name": info['name'],
                                "language": language,
                                "sample_index": info['sample_idx'],
                                "problem_prompt": info['problem_prompt'],
                                "entry_point": info['entry_point'],
                                "full_llm_prompt": info['llm_prompt'],
                                "raw_generation": "[GENERATION_ERROR]",
                                "extracted_completion": f"// GENERATION ERROR: {e_batch}",
                                "full_code_for_eval": error_code,
                                "test_script": info['test_script'],
                                "stop_tokens": info['stop_tokens'],
                                "generation_successful": False,
                                "compilation_successful": False,
                                "passed": False,
                                "error_message": str(e_batch)
                            }
                            detailed_results.append(detailed_result)
                
                # Reset batch
                prompts_for_batch, original_items_for_batch_info = [], []

    # Prepare for code evaluation
    final_predictions = []
    final_references = []
    
    # Sort names to ensure consistent ordering
    sorted_names = sorted(problem_references.keys())
    
    for name in sorted_names:
        if name in predictions_by_name and problem_references[name]:
            final_predictions.append(predictions_by_name[name])
            final_references.append(problem_references[name])

    if not final_predictions or not final_references:
        logger.error("No valid predictions or references for evaluation")
        return {"MultiPL-E": 0.0, "error_message": "NoSamplesForCodeEval"}

    logger.info(f"P{process_id}: Running code evaluation for {len(final_references)} {language} problems with {sum(len(preds) for preds in final_predictions)} total samples")
    
    # Run code evaluation
    final_scores = {}
    
    try:
        eval_output = code_eval_metric.compute(
            references=final_references,
            predictions=final_predictions,
            k=k_values
        )
        
        # Handle different output formats
        if isinstance(eval_output, tuple):
            scores = eval_output[0]
            detailed_eval_results = eval_output[1] if len(eval_output) > 1 else None
        else:
            scores = eval_output
            detailed_eval_results = None
        
        if scores:
            logger.info(f"P{process_id} - MultiPL-E {language} Pass@k scores: {scores}")
            for k_val in k_values:
                metric_key = f"pass@{k_val}"
                score_value = scores.get(metric_key, 0.0) * 100  # Convert to percentage
                
                if k_val == k_values[0]:  # Primary metric
                    final_scores["MultiPL-E"] = score_value
                final_scores[f"MultiPL-E_{language}_pass@{k_val}"] = score_value
        else:
            logger.error("code_eval did not return valid scores")
            final_scores["MultiPL-E"] = 0.0

        # Update detailed results with pass/fail status
        if detailed_eval_results and isinstance(detailed_eval_results, list) and save_detailed:
            for problem_idx, problem_results in enumerate(detailed_eval_results):
                if problem_idx < len(sorted_names):
                    name = sorted_names[problem_idx]
                    
                    if problem_results and isinstance(problem_results, list):
                        for sample_idx, sample_result in enumerate(problem_results):
                            # Find corresponding detailed result
                            matching_result = next(
                                (dr for dr in detailed_results 
                                 if dr["name"] == name and dr["sample_index"] == sample_idx), 
                                None
                            )
                            
                            if matching_result and sample_result:
                                if isinstance(sample_result, tuple) and len(sample_result) == 2:
                                    result_dict = sample_result[1]
                                    if isinstance(result_dict, dict):
                                        matching_result["passed"] = result_dict.get('passed', False)
                                        if not matching_result["passed"]:
                                            matching_result["error_message"] = result_dict.get('result', '')

    except Exception as e:
        logger.error(f"P{process_id}: Error during code evaluation: {e}", exc_info=True)
        final_scores["MultiPL-E"] = 0.0
        final_scores["error_message"] = "CodeEvalComputationError"

    # Calculate summary stats
    if final_scores.get("MultiPL-E") is not None:
        pass_at_1 = final_scores["MultiPL-E"]
    else:
        pass_at_1 = 0.0

    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_multiple_results(
            detailed_results,
            model_name_for_logging,
            language,
            dataset_name,
            num_few_shot,
            pass_at_1,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed MultiPL-E {language} results with {len(detailed_results)} samples saved to: {saved_path}")

    logger.info(f"P{process_id} - Final MultiPL-E {language} Pass@1: {pass_at_1:.2f}% on {len(final_references)} problems.")
    
    if "MultiPL-E" not in final_scores:
        final_scores["MultiPL-E"] = 0.0

    return final_scores


def evaluate_multiple_all_languages(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    languages: List[str] = ["cpp", "java", "js", "ts", "go", "cs", "php", "rs"],
    source_benchmark: str = "humaneval",  # "humaneval" or "mbpp"
    dataset_split: str = "test",
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_MULTIPLE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MULTIPLE,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_MULTIPLE,
    num_samples_per_task: int = DEFAULT_NUM_SAMPLES_PER_TASK,
    k_values: List[int] = None,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    use_compilation_check: bool = False,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate model on MultiPL-E across multiple programming languages.
    
    Args:
        languages: List of programming languages to evaluate
        source_benchmark: Base benchmark ("humaneval" or "mbpp")
        Other args: Same as evaluate_multiple
    
    Returns:
        Dictionary with aggregated results across all languages
    """
    
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    
    logger.info(f"Starting MultiPL-E evaluation across {len(languages)} languages: {languages}")
    logger.info(f"Source benchmark: {source_benchmark}")
    
    all_results = {}
    language_scores = {}
    
    for language in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {language.upper()}")
        logger.info(f"{'='*60}")
        
        dataset_name = f"{source_benchmark}-{language}"
        
        try:
            lang_results = evaluate_multiple(
                pipe=pipe,
                tokenizer=tokenizer,
                model_name_for_logging=model_name_for_logging,
                device=device,
                language=language,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                num_few_shot=num_few_shot,
                max_new_tokens=max_new_tokens,
                generation_batch_size=generation_batch_size,
                num_samples_per_task=num_samples_per_task,
                k_values=k_values,
                process_id=process_id,
                gpu_id=gpu_id,
                num_gpus=num_gpus,
                results_dir=results_dir,
                save_detailed=save_detailed,
                use_compilation_check=use_compilation_check,
                **kwargs
            )
            
            # Store language-specific results
            for key, value in lang_results.items():
                if key != "error_message":
                    all_results[f"{key}_{language}"] = value
            
            # Track primary score for each language
            primary_score = lang_results.get("MultiPL-E", 0.0)
            language_scores[language] = primary_score
            
            logger.info(f"✅ {language.upper()} completed: {primary_score:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {language}: {e}", exc_info=True)
            language_scores[language] = 0.0
            all_results[f"MultiPL-E_{language}"] = 0.0
            all_results[f"MultiPL-E_{language}_error"] = str(e)
    
    # Calculate aggregate metrics
    valid_scores = [score for score in language_scores.values() if score > 0]
    
    if valid_scores:
        all_results["MultiPL-E_average"] = sum(valid_scores) / len(valid_scores)
        all_results["MultiPL-E_total"] = sum(language_scores.values()) / len(language_scores)
        all_results["MultiPL-E_best"] = max(language_scores.values())
        all_results["MultiPL-E_worst"] = min(language_scores.values())
        all_results["MultiPL-E_languages_passed"] = len(valid_scores)
    else:
        all_results["MultiPL-E_average"] = 0.0
        all_results["MultiPL-E_total"] = 0.0
        all_results["MultiPL-E_best"] = 0.0
        all_results["MultiPL-E_worst"] = 0.0
        all_results["MultiPL-E_languages_passed"] = 0
    
    # Set primary metric to average
    all_results["MultiPL-E"] = all_results["MultiPL-E_average"]
    
    # Log summary
    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 MULTI-LANGUAGE EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Languages evaluated: {len(languages)}")
    logger.info(f"✅ Languages passed: {all_results['MultiPL-E_languages_passed']}")
    logger.info(f"📈 Average score: {all_results['MultiPL-E_average']:.2f}%")
    logger.info(f"🥇 Best score: {all_results['MultiPL-E_best']:.2f}%")
    logger.info(f"📉 Worst score: {all_results['MultiPL-E_worst']:.2f}%")
    
    logger.info(f"\n📋 Per-language breakdown:")
    for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if score > 0 else "❌"
        logger.info(f"  {status} {lang.upper()}: {score:.2f}%")
    
    logger.info(f"{'='*80}")
    
    return all_results


if __name__ == '__main__':
    # Test functionality
    if __package__ is None or __package__ == '':
        current_script_path = os.path.abspath(__file__)
        project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
        if project_root_for_test not in sys.path:
            sys.path.insert(0, project_root_for_test)
        print(f"DEBUG (multiple.py __main__): Added to sys.path for standalone test: {project_root_for_test}")
    
    # Set CUDA_VISIBLE_DEVICES to use available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    print(f"DEBUG: Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    setup_logging(level=logging.DEBUG, worker_id="MultiPLEFileTest")
    
    TEST_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    TEST_LANGUAGE = "cpp"
    print(f"--- Standalone MultiPL-E Test: {TEST_MODEL_NAME} on {TEST_LANGUAGE} ---")
    
    test_task_args = {
        "language": TEST_LANGUAGE,
        "dataset_name": f"humaneval-{TEST_LANGUAGE}",
        "dataset_split": "test[:5]",  # Small test
        "num_few_shot": 0,
        "prompt_template_name_zeroshot": "multiple_0shot",
        "prompt_template_name_fewshot": "multiple_3shot",
        "prompt_file_benchmark_key": "multiple",
        "prompt_file_category": "code",
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS_MULTIPLE,
        "generation_batch_size": 1,
        "num_samples_per_task": 1,
        "k_values": [1],
        "use_compilation_check": False,  
    }
    
    test_general_args = {
        "model_name_for_logging": TEST_MODEL_NAME,
        "process_id": 0,
        "gpu_id": 0,
        "num_gpus": 1,
        "results_dir": "results_output_test"
    }

    test_pipe, *_ = initialize_model_pipeline(TEST_MODEL_NAME, target_device_id=0)
    
    if test_pipe:
        eval_args_for_test = {
            "pipe": test_pipe,
            "tokenizer": test_pipe.tokenizer,
            "device": test_pipe.device,
            **test_general_args,
            **test_task_args
        }
        
        try:
            print(f"\n🧪 Testing single language evaluation ({TEST_LANGUAGE})...")
            results = evaluate_multiple(**eval_args_for_test)
            print("🎉 Single language results:", json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cleanup_model_resources(test_pipe, getattr(test_pipe, 'model', None))
    else:
        logger.error(f"Failed to initialize model {TEST_MODEL_NAME} for MultiPL-E standalone test.")
    
    print("🏁 MultiPL-E test completed!")