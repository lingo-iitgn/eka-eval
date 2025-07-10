# your_main_script.py (or whatever you named it)

import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict
import evaluate as hf_evaluate
import gc
from typing import List, Dict, Union
import argparse 


BENCHMARK_CONFIG = {
    "CODE GENERATION": {
        "HumanEval": {"description": "Average pass@1 scores on HumanEval.", "evaluation_function": "evaluate_humaneval"},
        "MBPP": {"description": "Average pass@1 scores on MBPP.", "evaluation_function": "evaluate_mbpp"}
    },
    "COMMONSENSE REASONING": {
        "PIQA": {"description": "0-shot"},
        "SIQA": {"description": "0-shot"},
        "HellaSwag": {"description": "0-shot"},
        "WinoGrande": {"description": "0-shot"},
        "ARC (easy + challenge)": {"description": "0-shot"},
        "OpenBookQA": {"description": "0-shot"},
        "CommonsenseQA": {"description": "7-shot"},
    },
    "WORLD KNOWLEDGE": {
        "NaturalQuestions": {"description": "5-shot average"},
        "TriviaQA": {"description": "5-shot average"},
    },
    "READING COMPREHENSION": {
        "SQuAD": {"description": "0-shot average"},
        "QuAC": {"description": "0-shot average"},
        "BoolQ": {"description": "0-shot average"},
    },
    "MATH": {
        "GSM8K": {"description": "8-shot average top-1 accuracy", "evaluation_function": "evaluate_gsm8k"}, # ADDED THIS LINE
        "MATH": {"description": "4-shot average top-1 accuracy"},
    },
    "MMLU": {"description": "5-shot overall results", "evaluation_function": "evaluate_mmlu"},
    "BBH": {"description": "3-shot overall results", "evaluation_function": "evaluate_bbh"},
    "AGIEval": {"description": "3–5 shot overall results (English tasks averaged)", "evaluation_function": "evaluate_agieval"},
}

PRE_CALCULATED_RESULTS = pd.DataFrame()

# --- Placeholder/Mock evaluation functions for other benchmarks ---
# (Keep these if you don't have real implementations yet, so the code runs)
def evaluate_humaneval(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    print(f"\n--- Running HumanEval evaluation for {model_name} (placeholder) ---")
    return {"HumanEval": 0.0}

def evaluate_mbpp(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    print(f"\n--- Running MBPP evaluation for {model_name} (placeholder) ---")
    return {"MBPP": 0.0}

def evaluate_commonsense_reasoning(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} evaluation for {model_name} (placeholder) ---")
    return {benchmark: 60.0 + (hash(model_name + benchmark) % 10)}

def evaluate_world_knowledge(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} evaluation for {model_name} (placeholder) ---")
    return {benchmark: 75.0 + (hash(model_name + benchmark) % 10)}

def evaluate_reading_comprehension(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} evaluation for {model_name} (placeholder) ---")
    return {benchmark: 80.0 + (hash(model_name + benchmark) % 10)}

def evaluate_math(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    # This one will be replaced by evaluate_gsm8k for GSM8K, but needed for others
    print(f"\n--- Running {benchmark} evaluation for {model_name} (placeholder) ---")
    return {benchmark: 15.0 + (hash(model_name + benchmark) % 5)}

def evaluate_mmlu(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    print(f"\n--- Running MMLU evaluation for {model_name} (placeholder) ---")
    return {"MMLU": 65.0 + (hash(model_name) % 15)}

def evaluate_bbh(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    print(f"\n--- Running BBH evaluation for {model_name} (placeholder) ---")
    return {"BBH": 50.0 + (hash(model_name) % 10)}

def evaluate_agieval(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    """Placeholder for AGIEval evaluation."""
    print(f"\n--- Running AGIEval evaluation for {model_name} (placeholder) ---")
    return {"AGIEval": 40.0 + (hash(model_name) % 10)}

def safe_generate(pipe, prompts, max_retries=3):
    """Safely generate text with error handling and retries."""
    for attempt in range(max_retries):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results = []
            for i, prompt in enumerate(prompts):
                try:
                    generation_params = {
                        "do_sample": True,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_new_tokens": 512,
                        "num_return_sequences": 1,
                        "pad_token_id": pipe.tokenizer.eos_token_id,
                        "eos_token_id": pipe.tokenizer.eos_token_id,
                        "return_full_text": True,
                        "repetition_penalty": 1.1,
                        "length_penalty": 1.0,
                    }
                    # IMPORTANT: For models like Gemma/Llama, `return_full_text=False`
                    # in pipeline initialization is usually better for generation,
                    # but if your `format_prompt` expects a specific output structure
                    # with the full prompt, then `return_full_text=True` might be necessary.
                    # Your current format_prompt adds few-shot examples, so you'll want
                    # to correctly handle whether the prompt is included in the output
                    # for answer extraction. Let's assume it should return only generated.
                    # If issues, check `return_full_text` here vs. pipeline init.
                    output = pipe(prompt, **generation_params)
                    generated_text = output[0].get('generated_text', '') if output and len(output) > 0 else "# Generation failed: No output"

                    # If return_full_text was True, you'd need to strip the prompt:
                    # if generated_text.startswith(prompt):
                    #     generated_text = generated_text[len(prompt):].strip()

                    results.append([{"generated_text": generated_text}])
                except Exception as e:
                    results.append([{"generated_text": "# Generation failed"}])
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            else:
                return [[{"generated_text": "# Generation failed"}] for _ in prompts]
    return [[{"generated_text": "# Generation failed"}] for _ in prompts]


def initialize_pipeline(model_name: str, device_id: int = 0):
    """Initializes a Hugging Face model and tokenizer pipeline."""
    # Ensure device_id is 0 as CUDA_VISIBLE_DEVICES handles the actual device
    device_map_arg = {'': 0} if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For 4-bit quantization, as discussed
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map_arg, # Use the mapped device
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config if torch.cuda.is_available() else None,
            attn_implementation="eager", # Explicitly set to eager for V100 compatibility
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"Error loading model with 4-bit quantization: {e}. Trying without quantization.")
        # Fallback to no quantization if 4-bit fails (e.g., if it's not a proper model for bnb)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device_map_arg,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        except Exception as e_no_quant:
            print(f"Failed to load model {model_name} even without quantization: {e_no_quant}")
            return None, None

    model_size_gb = 0
    if model:
        try:
            # Correct calculation of model size in GB
            total_params = sum(p.numel() for p in model.parameters())
            # Use model.config.torch_dtype.itemsize if model is loaded with a specific dtype
            # Otherwise, assume average 8-bit for quantized, or bfloat16 for unquantized if loaded as such.
            # For 4-bit quantized, it's roughly 0.5 bytes per parameter (4 bits).
            # For bfloat16, it's 2 bytes per parameter.
            if hasattr(model, 'quantization_config') and model.quantization_config:
                if model.quantization_config.load_in_4bit:
                    model_size_gb = total_params * 4 / (8 * (1024**3)) # 4 bits per parameter
                elif model.quantization_config.load_in_8bit:
                    model_size_gb = total_params * 8 / (8 * (1024**3)) # 8 bits per parameter
                else: # Default to bfloat16 for non-quantized if loaded that way
                    model_size_gb = total_params * 2 / (1024**3)
            elif model.dtype == torch.bfloat16:
                model_size_gb = total_params * torch.finfo(torch.bfloat16).bits / (8 * (1024**3))
            elif model.dtype == torch.float16:
                model_size_gb = total_params * torch.finfo(torch.float16).bits / (8 * (1024**3))
            else: # Fallback to float32 if no specific dtype detected
                model_size_gb = total_params * torch.finfo(torch.float32).bits / (8 * (1024**3))

        except Exception as e:
            print(f"Warning: Could not calculate model size: {e}")
            model_size_gb = 'N/A'


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else "cpu", # Pipeline explicitly takes device ID or "cpu"
        # The torch_dtype here in pipeline is often redundant if already set in from_pretrained
        # It's better practice to ensure it's set during model loading.
        # However, keeping it doesn't hurt.
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # Important: set return_full_text=False so pipeline only returns generated new tokens.
        # This simplifies answer extraction.
        return_full_text=False
    )
    return pipe, model_size_gb

def main():
    global PRE_CALCULATED_RESULTS

    parser = argparse.ArgumentParser(description="Evaluate LLM on various benchmarks.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use (0, 1, etc.).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs being used for this run.")
    parser.add_argument("--process_id", type=int, default=0, help="Unique ID for this process (0 to num_gpus-1).")
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES for this specific process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Process {args.process_id} using GPU {args.gpu_id}")


    input_model_name = input("Enter the model name (e.g., tiiuae/falcon-7b, tiiuae/falcon-40b, google/gemma-2b, or your custom model path): ").strip()
    model_name_lower = input_model_name.lower()

    model_pipeline = None
    model_size_value = 'N/A'

    csv_file_path = 'calculated.csv'
    if os.path.exists(csv_file_path):
        try:
            temp_df = pd.read_csv(csv_file_path)
            PRE_CALCULATED_RESULTS = temp_df[temp_df['Model'].str.lower() == model_name_lower].copy()
            if not PRE_CALCULATED_RESULTS.empty:
                # Ensure 'Size (B)' is read correctly, handle cases where it's already GB
                size_str = PRE_CALCULATED_RESULTS['Size (B)'].iloc[0]
                if 'gb' in str(size_str).lower():
                    model_size_value = size_str
                else: # Assume it's in Billions
                    model_size_value = f"{float(size_str):.2f}B"
        except Exception as e:
            print(f"Warning: Could not load pre-calculated results from '{csv_file_path}': {e}")
            PRE_CALCULATED_RESULTS = pd.DataFrame()
    
    # Initialize pipeline only if no pre-calculated results or if force re-evaluate
    if PRE_CALCULATED_RESULTS.empty or \
       any(task_name not in PRE_CALCULATED_RESULTS['Task'].values for task_name in BENCHMARK_CONFIG.keys()) or \
       (input("Model found in pre-calculated results. Re-evaluate? (yes/no): ").lower() == 'yes' if not PRE_CALCULATED_RESULTS.empty else False):
        
        # Pass args.process_id to initialize_pipeline if it needs to know its unique ID
        model_pipeline, calculated_model_size_gb = initialize_pipeline(input_model_name, device_id=args.gpu_id)
        if model_pipeline is None:
            print(f"Failed to initialize model: {input_model_name}. Exiting.")
            return
        model_size_value = f"{calculated_model_size_gb:.2f} GB" if isinstance(calculated_model_size_gb, (int, float)) else 'N/A'
    else:
        print(f"Using pre-calculated results for model '{input_model_name}'.")


    print("\n--- Available Tasks ---")
    tasks = list(BENCHMARK_CONFIG.keys())
    for i, task in enumerate(tasks):
        print(f"{i+1}. {task}")
    print(f"{len(tasks)+1}. ALL (Evaluate on all possible tasks)")

    selected_task_indices = input(f"Enter the numbers of tasks you want to evaluate (e.g., '1' for Code Generation, '1 3' for Code Generation and World Knowledge, '{len(tasks)+1}' for ALL): ").strip().split()

    chosen_tasks = []
    if str(len(tasks) + 1) in selected_task_indices:
        chosen_tasks = tasks
    else:
        for idx_str in selected_task_indices:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(tasks):
                    chosen_tasks.append(tasks[idx])
                else:
                    print(f"Warning: Invalid task number '{idx_str}' ignored.")
            except ValueError:
                print(f"Warning: Invalid input '{idx_str}' ignored.")

    if not chosen_tasks:
        print("No valid tasks selected. Exiting.")
        return

    current_model_row_data = {
        ('Model', ''): input_model_name,
        ('Size (B)', ''): model_size_value
    }
    
    # Keep track of results for new/overwritten entries to save to CSV
    new_results_to_save = []

    for task_name in chosen_tasks:
        task_config = BENCHMARK_CONFIG[task_name]
        
        # Determine if it's a top-level benchmark (MMLU, BBH, AGIEval)
        is_top_level_benchmark = task_name in ["MMLU", "BBH", "AGIEval"]

        scores_for_averaging = [] # For sub-benchmark averages

        # Handle top-level benchmarks first
        if is_top_level_benchmark:
            score = pd.NA
            evaluation_function_name = task_config.get("evaluation_function")
            
            # Check pre-calculated results
            pre_calc_entry = PRE_CALCULATED_RESULTS[
                (PRE_CALCULATED_RESULTS['Model'].str.lower() == model_name_lower) &
                (PRE_CALCULATED_RESULTS['Task'].str.lower() == task_name.lower())
            ]

            if not pre_calc_entry.empty:
                score = pre_calc_entry['Score'].iloc[0]
                print(f"Using pre-calculated score for {task_name}: {score}")
            elif evaluation_function_name:
                if model_pipeline is None:
                    print(f"Model pipeline not initialized for {task_name}. Skipping evaluation.")
                else:
                    # Dynamically get the evaluation function
                    eval_func = globals().get(evaluation_function_name)
                    if eval_func:
                        try:
                            # Pass relevant args for multi-GPU distribution to eval function if needed
                            # For simplicity here, we assume GSM8K handles its own data loading for now.
                            # If you want to distribute GSM8K problems across GPUs, modify evaluate_gsm8k to accept args.
                            scores = eval_func(input_model_name, model_pipeline, calculated_model_size_gb)
                            score = scores.get(task_name, pd.NA) # For top-level, key is task_name
                            if isinstance(score, (int, float)):
                                new_results_to_save.append({
                                    'Model': input_model_name,
                                    'Size (B)': model_size_value,
                                    'Task': task_name,
                                    'Benchmark': task_name, # Benchmark is same as Task for top-level
                                    'Score': score,
                                    'Timestamp': datetime.now().isoformat()
                                })
                        except Exception as e:
                            print(f"Error evaluating {task_name}: {e}")
                    else:
                        print(f"Evaluation function '{evaluation_function_name}' not found for {task_name}.")
            
            current_model_row_data[(task_name, '')] = round(score, 3) if isinstance(score, (int, float)) else score
            current_model_row_data[(task_name, 'Average')] = round(score, 3) if isinstance(score, (int, float)) else score # For display consistency
            continue # Move to next task

        # Handle tasks with sub-benchmarks
        benchmarks_for_task = {k: v for k, v in task_config.items() if k != "evaluation_function"}
        benchmark_options = list(benchmarks_for_task.keys())

        if not benchmark_options:
            print(f"Warning: Task '{task_name}' has no benchmarks defined or no evaluable benchmarks.")
            continue
        
        print(f"\n--- Benchmarks for {task_name} ---")
        for i, benchmark in enumerate(benchmark_options):
            desc = benchmarks_for_task[benchmark].get("description", "")
            print(f"{i+1}. {benchmark} ({desc})")
        print(f"{len(benchmark_options)+1}. ALL (Evaluate on all benchmarks for {task_name})")

        selected_benchmark_indices = input(f"Enter the numbers of benchmarks for {task_name} (e.g., '1', '{len(benchmark_options)+1}' for ALL): ").strip().split()

        chosen_benchmarks_for_task = []
        if str(len(benchmark_options) + 1) in selected_benchmark_indices:
            chosen_benchmarks_for_task = benchmark_options
        else:
            for idx_str in selected_benchmark_indices:
                try:
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(benchmark_options):
                        chosen_benchmarks_for_task.append(benchmark_options[idx])
                    else:
                        print(f"Warning: Invalid benchmark number '{idx_str}' for task '{task_name}' ignored.")
                except ValueError:
                    print(f"Warning: Invalid input '{idx_str}' for task '{task_name}' ignored.")

        if not chosen_benchmarks_for_task:
            print(f"No valid benchmarks selected for task '{task_name}'. Skipping.")
            continue

        for benchmark_name in chosen_benchmarks_for_task:
            score = pd.NA
            evaluation_function_name = benchmarks_for_task[benchmark_name].get("evaluation_function") # Check sub-benchmark for func

            pre_calc_entry = PRE_CALCULATED_RESULTS[
                (PRE_CALCULATED_RESULTS['Model'].str.lower() == model_name_lower) &
                (PRE_CALCULATED_RESULTS['Task'].str.lower() == task_name.lower()) &
                (PRE_CALCULATED_RESULTS['Benchmark'].str.lower() == benchmark_name.lower())
            ]

            if not pre_calc_entry.empty:
                score = pre_calc_entry['Score'].iloc[0]
                print(f"Using pre-calculated score for {task_name} - {benchmark_name}: {score}")
            elif evaluation_function_name:
                if model_pipeline is None:
                    print(f"Model pipeline not initialized for {task_name} - {benchmark_name}. Skipping evaluation.")
                else:
                    eval_func = globals().get(evaluation_function_name)
                    if eval_func:
                        try:
                            # Pass relevant args for multi-GPU distribution to eval function if needed
                            scores = eval_func(input_model_name, model_pipeline, calculated_model_size_gb, benchmark=benchmark_name)
                            score = scores.get(benchmark_name, pd.NA) # For sub-benchmarks, key is benchmark_name
                            if isinstance(score, (int, float)):
                                new_results_to_save.append({
                                    'Model': input_model_name,
                                    'Size (B)': model_size_value,
                                    'Task': task_name,
                                    'Benchmark': benchmark_name,
                                    'Score': score,
                                    'Timestamp': datetime.now().isoformat()
                                })
                        except Exception as e:
                            print(f"Error evaluating {task_name} - {benchmark_name}: {e}")
                    else:
                        print(f"Evaluation function '{evaluation_function_name}' not found for {task_name} - {benchmark_name}.")
            else:
                print(f"No evaluation function defined for task '{task_name}' and benchmark '{benchmark_name}'.")
                score = pd.NA

            current_model_row_data[(task_name, benchmark_name)] = round(score, 3) if isinstance(score, (int, float)) else score
            if isinstance(score, (int, float)):
                scores_for_averaging.append(score)

        if len(scores_for_averaging) > 1:
            avg_score = sum(scores_for_averaging) / len(scores_for_averaging)
            current_model_row_data[(task_name, 'Average')] = round(avg_score, 3)
        elif len(scores_for_averaging) == 1:
            current_model_row_data[(task_name, 'Average')] = round(scores_for_averaging[0], 3)
        else:
            current_model_row_data[(task_name, 'Average')] = pd.NA

    # --- After all evaluations, save results to CSV ---
    if new_results_to_save:
        new_df = pd.DataFrame(new_results_to_save)
        # Load existing data if file exists, then append/update
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            # Combine and remove duplicates, keeping new results
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=['Model', 'Task', 'Benchmark'], keep='last'
            )
            combined_df.to_csv(csv_file_path, index=False)
        else:
            new_df.to_csv(csv_file_path, index=False)
        print(f"New and updated results saved to '{csv_file_path}'.")

    # --- Display Logic (remains largely same) ---
    all_possible_multiindex_columns = [('Model', ''), ('Size (B)', '')]
    for task_name in BENCHMARK_CONFIG:
        # Check if it's a top-level benchmark that doesn't have sub-benchmarks in its config
        if task_name in ["MMLU", "BBH", "AGIEval"]:
            all_possible_multiindex_columns.append((task_name, '')) # Main score column
            all_possible_multiindex_columns.append((task_name, 'Average'))
        else:
            # tasks with sub-benchmarks
            for benchmark_name in BENCHMARK_CONFIG[task_name]:
                if benchmark_name != "evaluation_function": # Ensure this is always true for sub-benchmarks
                    all_possible_multiindex_columns.append((task_name, benchmark_name))
            all_possible_multiindex_columns.append((task_name, 'Average'))

    df_for_display = pd.DataFrame(columns=pd.MultiIndex.from_tuples(all_possible_multiindex_columns))

    df_for_display.loc[0] = {col: current_model_row_data.get(col, pd.NA) for col in all_possible_multiindex_columns}

    actual_display_columns_tuples = [
        col for col in all_possible_multiindex_columns
        if col in current_model_row_data or col[0] in ['Model', 'Size (B)']
    ]

    def sort_key_for_display_cols(col_tuple):
        if col_tuple[0] == 'Model': return (0, '')
        if col_tuple[0] == 'Size (B)': return (1, '')

        top_level_order = {"MMLU": 0, "BBH": 1, "AGIEval": 2}

        if col_tuple[0] in top_level_order:
            order_idx = top_level_order[col_tuple[0]]
            # Ensure the top-level score column comes before its average (if 'Average' is also there)
            return (2, order_idx, 0 if col_tuple[1] == '' else 1)

        task_main_order = list(BENCHMARK_CONFIG.keys()).index(col_tuple[0]) if col_tuple[0] in BENCHMARK_CONFIG else 999
        if col_tuple[1] == 'Average': return (3, task_main_order, 'ZZZ')
        return (3, task_main_order, col_tuple[1])

    sorted_actual_display_columns = sorted(actual_display_columns_tuples, key=sort_key_for_display_cols)

    df_for_display = df_for_display[sorted_actual_display_columns]

    # Convert MultiIndex columns to single-level for markdown if the second level is empty string
    df_for_display.columns = pd.MultiIndex.from_tuples([
        (col[0] if col[1] == '' else col[0], col[1]) for col in df_for_display.columns
    ])

    print("\n--- Evaluation Results ---")
    print(df_for_display.to_markdown(index=False, numalign="left", stralign="left"))


    if model_pipeline:
        try:
            # Ensure proper cleanup
            del model_pipeline.model
            del model_pipeline.tokenizer
            del model_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model pipeline and GPU memory cleaned up.")
        except Exception as e:
            print(f"Error during model cleanup: {e}")

if __name__ == "__main__":
    main()