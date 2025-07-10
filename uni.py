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

# have created dictionary
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
        "GSM8K": {"description": "8-shot average top-1 accuracy"},
        "MATH": {"description": "4-shot average top-1 accuracy"},
    },
    # MMLU, BBH, and AGIEval are now top-level benchmarks
    "MMLU": {"description": "5-shot overall results", "evaluation_function": "evaluate_mmlu"},
    "BBH": {"description": "3-shot overall results", "evaluation_function": "evaluate_bbh"},
    "AGIEval": {"description": "3–5 shot overall results (English tasks averaged)", "evaluation_function": "evaluate_agieval"},
}

PRE_CALCULATED_RESULTS = pd.DataFrame()

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
                    output = pipe(prompt, **generation_params)
                    generated_text = output[0].get('generated_text', '') if output and len(output) > 0 else "# Generation failed: No output"
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
    if not torch.cuda.is_available():
        device_id = -1

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config if torch.cuda.is_available() else None,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        except Exception as e_no_quant:
            return None, None

    model_size_gb = 0
    if model:
        try:
            total_params = sum(p.numel() for p in model.parameters())
            model_size_gb = total_params * model.dtype.itemsize / (1024**3)
        except Exception as e:
            pass 

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return pipe, model_size_gb

# function placeholders !

def evaluate_humaneval(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    """Placeholder for HumanEval evaluation."""
    print(f"\n--- Running HumanEval evaluation for {model_name} (placeholder) ---")
    return {"HumanEval": 0.0}

def evaluate_mbpp(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    """Placeholder for MBPP evaluation."""
    print(f"\n--- Running MBPP evaluation for {model_name} (placeholder) ---")
    return {"MBPP": 0.0}

def evaluate_commonsense_reasoning(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    """Placeholder for Commonsense Reasoning evaluation."""
    return {benchmark: 60.0 + (hash(model_name + benchmark) % 10)}

def evaluate_world_knowledge(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    """Placeholder for World Knowledge evaluation."""
    return {benchmark: 75.0 + (hash(model_name + benchmark) % 10)}

def evaluate_reading_comprehension(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    """Placeholder for Reading Comprehension evaluation."""
    return {benchmark: 80.0 + (hash(model_name + benchmark) % 10)}

def evaluate_math(model_name: str, pipe, model_size_gb: float, benchmark: str) -> Dict[str, float]:
    """Placeholder for Math evaluation."""
    return {benchmark: 15.0 + (hash(model_name + benchmark) % 5)}

def evaluate_mmlu(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    """Placeholder for MMLU evaluation."""
    print(f"\n--- Running MMLU evaluation for {model_name} (placeholder) ---")
    return {"MMLU": 65.0 + (hash(model_name) % 15)}

def evaluate_bbh(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    """Placeholder for BBH evaluation."""
    print(f"\n--- Running BBH evaluation for {model_name} (placeholder) ---")
    return {"BBH": 50.0 + (hash(model_name) % 10)}

def evaluate_agieval(model_name: str, pipe, model_size_gb: float) -> Dict[str, float]:
    """Placeholder for AGIEval evaluation."""
    print(f"\n--- Running AGIEval evaluation for {model_name} (placeholder) ---")
    return {"AGIEval": 40.0 + (hash(model_name) % 10)}

def main():
    global PRE_CALCULATED_RESULTS

    input_model_name = input("Enter the model name (e.g., tiiuae/falcon-7b, tiiuae/falcon-40b, or your custom model path): ").strip()
    model_name_lower = input_model_name.lower()

    model_pipeline = None
    model_size_value = 'N/A'

    # precalculated tasks
    csv_file_path = 'calculated.csv'
    if os.path.exists(csv_file_path):
        try:
            temp_df = pd.read_csv(csv_file_path)
            PRE_CALCULATED_RESULTS = temp_df[temp_df['Model'].str.lower() == model_name_lower].copy()
            if not PRE_CALCULATED_RESULTS.empty:
                model_size_value = f"{PRE_CALCULATED_RESULTS['Size (B)'].iloc[0]}B"
        except Exception as e:
            PRE_CALCULATED_RESULTS = pd.DataFrame()
    
    if PRE_CALCULATED_RESULTS.empty: #when model not present //
        model_pipeline, calculated_model_size_gb = initialize_pipeline(input_model_name)
        if model_pipeline is None:
            print(f"Failed to initialize model: {input_model_name}. Exiting.")
            return
        model_size_value = f"{calculated_model_size_gb:.2f} GB" if calculated_model_size_gb is not None else 'N/A'

    print("\n--- Available Tasks ---")
    tasks = list(BENCHMARK_CONFIG.keys())
    for i, task in enumerate(tasks):
        print(f"{i+1}. {task}")
    print(f"{len(tasks)+1}. ALL (Evaluate on all possible tasks)")

    selected_task_indices = input(f"Enter the numbers of tasks you want to evaluate (e.g., '1' for Code Generation, '1 3' for Code Generation and World Knowledge, '{len(tasks)+1}' for ALL): ").strip().split()

    chosen_tasks = []
    if str(len(tasks) + 1) in selected_task_indices: # if ALL tasks chosen
        chosen_tasks = tasks
    else:
        for idx_str in selected_task_indices:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(tasks): # if other tasks chosen
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

    for task_name in chosen_tasks:
        benchmarks_for_task = BENCHMARK_CONFIG[task_name]
        benchmark_options = [b for b in benchmarks_for_task.keys() if b != "evaluation_function"]
        if task_name in ["MMLU", "BBH", "AGIEval"]: # sep
            score = pd.NA
            pre_calc_entry = PRE_CALCULATED_RESULTS[
                (PRE_CALCULATED_RESULTS['Model'].str.lower() == model_name_lower) &
                (PRE_CALCULATED_RESULTS['Task'].str.lower() == task_name.lower())
            ]

            if not pre_calc_entry.empty:
                score = pre_calc_entry['Score'].iloc[0]
            else:
                if model_pipeline is None:
                    print(f"Model pipeline not initialized for {task_name}. Skipping evaluation.")
                    score = pd.NA
                else:
                    if task_name == "CODE GENERATION":
                        if benchmark_name == "HumanEval":
                            scores = evaluate_humaneval(input_model_name, model_pipeline, model_size_value)
                            score = scores.get("HumanEval", pd.NA)
                        elif benchmark_name == "MBPP":
                            scores = evaluate_mbpp(input_model_name, model_pipeline, model_size_value)
                            score = scores.get("MBPP", pd.NA)
                    elif task_name == "COMMONSENSE REASONING":
                        scores = evaluate_commonsense_reasoning(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "WORLD KNOWLEDGE":
                        scores = evaluate_world_knowledge(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "READING COMPREHENSION":
                        scores = evaluate_reading_comprehension(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "MATH":
                        scores = evaluate_math(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "MMLU":
                        scores = evaluate_mmlu(input_model_name, model_pipeline, model_size_value)
                        score = scores.get("MMLU", pd.NA)
                    elif task_name == "BBH":
                        scores = evaluate_bbh(input_model_name, model_pipeline, model_size_value)
                        score = scores.get("BBH", pd.NA)
                    elif task_name == "AGIEval":
                        scores = evaluate_agieval(input_model_name, model_pipeline, model_size_value)
                        score = scores.get("AGIEval", pd.NA)
                    else:
                         score = pd.NA
            

            current_model_row_data[(task_name, '')] = round(score, 3) if isinstance(score, (int, float)) else score
            current_model_row_data[(task_name, 'Average')] = round(score, 3) if isinstance(score, (int, float)) else score # For display consistency
            continue 

        benchmarks_for_task = BENCHMARK_CONFIG[task_name]
        benchmark_options = [b for b in benchmarks_for_task.keys() if b != "evaluation_function"]

        chosen_benchmarks_for_task = []
        if not benchmark_options:
            print(f"Warning: Task '{task_name}' has no benchmarks defined or no evaluable benchmarks.")
            continue 
        else:
            print(f"\n--- Benchmarks for {task_name} ---")
            for i, benchmark in enumerate(benchmark_options):
                desc = benchmarks_for_task[benchmark].get("description", "")
                print(f"{i+1}. {benchmark} ({desc})")
            print(f"{len(benchmark_options)+1}. ALL (Evaluate on all benchmarks for {task_name})")

            selected_benchmark_indices = input(f"Enter the numbers of benchmarks for {task_name} (e.g., '1', '{len(benchmark_options)+1}' for ALL): ").strip().split()

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

        scores_for_averaging = []

        for benchmark_name in chosen_benchmarks_for_task:
            score = pd.NA

            pre_calc_entry = PRE_CALCULATED_RESULTS[
                (PRE_CALCULATED_RESULTS['Model'].str.lower() == model_name_lower) &
                (PRE_CALCULATED_RESULTS['Task'].str.lower() == task_name.lower()) &
                (PRE_CALCULATED_RESULTS['Benchmark'].str.lower() == benchmark_name.lower())
            ]

            if not pre_calc_entry.empty:
                score = pre_calc_entry['Score'].iloc[0]
            else:
                if model_pipeline is None:
                    print(f"Model pipeline not initialized for {task_name} - {benchmark_name}. Skipping evaluation.")
                    score = pd.NA
                else:
                    if task_name == "CODE GENERATION":
                        if benchmark_name == "HumanEval":
                            scores = evaluate_humaneval(input_model_name, model_pipeline, model_size_value)
                            score = scores.get("HumanEval", pd.NA)
                        elif benchmark_name == "MBPP":
                            scores = evaluate_mbpp(input_model_name, model_pipeline, model_size_value)
                            score = scores.get("MBPP", pd.NA)
                    elif task_name == "COMMONSENSE REASONING":
                        scores = evaluate_commonsense_reasoning(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "WORLD KNOWLEDGE":
                        scores = evaluate_world_knowledge(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "READING COMPREHENSION":
                        scores = evaluate_reading_comprehension(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
                    elif task_name == "MATH":
                        scores = evaluate_math(input_model_name, model_pipeline, model_size_value, benchmark_name)
                        score = scores.get(benchmark_name, pd.NA)
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

    all_possible_multiindex_columns = [('Model', ''), ('Size (B)', '')]
    for task_name in BENCHMARK_CONFIG:
        if task_name in ["MMLU", "BBH", "AGIEval"]:
            all_possible_multiindex_columns.append((task_name, '')) # Main score column
            all_possible_multiindex_columns.append((task_name, 'Average')) 
        else:
            # tasks with sub-benchmarks
            for benchmark_name in BENCHMARK_CONFIG[task_name]:
                if benchmark_name != "evaluation_function":
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
            return (2, order_idx, 0 if col_tuple[1] == '' else 1)
        
        task_main_order = list(BENCHMARK_CONFIG.keys()).index(col_tuple[0]) if col_tuple[0] in BENCHMARK_CONFIG else 999 
        if col_tuple[1] == 'Average': return (3, task_main_order, 'ZZZ') 
        return (3, task_main_order, col_tuple[1]) 

    sorted_actual_display_columns = sorted(actual_display_columns_tuples, key=sort_key_for_display_cols)

    df_for_display = df_for_display[sorted_actual_display_columns]

    df_for_display.columns = df_for_display.columns.map(lambda x: x[0] if x[1] == '' else x)

    print("\n--- Evaluation Results ---")
    print(df_for_display.to_markdown(index=False, numalign="left", stralign="left"))

    if model_pipeline:
        try:
            del model_pipeline.model
            del model_pipeline.tokenizer
            del model_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during model cleanup: {e}")

if __name__ == "__main__":
    main()