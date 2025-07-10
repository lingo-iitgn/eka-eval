import subprocess
import multiprocessing
import os
import pandas as pd
import time
from typing import List, Dict as PyDict, Tuple, Set, Any 
from collections import defaultdict
import argparse
import torch
import sys 
import json 
from main import BENCHMARK_CONFIG

def get_available_gpus():#-> List[int]
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            return list(range(num_gpus)) 
        else:
            print("Warning PyTorch: CUDA is available, but no GPUs were found.")
            return []
    else:
        return []

# for the run_evaluation_models_and_taksks in main.py
def worker_process(gpu_id: int, process_id: int, model_name: str, total_num_workers: int,
                   task_group_to_run: str, 
                   selected_benchmarks_for_group: List[str], 
                   batch_size: int):
    print(f"Worker {process_id} (Targeting GPU {gpu_id}): Starting model '{model_name}' for task group: '{task_group_to_run}', specific BMs: {selected_benchmarks_for_group}, batch_size: {batch_size}")
    try:
        python_executable = sys.executable or "python3"  #this is for main testing script
        command = [
            python_executable, "-u", "main.py",
            "--gpu_id", str(gpu_id),
            "--num_gpus", str(total_num_workers), #command created
            "--process_id", str(process_id),   
            "--model_name", model_name,
            "--batch_size", str(batch_size),   
            "--task_group", task_group_to_run,  
            "--selected_benchmarks_json", json.dumps({task_group_to_run: selected_benchmarks_for_group})
        ]
        
        print(f"DEBUG Worker {process_id}: Executing command: {' '.join(command)}")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
        
        print(f"\n--------- Output from Worker {process_id} (GPU {gpu_id}) for Task Group '{task_group_to_run}' ---------\n")
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
            process.stdout.close()
        return_code = process.wait()
        print(f"\n--------- End Output from Worker {process_id} (GPU {gpu_id}) for Task Group '{task_group_to_run}' ---------")
        if return_code == 0:
            print(f"Worker {process_id} (GPU {gpu_id}): Finished task group '{task_group_to_run}' successfully (RC: {return_code}).")
        else:
            print(f"Worker {process_id} (GPU {gpu_id}): Task group '{task_group_to_run}' exited with error (RC: {return_code}).")

    except Exception as e:
        print(f"Worker {process_id} (GPU {gpu_id}): FATAL error launching/monitoring main.py for task group '{task_group_to_run}': {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description=" LLM benchmark evaluations.") #diff bew main and run
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs/workers to use. Default: all available or 1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=1, help="Default batch size passed to worker tasks (can be overridden by specific task logic in main.py).")
    args = parser.parse_args()

    if not BENCHMARK_CONFIG: # Check if BENCHMARK_CONFIG loaded
        print("CRITICAL: BENCHMARK_CONFIG is not available. Exiting.")
        return

    # USER INPUT/CHOICE BASED
    csv_file_path = 'calculated.csv'
    print("\n--- Model Selection ---")
    model_source_choice = input("Enter model source ('1' for Hugging Face, '2' for Local Path): ").strip()
    model_path = ""
    if model_source_choice == '1':
        model_path = input("Enter Hugging Face model name (e.g., google/gemma-2b): ").strip()
    elif model_source_choice == '2': # for local models
        model_path = input("Enter the full local path to the model directory: ").strip()
        if not os.path.isdir(model_path):
            print(f"Warning: Local path '{model_path}' does not seem to be a valid directory.")
    else:
        print("Invalid model source choice. Exiting.")
        return
    
    input_model_name = model_path
    if not input_model_name: 
        print("Model name cannot be empty. Exiting."); return #>2!
    model_name_lower = input_model_name.lower()
    
    current_session_benchmark_config = json.loads(json.dumps(BENCHMARK_CONFIG)) 

    add_custom = input("Do you want to add any custom/internal benchmarks for this session? (yes/no): ").strip().lower()
    cust_ben_dety = {} 

# condition for adding internal benchmarks
    if add_custom == 'yes':
        while True:
            print("\n--- Adding Custom Benchmark ---")
            custom_task_group = input("Enter Task Group name for this custom benchmark (e.g., CUSTOM EVALS): ").strip()
            custom_bm_name = input("Enter a unique display name for this custom benchmark (e.g., MySpecialTest): ").strip()
            custom_module_path = input("Enter Python module path (e.g., my_custom_evals.script_name): ").strip()
            custom_func_name = input(f"Enter function name in '{custom_module_path}' to call (e.g., evaluate_my_test): ").strip()

            if not all([custom_task_group, custom_bm_name, custom_module_path, custom_func_name]):
                print("All fields are required for a custom benchmark. Skipping this one.")
            else:
                if custom_task_group not in current_session_benchmark_config:
                    current_session_benchmark_config[custom_task_group] = {}
                
                current_session_benchmark_config[custom_task_group][custom_bm_name] = {
                    "description": f"Custom benchmark: {custom_bm_name}",
                    "evaluation_function": "evaluate_dynamically_loaded_benchmark", 
                    "custom_module_path": custom_module_path,
                    "custom_function_name": custom_func_name
                }
    
                if task_group_name not in cust_ben_dety:
                    cust_ben_dety[task_group_name] = {}
                cust_ben_dety[task_group_name][custom_bm_name] = {
                    "module": custom_module_path, 
                    "func": custom_func_name
                }
                print(f"Added custom benchmark '{custom_bm_name}' under task group '{custom_task_group}'.")

            if input("Add another custom benchmark? (yes/no): ").strip().lower() != 'yes':
                break
            
    
    # selection benchmarks
    print("\n--- Available Benchmarks ---")
    all_config_task_groups = list(BENCHMARK_CONFIG.keys())
    for i, task_group in enumerate(all_config_task_groups): print(f"{i+1}. {task_group}")
    print(f"{len(all_config_task_groups)+1}. ALL")
    
    selected_indices_str = input(f"Enter task group #(s) (e.g., '1', '1 3', 'ALL'): ").strip().lower().split()
    
    chosen_initial_task_groups: List[str] = []
    if "all" in selected_indices_str or str(len(all_config_task_groups) + 1) in selected_indices_str :
        chosen_initial_task_groups = all_config_task_groups
    else:
        for idx_str in selected_indices_str:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(all_config_task_groups):
                    chosen_initial_task_groups.append(all_config_task_groups[idx])
                else: 
                    print(f"Warn: Invalid task group # '{idx_str}' ignored.")
            except ValueError: 
                print(f"Warn: Invalid input '{idx_str}' ignored.")
    if not chosen_initial_task_groups: 
        print("No valid top-level task groups selected. Exiting."); return #?!
    print(f"Selected top-level task groups: {chosen_initial_task_groups}")

    user_selected_benchmarks: PyDict[str, List[str]] = {}  #?!
    ordered_selected_task_groups_for_processing: List[str] = [] 

    for task_group_name in chosen_initial_task_groups:
        if task_group_name not in BENCHMARK_CONFIG: 
            continue 

        is_single_bm_task_group = task_group_name in ["MMLU", "BBH", "AGIEval"]
        if is_single_bm_task_group:
            user_selected_benchmarks[task_group_name] = [task_group_name] 
            if task_group_name not in ordered_selected_task_groups_for_processing:
                ordered_selected_task_groups_for_processing.append(task_group_name)
        else:
            print(f"\n--- Select benchmarks for Task Group: {task_group_name} ---")
            available_sub_benchmarks = [bm for bm in BENCHMARK_CONFIG[task_group_name] if bm != "evaluation_function"]
            if not available_sub_benchmarks: print(f"Warn: No sub-benchmarks for {task_group_name}. Skipping."); continue

            for i, sub_bm in enumerate(available_sub_benchmarks): print(f"{i+1}. {sub_bm}")
            print(f"{len(available_sub_benchmarks)+1}. ALL (within {task_group_name})")
            print(f"{len(available_sub_benchmarks)+2}. SKIP THIS TASK GROUP")
            selected_sub_indices_str = input(f"Select benchmark #(s) for {task_group_name} ('ALL', 'SKIP', nums): ").strip().lower().split()
            
            selected_for_this_group: List[str] = []
            if "skip" in selected_sub_indices_str or str(len(available_sub_benchmarks)+2) in selected_sub_indices_str:
                print(f"Skipping task group: {task_group_name}"); continue 
            if "all" in selected_sub_indices_str or str(len(available_sub_benchmarks)+1) in selected_sub_indices_str:
                selected_for_this_group = available_sub_benchmarks
            else:
                for sub_idx_str in selected_sub_indices_str:
                    try:
                        sub_idx = int(sub_idx_str) - 1
                        if 0 <= sub_idx < len(available_sub_benchmarks): selected_for_this_group.append(available_sub_benchmarks[sub_idx])
                        else: print(f"Warn: Invalid benchmark # '{sub_idx_str}' for {task_group_name} ignored.")
                    except ValueError: print(f"Warn: Invalid input '{sub_idx_str}' for {task_group_name} ignored.")
            
            if selected_for_this_group:
                user_selected_benchmarks[task_group_name] = sorted(list(set(selected_for_this_group)))
                if task_group_name not in ordered_selected_task_groups_for_processing:
                     ordered_selected_task_groups_for_processing.append(task_group_name)

    if not user_selected_benchmarks: print("No benchmarks selected, exiting."); return

    print("\n--- Final Benchmarks Selected for Evaluation (Task Group: [Specific Benchmarks]) ---")
    for tg_name in ordered_selected_task_groups_for_processing:
        if tg_name in user_selected_benchmarks: print(f"- {tg_name}: {user_selected_benchmarks[tg_name]}")

    # already computed benchmarks
    completed_benchmarks_set: Set[Tuple[str, str]] = set()
    if os.path.exists(csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            if all(col in df.columns for col in ['Model', 'Task', 'Benchmark', 'Score']):
                model_df = df[df['Model'].str.lower() == model_name_lower]
                for _, row in model_df.iterrows():
                    if pd.notna(row['Score']): completed_benchmarks_set.add((row['Task'], row['Benchmark']))
                print(f"Found {len(completed_benchmarks_set)} completed benchmarks for '{model_name_lower}' in '{csv_file_path}'.")
        except Exception as e: print(f"Error loading '{csv_file_path}': {e}. No completed assumed.")

    # tasks scheduling
    tasks_to_schedule_for_workers: PyDict[str, List[str]] = defaultdict(list) 
    for task_group, selected_bms_for_group in user_selected_benchmarks.items():
        bms_needing_eval_for_group = [bm for bm in selected_bms_for_group if (task_group, bm) not in completed_benchmarks_set]
        if bms_needing_eval_for_group:
            tasks_to_schedule_for_workers[task_group] = bms_needing_eval_for_group
            
    if not tasks_to_schedule_for_workers: #displaying precomputed
        print(f"All specifically selected benchmarks for model '{input_model_name}' are already completed.")
        display_consolidated_results(input_model_name, csv_file_path, user_selected_benchmarks, ordered_selected_task_groups_for_processing)
        return

    print(f"\n--- Tasks Requiring Evaluation (Task Group: [Benchmarks]) ---") # live evaluation
    for tg_name in ordered_selected_task_groups_for_processing:
        if tg_name in tasks_to_schedule_for_workers: print(f"- {tg_name}: {tasks_to_schedule_for_workers[tg_name]}")

    available_gpu_ids_from_smi = get_available_gpus() #
    is_cpu_run = not available_gpu_ids_from_smi or not torch.cuda.is_available()
    effective_gpu_ids_for_worker_assignment = available_gpu_ids_from_smi if not is_cpu_run else [0] 
    num_available_gpu_slots = len(effective_gpu_ids_for_worker_assignment) if not is_cpu_run else 1

    total_workers_to_use = num_available_gpu_slots 
    if args.num_gpus is not None: 
        if args.num_gpus <= 0: total_workers_to_use = num_available_gpu_slots
        elif args.num_gpus > num_available_gpu_slots and not is_cpu_run :
            print(f"Warn: Req {args.num_gpus} GPUs, have {num_available_gpu_slots}. Using {num_available_gpu_slots}.")
            total_workers_to_use = num_available_gpu_slots
        else: total_workers_to_use = args.num_gpus
    if is_cpu_run: total_workers_to_use = 1 
    if total_workers_to_use == 0 and not is_cpu_run: print("Err: No GPU workers. Exit."); return
    print(f"Using {total_workers_to_use} {'CPU process' if is_cpu_run else 'GPU worker(s)'} for evaluation.")

    work_items_to_distribute: List[PyDict[str, Any]] = [] 
    for tg_name_ordered in ordered_selected_task_groups_for_processing:
        if tg_name_ordered in tasks_to_schedule_for_workers:
            work_items_to_distribute.append({
                'task_group': tg_name_ordered,
                'benchmarks': tasks_to_schedule_for_workers[tg_name_ordered]
            })

    processes = []
    for i, work_item in enumerate(work_items_to_distribute):
        worker_local_id = i % total_workers_to_use
        assigned_physical_gpu_id = 0 
        if not is_cpu_run and effective_gpu_ids_for_worker_assignment:
            assigned_physical_gpu_id = effective_gpu_ids_for_worker_assignment[worker_local_id % len(effective_gpu_ids_for_worker_assignment)]

        task_group_to_run = work_item['task_group']
        specific_benchmarks_for_group = work_item['benchmarks']
        
        subprocess_unique_id = i 

        print(f"Preparing Worker (Subprocess {subprocess_unique_id} mapped to logical worker slot {worker_local_id}, targeting GPU {assigned_physical_gpu_id}): TG '{task_group_to_run}', BMs: {specific_benchmarks_for_group}")
        p = multiprocessing.Process(
            target=worker_process,
            args=(
                assigned_physical_gpu_id, 
                subprocess_unique_id, 
                input_model_name,
                total_workers_to_use, 
                task_group_to_run, 
                specific_benchmarks_for_group, 
                args.batch_size 
            )
        )
        processes.append(p)
        p.start()
        if total_workers_to_use > 1 and not is_cpu_run and len(processes) % total_workers_to_use == 0 :
            time.sleep(max(1, 5 // total_workers_to_use if total_workers_to_use > 0 else 5)) 

    for p in processes: p.join()

    print("\nAll evaluation worker processes have finished.")
    print("Consolidating and displaying results...")
    display_consolidated_results(input_model_name, csv_file_path, user_selected_benchmarks, ordered_selected_task_groups_for_processing)

def display_consolidated_results(model_name: str, csv_path: str,
                                 user_selected_benchmarks: PyDict[str, List[str]],
                                 ordered_task_groups_for_display: List[str]):
    if not os.path.exists(csv_path): print(f"Results file '{csv_path}' not found."); return
    try:
        final_df = pd.read_csv(csv_path)
        model_df_display = final_df[final_df['Model'].str.lower() == model_name.lower()].copy()
        if model_df_display.empty: print(f"\nNo results for '{model_name}' in '{csv_path}'."); return
        model_df_display['Score'] = pd.to_numeric(model_df_display['Score'], errors='coerce')

        size_b_val = 'N/A'
        if 'Size (B)' in model_df_display.columns and not model_df_display['Size (B)'].dropna().empty:
            size_b_val = model_df_display['Size (B)'].dropna().iloc[0]
        current_model_row_data = {('Model', ''): model_name, ('Size (B)', ''): size_b_val}

        task_bm_scores = defaultdict(lambda: defaultdict(lambda: pd.NA))
        for _, row in model_df_display.iterrows(): task_bm_scores[row['Task']][row['Benchmark']] = row['Score']
        
        multi_idx_cols = [('Model', ''), ('Size (B)', '')]
        for task_group_name in ordered_task_groups_for_display: 
            selected_bms_in_group = user_selected_benchmarks.get(task_group_name, [])
            if not selected_bms_in_group : continue
            is_single_bm_task_group = task_group_name in ["MMLU", "BBH", "AGIEval"]
            if is_single_bm_task_group:
                if task_group_name in selected_bms_in_group:
                    score = task_bm_scores[task_group_name].get(task_group_name, pd.NA)
                    current_model_row_data[(task_group_name, '')] = round(score,2) if pd.notna(score) else pd.NA
                    current_model_row_data[(task_group_name, 'Average')] = round(score,2) if pd.notna(score) else pd.NA
                    multi_idx_cols.extend([(task_group_name, ''), (task_group_name, 'Average')])
            else:
                actual_scores_for_group_avg = []
                for bm_name in selected_bms_in_group: 
                    score = task_bm_scores[task_group_name].get(bm_name, pd.NA)
                    current_model_row_data[(task_group_name, bm_name)] = round(score,2) if pd.notna(score) else pd.NA
                    multi_idx_cols.append((task_group_name, bm_name))
                    if pd.notna(score): actual_scores_for_group_avg.append(score)
                if len(selected_bms_in_group) > 1: 
                    avg_score_from_csv = task_bm_scores[task_group_name].get('Average', pd.NA)
                    if pd.notna(avg_score_from_csv):
                         current_model_row_data[(task_group_name, 'Average')] = round(avg_score_from_csv,2)
                         multi_idx_cols.append((task_group_name, 'Average')) 
        
        seen_cols, unique_cols = set(), []
        for col in multi_idx_cols:
            if col not in seen_cols: unique_cols.append(col); seen_cols.add(col)
        if not unique_cols: print("Warn: No columns generated for display table."); return

        df_disp = pd.DataFrame(columns=pd.MultiIndex.from_tuples(unique_cols))
        row_data = {col_t: current_model_row_data.get(col_t, pd.NA) for col_t in unique_cols}
        series_data = pd.Series(row_data, index=pd.MultiIndex.from_tuples(unique_cols))
        if not series_data.empty: df_disp.loc[0] = series_data
        elif unique_cols : df_disp.loc[0] = pd.NA  

        def sort_key_final(col_tuple):
            tg_name, bm_name = col_tuple[0], col_tuple[1]
            if tg_name == 'Model': return (0,0,0)
            if tg_name == 'Size (B)': return (1,0,0)
            try: task_order = ordered_task_groups_for_display.index(tg_name)
            except ValueError: task_order = 9999
            is_special = tg_name in ["MMLU", "BBH", "AGIEval"]
            if is_special: sub_o = 0 if bm_name == '' else 1; return (2, task_order, sub_o)
            else:
                if bm_name == 'Average': bm_order = 99999
                else:
                    try:
                        bm_keys_orig_order = [k for k in BENCHMARK_CONFIG.get(tg_name, {}) if k != "evaluation_function"]
                        bm_order = bm_keys_orig_order.index(bm_name) if bm_name in bm_keys_orig_order else 99998
                    except (KeyError, ValueError): bm_order = 99998 
                return (3, task_order, bm_order)

        cols_to_sort_from_df = [col for col in df_disp.columns.tolist() if col in unique_cols]
        if not cols_to_sort_from_df and not df_disp.empty: cols_to_sort_from_df = df_disp.columns.tolist()
        if cols_to_sort_from_df:
            sorted_cols = sorted(cols_to_sort_from_df, key=sort_key_final)
            df_disp = df_disp[sorted_cols]
        elif df_disp.empty: print("No results to display in table."); return
        
        print("\n--- Consolidated Evaluation Results ---")
        print(df_disp.to_markdown(index=False, floatfmt=".2f"))
    except Exception as e:
        print(f"Error displaying consolidated results from '{csv_path}': {e}")
        import traceback; traceback.print_exc() #point?
        
# to add deletion of results option

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()