# scripts/run_evaluation_suite.py

import subprocess
import multiprocessing
import os
import pandas as pd # Still needed for display_consolidated_results and completed check
import time
import sys
import json
import argparse
import logging # For logging levels
from typing import List, Dict as PyDict, Tuple, Set, Any
from collections import defaultdict

# --- Eka Eval Core Imports ---
from eka_eval.benchmarks.benchmark_registry import BenchmarkRegistry
from eka_eval.utils.gpu_utils import get_available_gpus
from eka_eval.utils.logging_setup import setup_logging
# from eka_eval.utils.file_utils import ensure_dir_exists # If needed for results dir
from eka_eval.utils import constants # If you created constants.py

# Configure logger for this orchestrator script
logger = logging.getLogger(__name__) # Will be configured by setup_logging

# --- Worker Process Function (Handles subprocess execution) ---
def worker_process(
    assigned_physical_gpu_id: int,
    subprocess_unique_id: int,
    model_name_or_path: str,
    total_num_workers: int,
    task_group_to_run: str,
    selected_benchmarks_for_group: List[str],
    orchestrator_batch_size: int,
    # Potentially pass path to benchmark config if worker needs to init its own registry
    # or path to results dir if not using a shared ResultManager instance.
    # For now, assuming worker initializes its own based on standard paths or args.
):
    """
    Manages the execution of a single worker (evaluation_worker.py) as a subprocess.
    """
    worker_log_prefix = f"Worker {subprocess_unique_id} (GPU {assigned_physical_gpu_id})"
    logger.info(
        f"{worker_log_prefix}: Starting model '{model_name_or_path}' for TG: '{task_group_to_run}', "
        f"BMs: {selected_benchmarks_for_group}, BatchSize: {orchestrator_batch_size}"
    )
    try:
        python_executable = sys.executable or "python3"
        # --- IMPORTANT: Define the path to your worker script ---
        # This assumes your project structure is:
        # eka-eval/
        #   scripts/
        #     run_evaluation_suite.py (this file)
        #     evaluation_worker.py    (the worker)
        #   eka_eval/
        #     ... (library code)
        # Determine the root directory of your project
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        worker_script_path = os.path.join(project_root, "scripts", "evaluation_worker.py")

        if not os.path.exists(worker_script_path):
            logger.error(f"{worker_log_prefix}: CRITICAL - Worker script not found at {worker_script_path}. Aborting this worker.")
            return

        command = [
            python_executable, "-u", worker_script_path,
            "--gpu_id", str(assigned_physical_gpu_id), # Physical GPU ID for CUDA_VISIBLE_DEVICES in worker
            "--num_gpus", str(total_num_workers),
            "--process_id", str(subprocess_unique_id), # Logical worker ID
            "--model_name", model_name_or_path,
            "--batch_size", str(orchestrator_batch_size),
            "--task_group", task_group_to_run,
            # Pass benchmarks as a JSON string mapping task_group to its list of BMs
            "--selected_benchmarks_json", json.dumps({task_group_to_run: selected_benchmarks_for_group}),
            # Add other necessary args for the worker, e.g., results_dir
            "--results_dir", constants.DEFAULT_RESULTS_DIR if hasattr(constants, 'DEFAULT_RESULTS_DIR') else "results_output"
        ]

        logger.debug(f"{worker_log_prefix}: Executing command: {' '.join(command)}")

        # Using Popen for live output streaming
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Combine stdout and stderr
            text=True,
            encoding='utf-8',
            bufsize=1 # Line buffered
        )

        logger.info(f"\n--------- Output from {worker_log_prefix} for TG '{task_group_to_run}' ---------\n")
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                # Prepend worker ID to each line for clarity if logs get interleaved
                sys.stdout.write(f"[{worker_log_prefix}] {line}")
                sys.stdout.flush()
            process.stdout.close()

        return_code = process.wait()
        logger.info(f"\n--------- End Output from {worker_log_prefix} for TG '{task_group_to_run}' ---------")

        if return_code == 0:
            logger.info(f"{worker_log_prefix}: Finished TG '{task_group_to_run}' successfully (RC: {return_code}).")
        else:
            logger.error(f"{worker_log_prefix}: TG '{task_group_to_run}' exited with error (RC: {return_code}).")

    except Exception as e:
        logger.critical(
            f"{worker_log_prefix}: FATAL error launching/monitoring worker for TG '{task_group_to_run}': {e}",
            exc_info=True
        )

# --- Main Orchestrator Logic ---
def main_orchestrator():
    """
    Main function to orchestrate the LLM benchmark evaluations.
    Handles user input, task scheduling, and worker management.
    """
    parser = argparse.ArgumentParser(description="Eka-Eval: LLM Benchmark Evaluation Suite.")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs/workers to use. Default: all available or 1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=1, help="Default batch size for worker tasks.")
    parser.add_argument("--results_dir", type=str, default=constants.DEFAULT_RESULTS_DIR if hasattr(constants, 'DEFAULT_RESULTS_DIR') else "results_output", help="Directory to save evaluation results.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    # Add other orchestrator-level arguments if needed

    args = parser.parse_args()

    # Setup logging for the orchestrator
    # The worker_id for orchestrator can be fixed or omitted if it's always the main process
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    setup_logging(level=log_level_map.get(args.log_level.upper(), logging.INFO), worker_id="Orchestrator")


    logger.info("--- Eka-Eval Orchestrator Starting ---")

    # Ensure results directory exists
    # ensure_dir_exists(args.results_dir) # If using file_utils

    # --- Initialize Benchmark Registry ---
    benchmark_registry = BenchmarkRegistry() # Assumes config is at eka_eval.config.benchmark_config
    if not benchmark_registry.benchmarks:
        logger.critical("Benchmark configuration is empty or failed to load. Exiting.")
        return

    # --- Model Selection ---
    logger.info("\n--- Model Selection ---")
    model_source_choice = input("Enter model source ('1' for Hugging Face, '2' for Local Path): ").strip()
    model_path = ""
    if model_source_choice == '1':
        model_path = input("Enter Hugging Face model name (e.g., google/gemma-2b): ").strip()
    elif model_source_choice == '2':
        model_path = input("Enter the full local path to the model directory: ").strip()
        if not os.path.isdir(model_path): # Basic check
            logger.warning(f"Local path '{model_path}' does not seem to be a valid directory.")
    else:
        logger.error("Invalid model source choice. Exiting.")
        return

    input_model_name = model_path
    if not input_model_name:
        logger.error("Model name/path cannot be empty. Exiting.")
        return
    model_name_lower = input_model_name.lower()
    results_csv_path = os.path.join(args.results_dir, f"{model_name_lower.replace('/', '_')}_results.csv") # Model-specific CSV
    # Or a single 'calculated.csv' as before:
    # results_csv_path = os.path.join(args.results_dir, 'calculated.csv')


    # --- Custom Benchmark Addition ---
    add_custom = input("Do you want to add any custom/internal benchmarks for this session? (yes/no): ").strip().lower()
    if add_custom == 'yes':
        while True:
            logger.info("\n--- Adding Custom Benchmark ---")
            custom_task_group = input("Enter Task Group name for this custom benchmark (e.g., CUSTOM EVALS): ").strip()
            custom_bm_name = input("Enter a unique display name for this custom benchmark (e.g., MySpecialTest): ").strip()
            custom_module_path = input("Enter Python module path (e.g., my_custom_evals.script_name): ").strip()
            custom_func_name = input(f"Enter function name in '{custom_module_path}' to call (e.g., evaluate_my_test): ").strip()

            if not all([custom_task_group, custom_bm_name, custom_module_path, custom_func_name]):
                logger.warning("All fields are required for a custom benchmark. Skipping this one.")
            else:
                success = benchmark_registry.add_custom_benchmark_definition(
                    custom_task_group, custom_bm_name,
                    custom_module_path, custom_func_name,
                    description=f"Custom benchmark: {custom_bm_name}"
                )
                if success:
                    logger.info(f"Added custom benchmark '{custom_bm_name}' under task group '{custom_task_group}'.")

            if input("Add another custom benchmark? (yes/no): ").strip().lower() != 'yes':
                break

    # --- Benchmark Selection (using BenchmarkRegistry) ---
    logger.info("\n--- Available Benchmark Task Groups ---")
    all_task_groups = benchmark_registry.get_task_groups()
    for i, tg_name in enumerate(all_task_groups):
        print(f"{i+1}. {tg_name}")
    print(f"{len(all_task_groups)+1}. ALL Task Groups")

    selected_indices_str = input(f"Select task group #(s) (e.g., '1', '1 3', 'ALL'): ").strip().lower().split()
    chosen_initial_task_groups: List[str] = []
    if "all" in selected_indices_str or str(len(all_task_groups) + 1) in selected_indices_str:
        chosen_initial_task_groups = all_task_groups
    else:
        for idx_str in selected_indices_str:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(all_task_groups):
                    chosen_initial_task_groups.append(all_task_groups[idx])
                else:
                    logger.warning(f"Invalid task group # '{idx_str}' ignored.")
            except ValueError:
                logger.warning(f"Invalid input '{idx_str}' for task group selection ignored.")

    if not chosen_initial_task_groups:
        logger.error("No valid top-level task groups selected. Exiting.")
        return
    logger.info(f"Selected top-level task groups for further benchmark selection: {chosen_initial_task_groups}")

    user_selected_benchmarks: PyDict[str, List[str]] = {}
    ordered_selected_task_groups_for_processing: List[str] = []

    for task_group_name in chosen_initial_task_groups:
        # Check if this task group itself is a single benchmark (e.g., MMLU)
        group_benchmarks = benchmark_registry.get_benchmarks_for_group(task_group_name)
        is_single_bm_task_group = len(group_benchmarks) == 1 and group_benchmarks[0] == task_group_name

        if is_single_bm_task_group:
            user_selected_benchmarks[task_group_name] = [task_group_name]
            if task_group_name not in ordered_selected_task_groups_for_processing:
                ordered_selected_task_groups_for_processing.append(task_group_name)
        else:
            logger.info(f"\n--- Select benchmarks for Task Group: {task_group_name} ---")
            available_sub_benchmarks = group_benchmarks # Already fetched
            if not available_sub_benchmarks:
                logger.warning(f"No sub-benchmarks defined for task group '{task_group_name}'. Skipping.")
                continue

            for i, sub_bm in enumerate(available_sub_benchmarks):
                print(f"{i+1}. {sub_bm}")
            print(f"{len(available_sub_benchmarks)+1}. ALL (within {task_group_name})")
            print(f"{len(available_sub_benchmarks)+2}. SKIP THIS TASK GROUP")

            selected_sub_indices_str = input(f"Select benchmark #(s) for {task_group_name} ('ALL', 'SKIP', nums): ").strip().lower().split()
            selected_for_this_group: List[str] = []

            if "skip" in selected_sub_indices_str or str(len(available_sub_benchmarks)+2) in selected_sub_indices_str:
                logger.info(f"Skipping task group: {task_group_name}")
                continue
            if "all" in selected_sub_indices_str or str(len(available_sub_benchmarks)+1) in selected_sub_indices_str:
                selected_for_this_group = available_sub_benchmarks
            else:
                for sub_idx_str in selected_sub_indices_str:
                    try:
                        sub_idx = int(sub_idx_str) - 1
                        if 0 <= sub_idx < len(available_sub_benchmarks):
                            selected_for_this_group.append(available_sub_benchmarks[sub_idx])
                        else:
                            logger.warning(f"Invalid benchmark # '{sub_idx_str}' for {task_group_name} ignored.")
                    except ValueError:
                        logger.warning(f"Invalid input '{sub_idx_str}' for {task_group_name} ignored.")

            if selected_for_this_group:
                user_selected_benchmarks[task_group_name] = sorted(list(set(selected_for_this_group)))
                if task_group_name not in ordered_selected_task_groups_for_processing:
                    ordered_selected_task_groups_for_processing.append(task_group_name)

    if not user_selected_benchmarks:
        logger.info("No benchmarks were selected for evaluation. Exiting.")
        return

    logger.info("\n--- Final Benchmarks Selected for Evaluation (Task Group: [Specific Benchmarks]) ---")
    for tg_name in ordered_selected_task_groups_for_processing:
        if tg_name in user_selected_benchmarks:
            logger.info(f"- {tg_name}: {user_selected_benchmarks[tg_name]}")


    # --- Check for Already Computed Benchmarks ---
    completed_benchmarks_set: Set[Tuple[str, str]] = set()
    # ensure_dir_exists(args.results_dir) # Make sure results dir exists before trying to read
    if os.path.exists(results_csv_path):
        try:
            df = pd.read_csv(results_csv_path)
            if all(col in df.columns for col in ['Model', 'Task', 'Benchmark', 'Score']):
                # Filter for the current model if 'Model' column matches input_model_name exactly
                # (or use model_name_lower if CSV stores it that way)
                model_df = df[df['Model'] == input_model_name] # Case-sensitive match
                for _, row in model_df.iterrows():
                    if pd.notna(row['Score']):
                        completed_benchmarks_set.add((row['Task'], row['Benchmark']))
                logger.info(f"Found {len(completed_benchmarks_set)} completed benchmarks for '{input_model_name}' in '{results_csv_path}'.")
            else:
                logger.warning(f"Results file '{results_csv_path}' is missing required columns. Assuming no completed benchmarks.")
        except Exception as e:
            logger.error(f"Error loading completed benchmarks from '{results_csv_path}': {e}. Assuming no completed benchmarks.", exc_info=True)


    # --- Prepare Tasks for Workers ---
    tasks_to_schedule_for_workers: PyDict[str, List[str]] = defaultdict(list)
    for task_group, selected_bms_for_group in user_selected_benchmarks.items():
        bms_needing_eval_for_group = [
            bm for bm in selected_bms_for_group if (task_group, bm) not in completed_benchmarks_set
        ]
        if bms_needing_eval_for_group:
            tasks_to_schedule_for_workers[task_group] = bms_needing_eval_for_group

    if not tasks_to_schedule_for_workers:
        logger.info(f"All specifically selected benchmarks for model '{input_model_name}' are already completed and found in '{results_csv_path}'.")
        display_consolidated_results(input_model_name, results_csv_path, user_selected_benchmarks, ordered_selected_task_groups_for_processing, benchmark_registry)
        return

    logger.info(f"\n--- Tasks Requiring Evaluation (Task Group: [Benchmarks]) ---")
    for tg_name in ordered_selected_task_groups_for_processing: # Use ordered list for display
        if tg_name in tasks_to_schedule_for_workers:
            logger.info(f"- {tg_name}: {tasks_to_schedule_for_workers[tg_name]}")


    # --- GPU Allocation and Worker Setup ---
    available_physical_gpu_ids = get_available_gpus()
    is_cpu_run = not available_physical_gpu_ids # True if list is empty

    # Determine number of workers
    if is_cpu_run:
        total_workers_to_use = 1
        effective_gpu_ids_for_assignment = [-1] # Placeholder for CPU
        logger.info("No GPUs found or CUDA not available. Running in CPU mode with 1 worker.")
    else: # GPU run
        num_available_gpu_slots = len(available_physical_gpu_ids)
        if args.num_gpus is not None and args.num_gpus > 0:
            if args.num_gpus > num_available_gpu_slots:
                logger.warning(
                    f"Requested {args.num_gpus} GPUs, but only {num_available_gpu_slots} are available. "
                    f"Using {num_available_gpu_slots} worker(s)."
                )
                total_workers_to_use = num_available_gpu_slots
            else:
                total_workers_to_use = args.num_gpus
        else: # Default to using all available GPUs
            total_workers_to_use = num_available_gpu_slots
        effective_gpu_ids_for_assignment = available_physical_gpu_ids[:total_workers_to_use]
        logger.info(f"Using {total_workers_to_use} GPU worker(s) targeting physical GPUs: {effective_gpu_ids_for_assignment}.")

    if total_workers_to_use == 0 and not is_cpu_run: # Should not happen if logic is correct
        logger.error("Error: No GPU workers available, but not a CPU run. Exiting.")
        return


    # --- Distribute Tasks to Worker Processes ---
    # Each item in work_items_to_distribute is a task group with its list of benchmarks to run
    work_items_to_distribute: List[PyDict[str, Any]] = []
    for tg_name_ordered in ordered_selected_task_groups_for_processing:
        if tg_name_ordered in tasks_to_schedule_for_workers:
            work_items_to_distribute.append({
                'task_group': tg_name_ordered,
                'benchmarks': tasks_to_schedule_for_workers[tg_name_ordered]
            })

    processes = []
    logger.info(f"\n--- Launching {len(work_items_to_distribute)} evaluation tasks across {total_workers_to_use} worker(s) ---")

    for i, work_item in enumerate(work_items_to_distribute):
        # Assign work to workers cyclically
        worker_slot_index = i % total_workers_to_use
        assigned_physical_gpu_id = effective_gpu_ids_for_assignment[worker_slot_index] if not is_cpu_run else -1 # -1 for CPU

        task_group_to_run = work_item['task_group']
        specific_benchmarks_for_group = work_item['benchmarks']
        subprocess_unique_id = i # A unique ID for each task execution (subprocess)

        logger.info(
            f"Preparing Subprocess {subprocess_unique_id} (mapped to worker slot {worker_slot_index}, "
            f"targeting {'CPU' if assigned_physical_gpu_id == -1 else f'GPU {assigned_physical_gpu_id}'}): "
            f"TG '{task_group_to_run}', BMs: {specific_benchmarks_for_group}"
        )
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

        # Optional: Stagger process starts slightly if many GPUs and many tasks
        if total_workers_to_use > 1 and not is_cpu_run and len(processes) >= total_workers_to_use:
            # If all worker slots have been filled once, wait for one to finish before launching more
            # This implements a queue of size `total_workers_to_use`
            # For true parallelism up to `total_workers_to_use`, just start them all.
            # The current logic starts all tasks then joins.
            # If you want to limit concurrent processes to `total_workers_to_use`:
            if len(processes) % total_workers_to_use == 0 or len(processes) == len(work_items_to_distribute):
                 logger.info(f"Launched a batch of {len(processes) % total_workers_to_use or total_workers_to_use} processes. Waiting for some to complete if needed.")
                 # This part is tricky. The original code launched all then joined.
                 # Sticking to that for now unless you want a fixed-size worker pool.
                 pass # Let all processes start, then join below.
            time.sleep(max(1, 3 // total_workers_to_use if total_workers_to_use > 0 else 3))


    logger.info(f"All {len(processes)} worker processes launched. Waiting for completion...")
    for i, p in enumerate(processes):
        p.join() # Wait for each process to complete
        logger.info(f"Worker process {i} (Subprocess UID {i}) has finished.")


    logger.info("\n--- All evaluation worker processes have completed. ---")
    logger.info("Consolidating and displaying results...")
    display_consolidated_results(
        input_model_name,
        results_csv_path,
        user_selected_benchmarks, # User's initial selection for display filtering
        ordered_selected_task_groups_for_processing,
        benchmark_registry # Pass registry for canonical benchmark order
    )

# --- Display Consolidated Results (largely unchanged, but uses BenchmarkRegistry for order) ---
def display_consolidated_results(
    model_name_to_display: str,
    csv_path: str,
    user_selected_benchmarks_map: PyDict[str, List[str]], # The benchmarks user wanted to see
    ordered_task_groups_for_display: List[str],
    registry: BenchmarkRegistry # For canonical order of benchmarks within a group
):
    if not os.path.exists(csv_path):
        logger.error(f"Results file '{csv_path}' not found. Cannot display results.")
        return
    try:
        final_df = pd.read_csv(csv_path)
        # Filter for the specific model, ensure case matches how it's stored in CSV
        model_df_display = final_df[final_df['Model'] == model_name_to_display].copy()

        if model_df_display.empty:
            logger.info(f"\nNo results found for model '{model_name_to_display}' in '{csv_path}'.")
            return

        model_df_display['Score'] = pd.to_numeric(model_df_display['Score'], errors='coerce')

        size_b_val = 'N/A'
        if 'Size (B)' in model_df_display.columns and not model_df_display['Size (B)'].dropna().empty:
            size_b_val = model_df_display['Size (B)'].dropna().iloc[0]

        # --- Build the display table ---
        # Use a dictionary to collect scores for the current model row
        current_model_row_data = {('Model', ''): model_name_to_display, ('Size (B)', ''): size_b_val}
        # Store all task_bm_scores from the CSV for the current model
        task_bm_scores_from_csv = defaultdict(lambda: defaultdict(lambda: pd.NA))
        for _, row in model_df_display.iterrows():
            task_bm_scores_from_csv[row['Task']][row['Benchmark']] = row['Score']

        multi_index_columns_for_df = [('Model', ''), ('Size (B)', '')]

        for task_group_name in ordered_task_groups_for_display:
            # Only consider task groups the user actually selected for processing
            if task_group_name not in user_selected_benchmarks_map:
                continue

            selected_bms_in_this_group = user_selected_benchmarks_map.get(task_group_name, [])
            if not selected_bms_in_this_group:
                continue

            # Is this task group a single benchmark type (like MMLU)?
            registry_benchmarks_for_group = registry.get_benchmarks_for_group(task_group_name)
            is_single_bm_task_group = len(registry_benchmarks_for_group) == 1 and registry_benchmarks_for_group[0] == task_group_name

            if is_single_bm_task_group:
                # For MMLU, BBH etc., the benchmark name is the task group name
                # And the user would have selected it as [task_group_name]
                if task_group_name in selected_bms_in_this_group: # Should always be true if we are here
                    score = task_bm_scores_from_csv[task_group_name].get(task_group_name, pd.NA)
                    current_model_row_data[(task_group_name, '')] = round(score, 2) if pd.notna(score) else pd.NA # Main score
                    multi_index_columns_for_df.append((task_group_name, '')) # Header: TaskGroup, Sub-header: ''
                    # For single benchmarks, "Average" is just the score itself
                    current_model_row_data[(task_group_name, 'Average')] = round(score, 2) if pd.notna(score) else pd.NA
                    multi_index_columns_for_df.append((task_group_name, 'Average'))
            else: # It's a group with multiple sub-benchmarks
                actual_scores_for_group_avg = []
                # Use registry's canonical order for benchmarks within this group for consistent column order
                canonical_bms_in_group = registry.get_benchmarks_for_group(task_group_name)

                for bm_name in canonical_bms_in_group:
                    if bm_name in selected_bms_in_this_group: # Only display if user selected it
                        score = task_bm_scores_from_csv[task_group_name].get(bm_name, pd.NA)
                        current_model_row_data[(task_group_name, bm_name)] = round(score, 2) if pd.notna(score) else pd.NA
                        multi_index_columns_for_df.append((task_group_name, bm_name))
                        if pd.notna(score):
                            actual_scores_for_group_avg.append(score)

                # Add average for multi-benchmark groups if more than one benchmark was selected and run
                if len([bm for bm in canonical_bms_in_group if bm in selected_bms_in_this_group]) > 1:
                    # Try to get 'Average' score directly from CSV (worker should have calculated it)
                    avg_score_from_csv = task_bm_scores_from_csv[task_group_name].get('Average', pd.NA)
                    if pd.notna(avg_score_from_csv):
                        current_model_row_data[(task_group_name, 'Average')] = round(avg_score_from_csv, 2)
                    elif actual_scores_for_group_avg: # Fallback: calculate if not in CSV (should not be needed)
                        avg_score_calculated = sum(actual_scores_for_group_avg) / len(actual_scores_for_group_avg)
                        current_model_row_data[(task_group_name, 'Average')] = round(avg_score_calculated, 2)
                        logger.warning(f"Calculated average for {task_group_name} as it was missing from CSV.")
                    else:
                        current_model_row_data[(task_group_name, 'Average')] = pd.NA
                    multi_index_columns_for_df.append((task_group_name, 'Average'))

        # Ensure unique columns for the DataFrame, maintaining order as much as possible
        seen_cols, unique_multi_index_cols = set(), []
        for col_tuple in multi_index_columns_for_df:
            if col_tuple not in seen_cols:
                unique_multi_index_cols.append(col_tuple)
                seen_cols.add(col_tuple)

        if not unique_multi_index_cols or (len(unique_multi_index_cols) == 2 and unique_multi_index_cols[0][0] == 'Model' and unique_multi_index_cols[1][0] == 'Size (B)' ):
             logger.warning("No benchmark score columns to display in table for the selected model.")
             # Still print model and size if that's all
             if ('Model', '') in current_model_row_data :
                 print(f"\nModel: {current_model_row_data[('Model', '')]}, Size (B): {current_model_row_data.get(('Size (B)', ''), 'N/A')}")
                 print("No benchmark scores were found or selected for display for this model.")
             return


        df_for_display = pd.DataFrame(columns=pd.MultiIndex.from_tuples(unique_multi_index_cols))
        # Populate the DataFrame row
        row_data_for_series = {col_t: current_model_row_data.get(col_t, pd.NA) for col_t in unique_multi_index_cols}
        series_for_df_row = pd.Series(row_data_for_series, index=df_for_display.columns)

        if not series_for_df_row.empty:
            df_for_display.loc[0] = series_for_df_row
        elif unique_multi_index_cols: # If columns exist but no data, fill with NA
            df_for_display.loc[0] = pd.NA


        # --- Sorting logic for display table columns ---
        def sort_key_for_display_table(col_tuple: Tuple[str, str]):
            tg_name, bm_name = col_tuple[0], col_tuple[1]
            if tg_name == 'Model': return (0, 0, 0) # Model first
            if tg_name == 'Size (B)': return (1, 0, 0) # Size second

            try: # Order by user's task group selection order
                task_order_idx = ordered_task_groups_for_display.index(tg_name)
            except ValueError:
                task_order_idx = 9999 # Should not happen if tg_name comes from ordered_task_groups

            # Is it a single benchmark group?
            registry_bms_for_group_sort = registry.get_benchmarks_for_group(tg_name)
            is_single_bm_group_sort = len(registry_bms_for_group_sort) == 1 and registry_bms_for_group_sort[0] == tg_name

            if is_single_bm_group_sort:
                sub_order = 0 if bm_name == '' else 1 # '' (main score) before 'Average'
                return (2, task_order_idx, sub_order) # Group single benchmarks after Model/Size
            else: # Multi-benchmark group
                if bm_name == 'Average':
                    bm_order_idx = 99999 # Average last within its group
                else:
                    try:
                        # Use registry's canonical order for benchmarks within the group
                        canonical_bms_in_group_sort = registry.get_benchmarks_for_group(tg_name)
                        bm_order_idx = canonical_bms_in_group_sort.index(bm_name)
                    except (KeyError, ValueError):
                        bm_order_idx = 99998 # Should not happen if bm_name is valid
                return (3, task_order_idx, bm_order_idx) # Group multi-benchmarks last

        # Get columns from the DataFrame to sort (ensure they exist)
        cols_to_sort = [col for col in df_for_display.columns.tolist() if col in unique_multi_index_cols]
        if not cols_to_sort and not df_for_display.empty: # If unique_multi_index_cols somehow missed df columns
             cols_to_sort = df_for_display.columns.tolist()

        if cols_to_sort:
            sorted_display_columns = sorted(cols_to_sort, key=sort_key_for_display_table)
            df_for_display = df_for_display[sorted_display_columns]
        elif df_for_display.empty:
            logger.info("No results data to display in table.")
            return

        logger.info("\n--- Consolidated Evaluation Results ---")
        # Use to_markdown for better console table display
        print(df_for_display.to_markdown(index=False, floatfmt=".2f"))

    except FileNotFoundError: # Double check, already handled above
        logger.error(f"Results file '{csv_path}' not found for display.")
    except Exception as e:
        logger.error(f"Error displaying consolidated results from '{csv_path}': {e}", exc_info=True)


# --- Entry Point ---
if __name__ == "__main__":
    # This is crucial for multiprocessing on Windows, good practice elsewhere.
    multiprocessing.freeze_support()
    main_orchestrator()