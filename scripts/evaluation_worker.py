import os
import pandas as pd
import torch
import json
from datetime import datetime
import argparse
import logging
import sys
from datasets import disable_progress_bar

""" For configuring project root path"""
disable_progress_bar()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
from eka_eval.benchmarks.benchmark_registry import BenchmarkRegistry
from eka_eval.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# --- SimpleResultManager class remains the same ---
class SimpleResultManager:
    """Manages loading and saving results to CSV."""
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
        
    def load_pre_calculated_for_model(self, model_name_lower: str) -> pd.DataFrame:
        if os.path.exists(self.csv_file_path):
            try:
                temp_df = pd.read_csv(self.csv_file_path)
                if 'Model' in temp_df.columns:
                    return temp_df[temp_df['Model'].str.lower() == model_name_lower].copy()
            except Exception as e:
                logger.warning(f"Failed to load pre-calculated results from {self.csv_file_path}: {e}", exc_info=True)
        return pd.DataFrame()
        
    def save_results(self, new_results_df: pd.DataFrame):
        if new_results_df.empty:
            logger.info("No new results to save.")
            return
        try:
            if os.path.exists(self.csv_file_path):
                existing_df = pd.read_csv(self.csv_file_path)
                combined_df = pd.concat([existing_df, new_results_df]).drop_duplicates(
                    subset=['Model', 'Task', 'Benchmark'], keep='last'
                )
            else:
                combined_df = new_results_df
            combined_df.to_csv(self.csv_file_path, index=False)
            logger.info(f"Results saved/updated in '{self.csv_file_path}'.")
        except Exception as e:
            logger.error(f"Error saving results to CSV '{self.csv_file_path}': {e}", exc_info=True)

def initialize_worker_model(args):
    """Initialize model pipeline based on worker arguments."""
    is_api_model = args.is_api_model.lower() == 'true'
    
    pipeline, param_count_str = initialize_model_pipeline(
        model_name_or_path=args.model_name,
        target_device_id=0, 
        is_api_model=is_api_model,
        api_provider=args.api_provider,
        api_key=args.api_key
    )
    return pipeline, param_count_str, is_api_model
# --- run_evaluation_for_model_and_tasks function is REFACTORED ---
def run_evaluation_for_model_and_tasks(
    worker_args: argparse.Namespace,
    benchmark_registry: BenchmarkRegistry,
    result_manager: SimpleResultManager
):
    """Runs evaluation for a model on a task group and benchmarks, driven by config."""
    worker_log_id = f"P{worker_args.process_id}"
    
    model_pipeline, model_param_count_str, is_api = initialize_worker_model(worker_args)
    
    if model_pipeline is None:
        logger.error(f"{worker_log_id}: Failed to initialize model '{worker_args.model_name}'.")
        # Handle failed model load and exit
        return

    device_info = "API" if is_api else f"GPU{worker_args.gpu_id}"
    logger.info(f"{worker_log_id} ({device_info}): Initialized '{worker_args.model_name}' ({model_param_count_str}B).")
    
    selected_benchmarks_map = json.loads(worker_args.selected_benchmarks_json)
    task_group_to_evaluate = worker_args.task_group
    selected_benchmarks_for_group = selected_benchmarks_map.get(task_group_to_evaluate, [])
    
    new_results_list = []
    for bm_name in selected_benchmarks_for_group:
        logger.info(f"{worker_log_id}: Preparing to evaluate benchmark: {bm_name}")
        
        actual_eval_function = benchmark_registry.resolve_evaluation_function(task_group_to_evaluate, bm_name)
        
        if not actual_eval_function:
            logger.error(f"{worker_log_id}: Could not resolve evaluation function for {bm_name}. Skipping.")
            # Record error and continue
            continue
            
        logger.info(f"{worker_log_id}: Evaluating with function: {actual_eval_function.__name__}")
        
        try:
            # --- THIS IS THE KEY CHANGE ---
            # 1. Start with a base set of arguments that every function needs.
            eval_args = {
                "pipe": model_pipeline,
                "tokenizer": getattr(model_pipeline, 'tokenizer', None),
                "model_name_for_logging": worker_args.model_name,
                "device": getattr(model_pipeline, 'device', 'cpu'),
                "process_id": worker_args.process_id,
                "gpu_id": worker_args.gpu_id,
                "num_gpus": worker_args.num_gpus,
            }

            # 2. Get the specific `task_args` from the benchmark config.
            benchmark_details = benchmark_registry.get_benchmark_details(task_group_to_evaluate, bm_name)
            task_specific_args = benchmark_details.get("task_args", {})
            logger.info(f"Loaded {len(task_specific_args)} task-specific args from config for {bm_name}: {list(task_specific_args.keys())}")

            # 3. Merge them. The task-specific args will override any defaults if needed.
            eval_args.update(task_specific_args)
            # --- END OF KEY CHANGE ---
            
            # Now, the hardcoded overrides are GONE!
            logger.debug(f"{worker_log_id}: Final args for {actual_eval_function.__name__}: {list(eval_args.keys())}")
            
            scores_dict_from_eval = actual_eval_function(**eval_args)
            
            # ... (Result processing logic remains the same) ...
            status = "Completed"
            if isinstance(scores_dict_from_eval, dict):
                # Prefer a key that matches the benchmark name for clarity
                if bm_name in scores_dict_from_eval:
                    current_score = scores_dict_from_eval[bm_name]
                # Fallback to common metric names
                elif "accuracy" in scores_dict_from_eval:
                    current_score = scores_dict_from_eval["accuracy"]
                elif "f1" in scores_dict_from_eval:
                    current_score = scores_dict_from_eval["f1"]
                else:
                    logger.warning(f"No standard score key found for {bm_name}. Result: {scores_dict_from_eval}")
                    current_score = pd.NA
            else:
                logger.error(f"Eval function for {bm_name} did not return a dict.")
                current_score = pd.NA
                status = "InvalidReturnFormat"

        except Exception as e_eval:
            logger.error(f"{worker_log_id}: Error during evaluation of {bm_name}: {e_eval}", exc_info=True)
            status = "EvaluationError"
            current_score = pd.NA
            
        # Store result for this benchmark
        new_results_list.append({
            'Model': worker_args.model_name, 'Size (B)': model_param_count_str,
            'Task': task_group_to_evaluate, 'Benchmark': bm_name,
            'Score': current_score, 'Timestamp': datetime.now().isoformat(), 'Status': status
        })
    
    # Save all results from this worker's run
    if new_results_list:
        result_manager.save_results(pd.DataFrame(new_results_list))
    
    # Cleanup
    logger.info(f"{worker_log_id}: Cleaning up model resources for '{worker_args.model_name}'.")
    cleanup_model_resources(model_pipeline)
    
    logger.info(f"{worker_log_id}: Finished processing task group '{task_group_to_evaluate}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eka-Eval: Evaluation Worker.")
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--process_id", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_group", type=str, required=True)
    parser.add_argument("--selected_benchmarks_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--results_dir", type=str, default="results_output")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--is_api_model", type=str, default="false")
    parser.add_argument("--api_provider", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    
    args = parser.parse_args()
    
    # ... (logging and environment setup remains the same) ...
    
    worker_benchmark_registry = BenchmarkRegistry()
    full_results_path = os.path.join(args.results_dir, "calculated.csv")
    worker_result_manager = SimpleResultManager(csv_file_path=full_results_path)
    
    if not worker_benchmark_registry.benchmarks:
        logger.critical(f"W{args.process_id}: Benchmark configuration failed to load. Exiting.")
        sys.exit(1)
        
    # --- Refactored main execution call ---
    run_evaluation_for_model_and_tasks(
        worker_args=args,
        benchmark_registry=worker_benchmark_registry,
        result_manager=worker_result_manager
    )
    
    logger.info(f"Worker W{args.process_id}: Finished all assigned work for this instance.")