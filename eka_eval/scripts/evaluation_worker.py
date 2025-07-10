import os
import pandas as pd
import torch
import json
from datetime import datetime
import argparse
import logging
import sys
import os  
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
    """Initialize model pipeline based on whether it's local or API"""
    is_api_model = args.is_api_model.lower() == "true"
    
    if is_api_model:
        logger.info(f"Initializing API model: {args.api_provider}/{args.model_name}")
        pipeline, param_count_str = initialize_model_pipeline(
            model_name_or_path=args.model_name,
            target_device_id=args.gpu_id,
            is_api_model=True,
            api_provider=args.api_provider,
            api_key=args.api_key
        )
        # For API models, set device attribute for compatibility
        if pipeline and not hasattr(pipeline, 'device'):
            pipeline.device = "api"
    else:
        logger.info(f"Initializing local model: {args.model_name}")
        pipeline, param_count_str = initialize_model_pipeline(
            model_name_or_path=args.model_name,
            target_device_id=args.gpu_id,
            is_api_model=False
        )
    
    return pipeline, param_count_str, is_api_model

def run_evaluation_for_model_and_tasks(
    input_model_name: str,
    task_group_to_evaluate: str,
    selected_benchmarks_for_group: list,
    process_id: int,
    physical_gpu_id: int,
    num_total_workers: int,
    orchestrator_batch_size: int,
    benchmark_registry: BenchmarkRegistry,
    result_manager: SimpleResultManager,
    is_api_model: bool = False,
    api_provider: str = None,
    api_key: str = None
):
    """Runs evaluation for a model on a task group and benchmarks."""
    worker_log_id = f"P{process_id}"
    model_type_str = f"{api_provider} API" if is_api_model else "Local"
    logger.info(f"{worker_log_id}: Starting evaluation for {model_type_str} model '{input_model_name}' on task group '{task_group_to_evaluate}'.")
    logger.info(f"{worker_log_id}: Benchmarks to evaluate in this group: {selected_benchmarks_for_group}")
    
    pre_calculated_df = result_manager.load_pre_calculated_for_model(input_model_name.lower())
    
    # Create args object for initialize_worker_model
    class Args:
        def __init__(self):
            self.model_name = input_model_name
            self.gpu_id = physical_gpu_id
            self.is_api_model = "true" if is_api_model else "false"
            self.api_provider = api_provider or ""
            self.api_key = api_key or ""
    
    args = Args()
    model_pipeline, model_param_count_str, is_api = initialize_worker_model(args)
    
    if model_pipeline is None:
        logger.error(f"{worker_log_id}: Failed to initialize {model_type_str} model '{input_model_name}'. Skipping this task group.")
        failed_results = []
        for bm_name in selected_benchmarks_for_group:
            failed_results.append({
                'Model': input_model_name, 'Size (B)': 'N/A', 'Task': task_group_to_evaluate,
                'Benchmark': bm_name, 'Score': pd.NA, 'Timestamp': datetime.now().isoformat(),
                'Status': 'ModelLoadFailed'
            })
        if failed_results:
             result_manager.save_results(pd.DataFrame(failed_results))
        return
    
    device_info = "API" if is_api else f"GPU{physical_gpu_id}"
    logger.info(f"{worker_log_id} ({device_info}): Initialized '{input_model_name}' ({model_param_count_str}B). Effective BatchSize: {orchestrator_batch_size}")
    
    new_results_list = []
    scores_for_group_average = []
    group_benchmarks_from_registry = benchmark_registry.get_benchmarks_for_group(task_group_to_evaluate)
    is_single_benchmark_task_group = (
        len(group_benchmarks_from_registry) == 1 and
        group_benchmarks_from_registry[0] == task_group_to_evaluate
    )
    
    for bm_name in selected_benchmarks_for_group:
        current_score = pd.NA
        status = "Pending"
        
        # Check for pre-calculated results
        if not pre_calculated_df.empty:
            pre_calc_entry = pre_calculated_df[
                (pre_calculated_df['Task'].str.lower() == task_group_to_evaluate.lower()) &
                (pre_calculated_df['Benchmark'].str.lower() == bm_name.lower()) &
                (pre_calculated_df['Score'].notna())
            ]
            if not pre_calc_entry.empty:
                current_score = pre_calc_entry['Score'].iloc[0]
                logger.info(f"{worker_log_id}: Using pre-calculated score for {task_group_to_evaluate}-{bm_name}: {current_score}")
                if pd.notna(current_score):
                    scores_for_group_average.append(float(current_score))
                new_results_list.append({
                    'Model': input_model_name, 'Size (B)': model_param_count_str,
                    'Task': task_group_to_evaluate, 'Benchmark': bm_name,
                    'Score': current_score, 'Timestamp': datetime.now().isoformat(),
                    'Status': 'PreCalculated'
                })
                continue
        
        logger.info(f"{worker_log_id}: Attempting to resolve evaluation function for {task_group_to_evaluate}-{bm_name}.")
        actual_eval_function = benchmark_registry.resolve_evaluation_function(task_group_to_evaluate, bm_name)
        
        if not actual_eval_function:
            logger.error(f"{worker_log_id}: Could not resolve evaluation function for {task_group_to_evaluate}-{bm_name}. Skipping.")
            status = "EvalFunctionNotFound"
        else:
            logger.info(f"{worker_log_id}: Evaluating {task_group_to_evaluate}-{bm_name} using function: {actual_eval_function.__name__}")
            try:
                # Prepare evaluation arguments
                eval_args = {
                    "pipe": model_pipeline,
                    "tokenizer": model_pipeline.tokenizer,
                    "model_name_for_logging": input_model_name,
                    "device": model_pipeline.device,
                    "process_id": process_id,
                    "gpu_id": physical_gpu_id,
                    "num_gpus": num_total_workers,
                    "batch_size": orchestrator_batch_size,
                }
                
                # Function-specific configurations
                fn_name_for_switch = actual_eval_function.__name__
                if fn_name_for_switch == "evaluate_mbpp":
                    eval_args.update({"num_samples_per_task": 1, "dataset_split": "test[:5]", "include_tests_in_prompt": False})
                    # For API models, use smaller batch sizes to avoid rate limits
                    eval_args["batch_size"] = 1 if is_api else 1
                elif fn_name_for_switch == "evaluate_humaneval":
                    eval_args.update({"num_samples_per_task": 1, "dataset_split": "test", "k_values": [1], "use_fewshot": True})
                    eval_args["batch_size"] = 1 if is_api else 1
                elif fn_name_for_switch == "evaluate_boolq":
                    cp_dir = os.path.join("checkpoints", "boolq_checkpoints"); os.makedirs(cp_dir, exist_ok=True)
                    eval_args.update({"dataset_split": "validation[:20]", "checkpoint_dir": cp_dir, "resume": True})
                    # For API models, use smaller dataset splits to manage costs
                    if is_api:
                        eval_args["dataset_split"] = "validation[:10]"
                elif fn_name_for_switch == "evaluate_squad":
                    cp_dir_squad = os.path.join("checkpoints", "squad_checkpoints"); os.makedirs(cp_dir_squad, exist_ok=True)
                    eval_args.update({"dataset_split": "validation[:10]", "checkpoint_dir": cp_dir_squad, "resume": True})
                    if is_api:
                        eval_args["dataset_split"] = "validation[:5]"
                elif fn_name_for_switch == "evaluate_quac":
                    cp_dir_quac = os.path.join("checkpoints", "quac_checkpoints"); os.makedirs(cp_dir_quac, exist_ok=True)
                    eval_args.update({"dataset_split": "validation[:10]", "checkpoint_dir": cp_dir_quac, "resume": True})
                    if is_api:
                        eval_args["dataset_split"] = "validation[:5]"
                elif fn_name_for_switch == "evaluate_gsm8k":
                    eval_args.update({"dataset_split": "test[:10]", "resume": True})
                    if is_api:
                        eval_args["dataset_split"] = "test[:5]"
                elif fn_name_for_switch == "evaluate_math":
                    eval_args.update({"dataset_split": "test[:10]", "resume": True})
                    if is_api:
                        eval_args["dataset_split"] = "test[:5]"
                
                logger.debug(f"{worker_log_id}: Calling {fn_name_for_switch} with args: {{k: type(v) for k,v in eval_args.items()}}")
                
                scores_dict_from_eval = {}
                try:
                    scores_dict_from_eval = actual_eval_function(**eval_args)
                except TypeError as te:
                    functions_with_own_batch_handling = [
                        "evaluate_mbpp", "evaluate_humaneval", "evaluate_boolq",
                        "evaluate_squad", "evaluate_quac", "evaluate_gsm8k", "evaluate_math"
                    ]
                    should_retry_without_batch_size = "batch_size" in str(te).lower() and \
                                                      fn_name_for_switch not in functions_with_own_batch_handling
                    if should_retry_without_batch_size:
                        logger.warning(f"{worker_log_id}: {fn_name_for_switch} got TypeError related to 'batch_size'. Retrying without it.")
                        eval_args.pop("batch_size", None)
                        scores_dict_from_eval = actual_eval_function(**eval_args)
                    else:
                        logger.error(f"{worker_log_id}: TypeError for {fn_name_for_switch} not handled by retry. Error: {te}", exc_info=True)
                        raise
                
                # Process evaluation results
                if isinstance(scores_dict_from_eval, dict):
                    if bm_name in scores_dict_from_eval:
                        current_score = scores_dict_from_eval[bm_name]
                    elif "accuracy" in scores_dict_from_eval:
                        current_score = scores_dict_from_eval["accuracy"]
                    elif "f1" in scores_dict_from_eval:
                        current_score = scores_dict_from_eval["f1"]
                    else:
                        logger.warning(f"{worker_log_id}: No standard score key found in result from {fn_name_for_switch}. Result: {scores_dict_from_eval}")
                        current_score = pd.NA
                    status = "Completed"
                else:
                    logger.error(f"{worker_log_id}: Evaluation function {fn_name_for_switch} did not return a dictionary. Got: {type(scores_dict_from_eval)}")
                    status = "InvalidReturnFormat"
                    
            except Exception as e_eval:
                logger.error(f"{worker_log_id}: Error during evaluation of {task_group_to_evaluate}-{bm_name} with {actual_eval_function.__name__}: {e_eval}", exc_info=True)
                status = "EvaluationError"
                current_score = pd.NA
        
        # Store results
        new_results_list.append({
            'Model': input_model_name, 'Size (B)': model_param_count_str,
            'Task': task_group_to_evaluate, 'Benchmark': bm_name,
            'Score': current_score if pd.notna(current_score) else pd.NA,
            'Timestamp': datetime.now().isoformat(),
            'Status': status
        })
        
        if pd.notna(current_score) and status == "Completed":
            try:
                scores_for_group_average.append(float(current_score))
            except ValueError:
                logger.warning(f"{worker_log_id}: Score '{current_score}' for {bm_name} is not a valid float, not included in average.")
    
    # Calculate group average if applicable
    if not is_single_benchmark_task_group and len(scores_for_group_average) > 0:
        if len(scores_for_group_average) > 1:
            group_average_score = sum(scores_for_group_average) / len(scores_for_group_average)
            logger.info(f"{worker_log_id}: Average score for task group '{task_group_to_evaluate}' ({len(scores_for_group_average)} scores): {group_average_score:.2f}")
            new_results_list.append({
                'Model': input_model_name, 'Size (B)': model_param_count_str,
                'Task': task_group_to_evaluate, 'Benchmark': 'Average',
                'Score': group_average_score, 'Timestamp': datetime.now().isoformat(),
                'Status': 'Aggregated'
            })
        elif scores_for_group_average:
             logger.info(f"{worker_log_id}: Only one score ({scores_for_group_average[0]:.2f}) for task group '{task_group_to_evaluate}'. No 'Average' row added.")
    
    # Save results
    if new_results_list:
        new_results_df = pd.DataFrame(new_results_list)
        result_manager.save_results(new_results_df)
    else:
        logger.info(f"{worker_log_id}: No new results generated for model '{input_model_name}', task group '{task_group_to_evaluate}'.")
    
    # Cleanup
    logger.info(f"{worker_log_id}: Cleaning up model resources for '{input_model_name}'.")
    if is_api:
        # API models have different cleanup
        cleanup_model_resources(model_pipeline, model_ref=None)
    else:
        # Local models
        model_instance_ref = model_pipeline.model if hasattr(model_pipeline, 'model') else None
        cleanup_model_resources(model_pipeline, model_ref=model_instance_ref)
    
    logger.info(f"{worker_log_id}: Finished processing task group '{task_group_to_evaluate}' for model '{input_model_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eka-Eval: Evaluation Worker.")
    parser.add_argument("--gpu_id", type=int, required=True, help="Physical GPU ID assigned to this worker (for CUDA_VISIBLE_DEVICES). Use -1 for CPU/API.")
    parser.add_argument("--num_gpus", type=int, required=True, help="Total number of GPU workers in this run.")
    parser.add_argument("--process_id", type=int, required=True, help="Unique logical ID for this worker process.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to evaluate.")
    parser.add_argument("--task_group", type=str, required=True, help="Single task group for this worker to evaluate.")
    parser.add_argument("--selected_benchmarks_json", type=str, required=True,
                        help="JSON string mapping: {task_group: [list_of_benchmarks_for_this_group]}.")
    parser.add_argument("--batch_size", type=int, default=1, help="Default batch size suggested by orchestrator.")
    parser.add_argument("--results_dir", type=str, default="results_output", help="Directory where results CSV is stored.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level for this worker.")
    
    # API model specific arguments
    parser.add_argument("--is_api_model", type=str, default="false", help="Whether this is an API model")
    parser.add_argument("--api_provider", type=str, default="", help="API provider (openai, gemini, claude)")
    parser.add_argument("--api_key", type=str, default="", help="API key for authentication")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    setup_logging(level=log_level_map.get(args.log_level.upper(), logging.INFO), worker_id=f"W{args.process_id}")
    
    logger.info(f"Worker W{args.process_id} starting with model: {args.model_name}")
    
    is_api_model = args.is_api_model.lower() == "true"
    
    # Setup CUDA environment for local models
    if not is_api_model:
        if torch.cuda.is_available():
            if args.gpu_id >= 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
                logger.info(f"W{args.process_id}: Set CUDA_VISIBLE_DEVICES={args.gpu_id}. This worker will see GPU {args.gpu_id} as logical cuda:0.")
            else:
                logger.info(f"W{args.process_id}: gpu_id is {args.gpu_id}, running on CPU (CUDA_VISIBLE_DEVICES not set).")
        else:
            logger.info(f"W{args.process_id}: CUDA not available. Running on CPU.")
    else:
        logger.info(f"W{args.process_id}: Running API model ({args.api_provider}). No GPU setup needed.")
    
    # Initialize benchmark registry
    worker_benchmark_registry = BenchmarkRegistry()
    results_csv_filename = "calculated.csv"
    full_results_path = os.path.join(args.results_dir, results_csv_filename)
    worker_result_manager = SimpleResultManager(csv_file_path=full_results_path)
    
    if not worker_benchmark_registry.benchmarks:
        logger.critical(f"W{args.process_id}: Benchmark configuration failed to load in worker. Exiting.")
        sys.exit(1)
    
    # Parse selected benchmarks
    try:
        selected_benchmarks_map = json.loads(args.selected_benchmarks_json)
        benchmarks_for_this_worker_in_group = selected_benchmarks_map.get(args.task_group)
    except json.JSONDecodeError as e:
        logger.critical(f"W{args.process_id}: FATAL - Failed to decode JSON for selected benchmarks: {e}. JSON string: '{args.selected_benchmarks_json}'", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"W{args.process_id}: FATAL - Error processing selected_benchmarks_json: {e}. JSON string: '{args.selected_benchmarks_json}'", exc_info=True)
        sys.exit(1)
    
    # Validate benchmarks
    if benchmarks_for_this_worker_in_group is None:
        logger.error(f"W{args.process_id}: No specific benchmarks found for task group '{args.task_group}' in the provided JSON map. Map: {selected_benchmarks_map}. Skipping.")
    elif not isinstance(benchmarks_for_this_worker_in_group, list) or not benchmarks_for_this_worker_in_group:
        logger.error(f"W{args.process_id}: Invalid or empty benchmark list for '{args.task_group}'. List: {benchmarks_for_this_worker_in_group}. Skipping.")
    else:
        device_info = f"{args.api_provider} API" if is_api_model else ("CPU" if args.gpu_id < 0 else f"GPU {args.gpu_id}")
        logger.info(f"W{args.process_id}: Will evaluate model '{args.model_name}' on Task Group: '{args.task_group}', "
                    f"Specific Benchmarks: {benchmarks_for_this_worker_in_group} "
                    f"on {device_info}.")
        
        # Run the evaluation
        run_evaluation_for_model_and_tasks(
            input_model_name=args.model_name,
            task_group_to_evaluate=args.task_group,
            selected_benchmarks_for_group=benchmarks_for_this_worker_in_group,
            process_id=args.process_id,
            physical_gpu_id=args.gpu_id,
            num_total_workers=args.num_gpus,
            orchestrator_batch_size=args.batch_size,
            benchmark_registry=worker_benchmark_registry,
            result_manager=worker_result_manager,
            is_api_model=is_api_model,
            api_provider=args.api_provider,
            api_key=args.api_key
        )
    
    logger.info(f"Worker W{args.process_id}: Finished all assigned work for this instance.")