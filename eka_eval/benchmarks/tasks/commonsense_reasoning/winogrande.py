import torch
import sys
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import logging
from typing import Dict, List, Any
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_WINO = "winogrande"
DEFAULT_CONFIG_WINO = "winogrande_xl" 
DEFAULT_SPLIT_WINO = "validation"
DEFAULT_CHECKPOINT_DIR_WINO = "checkpoints/winogrande_checkpoints"

try:
    wino_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for Winogrande loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for Winogrande: {e}. Winogrande will not run correctly.", exc_info=True)
    wino_accuracy_metric = None

def _format_winogrande_choices(item: Dict) -> List[str]:
    """Format winogrande choices like lm-eval-harness."""
    sentence = item.get('sentence', '')
    idx = sentence.index("_")
    option1 = item.get('option1', '')
    option2 = item.get('option2', '')
    
    # Create two complete sentences with options filled in
    choice1 = sentence[:idx] + option1 + sentence[idx+1:]
    choice2 = sentence[:idx] + option2 + sentence[idx+1:]
    
    return [choice1, choice2]

def evaluate_winogrande(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_WINO,
    dataset_config_name: str = DEFAULT_CONFIG_WINO,
    dataset_split: str = DEFAULT_SPLIT_WINO,
    generation_batch_size: int = 8,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if wino_accuracy_metric is None:
        return {"WinoGrande": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting Winogrande: {model_name_for_logging} on {dataset_name}/{dataset_config_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Using log-likelihood scoring (multiple choice)")

    try:
        full_data_for_split = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"WinoGrande": 0.0, "error_message": f"DatasetLoadFailed Wino: {e}"}
    
    logger.info(f"P{process_id}: Loaded Winogrande '{dataset_name}/{dataset_config_name}' (split: '{dataset_split}') with {len(full_data_for_split)} examples.")

    if num_gpus > 1:
        total_examples = len(full_data_for_split)
        examples_per_instance = total_examples // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1: end_idx = total_examples
        dataset_subset_to_process = full_data_for_split.select(range(start_idx, end_idx))
    else:
        dataset_subset_to_process = full_data_for_split
        
    if len(dataset_subset_to_process) == 0: return {"WinoGrande": 0.0}

    model = pipe.model
    predictions, references = [], []
    
    for item in tqdm(dataset_subset_to_process, desc=f"P{process_id} - Wino Eval"):
        answer_str = item.get('answer', '')
        if answer_str not in ['1', '2']:
            logger.warning(f"P{process_id}: Skipping item with invalid answer '{answer_str}'")
            continue

        choices = _format_winogrande_choices(item)
        answer_idx = int(answer_str) - 1  # Convert 1/2 to 0/1
        
        # Compute log-likelihood for each choice
        log_likelihoods = []
        for choice_text in choices:
            try:
                inputs = tokenizer(choice_text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
                    log_likelihoods.append(log_likelihood)
            except Exception as e:
                logger.debug(f"Error computing log-likelihood: {e}")
                log_likelihoods.append(float('-inf'))
        
        # Predict the choice with higher log-likelihood
        pred_idx = 0 if log_likelihoods[0] > log_likelihoods[1] else 1
        predictions.append(pred_idx)
        references.append(answer_idx)

    if not references: return {"WinoGrande": 0.0}
    
    accuracy_score = 0.0
    try:
        accuracy_results = wino_accuracy_metric.compute(predictions=predictions, references=references)
        accuracy_score = accuracy_results.get("accuracy", 0.0) * 100
    except Exception as e_metric: 
        logger.error(f"P{process_id}: Error computing Wino accuracy: {e_metric}", exc_info=True)

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final Winogrande Accuracy: {accuracy_score:.2f}% on {len(references)} examples.")
    return {"WinoGrande": accuracy_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser_wino = argparse.ArgumentParser(description="Standalone Test Winogrande")
    test_parser_wino.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser_wino.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser_wino.add_argument("--dataset_config_test", type=str, default="winogrande_xs")
    test_parser_wino.add_argument("--gen_batch_size_test", type=int, default=2)
    w_args = test_parser_wino.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="WinoFileTest")
    logger.info(f"--- Standalone Winogrande Test: {w_args.model_name_test} ---")
    w_pipe, _ = initialize_model_pipeline(w_args.model_name_test, target_device_id=0)
    if w_pipe:
        w_eval_args = {
            "pipe": w_pipe, "tokenizer": w_pipe.tokenizer, "model_name_for_logging": w_args.model_name_test,
            "device": w_pipe.device, "dataset_config_name": w_args.dataset_config_test,
            "dataset_split": w_args.dataset_split_test, "generation_batch_size": w_args.gen_batch_size_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_winogrande(**w_eval_args), indent=2))
        finally: cleanup_model_resources(w_pipe, getattr(w_pipe, 'model', None))
    else: logger.error(f"Failed to init model {w_args.model_name_test} for Winogrande test.")