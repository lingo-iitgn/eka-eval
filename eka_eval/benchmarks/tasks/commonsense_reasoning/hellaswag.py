import torch
import torch.nn.functional as F
import sys
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
DEFAULT_DATASET_NAME_HELLASWAG = "hellaswag"
DEFAULT_CONFIG_HELLASWAG = "default" # HellaSwag has a 'default' config
DEFAULT_SPLIT_HELLASWAG = "validation"

# --- Main Evaluation Function ---

def evaluate_hellaswag(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_HELLASWAG,
    dataset_config_name: str = DEFAULT_CONFIG_HELLASWAG,
    dataset_split: str = DEFAULT_SPLIT_HELLASWAG,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting HellaSwag (likelihood eval): {model_name_for_logging} on {dataset_name}/{dataset_config_name}")

    try:
        full_data = load_dataset(dataset_name, name=dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"HellaSwag": 0.0, "error_message": f"DatasetLoadFailed HellaSwag: {e}"}
    
    logger.info(f"P{process_id}: Loaded HellaSwag ({len(full_data)} examples) for split '{dataset_split}'.")

    # Handle multi-GPU data sharding
    if num_gpus > 1:
        subset_to_process = full_data.shard(num_shards=num_gpus, index=process_id)
    else:
        subset_to_process = full_data

    if len(subset_to_process) == 0:
        return {"HellaSwag": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} HellaSwag examples.")

    correct_predictions = 0
    total_evaluated = 0

    for item_data in tqdm(subset_to_process, desc=f"P{process_id} - HellaSwag Eval"):
        context = item_data.get('ctx', '')
        endings = item_data.get('endings', [])
        true_label_str = item_data.get('label', '-1')
        
        if not context or not endings or len(endings) != 4 or not true_label_str.isdigit():
            logger.warning(f"Skipping HellaSwag item with invalid data. ID: {item_data.get('ind')}")
            continue

        true_label = int(true_label_str)
        
        log_likelihoods = []
        
        # Calculate the log-likelihood of each possible completion
        for i in range(len(endings)):
            try:
                context_tokens = tokenizer(context, return_tensors="pt").to(device)
                ending_tokens = tokenizer(endings[i], return_tensors="pt").to(device)
                
                # Concatenate context and ending tokens
                full_input_ids = torch.cat([context_tokens.input_ids, ending_tokens.input_ids], dim=-1)
                
                with torch.no_grad():
                    outputs = pipe.model(full_input_ids)
                    logits = outputs.logits

                # We only care about the logits for the 'ending' part of the sequence
                # Shift logits and labels for next-token prediction loss calculation
                context_len = context_tokens.input_ids.shape[-1]
                shift_logits = logits[..., context_len-1:-1, :].contiguous()
                shift_labels = ending_tokens.input_ids.contiguous()
                
                # Calculate negative log likelihood (cross-entropy loss)
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                log_likelihood = -loss.item()
                log_likelihoods.append(log_likelihood)

            except Exception as e:
                logger.debug(f"Error calculating likelihood for a choice: {e}")
                log_likelihoods.append(float('-inf')) # Assign a very low score on error

        if log_likelihoods:
            # The prediction is the choice with the highest log-likelihood (lowest negative log-likelihood)
            predicted_label = log_likelihoods.index(max(log_likelihoods))
            if predicted_label == true_label:
                correct_predictions += 1
            
            total_evaluated += 1
    
    if total_evaluated == 0:
        logger.warning(f"P{process_id}: No valid HellaSwag examples were evaluated.")
        return {"HellaSwag": 0.0}

    accuracy = (correct_predictions / total_evaluated) * 100
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final HellaSwag Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_evaluated}).")
    
    return {"HellaSwag": accuracy}

# --- Standalone Testing Block ---
if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test HellaSwag")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    
    hs_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="HellaSwagFileTest")
    logger.info(f"--- Standalone HellaSwag Test: {hs_args.model_name_test} ---")
    
    hs_pipe, _ = initialize_model_pipeline(hs_args.model_name_test, target_device_id=0)
    if hs_pipe:
        hs_eval_args = {
            "pipe": hs_pipe,
            "tokenizer": hs_pipe.tokenizer,
            "model_name_for_logging": hs_args.model_name_test,
            "device": hs_pipe.device,
            "dataset_split": hs_args.dataset_split_test,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1
        }
        try:
            print(json.dumps(evaluate_hellaswag(**hs_eval_args), indent=2))
        finally:
            cleanup_model_resources(hs_pipe, getattr(hs_pipe, 'model', None))