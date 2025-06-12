import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import hashlib
import logging
from typing import Dict, List, Any, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "google/boolq"
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_NEW_TOKENS_BOOLQ = 10
DEFAULT_CHECKPOINT_DIR_BOOLQ = "checkpoints/boolq_checkpoints"

def _format_prompt_for_standard_boolq(passage: str, question: str) -> str:
    return f"""Answer the following question with only 'yes' or 'no' based on the provided passage.
Passage: {passage}
Question: {question}
Answer:"""

def _normalize_answer_standard_boolq(generated_text: str) -> int:
    if not isinstance(generated_text, str):
        return -1
    text_lower_stripped = generated_text.lower().strip()
    if text_lower_stripped.startswith("yes"): return 1
    if text_lower_stripped.startswith("no"): return 0
    match = re.search(r'\b(yes|no)\b', text_lower_stripped)
    if match:
        return 1 if match.group(0) == "yes" else 0
    return -1

def _save_checkpoint_boolq(checkpoint_filepath: str, predictions_so_far: List[Dict], references_so_far: List[Dict], processed_indices: set):
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    try:
        with open(checkpoint_filepath, 'w') as f:
            json.dump({
                'predictions': predictions_so_far,
                'references': references_so_far,
                'processed_indices': list(processed_indices)
            }, f, indent=4)
        logger.info(f"BoolQ Checkpoint saved to {checkpoint_filepath}")
    except Exception as e:
        logger.error(f"Failed to save BoolQ checkpoint to {checkpoint_filepath}: {e}", exc_info=True)

def evaluate_boolq(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_BOOLQ,
    generation_batch_size: int = 8,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR_BOOLQ,
    resume: bool = False,
    checkpoint_save_interval_batches: int = 50,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    **kwargs
) -> Dict[str, float]:
    logger.info(f"Starting BoolQ evaluation for model: {model_name_for_logging} on dataset: {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Params: split='{dataset_split}', gen_batch_size={generation_batch_size}, checkpoint_dir='{checkpoint_dir}', resume={resume}")

    try:
        accuracy_metric = hf_evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"BoolQ: Failed to load 'accuracy' metric: {e}", exc_info=True)
        return {"BoolQ": 0.0, "error_message": "MetricLoadFailed"}

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"P{process_id}: Loaded BoolQ dataset '{dataset_name}' (split: '{dataset_split}') with {len(full_data_for_split)} total examples for this worker's initial slice.")
    except Exception as e:
        logger.critical(f"BoolQ: Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        return {"BoolQ": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}

    if num_gpus > 1:
        total_examples_for_this_worker_slice = len(full_data_for_split)
        examples_per_process_instance = total_examples_for_this_worker_slice // num_gpus
        slice_start_idx = process_id * examples_per_process_instance
        slice_end_idx = slice_start_idx + examples_per_process_instance
        if process_id == num_gpus - 1:
            slice_end_idx = total_examples_for_this_worker_slice
        dataset_subset_to_process = full_data_for_split.select(range(slice_start_idx, slice_end_idx))
        logger.info(f"P{process_id}: Further sliced data for this instance. Processing {len(dataset_subset_to_process)} examples (from {slice_start_idx} to {slice_end_idx-1}).")
    else:
        dataset_subset_to_process = full_data_for_split
        logger.info(f"P{process_id}: Processing all {len(dataset_subset_to_process)} examples from the provided dataset split.")

    if len(dataset_subset_to_process) == 0:
        logger.warning(f"P{process_id}: No BoolQ examples to process for this instance after slicing. Returning 0.")
        return {"BoolQ": 0.0}

    checkpoint_filename = f"boolq_checkpoint_p{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    predictions_log: List[Dict] = []
    references_log: List[Dict] = []
    processed_indices_from_checkpoint: set = set()

    if resume and os.path.exists(checkpoint_filepath):
        logger.info(f"P{process_id}: Resuming from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f:
                checkpoint_data = json.load(f)
            predictions_log = checkpoint_data.get('predictions', [])
            references_log = checkpoint_data.get('references', [])
            processed_indices_from_checkpoint = set(checkpoint_data.get('processed_indices', []))
            logger.info(f"P{process_id}: Loaded {len(predictions_log)} entries from checkpoint. Skipping {len(processed_indices_from_checkpoint)} already processed examples.")
        except Exception as e:
            logger.error(f"P{process_id}: Error reading checkpoint {checkpoint_filepath}: {e}. Starting fresh.", exc_info=True)
            predictions_log, references_log, processed_indices_from_checkpoint = [], [], set()
    else:
        logger.info(f"P{process_id}: No checkpoint found at {checkpoint_filepath} or resume not requested. Starting fresh.")

    prompts_to_generate, original_indices, ground_truth_booleans = [], [], []
    for example_idx_in_subset, example_data in enumerate(dataset_subset_to_process):
        q_id = example_data.get('idx')
        if q_id is None:
            unique_str = f"{example_data.get('question','')}_{example_data.get('passage','')}_{example_idx_in_subset}"
            q_id = hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:16]
        if q_id in processed_indices_from_checkpoint:
            continue
        passage = example_data.get('passage', "")
        question = example_data.get('question', "")
        answer_bool = example_data.get('answer')
        if not passage or not question or answer_bool is None:
            continue
        prompts_to_generate.append(_format_prompt_for_standard_boolq(passage, question))
        original_indices.append(q_id)
        ground_truth_booleans.append(answer_bool)

    logger.info(f"P{process_id}: Total examples in assigned subset: {len(dataset_subset_to_process)}. Previously processed: {len(processed_indices_from_checkpoint)}. New examples to process: {len(prompts_to_generate)}.")

    if prompts_to_generate:
        logger.info(f"P{process_id}: Starting BoolQ batch inference (batch_size={generation_batch_size}).")
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "return_full_text": False
        }

        for i in tqdm(range(0, len(prompts_to_generate), generation_batch_size), desc=f"P{process_id} - BoolQ Batches", unit="batch"):
            batch_prompts_slice = prompts_to_generate[i : i + generation_batch_size]
            batch_indices_slice = original_indices[i : i + generation_batch_size]
            batch_gt_bools_slice = ground_truth_booleans[i : i + generation_batch_size]

            try:
                with torch.no_grad():
                    batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)
                for j, output_item_list in enumerate(batch_outputs_raw):
                    q_id_current = batch_indices_slice[j]
                    gt_bool_current = batch_gt_bools_slice[j]
                    generated_text_part = ""
                    if output_item_list and output_item_list[0] and 'generated_text' in output_item_list[0]:
                        generated_text_part = output_item_list[0]['generated_text'].strip()
                    predicted_normalized_val = _normalize_answer_standard_boolq(generated_text_part)
                    reference_val = 1 if gt_bool_current else 0
                    if predicted_normalized_val == -1:
                        predicted_normalized_val = 1 - reference_val
                    predictions_log.append({'idx': q_id_current, 'prediction_value': predicted_normalized_val})
                    references_log.append({'idx': q_id_current, 'reference_value': reference_val})
                    processed_indices_from_checkpoint.add(q_id_current)
                current_batch_num = (i // generation_batch_size) + 1
                if current_batch_num % checkpoint_save_interval_batches == 0:
                    _save_checkpoint_boolq(checkpoint_filepath, predictions_log, references_log, processed_indices_from_checkpoint)
            except Exception as e_batch:
                logger.error(f"P{process_id}: Error during BoolQ inference batch starting at index {i}: {e_batch}", exc_info=True)

        _save_checkpoint_boolq(checkpoint_filepath, predictions_log, references_log, processed_indices_from_checkpoint)

    logger.info(f"P{process_id}: BoolQ batch inference complete. Total processed+loaded from checkpoint: {len(predictions_log)}.")

    if not predictions_log or not references_log:
        logger.warning(f"P{process_id}: No predictions or references available for BoolQ metric computation.")
        return {"BoolQ": 0.0, "total_examples_final": 0}

    final_preds_for_metric = [item['prediction_value'] for item in predictions_log]
    final_refs_for_metric = [item['reference_value'] for item in references_log]

    if len(final_preds_for_metric) != len(final_refs_for_metric):
        logger.error(f"P{process_id}: Mismatch between number of BoolQ predictions ({len(final_preds_for_metric)}) and references ({len(final_refs_for_metric)}). Cannot compute accuracy.")
        return {"BoolQ": 0.0, "error_message": "PredictionReferenceMismatch", "total_examples_final": len(predictions_log)}

    if not final_preds_for_metric:
        logger.info(f"P{process_id}: No data points for BoolQ accuracy calculation.")
        return {"BoolQ": 0.0, "total_examples_final": 0}

    results = accuracy_metric.compute(predictions=final_preds_for_metric, references=final_refs_for_metric)
    accuracy_score = results.get('accuracy', 0.0) * 100

    logger.info(f"P{process_id}(GPU{gpu_id}) - BoolQ Accuracy: {accuracy_score:.2f}% on {len(final_preds_for_metric)} examples.")
    
    return {"BoolQ": accuracy_score, "total_examples_final": len(final_preds_for_metric)}
