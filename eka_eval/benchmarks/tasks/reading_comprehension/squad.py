import torch
import sys
import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import string
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_SQUAD = "squad"
DEFAULT_SPLIT_SQUAD = "validation"
DEFAULT_MAX_NEW_TOKENS_SQUAD = 64
DEFAULT_CHECKPOINT_DIR_SQUAD = "checkpoints/squad_checkpoints"

def _normalize_answer_squad(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    if not isinstance(s, str): return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _save_checkpoint_squad(checkpoint_filepath: str, predictions_so_far: List[Dict], references_so_far: List[Dict], processed_qas_ids: set):
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    try:
        data_to_save = {
            'predictions': predictions_so_far,
            'references': references_so_far,
            'processed_qas_ids': list(processed_qas_ids)
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logger.info(f"SQuAD Checkpoint saved to {checkpoint_filepath}")
    except Exception as e:
        logger.error(f"Failed to save SQuAD checkpoint to {checkpoint_filepath}: {e}", exc_info=True)

def evaluate_squad(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_SQUAD,
    dataset_split: str = DEFAULT_SPLIT_SQUAD,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_SQUAD,
    generation_batch_size: int = 8,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR_SQUAD,
    resume: bool = False,
    checkpoint_save_interval_batches: int = 50,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    **kwargs
) -> Dict[str, float]:
    logger.info(f"Starting SQuAD evaluation for model: {model_name_for_logging} on dataset: {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Params: split='{dataset_split}', gen_batch_size={generation_batch_size}, checkpoint_dir='{checkpoint_dir}', resume={resume}")

    try:
        squad_metric = hf_evaluate.load("squad")
    except Exception as e:
        logger.error(f"SQuAD: Failed to load 'squad' metric: {e}", exc_info=True)
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": "MetricLoadFailed"}

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}
    
    logger.info(f"P{process_id}: Loaded SQuAD dataset '{dataset_name}' (split: '{dataset_split}') with {len(full_data_for_split)} examples for this worker's initial slice.")

    if num_gpus > 1:
        total_examples_initial_slice = len(full_data_for_split)
        examples_per_instance = total_examples_initial_slice // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1: end_idx = total_examples_initial_slice
        dataset_subset_to_process = full_data_for_split.select(range(start_idx, end_idx))
        logger.info(f"P{process_id}: Sliced for this instance. Processing {len(dataset_subset_to_process)} examples (from {start_idx} to {end_idx-1}).")
    else:
        dataset_subset_to_process = full_data_for_split

    if len(dataset_subset_to_process) == 0:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": "NoSamplesAfterSplit"}

    checkpoint_filename = f"squad_checkpoint_p{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
    predictions_log: List[Dict[str, str]] = []
    references_log: List[Dict[str, Any]] = []
    processed_qas_ids_from_checkpoint: set = set()

    if resume and os.path.exists(checkpoint_filepath):
        logger.info(f"P{process_id}: Resuming SQuAD from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f: checkpoint_data = json.load(f)
            predictions_log = checkpoint_data.get('predictions', [])
            references_log = checkpoint_data.get('references', [])
            processed_qas_ids_from_checkpoint = set(checkpoint_data.get('processed_qas_ids', []))
            logger.info(f"P{process_id}: Loaded {len(predictions_log)} preds from SQuAD checkpoint.")
        except Exception as e:
            logger.error(f"P{process_id}: Error reading SQuAD checkpoint {checkpoint_filepath}: {e}. Starting fresh.", exc_info=True)
            predictions_log, references_log, processed_qas_ids_from_checkpoint = [], [], set()

    prompts_to_generate, current_batch_info_for_processing = [], []
    for example_data in tqdm(dataset_subset_to_process, desc=f"P{process_id} - Preparing SQuAD", disable=False):
        qas_id = example_data.get('id')
        if qas_id is None:
            logger.warning("SQuAD: Example missing 'id'. Skipping.")
            continue
        if qas_id in processed_qas_ids_from_checkpoint: continue

        context = example_data.get('context', "")
        question = example_data.get('question', "")
        answers_dict = example_data.get('answers')

        if not context or not question or not answers_dict or not answers_dict.get('text'):
            logger.warning(f"SQuAD: Skipping QAS ID {qas_id} due to missing context, question, or answers text.")
            continue
        
        prompt = f"Based on the following passage, please answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
        prompts_to_generate.append(prompt)
        current_batch_info_for_processing.append({'id': qas_id, 'answers_dict': answers_dict})

    if not prompts_to_generate:
        logger.info(f"P{process_id}: No new SQuAD examples to process after filtering from checkpoint.")
        if predictions_log and references_log:
            try:
                norm_preds = [{'id': p['id'], 'prediction_text': _normalize_answer_squad(p['prediction_text'])} for p in predictions_log]
                norm_refs = [{'id': r['id'], 'answers': {'text': [_normalize_answer_squad(ans) for ans in r['answers']['text']], 'answer_start': r['answers']['answer_start']}} for r in references_log]
                if norm_preds and norm_refs:
                    final_results = squad_metric.compute(predictions=norm_preds, references=norm_refs)
                    f1, em = final_results.get('f1', 0.0), final_results.get('exact_match', 0.0)
                    return {"SQuAD": f1, "SQuAD_exact_match": em, "SQuAD_f1": f1}
            except Exception as e_metric:
                logger.error(f"P{process_id}: Error computing SQuAD metrics on resumed data: {e_metric}")
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0}

    logger.info(f"P{process_id}: Starting SQuAD batch inference for {len(prompts_to_generate)} prompts (batch_size={generation_batch_size}).")

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "return_full_text": False
    }

    for i in tqdm(range(0, len(prompts_to_generate), generation_batch_size), desc=f"P{process_id} - Generating SQuAD", unit="batch"):
        batch_prompts_slice = prompts_to_generate[i : i + generation_batch_size]
        batch_info_slice = current_batch_info_for_processing[i : i + generation_batch_size]
        
        try:
            with torch.no_grad():
                batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)

            for j, output_list_item in enumerate(batch_outputs_raw):
                info_item = batch_info_slice[j]
                qas_id = info_item['id']
                answers_dict = info_item['answers_dict']
                
                pred_text = "#GenFail"
                if output_list_item and output_list_item[0] and 'generated_text' in output_list_item[0]:
                    pred_text = output_list_item[0]['generated_text'].strip()
                
                predictions_log.append({'id': qas_id, 'prediction_text': pred_text})
                references_log.append({'id': qas_id, 'answers': answers_dict})
                processed_qas_ids_from_checkpoint.add(qas_id)

        except Exception as e_batch_gen:
            logger.error(f"P{process_id}: Error during SQuAD generation batch {i//generation_batch_size}: {e_batch_gen}", exc_info=True)
            for info_item_err in batch_info_slice:
                qas_id, answers_dict = info_item_err['id'], info_item_err['answers_dict']
                if qas_id not in processed_qas_ids_from_checkpoint:
                    predictions_log.append({'id': qas_id, 'prediction_text': "#PipelineError"})
                    references_log.append({'id': qas_id, 'answers': answers_dict})
                    processed_qas_ids_from_checkpoint.add(qas_id)

        current_batch_num = (i // generation_batch_size) + 1
        if current_batch_num % checkpoint_save_interval_batches == 0:
            _save_checkpoint_squad(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)

    if prompts_to_generate:
        _save_checkpoint_squad(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)
    
    logger.info(f"P{process_id}: SQuAD inference complete. Total items for metric: {len(predictions_log)}.")

    if not predictions_log or not references_log:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": "NoPredsOrRefsForMetric"}

    final_metric_predictions = [{'id': p['id'], 'prediction_text': _normalize_answer_squad(p['prediction_text'])} for p in predictions_log]
    final_metric_references = []
    for r_item in references_log:
        final_metric_references.append({
            'id': r_item['id'],
            'answers': {
                'text': [_normalize_answer_squad(ans_text) for ans_text in r_item['answers']['text']],
                'answer_start': r_item['answers']['answer_start']
            }
        })

    em_score, f1_score = 0.0, 0.0
    try:
        if final_metric_predictions and final_metric_references:
            squad_eval_results = squad_metric.compute(predictions=final_metric_predictions, references=final_metric_references)
            em_score = squad_eval_results.get('exact_match', 0.0)
            f1_score = squad_eval_results.get('f1', 0.0)
        else:
            logger.warning(f"P{process_id}: Not enough data for SQuAD metric computation after normalization.")
    except Exception as e_metric_final:
        logger.error(f"P{process_id}: Error computing final SQuAD metrics: {e_metric_final}", exc_info=True)
        f1_score, em_score = 0.0, 0.0

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final SQuAD: EM={em_score:.2f}%, F1={f1_score:.2f}% on {len(final_metric_predictions)} examples.")
    
    return {"SQuAD": f1_score, "SQuAD_exact_match": em_score, "SQuAD_f1": f1_score}
