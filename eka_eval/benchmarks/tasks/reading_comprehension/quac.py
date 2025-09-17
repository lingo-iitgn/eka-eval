# eka_eval/benchmarks/tasks/reading_comprehension/quac.py

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
import argparse
import string
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_QUAC = "allenai/quac"
DEFAULT_SPLIT_QUAC = "validation"
DEFAULT_MAX_NEW_TOKENS_QUAC = 64 
DEFAULT_CHECKPOINT_DIR_QUAC = "checkpoints/quac_checkpoints"

def _normalize_answer_quac(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    if not isinstance(s, str): return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _save_checkpoint_quac(checkpoint_filepath: str, predictions_so_far: List[Dict], references_so_far: List[Dict], processed_qas_ids: set):
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    try:
        data_to_save = {
            'predictions': predictions_so_far, 
            'references': references_so_far,   
            'processed_qas_ids': list(processed_qas_ids)
        }
        with open(checkpoint_filepath, 'w') as f: json.dump(data_to_save, f, indent=4)
        logger.info(f"QuAC Checkpoint saved to {checkpoint_filepath}")
    except Exception as e:
        logger.error(f"Failed to save QuAC checkpoint: {e}", exc_info=True)

# --- Main Evaluation Function ---
def evaluate_quac(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_QUAC,
    dataset_split: str = DEFAULT_SPLIT_QUAC,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_QUAC,
    generation_batch_size: int = 8,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR_QUAC,
    resume: bool = False, checkpoint_save_interval_batches: int = 50,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:
    """Evaluates the model on the QuAC benchmark."""
    logger.info(f"Starting QuAC: {model_name_for_logging} on {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Params: split='{dataset_split}', gen_batch_size={generation_batch_size}, ckpt='{checkpoint_dir}', resume={resume}")

    try:
        squad_metric_for_quac = hf_evaluate.load("squad") # QuAC uses SQuAD F1/EM
    except Exception as e:
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0, "error_message": f"MetricLoadFailed: {e}"}

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name} - {e}"}
    
    logger.info(f"P{process_id}: Loaded QuAC '{dataset_name}' (split: '{dataset_split}') with {len(full_data_for_split)} dialogues.")

    if num_gpus > 1:
        total_dialogues = len(full_data_for_split)
        dialogues_per_instance = total_dialogues // num_gpus
        start_idx = process_id * dialogues_per_instance
        end_idx = start_idx + dialogues_per_instance
        if process_id == num_gpus - 1: end_idx = total_dialogues
        dataset_subset_dialogues = full_data_for_split.select(range(start_idx, end_idx))
    else:
        dataset_subset_dialogues = full_data_for_split
    
    if len(dataset_subset_dialogues) == 0: return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0, "error_message": "NoDialoguesAfterSplit"}

    checkpoint_filename = f"quac_checkpoint_p{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
    predictions_log, references_log, processed_qas_ids = [], [], set()

    if resume and os.path.exists(checkpoint_filepath):
        logger.info(f"P{process_id}: Resuming QuAC from {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f: ckpt_data = json.load(f)
            predictions_log, references_log, processed_qas_ids = ckpt_data.get('predictions',[]), ckpt_data.get('references',[]), set(ckpt_data.get('processed_qas_ids',[]))
            logger.info(f"P{process_id}: Loaded {len(predictions_log)} preds from QuAC checkpoint.")
        except Exception as e: logger.error(f"P{process_id}: Error reading QuAC ckpt: {e}. Starting fresh.", exc_info=True)

    # Flatten dialogues into individual question-answer turns
    all_prompts, all_turn_infos_for_processing = [], []
    for dialogue in tqdm(dataset_subset_dialogues, desc=f"P{process_id} - Preparing QuAC Turns"):
        context = dialogue.get('context', "")
        for turn_idx_in_dialogue in range(len(dialogue.get('questions', []))):
            turn_id = dialogue['turn_ids'][turn_idx_in_dialogue] # QuAC has 'turn_ids'
            if turn_id in processed_qas_ids: continue

            question = dialogue['questions'][turn_idx_in_dialogue]
            answer_texts_for_turn = dialogue['answers']['texts'][turn_idx_in_dialogue]
            answer_starts_for_turn = dialogue['answers']['answer_starts'][turn_idx_in_dialogue]

            if not context or not question or not answer_texts_for_turn: continue
            
            prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            all_prompts.append(prompt)
            all_turn_infos_for_processing.append({
                'id': turn_id,
                'answers_dict': {'text': answer_texts_for_turn, 'answer_start': answer_starts_for_turn}
            })

    if not all_prompts: 
        logger.info(f"P{process_id}: No new QuAC turns to process.")
        if predictions_log and references_log:
            try:
                norm_preds = [{'id': p['id'], 'prediction_text': _normalize_answer_quac(p['prediction_text'])} for p in predictions_log]
                norm_refs = [{'id': r['id'], 'answers': {'text': [_normalize_answer_quac(ans) for ans in r['answers']['text']], 'answer_start': r['answers']['answer_start']}} for r in references_log]
                if norm_preds and norm_refs:
                    final_results = squad_metric_for_quac.compute(predictions=norm_preds, references=norm_refs)
                    f1, em = final_results.get('f1', 0.0), final_results.get('exact_match', 0.0)
                    return {"QuAC": f1, "QuAC_exact_match": em, "QuAC_f1": f1}
            except Exception as e_m: logger.error(f"P{process_id}: Error computing QuAC metrics on resumed data: {e_m}")
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0}

    logger.info(f"P{process_id}: Starting QuAC batch inference for {len(all_prompts)} turns (batch_size={generation_batch_size}).")
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, "return_full_text": False}

    for i in tqdm(range(0, len(all_prompts), generation_batch_size), desc=f"P{process_id} - Generating QuAC", unit="batch"):
        batch_prompts_slice = all_prompts[i : i + generation_batch_size]
        batch_info_slice = all_turn_infos_for_processing[i : i + generation_batch_size]
        try:
            with torch.no_grad(): batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)
            for j, out_list in enumerate(batch_outputs_raw):
                info, qas_id = batch_info_slice[j], batch_info_slice[j]['id']
                pred_txt = out_list[0]['generated_text'].strip() if out_list and out_list[0] else "#GenFail"
                predictions_log.append({'id': qas_id, 'prediction_text': pred_txt})
                references_log.append({'id': qas_id, 'answers': info['answers_dict']})
                processed_qas_ids.add(qas_id)
        except Exception as e_b:
            logger.error(f"P{process_id}: Error in QuAC gen batch {i//generation_batch_size}: {e_b}", exc_info=True)
            for info_err in batch_info_slice:
                if info_err['id'] not in processed_qas_ids:
                    predictions_log.append({'id': info_err['id'], 'prediction_text': "#PipelineError"})
                    references_log.append({'id': info_err['id'], 'answers': info_err['answers_dict']})
                    processed_qas_ids.add(info_err['id'])
        if ((i // generation_batch_size) + 1) % checkpoint_save_interval_batches == 0:
            _save_checkpoint_quac(checkpoint_filepath, predictions_log, references_log, processed_qas_ids)
    if all_prompts: _save_checkpoint_quac(checkpoint_filepath, predictions_log, references_log, processed_qas_ids)

    if not predictions_log: return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0, "error_message": "NoPredsForMetric"}
    
    metric_preds = [{'id': p['id'], 'prediction_text': _normalize_answer_quac(p['prediction_text'])} for p in predictions_log]
    metric_refs = [{'id': r['id'], 'answers': {'text': [_normalize_answer_quac(ans) for ans in r['answers']['text']], 'answer_start': r['answers']['answer_start']}} for r in references_log]
    em, f1 = 0.0, 0.0
    try:
        if metric_preds and metric_refs:
            res = squad_metric_for_quac.compute(predictions=metric_preds, references=metric_refs)
            em, f1 = res.get('exact_match', 0.0), res.get('f1', 0.0)
    except Exception as e_m_final: logger.error(f"P{process_id}: Error computing final QuAC metrics: {e_m_final}", exc_info=True)
    
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final QuAC: EM={em:.2f}%, F1={f1:.2f}% on {len(metric_preds)} turns.")
    return {"QuAC": f1, "QuAC_exact_match": em, "QuAC_f1": f1}

