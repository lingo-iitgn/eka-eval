import re
import evaluate
from transformers import pipeline
from datasets import load_dataset
import string
from tqdm import tqdm
import json
import os
import sys
from typing import List, Dict, Any


def normalize_answer_quac(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def save_checkpoint_quac(checkpoint_filepath: str, predictions_so_far: list, references_so_far: list, processed_qas_ids: set):
    """Saves the current state for QuAC evaluation."""
    try:
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        data_to_save = {
            'predictions': predictions_so_far,
            'references': references_so_far,
            'processed_qas_ids': list(processed_qas_ids)
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    except Exception:
        pass # Suppress checkpoint save errors in cleaner version

def evaluate_quac(
    model_name: str, # For logging/reporting
    pipe: pipeline, # The initialized Hugging Face pipeline
    model_size_gb: float, # For logging/reporting (currently unused within)
    batch_size: int = 8, # Batch size for inference by the pipe
    dataset_split: str = "validation", # e.g., "validation"
    # Parameters for multi-processing/GPU context
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1, # Total GPUs used in the run, for data splitting
    # Checkpointing parameters
    checkpoint_dir: str = "checkpoints_quac",
    resume: bool = False,
    checkpoint_interval_batches: int = 50 # Save checkpoint every N batches
) -> Dict[str, float]:
    """
    Evaluates the given model pipeline on a subset of the QuAC dataset
    with support for multi-GPU data splitting and checkpointing.
    Returns a dictionary with 'QuAC_exact_match' and 'QuAC_f1' scores.
    """
    print(f"\n--- Starting QuAC evaluation for {model_name} (Proc {process_id}, GPU {gpu_id}) ---")
    print(f"Parameters: batch_size={batch_size}, dataset_split='{dataset_split}', resume={resume}")

    try:
        squad_metric = evaluate.load("squad") # QuAC uses SQuAD metric
    except Exception:
        print(f"Failed to load 'squad' metric. Skipping QuAC evaluation.")
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0}

    # --- Load and Split Dataset ---
    try:
        full_quac_data = load_dataset("allenai/quac", split=dataset_split, trust_remote_code=True)
        print(f"Proc {process_id}: Loaded QuAC dataset split '{dataset_split}', total size: {len(full_quac_data)}")
    except Exception:
        print(f"Proc {process_id}: Error loading QuAC dataset. Skipping.")
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0}

    total_dialogues_in_split = len(full_quac_data)
    if num_gpus > 1:
        dialogues_per_process = total_dialogues_in_split // num_gpus
        start_idx = process_id * dialogues_per_process
        end_idx = start_idx + dialogues_per_process
        if process_id == num_gpus - 1: end_idx = total_dialogues_in_split
        dataset_subset = full_quac_data.select(range(start_idx, end_idx))
        print(f"Proc {process_id}: Processing QuAC subset from index {start_idx} to {end_idx-1} ({len(dataset_subset)} dialogues).")
    else:
        dataset_subset = full_quac_data
        print(f"Proc {process_id}: Processing all {len(dataset_subset)} QuAC dialogues in the dataset split.")

    # --- Checkpointing ---
    predictions_log: List[Dict[str, str]] = [] # {'id': qas_id, 'prediction_text': str}
    references_log: List[Dict[str, Any]] = [] # {'id': qas_id, 'answers': {'text': [...], 'answer_start': [...]}}
    processed_qas_ids_from_checkpoint = set()

    checkpoint_filename = f"quac_checkpoint_proc{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    if resume and os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Resuming QuAC from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f: checkpoint_data = json.load(f)
            predictions_log = checkpoint_data.get('predictions', [])
            references_log = checkpoint_data.get('references', [])
            processed_qas_ids_from_checkpoint = set(checkpoint_data.get('processed_qas_ids', []))
            print(f"Proc {process_id}: Loaded {len(predictions_log)} preds, {len(references_log)} refs, {len(processed_qas_ids_from_checkpoint)} processed IDs from QuAC checkpoint.")
        except json.JSONDecodeError:
            print(f"Proc {process_id}: Error reading QuAC checkpoint {checkpoint_filepath}. Starting fresh.")
            predictions_log, references_log, processed_qas_ids_from_checkpoint = [], [], set()
    else:
        print(f"Proc {process_id}: No QuAC checkpoint found or not resuming. Starting fresh.")

    # --- Prepare Prompts and Ground Truths (flattened for batching) ---
    all_prompts = []
    all_qas_ids = []
    all_ground_truth_answers_texts = []
    all_ground_truth_answers_starts = []

    print(f"Proc {process_id}: Preparing QuAC turns for evaluation...")
    total_turns_in_subset = 0
    for dialogue_item in tqdm(dataset_subset, desc=f"P{process_id} - Preparing QuAC"):
        context = dialogue_item['context']
        for turn_id, question, turn_answers_texts, turn_answers_starts in zip(
            dialogue_item['turn_ids'],
            dialogue_item['questions'],
            dialogue_item['answers']['texts'],
            dialogue_item['answers']['answer_starts']
        ):
            total_turns_in_subset += 1
            if turn_id in processed_qas_ids_from_checkpoint:
                continue

            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            all_prompts.append(prompt)
            all_qas_ids.append(turn_id)
            all_ground_truth_answers_texts.append(turn_answers_texts)
            all_ground_truth_answers_starts.append(turn_answers_starts)

    if not all_prompts:
        print(f"Proc {process_id}: No new QuAC turns to process after filtering from checkpoint.")
        if predictions_log and references_log:
            norm_preds = [{'id': p['id'], 'prediction_text': normalize_answer_quac(p['prediction_text'])} for p in predictions_log]
            if norm_preds and references_log:
                final_results = squad_metric.compute(predictions=norm_preds, references=references_log)
                f1 = final_results.get('f1', 0.0)
                em = final_results.get('exact_match', 0.0)
                print(f"Proc {process_id}: QuAC scores from (resumed) checkpoint: EM={em:.2f}, F1={f1:.2f} on {len(norm_preds)} examples.")
                return {"QuAC": f1, "QuAC_exact_match": em, "QuAC_f1": f1}
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0}

    print(f"Proc {process_id}: Starting batch inference for {len(all_prompts)} QuAC turns with batch size: {batch_size}")

    # --- Generation Loop ---
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"P{process_id} - Generating QuAC Answers", file=sys.stdout):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_qas_ids = all_qas_ids[i : i + batch_size]
        batch_ground_truth_answers_texts = all_ground_truth_answers_texts[i : i + batch_size]
        batch_ground_truth_answers_starts = all_ground_truth_answers_starts[i : i + batch_size]

        try:
            batch_outputs_raw = pipe(batch_prompts, max_new_tokens=64, do_sample=False, temperature=0.1, top_p=0.9)
        except Exception:
            for qas_id_fail, gt_answers_fail, gt_starts_fail in zip(
                batch_qas_ids, batch_ground_truth_answers_texts, batch_ground_truth_answers_starts
            ):
                predictions_log.append({'id': qas_id_fail, 'prediction_text': "#PipelineError"})
                references_log.append({'id': qas_id_fail, 'answers': {'text': gt_answers_fail, 'answer_start': gt_starts_fail}})
                processed_qas_ids_from_checkpoint.add(qas_id_fail)
            continue

        for j, output_list in enumerate(batch_outputs_raw):
            qas_id = batch_qas_ids[j]
            ground_truth_answers_texts = batch_ground_truth_answers_texts[j]
            ground_truth_answers_starts = batch_ground_truth_answers_starts[j]

            predicted_raw_text = "#GenFail"
            if output_list and isinstance(output_list, list) and len(output_list) > 0 and isinstance(output_list[0], dict):
                predicted_raw_text = output_list[0].get('generated_text', "#GenFail").strip()

            predictions_log.append({'id': qas_id, 'prediction_text': predicted_raw_text})
            references_log.append({'id': qas_id, 'answers': {'text': ground_truth_answers_texts, 'answer_start': ground_truth_answers_starts}})
            processed_qas_ids_from_checkpoint.add(qas_id)

        if (i // batch_size + 1) % checkpoint_interval_batches == 0 and i > 0:
            save_checkpoint_quac(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)

    save_checkpoint_quac(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)
    print(f"Proc {process_id}: QuAC batch inference complete. Total items logged: {len(predictions_log)}")

    # --- Compute Metrics ---
    if not predictions_log or not references_log:
        print(f"Proc {process_id}: No QuAC predictions or references to compute metrics.")
        return {"QuAC": 0.0, "QuAC_exact_match": 0.0, "QuAC_f1": 0.0}

    final_predictions_for_metric = []
    for pred_item in predictions_log:
        final_predictions_for_metric.append({
            'id': pred_item['id'],
            'prediction_text': normalize_answer_quac(pred_item['prediction_text'])
        })

    final_references_for_metric = []
    for ref_item in references_log:
        final_references_for_metric.append({
            'id': ref_item['id'],
            'answers': {
                'text': [normalize_answer_quac(ans_text) for ans_text in ref_item['answers']['text']],
                'answer_start': ref_item['answers']['answer_start']
            }
        })

    exact_match_score = 0.0
    f1_score = 0.0
    try:
        eval_results = squad_metric.compute(predictions=final_predictions_for_metric, references=final_references_for_metric)
        exact_match_score = eval_results.get('exact_match', 0.0)
        f1_score = eval_results.get('f1', 0.0)
    except Exception:
        pass 

    print(f"Proc {process_id} (GPU {gpu_id}) - Final QuAC: EM={exact_match_score:.2f}, F1={f1_score:.2f} on {len(final_predictions_for_metric)} examples.")

    return {"QuAC": f1_score, "QuAC_exact_match": exact_match_score, "QuAC_f1": f1_score}