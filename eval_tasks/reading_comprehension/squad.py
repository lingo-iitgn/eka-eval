import re
import evaluate
from transformers import pipeline
from datasets import load_dataset
import string
from tqdm import tqdm
import json
import os
import hashlib
import sys
from typing import List, Dict, Any

def normalize_answer_squad(s: str) -> str:
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

def save_checkpoint_squad(checkpoint_filepath: str, predictions_so_far: list, references_so_far: list, processed_qas_ids: set):
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

def evaluate_squad(
    model_name: str,
    pipe: pipeline,
    model_size_gb: float,
    batch_size: int = 8,
    dataset_split: str = "validation",
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    checkpoint_dir: str = "checkpoints_squad",
    resume: bool = False,
    checkpoint_interval_batches: int = 50
) -> Dict[str, float]:
    print(f"\n--- Starting SQuAD evaluation for {model_name} (Proc {process_id}, GPU {gpu_id}) ---")
    print(f"Parameters: batch_size={batch_size}, dataset_split='{dataset_split}', resume={resume}")

    try:
        squad_metric = evaluate.load("squad")
    except Exception:
        print(f"Failed to load 'squad' metric. Skipping SQuAD evaluation.")
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0}

    try:
        full_squad_data = load_dataset("squad", split=dataset_split, trust_remote_code=True)
        print(f"Proc {process_id}: Loaded SQuAD dataset split '{dataset_split}', total size: {len(full_squad_data)}")
    except Exception:
        print(f"Proc {process_id}: Error loading SQuAD dataset. Skipping.")
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0}

    total_examples_in_split = len(full_squad_data)
    if num_gpus > 1:
        examples_per_process = total_examples_in_split // num_gpus
        start_idx = process_id * examples_per_process
        end_idx = start_idx + examples_per_process
        if process_id == num_gpus - 1: end_idx = total_examples_in_split
        dataset_subset = full_squad_data.select(range(start_idx, end_idx))
        print(f"Proc {process_id}: Processing SQuAD subset from index {start_idx} to {end_idx-1} ({len(dataset_subset)} examples).")
    else:
        dataset_subset = full_squad_data
        print(f"Proc {process_id}: Processing all {len(dataset_subset)} SQuAD examples in the dataset split.")

    predictions_log = []
    references_log = []
    processed_qas_ids_from_checkpoint = set()

    checkpoint_filename = f"squad_checkpoint_proc{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    if resume and os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Resuming SQuAD from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f: checkpoint_data = json.load(f)
            predictions_log = checkpoint_data.get('predictions', [])
            references_log = checkpoint_data.get('references', [])
            processed_qas_ids_from_checkpoint = set(checkpoint_data.get('processed_qas_ids', []))
            print(f"Proc {process_id}: Loaded {len(predictions_log)} preds, {len(references_log)} refs, {len(processed_qas_ids_from_checkpoint)} processed IDs from SQuAD checkpoint.")
        except json.JSONDecodeError:
            print(f"Proc {process_id}: Error reading SQuAD checkpoint {checkpoint_filepath}. Starting fresh.")
            predictions_log, references_log, processed_qas_ids_from_checkpoint = [], [], set()
    else:
        print(f"Proc {process_id}: No SQuAD checkpoint found or not resuming. Starting fresh.")

    prompts_to_generate = []
    current_batch_info_map: List[Dict[str, Any]] = []

    for example_data in tqdm(dataset_subset, desc=f"P{process_id} - Preparing SQuAD", file=sys.stdout):
        qas_id = example_data.get('id')
        context = example_data.get('context', '')
        question = example_data.get('question', '')
        answers_dict = example_data.get('answers')

        if qas_id is None:
            continue
        if qas_id in processed_qas_ids_from_checkpoint: continue
        if not context or not question or not answers_dict or not answers_dict.get('text'):
            continue

        prompt = f"Based on the passage, answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
        prompts_to_generate.append(prompt)
        current_batch_info_map.append({'id': qas_id, 'answers_dict': answers_dict})

    if not prompts_to_generate:
        print(f"Proc {process_id}: No new SQuAD examples to process after filtering from checkpoint.")
        if predictions_log and references_log:
            norm_preds = [{'id': p['id'], 'prediction_text': normalize_answer_squad(p['prediction_text'])} for p in predictions_log]
            if norm_preds and references_log:
                final_results = squad_metric.compute(predictions=norm_preds, references=references_log)
                f1 = final_results.get('f1', 0.0)
                em = final_results.get('exact_match', 0.0)
                print(f"Proc {process_id}: SQuAD scores from (resumed) checkpoint: EM={em:.2f}, F1={f1:.2f} on {len(norm_preds)} examples.")
                return {"SQuAD": f1, "SQuAD_exact_match": em, "SQuAD_f1": f1}
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0}

    print(f"Proc {process_id}: Starting batch inference for {len(prompts_to_generate)} SQuAD prompts with batch size: {batch_size}")

    for i in tqdm(range(0, len(prompts_to_generate), batch_size), desc=f"P{process_id} - Generating SQuAD Answers", file=sys.stdout):
        batch_prompts = prompts_to_generate[i : i + batch_size]
        batch_map_info_list = current_batch_info_map[i : i + batch_size]

        try:
            batch_outputs_raw = pipe(batch_prompts, max_new_tokens=64, do_sample=False, temperature=0.1, top_p=0.9)
        except Exception:
            for map_info in batch_map_info_list:
                qas_id = map_info['id']; answers_dict = map_info['answers_dict']
                predictions_log.append({'id': qas_id, 'prediction_text': "#PipelineError"})
                references_log.append({'id': qas_id, 'answers': answers_dict})
                processed_qas_ids_from_checkpoint.add(qas_id)
            continue

        for j, output_list in enumerate(batch_outputs_raw):
            map_info = batch_map_info_list[j]
            qas_id = map_info['id']
            ground_truth_answers_dict = map_info['answers_dict']

            predicted_raw_text = "#GenFail"
            if output_list and isinstance(output_list, list) and len(output_list) > 0 and isinstance(output_list[0], dict):
                predicted_raw_text = output_list[0].get('generated_text', "#GenFail").strip()

            predictions_log.append({'id': qas_id, 'prediction_text': predicted_raw_text})
            references_log.append({'id': qas_id, 'answers': ground_truth_answers_dict})
            processed_qas_ids_from_checkpoint.add(qas_id)

        if (i // batch_size + 1) % checkpoint_interval_batches == 0 and i > 0:
            save_checkpoint_squad(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)

    save_checkpoint_squad(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)
    print(f"Proc {process_id}: SQuAD batch inference complete. Total items logged: {len(predictions_log)}")

    if not predictions_log or not references_log:
        print(f"Proc {process_id}: No SQuAD predictions or references to compute metrics.")
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0}

    final_predictions_for_metric = []
    for pred_item in predictions_log:
        final_predictions_for_metric.append({
            'id': pred_item['id'],
            'prediction_text': normalize_answer_squad(pred_item['prediction_text'])
        })

    final_references_for_metric = []
    for ref_item in references_log:
        final_references_for_metric.append({
            'id': ref_item['id'],
            'answers': {
                'text': [normalize_answer_squad(ans_text) for ans_text in ref_item['answers']['text']],
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

    print(f"Proc {process_id} (GPU {gpu_id}) - Final SQuAD: EM={exact_match_score:.2f}, F1={f1_score:.2f} on {len(final_predictions_for_metric)} examples.")

    return {"SQuAD": f1_score, "SQuAD_exact_match": exact_match_score, "SQuAD_f1": f1_score}