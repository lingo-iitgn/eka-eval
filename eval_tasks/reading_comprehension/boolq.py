import torch
import re
import evaluate
from transformers import pipeline
from datasets import load_dataset
import string
from tqdm import tqdm
import json
import os
from typing import List, Dict
import hashlib
import sys

def normalize_answer_boolq(s: str) -> str:
    s = str(s).lower().strip()
    if "yes" in s: return "yes"
    if "no" in s: return "no"
    return s

def save_checkpoint_boolq(checkpoint_filepath: str, predictions_so_far: list, references_so_far: list, processed_indices: set):
    try:
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        data_to_save = {
            'predictions': predictions_so_far,
            'references': references_so_far,
            'processed_indices': list(processed_indices)
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    except Exception:
        pass 

def evaluate_boolq(
    model_name: str,
    pipe: pipeline,
    model_size_gb: float,
    batch_size: int = 8,
    dataset_split: str = "validation",
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    checkpoint_dir: str = "checkpoints_boolq",
    resume: bool = False,
    checkpoint_interval_batches: int = 50
) -> Dict[str, float]:

    print(f"\n--- Starting BoolQ evaluation for {model_name} (Proc {process_id}, GPU {gpu_id}) ---")
    print(f"Parameters: batch_size={batch_size}, dataset_split='{dataset_split}', resume={resume}")

    try:
        accuracy_metric = evaluate.load("accuracy")
    except Exception:
        print(f"Failed to load 'accuracy' metric. Skipping BoolQ evaluation.")
        return {"BoolQ": 0.0}

    try:
        full_boolq_data = load_dataset("google/boolq", split=dataset_split, trust_remote_code=True)
        print(f"Proc {process_id}: Loaded BoolQ dataset split '{dataset_split}', total size: {len(full_boolq_data)}")
    except Exception:
        print(f"Proc {process_id}: Error loading BoolQ dataset. Skipping.")
        return {"BoolQ": 0.0}

    total_examples_in_split = len(full_boolq_data)
    if num_gpus > 1:
        examples_per_process = total_examples_in_split // num_gpus
        start_idx = process_id * examples_per_process
        end_idx = start_idx + examples_per_process
        if process_id == num_gpus - 1:
            end_idx = total_examples_in_split
        dataset_subset = full_boolq_data.select(range(start_idx, end_idx))
        print(f"Proc {process_id}: Processing subset from index {start_idx} to {end_idx-1} ({len(dataset_subset)} examples).")
    else:
        dataset_subset = full_boolq_data
        print(f"Proc {process_id}: Processing all {len(dataset_subset)} examples in the dataset split.")

    predictions_log = []
    references_log = []
    processed_indices_from_checkpoint = set()

    checkpoint_filename = f"boolq_checkpoint_proc{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    if resume and os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Resuming from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f:
                checkpoint_data = json.load(f)
            predictions_log = checkpoint_data.get('predictions', [])
            references_log = checkpoint_data.get('references', [])
            processed_indices_from_checkpoint = set(checkpoint_data.get('processed_indices', []))
            print(f"Proc {process_id}: Loaded {len(predictions_log)} predictions, {len(references_log)} references, and {len(processed_indices_from_checkpoint)} processed indices from checkpoint.")
        except json.JSONDecodeError:
            print(f"Proc {process_id}: Error reading checkpoint {checkpoint_filepath}. Starting fresh.")
            predictions_log, references_log, processed_indices_from_checkpoint = [], [], set()
    else:
        print(f"Proc {process_id}: No checkpoint found or not resuming. Starting fresh for BoolQ.")

    prompts_to_generate = []
    current_batch_indices_map = []

    for example_data in tqdm(dataset_subset, desc=f"P{process_id} - Preparing BoolQ examples", file=sys.stdout):
        passage = example_data.get('passage', '')
        question = example_data.get('question', '')
        ground_truth_bool = example_data.get('answer', None)

        boolq_idx = example_data.get('idx')
        if boolq_idx is None:
            unique_string = f"{question}_{passage}_{ground_truth_bool}"
            boolq_idx = hashlib.md5(unique_string.encode('utf-8')).hexdigest()

        if boolq_idx in processed_indices_from_checkpoint:
            continue

        if ground_truth_bool is None or not passage or not question:
            print(f"Warn (BoolQ Proc {process_id}): Skipping example due to missing data (idx: {boolq_idx}, passage/question/answer).")
            continue

        prompt = f"Based on the following passage, answer the question with only 'yes' or 'no'.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        prompts_to_generate.append(prompt)
        current_batch_indices_map.append({'idx': boolq_idx, 'ground_truth_bool': ground_truth_bool})

    if not prompts_to_generate:
        print(f"Proc {process_id}: No new BoolQ examples to process after filtering from checkpoint (if any).")
        if predictions_log and references_log:
            preds_metric = [item['prediction_value'] for item in predictions_log]
            refs_metric = [item['reference_value'] for item in references_log]
            if preds_metric and refs_metric:
                final_results = accuracy_metric.compute(predictions=preds_metric, references=refs_metric)
                accuracy = final_results.get('accuracy', 0.0) * 100
                print(f"Proc {process_id}: Accuracy from (resumed) checkpoint data: {accuracy:.2f}% on {len(preds_metric)} examples.")
                return {"BoolQ": accuracy}
        return {"BoolQ": 0.0}

    print(f"Proc {process_id}: Starting batch inference for {len(prompts_to_generate)} BoolQ prompts with batch size: {batch_size}")

    for i in tqdm(range(0, len(prompts_to_generate), batch_size), desc=f"P{process_id} - Generating BoolQ Answers", file=sys.stdout):
        batch_prompts = prompts_to_generate[i : i + batch_size]
        batch_map_info = current_batch_indices_map[i : i + batch_size]

        try:
            batch_outputs_raw = pipe(batch_prompts, max_new_tokens=10, do_sample=False, temperature=0.1, top_p=0.9)
        except Exception:
            for map_info_item in batch_map_info:
                boolq_idx = map_info_item['idx']
                ground_truth_bool = map_info_item['ground_truth_bool']
                true_value_for_metric = 1 if ground_truth_bool else 0
                
                predictions_log.append({'idx': boolq_idx, 'prediction_value': 0, 'raw_text': "#PipelineError"})
                references_log.append({'idx': boolq_idx, 'reference_value': true_value_for_metric})
                processed_indices_from_checkpoint.add(boolq_idx)
            continue

        for j, output_list in enumerate(batch_outputs_raw):
            map_info_item = batch_map_info[j]
            boolq_idx = map_info_item['idx']
            ground_truth_bool = map_info_item['ground_truth_bool']
            true_value_for_metric = 1 if ground_truth_bool else 0
            
            predicted_raw_text = "#GenFail"
            if output_list and isinstance(output_list, list) and len(output_list) > 0 and isinstance(output_list[0], dict):
                predicted_raw_text = output_list[0].get('generated_text', "#GenFail").strip()
            
            normalized_pred_text = normalize_answer_boolq(predicted_raw_text)
            pred_value_for_metric = 1 if normalized_pred_text == "yes" else 0

            predictions_log.append({'idx': boolq_idx, 'prediction_value': pred_value_for_metric, 'raw_text': predicted_raw_text})
            references_log.append({'idx': boolq_idx, 'reference_value': true_value_for_metric})
            processed_indices_from_checkpoint.add(boolq_idx)
        
        if (i // batch_size + 1) % checkpoint_interval_batches == 0:
            save_checkpoint_boolq(checkpoint_filepath, predictions_log, references_log, processed_indices_from_checkpoint)

    save_checkpoint_boolq(checkpoint_filepath, predictions_log, references_log, processed_indices_from_checkpoint)

    if not predictions_log or not references_log:
        print(f"Proc {process_id}: No predictions or references available to compute BoolQ accuracy.")
        return {"BoolQ": 0.0}

    predictions_for_metric = [item['prediction_value'] for item in predictions_log]
    references_for_metric = [item['reference_value'] for item in references_log]

    try:
        eval_results = accuracy_metric.compute(predictions=predictions_for_metric, references=references_for_metric)
        accuracy_score = eval_results.get('accuracy', 0.0) * 100
    except Exception:
        print(f"Proc {process_id}: Failed to compute accuracy.")
        accuracy_score = 0.0
        
    print(f"Proc {process_id} (GPU {gpu_id}) - Final BoolQ Accuracy: {accuracy_score:.2f}% on {len(predictions_for_metric)} examples.")
    
    return {"BoolQ": accuracy_score}