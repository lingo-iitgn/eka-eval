
import re
import torch
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
import json
import os
import sys
from typing import List, Dict, Any, Optional


FEWSHOT_EXAMPLES_AGIEVAL = [
    {
        "question": "Which of the following, if true, would most strengthen the argument?\n\nArgument: Consuming brightly colored fruits and vegetables is beneficial for eye health because they contain high levels of antioxidants, which protect cells from damage.",
        "choices": [
            "A. Antioxidants are also found in some grains.",
            "B. People who eat a lot of red meat have poor eye health.",
            "C. Damaged eye cells are a major cause of vision loss.",
            "D. Some darkly colored vegetables also contain antioxidants.",
            "E. Eye health is important for overall well-being."
        ],
        "answer": "C"
    },
    {
        "question": "A recent study found that students who listen to classical music while studying perform better on tests than those who study in silence. This suggests that listening to classical music enhances cognitive function.\n\nWhich of the following, if true, would most weaken the argument?",
        "choices": [
            "A. The students in the classical music group were already high academic achievers.",
            "B. Other types of music also have a positive effect on studying.",
            "C. The study was conducted on a diverse group of students.",
            "D. Test performance is not the only measure of cognitive function.",
            "E. Some students find classical music distracting."
        ],
        "answer": "A"
    },
    {
        "question": "All birds have feathers. My pet is a bird. Therefore, my pet has feathers.\n\nThis argument is an example of:",
        "choices": [
            "A. Inductive reasoning",
            "B. Deductive reasoning",
            "C. Abductive reasoning",
            "D. Analogical reasoning",
            "E. Faulty generalization"
        ],
        "answer": "B"
    }
]

def format_prompt_agieval(question: str, choices: List[str], fewshot_examples: List[Dict[str, Any]]) -> str:
    """
    Formats the prompt for AGIEval LSAT-LR questions, including few-shot examples.
    """
    prompt_parts = [
        "You are an expert in LSAT logical reasoning. Read the question carefully and choose the best answer.\n",
        "Answer using only the letter: A, B, C, D, or E.\n\n"
    ]

    # Add few-shot examples
    for ex in fewshot_examples:
        ex_choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(ex["choices"])])
        prompt_parts.append(f"Question:\n{ex['question'].strip()}\n\nChoices:\n{ex_choices_str}\n\nAnswer: {ex['answer']}\n\n")

    # Add the current question
    current_choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    prompt_parts.append(f"Question:\n{question.strip()}\n\nChoices:\n{current_choices_str}\n\nAnswer:")
    
    return "".join(prompt_parts)

def extract_answer_agieval(generated_text: str) -> Optional[str]:
    """
    Extracts the predicted answer letter (A-E) from the model's raw output.
    Looks for a single capital letter, prioritizing the first one found.
    """
    if not generated_text:
        return None
    
    match = re.search(r"[ABCDE]", generated_text.strip().upper())
    if match:
        return match.group(0)
    return "Invalid" 
def save_checkpoint_agieval(checkpoint_filepath: str, problem_results_so_far: list):
    """Saves the current state of problem results for AGIEval evaluation."""
    try:
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        data_to_save = {
            'problem_results': problem_results_so_far
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save AGIEval checkpoint to {checkpoint_filepath}: {e}") # Changed to warning


def evaluate_agieval(
    model_name: str,
    pipe: pipeline,
    batch_size: int = 4,
    dataset_split: str = "test", 
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    checkpoint_dir: str = "checkpoints_agieval",
    resume: bool = False,
    checkpoint_interval_batches: int = 50,
    subset_name: str = "agieval-lsat-lr" 
) -> Dict[str, float]:
    """
    Evaluates the given model pipeline on the AGIEval LSAT-LR dataset
    with support for multi-GPU data splitting and checkpointing.
    Returns a dictionary with 'AGIEval-LSAT-LR' (accuracy) score.
    """
    print(f"\n--- Starting AGIEval LSAT-LR evaluation for {model_name} (Proc {process_id}, GPU {gpu_id}) ---")
    print(f"Parameters: batch_size={batch_size}, dataset_split='{dataset_split}', resume={resume}, subset='{subset_name}'")

    # --- Load and Split Dataset ---
    try:
        # AGIEval LSAT-LR is in 'hails/agieval' with specific subset
        full_agieval_data = load_dataset("hails/agieval", subset_name, split=dataset_split)
        print(f"Proc {process_id}: Loaded AGIEval dataset: hails/{subset_name} split='{dataset_split}', total size: {len(full_agieval_data)}")
    except Exception as e:
        print(f"Proc {process_id}: Error loading AGIEval dataset hails/{subset_name}: {e}. Skipping.")
        return {"AGIEval-LSAT-LR": 0.0}

    total_problems_in_split = len(full_agieval_data)
    if num_gpus > 1:
        problems_per_process = total_problems_in_split // num_gpus
        start_idx = process_id * problems_per_process
        end_idx = start_idx + problems_per_process
        if process_id == num_gpus - 1:
            end_idx = total_problems_in_split
        dataset_subset = full_agieval_data.select(range(start_idx, end_idx))
        print(f"Proc {process_id}: Processing AGIEval subset from index {start_idx} to {end_idx-1} ({len(dataset_subset)} problems).")
    else:
        dataset_subset = full_agieval_data
        print(f"Proc {process_id}: Processing all {len(dataset_subset)} AGIEval problems in the dataset split.")

    # --- Checkpointing ---
    problem_results: List[Dict[str, Any]] = [] 
    processed_problem_indices_from_checkpoint = set()

    checkpoint_filename = f"{subset_name.replace('/', '_')}_checkpoint_proc{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    if resume and os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Resuming AGIEval from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f:
                checkpoint_data = json.load(f)
            problem_results = checkpoint_data.get('problem_results', [])
            for res in problem_results:
                if 'original_global_idx' in res:
                    processed_problem_indices_from_checkpoint.add(res['original_global_idx'])
            print(f"Proc {process_id}: Loaded {len(problem_results)} problem results from AGIEval checkpoint. Skipping {len(processed_problem_indices_from_checkpoint)} already processed problems.")
        except json.JSONDecodeError:
            print(f"Proc {process_id}: Error reading AGIEval checkpoint {checkpoint_filepath}. Starting fresh.")
            problem_results, processed_problem_indices_from_checkpoint = [], set()
    else:
        print(f"Proc {process_id}: No AGIEval checkpoint found or not resuming. Starting fresh.")

    # --- Prepare Prompts and Ground Truths ---
    all_prompts = []
    all_ground_truths = []
    all_problems_info = [] # To map generated outputs back to original problem details

    print(f"Proc {process_id}: Preparing AGIEval examples for evaluation...")
    for local_idx, item in tqdm(enumerate(dataset_subset), total=len(dataset_subset), desc=f"P{process_id} - Preparing AGIEval"):
        original_global_idx = start_idx + local_idx # Calculate global index

        if original_global_idx in processed_problem_indices_from_checkpoint:
            continue

        question = item["query"]
        choices = item["choices"]
        # gold is a list of integers, e.g., [0] for A, [1] for B.
        # We need to convert it to a letter.
        gold_letter = chr(65 + item["gold"][0]) if isinstance(item["gold"], list) and len(item["gold"]) > 0 else 'N/A'

        if not question or not choices or gold_letter == 'N/A':
            print(f"Warn (AGIEval Proc {process_id}): Skipping problem at original global index {original_global_idx} due to missing or invalid data.")
            continue

        prompt = format_prompt_agieval(question, choices, FEWSHOT_EXAMPLES_AGIEVAL)
        
        all_prompts.append(prompt)
        all_ground_truths.append(gold_letter) # Store the gold letter
        all_problems_info.append({
            'original_global_idx': original_global_idx,
            'question': question,
            'choices': choices,
            'gold_answer_raw': item["gold"] # Store original gold for detailed logging
        })

    if not all_prompts:
        print(f"Proc {process_id}: No new AGIEval examples to process after filtering from checkpoint.")
        # If no new prompts, calculate accuracy from loaded results if any
        if problem_results:
            correct_count = sum(1 for res in problem_results if res.get('is_correct'))
            total_count = len(problem_results)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            print(f"Proc {process_id}: AGIEval accuracy from (resumed) checkpoint: {accuracy:.2f}% on {total_count} examples.")
            return {"AGIEval-LSAT-LR": accuracy}
        return {"AGIEval-LSAT-LR": 0.0}

    print(f"Proc {process_id}: Starting batch inference for {len(all_prompts)} AGIEval problems with batch size: {batch_size}")

    # --- Generation Loop ---
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"P{process_id} - Generating AGIEval Solutions", file=sys.stdout):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_ground_truths = all_ground_truths[i : i + batch_size]
        batch_problems_info = all_problems_info[i : i + batch_size]

        try:
            # max_new_tokens for a single letter answer should be small, e.g., 5-10
            # stop_sequence could also be used to stop at the first newline or after a letter.
            batch_outputs = pipe(
                batch_prompts,
                max_new_tokens=10, # Expecting just a single letter, maybe a few tokens around it.
                do_sample=False,
                pad_token_id=pipe.tokenizer.pad_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
                stop_sequence=["\n", "Q:"] # Stop at newline or next question
            )

            for j, output_list in enumerate(batch_outputs):
                current_problem_info = batch_problems_info[j]
                ground_truth_letter = batch_ground_truths[j]

                generated_raw_text = "#GenFail"
                if output_list and isinstance(output_list, list) and len(output_list) > 0 and isinstance(output_list[0], dict):
                    generated_raw_text = output_list[0].get('generated_text', "#GenFail").strip()
                    # Remove the prompt itself if it's still present
                    if generated_raw_text.startswith(batch_prompts[j]):
                        generated_raw_text = generated_raw_text[len(batch_prompts[j]):].strip()

                predicted_letter = extract_answer_agieval(generated_raw_text)

                is_correct = (predicted_letter == ground_truth_letter)
                if is_correct:
                    pass # Correct predictions are counted later

                problem_results.append({
                    "original_global_idx": current_problem_info['original_global_idx'],
                    "question": current_problem_info['question'],
                    "choices": current_problem_info['choices'],
                    "gold_answer_letter": ground_truth_letter,
                    "model_raw_output": generated_raw_text,
                    "predicted_answer_letter": predicted_letter,
                    "is_correct": is_correct
                })
                processed_problem_indices_from_checkpoint.add(current_problem_info['original_global_idx'])

        except Exception as e:
            print(f"ERROR (AGIEval Proc {process_id}): Pipeline generation error in batch starting at index {i}: {e}")
            import traceback
            traceback.print_exc()
            for prob_info, gt_ans in zip(batch_problems_info, batch_ground_truths):
                problem_results.append({
                    "original_global_idx": prob_info['original_global_idx'],
                    "question": prob_info['question'],
                    "choices": prob_info['choices'],
                    "gold_answer_letter": gt_ans,
                    "model_raw_output": "#PipelineError",
                    "predicted_answer_letter": None,
                    "is_correct": False
                })
                processed_problem_indices_from_checkpoint.add(prob_info['original_global_idx'])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # No gc.collect() as it's already in main, re-initializing pipeline
            continue

        if (i // batch_size + 1) % checkpoint_interval_batches == 0 and i > 0:
            save_checkpoint_agieval(checkpoint_filepath, problem_results)

    save_checkpoint_agieval(checkpoint_filepath, problem_results) # Final save
    print(f"Proc {process_id}: AGIEval batch inference complete. Total items logged: {len(problem_results)}")

    # --- Compute Final Accuracy ---
    if not problem_results:
        print(f"Proc {process_id}: No AGIEval problem results to compute metrics.")
        return {"AGIEval-LSAT-LR": 0.0}

    correct_predictions_count = sum(1 for res in problem_results if res.get('is_correct'))
    total_processed_problems = len(problem_results)
    final_accuracy = (correct_predictions_count / total_processed_problems) * 100 if total_processed_problems > 0 else 0

    print(f"Proc {process_id} (GPU {gpu_id}) - Final AGIEval-LSAT-LR Accuracy: {final_accuracy:.2f}% on {total_processed_problems} examples.")

    return {"AGIEval-LSAT-LR": final_accuracy}

