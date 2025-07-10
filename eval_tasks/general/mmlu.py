# eval_tasks/mmlu.py

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
from typing import List, Dict, Any, Optional

# --- Improved Few-shot Examples for MMLU ---
# These examples aim for variety in subject matter and a clear Q&A format.
FEWSHOT_EXAMPLES_MMLU = [
    {
        "question": "The development of a new vaccine for a common viral infection is likely to have which of the following effects on the market for antiviral medications?",
        "subject": "medicine",
        "choices": [
            "A. Increase demand for antiviral medications.",
            "B. Decrease demand for antiviral medications.",
            "C. Have no effect on the demand for antiviral medications.",
            "D. Increase the price of antiviral medications."
        ],
        "answer_idx": 1, # Index 1 corresponds to 'B'
        "answer_letter": "B"
    },
    {
        "question": "Which of the following describes the primary function of the hippocampus?",
        "subject": "anatomy",
        "choices": [
            "A. Regulating heart rate and breathing.",
            "B. Processing sensory information.",
            "C. Formation of new memories.",
            "D. Coordinating voluntary movements."
        ],
        "answer_idx": 2, # Index 2 corresponds to 'C'
        "answer_letter": "C"
    },
    {
        "question": "In a democracy, which of the following best describes the principle of 'popular sovereignty'?",
        "subject": "political_science",
        "choices": [
            "A. The power of the government is derived from the consent of the governed.",
            "B. The judiciary has the final say on all legal matters.",
            "C. Citizens directly participate in all legislative decisions.",
            "D. The executive branch has absolute authority."
        ],
        "answer_idx": 0, # Index 0 corresponds to 'A'
        "answer_letter": "A"
    },
    {
        "question": "What is the primary characteristic of a capitalist economic system?",
        "subject": "economics",
        "choices": [
            "A. Centralized economic planning by the government.",
            "B. State ownership of all means of production.",
            "C. Private ownership of the means of production.",
            "D. Distribution of goods based on need rather than ability to pay."
        ],
        "answer_idx": 2, # Index 2 corresponds to 'C'
        "answer_letter": "C"
    },
    {
        "question": "Which of the following is an example of an exothermic reaction?",
        "subject": "chemistry",
        "choices": [
            "A. Melting ice.",
            "B. Photosynthesis.",
            "C. Burning wood.",
            "D. Evaporating water."
        ],
        "answer_idx": 2, # Index 2 corresponds to 'C'
        "answer_letter": "C"
    }
]

def format_mmlu_prompt(
    test_question: str,
    test_subject: str,
    test_choices: List[str],
    fewshot_examples: List[Dict[str, Any]]
) -> str:
    """
    Formats the prompt for MMLU questions, including few-shot examples.
    """
    subject_formatted = test_subject.replace('_', ' ')
    
    # --- Start of the improved prompt template ---
    prompt_parts = [
        f"The following are multiple choice questions (with answers) about {subject_formatted}.\n\n",
        "Answer the following questions by choosing the best option (A, B, C, or D).\n",
        "Your answer should be just the letter.\n\n"
    ]

    for ex in fewshot_examples:
        # Use 'choices' from the example directly as they are already A., B., C., D.
        ex_choices_str = "\n".join(ex['choices'])
        prompt_parts.append(f"Question: {ex['question']}\n{ex_choices_str}\nAnswer: {ex['answer_letter']}\n\n")

    # Add the current question
    current_choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(test_choices)])
    prompt_parts.append(f"Question: {test_question}\n{current_choices_str}\nAnswer:")
    # --- End of the improved prompt template ---
    
    return "".join(prompt_parts).strip()

def extract_mmlu_answer(generated_text: str) -> Optional[str]:
    """
    Extracts the predicted answer letter (A, B, C, D) from the model's raw output.
    Looks for a single capital letter, prioritizing the first one found.
    Returns the 0-indexed integer equivalent (0 for A, 1 for B, etc.)
    """
    if not generated_text:
        return None

    # Use a regex to find a single capital letter A-D, potentially surrounded by whitespace or other chars
    # We are looking for the *first* such letter, which is most likely the model's direct answer.
    match = re.search(r"\b([A-D])\b", generated_text.strip().upper())
    if match:
        # Convert letter to 0-indexed integer string (e.g., 'A' -> '0', 'B' -> '1')
        return str(ord(match.group(1)) - ord('A'))
    return None # Return None if no valid letter is found


def save_checkpoint_mmlu(checkpoint_filepath: str, problem_results_so_far: list):
    """Saves the current state of problem results for MMLU evaluation."""
    try:
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        data_to_save = {
            'problem_results': problem_results_so_far
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save MMLU checkpoint to {checkpoint_filepath}: {e}")


def evaluate_mmlu(
    model_name: str,
    pipe: Any, # Use Any for pipeline type hint to avoid circular imports if pipeline is in main
    model_size_gb: float, # Not directly used but kept for consistent signature
    batch_size: int = 8,
    dataset_split: str = "test", # MMLU typically uses 'test' split
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    checkpoint_dir: str = "checkpoints_mmlu",
    resume: bool = False,
    checkpoint_interval_batches: int = 50,
    subset_name: str = "all" # MMLU has many subjects, "all" for the combined test set
) -> Dict[str, float]:
    """
    Evaluates the given model pipeline on the MMLU dataset
    with support for multi-GPU data splitting and checkpointing.
    Returns a dictionary with 'MMLU' (accuracy) score.
    """
    print(f"\n--- Starting MMLU evaluation for {model_name} (Proc {process_id}, GPU {gpu_id}) ---")
    print(f"Parameters: batch_size={batch_size}, dataset_split='{dataset_split}', resume={resume}, subset='{subset_name}'")

    # --- Load and Split Dataset ---
    try:
        full_mmlu_data = load_dataset("cais/mmlu", subset_name, split=dataset_split)
        print(f"Proc {process_id}: Loaded MMLU dataset: cais/mmlu, subset='{subset_name}', split='{dataset_split}', total size: {len(full_mmlu_data)}")
    except Exception as e:
        print(f"Proc {process_id}: Error loading MMLU dataset cais/mmlu, subset='{subset_name}': {e}. Skipping.")
        return {"MMLU": 0.0}

    total_problems_in_split = len(full_mmlu_data)
    if num_gpus > 1:
        problems_per_process = total_problems_in_split // num_gpus
        start_idx = process_id * problems_per_process
        end_idx = start_idx + problems_per_process
        if process_id == num_gpus - 1:
            end_idx = total_problems_in_split
        dataset_subset = full_mmlu_data.select(range(start_idx, end_idx))
        print(f"Proc {process_id}: Processing MMLU subset from global index {start_idx} to {end_idx-1} ({len(dataset_subset)} problems).")
    else:
        dataset_subset = full_mmlu_data
        print(f"Proc {process_id}: Processing all {len(dataset_subset)} MMLU problems in the dataset split.")

    # --- Checkpointing ---
    problem_results: List[Dict[str, Any]] = [] # Stores detailed results for each problem
    processed_problem_indices_from_checkpoint = set()

    checkpoint_filename = f"mmlu_{subset_name.replace('/', '_')}_checkpoint_proc{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    if resume and os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Resuming MMLU from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f:
                checkpoint_data = json.load(f)
            problem_results = checkpoint_data.get('problem_results', [])
            for res in problem_results:
                if 'original_global_idx' in res:
                    processed_problem_indices_from_checkpoint.add(res['original_global_idx'])
            print(f"Proc {process_id}: Loaded {len(problem_results)} problem results from MMLU checkpoint. Skipping {len(processed_problem_indices_from_checkpoint)} already processed problems.")
        except json.JSONDecodeError:
            print(f"Proc {process_id}: Error reading MMLU checkpoint {checkpoint_filepath}. Starting fresh.")
            problem_results, processed_problem_indices_from_checkpoint = [], set()
    else:
        print(f"Proc {process_id}: No MMLU checkpoint found or not resuming. Starting fresh.")

    # --- Prepare Prompts and Ground Truths ---
    all_prompts = []
    all_ground_truths_idx_str = [] # Stores 0-indexed int as string ('0' for A, '1' for B, etc.)
    all_problems_info = [] # To map generated outputs back to original problem details

    print(f"Proc {process_id}: Preparing MMLU examples for evaluation...")
    for local_idx, item in tqdm(enumerate(dataset_subset), total=len(dataset_subset), desc=f"P{process_id} - Preparing MMLU"):
        original_global_idx = start_idx + local_idx # Calculate global index

        if original_global_idx in processed_problem_indices_from_checkpoint:
            continue

        question = item["question"]
        subject = item["subject"]
        choices = item["choices"]
        ground_truth_int = item["answer"] # 0-indexed integer

        if not question or not choices or ground_truth_int is None:
            print(f"Warn (MMLU Proc {process_id}): Skipping problem at original global index {original_global_idx} due to missing data.")
            continue

        prompt = format_mmlu_prompt(
            test_question=question,
            test_subject=subject,
            test_choices=choices,
            fewshot_examples=FEWSHOT_EXAMPLES_MMLU
        )

        all_prompts.append(prompt)
        all_ground_truths_idx_str.append(str(ground_truth_int)) # Convert to string for comparison
        all_problems_info.append({
            'original_global_idx': original_global_idx,
            'question': question,
            'subject': subject,
            'choices': choices,
            'ground_truth_raw': item["answer"] # Store original int gold for detailed logging
        })

    if not all_prompts:
        print(f"Proc {process_id}: No new MMLU examples to process after filtering from checkpoint.")
        if problem_results:
            correct_count = sum(1 for res in problem_results if res.get('is_correct'))
            total_count = len(problem_results)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            print(f"Proc {process_id}: MMLU accuracy from (resumed) checkpoint: {accuracy:.2f}% on {total_count} examples.")
            return {"MMLU": accuracy}
        return {"MMLU": 0.0}

    print(f"Proc {process_id}: Starting batch inference for {len(all_prompts)} MMLU problems with batch size: {batch_size}")

    # --- Generation Loop ---
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"P{process_id} - Generating MMLU Solutions", file=sys.stdout):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_ground_truths = all_ground_truths_idx_str[i : i + batch_size]
        batch_problems_info = all_problems_info[i : i + batch_size]

        try:
            batch_outputs = pipe(
                batch_prompts,
                do_sample=False,
                max_new_tokens=5, # Expecting just A, B, C, D
                pad_token_id=pipe.tokenizer.pad_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
                stop_sequence=["\n", "Question:"] # Stop at newline or next question
            )

            for j, output_list in enumerate(batch_outputs):
                current_problem_info = batch_problems_info[j]
                ground_truth_str = batch_ground_truths[j]

                generated_raw_text = "#GenFail"
                if output_list and isinstance(output_list, list) and len(output_list) > 0 and isinstance(output_list[0], dict):
                    generated_text = output_list[0].get('generated_text', "#GenFail")
                    # Remove the prompt itself if it's still present in the generated text
                    if generated_text.startswith(batch_prompts[j]):
                        generated_raw_text = generated_text[len(batch_prompts[j]):].strip()
                    else:
                        generated_raw_text = generated_text.strip()

                predicted_answer_idx_str = extract_mmlu_answer(generated_raw_text)

                is_correct = (predicted_answer_idx_str == ground_truth_str)
                
                problem_results.append({
                    "original_global_idx": current_problem_info['original_global_idx'],
                    "question": current_problem_info['question'],
                    "subject": current_problem_info['subject'],
                    "choices": current_problem_info['choices'],
                    "ground_truth_idx": ground_truth_str,
                    "model_raw_output": generated_raw_text,
                    "predicted_idx": predicted_answer_idx_str,
                    "is_correct": is_correct
                })
                processed_problem_indices_from_checkpoint.add(current_problem_info['original_global_idx'])

        except Exception as e:
            print(f"ERROR (MMLU Proc {process_id}): Pipeline generation error in batch starting at index {i}: {e}")
            import traceback
            traceback.print_exc()
            for prob_info, gt_ans in zip(batch_problems_info, batch_ground_truths):
                problem_results.append({
                    "original_global_idx": prob_info['original_global_idx'],
                    "question": prob_info['question'],
                    "subject": prob_info['subject'],
                    "choices": prob_info['choices'],
                    "ground_truth_idx": gt_ans,
                    "model_raw_output": "#PipelineError",
                    "predicted_idx": None,
                    "is_correct": False
                })
                processed_problem_indices_from_checkpoint.add(prob_info['original_global_idx'])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if (i // batch_size + 1) % checkpoint_interval_batches == 0 and i > 0:
            save_checkpoint_mmlu(checkpoint_filepath, problem_results)

    save_checkpoint_mmlu(checkpoint_filepath, problem_results)
    print(f"Proc {process_id}: MMLU batch inference complete. Total items logged: {len(problem_results)}")

    # --- Compute Final Accuracy ---
    if not problem_results:
        print(f"Proc {process_id}: No MMLU problem results to compute metrics.")
        return {"MMLU": 0.0}

    correct_predictions_count = sum(1 for res in problem_results if res.get('is_correct'))
    total_processed_problems = len(problem_results)
    final_accuracy = (correct_predictions_count / total_processed_problems) * 100 if total_processed_problems > 0 else 0

    print(f"Proc {process_id} (GPU {gpu_id}) - Final MMLU Accuracy: {final_accuracy:.2f}% on {total_processed_problems} examples.")

    return {"MMLU": final_accuracy}