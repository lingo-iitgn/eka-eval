import torch
import re
import evaluate
from transformers import pipeline
from datasets import load_dataset
import string
from tqdm import tqdm
import json
import os
import sys
from typing import List, Dict, Any, Union, Optional # Import Union and Optional

def normalize_answer_math(s: str) -> str:
    """Basic normalization for numerical answers."""
    return s.strip().lower()


def save_checkpoint_math(checkpoint_filepath: str, problem_results_so_far: list):
    """Saves the current state of problem results for MATH evaluation."""
    try:
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        data_to_save = {
            'problem_results': problem_results_so_far
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    except Exception:
        pass # Suppress checkpoint save errors in cleaner version


def format_prompt(question: str, subject: str) -> str:
    """Formats the prompt for the MATH problem."""
    return f"""
Given is the subject: {subject}
and question: {question}
Some examples to refer:
Example 1:
subject: Prealgebra
question: If $5x - 3 = 12$, what is the value of $5x + 3$?
solution: Adding 6 to both sides of $5x - 3 = 12$ gives $5x - 3 + 6 = 12 + 6$. Simplifying both sides gives $5x + 3 = \\boxed{{18}}$.
Final answer: 18
Example 2:
subject: Prealgebra
question: Alice wants to buy $3$ pounds of veal at the grocery store, but the scales at the store only show weight in kilograms. If one kilogram is $2.20$ pounds, how many kilograms of veal should Alice buy? (You may use a calculator on this problem; answer to the nearest hundredth.)
solution: Since Alice wants to buy $3$ pounds of veal, we multiply the quantity of $3$ pounds by the conversion factor $\\frac{{1\\ \\text{{kg}}}}{{2.20\\ \\text{{lb}}}}$ to obtain $3\\ \\text{{lb}} \\cdot \\frac{{1\\ \\text{{kg}}}}{{2.20\\ \\text{{lb}}}} \\approx \\boxed{{1.36}}\\ \\text{{kg}}$.
Final answer: 1.36
Example 3:
subject: Number Theory
question: One morning each member of Angela's family drank an 8-ounce mixture of coffee with milk. The amounts of coffee and milk varied from cup to cup, but were never zero. Angela drank a quarter of the total amount of milk and a sixth of the total amount of coffee. How many people are in the family?
solution: Suppose that the whole family drank $x$ cups of milk and $y$ cups of coffee. Let $n$ denote the number of people in the family. The information given implies that $\\frac{{x}}{{4}} + \\frac{{y}}{{6}} = \\frac{{x + y}}{{n}}$. This leads to
\\[
3x(n - 4) = 2y(6 - n).
\\]
Since $x$ and $y$ are positive, the only positive integer $n$ for which both sides have the same sign is $n = \\boxed{{5}}$.
 5
Example 4:
subject: Precalculus
question: Simplify
\\[
\\frac{{\\sin^4 x + \\cos^4 x - 1}}{{\\sin^6 x + \\cos^6 x - 1}}.
\\]
solution: Let $p = \\sin x \\cos x$. We know that $\\sin^2 x + \\cos^2 x = 1$. Squaring both sides, we get
\\[
\\sin^4 x + 2 \\sin^2 x \\cos^2 x + \\cos^4 x = 1.
\\]
Hence,
\\[
\\sin^4 x + \\cos^4 x = 1 - 2p^2.
\\]
Then,
\\[
(\\sin^2 x + \\cos^2 x)(\\sin^4 x + \\cos^4 x) = 1 - 2p^2.
\\]
Expanding, we get
\\[
\\sin^6 x + \\sin^2 x \\cos^4 x + \\cos^2 x \\sin^4 x + \\cos^6 x = 1 - 2p^2.
\\]
Hence,
\\[
\\begin{{aligned}}
\\sin^6 x + \\cos^6 x &= 1 - 2p^2 - (\\sin^2 x \\cos^4 x + \\cos^2 x \\sin^4 x) \\\\
&= 1 - 2p^2 - \\sin^2 x \\cos^2 x (\\sin^2 x + \\cos^2 x) \\\\
&= 1 - 3p^2.
\\end{{aligned}}
\\]
Therefore,
\\[
\\frac{{\\sin^4 x + \\cos^4 x - 1}}{{\\sin^6 x + \\cos^6 x - 1}} = \\frac{{-2p^2}}{{-3p^2}} = \\boxed{{\\frac{{2}}{{3}}}}.
\\]
 $\\frac{{2}}{{3}}$
Think stepwise, refer to examples and provide the answer in similar manner.
Answer:
"""

# Corrected type hint here: Use Union[str, None] or Optional[str]
def extract_final_answer(text: str) -> Optional[str]: # Changed str | None to Optional[str]
    """
    Extracts the final answer from the model's generated text.
    Prioritizes '\boxed{}' and then 'Final Answer: '.
    Attempts to normalize to float for consistent numerical comparison.
    """
    if text is None:
        return None

    # Try to find \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        extracted_string = match.group(1).strip()
    else:
        # Try to find "Final Answer: " followed by digits/decimals/fractions
        # The previous regex `\s*([0-9\-\./\s]+(?:\.[0-9]+)?(?:[a-zA-Z\\/]*))` was too broad.
        # Let's adjust it to be more specific to "Final answer: X" or just numbers.
        # This will also try to capture explicit "Final answer: " if present.
        match = re.search(r'\s*([0-9\-\./]+(?:\.[0-9]+)?(?:[a-zA-Z\\/]*))', text, re.IGNORECASE)
        if match:
            extracted_string = match.group(1).strip()
        else:
            # Fallback to simple number/fraction extraction but be careful
            # This regex is still broad but tries to get a numeric pattern.
            # For MATH, answers are typically numbers or simple fractions.
            match = re.search(r'([0-9\-\./]+(?:\.[0-9]+)?(?:[a-zA-Z\\/]*))', text)
            if match:
                extracted_string = match.group(1).strip()
            else:
                return None # No recognizable answer pattern

    if extracted_string is not None:
        try:
            # Attempt to convert to float for robust numerical comparison (e.g., "1" vs "1.0")
            # This is critical for MATH problems where answers are often numbers.
            return str(float(extracted_string))
        except ValueError:
            # If it's a fraction (e.g., "1/2") or an expression that can't be directly float, return as is.
            # Comparison will then be string-based.
            return extracted_string
    return None

def evaluate_math(
    model_name: str,
    pipe: pipeline,
    model_size_gb: float, # Not directly used in this function but passed for consistency
    batch_size: int = 4,
    dataset_split: str = "test", # For Hendrycks MATH, typically 'test' split
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    checkpoint_dir: str = "checkpoints_math",
    resume: bool = False,
    checkpoint_interval_batches: int = 50
) -> Dict[str, float]:
    """
    Evaluates the given model pipeline on the Hendrycks MATH dataset
    with support for multi-GPU data splitting and checkpointing.
    Returns a dictionary with 'MATH' (accuracy) score.
    """
    print(f"\n--- Starting MATH evaluation for {model_name} (Proc {process_id}, GPU {gpu_id}) ---")
    print(f"Parameters: batch_size={batch_size}, dataset_split='{dataset_split}', resume={resume}")

    # --- Load and Split Dataset ---
    try:
        full_math_data = load_dataset("nlile/hendrycks-MATH-benchmark", split=dataset_split, trust_remote_code=True)
        print(f"Proc {process_id}: Loaded MATH dataset split '{dataset_split}', total size: {len(full_math_data)}")
    except Exception:
        print(f"Proc {process_id}: Error loading MATH dataset. Skipping.")
        return {"MATH": 0.0}

    total_problems_in_split = len(full_math_data)
    if num_gpus > 1:
        problems_per_process = total_problems_in_split // num_gpus
        start_idx = process_id * problems_per_process
        end_idx = start_idx + problems_per_process
        if process_id == num_gpus - 1: end_idx = total_problems_in_split
        dataset_subset = full_math_data.select(range(start_idx, end_idx))
        print(f"Proc {process_id}: Processing MATH subset from index {start_idx} to {end_idx-1} ({len(dataset_subset)} problems).")
    else:
        dataset_subset = full_math_data
        print(f"Proc {process_id}: Processing all {len(dataset_subset)} MATH problems in the dataset split.")

    # --- Checkpointing ---
    problem_results: List[Dict[str, Any]] = [] # Stores detailed results for each problem
    processed_problem_indices_from_checkpoint = set() # To track which original dataset indices were processed

    checkpoint_filename = f"math_checkpoint_proc{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    if resume and os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Resuming MATH from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f: checkpoint_data = json.load(f)
            problem_results = checkpoint_data.get('problem_results', [])
            # Reconstruct processed_problem_indices_from_checkpoint
            for res in problem_results:
                if 'problem_idx' in res: # Ensure 'problem_idx' exists
                    processed_problem_indices_from_checkpoint.add(res['problem_idx'])
            print(f"Proc {process_id}: Loaded {len(problem_results)} problem results from MATH checkpoint. Skipping {len(processed_problem_indices_from_checkpoint)} already processed problems.")
        except json.JSONDecodeError:
            print(f"Proc {process_id}: Error reading MATH checkpoint {checkpoint_filepath}. Starting fresh.")
            problem_results, processed_problem_indices_from_checkpoint = [], set()
    else:
        print(f"Proc {process_id}: No MATH checkpoint found or not resuming. Starting fresh.")


    # --- Prepare Prompts and Ground Truths ---
    all_prompts = []
    all_ground_truths = []
    all_problem_info_for_batch = [] # To map generated outputs back to original problem details

    print(f"Proc {process_id}: Preparing MATH examples for evaluation...")
    for original_idx, problem_data in tqdm(enumerate(dataset_subset), total=len(dataset_subset), desc=f"P{process_id} - Preparing MATH"):
        # Map original_idx (0 to len(dataset_subset)-1) to its index in the full dataset
        full_dataset_idx = start_idx + original_idx
        if full_dataset_idx in processed_problem_indices_from_checkpoint:
            continue

        question = problem_data.get('problem')
        subject = problem_data.get('subject')
        solution = problem_data.get('solution') # The full solution text

        if not question or not subject or not solution:
            print(f"Warn (MATH Proc {process_id}): Skipping problem at original index {full_dataset_idx} due to missing data.")
            continue

        prompt = format_prompt(question, subject)
        ground_truth = extract_final_answer(solution)

        if ground_truth is None:
            print(f"Warn (MATH Proc {process_id}): Could not extract ground truth answer for problem {full_dataset_idx}. Skipping.")
            continue

        all_prompts.append(prompt)
        all_ground_truths.append(ground_truth)
        all_problem_info_for_batch.append({
            'full_dataset_idx': full_dataset_idx, # Store original index for checkpointing
            'question': question,
            'subject': subject,
            'ground_truth_solution': solution # Store original solution for results
        })

    if not all_prompts:
        print(f"Proc {process_id}: No new MATH examples to process after filtering from checkpoint.")
        # If no new prompts, calculate accuracy from loaded results if any
        if problem_results:
            correct_count = sum(1 for res in problem_results if res.get('is_correct'))
            total_count = len(problem_results)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            print(f"Proc {process_id}: MATH accuracy from (resumed) checkpoint: {accuracy:.2f}% on {total_count} examples.")
            return {"MATH": accuracy}
        return {"MATH": 0.0}

    print(f"Proc {process_id}: Starting batch inference for {len(all_prompts)} MATH prompts with batch size: {batch_size}")

    # --- Generation Loop ---
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"P{process_id} - Generating MATH Solutions", file=sys.stdout):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_ground_truths = all_ground_truths[i : i + batch_size]
        batch_problem_info = all_problem_info_for_batch[i : i + batch_size]

        try:
            batch_outputs_raw = pipe(
                batch_prompts,
                max_new_tokens=256, # Adjust as needed for solution length
                do_sample=False,
                temperature=0.1,
                top_p=0.9
            )
        except Exception:
            for prob_info, gt_ans in zip(batch_problem_info, batch_ground_truths):
                problem_results.append({
                    "problem_idx": prob_info['full_dataset_idx'],
                    "subject": prob_info['subject'],
                    "question": prob_info['question'],
                    "ground_truth_answer": gt_ans,
                    "model_raw_output": "#PipelineError",
                    "model_extracted_answer": None,
                    "is_correct": False
                })
                processed_problem_indices_from_checkpoint.add(prob_info['full_dataset_idx'])
            continue

        for j, output_list in enumerate(batch_outputs_raw):
            current_problem_info = batch_problem_info[j]
            ground_truth_answer = batch_ground_truths[j]

            generated_raw_text = "#GenFail"
            if output_list and isinstance(output_list, list) and len(output_list) > 0 and isinstance(output_list[0], dict):
                generated_raw_text = output_list[0].get('generated_text', "#GenFail").strip()

            predicted_answer = extract_final_answer(generated_raw_text)

            is_correct = (predicted_answer == ground_truth_answer)

            problem_results.append({
                "problem_idx": current_problem_info['full_dataset_idx'],
                "subject": current_problem_info['subject'],
                "question": current_problem_info['question'],
                "ground_truth_answer": ground_truth_answer,
                "model_raw_output": generated_raw_text,
                "model_extracted_answer": predicted_answer,
                "is_correct": is_correct
            })
            processed_problem_indices_from_checkpoint.add(current_problem_info['full_dataset_idx'])

        if (i // batch_size + 1) % checkpoint_interval_batches == 0 and i > 0:
            save_checkpoint_math(checkpoint_filepath, problem_results)

    save_checkpoint_math(checkpoint_filepath, problem_results) # Final save
    print(f"Proc {process_id}: MATH batch inference complete. Total items logged: {len(problem_results)}")

    # --- Compute Metrics ---
    if not problem_results:
        print(f"Proc {process_id}: No MATH problem results to compute metrics.")
        return {"MATH": 0.0}

    correct_predictions_count = sum(1 for res in problem_results if res.get('is_correct'))
    total_processed_problems = len(problem_results)
    final_accuracy = (correct_predictions_count / total_processed_problems) * 100 if total_processed_problems > 0 else 0

    print(f"Proc {process_id} (GPU {gpu_id}) - Final MATH Accuracy: {final_accuracy:.2f}% on {total_processed_problems} examples.")

    return {"MATH": final_accuracy}