# eval_tasks/gsm8k.py

import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json # Good practice to include if other similar files use it
import gc
import os
# import evaluate as hf_evaluate # Not used by this GSM8K version
from typing import List, Dict, Optional # Added for type hinting
import sys # For tqdm file output

GSM8K_FEWSHOT_EXAMPLES = """
Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Step 1: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Step 2: Working 50 minutes, she earned 0.2 x 50 = <<0.2*50=10>>10.
Step 3: The final answer is #### 10.

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15, and her grandparents gave her twice as much. How much more money does she needs?
A: Step 1: Betty has 100 / 2 = $<<100/2=50>>50.
Step 2: Grandparents gave her 15 * 2 = $<<15*2=30>>30.
Step 3: Total = 50 + 15 + 30 = <<50+15+30=95>>95.
Step 4: 100 - 95 = $<<100-95=5>>5.
Step 5: The final answer is #### 5.

Q: Julie is reading a 120-page book. Yesterday she read 12 pages, and today twice as many. If she wants to read half the remaining pages tomorrow, how many pages is that?
A: Step 1: Today she read 12 x 2 = <<12*2=24>>24.
Step 2: Total read so far = 12 + 24 = <<12+24=36>>36.
Step 3: Remaining = 120 - 36 = <<120-36=84>>84.
Step 4: Half of 84 = <<84/2=42>>42.
Step 5: The final answer is #### 42.

Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
A: Step 1: 3 pages x 2 friends = <<3*2=6>>6 pages per letter session.
Step 2: Twice a week = 6 x 2 = <<6*2=12>>12 pages per week.
Step 3: Per year = 12 x 52 = <<12*52=624>>624.
Step 4: The final answer is #### 624.

Q: Mark planted 10 yellow flowers. He planted 80% more purple ones. Then 25% as many green flowers as the total of yellow and purple. How many total flowers?
A: Step 1: Purple = 10 x 0.8 = <<10*0.8=8>>8.
Step 2: Total yellow + purple = 10 + 8 = <<10+8=18>>18.
Step 3: Green = 0.25 x 18 = <<0.25*18=4.5>>4.5.
Step 4: Total = 10 + 8 + 4.5 = <<10+8+4.5=22.5>>22.5.
Step 5: The final answer is #### 22.5.
"""

def extract_gsm8k_final_answer_from_string(text: str) -> Optional[str]: # Renamed from extract_final_answer
    if text is None:
        return None
    match = re.search(r'####\s*([0-9\-\.,/]+)', text) # Allow comma
    if match:
        extracted_num_str = match.group(1).strip().replace(',', '') # Remove comma
        try:
            if '/' in extracted_num_str:
                parts = extracted_num_str.split('/')
                if len(parts) == 2:
                    num, den = float(parts[0]), float(parts[1])
                    return str(num / den) if den != 0 else None
                return None # Malformed fraction
            else:
                return str(float(extracted_num_str))
        except ValueError:
            print(f"DEBUG GSM8K extract_answer: ValueError converting '{extracted_num_str}'")
            return None # Could also return original string if preferred for debugging
    return None

# --- MODIFIED FUNCTION SIGNATURE ---
def evaluate_gsm8k(
    model_name: str, 
    pipe: callable, 
    model_size_gb: float, # Added for consistency, though not used in this logic
    batch_size: int = 8,   # Use this passed batch_size for inference
    dataset_split: str = "test" # Use this passed dataset_split
    # Add **kwargs if you want to accept other args from main.py without using them
) -> Dict[str, float]:
# --- END MODIFIED SIGNATURE ---

    print(f"\n--- Running GSM8K evaluation for {model_name} ---")
    print(f"Parameters: batch_size (for generation)={batch_size}, dataset_split='{dataset_split}'")
    
    if not hasattr(pipe, 'tokenizer') or pipe.tokenizer is None or \
       pipe.tokenizer.pad_token_id is None or pipe.tokenizer.eos_token_id is None:
        print("ERROR (GSM8K): Pipeline's tokenizer is not properly configured. Skipping.")
        return {"GSM8K": 0.0}

    try:
        # Use the passed dataset_split
        dataset = load_dataset("openai/gsm8k", "main", split=dataset_split)
    except Exception as e:
        print(f"ERROR (GSM8K): Failed to load dataset 'openai/gsm8k' split '{dataset_split}': {e}")
        return {"GSM8K": 0.0}
    
    if len(dataset) == 0:
        print(f"Warn (GSM8K): Dataset split '{dataset_split}' is empty. Returning 0.")
        return {"GSM8K": 0.0}

    def format_prompt_internal_gsm8k(question: str) -> str: # Renamed to avoid conflict if imported elsewhere
        return f"{GSM8K_FEWSHOT_EXAMPLES.strip()}\n\nQ: {question}\nA:"

    all_prompts = []
    all_ground_truths_str = []
    # all_problems_info = [] # Not strictly needed if not used for detailed logging later

    print(f"Preparing {len(dataset)} GSM8K prompts and ground truths...")
    for i in tqdm(range(len(dataset)), desc="Preparing GSM8K problems", file=sys.stdout, mininterval=0.5):
        question = dataset[i]['question']
        ground_truth_full_answer_text = dataset[i]['answer']

        prompt = format_prompt_internal_gsm8k(question)
        ground_truth_final_num_str = extract_gsm8k_final_answer_from_string(ground_truth_full_answer_text)

        if ground_truth_final_num_str is None:
            print(f"Warn (GSM8K): Could not extract ground truth for problem index {i}. Skipping. Full answer: {ground_truth_full_answer_text[:100]}...")
            continue 

        all_prompts.append(prompt)
        all_ground_truths_str.append(ground_truth_final_num_str)
        # all_problems_info.append({'question': question, 'original_idx': i})

    if not all_prompts:
        print("Error (GSM8K): No valid prompts could be prepared after filtering. Aborting.")
        return {"GSM8K": 0.0}

    correct_predictions = 0
    total_valid_problems_processed = len(all_prompts)

    # Use the batch_size passed into the function
    print(f"Starting batched inference for {total_valid_problems_processed} GSM8K problems with batch size: {batch_size}")

    for i in tqdm(range(0, total_valid_problems_processed, batch_size), desc="Generating GSM8K solutions", file=sys.stdout, mininterval=0.5):
        batch_prompts_list = all_prompts[i : i + batch_size]
        batch_ground_truths_list_str = all_ground_truths_str[i : i + batch_size]
        # batch_questions_info = all_problems_info[i : i + BATCH_SIZE] # If needed

        try:
            # The 'stop_sequence' argument is not standard for HF pipeline.
            # Relies on EOS token or max_new_tokens.
            batch_outputs = pipe(
                batch_prompts_list,
                max_new_tokens=384, # Max length for CoT answer
                do_sample=False,    # Typically better for math
                temperature=0.0,    # For greedy decoding
                pad_token_id=pipe.tokenizer.pad_token_id, 
                eos_token_id=pipe.tokenizer.eos_token_id,
                # top_p=0.9 # Often not used when do_sample=False
            )

            for j in range(len(batch_outputs)):
                generated_full_text = ""
                if batch_outputs[j] and isinstance(batch_outputs[j], list) and \
                   len(batch_outputs[j]) > 0 and isinstance(batch_outputs[j][0], dict) and \
                   'generated_text' in batch_outputs[j][0]:
                    generated_full_text = batch_outputs[j][0]['generated_text']
                
                predicted_chain_of_thought_and_answer = generated_full_text.strip()
                predicted_final_num_str = extract_gsm8k_final_answer_from_string(predicted_chain_of_thought_and_answer)
                ground_truth_num_str = batch_ground_truths_list_str[j]
                
                # Optional: More detailed debug per item
                # print(f"\nDEBUG GSM8K Item {i+j}:")
                # print(f"Q: {batch_prompts_list[j].split('Q: ')[-1].split('A:')[0].strip()}")
                # print(f"Generated Full: {predicted_chain_of_thought_and_answer[:200]}...")
                # print(f"Pred Num: {predicted_final_num_str} | GT Num: {ground_truth_num_str}")

                if predicted_final_num_str is not None and ground_truth_num_str is not None:
                    try:
                        pred_float = float(predicted_final_num_str)
                        gt_float = float(ground_truth_num_str)
                        if abs(pred_float - gt_float) < 1e-5: # Tolerance
                            correct_predictions += 1
                            # print("Comparison: CORRECT")
                    except ValueError: # If float conversion fails, compare as strings
                        if predicted_final_num_str == ground_truth_num_str:
                            correct_predictions += 1
                            # print("Comparison: CORRECT (string match)")
                # else:
                    # print("Comparison: INCORRECT (extraction failed for pred or gt)")


        except Exception as e:
            print(f"\033[91mError processing GSM8K batch starting at index {i}: {str(e)}\033[0m")
            # import traceback; traceback.print_exc() # For more detail during debug
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            continue

    final_accuracy_gsm8k = correct_predictions / total_valid_problems_processed * 100 if total_valid_problems_processed > 0 else 0.0
    print(f"\nFinal Accuracy for GSM8K ({model_name}): {correct_predictions}/{total_valid_problems_processed} = {final_accuracy_gsm8k:.2f}%")

    return {"GSM8K": final_accuracy_gsm8k}