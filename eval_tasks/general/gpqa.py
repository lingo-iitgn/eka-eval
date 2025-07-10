import torch
import re
import evaluate as hf_evaluate # Still import for other potential metrics or if you revert
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset as HFDataset
import random
from tqdm import tqdm
import json
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import sys
import hashlib


@dataclass
class GPQAEvalResult:
    gpqa_id: str; question: str; choices: List[str]; correct_answer_text: str
    correct_answer_label: str; model_raw_output: str
    model_predicted_label: Optional[str]; is_correct: bool; error_message: str = ""

# --- Global Setup (Accuracy metric loaded but might not be used if manual calc is preferred) ---
try:
    accuracy_metric_gpqa_loaded = hf_evaluate.load("accuracy")
    print("DEBUG (gpqa.py): Accuracy metric loaded successfully for GPQA (available if needed).")
except Exception as e:
    print(f"WARN (gpqa.py): Could not load 'accuracy' metric: {e}. Manual calculation will be used.")
    accuracy_metric_gpqa_loaded = None

# --- Preprocessing ---
def preprocess_gpqa_text(text: Optional[str]) -> str:
    if text is None: return "" 
    text = str(text).strip(); text = text.replace(" [title]", ". ") 
    text = re.sub(r"\s*\[.*?\]\s*", " ", text); text = re.sub(r"\s+", " ", text) 
    return text.strip()

# --- Few-shot Examples ---
FEWSHOT_EXAMPLES_GPQA = [
    {"question_text": "What is the capital of France?", "choice_A": "Berlin", "choice_B": "Paris", "choice_C": "Rome", "choice_D": "Madrid", "correct_label": "(B)"},
    {"question_text": "Which planet is known as the 'Red Planet'?", "choice_A": "Venus", "choice_B": "Mars", "choice_C": "Jupiter", "choice_D": "Saturn", "correct_label": "(B)"},
    {"question_text": "Who painted the Mona Lisa?", "choice_A": "Vincent van Gogh", "choice_B": "Pablo Picasso", "choice_C": "Leonardo da Vinci", "choice_D": "Claude Monet", "correct_label": "(C)"},
]

# --- Dataset Processing ---
def process_gpqa_docs(raw_dataset: HFDataset, dataset_name_for_id: str = "gpqa") -> HFDataset:
    print("DEBUG GPQA: Inside process_gpqa_docs")
    processed_examples = []
    if not raw_dataset: return HFDataset.from_list([])
    for i, doc in enumerate(tqdm(raw_dataset, desc="Processing GPQA Docs", file=sys.stdout, mininterval=1.0)):
        question = doc.get("Question"); correct_ans_text_orig = doc.get("Correct Answer")
        inc1 = doc.get("Incorrect Answer 1"); inc2 = doc.get("Incorrect Answer 2"); inc3 = doc.get("Incorrect Answer 3")
        if not all([question, correct_ans_text_orig, inc1, inc2, inc3]):
            print(f"WARN (GPQA process_docs): Skipping ex {i} due to missing fields. Q: {str(question)[:50]}..."); continue
        
        gpqa_id = doc.get("id", f"{dataset_name_for_id}_{i}_{hashlib.md5(f'{question}{correct_ans_text_orig}'.encode()).hexdigest()[:8]}")

        choices_raw = [correct_ans_text_orig, inc1, inc2, inc3]
        preprocessed_choices = [preprocess_gpqa_text(c) if c else "[ChoiceErr:Empty]" for c in choices_raw]
        processed_correct_ans_text = preprocessed_choices[0] # Correct answer is always the first in choices_raw

        choices_to_shuffle = [{"text": text, "is_correct_orig": (idx == 0)} for idx, text in enumerate(preprocessed_choices)]
        random.shuffle(choices_to_shuffle)
        
        correct_answer_shuffled_pos = -1; shuffled_final_texts = []
        for j, item in enumerate(choices_to_shuffle):
            shuffled_final_texts.append(item["text"])
            if item["is_correct_orig"]: correct_answer_shuffled_pos = j
        
        if correct_answer_shuffled_pos == -1:
            print(f"CRITICAL (GPQA process_docs): Correct ans not found after shuffle for ID {gpqa_id}. Orig: '{processed_correct_ans_text}'. Shuffled: {shuffled_final_texts}")
            processed_examples.append({"gpqa_id": gpqa_id, "question_text": preprocess_gpqa_text(question),"choice_A": "ERR", "choice_B": "ERR", "choice_C": "ERR", "choice_D": "ERR", "correct_label": "(ERR_MAP)", "original_correct_answer_text": processed_correct_ans_text, "is_valid": False}); continue
        
        processed_examples.append({
            "gpqa_id": gpqa_id, "question_text": preprocess_gpqa_text(question),
            "choice_A": shuffled_final_texts[0], "choice_B": shuffled_final_texts[1],
            "choice_C": shuffled_final_texts[2], "choice_D": shuffled_final_texts[3],
            "correct_label": f"({chr(65 + correct_answer_shuffled_pos)})",
            "original_correct_answer_text": processed_correct_ans_text, "is_valid": True })
    if not processed_examples: return HFDataset.from_list([])
    return HFDataset.from_list(processed_examples)

# --- Prompt Formatting ---
def format_gpqa_prompt(processed_example: Dict, few_shot_examples: List[Dict]) -> str:
    # ... (Same as your last version, using FEWSHOT_EXAMPLES_GPQA)
    prompt_parts = ["The following are multiple-choice questions. Read each question and choose the best answer from (A), (B), (C), or (D).\nYour answer should be only the letter in parentheses, e.g., (A).\n"]
    for fs_example in few_shot_examples:
        prompt_parts.append(f"\nQuestion: {fs_example['question_text']}")
        prompt_parts.append(f"(A) {fs_example['choice_A']}\n(B) {fs_example['choice_B']}\n(C) {fs_example['choice_C']}\n(D) {fs_example['choice_D']}")
        prompt_parts.append(f"Answer: {fs_example['correct_label']}\n")
    prompt_parts.append("\n---\nNow, answer the following question:\n")
    prompt_parts.append(f"Question: {processed_example['question_text']}")
    prompt_parts.append(f"(A) {processed_example['choice_A']}\n(B) {processed_example['choice_B']}\n(C) {processed_example['choice_C']}\n(D) {processed_example['choice_D']}")
    prompt_parts.append(f"Answer: ")
    return "\n".join(prompt_parts)

# --- Answer Extraction ---
def extract_gpqa_answer_label(generated_text: str) -> Optional[str]:
    # ... (Same as your last version)
    if generated_text is None: return None; text = generated_text.strip().upper()
    match = re.search(r'\(([A-D])\)', text); 
    if match: return f"({match.group(1)})"
    match = re.search(r'\b([A-D])\.', text); 
    if match: return f"({match.group(1)})"
    match = re.search(r'^\s*([A-D])\s*$', text); 
    if match: return f"({match.group(1)})"
    match = re.search(r'[A-D]', text); # Last resort, first A,B,C,D found
    if match: return f"({match.group(0)})"
    return None

# --- Checkpointing ---
def save_checkpoint_gpqa(checkpoint_filepath: str, results_log: List[GPQAEvalResult]):
    # ... (Same as your last version - overwrites with full log)
    try:
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        with open(checkpoint_filepath, 'w', encoding='utf-8') as f_jsonl: 
            for res_item_dc in results_log: 
                if isinstance(res_item_dc, GPQAEvalResult): f_jsonl.write(json.dumps(asdict(res_item_dc)) + '\n')
                elif isinstance(res_item_dc, dict): f_jsonl.write(json.dumps(res_item_dc) + '\n')
    except Exception as e: print(f"ERROR (GPQA): Failed to save checkpoint: {e}")

def load_checkpoint_gpqa(checkpoint_filepath: str) -> List[GPQAEvalResult]:
    # ... (Same as your last version)
    loaded_results = []
    if not os.path.exists(checkpoint_filepath): return loaded_results
    try:
        with open(checkpoint_filepath, 'r', encoding='utf-8') as f_jsonl:
            for line_idx, line in enumerate(f_jsonl):
                if line.strip():
                    try: data = json.loads(line); loaded_results.append(GPQAEvalResult(**data))
                    except Exception as e_parse: print(f"WARN (GPQA load_checkpoint): Skipping malformed line {line_idx+1}: {e_parse}")
        print(f"Loaded {len(loaded_results)} results from GPQA checkpoint {checkpoint_filepath}.")
    except Exception as e: print(f"ERROR (GPQA load_checkpoint): Error reading {checkpoint_filepath}: {e}."); loaded_results = []
    return loaded_results


# --- Main GPQA Evaluation Function ---
def evaluate_gpqa(
    model_name: str, pipe: pipeline, model_size_gb: float, batch_size: int = 2,
    dataset_split: str = "train[:10]", dataset_name: str = "Idavidrein/gpqa",
    dataset_config: str = "gpqa_extended", process_id: int = 0, gpu_id: int = 0,
    num_gpus: int = 1, checkpoint_dir: str = "checkpoints_gpqa", resume: bool = False,
    gpqa_data_dir: Optional[str] = None, gpqa_script_path: Optional[str] = None
) -> Dict[str, float]:

    print(f"\n--- DEBUG GPQA STEP 1: Starting eval for {model_name} (P{process_id}, GPU{gpu_id}) ---")
    print(f"Params: BS={batch_size}, Dataset='{dataset_name}', Config='{dataset_config}', Split='{dataset_split}', Resume={resume}")

    # Metric is loaded globally, check if it's available
    if accuracy_metric_gpqa_loaded is None:
        print("ERROR (GPQA): Accuracy metric (accuracy_metric_gpqa_loaded) not available. Skipping.")
        return {"GPQA": 0.0, "error": "Accuracy metric global load failed"}

    raw_full_dataset = None
    try:
        if gpqa_script_path and os.path.exists(gpqa_script_path):
             raw_full_dataset = load_dataset(gpqa_script_path, name=dataset_config, data_dir=gpqa_data_dir, split=dataset_split, trust_remote_code=True)
        else:
            raw_full_dataset = load_dataset(dataset_name, name=dataset_config, split=dataset_split, trust_remote_code=True)
        print(f"DEBUG GPQA STEP 3: Loaded raw dataset '{dataset_name}/{dataset_config}/{dataset_split}'. Size: {len(raw_full_dataset) if raw_full_dataset else 'None'}")
    except Exception as e:
        print(f"P{process_id}: FATAL - Error loading GPQA dataset: {e}. Check config/split. Skipping.");
        return {"GPQA": 0.0, "error": f"Dataset load fail: {e}"}

    print(f"DEBUG GPQA STEP 4: Processing GPQA docs...")
    processed_dataset = process_gpqa_docs(raw_full_dataset, dataset_name_for_id=f"{dataset_name}_{dataset_config}")
    if not processed_dataset or len(processed_dataset) == 0: print(f"DEBUG GPQA STEP 4.1: No examples after process_gpqa_docs. Exiting."); return {"GPQA": 0.0, "error": "Processing docs yielded no data"}
    print(f"DEBUG GPQA STEP 4.2: Finished processing docs. Count: {len(processed_dataset)}")

    dataset_subset_for_this_process = processed_dataset
    if num_gpus > 1:
        total_proc_ex = len(processed_dataset); ex_per_proc = total_proc_ex // num_gpus
        start_idx = process_id * ex_per_proc; end_idx = start_idx + ex_per_proc if process_id != num_gpus - 1 else total_proc_ex
        if start_idx >= end_idx : print(f"DEBUG GPQA STEP 5.1: No data for P{process_id} post-split."); return {"GPQA": 0.0, "error": "No data post-split"}
        dataset_subset_for_this_process = processed_dataset.select(range(start_idx, end_idx))
    print(f"DEBUG GPQA STEP 5: Subset for P{process_id}: {len(dataset_subset_for_this_process)} examples.")
    if len(dataset_subset_for_this_process) == 0: print(f"DEBUG GPQA STEP 5.2: Subset for P{process_id} empty. Exit."); return {"GPQA": 0.0, "error": "Subset empty"}

    checkpoint_filename = f"gpqa_results_proc{process_id}_gpu{gpu_id}.jsonl"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
    detailed_results_log: List[GPQAEvalResult] = []
    processed_gpqa_ids_from_checkpoint = set()

    if resume:
        detailed_results_log = load_checkpoint_gpqa(checkpoint_filepath)
        for res in detailed_results_log: processed_gpqa_ids_from_checkpoint.add(res.gpqa_id)
    elif os.path.exists(checkpoint_filepath):
        print(f"Proc {process_id}: Not resuming, removing old checkpoint {checkpoint_filepath}")
        try: os.remove(checkpoint_filepath)
        except OSError as e: print(f"Error removing old checkpoint: {e}")
    print(f"DEBUG GPQA STEP 6: Checkpoint. Loaded {len(detailed_results_log)} from checkpoint.")

    prompts_to_generate = []; original_processed_examples_for_generation: List[Dict] = []
    print(f"DEBUG GPQA STEP 7: Preparing prompts (subset size: {len(dataset_subset_for_this_process)})...")
    skipped_checkpoint = 0; skipped_invalid_processed = 0
    for i in range(len(dataset_subset_for_this_process)):
        ex = dataset_subset_for_this_process[i]
        if not ex.get("is_valid", False): skipped_invalid_processed +=1; continue
        ex_id = ex.get("gpqa_id", f"fallback_id_{i}_{process_id}")
        if ex_id in processed_gpqa_ids_from_checkpoint: skipped_checkpoint +=1; continue
        prompts_to_generate.append(format_gpqa_prompt(ex, FEWSHOT_EXAMPLES_GPQA))
        original_processed_examples_for_generation.append(ex)
    print(f"DEBUG GPQA STEP 7.1: Skipped {skipped_invalid_processed} invalid, {skipped_checkpoint} from checkpoint. Prompts to generate: {len(prompts_to_generate)}")

    if not prompts_to_generate:
        print(f"DEBUG GPQA STEP 7.2: No new examples to process.")
        if detailed_results_log:
            # --- Perform manual accuracy calculation from checkpoint ---
            correct = sum(1 for r in detailed_results_log if r.is_correct)
            total_valid = len(detailed_results_log)
            acc = (correct / total_valid) * 100 if total_valid > 0 else 0.0
            print(f"DEBUG GPQA STEP 7.3: Final GPQA Accuracy (from checkpoint): {acc:.2f}%")
            return {"GPQA": acc}
        return {"GPQA": 0.0, "error": "No new prompts & no checkpoint data"}
    
    print(f"DEBUG GPQA STEP 8: Starting inference for {len(prompts_to_generate)} new prompts, BS: {batch_size}")
    newly_generated_this_run: List[GPQAEvalResult] = []
    for i in tqdm(range(0, len(prompts_to_generate), batch_size), desc=f"P{process_id} Gen GPQA", file=sys.stdout, mininterval=0.5):
        batch_prompts = prompts_to_generate[i : i + batch_size]
        batch_original_examples = original_processed_examples_for_generation[i : i + batch_size]
        try:
            raw_pipe_outputs = pipe(batch_prompts, max_new_tokens=10, do_sample=False, temperature=0.0)
            for j, pipe_output_item in enumerate(raw_pipe_outputs):
                original_example = batch_original_examples[j]; raw_model_output = "#GenFail"
                if isinstance(pipe_output_item, list) and pipe_output_item and isinstance(pipe_output_item[0], dict): raw_model_output = pipe_output_item[0].get('generated_text', "#GenFail")
                elif isinstance(pipe_output_item, dict): raw_model_output = pipe_output_item.get('generated_text', "#GenFail")
                completion_only = raw_model_output.strip() # Assuming pipe returns_full_text=False
                predicted_label = extract_gpqa_answer_label(completion_only)
                is_correct = (predicted_label is not None and predicted_label == original_example["correct_label"])
                choices_list = [original_example[f"choice_{chr(65+k)}"] for k in range(4)]
                newly_generated_this_run.append(GPQAEvalResult(
                    gpqa_id=original_example["gpqa_id"], question=original_example["question_text"], choices=choices_list,
                    correct_answer_text=original_example["original_correct_answer_text"], correct_answer_label=original_example["correct_label"],
                    model_raw_output=raw_model_output, model_predicted_label=predicted_label, is_correct=is_correct ))
        except Exception as gen_e:
            print(f"ERROR (GPQA Proc {process_id}) Pipeline error batch {i//batch_size}: {gen_e}")
            for original_example in batch_original_examples:
                choices_list = [original_example[f"choice_{chr(65+k)}"] for k in range(4)]
                newly_generated_this_run.append(GPQAEvalResult(
                    gpqa_id=original_example["gpqa_id"], question=original_example["question_text"], choices=choices_list,
                    correct_answer_text=original_example["original_correct_answer_text"], correct_answer_label=original_example["correct_label"],
                    model_raw_output="#PipelineError", model_predicted_label=None, is_correct=False, error_message=str(gen_e)))
            if torch.cuda.is_available(): torch.cuda.empty_cache(); continue

    if newly_generated_this_run:
        detailed_results_log.extend(newly_generated_this_run)
        save_checkpoint_gpqa(checkpoint_filepath, detailed_results_log)

    print(f"DEBUG GPQA STEP 9: Inference complete. Total results in log: {len(detailed_results_log)}")
    if not detailed_results_log: print(f"DEBUG GPQA STEP 9.1: No results to compute metrics."); return {"GPQA": 0.0, "error": "No results for metrics"}

    # --- Using MANUAL ACCURACY CALCULATION ---
    correct_count = 0; valid_comparisons = 0
    for res_item in detailed_results_log:
        pred_label = res_item.model_predicted_label; true_label = res_item.correct_answer_label
        if true_label and not true_label.startswith("(ERROR"): # Only count if true label is valid
            valid_comparisons += 1
            if pred_label is not None and pred_label == true_label:
                correct_count += 1
    accuracy_score = (correct_count / valid_comparisons) * 100 if valid_comparisons > 0 else 0.0
    print(f"DEBUG GPQA STEP 10: Manual Accuracy: {correct_count}/{valid_comparisons} = {accuracy_score:.2f}%")
    # --- END MANUAL ACCURACY CALCULATION ---
            
    print(f"Proc {process_id} (GPU {gpu_id}) - Final GPQA Accuracy: {accuracy_score:.2f}% on {len(detailed_results_log)} logged items ({valid_comparisons} valid comparisons).")
    return {"GPQA": accuracy_score}


# --- Standalone Testing Block (Ensure it's up-to-date) ---
def _initialize_pipeline_for_gpqa_test(model_name_test: str, device_id_test: int):
    if "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id_test) 
    effective_device = 0 if torch.cuda.is_available() else "cpu" # Will be cuda:0 if CUDA_VISIBLE_DEVICES worked
    print(f"Test Mode: Initializing on effective device: {effective_device} (requested system GPU {device_id_test})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_test, trust_remote_code=True)
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if tokenizer.pad_token is None and tokenizer.pad_token_id is not None:
        # Ensure pad_token is set if only pad_token_id exists
        if tokenizer.pad_token_id < tokenizer.vocab_size: # Check if it's a valid token ID
             tokenizer.add_special_tokens({'pad_token': tokenizer.decode([tokenizer.pad_token_id])})
        else: # Fallback if pad_token_id is out of vocab (e.g. from a bad default)
             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')


    quant_config_test = BitsAndBytesConfig(load_in_8bit=True) if torch.cuda.is_available() else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_test, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto", # Let HF handle mapping for the visible device
        quantization_config=quant_config_test,
        trust_remote_code=True
    )
    # Important: return_full_text=False so that the output is only the generated part
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False) 
    print(f"Standalone GPQA Test: Pipeline initialized for {model_name_test}. Pipeline device: {pipe.device}")
    return pipe

if __name__ == "__main__":
    print("--- Running gpqa.py standalone for testing ---")
    test_model_name = "google/gemma-2b" 
    test_gpu_id_to_use = 0 
    test_checkpoint_dir = "test_gpqa_checkpoints"
    test_num_examples_slice = "[:10]" # e.g. "[:10]" or "" for full split

    # For "Idavidrein/gpqa", choose a config and a valid split for that config
    test_dataset_name = "Idavidrein/gpqa"
    # Available configs: ['gpqa_extended', 'gpqa_main', 'gpqa_diamond', 'gpqa_experts']
    test_dataset_config = "gpqa_main" # Let's test with gpqa_main
    # For "gpqa_main" config, the available split is "test"
    test_actual_split_name = "train" 
    
    final_test_split_str = f"{test_actual_split_name}{test_num_examples_slice}" # e.g., "test[:10]"

    print(f"DEBUG Standalone: Using dataset_name='{test_dataset_name}', config='{test_dataset_config}', split='{final_test_split_str}'")

    if os.path.exists(test_checkpoint_dir):
        import shutil
        try: shutil.rmtree(test_checkpoint_dir)
        except Exception as e_clean: print(f"Warn: Could not clean checkpoint dir {test_checkpoint_dir}: {e_clean}")    
    os.makedirs(test_checkpoint_dir, exist_ok=True)

    try:
        test_pipe = _initialize_pipeline_for_gpqa_test(test_model_name, test_gpu_id_to_use)
        if test_pipe:
            results = evaluate_gpqa(
                model_name=test_model_name, pipe=test_pipe, model_size_gb=0.0,
                batch_size=2,
                dataset_name=test_dataset_name, 
                dataset_config=test_dataset_config, 
                dataset_split=final_test_split_str,
                process_id=0, gpu_id=test_gpu_id_to_use, num_gpus=1,
                checkpoint_dir=test_checkpoint_dir, resume=False
            )
            print("\n--- Standalone GPQA Test Results ---")
            print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during standalone GPQA test: {e}")
        import traceback; traceback.print_exc()
    print("--- Standalone GPQA test finished ---")