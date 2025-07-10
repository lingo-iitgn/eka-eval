import os
import torch
from datasets import load_dataset, Dataset as HFDataset
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import numpy as np
import json
from typing import Union, Dict, List, Any # For Python < 3.10 type hints and clarity
import traceback

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = 'google/IndicGenBench_crosssum_in'
# Languages present in the 'lang' field of IndicGenBench_crosssum_in
TARGET_LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
# TARGET_LANGUAGES = ["as", "hi"] # For quicker testing

SPLIT_TO_USE = 'validation' 
# Set to None to process all samples for each language, or an integer for a fixed number.
NUM_SAMPLES_PER_LANGUAGE = None # Process ALL samples for each language
# NUM_SAMPLES_PER_LANGUAGE = 10 # For testing: process first 10 samples per language

MAX_INPUT_LENGTH = 1024 # Max length for tokenized input (prompt + article)
MAX_SUMMARY_LENGTH = 150 # Max *new* tokens for the generated summary
OUTPUT_DIR = "crosssum_eval_outputs_full_v2" # Directory for results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Setup Model and Tokenizer
# -----------------------------
print(f"--- Model and Tokenizer Setup ---")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Let device_map handle this. If set, ensure consistency.
torch.cuda.empty_cache() 

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to: '{tokenizer.eos_token}'")

print(f"Loading model: {MODEL_NAME}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading model: {e}")
    traceback.print_exc()
    exit()

# -----------------------------
# Helper and Prompt Functions
# -----------------------------
def get_indic_language_name(lang_code: str) -> str:
    lang_map = {
        "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
        "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
        "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
        "en": "English"
    }
    return lang_map.get(lang_code, lang_code.upper())

def format_prompt_for_crosssum(article_text: str, target_summary_lang_code: str) -> str:
    target_language_name = get_indic_language_name(target_summary_lang_code)
    source_language_name = "English" # Articles in IndicGenBench CrossSum are English
    
    prompt = (
        f"You are a multilingual summarization expert. "
        f"Your task is to read the following news article provided in {source_language_name} "
        f"and generate a concise summary of it strictly in the {target_language_name} language.\n\n"
        f"Article ({source_language_name}):\n{article_text}\n\n"
        f"Provide the summary ONLY in {target_language_name}.\n"
        f"Summary ({target_language_name}):"
    )
    return prompt

# -----------------------------
# Metric and Results Storage
# -----------------------------
rouge_metric = evaluate.load("rouge")
language_overall_scores = {} # Stores final ROUGE-L F-measure per language
all_rouge_l_fmeasures_list = [] # For calculating overall average
all_examples_log_overall = [] # For a single JSON log of all examples

# -----------------------------
# Data Loading
# -----------------------------
print(f"Loading full dataset: {DATASET_NAME}, split: {SPLIT_TO_USE}")
try:
    full_indicgenbench_dataset = load_dataset(DATASET_NAME, split=SPLIT_TO_USE, trust_remote_code=True)
    print(f"Full dataset loaded. Total rows (each with canary+examples): {len(full_indicgenbench_dataset)}")
    
    actual_data_points = []
    if 'examples' in full_indicgenbench_dataset.features:
        for row in full_indicgenbench_dataset:
            if 'examples' in row and isinstance(row['examples'], dict):
                actual_data_points.append(row['examples'])
    
    if not actual_data_points:
        print("FATAL: No data points extracted from the 'examples' field. Check dataset structure.")
        exit()

    clean_dataset_all_langs = HFDataset.from_list(actual_data_points)
    print(f"Created clean dataset with {len(clean_dataset_all_langs)} examples across all languages.")
    print(f"Clean dataset features: {clean_dataset_all_langs.features}")

except Exception as e:
    print(f"FATAL: Error loading or processing dataset: {e}")
    traceback.print_exc()
    exit()

# -----------------------------
# Main Evaluation Loop
# -----------------------------
print(f"\n--- Starting IndicGenBench_CrossSum_IN Evaluation ---")
print(f"Target Languages: {', '.join(TARGET_LANGUAGES)}")
print(f"Samples per language: {'All' if NUM_SAMPLES_PER_LANGUAGE is None else NUM_SAMPLES_PER_LANGUAGE}")
print(f"Results will be saved in: {OUTPUT_DIR}")
print("======================================================================")

for lang_code in TARGET_LANGUAGES:
    display_name = f"crosssum_{lang_code}"
    print(f"\n--- Evaluating Language: {lang_code.upper()} (as {display_name}) ---")

    current_lang_predictions: List[str] = []
    current_lang_references: List[str] = []
    current_lang_detailed_logs: List[Dict[str, Any]] = []

    try:
        lang_specific_dataset = clean_dataset_all_langs.filter(
            lambda example: example.get('lang') == lang_code
        )

        if len(lang_specific_dataset) == 0:
            print(f"  No samples found for language '{lang_code}' after filtering. Skipping.")
            language_overall_scores[display_name] = {"rougeL_fmeasure": 0.0, "info": "No samples found"}
            continue

        num_to_eval_this_lang = len(lang_specific_dataset) 
        if NUM_SAMPLES_PER_LANGUAGE is not None and NUM_SAMPLES_PER_LANGUAGE > 0: # if a specific positive number is set
            num_to_eval_this_lang = min(NUM_SAMPLES_PER_LANGUAGE, len(lang_specific_dataset))
        
        if num_to_eval_this_lang == 0:
            print(f"  No samples to evaluate for '{lang_code}' (0 configured or available). Skipping.")
            language_overall_scores[display_name] = {"rougeL_fmeasure": 0.0, "info": "0 samples evaluated"}
            # Save empty detailed log for this language
            lang_log_filename_empty = os.path.join(OUTPUT_DIR, f"{display_name}_detailed_results.json")
            with open(lang_log_filename_empty, "w", encoding="utf-8") as f_json_empty:
                json.dump([], f_json_empty, ensure_ascii=False, indent=2)
            continue
            
        subset_to_eval = lang_specific_dataset.select(range(num_to_eval_this_lang))
        print(f"  Evaluating on {len(subset_to_eval)} samples for '{lang_code}'.")

        for i, example_data_point in tqdm(enumerate(subset_to_eval), desc=f"Eval {lang_code.upper()}", total=len(subset_to_eval)):
            english_article = example_data_point.get("text")
            target_summary = example_data_point.get("summary")
            # lang_code_from_example = example_data_point.get("lang") # Should match lang_code

            log_entry = {
                "language": lang_code,
                "example_index_in_subset": i,
                "source_url": example_data_point.get("source_url", "N/A"),
                "target_url": example_data_point.get("target_url", "N/A"),
                "english_article_snippet": english_article[:200]+"..." if english_article else "N/A",
                "reference_summary": target_summary,
                "prompt_preview": "",
                "generated_summary_raw": "[SKIPPED]",
            }

            if not english_article or not target_summary:
                print(f"    WARNING: Skipping example {i} for '{lang_code}' due to missing article or summary.")
                current_lang_detailed_logs.append(log_entry)
                continue

            try:
                prompt = format_prompt_for_crosssum(english_article, lang_code)
                log_entry["prompt_preview"] = prompt[:300]+"..."

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(model.device)
                prompt_token_length = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=MAX_SUMMARY_LENGTH,
                        num_beams=4, 
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                generated_ids = outputs[0][prompt_token_length:]
                generated_summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                log_entry["generated_summary_raw"] = generated_summary

                current_lang_predictions.append(generated_summary)
                current_lang_references.append(target_summary)

                if i < 2 : # Print first few examples for this language
                    print(f"\n--- Example {i+1} for {lang_code.upper()} ---")
                    print("Prompt (first 300 chars):", prompt[:300] + "...")
                    print("Reference summary:", target_summary)
                    print("Generated summary:", generated_summary)
                    print("-" * 80)

            except Exception as e_gen:
                print(f"    ERROR processing example {i} (lang: {lang_code}): {e_gen}")
                log_entry["generated_summary_raw"] = f"[ERROR: {str(e_gen)}]"
            current_lang_detailed_logs.append(log_entry)
        
        all_examples_log_overall.extend(current_lang_detailed_logs) # Add this lang's logs to overall log

        if current_lang_predictions and current_lang_references:
            print(f"  Computing ROUGE for {lang_code.upper()} with {len(current_lang_predictions)} pairs.")
            results = rouge_metric.compute(
                predictions=current_lang_predictions, 
                references=current_lang_references, 
                use_stemmer=True
            )
            
            rouge_l_fmeasure = 0.0
            if isinstance(results.get('rougeL'), float): # Direct float value
                rouge_l_fmeasure = results['rougeL']
            elif hasattr(results.get('rougeL'), 'mid') and hasattr(results['rougeL'].mid, 'fmeasure'): # AggregatedScore object
                rouge_l_fmeasure = results['rougeL'].mid.fmeasure
            else:
                print(f"    WARNING: Unexpected ROUGE output structure for {lang_code}: {results.get('rougeL')}")

            language_overall_scores[display_name] = {"rougeL_fmeasure": rouge_l_fmeasure, "full_rouge_results": results}
            all_rouge_l_fmeasures_list.append(rouge_l_fmeasure)
            print(f"  ✅ ROUGE-L F-measure for {lang_code.upper()}: {rouge_l_fmeasure:.4f}")
        elif len(subset_to_eval) > 0:
            print(f"  ⚠️ No valid predictions/references generated for '{lang_code}'. ROUGE-L set to 0.0000.")
            language_overall_scores[display_name] = {"rougeL_fmeasure": 0.0, "info": "No valid pairs for metric"}
            all_rouge_l_fmeasures_list.append(0.0)
        else: # Should not be reached if num_to_eval_this_lang > 0
            print(f"  No examples were processed for '{lang_code}'.")
            language_overall_scores[display_name] = None 
        
        # Save per-language detailed log
        lang_log_filename = os.path.join(OUTPUT_DIR, f"{display_name}_detailed_results.json")
        try:
            with open(lang_log_filename, "w", encoding="utf-8") as f_out:
                json.dump(current_lang_detailed_logs, f_out, ensure_ascii=False, indent=2)
            print(f"  Detailed logs for {lang_code.upper()} saved to {lang_log_filename}")
        except Exception as e_json:
            print(f"  Error saving JSON log for {lang_code.upper()}: {e_json}")

    except Exception as e_outer_lang:
        print(f"CRITICAL ERROR processing language {lang_code}: {e_outer_lang}")
        traceback.print_exc()
        language_overall_scores[display_name] = {"rougeL_fmeasure": 0.0, "error": str(e_outer_lang)}

# Save overall detailed log for ALL examples processed
overall_log_filename = os.path.join(OUTPUT_DIR, f"all_langs_crosssum_eval_{MODEL_NAME.replace('/','_')}_details.json")
try:
    with open(overall_log_filename, "w", encoding="utf-8") as f_overall:
        json.dump(all_examples_log_overall, f_overall, ensure_ascii=False, indent=2)
    print(f"\n💾 Overall detailed logs for all processed examples saved to {overall_log_filename}")
except Exception as e_json_all:
    print(f"Error saving overall detailed JSON log: {e_json_all}")

# Final Averaged Results
print("\n======================================================================")
print("🏆 Final IndicGenBench_CrossSum_IN Evaluation Summary (ROUGE-L F-measure) 🏆")
print(f"Model: {MODEL_NAME}")
print("======================================================================")
for disp_name_key, scores_dict_val in language_overall_scores.items():
    if scores_dict_val and "rougeL_fmeasure" in scores_dict_val and "error" not in scores_dict_val:
        print(f"  - {disp_name_key}: {scores_dict_val['rougeL_fmeasure']:.4f} {scores_dict_val.get('info','')}")
    elif scores_dict_val and "error" in scores_dict_val:
         print(f"  - {disp_name_key}: ERROR ({scores_dict_val.get('error')})")
    else: # Catches None or other unexpected structures
        print(f"  - {disp_name_key}: No Data / Not Evaluated or Error")

valid_scores = [score for score in all_rouge_l_fmeasures_list if isinstance(score, (float, int))]
if valid_scores:
    overall_average_rouge_l = np.mean(valid_scores)
    print(f"\n📈 Overall Average ROUGE-L F-measure across {len(valid_scores)} successfully evaluated languages: {overall_average_rouge_l:.4f}")
else:
    print("\n⚠️ No valid ROUGE-L scores to compute an overall average.")

print("\nCrossSum Evaluation complete.")