## https://github.com/AI4Bharat/MILU
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate
import torch
import re
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "ai4bharat/MILU"

TARGET_LANGUAGES = [
    "Bengali", "English", "Gujarati", "Hindi", "Kannada", 
    "Malayalam", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"
]
SPLIT_TO_USE = 'validation' 

NUM_SAMPLES_PER_LANGUAGE = None
MAX_NEW_TOKENS_FOR_ANSWER = 10

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to tokenizer.eos_token: '{tokenizer.eos_token}'")

print(f"Loading model: {MODEL_NAME}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def format_prompt_for_milu(question, option1, option2, option3, option4):
    choices = [option1, option2, option3, option4]
    choices_text = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""प्रश्न: {question}
विकल्प:
{choices_text}
सही उत्तर का अक्षर (A, B, C, या D) क्या है?
उत्तर:"""
    return prompt


HINDI_TO_ENGLISH_MAP = {
    "ए": "A", "अ": "A", "ए.": "A", "अ.": "A",
    "बी": "B", "ब": "B", "बी.": "B", "ब.": "B",
    "सी": "C", "स": "C", "सी.": "C", "स.": "C",
    "डी": "D", "द": "D", "डी.": "D", "द.": "D"
}

def parse_predicted_answer(raw_generated_text: str, current_lang_config: str) -> str :
    if not raw_generated_text:
        return None
    first_line_output = raw_generated_text.split('\n')[0].strip()
    if not first_line_output:
        return None

    predicted_letter_english = None
    # Regex for A-D (English) or common Hindi representations for A-D at the start.
    regex_pattern = r"([A-Dएअबीबसीसडीद])(?:[.)\s]|$)" 
    
    match_start = re.match(regex_pattern, first_line_output, re.IGNORECASE)
    if match_start:
        found_char = match_start.group(1).upper()
        # Apply Hindi map only if current language is Hindi (config name) and a Hindi char was found
        if current_lang_config.lower() == "hindi" and found_char in HINDI_TO_ENGLISH_MAP:
            predicted_letter_english = HINDI_TO_ENGLISH_MAP.get(found_char)
        elif found_char in "ABCD": # If it's an English letter directly
             predicted_letter_english = found_char
        
        if predicted_letter_english and predicted_letter_english not in "ABCD":
            predicted_letter_english = None
            
    if not predicted_letter_english:
        match_any = re.search(r"\b([A-D])\b", first_line_output, re.IGNORECASE)
        if match_any:
            predicted_letter_english = match_any.group(1).upper()

    return predicted_letter_english

def target_to_index(target_str: str) -> int:
    if not target_str: return -1
    target_str_lower = target_str.lower()
    if target_str_lower == "option1" or target_str_lower == "a": return 0
    if target_str_lower == "option2" or target_str_lower == "b": return 1
    if target_str_lower == "option3" or target_str_lower == "c": return 2
    if target_str_lower == "option4" or target_str_lower == "d": return 3
    # If it's already a letter A-D from parsing
    if 'A' <= target_str.upper() <= 'D':
        return ord(target_str.upper()) - ord('A')
    return -1 # Invalid

# -----------------------------
# Main Evaluation Loop
# -----------------------------
accuracy_metric = evaluate.load("accuracy")
language_accuracies = {}
all_individual_accuracies = []

print(f"\nStarting MILU evaluation for languages: {', '.join(TARGET_LANGUAGES)}")
print("======================================================================")

for lang_config_name in TARGET_LANGUAGES: # lang_config_name is "English", "Hindi", etc.
    print(f"\n--- Evaluating Language Config: {lang_config_name} ---")
    
    predictions_indices = []
    reference_indices = []
    raw_model_outputs_log = []

    try:
        print(f"Loading dataset: {DATASET_NAME}, config: {lang_config_name}, split: {SPLIT_TO_USE}")
        # For MILU, language name is the config name
        current_dataset = load_dataset(DATASET_NAME, name=lang_config_name, split=SPLIT_TO_USE)
        
        num_to_sample = NUM_SAMPLES_PER_LANGUAGE
        if num_to_sample is None or num_to_sample > len(current_dataset):
            num_to_sample = len(current_dataset)
            if NUM_SAMPLES_PER_LANGUAGE is not None:
                 print(f"Using all {num_to_sample} available samples for '{lang_config_name}'.")
        elif num_to_sample == 0:
            print(f"NUM_SAMPLES_PER_LANGUAGE is 0. Skipping '{lang_config_name}'.")
            language_accuracies[lang_config_name] = None
            continue
        
        if num_to_sample == 0:
            print(f"No samples to evaluate for language config '{lang_config_name}'. Skipping.")
            language_accuracies[lang_config_name] = None
            continue
            
        subset = current_dataset.select(range(num_to_sample)) if num_to_sample < len(current_dataset) else current_dataset
        print(f"Using {len(subset)} samples for '{lang_config_name}' evaluation.")

        for example_idx, example in tqdm(enumerate(subset), desc=f"Eval {lang_config_name}", total=len(subset)):
            question = example.get("question", "")
            option1 = example.get("option1", "")
            option2 = example.get("option2", "")
            option3 = example.get("option3", "")
            option4 = example.get("option4", "")
            target_str = example.get("target", "") # e.g., "option3"

            if not all([question, option1, option2, option3, option4, target_str]):
                print(f"Skipping example {example_idx} in lang config '{lang_config_name}' due to missing data.")
                raw_model_outputs_log.append("[SKIPPED - MISSING DATA]")
                predictions_indices.append(-1)
                reference_indices.append(target_to_index(target_str))
                continue

            prompt = format_prompt_for_milu(question, option1, option2, option3, option4)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False).to(model.device)
            prompt_token_length = inputs.input_ids.shape[1]

            generated_text_for_parsing = "[GENERATION ERROR]"
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS_FOR_ANSWER,
                        num_beams=1,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                generated_ids = outputs[0][prompt_token_length:]
                generated_text_for_parsing = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            except Exception as e:
                print(f"\nError during model generation for lang config '{lang_config_name}', example {example_idx}: {e}")
            
            raw_model_outputs_log.append(generated_text_for_parsing)
            
            predicted_letter = parse_predicted_answer(generated_text_for_parsing, lang_config_name)
            
            pred_idx = target_to_index(predicted_letter) # Converts 'A'->0, 'B'->1, etc. or None -> -1
            ref_idx = target_to_index(target_str)       # Converts "option1"->0, "option2"->1, etc.
            
            predictions_indices.append(pred_idx)
            reference_indices.append(ref_idx)
        
        valid_indices_pairs = [(p, r) for p, r in zip(predictions_indices, reference_indices) if r != -1]

        if valid_indices_pairs:
            valid_predictions = [p for p,r in valid_indices_pairs]
            valid_references = [r for p,r in valid_indices_pairs]

            acc_results = accuracy_metric.compute(
                predictions=valid_predictions, 
                references=valid_references
            )
            lang_accuracy = acc_results['accuracy']
            language_accuracies[lang_config_name] = lang_accuracy
            all_individual_accuracies.append(lang_accuracy)
            print(f"✅ Accuracy for {lang_config_name}: {lang_accuracy:.2%} on {len(valid_indices_pairs)} valid reference examples (out of {len(subset)} total examples).")
        elif len(subset) > 0:
            print(f"⚠️ No valid reference examples or predictions for language config '{lang_config_name}' out of {len(subset)} examples. Accuracy is 0.00%.")
            language_accuracies[lang_config_name] = 0.0
            all_individual_accuracies.append(0.0)
        else:
            print(f"No examples processed for language config '{lang_config_name}'.")
            language_accuracies[lang_config_name] = None


        if raw_model_outputs_log and len(subset) > 0:
            print(f"\n🔍 Detailed Results for first example in {lang_config_name}:\n")
            idx_to_show = 0
            ex_data = subset[idx_to_show]
            q = ex_data.get("question", "N/A")
            opts = [ex_data.get(f"option{k+1}", "") for k in range(4)]
            
            ref_idx_disp = reference_indices[idx_to_show] if idx_to_show < len(reference_indices) else -1
            pred_idx_disp = predictions_indices[idx_to_show] if idx_to_show < len(predictions_indices) else -1
            raw_out = raw_model_outputs_log[idx_to_show] if idx_to_show < len(raw_model_outputs_log) else ""

            corr_lttr_disp = chr(ord('A') + ref_idx_disp) if 0 <= ref_idx_disp <= 3 else ex_data.get("target", "N/A")
            pred_lttr_disp = chr(ord('A') + pred_idx_disp) if 0 <= pred_idx_disp <= 3 else '?'

            print(f"--- Lang Config {lang_config_name} Example {idx_to_show+1}/{len(subset)} ---")
            print(f"📌 प्रश्न (Question): {q}\n")
            print("विकल्प (Choices):")
            for j, choice_text in enumerate(opts):
                prefix = chr(ord('A') + j)
                correct_marker = " ✅ (सही)" if j == ref_idx_disp else ""
                model_marker = " 🔸 (मॉडल)" if j == pred_idx_disp else ""
                print(f"  {prefix}. {choice_text}{correct_marker}{model_marker}")

            print(f"\n✔️   सही उत्तर (Correct Answer): {corr_lttr_disp} (Target: {ex_data.get('target', 'N/A')})")
            print(f"🤖 मॉडल का अनुमान (Model's Prediction): {pred_lttr_disp}")
            print(f"🔍 मॉडल का रॉ आउटपुट (Raw Model Output): '{raw_out}'")
            print("-" * 70)

    except Exception as e:
        print(f"Major error processing language config {lang_config_name}: {e}")
        import traceback
        traceback.print_exc()
        language_accuracies[lang_config_name] = None

# -----------------------------
# Final Averaged Results
# -----------------------------
print("\n======================================================================")
print("🏆 Final ai4bharat/MILU Evaluation Summary 🏆")
print("======================================================================")
for lang_conf, acc in language_accuracies.items():
    if acc is not None:
        print(f"  - {lang_conf}: {acc:.2%}")
    else:
        print(f"  - {lang_conf}: Error, No Data, or 0 Samples Evaluated")

valid_accuracies = [acc for acc in all_individual_accuracies if acc is not None]
if valid_accuracies:
    overall_average_accuracy = np.mean(valid_accuracies)
    print(f"\n📈 Overall Average Accuracy across {len(valid_accuracies)} successfully evaluated language configs: {overall_average_accuracy:.2%}")
else:
    print("\n⚠️ No valid accuracies to compute an overall average.")

print("\nEvaluation complete.")