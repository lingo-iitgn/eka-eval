from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate
import torch
import re
import numpy as np
from typing import Union # <--- IMPORT THIS

MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "sarvamai/mmlu-indic"
TARGET_LANGUAGES = ["hi", "bn", "kn", "en", "gu", "ml", "mr", "or", "pa", "ta", "te"]
SPLIT = "validation"
NUM_SAMPLES_PER_LANGUAGE = -1 # Use -1 to process all samples for the split
MAX_NEW_TOKENS_FOR_ANSWER = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to tokenizer.eos_token: '{tokenizer.eos_token}'")

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto", # Automatically uses available CUDA if DEVICE is cuda, or CPU
    trust_remote_code=True
)
model.eval()
print("Model and tokenizer loaded successfully.")

def format_prompt_for_mmlu(question, choices):
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

# Corrected function signature for older Python versions
def parse_predicted_answer(raw_generated_text: str, current_lang_config: str) -> Union[str, None]:
    if not raw_generated_text:
        return None
    first_line_output = raw_generated_text.split('\n')[0].strip()
    if not first_line_output:
        return None
    
    predicted_letter_english = None
    # Using a regex that matches English A-D or specific Hindi letters for A-D sounds
    # (?:[.)\s]|$) ensures it's followed by punctuation, space, or end of line (standalone letter)
    regex_pattern = r"([A-Dएअबीबसीसडीद])(?:[.)\s]|$)"
    match_start = re.match(regex_pattern, first_line_output, re.IGNORECASE)
    
    if match_start:
        found_char = match_start.group(1).upper()
        # Apply Hindi map only if current language is Hindi and a known Hindi char was found
        if current_lang_config.lower() == "hi" and found_char in HINDI_TO_ENGLISH_MAP:
            predicted_letter_english = HINDI_TO_ENGLISH_MAP.get(found_char)
        elif found_char in "ABCD": # If it's an English letter A-D directly
             predicted_letter_english = found_char
        
        # Validate that the result (after potential mapping) is indeed one of A, B, C, D
        if predicted_letter_english and predicted_letter_english not in "ABCD":
            predicted_letter_english = None 
            
    if not predicted_letter_english: # Fallback if first char wasn't a direct match
        # Search for a standalone English letter A, B, C, or D
        match_any = re.search(r"\b([A-D])\b", first_line_output, re.IGNORECASE)
        if match_any:
            predicted_letter_english = match_any.group(1).upper()
            
    return predicted_letter_english

accuracy_metric = evaluate.load("accuracy")
language_accuracies = {}
all_individual_accuracies = []

print(f"\n--- Starting MMLU-Indic Evaluation ---")
print(f"Target Languages: {', '.join(TARGET_LANGUAGES)}")
print(f"Samples per language: {'All' if NUM_SAMPLES_PER_LANGUAGE == -1 else NUM_SAMPLES_PER_LANGUAGE}")
print("======================================================================")

for lang_code in TARGET_LANGUAGES:
    print(f"\n--- Evaluating Language: {lang_code.upper()} ---")
    predictions_indices = []
    reference_indices = []
    try:
        print(f"  Loading dataset: {DATASET_NAME}, config: {lang_code}, split: {SPLIT}")
        # For MMLU-Indic, lang_code is the config name
        dataset = load_dataset(DATASET_NAME, name=lang_code, split=SPLIT, trust_remote_code=True) # Added trust_remote_code
        
        current_num_samples = len(dataset) if NUM_SAMPLES_PER_LANGUAGE == -1 else min(NUM_SAMPLES_PER_LANGUAGE, len(dataset))
        
        if current_num_samples == 0:
            print(f"  No samples found or configured for '{lang_code}'. Skipping.")
            language_accuracies[lang_code] = None
            continue
            
        subset = dataset.select(range(current_num_samples))
        print(f"  Evaluating on {len(subset)} samples for '{lang_code}'.")

        for example_idx, example in tqdm(enumerate(subset), desc=f"Eval {lang_code.upper()}", total=len(subset)):
            question = example["question"]
            choices = example["choices"] # List of choice strings
            correct_answer_index = example["answer"] # Integer index (0, 1, 2, or 3)

            if not all([question, isinstance(choices, list), len(choices) > 0, isinstance(correct_answer_index, int)]):
                print(f"    WARNING: Skipping example {example_idx} for '{lang_code}' due to malformed data.")
                continue


            prompt = format_prompt_for_mmlu(question, choices)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE) # Removed padding=False as pipeline handles it
            prompt_token_length = inputs.input_ids.shape[1]
            
            generated_text_for_parsing = "" # Default to empty string
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
            except Exception as e_gen:
                print(f"    ERROR during generation for example {example_idx}, lang '{lang_code}': {e_gen}")
                generated_text_for_parsing = "[GENERATION_ERROR]"
            
            predicted_letter = parse_predicted_answer(generated_text_for_parsing, lang_code)
            
            predicted_answer_index = -1 # Default for unparsed or invalid
            if predicted_letter and predicted_letter in "ABCD":
                predicted_answer_index = ord(predicted_letter) - ord('A')
            
            predictions_indices.append(predicted_answer_index)
            reference_indices.append(correct_answer_index)

        if predictions_indices and reference_indices: # Ensure lists are not empty
            # Filter out pairs where prediction was unparseable (-1), if desired, or score them as wrong.
            # Current approach scores -1 as wrong, which is standard for accuracy.
            acc_results = accuracy_metric.compute(predictions=predictions_indices, references=reference_indices)
            lang_accuracy = acc_results['accuracy']
            language_accuracies[lang_code] = lang_accuracy
            all_individual_accuracies.append(lang_accuracy)
            print(f"  ✅ Accuracy for {lang_code.upper()}: {lang_accuracy:.4f} on {len(predictions_indices)} examples.")
        elif current_num_samples > 0 :
             print(f"  ⚠️ No valid predictions made for '{lang_code}' (processed {current_num_samples} examples). Accuracy set to 0.0000.")
             language_accuracies[lang_code] = 0.0
             all_individual_accuracies.append(0.0)
        else:
            print(f"  No examples were processed for '{lang_code}'.")
            language_accuracies[lang_code] = None # Or 0.0 if preferred for averaging

    except Exception as e_lang:
        print(f"CRITICAL ERROR processing language {lang_code}: {e_lang}")
        import traceback
        traceback.print_exc()
        language_accuracies[lang_code] = None

print("\n--- Final MMLU-Indic Results ---")
for lang, acc in language_accuracies.items():
    if acc is not None:
        print(f"  - {lang.upper()}: {acc:.4f}")
    else:
        print(f"  - {lang.upper()}: ERROR or No Data")

valid_accuracies = [acc for acc in all_individual_accuracies if acc is not None]
if valid_accuracies:
    overall_average_accuracy = np.mean(valid_accuracies)
    print(f"\nOverall Average Accuracy: {overall_average_accuracy:.4f} across {len(valid_accuracies)} languages.")
else:
    print("\nNo valid accuracies to compute an overall average.")

print("\nMMLU-Indic Evaluation complete.")