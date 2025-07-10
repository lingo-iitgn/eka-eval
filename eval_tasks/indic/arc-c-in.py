import os
import json
import re
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "sarvamai/arc-challenge-indic"
TARGET_LANGUAGES = ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
PREFERRED_SPLITS = ['validation']
NUM_SAMPLES_PER_LANGUAGE = None
MAX_NEW_TOKENS_FOR_ANSWER = 10

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to tokenizer.eos_token: '{tokenizer.eos_token}'")

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

def format_prompt_for_arc(question, choice_dict):
    if not isinstance(choice_dict, dict) or 'label' not in choice_dict or 'text' not in choice_dict:
        return None
    if not (isinstance(choice_dict['label'], list) and isinstance(choice_dict['text'], list) and
            len(choice_dict['label']) == len(choice_dict['text'])):
        return None
    choices_text_parts = []
    expected_labels = ['A', 'B', 'C', 'D']
    for label, text in zip(choice_dict['label'], choice_dict['text']):
        if label.upper() in expected_labels:
            choices_text_parts.append(f"{label.upper()}. {text}")
    if not choices_text_parts:
        return None
    choices_text = "\n".join(choices_text_parts)
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

def parse_predicted_answer(raw_generated_text, current_lang_config: str):
    if not raw_generated_text:
        return None
    first_line_output = raw_generated_text.split('\n')[0].strip()
    if not first_line_output:
        return None
    predicted_letter_english = None
    regex_pattern = r"([A-Dएअबीबसीसडीद])(?:[.)\s]|$)"
    match_start = re.match(regex_pattern, first_line_output, re.IGNORECASE)
    if match_start:
        found_char = match_start.group(1).upper()
        if current_lang_config == "hi" and found_char in HINDI_TO_ENGLISH_MAP:
            predicted_letter_english = HINDI_TO_ENGLISH_MAP.get(found_char)
        elif found_char in "ABCD":
            predicted_letter_english = found_char
        if predicted_letter_english and predicted_letter_english not in "ABCD":
            predicted_letter_english = None
    if not predicted_letter_english:
        match_any = re.search(r"\b([A-D])\b", first_line_output, re.IGNORECASE)
        if match_any:
            predicted_letter_english = match_any.group(1).upper()
    return predicted_letter_english

def letter_to_index(letter: str):
    if letter and 'A' <= letter.upper() <= 'D':
        return ord(letter.upper()) - ord('A')
    return -1

accuracy_metric = evaluate.load("accuracy")
language_accuracies = {}
all_individual_accuracies = []
all_results = {}

print(f"\nStarting ARC-Challenge-Indic evaluation for languages: {', '.join(TARGET_LANGUAGES)}")
print("======================================================================")

for lang_code in TARGET_LANGUAGES:
    print(f"\n--- Evaluating Language: {lang_code.upper()} ---")
    predictions_indices = []
    reference_indices = []
    per_example_results = []
    try:
        loaded_data = load_dataset(DATASET_NAME, name=lang_code)
        actual_dataset_to_use = None
        if isinstance(loaded_data, dict):
            for split_name in PREFERRED_SPLITS:
                if split_name in loaded_data:
                    actual_dataset_to_use = loaded_data[split_name]
                    break
            if not actual_dataset_to_use and loaded_data:
                first_available_split = list(loaded_data.keys())[0]
                actual_dataset_to_use = loaded_data[first_available_split]
            elif not loaded_data:
                language_accuracies[lang_code] = None
                continue
        elif loaded_data is not None:
            actual_dataset_to_use = loaded_data
        else:
            language_accuracies[lang_code] = None
            continue

        num_to_sample = NUM_SAMPLES_PER_LANGUAGE
        if num_to_sample is None or num_to_sample > len(actual_dataset_to_use):
            num_to_sample = len(actual_dataset_to_use)
        elif num_to_sample == 0:
            language_accuracies[lang_code] = None
            continue

        if num_to_sample == 0:
            language_accuracies[lang_code] = None
            continue

        subset = actual_dataset_to_use.select(range(num_to_sample)) if num_to_sample < len(actual_dataset_to_use) else actual_dataset_to_use

        for example_idx, example in tqdm(enumerate(subset), desc=f"Eval {lang_code.upper()}", total=len(subset)):
            question = example.get("question", "")
            choices_dict = example.get("choices", {})
            correct_answer_letter = example.get("answerKey", "").upper()
            if not question or not choices_dict or not correct_answer_letter or correct_answer_letter not in "ABCD":
                predictions_indices.append(-1)
                reference_indices.append(letter_to_index(correct_answer_letter))
                per_example_results.append({
                    "index": example_idx,
                    "question": question,
                    "choices": choices_dict,
                    "reference": correct_answer_letter,
                    "prediction": None,
                    "raw_output": "[SKIPPED - MISSING/INVALID DATA]"
                })
                continue
            prompt = format_prompt_for_arc(question, choices_dict)
            if prompt is None:
                predictions_indices.append(-1)
                reference_indices.append(letter_to_index(correct_answer_letter))
                per_example_results.append({
                    "index": example_idx,
                    "question": question,
                    "choices": choices_dict,
                    "reference": correct_answer_letter,
                    "prediction": None,
                    "raw_output": "[SKIPPED - PROMPT FORMATTING ERROR]"
                })
                continue
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
                pass
            predicted_letter = parse_predicted_answer(generated_text_for_parsing, lang_code)
            pred_idx = letter_to_index(predicted_letter)
            ref_idx = letter_to_index(correct_answer_letter)
            predictions_indices.append(pred_idx)
            reference_indices.append(ref_idx)
            per_example_results.append({
                "index": example_idx,
                "question": question,
                "choices": choices_dict,
                "reference": correct_answer_letter,
                "prediction": predicted_letter,
                "raw_output": generated_text_for_parsing
            })

        valid_indices_pairs = [(p, r) for p, r in zip(predictions_indices, reference_indices) if r != -1]
        if valid_indices_pairs:
            valid_predictions = [p for p, r in valid_indices_pairs]
            valid_references = [r for p, r in valid_indices_pairs]
            acc_results = accuracy_metric.compute(
                predictions=valid_predictions,
                references=valid_references
            )
            lang_accuracy = acc_results['accuracy']
            language_accuracies[lang_code] = lang_accuracy
            all_individual_accuracies.append(lang_accuracy)
        elif len(subset) > 0:
            language_accuracies[lang_code] = 0.0
            all_individual_accuracies.append(0.0)
        else:
            language_accuracies[lang_code] = None

        all_results[lang_code] = {
            "accuracy": language_accuracies[lang_code],
            "per_example": per_example_results
        }

    except Exception as e:
        language_accuracies[lang_code] = None
        all_results[lang_code] = {
            "accuracy": None,
            "per_example": [],
            "error": str(e)
        }

print("\n======================================================================")
print("🏆 Final ARC-Challenge-Indic Evaluation Summary 🏆")
print("======================================================================")
for lang, acc in language_accuracies.items():
    if acc is not None:
        print(f"  - {lang.upper()}: {acc:.2%}")
    else:
        print(f"  - {lang.upper()}: Error, No Data, or 0 Samples Evaluated")

valid_accuracies = [acc for acc in all_individual_accuracies if acc is not None]
if valid_accuracies:
    overall_average_accuracy = float(np.mean(valid_accuracies))
    print(f"\n📈 Overall Average Accuracy across {len(valid_accuracies)} successfully evaluated languages: {overall_average_accuracy:.2%}")
    all_results["overall_average_accuracy"] = overall_average_accuracy
else:
    print("\n⚠️ No valid accuracies to compute an overall average.")
    all_results["overall_average_accuracy"] = None

print("\nEvaluation complete.")

# Save all results to JSON
with open("arc_challenge_indic_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("Results saved to arc_challenge_indic_eval_results.json")
