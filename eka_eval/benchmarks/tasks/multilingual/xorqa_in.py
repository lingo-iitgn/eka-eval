import torch
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import numpy as np
import re 
import json 
import os 
import string
import logging
from collections import Counter
from typing import Dict, List, Any

# All imports from your original script that are still needed
# Removed: accelerate, DDP, mp, traceback, etc.

logger = logging.getLogger(__name__)

# --- All Helper Functions from Your Original Script ---
# These are preserved with minor adaptations to remove 'accelerator'

LANGUAGE_FULL_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
}

def format_xorqa_prompt(question: str, lang_name: str) -> str:
    # Your original prompt formatting logic
    question_str = str(question) if question is not None else "[QUESTION MISSING]"
    if lang_name == "Hindi":
        prompt = f"""Question: फ्रांस की राजधानी क्या है?\nAnswer: पेरिस\n\nQuestion: {question_str}\nAnswer:"""
    else: # Fallback to English examples for other languages
        prompt = f"""Question: What is the capital of France?\nAnswer: Paris\n\nQuestion: {question_str}\nAnswer:"""
    return prompt

def load_xorqa_data_robust(dataset_name: str, indic_lang_code: str, split: str) -> HFDataset:
    # Your original robust data loading logic
    from datasets import Dataset
    data_file = f"xorqa_{indic_lang_code}_{split}.json"
    local_path = os.path.join("indic_data", "xorqa_in", data_file)

    if os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            # The data is nested under 'examples'
            examples = [item['examples'] for item in raw_data if 'examples' in item]
            return Dataset.from_list(examples)
        except Exception as e:
            logger.warning(f"Failed to load local file {local_path}: {e}")

    # Fallback to streaming from Hugging Face Hub if local file not found
    try:
        dataset = load_dataset(dataset_name, data_files={split: data_file}, split=split, trust_remote_code=True)
        examples = [item['examples'] for item in dataset if 'examples' in item]
        return Dataset.from_list(examples)
    except Exception as e:
        logger.error(f"All loading methods failed for {indic_lang_code}: {e}")
        return None

def clean_xorqa_output(text: str) -> str:
    # Your original cleaning logic
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[0]
    return ""

def batch_xorqa_generate(pipe: Any, prompts: List[str], max_new_tokens: int, batch_size: int) -> List[str]:
    # Your original batching logic, adapted for the framework's pipeline
    all_answers = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating answers"):
        batch_prompts = prompts[i:i + batch_size]
        try:
            with torch.no_grad():
                results = pipe(
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                    batch_size=len(batch_prompts)
                )
            batch_answers = [clean_xorqa_output(res[0]['generated_text']) for res in results]
            all_answers.extend(batch_answers)
        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            all_answers.extend(["[GENERATION ERROR]"] * len(batch_prompts))
    return all_answers

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths: return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)

def evaluate_language_xorqa(
    pipe: Any,
    indic_lang_code: str,
    dataset_name: str,
    dataset_split: str,
    num_samples_per_lang: int,
    max_new_tokens: int,
    batch_size: int
) -> List[float]:
    # Your original language evaluation logic, refactored
    lang_full_name = LANGUAGE_FULL_NAMES.get(indic_lang_code, indic_lang_code.upper())
    logger.info(f"\n--- Evaluating: {lang_full_name} XOR-QA ---")

    dataset = load_xorqa_data_robust(dataset_name, indic_lang_code, dataset_split)
    if dataset is None:
        return []

    num_to_eval = min(num_samples_per_lang, len(dataset))
    subset = dataset.select(range(num_to_eval))
    
    prompts, references = [], []
    for example in subset:
        question = example.get("question", "")
        answers = example.get("answers", [])
        if question and answers:
            prompts.append(format_xorqa_prompt(question, lang_full_name))
            references.append(answers)

    if not prompts:
        logger.warning(f"No valid prompts created for {indic_lang_code}.")
        return []

    predictions = batch_xorqa_generate(pipe, prompts, max_new_tokens, batch_size)
    
    f1_scores = [
        metric_max_over_ground_truths(f1_score, pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    
    logger.info(f"Average F1 for {indic_lang_code.upper()}: {np.mean(f1_scores) * 100:.2f}")
    return f1_scores

# --- The Main Wrapper Function for eka_eval ---

def evaluate_xorqa_in(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    dataset_name: str,
    target_languages: List[str],
    dataset_split: str,
    max_new_tokens: int,
    num_samples_per_lang: int,
    generation_batch_size: int,
    **kwargs
) -> Dict[str, float]:
    
    logger.info(f"Starting XorQA-IN for {model_name_for_logging}")
    all_f1_scores = []

    for lang_code in target_languages:
        lang_f1 = evaluate_language_xorqa(
            pipe=pipe,
            indic_lang_code=lang_code,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_samples_per_lang=num_samples_per_lang,
            max_new_tokens=max_new_tokens,
            batch_size=generation_batch_size
        )
        if lang_f1:
            all_f1_scores.extend(lang_f1)

    if not all_f1_scores:
        logger.error("No valid examples processed for XorQA-IN.")
        return {"XorQA-IN": 0.0}

    overall_avg_f1 = np.mean(all_f1_scores) * 100
    logger.info(f"Overall XorQA-IN Average F1 Score: {overall_avg_f1:.2f}%")
    return {"XorQA-IN": overall_avg_f1}