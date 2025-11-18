import torch
import re
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Any, Optional
import evaluate as hf_evaluate
import numpy as np
from datetime import datetime

from eka_eval.utils.prompt_utils import get_prompt_template

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "sarvamai/arc-challenge-indic"
DEFAULT_TARGET_LANGUAGES = ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation'
DEFAULT_MAX_NEW_TOKENS = 5
PROMPT_FILE_BENCHMARK_KEY = "arc_c_in"
PROMPT_FILE_CATEGORY = "indic"
CHECKPOINT_DIR = "checkpoints/arc_c_in_checkpoints"

def _get_language_mappings() -> Dict[str, Dict[str, str]]:
    """Get comprehensive language-specific letter mappings."""
    return {
        "hi": {"ए": "A", "अ": "A", "बी": "B", "ब": "B", "सी": "C", "स": "C", "डी": "D", "द": "D"},
        "bn": {"এ": "A", "বি": "B", "সি": "C", "ডি": "D"},
        "gu": {"એ": "A", "બી": "B", "સી": "C", "ડી": "D"},
        "kn": {"ಎ": "A", "ಬಿ": "B", "ಸಿ": "C", "ಡಿ": "D"},
        "ml": {"എ": "A", "ബി": "B", "സി": "C", "ഡി": "D"},
        "mr": {"ए": "A", "बी": "B", "सी": "C", "डी": "D"},
        "or": {"ଏ": "A", "ବି": "B", "ସି": "C", "ଡି": "D"},
        "pa": {"ਏ": "A", "ਬੀ": "B", "ਸੀ": "C", "ਡੀ": "D"},
        "ta": {"ஏ": "A", "பி": "B", "சி": "C", "டி": "D"},
        "te": {"ఏ": "A", "బి": "B", "సి": "C", "డి": "D"},
        "en": {}  # English doesn't need mapping
    }

def _create_arc_c_in_prompt(question: str, choices_dict: Dict, language: str, prompt_template_dict: Dict) -> Optional[str]:
    """Create optimized language-specific prompt for ARC-Challenge-Indic."""
    if not isinstance(choices_dict, dict) or 'label' not in choices_dict or 'text' not in choices_dict:
        return None
    
    labels = choices_dict['label']
    texts = choices_dict['text']
    
    if not (isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts)):
        return None
    
    choices_str = "\n".join([f"{label.upper()}. {text}" for label, text in zip(labels, texts)])
    
    # Get language-specific template
    lang_prompts = prompt_template_dict.get("language_specific_prompts", {})
    if language in lang_prompts:
        template = lang_prompts[language]
    else:
        template = lang_prompts.get("default", prompt_template_dict.get("template", "Question: {question}\n\nOptions:\n{choices_str}\n\nAnswer:"))
    
    return template.format(question=question, choices_str=choices_str)

def _parse_predicted_answer(generated_text: str, language: str, all_mappings: Dict[str, Dict[str, str]]) -> Optional[str]:
    """Enhanced answer parsing with better accuracy."""
    if not generated_text:
        return None
    
    # Clean the text
    text = generated_text.strip()
    
    # Pattern 1: Exact letter match at start (A, B, C, D)
    match = re.match(r'^([A-D])[.\s)]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: Letter with punctuation (A., B), C:, D.)
    match = re.search(r'\b([A-D])[.)\s:,]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Native script letters (language-specific)
    lang_mapping = all_mappings.get(language, {})
    for native_char, english_char in lang_mapping.items():
        if text.startswith(native_char):
            return english_char
        if native_char in text[:10]:  # Check first 10 chars
            return english_char
    
    # Pattern 4: Word-based answers (option one, first, etc.)
    option_words = {
        'hi': {'पहला': 'A', 'पहली': 'A', 'दूसरा': 'B', 'दूसरी': 'B', 'तीसरा': 'C', 'तीसरी': 'C', 'चौथा': 'D', 'चौथी': 'D'},
        'en': {'first': 'A', 'second': 'B', 'third': 'C', 'fourth': 'D', 'option a': 'A', 'option b': 'B', 'option c': 'C', 'option d': 'D'},
        'bn': {'প্রথম': 'A', 'দ্বিতীয়': 'B', 'তৃতীয়': 'C', 'চতুর্থ': 'D'}
    }
    
    text_lower = text.lower()
    if language in option_words:
        for word, letter in option_words[language].items():
            if word in text_lower:
                return letter
    
    # Pattern 5: Any letter A-D anywhere in first 20 chars
    match = re.search(r'([A-D])', text[:20], re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None

def save_checkpoint(
    checkpoint_data: Dict,
    model_name: str,
    lang_code: str,
    process_id: int = 0
) -> str:
    """Save checkpoint for resuming evaluation."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    filename = f"arc_c_in_{model_clean}_{lang_code}_p{process_id}.json"
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        return filepath
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return ""

def load_checkpoint(
    model_name: str,
    lang_code: str,
    process_id: int = 0
) -> Optional[Dict]:
    """Load checkpoint if exists."""
    model_clean = model_name.replace("/", "_").replace(":", "_")
    filename = f"arc_c_in_{model_clean}_{lang_code}_p{process_id}.json"
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {filepath}: {e}")
    return None

def save_detailed_arc_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language_accuracies: Dict[str, float],
    overall_accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed ARC-Challenge-Indic results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arc_c_in_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_questions": len(results_data),
        "correct_answers": sum(1 for r in results_data if r["is_correct"]),
        "overall_accuracy": overall_accuracy,
        "language_accuracies": language_accuracies,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed ARC-Challenge-Indic results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def evaluate_arc_c_in(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = 1,
    prompt_template_name_zeroshot: str = "arc_c_in_0shot",
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    use_checkpoints: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting ARC-Challenge-Indic: {model_name_for_logging}")
    logger.info(f"Using optimized generation-based evaluation")

    # Load prompt template
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=prompt_template_name_zeroshot,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template not found")
        return {"ARC-Challenge-Indic": 0.0, "error_message": "PromptTemplateNotFound"}

    all_mappings = _get_language_mappings()

    try:
        accuracy_metric = hf_evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load accuracy metric: {e}")
        return {"ARC-Challenge-Indic": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    language_accuracies = {}
    all_individual_accuracies = []
    detailed_results = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        
        # Try to load checkpoint
        checkpoint = None
        if use_checkpoints:
            checkpoint = load_checkpoint(model_name_for_logging, lang_code, process_id)
            if checkpoint:
                logger.info(f"Loaded checkpoint for {lang_code}, resuming from example {checkpoint['last_processed_idx'] + 1}")
        
        try:
            dataset = load_dataset(dataset_name, name=lang_code, split=dataset_split, trust_remote_code=True)
            if isinstance(dataset, dict) and 'validation' in dataset:
                lang_dataset = dataset['validation']
            elif hasattr(dataset, 'select'):
                lang_dataset = dataset
            else:
                logger.warning(f"Could not load dataset for {lang_code}")
                language_accuracies[lang_code] = None
                continue

            if len(lang_dataset) == 0:
                language_accuracies[lang_code] = None
                continue

            # Resume from checkpoint if available
            start_idx = 0
            predictions = []
            references = []
            if checkpoint:
                start_idx = checkpoint['last_processed_idx'] + 1
                predictions = checkpoint.get('predictions', [])
                references = checkpoint.get('references', [])
                detailed_results.extend(checkpoint.get('detailed_results', []))

            for example_idx in tqdm(range(start_idx, len(lang_dataset)), 
                                   desc=f"Eval {lang_code.upper()}", 
                                   initial=start_idx, 
                                   total=len(lang_dataset)):
                example = lang_dataset[example_idx]
                question = example.get("question", "")
                choices_dict = example.get("choices", {})
                answer_key = example.get("answerKey", "").upper()

                if not question or not choices_dict or answer_key not in "ABCD":
                    if save_detailed:
                        detailed_results.append({
                            "language": lang_code,
                            "example_id": example_idx,
                            "question": question,
                            "choices": choices_dict,
                            "correct_answer": answer_key,
                            "predicted_answer": None,
                            "is_correct": False,
                            "error": "Missing data"
                        })
                    continue

                prompt = _create_arc_c_in_prompt(question, choices_dict, lang_code, prompt_template_dict)
                if not prompt:
                    if save_detailed:
                        detailed_results.append({
                            "language": lang_code,
                            "example_id": example_idx,
                            "question": question,
                            "choices": choices_dict,
                            "correct_answer": answer_key,
                            "predicted_answer": None,
                            "is_correct": False,
                            "error": "Prompt creation failed"
                        })
                    continue

                try:
                    gen_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        "return_full_text": False
                    }
                    
                    with torch.no_grad():
                        outputs = pipe(prompt, **gen_config)
                    
                    generated_text = ""
                    if outputs and isinstance(outputs, list) and outputs[0]:
                        generated_text = outputs[0].get('generated_text', "").strip()

                except Exception as e:
                    logger.debug(f"Generation error for {lang_code} example {example_idx}: {e}")
                    generated_text = ""

                predicted_letter = _parse_predicted_answer(generated_text, lang_code, all_mappings)
                
                # Convert to indices for metric computation
                pred_idx = ord(predicted_letter) - ord('A') if predicted_letter and predicted_letter in "ABCD" else -1
                ref_idx = ord(answer_key) - ord('A')

                predictions.append(pred_idx)
                references.append(ref_idx)

                if save_detailed:
                    is_correct = pred_idx == ref_idx and pred_idx != -1
                    detailed_results.append({
                        "language": lang_code,
                        "example_id": example_idx,
                        "question": question,
                        "choices": choices_dict,
                        "correct_answer": answer_key,
                        "predicted_answer": predicted_letter,
                        "is_correct": is_correct,
                        "error": None,
                        "prompt": prompt,
                        "raw_response": generated_text
                    })

                # Save checkpoint every 50 examples
                if use_checkpoints and (example_idx + 1) % 50 == 0:
                    checkpoint_data = {
                        "last_processed_idx": example_idx,
                        "predictions": predictions,
                        "references": references,
                        "detailed_results": [r for r in detailed_results if r['language'] == lang_code]
                    }
                    save_checkpoint(checkpoint_data, model_name_for_logging, lang_code, process_id)

            # Compute accuracy
            valid_pairs = [(p, r) for p, r in zip(predictions, references) if r != -1]
            if valid_pairs:
                valid_predictions = [p for p, r in valid_pairs]
                valid_references = [r for p, r in valid_pairs]
                
                acc_results = accuracy_metric.compute(predictions=valid_predictions, references=valid_references)
                lang_accuracy = acc_results['accuracy']
                language_accuracies[lang_code] = lang_accuracy
                all_individual_accuracies.append(lang_accuracy)
                logger.info(f"Accuracy for {lang_code.upper()}: {lang_accuracy:.4f} ({sum(1 for p, r in valid_pairs if p == r)}/{len(valid_pairs)})")
            else:
                language_accuracies[lang_code] = 0.0
                all_individual_accuracies.append(0.0)

            # Clean up checkpoint after completion
            if use_checkpoints:
                model_clean = model_name_for_logging.replace("/", "_").replace(":", "_")
                checkpoint_file = os.path.join(CHECKPOINT_DIR, f"arc_c_in_{model_clean}_{lang_code}_p{process_id}.json")
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            language_accuracies[lang_code] = None

    overall_average = np.mean(all_individual_accuracies) if all_individual_accuracies else 0.0

    if save_detailed and detailed_results:
        saved_path = save_detailed_arc_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            language_accuracies,
            overall_average,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed results with {len(detailed_results)} examples saved to: {saved_path}")

    final_scores = {"ARC-Challenge-Indic": overall_average * 100}
    for lang, acc in language_accuracies.items():
        final_scores[f"ARC-Challenge-Indic_{lang}"] = (acc * 100) if acc is not None else 0.0

    logger.info(f"Overall ARC-Challenge-Indic Average: {overall_average:.4f} ({overall_average*100:.2f}%)")
    return final_scores