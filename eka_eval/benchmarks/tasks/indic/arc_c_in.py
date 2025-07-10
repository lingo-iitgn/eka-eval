# eka_eval/benchmarks/tasks/indic/arc_c_in.py
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

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "sarvamai/arc-challenge-indic"
DEFAULT_TARGET_LANGUAGES = ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation[:50]'
DEFAULT_MAX_NEW_TOKENS = 10
PROMPT_FILE_BENCHMARK_KEY = "arc_c_in"
PROMPT_FILE_CATEGORY = "indic"

def _get_hindi_to_english_mapping(prompt_template_dict: Dict) -> Dict[str, str]:
    """Get Hindi to English letter mapping from config."""
    return prompt_template_dict.get("hindi_to_english_mapping", {
        "ए": "A", "अ": "A", "बी": "B", "ब": "B", 
        "सी": "C", "स": "C", "डी": "D", "द": "D"
    })

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
        "te": {"ఏ": "A", "బి": "B", "సి": "C", "డి": "D"}
    }

def _create_arc_c_in_prompt(question: str, choices_dict: Dict, language: str, prompt_template_dict: Dict) -> Optional[str]:
    """Create language-specific prompt for ARC-Challenge-Indic."""
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
        template = lang_prompts.get("default", prompt_template_dict.get("template", "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:"))
    
    return template.format(question=question, choices_str=choices_str)

def _normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common prefixes and suffixes
    text = re.sub(r'^(Answer|उत्तर|जवाब|সমাধান|ответ)\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(The answer is|উত্তর হল|उत्तर है)\s*', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def _parse_predicted_answer(generated_text: str, language: str, all_mappings: Dict[str, Dict[str, str]]) -> Optional[str]:
    """Parse predicted answer from generated text with improved accuracy."""
    if not generated_text:
        return None
    
    # Normalize the text
    generated_text = _normalize_text(generated_text)
    
    # Get first meaningful line
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    
    # Get language-specific mappings
    lang_mapping = all_mappings.get(language, {})
    hindi_mapping = all_mappings.get("hi", {})  # Fallback to Hindi mapping
    
    # Pattern 1: Look for exact letter matches (A, B, C, D)
    letter_match = re.search(r'\b([A-D])\b', first_line, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    
    # Pattern 2: Look for letter with punctuation (A., B), C:, etc.)
    punct_match = re.search(r'([A-D])[.)\s:,]', first_line, re.IGNORECASE)
    if punct_match:
        return punct_match.group(1).upper()
    
    # Pattern 3: Look for language-specific characters
    for native_char, english_char in lang_mapping.items():
        if native_char in first_line:
            return english_char
    
    # Pattern 4: Fallback to Hindi mapping
    for native_char, english_char in hindi_mapping.items():
        if native_char in first_line:
            return english_char
    
    # Pattern 5: Look for spelled out options
    option_words = {
        'hi': {'पहला': 'A', 'दूसरा': 'B', 'तीसरा': 'C', 'चौथा': 'D'},
        'en': {'first': 'A', 'second': 'B', 'third': 'C', 'fourth': 'D'},
        'bn': {'প্রথম': 'A', 'দ্বিতীয়': 'B', 'তৃতীয়': 'C', 'চতুর্থ': 'D'}
    }
    
    if language in option_words:
        for word, letter in option_words[language].items():
            if word in first_line.lower():
                return letter
    
    # Pattern 6: Look for any single character that could be an answer
    single_char_match = re.search(r'([A-D])', first_line, re.IGNORECASE)
    if single_char_match:
        return single_char_match.group(1).upper()
    
    return None

def _letter_to_index(letter: str) -> int:
    """Convert letter to index."""
    if letter and 'A' <= letter.upper() <= 'D':
        return ord(letter.upper()) - ord('A')
    return -1

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
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = 8,
    prompt_template_name_zeroshot: str = "arc_c_in_0shot",
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting ARC-Challenge-Indic: {model_name_for_logging}")

    # Load prompt template
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=prompt_template_name_zeroshot,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template not found")
        return {"ARC-Challenge-Indic": 0.0, "error_message": "PromptTemplateNotFound"}

    # Get comprehensive language mappings
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
        
        try:
            dataset = load_dataset(dataset_name, name=lang_code)
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

            predictions_indices = []
            reference_indices = []

            for example_idx, example in enumerate(tqdm(lang_dataset, desc=f"Eval {lang_code.upper()}")):
                question = example.get("question", "")
                choices_dict = example.get("choices", {})
                answer_key = example.get("answerKey", "").upper()

                if not question or not choices_dict or answer_key not in "ABCD":
                    predictions_indices.append(-1)
                    reference_indices.append(_letter_to_index(answer_key))
                    
                    if save_detailed:
                        detailed_results.append({
                            "language": lang_code,
                            "example_id": example_idx,
                            "question": question,
                            "choices": choices_dict,
                            "correct_answer": answer_key,
                            "predicted_answer": None,
                            "is_correct": False,
                            "error": "Missing data",
                            "prompt": None,
                            "raw_response": None
                        })
                    continue

                prompt = _create_arc_c_in_prompt(question, choices_dict, lang_code, prompt_template_dict)
                if not prompt:
                    predictions_indices.append(-1)
                    reference_indices.append(_letter_to_index(answer_key))
                    
                    if save_detailed:
                        detailed_results.append({
                            "language": lang_code,
                            "example_id": example_idx,
                            "question": question,
                            "choices": choices_dict,
                            "correct_answer": answer_key,
                            "predicted_answer": None,
                            "is_correct": False,
                            "error": "Prompt creation failed",
                            "prompt": None,
                            "raw_response": None
                        })
                    continue

                try:
                    gen_config = {
                        "max_new_tokens": max_new_tokens,
                        "num_beams": 1,
                        "do_sample": False,
                        "temperature": 0.0,
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
                pred_idx = _letter_to_index(predicted_letter)
                ref_idx = _letter_to_index(answer_key)

                predictions_indices.append(pred_idx)
                reference_indices.append(ref_idx)

                # Save detailed results
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

            # Calculate accuracy
            valid_pairs = [(p, r) for p, r in zip(predictions_indices, reference_indices) if r != -1]
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

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            language_accuracies[lang_code] = None

    # Calculate overall average
    overall_average = np.mean(all_individual_accuracies) if all_individual_accuracies else 0.0

    # Save detailed results
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

    # Prepare final results
    final_scores = {"ARC-Challenge-Indic": overall_average * 100}  # Convert to percentage
    for lang, acc in language_accuracies.items():
        final_scores[f"ARC-Challenge-Indic_{lang}"] = (acc * 100) if acc is not None else 0.0

    logger.info(f"Overall ARC-Challenge-Indic Average: {overall_average:.4f} ({overall_average*100:.2f}%)")
    return final_scores

# Test function for standalone execution
if __name__ == '__main__':
    import sys
    import os
    import argparse
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test ARC-Challenge-Indic")
    test_parser.add_argument("--model_name_test", type=str, default="sarvamai/sarvam-1")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:50]")
    test_parser.add_argument("--target_languages", nargs='+', default=["hi", "en","bn","gu","kn","ml","mr","or","pa","ta","te"])
    test_parser.add_argument("--save_detailed", action="store_true", help="Save detailed outputs to JSON file")
    
    arc_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="ARCIndicFileTest")
    logger.info(f"--- Standalone ARC-Challenge-Indic Test: {arc_args.model_name_test} ---")
    
    arc_pipe, _ = initialize_model_pipeline(arc_args.model_name_test, target_device_id=0)
    if arc_pipe:
        arc_eval_args = {
            "pipe": arc_pipe,
            "tokenizer": arc_pipe.tokenizer,
            "model_name_for_logging": arc_args.model_name_test,
            "device": arc_pipe.device,
            "dataset_split": arc_args.dataset_split_test,
            "target_languages": arc_args.target_languages,
            "save_detailed": arc_args.save_detailed,
            "process_id": 0
        }
        try:
            print(json.dumps(evaluate_arc_c_in(**arc_eval_args), indent=2))
        finally:
            cleanup_model_resources(arc_pipe, getattr(arc_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {arc_args.model_name_test} for ARC-Challenge-Indic test.")