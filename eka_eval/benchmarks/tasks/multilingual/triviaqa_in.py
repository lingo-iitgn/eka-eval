# eka_eval/benchmarks/tasks/indic/triviaqa_indic_mcq.py

import torch
import re
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Any, Optional
import evaluate as hf_evaluate
import numpy as np
from datetime import datetime
import string
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data
except ImportError:
    # Fallback if eka_eval modules are not available
    def get_prompt_template(benchmark_name, template_name, specific_task_group):
        """Fallback prompt template function."""
        return {
            "template": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:",
            "language_specific_prompts": {
                "en": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:",
                "default": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"
            }
        }
    
    def format_prompt(template_dict, **kwargs):
        """Fallback format prompt function."""
        template = template_dict.get("template", "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:")
        return template.format(**kwargs)
    
    def format_few_shot_prompt(template_dict, examples, main_data):
        """Fallback few-shot prompt function."""
        return format_prompt(template_dict, **main_data)
    
    def get_prompt_data(benchmark_name, data_key, specific_task_group):
        """Fallback prompt data function."""
        return []

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "sarvamai/trivia-qa-indic-mcq"
DEFAULT_TARGET_LANGUAGES = ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation[:100]'
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_FEW_SHOT_COUNT = 3
PROMPT_FILE_BENCHMARK_KEY = "triviaqa_indic_mcq"
PROMPT_FILE_CATEGORY = "indic"

def remove_prefixes(aliases: List[str]) -> List[str]:
    """Remove any alias that has a strict prefix elsewhere in the list."""
    if not aliases:
        return []
    aliases_sorted = sorted(aliases)
    ret = [aliases_sorted[0]] if aliases_sorted else []
    for alias in aliases_sorted[1:]:
        if not alias.startswith(ret[-1]):
            ret.append(alias)
    return ret

def _get_language_mappings_from_config(prompt_template_dict: Dict) -> Dict[str, Dict[str, str]]:
    """Get language mappings from configuration."""
    return prompt_template_dict.get("language_mappings", {
        "hi": {"ए": "A", "अ": "A", "बी": "B", "ब": "B", "सी": "C", "स": "C", "डी": "D", "द": "D"},
        "bn": {"এ": "A", "ক": "A", "বি": "B", "খ": "B", "সি": "C", "গ": "C", "ডি": "D", "ঘ": "D"},
        "en": {}
    })

def _create_triviaqa_mcq_prompt(question: str, choices: List[str], language: str, 
                               prompt_template_dict: Dict, few_shot_examples: List[Dict] = None) -> str:
    """Create language-specific prompt for TriviaQA-Indic MCQ using configuration."""
    if not choices or len(choices) != 4:
        return None
    

    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

    lang_prompts = prompt_template_dict.get("language_specific_prompts", {})
    
    if few_shot_examples and len(few_shot_examples) > 0:
        if language in lang_prompts and isinstance(lang_prompts[language], dict):
            example_template = lang_prompts[language].get("few_shot_example_template", 
                                                        "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer: {answer_letter}")
            template_suffix = lang_prompts[language].get("template_suffix",
                                                       "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:")
            few_shot_separator = prompt_template_dict.get("few_shot_separator", "\n\n")
            
            few_shot_text = ""
            for example in few_shot_examples:
                few_shot_text += example_template.format(**example) + few_shot_separator
            
            full_prompt = few_shot_text + template_suffix.format(question=question, choices_str=choices_str)
            return full_prompt
        else:
            template = lang_prompts.get(language, lang_prompts.get("default", 
                                      "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"))
    else:

        template = lang_prompts.get(language, lang_prompts.get("default", 
                                  "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"))
    
    return template.format(question=question, choices_str=choices_str)

def _get_few_shot_examples_from_config(language: str, num_shots: int, prompt_template_dict: Dict) -> List[Dict]:
    """Get few-shot examples from configuration."""
    examples_data = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_triviaqa_indic_mcq",
        specific_task_group=PROMPT_FILE_CATEGORY
    )
    
    if examples_data and isinstance(examples_data, dict):
        lang_examples = examples_data.get(language, examples_data.get("en", []))
        return lang_examples[:num_shots] if lang_examples else []
    
    fallback_examples = {
        "hi": [
            {
                "question": "फ्रांस की राजधानी क्या है?",
                "choices_str": "A. लंदन\nB. पेरिस\nC. मैड्रिड\nD. रोम",
                "answer_letter": "B"
            }
        ],
        "en": [
            {
                "question": "What is the capital of France?",
                "choices_str": "A. London\nB. Paris\nC. Madrid\nD. Rome",
                "answer_letter": "B"
            }
        ]
    }
    
    lang_examples = fallback_examples.get(language, fallback_examples["en"])
    return lang_examples[:num_shots]

def _normalize_text_with_config(text: str, prompt_template_dict: Dict) -> str:
    """Normalize text using patterns from configuration."""
    if not text:
        return ""
    
    # Get cleanup patterns from config
    cleanup_patterns = prompt_template_dict.get("response_parsing_patterns", {}).get("cleanup_patterns", [
        "^(Answer|উত্তর|जवाब)\\s*:?\\s*",
        "^(The answer is|উত্তর হল|उत्तर है)\\s*",
        "\\s+",
        "[.!?]+$"
    ])
    
    # Apply cleanup patterns
    for pattern in cleanup_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    return text.strip()

def _parse_predicted_answer_with_config(generated_text: str, language: str, prompt_template_dict: Dict) -> Optional[str]:
    """Parse predicted answer using configuration patterns."""
    if not generated_text:
        return None
    
    # Normalize the text
    generated_text = _normalize_text_with_config(generated_text, prompt_template_dict)
    
    # Get first meaningful line
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    
    # Get parsing patterns from config
    parsing_config = prompt_template_dict.get("response_parsing_patterns", {})
    letter_patterns = parsing_config.get("letter_patterns", ["\\b([A-D])\\b", "([A-D])[.)\\s:,]"])
    word_patterns = parsing_config.get("word_patterns", {})
    
    # Pattern 1: Look for exact letter matches using config patterns
    for pattern in letter_patterns:
        match = re.search(pattern, first_line, re.IGNORECASE)
        if match:
            found_char = match.group(1).upper()
            if found_char in "ABCD":
                return found_char
            elif found_char.isdigit() and 1 <= int(found_char) <= 4:
                return chr(64 + int(found_char)) 
    
    # Pattern 2: Look for language-specific mappings
    lang_mappings = _get_language_mappings_from_config(prompt_template_dict)
    if language in lang_mappings:
        for native_char, english_char in lang_mappings[language].items():
            if native_char in first_line:
                return english_char
    
    # Pattern 3: Look for spelled out options using config
    if language in word_patterns:
        for word, letter in word_patterns[language].items():
            if word in first_line.lower():
                return letter
    
    # Pattern 4: Fallback - look for any A-D character
    fallback_match = re.search(r'([A-D])', first_line, re.IGNORECASE)
    if fallback_match:
        return fallback_match.group(1).upper()
    
    return None

def _letter_to_index(letter: str) -> int:
    """Convert letter to index."""
    if letter and 'A' <= letter.upper() <= 'D':
        return ord(letter.upper()) - ord('A')
    return -1

def save_detailed_triviaqa_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language_accuracies: Dict[str, float],
    overall_accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed TriviaQA-Indic results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"triviaqa_indic_mcq_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
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
        logger.info(f"Detailed TriviaQA-Indic MCQ results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def evaluate_triviaqa_indic_mcq(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = 4,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT,
    prompt_template_name_zeroshot: str = "triviaqa_indic_mcq_0shot",
    prompt_template_name_fewshot: str = "triviaqa_indic_mcq_3shot",
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting TriviaQA-Indic MCQ: {model_name_for_logging}")

    # Load prompt template based on few-shot count
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.warning(f"Prompt template '{current_prompt_template_name}' not found, using fallback")
        prompt_template_dict = {
            "template": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:",
            "language_specific_prompts": {
                "en": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:",
                "default": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"
            }
        }

    try:
        accuracy_metric = hf_evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load accuracy metric: {e}")
        return {"TriviaQA-Indic-MCQ": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    language_accuracies = {}
    all_individual_accuracies = []
    detailed_results = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        
        try:
            # Use HuggingFace datasets slicing syntax directly
            dataset = load_dataset(dataset_name, name=lang_code, split=dataset_split, trust_remote_code=True)
            lang_dataset = dataset

            if len(lang_dataset) == 0:
                language_accuracies[lang_code] = None
                continue
                
            logger.info(f"Processing {len(lang_dataset)} examples for {lang_code.upper()} (split: {dataset_split})")

            # Get few-shot examples for this language
            few_shot_examples = []
            if num_few_shot > 0:
                few_shot_examples = _get_few_shot_examples_from_config(lang_code, num_few_shot, prompt_template_dict)
                if not few_shot_examples:
                    logger.warning(f"No few-shot examples found for {lang_code}, falling back to zero-shot")
                    num_few_shot = 0

            predictions_indices = []
            reference_indices = []

            for example_idx, example in enumerate(tqdm(lang_dataset, desc=f"Eval {lang_code.upper()}")):
                question = example.get("question", "")
                choices = example.get("choices", [])
                answer_idx = example.get("answer", -1)

                if not question or not choices or len(choices) != 4 or answer_idx < 0 or answer_idx > 3:
                    predictions_indices.append(-1)
                    reference_indices.append(answer_idx if 0 <= answer_idx <= 3 else -1)
                    
                    if save_detailed:
                        detailed_results.append({
                            "language": lang_code,
                            "example_id": example_idx,
                            "question": question,
                            "choices": choices,
                            "correct_answer": answer_idx,
                            "correct_answer_letter": chr(65 + answer_idx) if 0 <= answer_idx <= 3 else None,
                            "predicted_answer": None,
                            "predicted_letter": None,
                            "is_correct": False,
                            "error": "Missing or invalid data",
                            "prompt": None,
                            "raw_response": None
                        })
                    continue

                prompt = _create_triviaqa_mcq_prompt(question, choices, lang_code, prompt_template_dict, few_shot_examples)
                if not prompt:
                    predictions_indices.append(-1)
                    reference_indices.append(answer_idx)
                    continue

                try:
                   
                    eval_config = prompt_template_dict.get("evaluation_config", {})
                    gen_config = {
                        "max_new_tokens": eval_config.get("max_new_tokens", max_new_tokens),
                        "do_sample": eval_config.get("do_sample", False),
                        "temperature": eval_config.get("temperature", 0.1),
                        "top_p": eval_config.get("top_p", 0.9),
                        "repetition_penalty": eval_config.get("repetition_penalty", 1.1),
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        "return_full_text": eval_config.get("return_full_text", False)
                    }
                    
                    with torch.no_grad():
                        outputs = pipe(prompt, **gen_config)
                    
                    generated_text = ""
                    if outputs and isinstance(outputs, list) and outputs[0]:
                        generated_text = outputs[0].get('generated_text', "").strip()

                except Exception as e:
                    logger.debug(f"Generation error for {lang_code} example {example_idx}: {e}")
                    generated_text = ""

                predicted_letter = _parse_predicted_answer_with_config(generated_text, lang_code, prompt_template_dict)
                pred_idx = _letter_to_index(predicted_letter)

                predictions_indices.append(pred_idx)
                reference_indices.append(answer_idx)

                # Save detailed results
                if save_detailed:
                    is_correct = pred_idx == answer_idx and pred_idx != -1
                    detailed_results.append({
                        "language": lang_code,
                        "example_id": example_idx,
                        "question": question,
                        "choices": choices,
                        "correct_answer": answer_idx,
                        "correct_answer_letter": chr(65 + answer_idx),
                        "predicted_answer": pred_idx,
                        "predicted_letter": predicted_letter,
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
        saved_path = save_detailed_triviaqa_results(
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


    final_scores = {"TriviaQA-IN": overall_average * 100} 

    for lang, acc in language_accuracies.items():
        final_scores[f"TriviaQA-IN_{lang}"] = (acc * 100) if acc is not None else 0.0

    logger.info(f"Overall TriviaQA-IN Average: {overall_average:.4f} ({overall_average*100:.2f}%)")
    return final_scores
