# eka_eval/benchmarks/tasks/indic/milu_in.py
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

DEFAULT_DATASET_NAME = "ai4bharat/MILU"
DEFAULT_TARGET_LANGUAGES = [
    "Bengali", "English", "Gujarati", "Hindi", "Kannada", 
    "Malayalam", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"
]
DEFAULT_SPLIT = 'validation'
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_GENERATION_BATCH_SIZE = 8
DEFAULT_FEW_SHOT_COUNT = 0
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "milu_in_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT = "milu_in_5shot"
PROMPT_FILE_BENCHMARK_KEY = "milu_in"
PROMPT_FILE_CATEGORY = "indic"

def save_detailed_milu_in_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language: str,
    accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed MILU-Indic results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"milu_in_{model_clean}_{dataset_clean}_{language}_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "language": language,
        "total_questions": len(results_data),
        "correct_answers": sum(1 for r in results_data if r["is_correct"]),
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id,
        "generation_failures": sum(1 for r in results_data if not r.get("extraction_successful", True))
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed MILU-Indic results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed MILU-Indic results: {e}")
        return ""

def _get_milu_in_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    """Load few-shot examples from prompt configuration"""
    if num_few_shot <= 0:
        return []
    
    loaded_examples_list = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_milu_in",
        specific_task_group=prompt_file_category
    )
    
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        logger.info(f"Successfully loaded {len(loaded_examples_list)} few-shot examples from JSON for MILU-Indic.")
        return loaded_examples_list[:num_few_shot]
    
    logger.warning(f"Could not load default_few_shot_examples_milu_in from prompts/{prompt_file_category}/{PROMPT_FILE_BENCHMARK_KEY}.json")
    return []

def _get_hindi_to_english_mapping(prompt_template_dict: Dict) -> Dict[str, str]:
    """Get Hindi to English letter mapping from config."""
    return prompt_template_dict.get("hindi_to_english_mapping", {
        "ए": "A", "अ": "A", "बी": "B", "ब": "B", 
        "सी": "C", "स": "C", "डी": "D", "द": "D"
    })

def _get_option_mapping(prompt_template_dict: Dict) -> Dict[str, str]:
    """Get option to letter mapping from config."""
    return prompt_template_dict.get("option_mapping", {
        "option1": "A", "option2": "B", "option3": "C", "option4": "D"
    })

def _create_milu_in_prompt(question: str, option1: str, option2: str, option3: str, option4: str, language: str, prompt_template_dict: Dict, few_shot_examples: List[Dict] = None) -> str:
    """Create language-specific prompt for MILU-Indic using templates."""
    choices = [option1, option2, option3, option4]
    choices_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    main_q_data = {"question": question, "choices_str": choices_str}
    
    
    if few_shot_examples and len(few_shot_examples) > 0:
        lang_prompts = prompt_template_dict.get("language_specific_prompts", {})
        if language in lang_prompts and isinstance(lang_prompts[language], dict):
            lang_config = lang_prompts[language]
        else:
            lang_config = lang_prompts.get("default", {
                "few_shot_example_template": prompt_template_dict.get("few_shot_example_template", "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर: {answer_letter}"),
                "template_suffix": prompt_template_dict.get("template_suffix", "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:")
            })
        
        custom_template_dict = {
            "template_prefix": prompt_template_dict.get("template_prefix", ""),
            "few_shot_example_template": lang_config.get("few_shot_example_template"),
            "few_shot_separator": prompt_template_dict.get("few_shot_separator", "\n\n"),
            "template_suffix": lang_config.get("template_suffix")
        }
        
        return format_few_shot_prompt(custom_template_dict, few_shot_examples, main_q_data)
    else:
        lang_prompts = prompt_template_dict.get("language_specific_prompts", {})
        if language in lang_prompts:
            if isinstance(lang_prompts[language], str):
                template = lang_prompts[language]
            else:
                template = lang_prompts[language].get("template", prompt_template_dict.get("template", "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:"))
        else:
            template = lang_prompts.get("default", prompt_template_dict.get("template", "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:"))
        
        return template.format(**main_q_data)

def _parse_predicted_answer(generated_text: str, language: str, hindi_mapping: Dict[str, str]) -> Optional[str]:
    """Parse predicted answer from generated text with enhanced debugging."""
    if not generated_text:
        return None
    
    first_line = generated_text.split('\n')[0].strip()
    if not first_line:
        return None
    
    logger.debug(f"MILU-IN extraction input for {language}: '{first_line[:100]}'")
    
    # Try regex pattern
    regex_pattern = r"([A-Dएअबीबसीसडीद])(?:[.)\s]|$)"
    match = re.match(regex_pattern, first_line, re.IGNORECASE)
    
    if match:
        found_char = match.group(1).upper()
        if language.lower() == "hindi" and found_char in hindi_mapping:
            result = hindi_mapping[found_char]
            logger.debug(f"MILU-IN: Mapped Hindi '{found_char}' to '{result}'")
            return result
        elif found_char in "ABCD":
            logger.debug(f"MILU-IN: Found English letter '{found_char}'")
            return found_char
    
    fallback_match = re.search(r"\b([A-D])\b", first_line, re.IGNORECASE)
    if fallback_match:
        result = fallback_match.group(1).upper()
        logger.debug(f"MILU-IN: Fallback found '{result}'")
        return result
    
    logger.debug(f"MILU-IN: No pattern matched for: '{first_line[:50]}'")
    return None

def _target_to_index(target_str: str, option_mapping: Dict[str, str]) -> int:
    """Convert target string to index."""
    if not target_str:
        return -1
    
    target_lower = target_str.lower()
    
    # Direct option mapping
    if target_str in option_mapping:
        letter = option_mapping[target_str]
        return ord(letter) - ord('A')
    
    # Handle direct letters
    if target_str.upper() in "ABCD":
        return ord(target_str.upper()) - ord('A')
    
    # Handle option1-4 format
    if target_lower.startswith("option") and target_lower[-1].isdigit():
        option_num = int(target_lower[-1])
        if 1 <= option_num <= 4:
            return option_num - 1
    
    return -1

def evaluate_milu_in(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT,
    prompt_template_name_zeroshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT,
    prompt_template_name_fewshot: str = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting MILU-Indic ({num_few_shot}-shot): {model_name_for_logging}")
    logger.info(f"Target languages: {target_languages}")

    # Get prompt template
    current_prompt_template_name = prompt_template_name_fewshot if num_few_shot > 0 else prompt_template_name_zeroshot
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=current_prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{current_prompt_template_name}' not found")
        return {"MILU-Indic": 0.0, "error_message": f"PromptTemplate '{current_prompt_template_name}' NotFound"}

    # Load few-shot examples
    few_shot_examples_to_use = []
    if num_few_shot > 0:
        few_shot_examples_to_use = _get_milu_in_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        if not few_shot_examples_to_use:
            logger.warning("MILU-Indic: Failed to load few-shot examples from JSON, falling back to 0-shot.")
            num_few_shot = 0
            current_prompt_template_name = prompt_template_name_zeroshot
            prompt_template_dict = get_prompt_template(prompt_file_benchmark_key, current_prompt_template_name, prompt_file_category)
            if not prompt_template_dict:
                return {"MILU-Indic": 0.0, "error_message": "ZeroShotPromptTemplateNotFound"}

    hindi_mapping = _get_hindi_to_english_mapping(prompt_template_dict)
    option_mapping = _get_option_mapping(prompt_template_dict)

    try:
        accuracy_metric = hf_evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load accuracy metric: {e}")
        return {"MILU-Indic": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    language_accuracies = {}
    all_individual_accuracies = []

    for lang_config_name in target_languages:
        logger.info(f"--- Evaluating Language: {lang_config_name} ---")
        
        try:
            # Load dataset for specific language config
            dataset = load_dataset(dataset_name, name=lang_config_name, split=dataset_split)
            
            if len(dataset) == 0:
                logger.warning(f"No samples for {lang_config_name}")
                language_accuracies[lang_config_name] = None
                continue

            logger.info(f"Evaluating on {len(dataset)} samples for '{lang_config_name}'.")

            predictions_indices = []
            reference_indices = []
            detailed_results = []

            for example_idx, example in enumerate(tqdm(dataset, desc=f"Eval {lang_config_name}")):
                question = example.get("question", "")
                option1 = example.get("option1", "")
                option2 = example.get("option2", "")
                option3 = example.get("option3", "")
                option4 = example.get("option4", "")
                target_str = example.get("target", "")

                if not all([question, option1, option2, option3, option4, target_str]):
                    logger.warning(f"Skipping example in '{lang_config_name}' due to missing data.")
                    predictions_indices.append(-1)
                    reference_indices.append(_target_to_index(target_str, option_mapping))
                    if save_detailed:
                        detailed_results.append({
                            "question_id": f"item_{example_idx}",
                            "question": question,
                            "options": [option1, option2, option3, option4],
                            "target": target_str,
                            "predicted_answer": "SKIP",
                            "is_correct": False,
                            "extraction_successful": False,
                            "skip_reason": "Missing data"
                        })
                    continue

                prompt = _create_milu_in_prompt(question, option1, option2, option3, option4, lang_config_name, prompt_template_dict, few_shot_examples_to_use)

                try:
                    gen_config = {
                        "max_new_tokens": max_new_tokens,
                        "num_beams": 1,
                        "do_sample": False,
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
                    logger.debug(f"Generation error: {e}")
                    generated_text = ""

                predicted_letter = _parse_predicted_answer(generated_text, lang_config_name, hindi_mapping)
                
                pred_idx = _target_to_index(predicted_letter, {"A": "A", "B": "B", "C": "C", "D": "D"}) if predicted_letter else -1
                ref_idx = _target_to_index(target_str, option_mapping)
                
                extraction_successful = predicted_letter is not None
                if pred_idx == -1:
                    import random
                    pred_idx = random.choice([i for i in range(4) if i != ref_idx]) if ref_idx != -1 else 0
                
                is_correct = pred_idx == ref_idx
                predictions_indices.append(pred_idx)
                reference_indices.append(ref_idx)

                if save_detailed:
                    detailed_results.append({
                        "question_id": f"item_{example_idx}",
                        "question": question,
                        "options": [option1, option2, option3, option4],
                        "target": target_str,
                        "target_index": ref_idx,
                        "predicted_answer": predicted_letter or "UNKNOWN",
                        "predicted_index": pred_idx,
                        "is_correct": is_correct,
                        "prompt_used": prompt,
                        "generated_text": generated_text,
                        "extraction_successful": extraction_successful,
                        "language": lang_config_name
                    })

            # Calculate accuracy
            valid_pairs = [(p, r) for p, r in zip(predictions_indices, reference_indices) if r != -1]
            
            if valid_pairs:
                valid_predictions = [p for p, r in valid_pairs]
                valid_references = [r for p, r in valid_pairs]
                
                acc_results = accuracy_metric.compute(predictions=valid_predictions, references=valid_references)
                lang_accuracy = acc_results['accuracy']
                language_accuracies[lang_config_name] = lang_accuracy
                all_individual_accuracies.append(lang_accuracy)
                logger.info(f"Accuracy for {lang_config_name}: {lang_accuracy:.4f}")
                
                # Save detailed results for this language
                if save_detailed and detailed_results:
                    save_detailed_milu_in_results(
                        detailed_results,
                        model_name_for_logging,
                        dataset_name,
                        lang_config_name,
                        lang_accuracy,
                        results_dir,
                        process_id
                    )
            else:
                language_accuracies[lang_config_name] = 0.0
                all_individual_accuracies.append(0.0)

        except Exception as e:
            logger.error(f"Error processing language {lang_config_name}: {e}")
            language_accuracies[lang_config_name] = None

    overall_average = np.mean(all_individual_accuracies) if all_individual_accuracies else 0.0

    final_scores = {"MILU": overall_average}
    for lang, acc in language_accuracies.items():
        final_scores[f"MILU_{lang}"] = acc if acc is not None else 0.0

    logger.info(f"Overall MILU-Indic Average: {overall_average:.4f}")
    return final_scores

