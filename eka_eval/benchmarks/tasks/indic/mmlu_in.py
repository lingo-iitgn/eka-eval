# eka_eval/benchmarks/tasks/indic/mmlu_in.py
import torch
import re
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Any, Optional, Union
import evaluate as hf_evaluate
import numpy as np
from datetime import datetime

# Add project root to path for imports
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
            "template": "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:",
            "language_specific_prompts": {
                "hi": "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:",
                "en": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:",
                "default": "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:"
            },
            "hindi_to_english_mapping": {
                "ए": "A", "अ": "A", "बी": "B", "ब": "B",
                "सी": "C", "स": "C", "डी": "D", "द": "D"
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

DEFAULT_DATASET_NAME = "sarvamai/mmlu-indic"
DEFAULT_TARGET_LANGUAGES = ["hi", "bn", "kn", "en", "gu", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation[:100]'
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_FEW_SHOT_COUNT = 5
PROMPT_FILE_BENCHMARK_KEY = "mmlu_in"
PROMPT_FILE_CATEGORY = "indic"

def _get_comprehensive_language_mappings() -> Dict[str, Dict[str, str]]:
    """Get comprehensive language-specific letter mappings."""
    return {
        "hi": {"ए": "A", "अ": "A", "बी": "B", "ब": "B", "सी": "C", "स": "C", "डी": "D", "द": "D"},
        "bn": {"এ": "A", "ক": "A", "বি": "B", "খ": "B", "সি": "C", "গ": "C", "ডি": "D", "ঘ": "D"},
        "gu": {"એ": "A", "અ": "A", "બી": "B", "બ": "B", "સી": "C", "સ": "C", "ડી": "D", "દ": "D"},
        "kn": {"ಎ": "A", "ಅ": "A", "ಬಿ": "B", "ಬ": "B", "ಸಿ": "C", "ಸ": "C", "ಡಿ": "D", "ದ": "D"},
        "ml": {"എ": "A", "അ": "A", "ബി": "B", "ബ": "B", "സി": "C", "സ": "C", "ഡി": "D", "ദ": "D"},
        "mr": {"ए": "A", "अ": "A", "बी": "B", "ब": "B", "सी": "C", "स": "C", "डी": "D", "द": "D"},
        "or": {"ଏ": "A", "ଅ": "A", "ବି": "B", "ବ": "B", "ସି": "C", "ସ": "C", "ଡି": "D", "ଦ": "D"},
        "pa": {"ਏ": "A", "ਅ": "A", "ਬੀ": "B", "ਬ": "B", "ਸੀ": "C", "ਸ": "C", "ਡੀ": "D", "ਦ": "D"},
        "ta": {"ஏ": "A", "அ": "A", "பி": "B", "ப": "B", "சி": "C", "ச": "C", "டி": "D", "த": "D"},
        "te": {"ఏ": "A", "అ": "A", "బి": "B", "బ": "B", "సి": "C", "స": "C", "డి": "D", "ద": "D"},
        "en": {}  # English doesn't need mapping
    }

def _create_mmlu_in_prompt(question: str, choices: List[str], language: str, 
                          prompt_template_dict: Dict, few_shot_examples: List[Dict] = None) -> str:
    """Create language-specific prompt for MMLU-Indic with improved formatting."""
    if not choices:
        return None
    
    # Format choices as A, B, C, D (handle variable number of choices)
    choices_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    
    # Get language-specific template
    lang_prompts = prompt_template_dict.get("language_specific_prompts", {})
    
    # Check if this is a few-shot template
    if few_shot_examples and len(few_shot_examples) > 0:
        if language in lang_prompts and isinstance(lang_prompts[language], dict):
            # Few-shot template format
            example_template = lang_prompts[language].get("few_shot_example_template", 
                                                        "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer: {answer_letter}")
            template_suffix = lang_prompts[language].get("template_suffix",
                                                       "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:")
            few_shot_separator = prompt_template_dict.get("few_shot_separator", "\n\n")
            
            # Build few-shot examples
            few_shot_text = ""
            for example in few_shot_examples:
                few_shot_text += example_template.format(**example) + few_shot_separator
            
            # Combine with main question
            full_prompt = few_shot_text + template_suffix.format(question=question, choices_str=choices_str)
            return full_prompt
        else:
            # Fallback to simple template
            template = lang_prompts.get(language, lang_prompts.get("default", 
                                      "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"))
    else:
        # Zero-shot template
        template = lang_prompts.get(language, lang_prompts.get("default", 
                                  "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:"))
    
    return template.format(question=question, choices_str=choices_str)

def _get_few_shot_examples_from_config(language: str, num_shots: int, prompt_template_dict: Dict) -> List[Dict]:
    """Get few-shot examples from configuration with language adaptation."""
    examples_data = get_prompt_data(
        benchmark_name=PROMPT_FILE_BENCHMARK_KEY,
        data_key="default_few_shot_examples_mmlu_in",
        specific_task_group=PROMPT_FILE_CATEGORY
    )
    
    if examples_data and isinstance(examples_data, list):
        return examples_data[:num_shots]
    
    # Fallback examples if config not available
    fallback_examples = [
        {
            "question": "भारत की राजधानी कौन सी है?" if language == "hi" else "What is the capital of India?",
            "choices_str": "A. मुंबई\nB. दिल्ली\nC. कोलकाता\nD. चेन्नई" if language == "hi" else "A. Mumbai\nB. Delhi\nC. Kolkata\nD. Chennai",
            "answer_letter": "B"
        },
        {
            "question": "गुरुत्वाकर्षण की खोज किसने की थी?" if language == "hi" else "Who discovered gravity?",
            "choices_str": "A. आइंस्टीन\nB. गैलीलियो\nC. न्यूटन\nD. केप्लर" if language == "hi" else "A. Einstein\nB. Galileo\nC. Newton\nD. Kepler",
            "answer_letter": "C"
        }
    ]
    
    return fallback_examples[:num_shots]

def _normalize_text_advanced(text: str) -> str:
    """Advanced text normalization for better parsing."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common prefixes in multiple languages
    prefixes = [
        r'^(Answer|উত্তর|जवाब|ಉತ್ತರ|જવાબ|ответ|উত্তর|ਜਵਾਬ|பதில்|సమాధానం|ഉത്തരം|उत्तर|ଉତ୍ତର)\s*:?\s*',
        r'^(The answer is|উত্তর হল|उत्तर है|ಉತ್ತರವು|જવાબ છે|উত্তর হচ্ছে|ਜਵਾਬ ਹੈ|பதில்|సమాధానం|ഉത്തരം|उत्तर आहे|ଉତ୍ତର ହେଉଛି)\s*',
        r'^(Option|विकल्प|বিকল্প|વિકલ્પ|ಆಯ್ಕೆ|ഓപ്ഷൻ|पर्याय|ବିକଳ୍ପ|ਵਿਕਲਪ|விருப்பம்|ఎంపిక)\s*',
        r'^(Choice|चुनाव|পছন্দ|પસંદગી|ಆಯ್ಕೆ|തിരഞ്ഞെടുക്കൽ|निवड|ପସନ୍ଦ|ਚੋਣ|தேர்வு|ఎంపిక)\s*'
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def _parse_predicted_answer_enhanced(generated_text: str, language: str, all_mappings: Dict[str, Dict[str, str]]) -> Union[str, None]:
    """Enhanced answer parsing with multiple strategies and language support."""
    if not generated_text:
        return None
    
    # Normalize the text
    generated_text = _normalize_text_advanced(generated_text)
    
    # Get first meaningful line
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    
    # Get language-specific mappings
    lang_mapping = all_mappings.get(language, {})
    
    # Strategy 1: Look for exact letter matches (A, B, C, D) - highest priority
    letter_match = re.search(r'\b([A-D])\b', first_line, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    
    # Strategy 2: Look for letter with punctuation (A., B), C:, etc.)
    punct_match = re.search(r'([A-D])[.)\s:,]', first_line, re.IGNORECASE)
    if punct_match:
        return punct_match.group(1).upper()
    
    # Strategy 3: Look for language-specific characters
    for native_char, english_char in lang_mapping.items():
        if native_char in first_line:
            return english_char
    
    # Strategy 4: Look for spelled out options
    option_words = {
        'hi': {'पहला': 'A', 'दूसरा': 'B', 'तीसरा': 'C', 'चौथा': 'D', 'प्रथम': 'A', 'द्वितीय': 'B', 'तृतीय': 'C', 'चतुर्थ': 'D'},
        'en': {'first': 'A', 'second': 'B', 'third': 'C', 'fourth': 'D', 'one': 'A', 'two': 'B', 'three': 'C', 'four': 'D'},
        'bn': {'প্রথম': 'A', 'দ্বিতীয়': 'B', 'তৃতীয়': 'C', 'চতুর্থ': 'D', 'এক': 'A', 'দুই': 'B', 'তিন': 'C', 'চার': 'D'}
    }
    
    if language in option_words:
        for word, letter in option_words[language].items():
            if word in first_line.lower():
                return letter
    
    # Strategy 5: Look for numbers and convert to letters
    number_match = re.search(r'\b([1-4])\b', first_line)
    if number_match:
        num = int(number_match.group(1))
        return chr(64 + num)  # 1->A, 2->B, 3->C, 4->D
    
    # Strategy 6: Look for any single character that could be an answer (fallback)
    single_char_match = re.search(r'([A-D])', first_line, re.IGNORECASE)
    if single_char_match:
        return single_char_match.group(1).upper()
    
    return None

def save_detailed_mmlu_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language_accuracies: Dict[str, float],
    overall_accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed MMLU-Indic results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mmlu_indic_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
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
        logger.info(f"Detailed MMLU-Indic results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def evaluate_mmlu_in(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = 4,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT,
    prompt_template_name_zeroshot: str = "mmlu_in_0shot",
    prompt_template_name_fewshot: str = "mmlu_in_5shot",
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting MMLU-Indic: {model_name_for_logging}")
    logger.info(f"Target languages: {target_languages}")

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
            "template": "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:",
            "language_specific_prompts": {
                "en": "Question: {question}\n\nChoices:\n{choices_str}\n\nAnswer:",
                "default": "प्रश्न: {question}\n\nविकल्प:\n{choices_str}\n\nउत्तर:"
            }
        }

    # Get comprehensive language mappings
    all_mappings = _get_comprehensive_language_mappings()

    try:
        accuracy_metric = hf_evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load accuracy metric: {e}")
        return {"MMLU-Indic": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    language_accuracies = {}
    all_individual_accuracies = []
    detailed_results = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        
        try:
            # Use HuggingFace datasets slicing syntax directly
            dataset = load_dataset(dataset_name, name=lang_code, split=dataset_split, trust_remote_code=True)
            
            if len(dataset) == 0:
                logger.warning(f"No samples for {lang_code}")
                language_accuracies[lang_code] = None
                continue

            logger.info(f"Processing {len(dataset)} examples for {lang_code.upper()} (split: {dataset_split})")

            # Get few-shot examples for this language
            few_shot_examples = []
            if num_few_shot > 0:
                few_shot_examples = _get_few_shot_examples_from_config(lang_code, num_few_shot, prompt_template_dict)
                if not few_shot_examples:
                    logger.warning(f"No few-shot examples found for {lang_code}, falling back to zero-shot")
                    num_few_shot = 0

            predictions_indices = []
            reference_indices = []

            for example_idx, example in enumerate(tqdm(dataset, desc=f"Eval {lang_code.upper()}")):
                question = example.get("question", "")
                choices = example.get("choices", [])
                correct_answer_index = example.get("answer", -1)

                if not all([question, isinstance(choices, list), len(choices) > 0, isinstance(correct_answer_index, int)]):
                    logger.warning(f"Skipping malformed example for '{lang_code}' at index {example_idx}")
                    predictions_indices.append(-1)
                    reference_indices.append(correct_answer_index if correct_answer_index >= 0 else -1)
                    
                    if save_detailed:
                        detailed_results.append({
                            "language": lang_code,
                            "example_id": example_idx,
                            "question": question,
                            "choices": choices,
                            "correct_answer": correct_answer_index,
                            "correct_answer_letter": chr(65 + correct_answer_index) if 0 <= correct_answer_index <= 3 else None,
                            "predicted_answer": None,
                            "predicted_letter": None,
                            "is_correct": False,
                            "error": "Malformed example",
                            "prompt": None,
                            "raw_response": None
                        })
                    continue

                prompt = _create_mmlu_in_prompt(question, choices, lang_code, prompt_template_dict, few_shot_examples)
                if not prompt:
                    predictions_indices.append(-1)
                    reference_indices.append(correct_answer_index)
                    continue

                try:
                    gen_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
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

                predicted_letter = _parse_predicted_answer_enhanced(generated_text, lang_code, all_mappings)
                
                predicted_answer_index = -1
                if predicted_letter and predicted_letter in "ABCD":
                    predicted_answer_index = ord(predicted_letter) - ord('A')
                
                predictions_indices.append(predicted_answer_index)
                reference_indices.append(correct_answer_index)

                # Save detailed results
                if save_detailed:
                    is_correct = predicted_answer_index == correct_answer_index and predicted_answer_index != -1
                    detailed_results.append({
                        "language": lang_code,
                        "example_id": example_idx,
                        "question": question,
                        "choices": choices,
                        "correct_answer": correct_answer_index,
                        "correct_answer_letter": chr(65 + correct_answer_index) if 0 <= correct_answer_index <= 3 else None,
                        "predicted_answer": predicted_answer_index,
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
        saved_path = save_detailed_mmlu_results(
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
    final_scores = {"MMLU-Indic": overall_average * 100}  # Convert to percentage
    for lang, acc in language_accuracies.items():
        final_scores[f"MMLU-Indic_{lang}"] = (acc * 100) if acc is not None else 0.0

    logger.info(f"Overall MMLU-Indic Average: {overall_average:.4f} ({overall_average*100:.2f}%)")
    return final_scores

# Test function for standalone execution
if __name__ == '__main__':
    import argparse
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s - %(message)s'
    )
    
    # Try to import eka_eval modules, use fallbacks if not available
    try:
        from eka_eval.utils.logging_setup import setup_logging
        from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
        setup_logging(level=logging.DEBUG, worker_id="MMLUIndicTest")
    except ImportError:
        logger.warning("eka_eval modules not available, using fallback implementations")
        
        def initialize_model_pipeline(model_name, target_device_id=0):
            """Fallback model initialization."""
            try:
                from transformers import pipeline
                pipe = pipeline("text-generation", model=model_name, device=target_device_id, trust_remote_code=True)
                return pipe, None
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                return None, None
        
        def cleanup_model_resources(pipe, model):
            """Fallback cleanup function."""
            if pipe:
                del pipe
            if model:
                del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    test_parser = argparse.ArgumentParser(description="Standalone Test MMLU-Indic")
    test_parser.add_argument("--model_name_test", type=str, default="sarvamai/sarvam-1")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:100]")
    test_parser.add_argument("--target_languages", nargs='+', default=["hi", "en", "bn","gu","kn","ml","mr","or","pa","ta","te"])
    test_parser.add_argument("--num_few_shot_test", type=int, default=5)
    test_parser.add_argument("--save_detailed", action="store_true", help="Save detailed outputs to JSON file")
    
    mmlu_args = test_parser.parse_args()
    logger.info(f"--- Standalone MMLU-Indic Test: {mmlu_args.model_name_test} ---")
    
    mmlu_pipe, _ = initialize_model_pipeline(mmlu_args.model_name_test, target_device_id=0)
    if mmlu_pipe:
        mmlu_eval_args = {
            "pipe": mmlu_pipe,
            "tokenizer": mmlu_pipe.tokenizer,
            "model_name_for_logging": mmlu_args.model_name_test,
            "device": mmlu_pipe.device,
            "dataset_split": mmlu_args.dataset_split_test,
            "target_languages": mmlu_args.target_languages,
            "num_few_shot": mmlu_args.num_few_shot_test,
            "save_detailed": mmlu_args.save_detailed,
            "process_id": 0
        }
        try:
            print(json.dumps(evaluate_mmlu_in(**mmlu_eval_args), indent=2))
        finally:
            cleanup_model_resources(mmlu_pipe, getattr(mmlu_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {mmlu_args.model_name_test} for MMLU-Indic test.")