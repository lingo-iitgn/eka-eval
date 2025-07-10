import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from .prompts import format_prompt_for_boolq
from .normalizer import normalize_answer_to_bool_int
import os
import sys

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "sarvamai/boolq-indic"
DEFAULT_TARGET_LANGUAGES = ["en", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation'
DEFAULT_MAX_NEW_TOKENS = 10


def evaluate_boolq_in(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = 1,
    do_sample: bool = False,
    process_id: int = 0,
    gpu_id: int = -1,
    num_gpus: int = 1,
    batch_size: int = 1,
    **kwargs
) -> Dict[str, float]:
    logger.info(f"Starting BoolQ-Indic evaluation for model: {model_name_for_logging} on dataset: {dataset_name} (Split: {dataset_split})")

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES
        logger.info(f"Using default target languages: {target_languages}")

    try:
        accuracy_metric = evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load 'accuracy' metric: {e}", exc_info=True)
        return {"BOOLQ-IN": 0.0, "error_message": "MetricLoadFailed"}

    language_accuracies: Dict[str, float | None] = {}
    all_individual_accuracies_list: List[float] = []

    try:
        full_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"Loaded dataset with {len(full_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return {"BOOLQ-IN": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}

    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        predictions_normalized: List[int] = []
        references_normalized: List[int] = []

        try:
            lang_specific_dataset = full_dataset.filter(
                lambda example: example.get('language') == lang_code,
                load_from_cache_file=True
            )

            if not lang_specific_dataset:
                logger.warning(f"No samples found for language '{lang_code}'. Skipping.")
                language_accuracies[lang_code] = None
                continue

            for example_idx, example in tqdm(enumerate(lang_specific_dataset), desc=f"Eval {lang_code.upper()}", total=len(lang_specific_dataset)):
                question = example.get("question", "")
                passage = example.get("passage", "")
                ground_truth_answer_str = example.get("answer", "")

                if not question or not passage:
                    predictions_normalized.append(-1)
                    references_normalized.append(normalize_answer_to_bool_int(ground_truth_answer_str))
                    continue

                prompt = format_prompt_for_boolq(passage, question)
                if "Error: Invalid input" in prompt:
                    predictions_normalized.append(-1)
                    references_normalized.append(normalize_answer_to_bool_int(ground_truth_answer_str))
                    continue

                try:
                    with torch.no_grad():
                        pipeline_output = pipe(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                            do_sample=do_sample,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            return_full_text=False
                        )
                    if pipeline_output and isinstance(pipeline_output, list):
                        generated_text = pipeline_output[0].get('generated_text', "").strip()
                    else:
                        generated_text = "[PIPELINE_OUTPUT_ERROR]"
                except Exception as e_gen:
                    logger.error(f"Error during generation: {e_gen}", exc_info=True)
                    generated_text = "[GENERATION_ERROR]"

                predicted_normalized_answer = normalize_answer_to_bool_int(generated_text)
                reference_normalized_answer = normalize_answer_to_bool_int(ground_truth_answer_str)

                predictions_normalized.append(predicted_normalized_answer)
                references_normalized.append(reference_normalized_answer)

            valid_pairs_for_metric = [
                (pred if pred != -1 else -99, ref)
                for pred, ref in zip(predictions_normalized, references_normalized)
                if ref != -1
            ]

            if valid_pairs_for_metric:
                preds, refs = zip(*valid_pairs_for_metric)
                lang_accuracy_value = accuracy_metric.compute(predictions=list(preds), references=list(refs))['accuracy']
            else:
                lang_accuracy_value = 0.0

            language_accuracies[lang_code] = lang_accuracy_value
            all_individual_accuracies_list.append(lang_accuracy_value)
            logger.info(f"  Accuracy for {lang_code.upper()}: {lang_accuracy_value:.4f}")

        except Exception as e_lang_processing:
            logger.error(f"Error processing language {lang_code}: {e_lang_processing}", exc_info=True)
            language_accuracies[lang_code] = None

    valid_accuracies = [acc for acc in all_individual_accuracies_list if acc is not None]
    overall_average_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0.0

    logger.info(f"Final BOOLQ-IN Score: {overall_average_accuracy:.4f}")
    final_scores: Dict[str, float] = {"BOOLQ-IN": float(overall_average_accuracy)}
    for lang, acc in language_accuracies.items():
        final_scores[f"BOOLQ-IN_{lang}"] = float(acc) if acc is not None else 0.0

    return final_scores

