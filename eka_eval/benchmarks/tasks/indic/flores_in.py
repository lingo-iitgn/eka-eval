import torch
import os
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sacrebleu import corpus_chrf

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_ENXX = "Cognitive-Lab/GoogleIndicGenBench_flores_enxx_in"
DEFAULT_DATASET_NAME_XXEN = "Cognitive-Lab/GoogleIndicGenBench_flores_xxen_in"
DEFAULT_TARGET_LANGUAGES = ["hi", "kn", "ta", "te", "gu", "mr", "ml"]
DEFAULT_SPLIT = "test"

def _compute_translation_loglikelihood(model, tokenizer, source: str, target: str, device) -> float:
    """Compute log-likelihood of target translation given source."""
    # Tokenize source and full text separately
    source_tokens = tokenizer(source, return_tensors="pt", add_special_keys=True)
    full_tokens = tokenizer(source + " " + target, return_tensors="pt", add_special_tokens=True)
    
    source_len = source_tokens["input_ids"].shape[1]
    full_len = full_tokens["input_ids"].shape[1]
    
    full_tokens = {k: v.to(device) for k, v in full_tokens.items()}
    
    with torch.no_grad():
        outputs = model(**full_tokens)
        logits = outputs.logits
        
        # Compute log-likelihood only for target tokens
        log_likelihood = 0.0
        for i in range(source_len - 1, full_len - 1):
            token_logits = logits[0, i, :]
            token_id = full_tokens["input_ids"][0, i + 1]
            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
            log_likelihood += log_probs[token_id].item()
    
    return log_likelihood

def evaluate_flores_enxx(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_ENXX,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:
    """Evaluate English to Indic translation using log-likelihood scoring."""
    
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting Flores-ENXX (en->xx): {model_name_for_logging}")
    logger.info(f"Using log-likelihood scoring for translation")

    model = pipe.model
    language_scores = {}
    all_scores = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating en -> {lang_code.upper()} ---")
        
        try:
            dataset = load_dataset(dataset_name, name=lang_code, split=dataset_split, trust_remote_code=True)
            
            if not dataset or len(dataset) == 0:
                logger.warning(f"No data for en->{lang_code}")
                language_scores[lang_code] = None
                continue

            correct = 0
            total = 0

            for example in tqdm(dataset, desc=f"Eval en->{lang_code.upper()}"):
                source_text = example.get("source", "")
                target_text = example.get("target", "")
                distractor_text = example.get("distractor", "")
                
                if not source_text or not target_text or not distractor_text:
                    continue

                try:
                    # Compute log-likelihood for correct translation
                    ll_correct = _compute_translation_loglikelihood(
                        model, tokenizer, source_text, target_text, device
                    )
                    
                    # Compute log-likelihood for distractor
                    ll_distractor = _compute_translation_loglikelihood(
                        model, tokenizer, source_text, distractor_text, device
                    )
                    
                    # Predict: choose the one with higher log-likelihood
                    if ll_correct > ll_distractor:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.debug(f"Error computing log-likelihood for {lang_code}: {e}")
                    continue

            if total > 0:
                accuracy = correct / total
                language_scores[lang_code] = accuracy
                all_scores.append(accuracy)
                logger.info(f"Accuracy for en->{lang_code.upper()}: {accuracy:.4f} ({correct}/{total})")
            else:
                language_scores[lang_code] = 0.0
                all_scores.append(0.0)

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            language_scores[lang_code] = None

    overall_avg = np.mean(all_scores) if all_scores else 0.0

    final_scores = {"Flores-ENXX": overall_avg * 100}
    for lang, score in language_scores.items():
        final_scores[f"Flores-ENXX_{lang}"] = (score * 100) if score is not None else 0.0

    logger.info(f"Overall Flores-ENXX Average: {overall_avg:.4f} ({overall_avg*100:.2f}%)")
    return final_scores

def evaluate_flores_xxen(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_XXEN,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:
    """Evaluate Indic to English translation using log-likelihood scoring."""
    
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting Flores-XXEN (xx->en): {model_name_for_logging}")
    logger.info(f"Using log-likelihood scoring for translation")

    model = pipe.model
    language_scores = {}
    all_scores = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating {lang_code.upper()} -> en ---")
        
        try:
            dataset = load_dataset(dataset_name, name=lang_code, split=dataset_split, trust_remote_code=True)
            
            if not dataset or len(dataset) == 0:
                logger.warning(f"No data for {lang_code}->en")
                language_scores[lang_code] = None
                continue

            correct = 0
            total = 0

            for example in tqdm(dataset, desc=f"Eval {lang_code.upper()}->en"):
                source_text = example.get("source", "")
                target_text = example.get("target", "")
                distractor_text = example.get("distractor", "")
                
                if not source_text or not target_text or not distractor_text:
                    continue

                try:
                    # Compute log-likelihood for correct translation
                    ll_correct = _compute_translation_loglikelihood(
                        model, tokenizer, source_text, target_text, device
                    )
                    
                    # Compute log-likelihood for distractor
                    ll_distractor = _compute_translation_loglikelihood(
                        model, tokenizer, source_text, distractor_text, device
                    )
                    
                    # Predict: choose the one with higher log-likelihood
                    if ll_correct > ll_distractor:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.debug(f"Error computing log-likelihood for {lang_code}: {e}")
                    continue

            if total > 0:
                accuracy = correct / total
                language_scores[lang_code] = accuracy
                all_scores.append(accuracy)
                logger.info(f"Accuracy for {lang_code.upper()}->en: {accuracy:.4f} ({correct}/{total})")
            else:
                language_scores[lang_code] = 0.0
                all_scores.append(0.0)

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            language_scores[lang_code] = None

    overall_avg = np.mean(all_scores) if all_scores else 0.0

    final_scores = {"Flores-XXEN": overall_avg * 100}
    for lang, score in language_scores.items():
        final_scores[f"Flores-XXEN_{lang}"] = (score * 100) if score is not None else 0.0

    logger.info(f"Overall Flores-XXEN Average: {overall_avg:.4f} ({overall_avg*100:.2f}%)")
    return final_scores