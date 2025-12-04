import torch
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "ai4bharat/MILU"
DEFAULT_TARGET_LANGUAGES = [
    "Bengali", "English", "Gujarati", "Hindi", "Kannada", 
    "Malayalam", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"
]
DEFAULT_SPLIT = 'test'
CHECKPOINT_DIR = "checkpoints/milu_checkpoints"

def _compute_choice_loglikelihood_normalized(
    model, tokenizer, question: str, choice: str, device, ignore_first_space: bool = True
) -> float:
    """
    Compute normalized log-likelihood of choice given question.
    Matches indic-eval's LoglikelihoodAcc with length_normalization=True.
    """
    try:
        # Tokenize question and full text
        question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=True)
        full_text = question + " " + choice
        full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        
        question_len = question_tokens["input_ids"].shape[1]
        full_len = full_tokens["input_ids"].shape[1]
        choice_len = full_len - question_len
        
        # Move to device
        full_tokens = {k: v.to(device) for k, v in full_tokens.items()}
        
        with torch.no_grad():
            outputs = model(**full_tokens)
            logits = outputs.logits
            
            # Compute log-likelihood for choice tokens only
            log_likelihood = 0.0
            actual_choice_tokens = 0
            
            for i in range(question_len - 1, full_len - 1):
                token_logits = logits[0, i, :]
                token_id = full_tokens["input_ids"][0, i + 1]
                log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                
                # Skip first space token if requested (common in indic-eval)
                if ignore_first_space and i == question_len - 1:
                    token_text = tokenizer.decode([token_id])
                    if token_text.strip() == "":
                        continue
                
                log_likelihood += log_probs[token_id].item()
                actual_choice_tokens += 1
        
        # Length normalization (key for accuracy!)
        if actual_choice_tokens > 0:
            normalized_ll = log_likelihood / actual_choice_tokens
        else:
            normalized_ll = log_likelihood / max(1, choice_len)
        
        return normalized_ll
    
    except Exception as e:
        logger.debug(f"Error computing log-likelihood: {e}")
        return float('-inf')

def save_checkpoint(
    checkpoint_data: Dict,
    model_name: str,
    lang_code: str,
    process_id: int = 0
) -> str:
    """Save checkpoint for resuming evaluation."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    filename = f"milu_{model_clean}_{lang_code}_p{process_id}.json"
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
    filename = f"milu_{model_clean}_{lang_code}_p{process_id}.json"
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {filepath}: {e}")
    return None

def save_detailed_milu_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language_accuracies: Dict[str, float],
    overall_accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed MILU results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"milu_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
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
        logger.info(f"Detailed MILU results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def evaluate_milu_in(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    use_checkpoints: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate MILU using log-likelihood scoring (matching indic-eval methodology).
    """
    
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting MILU: {model_name_for_logging}")
    logger.info(f"Using log-likelihood scoring with length normalization")

    model = pipe.model
    language_accuracies = {}
    all_scores = []
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
            
            if not dataset or len(dataset) == 0:
                logger.warning(f"No data for {lang_code}")
                language_accuracies[lang_code] = None
                continue

            # Resume from checkpoint if available
            start_idx = 0
            correct = 0
            total = 0
            if checkpoint:
                start_idx = checkpoint['last_processed_idx'] + 1
                correct = checkpoint.get('correct', 0)
                total = checkpoint.get('total', 0)
                detailed_results.extend(checkpoint.get('detailed_results', []))

            for example_idx in tqdm(range(start_idx, len(dataset)), 
                                   desc=f"Eval {lang_code.upper()}",
                                   initial=start_idx,
                                   total=len(dataset)):
                example = dataset[example_idx]
                
                question = example.get("question", "")
                option1 = example.get("option1", "")
                option2 = example.get("option2", "")
                option3 = example.get("option3", "")
                option4 = example.get("option4", "")
                target_str = example.get("target", "")
                
                if not all([question, option1, option2, option3, option4, target_str]):
                    continue

                # Get target index (option1->0, option2->1, etc.)
                target_map = {"option1": 0, "option2": 1, "option3": 2, "option4": 3}
                target_idx = target_map.get(target_str, -1)
                
                if target_idx == -1:
                    continue

                # Compute log-likelihood for each choice
                choices = [option1, option2, option3, option4]
                log_likelihoods = []
                
                for choice in choices:
                    ll = _compute_choice_loglikelihood_normalized(
                        model, tokenizer, question, choice, device, ignore_first_space=True
                    )
                    log_likelihoods.append(ll)

                # Predict: highest log-likelihood
                pred_idx = int(np.argmax(log_likelihoods))
                
                is_correct = (pred_idx == target_idx)
                if is_correct:
                    correct += 1
                total += 1

                if save_detailed:
                    detailed_results.append({
                        "language": lang_code,
                        "example_id": example_idx,
                        "question": question,
                        "choices": choices,
                        "target": target_str,
                        "target_idx": target_idx,
                        "predicted_idx": pred_idx,
                        "is_correct": is_correct,
                        "log_likelihoods": log_likelihoods
                    })

                # Save checkpoint every 100 examples
                if use_checkpoints and (example_idx + 1) % 100 == 0:
                    checkpoint_data = {
                        "last_processed_idx": example_idx,
                        "correct": correct,
                        "total": total,
                        "detailed_results": [r for r in detailed_results if r['language'] == lang_code]
                    }
                    save_checkpoint(checkpoint_data, model_name_for_logging, lang_code, process_id)

            if total > 0:
                accuracy = correct / total
                language_accuracies[lang_code] = accuracy
                all_scores.append(accuracy)
                logger.info(f"Accuracy for {lang_code.upper()}: {accuracy:.4f} ({correct}/{total})")
            else:
                language_accuracies[lang_code] = 0.0
                all_scores.append(0.0)

            # Clean up checkpoint after completion
            if use_checkpoints:
                model_clean = model_name_for_logging.replace("/", "_").replace(":", "_")
                checkpoint_file = os.path.join(CHECKPOINT_DIR, f"milu_{model_clean}_{lang_code}_p{process_id}.json")
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            language_accuracies[lang_code] = None

    overall_avg = np.mean(all_scores) if all_scores else 0.0

    if save_detailed and detailed_results:
        saved_path = save_detailed_milu_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            language_accuracies,
            overall_avg,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed results with {len(detailed_results)} examples saved to: {saved_path}")

    final_scores = {"MILU": overall_avg * 100}
    for lang, score in language_accuracies.items():
        final_scores[f"MILU_{lang}"] = (score * 100) if score is not None else 0.0

    logger.info(f"Overall MILU Average: {overall_avg:.4f} ({overall_avg*100:.2f}%)")
    return final_scores