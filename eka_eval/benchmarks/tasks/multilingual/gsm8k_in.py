import torch
from datasets import load_dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Any, Optional
import re
import numpy as np

logger = logging.getLogger(__name__)

def _format_gsm8k_prompt(question: str) -> str:
    """Creates a simple zero-shot prompt for GSM-8K."""
    return f"Question: {question}\nAnswer:"

def _extract_answer(text: Optional[str]) -> Optional[str]:
    """Extracts the final numerical answer."""
    if text is None: return None
    # More robust regex to find the number after "####" or at the end
    match = re.search(r'####\s*([0-9\-\.,/]+)', text)
    if not match:
        # Fallback: find the last number in the string
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        if numbers:
            return numbers[-1].replace(',', '')
        return None
    return match.group(1).strip().replace(',', '')

def evaluate_gsm8k_in(
    pipe: Any, tokenizer: Any, model_name_for_logging: str,
    dataset_name: str,
    target_languages: List[str],
    dataset_split: str,
    max_new_tokens: int,
    **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting GSM-8K-IN for {model_name_for_logging}")

    all_correct = 0
    total_evaluated = 0

    for lang_code in target_languages:
        try:
            dataset = load_dataset(dataset_name, name=lang_code, split=dataset_split, trust_remote_code=True)
            logger.info(f"Evaluating {lang_code.upper()} on {len(dataset)} samples.")
            
            lang_correct = 0
            for item in tqdm(dataset, desc=f"Eval {lang_code.upper()}"):
                question = item.get("question")
                true_answer = str(item.get("answer"))
                if not question or not true_answer:
                    continue

                prompt = _format_gsm8k_prompt(question)
                with torch.no_grad():
                    result = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False, do_sample=False)
                prediction_text = result[0]['generated_text']

                predicted_answer = _extract_answer(prediction_text)
                true_final_answer = _extract_answer(true_answer)
                
                if predicted_answer is not None and true_final_answer is not None and predicted_answer == true_final_answer:
                    lang_correct += 1
            
            all_correct += lang_correct
            total_evaluated += len(dataset)

        except Exception as e:
            logger.error(f"Error processing language {lang_code} for GSM-8K-IN: {e}")

    if total_evaluated == 0:
        logger.error("No valid examples were processed for GSM-8K-IN.")
        return {"GSM-8K-IN": 0.0}

    overall_accuracy = (all_correct / total_evaluated) * 100
    
    logger.info(f"Overall GSM-8K-IN Accuracy: {overall_accuracy:.2f}% ({all_correct}/{total_evaluated})")
    return {"GSM-8K-IN": overall_accuracy}

####

##[Worker 0 (GPU 0)] Error processing language as for GSM-8K-IN: BuilderConfig 'as' not found. Available: ['bn', 'bn_roman', 'en', 'gu', 'gu_roman', 'hi', 'hi_roman', 'kn', 'kn_roman', 'ml', 'ml_roman', 'mr', 'mr_roman', 'or', 'or_roman', 'pa', 'pa_roman', 'ta', 'ta_roman', 'te', 'te_roman'###