import torch
from datasets import load_dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Any
import numpy as np
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

LANGUAGE_MAPPINGS = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
}

def format_crosssum_prompt(article_text: str, source_lang: str, target_lang: str) -> str:
    """Creates a simple zero-shot prompt for cross-lingual summarization."""
    source_lang_full = LANGUAGE_MAPPINGS.get(source_lang, source_lang)
    target_lang_full = LANGUAGE_MAPPINGS.get(target_lang, target_lang)
    # Truncate long articles to fit within context window
    truncated_article = article_text[:3000]
    return f"Summarize the following {source_lang_full} article in {target_lang_full}:\n\nArticle: {truncated_article}\n\nSummary in {target_lang_full}:"

def evaluate_cross_sum(
    pipe: Any, tokenizer: Any, model_name_for_logging: str,
    dataset_name: str,
    target_languages: List[str],
    dataset_split: str,
    max_new_tokens: int,
    num_samples_per_lang: int,
    **kwargs
) -> Dict[str, float]:
    
    logger.info(f"Starting CROSS SUM for {model_name_for_logging}")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    all_rouge1_scores = []

    for lang_code in target_languages:
        try:
            # The data files are nested under a field and named by language
            dataset = load_dataset(dataset_name, field="examples", trust_remote_code=True)
            
            # Filter for the specific target language
            lang_specific_data = [item for item in dataset[dataset_split] if item.get('lang') == lang_code]
            
            if not lang_specific_data:
                logger.warning(f"No data found for language {lang_code} in split {dataset_split}")
                continue

            num_to_eval = min(num_samples_per_lang, len(lang_specific_data))
            subset = lang_specific_data[:num_to_eval]
            logger.info(f"Evaluating {lang_code.upper()} on {len(subset)} samples.")
            
            predictions, references = [], []
            for item in tqdm(subset, desc=f"Eval {lang_code.upper()}"):
                source_text = item.get("text")
                reference_summary = item.get("summary")
                if not source_text or not reference_summary:
                    continue
                
                prompt = format_crosssum_prompt(source_text, "en", lang_code)
                with torch.no_grad():
                    result = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False, do_sample=False)
                prediction = result[0]['generated_text']

                predictions.append(prediction)
                references.append(reference_summary)

            if not predictions:
                continue

            # Calculate ROUGE scores for the language
            lang_rouge1 = []
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                lang_rouge1.append(scores['rouge1'].fmeasure)
            
            if lang_rouge1:
                all_rouge1_scores.extend(lang_rouge1)

        except Exception as e:
            logger.error(f"Error processing language {lang_code} for CROSS SUM: {e}")

    if not all_rouge1_scores:
        logger.error("No valid examples were processed for CROSS SUM.")
        return {"CROSS SUM": 0.0}

   
    overall_rouge1 = np.mean(all_rouge1_scores) * 100
    
    logger.info(f"Overall CROSS SUM ROUGE-1 Score: {overall_rouge1:.2f}")
    return {"CROSS SUM": overall_rouge1}