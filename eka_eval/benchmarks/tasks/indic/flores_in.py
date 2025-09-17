import torch
import re
import os
import json
import logging
from typing import Dict, List, Any, Tuple,Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from sacrebleu import corpus_chrf

logger = logging.getLogger(__name__)


@dataclass
class FloresConfig:
    """Internal config class to pass parameters between helpers."""
    translation_direction: str
    target_languages: List[str]
    num_samples_per_lang: int
    batch_size: int
    max_new_tokens: int
    use_few_shot: bool
    num_few_shot_examples: int

class FloresPromptManager:
    def __init__(self, config: FloresConfig):
        self.config = config
        self.language_mappings = {
            "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
            "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia", 
            "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
        }
        self.few_shot_examples = { # This could be loaded from a prompt JSON file
            "en_to_hi": [{"source_text": "Technology is advancing rapidly.", "target_text": "तकनीक तेज़ी से आगे बढ़ रही है।"}],
            "en_to_bn": [{"source_text": "Technology is advancing rapidly.", "target_text": "প্রযুক্তি দ্রুত এগিয়ে চলেছে।"}]
        }

    def create_translation_prompt(self, source_text: str, source_lang: str, target_lang: str) -> str:
        source_lang_full = self.language_mappings.get(source_lang, source_lang.capitalize())
        target_lang_full = self.language_mappings.get(target_lang, target_lang.capitalize())
        
        examples = []
        if self.config.use_few_shot:
            examples_key = f"{source_lang}_to_{target_lang}"
            examples = self.few_shot_examples.get(examples_key, [])[:self.config.num_few_shot_examples]

        prompt_parts = [f"English: {ex['source_text']}\n{self.language_mappings.get(target_lang)}: {ex['target_text']}" for ex in examples]
        prompt_parts.append(f"{source_lang_full}: {source_text}\n{target_lang_full}:")
        return "\n\n".join(prompt_parts)

class FloresDataManager:
    def __init__(self, dataset_name: str, config: FloresConfig):
        self.dataset_name = dataset_name
        self.config = config

    def load_language_data(self, indic_lang_code: str, split: str) -> Optional[HFDataset]:
        local_dir = "indic_data/flores_in"
        data_file = f"flores_en-{indic_lang_code}_{split}.json"
        local_path = os.path.join(local_dir, data_file)

        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                examples = [item['examples'] for item in raw_data if 'examples' in item]
                return HFDataset.from_list(examples)
            except Exception as e:
                logger.warning(f"Failed to load local file {local_path}: {e}")
        
        # Fallback to streaming from HF Hub
        try:
            dataset = load_dataset(self.dataset_name, data_files={split: data_file}, split=split, trust_remote_code=True)
            examples = [item['examples'] for item in dataset if 'examples' in item]
            return HFDataset.from_list(examples)
        except Exception as e:
            logger.error(f"All loading methods failed for {indic_lang_code}: {e}")
            return None

class FloresTranslationEngine:
    def __init__(self, pipe: Any, config: FloresConfig):
        self.pipe = pipe
        self.config = config
        self.prompt_manager = FloresPromptManager(config)

    def batch_translate(self, prompts: List[str]) -> List[str]:
        all_translations = []
        for i in tqdm(range(0, len(prompts), self.config.batch_size), desc="Translating"):
            batch_prompts = prompts[i:i + self.config.batch_size]
            try:
                with torch.no_grad():
                    results = self.pipe(
                        batch_prompts,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        return_full_text=False,
                        batch_size=len(batch_prompts)
                    )
                batch_translations = [res[0]['generated_text'].strip().split('\n')[0] for res in results]
                all_translations.extend(batch_translations)
            except Exception as e:
                logger.error(f"Batch translation error: {e}")
                all_translations.extend(["[TRANSLATION ERROR]"] * len(batch_prompts))
        return all_translations


def evaluate_flores_in(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    dataset_name: str,
    translation_direction: str,
    target_languages: List[str],
    num_samples_per_lang: int,
    batch_size: int,
    max_new_tokens: int,
    use_few_shot: bool,
    num_few_shot_examples: int,
    dataset_split: str,
    **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting Flores-IN Evaluation for {model_name_for_logging}")

    config = FloresConfig(
        translation_direction=translation_direction,
        target_languages=target_languages,
        num_samples_per_lang=num_samples_per_lang,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        use_few_shot=use_few_shot,
        num_few_shot_examples=num_few_shot_examples
    )

    data_manager = FloresDataManager(dataset_name, config)
    translation_engine = FloresTranslationEngine(pipe, config)
    
    all_chrf_scores = []
    source_lang = "en" # Based on direction 'enxx'

    for target_lang_code in config.target_languages:
        logger.info(f"--- Evaluating en -> {target_lang_code.upper()} ---")
        dataset = data_manager.load_language_data(target_lang_code, dataset_split)

        if not dataset or len(dataset) == 0:
            logger.warning(f"No data for en->{target_lang_code}, skipping.")
            continue

        subset = dataset.select(range(min(config.num_samples_per_lang, len(dataset))))
        
        prompts, references = [], []
        for item in subset:
            source_text, ref_translation = item.get("source"), item.get("target")
            if source_text and ref_translation:
                prompts.append(translation_engine.prompt_manager.create_translation_prompt(source_text, source_lang, target_lang_code))
                references.append(ref_translation)

        if not prompts: continue

        predictions = translation_engine.batch_translate(prompts)
        
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p != "[TRANSLATION ERROR]"]
        if not valid_pairs: continue

        valid_preds, valid_refs = zip(*valid_pairs)
        
        chrf_score = corpus_chrf(valid_preds, [valid_refs], word_order=2, beta=2).score
        all_chrf_scores.append(chrf_score)
        logger.info(f"chrF++ score for en->{target_lang_code.upper()}: {chrf_score:.2f}")

    if not all_chrf_scores:
        logger.error("No scores were calculated for Flores-IN.")
        return {"Flores-IN": 0.0}
    
    overall_avg_chrf = np.mean(all_chrf_scores)
    logger.info(f"Overall Flores-IN Average chrF++ Score: {overall_avg_chrf:.2f}")
    
    return {"Flores-IN": overall_avg_chrf}