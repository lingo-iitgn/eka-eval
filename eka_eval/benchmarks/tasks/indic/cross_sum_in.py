# crosssum_eval.py
import torch
import re
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import traceback
from accelerate import Accelerator, DistributedDataParallelKwargs

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "google/IndicGenBench_crosssum_in"
DEFAULT_TARGET_LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
DEFAULT_SPLIT = 'test[:100]'
DEFAULT_MAX_NEW_TOKENS = 256
PROMPT_FILE_BENCHMARK_KEY = "crosssum_in"
PROMPT_FILE_CATEGORY = "indic"

class CrossSumPromptManager:
    """Enhanced prompt management for cross-lingual summarization"""
    
    def __init__(self, prompt_templates: Dict):
        self.prompt_templates = prompt_templates
        self.language_mappings = {
            "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
            "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
            "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
        }
    
    def get_prompt_template(self, template_name: str) -> Dict:
        """Get prompt template by name"""
        return self.prompt_templates.get(template_name, {})
    
    def create_summarization_prompt(self, article_text: str, source_lang: str, target_lang: str, template_name: str = "crosssum_0shot") -> str:
        """Create language-specific summarization prompt"""
        template_config = self.get_prompt_template(template_name)
        
        if not template_config:
            # Fallback to basic template
            return f"Summarize the following {source_lang} article in {target_lang}:\n\nArticle: {article_text}\n\nSummary in {target_lang}:"
        
        # Get language-specific prompts
        lang_prompts = template_config.get("language_specific_prompts", {})
        lang_key = f"{source_lang}_to_{target_lang}"
        
        if lang_key in lang_prompts:
            prompt_template = lang_prompts[lang_key]
        else:
            prompt_template = template_config.get("template", lang_prompts.get("default", ""))
        
        # Use few-shot examples if available
        if template_config.get("use_few_shot", False):
            few_shot_examples = template_config.get("few_shot_examples", {}).get(lang_key, [])
            if few_shot_examples:
                examples_text = []
                for example in few_shot_examples:
                    example_text = f"Article: {example['source_text']}\nSummary: {example['target_text']}"
                    examples_text.append(example_text)
                
                examples_str = "\n\n".join(examples_text)
                prompt_template = f"{examples_str}\n\n{prompt_template}"
        
        # Format the prompt
        source_lang_full = self.language_mappings.get(source_lang, source_lang.capitalize())
        target_lang_full = self.language_mappings.get(target_lang, target_lang.capitalize())
        
        # Truncate very long articles
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."
        
        formatted_prompt = prompt_template.format(
            article_text=article_text,
            source_language=source_lang_full,
            target_language=target_lang_full
        )
        
        return formatted_prompt

def setup_accelerator():
    """Setup Accelerator for multi-GPU training"""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )
    return accelerator

def load_prompt_templates() -> Dict:
    """Load CrossSum prompt templates"""
    return {
        "crosssum_0shot": {
            "template": "Article: {article_text}\nSummary in {target_language}:",
            "description": "Zero-shot cross-lingual summarization prompt",
            "use_few_shot": False,
            "language_specific_prompts": {
                "en_to_hi": "Article: {article_text}\nSummary in Hindi:",
                "en_to_bn": "Article: {article_text}\nSummary in Bengali:",
                "en_to_gu": "Article: {article_text}\nSummary in Gujarati:",
                "en_to_ta": "Article: {article_text}\nSummary in Tamil:",
                "en_to_te": "Article: {article_text}\nSummary in Telugu:",
                "en_to_kn": "Article: {article_text}\nSummary in Kannada:",
                "en_to_ml": "Article: {article_text}\nSummary in Malayalam:",
                "en_to_mr": "Article: {article_text}\nSummary in Marathi:",
                "en_to_pa": "Article: {article_text}\nSummary in Punjabi:",
                "en_to_or": "Article: {article_text}\nSummary in Odia:",
                "en_to_as": "Article: {article_text}\nSummary in Assamese:",
                "en_to_ur": "Article: {article_text}\nSummary in Urdu:",
                "default": "Article: {article_text}\nSummary in {target_language}:"
            }
        },
        "crosssum_3shot": {
            "template": "{few_shot_examples}\n\nArticle: {article_text}\nSummary in {target_language}:",
            "description": "Few-shot cross-lingual summarization prompt",
            "use_few_shot": True,
            "language_specific_prompts": {
                "en_to_hi": "Article: {article_text}\nSummary in Hindi:",
                "en_to_bn": "Article: {article_text}\nSummary in Bengali:",
                "en_to_gu": "Article: {article_text}\nSummary in Gujarati:",
                "en_to_ta": "Article: {article_text}\nSummary in Tamil:",
                "en_to_te": "Article: {article_text}\nSummary in Telugu:",
                "en_to_kn": "Article: {article_text}\nSummary in Kannada:",
                "en_to_ml": "Article: {article_text}\nSummary in Malayalam:",
                "en_to_mr": "Article: {article_text}\nSummary in Marathi:",
                "en_to_pa": "Article: {article_text}\nSummary in Punjabi:",
                "en_to_or": "Article: {article_text}\nSummary in Odia:",
                "en_to_as": "Article: {article_text}\nSummary in Assamese:",
                "en_to_ur": "Article: {article_text}\nSummary in Urdu:",
                "default": "Article: {article_text}\nSummary in {target_language}:"
            },
            "few_shot_examples": {
                "en_to_hi": [
                    {
                        "source_text": "The Indian Space Research Organisation (ISRO) successfully launched its latest satellite mission today. The rocket carried multiple communication satellites into orbit, marking another milestone in India's space program.",
                        "target_text": "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने आज अपना नवीनतम उपग्रह मिशन सफलतापूर्वक लॉन्च किया। रॉकेट ने कई संचार उपग्रहों को कक्षा में पहुंचाया।"
                    },
                    {
                        "source_text": "Climate change continues to affect weather patterns globally. Scientists report increased frequency of extreme weather events.",
                        "target_text": "जलवायु परिवर्तन का वैश्विक मौसम पैटर्न पर प्रभाव जारी है। वैज्ञानिकों की रिपोर्ट के अनुसार चरम मौसमी घटनाओं की आवृत्ति बढ़ रही है।"
                    },
                    {
                        "source_text": "Technology companies are investing heavily in artificial intelligence research. New developments are expected to transform various industries.",
                        "target_text": "प्रौद्योगिकी कंपनियां कृत्रिम बुद्धिमत्ता अनुसंधान में भारी निवेश कर रही हैं। नए विकास से विभिन्न उद्योगों में बदलाव की उम्मीद है।"
                    }
                ],
                "en_to_bn": [
                    {
                        "source_text": "The Indian Space Research Organisation (ISRO) successfully launched its latest satellite mission today.",
                        "target_text": "ভারতীয় মহাকাশ গবেষণা সংস্থা (ইসরো) আজ তাদের সর্বশেষ স্যাটেলাইট মিশন সফলভাবে উৎক্ষেপণ করেছে।"
                    },
                    {
                        "source_text": "Technology companies are investing heavily in artificial intelligence research.",
                        "target_text": "প্রযুক্তি কোম্পানিগুলো কৃত্রিম বুদ্ধিমত্তা গবেষণায় ব্যাপক বিনিয়োগ করছে।"
                    },
                    {
                        "source_text": "Climate change affects global weather patterns significantly.",
                        "target_text": "জলবায়ু পরিবর্তন বিশ্বব্যাপী আবহাওয়ার ধরনকে উল্লেখযোগ্যভাবে প্রভাবিত করে।"
                    }
                ]
            }
        }
    }

def load_crosssum_data_robust(indic_lang_code: str, dataset_name: str, split: str) -> Any:
    """Robust data loading for CrossSum dataset"""
    cache_file = f"crosssum_{indic_lang_code}_test.json"
    
    # Try loading from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples from cached file")
            from datasets import Dataset
            return Dataset.from_list(examples)
        except Exception as e:
            logger.warning(f"Failed to load cached file {cache_file}: {e}")

    # Load the full dataset
    logger.info(f"Loading CrossSum dataset for {indic_lang_code}...")
    try:
        dataset = load_dataset(dataset_name, field="examples", trust_remote_code=True)
        logger.info(f"✅ Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        raise ValueError("Failed to load CrossSum dataset")
    
    # Extract relevant examples
    extracted_examples = []
    
    # Check all splits
    for split_name in ['test', 'validation', 'train']:
        if split_name not in dataset:
            continue
            
        logger.info(f"Searching in {split_name} split...")
        split_data = dataset[split_name]
        
        for item in split_data:
            if isinstance(item, dict):
                item_lang = item.get('lang', '')
                
                # For English->Indic: text is in English, summary is in target language
                if item_lang == indic_lang_code:
                    if item.get('text') and item.get('summary'):
                        extracted_examples.append(item)
            
            # Break early if we have enough examples
            if len(extracted_examples) >= 200:  # Get extra examples
                break
        
        if extracted_examples:
            logger.info(f"Found {len(extracted_examples)} examples in {split_name} split")
            break
    
    if not extracted_examples:
        raise ValueError(f"No examples found for {indic_lang_code}")
    
    # Save cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_examples, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved examples to {cache_file}")
    except Exception as e:
        logger.warning(f"Could not save cache file: {e}")
    
    from datasets import Dataset
    return Dataset.from_list(extracted_examples)

def batch_summarize(pipe: Any, prompts: List[str], max_new_tokens: int = 256, batch_size: int = 2) -> List[str]:
    """Perform batch summarization with optimized generation"""
    summaries = []
    
    with tqdm(total=len(prompts), desc="Summarizing") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_summaries = []
            
            for prompt in batch_prompts:
                try:
                    gen_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_beams": 1,
                        "repetition_penalty": 1.1,
                        "early_stopping": True,
                        "return_full_text": False
                    }
                    
                    with torch.no_grad():
                        outputs = pipe(prompt, **gen_config)
                    
                    summary = ""
                    if outputs and isinstance(outputs, list) and outputs[0]:
                        summary = outputs[0].get('generated_text', "").strip()
                    
                    summary = clean_summary_output(summary)
                    batch_summaries.append(summary)
                    
                except Exception as e:
                    logger.debug(f"Error in summarization: {e}")
                    batch_summaries.append("[SUMMARIZATION ERROR]")
            
            summaries.extend(batch_summaries)
            pbar.update(len(batch_prompts))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return summaries

def clean_summary_output(text: str) -> str:
    """Enhanced cleaning for summary output"""
    if not text:
        return text
    
    text = text.strip()
    
    # Take first meaningful paragraph
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        text = lines[0]
    
    # Remove common prefixes
    patterns_to_remove = [
        r'^(Summary in [A-Za-z]+|Summary|सारांश|সারাংশ|સારાંશ|ಸಾರಾಂಶ|സംഗ്രഹം|सारांश|ସାରାଂଶ|ਸਾਰ|सारांश|சுருக்கம்|సారాంశం|خلاصہ):\s*',
        r'^(Output|Result|Answer):\s*',
        r'^(Article|लेख|নিবন্ধ):\s*',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Stop at certain patterns
    stop_patterns = [
        r'\n\n',
        r'\nArticle:',
        r'\nSummary:',
        r'\n[A-Z][a-z]+:',
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    
    # Apply length constraints
    if len(text) > 500:
        text = text[:500].rsplit(' ', 1)[0]  # Cut at word boundary
    
    if len(text) < 10:
        return "[INVALID_SUMMARY]"
    
    return text.strip()

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for summarization evaluation"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip() or pred == "[SUMMARIZATION ERROR]" or pred == "[INVALID_SUMMARY]":
            continue
            
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    if not rouge1_scores:
        return {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
    
    return {
        "ROUGE-1": np.mean(rouge1_scores) * 100,
        "ROUGE-2": np.mean(rouge2_scores) * 100,
        "ROUGE-L": np.mean(rougeL_scores) * 100,
    }

def save_detailed_crosssum_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language_scores: Dict[str, Dict],
    overall_scores: Dict[str, float],
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed CrossSum results to JSON file"""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crosssum_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_examples": len(results_data),
        "valid_examples": sum(1 for r in results_data if r.get("is_valid", False)),
        "overall_scores": overall_scores,
        "language_scores": language_scores,
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
        logger.info(f"Detailed CrossSum results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def evaluate_crosssum_in(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = 2,
    prompt_template_name: str = "crosssum_0shot",
    results_dir: str = "results_output",
    save_detailed: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:
    
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting CrossSum Evaluation: {model_name_for_logging}")
    
    # Load prompt templates
    prompt_templates = load_prompt_templates()
    prompt_manager = CrossSumPromptManager(prompt_templates)
    
    if prompt_template_name not in prompt_templates:
        logger.error(f"Prompt template '{prompt_template_name}' not found")
        return {"CrossSum": 0.0, "error_message": "PromptTemplateNotFound"}

    language_scores = {}
    all_rouge1_scores = []
    all_rouge2_scores = []
    all_rougeL_scores = []
    detailed_results = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        
        try:
            # Load dataset for this language
            dataset = load_crosssum_data_robust(lang_code, dataset_name, dataset_split)
            
            if len(dataset) == 0:
                logger.warning(f"No data found for {lang_code}")
                language_scores[lang_code] = None
                continue

            # Parse split to get number of samples
            if ':' in dataset_split:
                num_samples = int(dataset_split.split(':')[1].rstrip(']'))
            else:
                num_samples = len(dataset)
            
            num_samples = min(num_samples, len(dataset))
            subset = dataset.select(range(num_samples))

            prompts = []
            references = []
            lang_detailed_results = []

            for example_idx, example in enumerate(tqdm(subset, desc=f"Preparing {lang_code.upper()}")):
                source_text = example.get("text", "")
                reference_summary = example.get("summary", "")

                if not source_text.strip() or not reference_summary.strip():
                    continue

                # Create prompt
                prompt = prompt_manager.create_summarization_prompt(
                    source_text, "en", lang_code, prompt_template_name
                )

                prompts.append(prompt)
                references.append(reference_summary)

                lang_detailed_results.append({
                    "language": lang_code,
                    "example_id": example_idx,
                    "source_text": source_text[:500] + "..." if len(source_text) > 500 else source_text,
                    "reference_summary": reference_summary,
                    "model_summary": "",
                    "prompt": prompt[-500:] + "..." if len(prompt) > 500 else prompt,
                    "lang_code": example.get("lang", ""),
                    "source_url": example.get("source_url", ""),
                    "target_url": example.get("target_url", ""),
                    "is_valid": False
                })

            if not prompts:
                logger.warning(f"No valid examples for {lang_code}")
                language_scores[lang_code] = {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
                continue

            logger.info(f"Starting summarization for {len(prompts)} examples in {lang_code}")

            # Generate summaries
            predictions = batch_summarize(pipe, prompts, max_new_tokens, generation_batch_size)

            # Update detailed results with predictions
            for i, prediction in enumerate(predictions):
                if i < len(lang_detailed_results):
                    lang_detailed_results[i]["model_summary"] = prediction
                    lang_detailed_results[i]["is_valid"] = (
                        prediction.strip() and 
                        prediction not in ["[SUMMARIZATION ERROR]", "[INVALID_SUMMARY]"]
                    )

            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(predictions, references)
            language_scores[lang_code] = rouge_scores

            # Add to overall scores
            if rouge_scores["ROUGE-1"] > 0:
                all_rouge1_scores.append(rouge_scores["ROUGE-1"])
                all_rouge2_scores.append(rouge_scores["ROUGE-2"])
                all_rougeL_scores.append(rouge_scores["ROUGE-L"])

            detailed_results.extend(lang_detailed_results)

            logger.info(f"✅ {lang_code.upper()}: ROUGE-1={rouge_scores['ROUGE-1']:.2f}, "
                       f"ROUGE-2={rouge_scores['ROUGE-2']:.2f}, ROUGE-L={rouge_scores['ROUGE-L']:.2f}")

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            traceback.print_exc()
            language_scores[lang_code] = None

    # Calculate overall averages
    overall_scores = {
        "ROUGE-1": np.mean(all_rouge1_scores) if all_rouge1_scores else 0.0,
        "ROUGE-2": np.mean(all_rouge2_scores) if all_rouge2_scores else 0.0,
        "ROUGE-L": np.mean(all_rougeL_scores) if all_rougeL_scores else 0.0
    }

    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_crosssum_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            language_scores,
            overall_scores,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed results saved to: {saved_path}")

    # Prepare final results
    final_scores = {"CrossSum": overall_scores["ROUGE-1"]}  # Use ROUGE-1 as primary metric
    for metric in ["ROUGE-1", "ROUGE-2", "ROUGE-L"]:
        final_scores[f"CrossSum_{metric}"] = overall_scores[metric]
    
    for lang, scores in language_scores.items():
        if scores:
            final_scores[f"CrossSum_{lang}"] = scores["ROUGE-1"]

    logger.info(f"Overall CrossSum ROUGE-1: {overall_scores['ROUGE-1']:.2f}")
    return final_scores

def inspect_dataset_structure(dataset_name: str = DEFAULT_DATASET_NAME):
    """Helper function to inspect the dataset structure"""
    logger.info("🔍 Inspecting CrossSum dataset structure...")
    try:
        dataset = load_dataset(dataset_name, field="examples", trust_remote_code=True)
        logger.info(f"Dataset splits: {list(dataset.keys())}")
        
        for split_name in dataset.keys():
            logger.info(f"\n--- {split_name.upper()} SPLIT ---")
            split_data = dataset[split_name]
            logger.info(f"Number of examples: {len(split_data)}")
            
            if len(split_data) > 0:
                first_example = split_data[0]
                logger.info(f"First example keys: {list(first_example.keys()) if isinstance(first_example, dict) else 'Not a dict'}")
                
                # Check language distribution
                lang_counts = {}
                for i in range(min(100, len(split_data))):
                    example = split_data[i]
                    if isinstance(example, dict) and 'lang' in example:
                        lang = example['lang']
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                logger.info(f"Languages found: {dict(sorted(lang_counts.items()))}")
            break
        
    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")

# Test function for standalone execution
if __name__ == '__main__':
    import sys
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set CUDA devices if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    parser = argparse.ArgumentParser(description="Standalone CrossSum Evaluation")
    parser.add_argument("--model_name", type=str, default="sarvamai/sarvam-1")
    parser.add_argument("--dataset_split", type=str, default="test[:50]")
    parser.add_argument("--target_languages", nargs='+',  default=["hi", "en", "bn","gu","kn","ml","mr","or","pa","ta","te"])
    parser.add_argument("--save_detailed", action="store_true", help="Save detailed outputs to JSON file")
    parser.add_argument("--inspect_dataset", action="store_true", help="Inspect dataset structure")
    parser.add_argument("--prompt_template", type=str, default="crosssum_0shot", choices=["crosssum_0shot", "crosssum_3shot"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    
    args = parser.parse_args()
    
    logger.info(f"--- Standalone CrossSum Test: {args.model_name} ---")
    
    if args.inspect_dataset:
        inspect_dataset_structure()
    
    try:
        # Initialize model pipeline
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Model pipeline created successfully!")
        
        # Run evaluation
        eval_args = {
            "pipe": pipe,
            "tokenizer": tokenizer,
            "model_name_for_logging": args.model_name,
            "device": pipe.device,
            "dataset_split": args.dataset_split,
            "target_languages": args.target_languages,
            "save_detailed": args.save_detailed,
            "prompt_template_name": args.prompt_template,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.batch_size,
            "process_id": 0
        }
        
        results = evaluate_crosssum_in(**eval_args)
        
        print("\n" + "="*70)
        print("🏆 CROSSSUM EVALUATION RESULTS 🏆")
        print("="*70)
        print(json.dumps(results, indent=2))
        print("="*70)
        
    except Exception as e:
        logger.error(f"Failed to run CrossSum evaluation: {e}")
        traceback.print_exc()
    
    logger.info("CrossSum evaluation completed!")