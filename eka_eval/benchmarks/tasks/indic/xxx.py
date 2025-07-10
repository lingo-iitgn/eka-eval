# xorqa_enhanced_eval.py
import torch
import re
import os
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import traceback
from accelerate import Accelerator, DistributedDataParallelKwargs
from collections import Counter
import string

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "google/IndicGenBench_xorqa_in"
DEFAULT_TARGET_LANGUAGES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
DEFAULT_SPLIT = 'test[:100]'
DEFAULT_MAX_NEW_TOKENS = 256
PROMPT_FILE_BENCHMARK_KEY = "xorqa_in"
PROMPT_FILE_CATEGORY = "indic"

class XORQAPromptManager:
    """Enhanced prompt management for XOR-QA"""
    
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
    
    def create_xorqa_prompt(self, question: str, language: str, template_name: str = "xorqa_3shot") -> str:
        """Create language-specific XOR-QA prompt"""
        template_config = self.get_prompt_template(template_name)
        
        if not template_config:
            # Fallback to basic template
            return f"Question: {question}\nAnswer:"
        
        # Get language-specific prompts
        lang_prompts = template_config.get("language_specific_prompts", {})
        
        if language in lang_prompts:
            prompt_template = lang_prompts[language]
        else:
            prompt_template = template_config.get("template", lang_prompts.get("default", ""))
        
        # Use few-shot examples if available
        if template_config.get("use_few_shot", False):
            few_shot_examples = template_config.get("few_shot_examples", {}).get(language, [])
            if few_shot_examples:
                examples_text = []
                for example in few_shot_examples:
                    if language in self.language_mappings:
                        lang_name = self.language_mappings[language]
                        if language == "hi":
                            example_text = f"प्रश्न: {example['question']}\nउत्तर: {example['answer']}"
                        elif language == "bn":
                            example_text = f"প্রশ্ন: {example['question']}\nউত্তর: {example['answer']}"
                        elif language == "gu":
                            example_text = f"પ્રશ્ન: {example['question']}\nજવાબ: {example['answer']}"
                        elif language == "ta":
                            example_text = f"கேள்வி: {example['question']}\nபதில்: {example['answer']}"
                        elif language == "te":
                            example_text = f"ప్రశ్న: {example['question']}\nసమాధానం: {example['answer']}"
                        elif language == "kn":
                            example_text = f"ಪ್ರಶ್ನೆ: {example['question']}\nಉತ್ತರ: {example['answer']}"
                        elif language == "ml":
                            example_text = f"ചോദ്യം: {example['question']}\nഉത്തരം: {example['answer']}"
                        elif language == "mr":
                            example_text = f"प्रश्न: {example['question']}\nउत्तर: {example['answer']}"
                        elif language == "pa":
                            example_text = f"ਸਵਾਲ: {example['question']}\nਜਵਾਬ: {example['answer']}"
                        elif language == "or":
                            example_text = f"ପ୍ରଶ୍ନ: {example['question']}\nଉତ୍ତର: {example['answer']}"
                        elif language == "as":
                            example_text = f"প্ৰশ্ন: {example['question']}\nউত্তৰ: {example['answer']}"
                        elif language == "ur":
                            example_text = f"سوال: {example['question']}\nجواب: {example['answer']}"
                        else:
                            example_text = f"Question: {example['question']}\nAnswer: {example['answer']}"
                    else:
                        example_text = f"Question: {example['question']}\nAnswer: {example['answer']}"
                    
                    examples_text.append(example_text)
                
                examples_str = "\n\n".join(examples_text)
                prompt_template = f"{examples_str}\n\n{prompt_template}"
        
        # Format the prompt
        formatted_prompt = prompt_template.format(question=question)
        
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
    """Load XOR-QA prompt templates"""
    return {
        "xorqa_0shot": {
            "template": "Question: {question}\nAnswer:",
            "description": "Zero-shot open-domain question answering prompt",
            "use_few_shot": False,
            "language_specific_prompts": {
                "hi": "प्रश्न: {question}\nउत्तर:",
                "bn": "প্রশ্ন: {question}\nউত্তর:",
                "gu": "પ્રશ્ન: {question}\nજવાબ:",
                "ta": "கேள்வி: {question}\nபதில்:",
                "te": "ప্রశ্న: {question}\nసమাధానం:",
                "kn": "ಪ್ರಶ್ನೆ: {question}\nಉತ್ತರ:",
                "ml": "ചോദ്യം: {question}\nഉത്തരം:",
                "mr": "प्रश्न: {question}\nउत्तर:",
                "pa": "ਸਵਾਲ: {question}\nਜਵਾਬ:",
                "or": "ପ୍ରଶ୍ନ: {question}\nଉତ୍ତର:",
                "as": "প্ৰশ্ন: {question}\nউত্তৰ:",
                "ur": "سوال: {question}\nجواب:",
                "en": "Question: {question}\nAnswer:",
                "default": "Question: {question}\nAnswer:"
            },
            "few_shot_examples": {
                "hi": [
                    {"question": "फ्रांस की राजधानी क्या है?", "answer": "पेरिस"},
                    {"question": "दुनिया का सबसे बड़ा महासागर कौन सा है?", "answer": "प्रशांत महासागर"},
                    {"question": "भारत के पहले प्रधानमंत्री कौन थे?", "answer": "जवाहरलाल नेहरू"}
                ],
                "bn": [
                    {"question": "ফ্রান্সের রাজধানী কী?", "answer": "প্যারিস"},
                    {"question": "বিশ্বের বৃহত্তম মহাসাগর কোনটি?", "answer": "প্রশান্ত মহাসাগর"},
                    {"question": "ভারতের প্রথম প্রধানমন্ত্রী কে ছিলেন?", "answer": "জওহরলাল নেহরু"}
                ],
                "gu": [
                    {"question": "ફ્રાન્સની રાજધાની કઈ છે?", "answer": "પેરિસ"},
                    {"question": "વિશ્વનો સૌથી મોટો મહાસાગર કયો છે?", "answer": "પ્રશાંત મહાસાગર"},
                    {"question": "ભારતના પ્રથમ વડાપ્રધાન કોણ હતા?", "answer": "જવાહરલાલ નેહરુ"}
                ],
                "ta": [
                    {"question": "பிரான்சின் தலைநகரம் என்ன?", "answer": "பாரிஸ்"},
                    {"question": "உலகின் மிகப்பெரிய கடல் எது?", "answer": "பசிபிக் பெருங்கடல்"},
                    {"question": "இந்தியாவின் முதல் பிரதமர் யார்?", "answer": "ஜவஹர்லால் நேரு"}
                ],
                "te": [
                    {"question": "ఫ్రాన్స్ రాజధాని ఏమిటి?", "answer": "పారిస్"},
                    {"question": "ప్రపంచంలోని అతిపెద్ద సముద్రం ఏది?", "answer": "పసిఫిక్ మహాసముద్రం"},
                    {"question": "భారతదేశం యొక్క మొదటి ప్రధానమంత్రి ఎవరు?", "answer": "జవహర్లాల్ నెహ్రూ"}
                ],
                "en": [
                    {"question": "What is the capital of France?", "answer": "Paris"},
                    {"question": "What is the largest ocean in the world?", "answer": "Pacific Ocean"},
                    {"question": "Who was the first Prime Minister of India?", "answer": "Jawaharlal Nehru"}
                ]
            }
        }
    }

def load_xorqa_data_robust(indic_lang_code: str, dataset_name: str, split: str) -> Any:
    """Robust data loading for XOR-QA dataset"""
    cache_file = f"xorqa_{indic_lang_code}_test.json"
    
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
    logger.info(f"Loading XOR-QA dataset for {indic_lang_code}...")
    try:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        logger.info(f"✅ Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        raise ValueError("Failed to load XOR-QA dataset")
    
    # Extract relevant examples
    extracted_examples = []
    
    # Check all splits
    for split_name in ['test', 'validation', 'train']:
        if split_name not in dataset:
            continue
            
        logger.info(f"Searching in {split_name} split...")
        split_data = dataset[split_name]
        
        for item in split_data:
            if 'examples' in item and isinstance(item['examples'], dict):
                examples = item['examples']
                if examples.get('lang') == indic_lang_code:
                    extracted_examples.append(examples)
            
            # Break early if we have enough examples
            if len(extracted_examples) >= 200:
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

def batch_xorqa_generate(pipe: Any, prompts: List[str], max_new_tokens: int = 256, batch_size: int = 2) -> List[str]:
    """Perform batch XOR-QA generation with optimized parameters"""
    answers = []
    
    with tqdm(total=len(prompts), desc="Generating answers") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_answers = []
            
            for prompt in batch_prompts:
                try:
                    gen_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": 0.6,  # Optimized for factual QA
                        "top_p": 0.85,
                        "top_k": 40,
                        "repetition_penalty": 1.15,
                        "early_stopping": True,
                        "return_full_text": False
                    }
                    
                    with torch.no_grad():
                        outputs = pipe(prompt, **gen_config)
                    
                    answer = ""
                    if outputs and isinstance(outputs, list) and outputs[0]:
                        answer = outputs[0].get('generated_text', "").strip()
                    
                    answer = clean_xorqa_output(answer)
                    batch_answers.append(answer)
                    
                except Exception as e:
                    logger.debug(f"Error in XOR-QA generation: {e}")
                    batch_answers.append("[XORQA ERROR]")
            
            answers.extend(batch_answers)
            pbar.update(len(batch_prompts))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return answers

def clean_xorqa_output(text: str) -> str:
    """Enhanced cleaning for XOR-QA output"""
    if not text:
        return text
    
    text = text.strip()
    
    # Take first meaningful line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Choose the longest meaningful line (likely the answer)
        text = max(lines, key=len) if len(lines) > 1 else lines[0]
    
    # Remove common prefixes
    patterns_to_remove = [
        r'^(Answer|उत्तर|উত্তর|જવાબ|பதில்|సమాధానం|ಉತ್ತರ|ഉത്തরം|उत्तर|ଉତ୍ତର|ਜਵਾਬ|উত্তৰ|جواب):\s*',
        r'^(Response|Output|Result):\s*',
        r'^(Question|प्रश्न|প্রশ্ন|પ્રશ્ન|கேள்வி|ప్రశ్న|ಪ್ರಶ್ನೆ|ചോദ്യം|प्रश্न|ପ୍ରଶ୍ନ|ਸਵਾਲ|প্ৰশ্ন|سوال):.*',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Stop at certain patterns
    stop_patterns = [
        r'\n\n',
        r'\nQuestion:',
        r'\nAnswer:',
        r'\nNote:',
        r'\nExplanation:',
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    
    # Apply length constraints
    if len(text) > 256:
        # Cut at sentence boundary if possible
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            text = sentences[0] + '.'
        else:
            text = text[:256].rsplit(' ', 1)[0] + "..."
    
    if len(text) < 1:
        return "[INVALID_ANSWER]"
    
    return text.strip()

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (enhanced version)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|एक|ek|একটি|eka|એક|ஒரு|ఒక|ಒಂದು|ഒരു|एक|ଗୋଟିଏ|ਇੱਕ|এটা|ایک)\b', ' ', text, flags=re.IGNORECASE)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate enhanced F1 score between prediction and ground truth"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0
    
    precision = 1.0 * num_common / len(prediction_tokens)
    recall = 1.0 * num_common / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction: str, ground_truth: str) -> int:
    """Calculate exact match score"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_xorqa_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """Evaluate XOR-QA predictions using enhanced F1 and EM metrics"""
    if not predictions or not references or len(predictions) != len(references):
        return {"f1": 0.0, "exact_match": 0.0}
    
    f1_scores = []
    em_scores = []
    
    for pred, ref_list in zip(predictions, references):
        if isinstance(ref_list, list):
            # Take maximum score across all reference answers
            max_f1 = max(f1_score(pred, ref) for ref in ref_list)
            max_em = max(exact_match_score(pred, ref) for ref in ref_list)
        else:
            max_f1 = f1_score(pred, ref_list)
            max_em = exact_match_score(pred, ref_list)
        
        f1_scores.append(max_f1)
        em_scores.append(max_em)
    
    return {
        "f1": np.mean(f1_scores) * 100,  # Convert to percentage
        "exact_match": np.mean(em_scores) * 100  # Convert to percentage
    }

def save_detailed_xorqa_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    language_scores: Dict[str, Dict],
    overall_scores: Dict[str, float],
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed XOR-QA results to JSON file"""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xorqa_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_examples": len(results_data),
        "valid_examples": sum(1 for r in results_data if r.get("is_valid", False)),
        "overall_scores": overall_scores,
        "language_scores": language_scores,
        "timestamp": datetime.now().isoformat(),
        "process_id": process_id,
        "metric": "F1"
    }
    
    full_data = {
        "summary": summary,
        "detailed_results": results_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed XOR-QA results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")
        return ""

def evaluate_xorqa_in(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None,
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generation_batch_size: int = 2,
    prompt_template_name: str = "xorqa_3shot",
    results_dir: str = "results_output",
    save_detailed: bool = True,
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:
    
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    logger.info(f"Starting XOR-QA Evaluation: {model_name_for_logging}")
    
    # Load prompt templates
    prompt_templates = load_prompt_templates()
    prompt_manager = XORQAPromptManager(prompt_templates)
    
    if prompt_template_name not in prompt_templates:
        logger.error(f"Prompt template '{prompt_template_name}' not found")
        return {"XOR-QA": 0.0, "error_message": "PromptTemplateNotFound"}

    language_scores = {}
    all_f1_scores = []
    all_em_scores = []
    detailed_results = []

    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        
        try:
            # Load dataset for this language
            dataset = load_xorqa_data_robust(lang_code, dataset_name, dataset_split)
            
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
                question = example.get("question", "")
                answers = example.get("answers", [])
                
                if isinstance(answers, str):
                    answers = [answers]
                elif not isinstance(answers, list):
                    answers = []

                if not question.strip() or not answers:
                    continue

                # Create prompt
                prompt = prompt_manager.create_xorqa_prompt(
                    question, lang_code, prompt_template_name
                )

                prompts.append(prompt)
                references.append(answers)

                lang_detailed_results.append({
                    "language": lang_code,
                    "example_id": example_idx,
                    "question": question,
                    "reference_answers": answers,
                    "model_answer": "",
                    "prompt": prompt[-500:] + "..." if len(prompt) > 500 else prompt,
                    "is_valid": False
                })

            if not prompts:
                logger.warning(f"No valid examples for {lang_code}")
                language_scores[lang_code] = {"f1": 0.0, "exact_match": 0.0}
                continue

            logger.info(f"Starting QA generation for {len(prompts)} examples in {lang_code}")

            # Generate answers
            predictions = batch_xorqa_generate(pipe, prompts, max_new_tokens, generation_batch_size)

            # Update detailed results with predictions
            for i, prediction in enumerate(predictions):
                if i < len(lang_detailed_results):
                    lang_detailed_results[i]["model_answer"] = prediction
                    lang_detailed_results[i]["is_valid"] = (
                        prediction.strip() and 
                        prediction not in ["[XORQA ERROR]", "[INVALID_ANSWER]"]
                    )

            # Calculate F1 and EM scores
            metrics = evaluate_xorqa_metrics(predictions, references)
            language_scores[lang_code] = metrics

            # Add to overall scores
            if metrics["f1"] > 0:
                all_f1_scores.append(metrics["f1"])
                all_em_scores.append(metrics["exact_match"])

            detailed_results.extend(lang_detailed_results)

            logger.info(f"✅ {lang_code.upper()}: F1={metrics['f1']:.2f}, "
                       f"EM={metrics['exact_match']:.2f}")

        except Exception as e:
            logger.error(f"Error processing language {lang_code}: {e}")
            traceback.print_exc()
            language_scores[lang_code] = None

    # Calculate overall averages
    overall_scores = {
        "f1": np.mean(all_f1_scores) if all_f1_scores else 0.0,
        "exact_match": np.mean(all_em_scores) if all_em_scores else 0.0,
        "languages_evaluated": len(all_f1_scores),
        "total_languages": len(target_languages)
    }

    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_xorqa_results(
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
    final_scores = {"XOR-QA": overall_scores["f1"]}  # Use F1 as primary metric
    final_scores["XOR-QA_F1"] = overall_scores["f1"]
    final_scores["XOR-QA_EM"] = overall_scores["exact_match"]
    
    for lang, scores in language_scores.items():
        if scores and "f1" in scores:
            final_scores[f"XOR-QA_{lang}"] = scores["f1"]

    logger.info(f"Overall XOR-QA F1: {overall_scores['f1']:.2f}")
    return final_scores

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
    
    parser = argparse.ArgumentParser(description="Standalone XOR-QA Evaluation")
    parser.add_argument("--model_name", type=str, default="sarvamai/sarvam-1")
    parser.add_argument("--dataset_split", type=str, default="test[:100]")
    parser.add_argument("--target_languages", nargs='+', default=["hi", "bn", "gu", "ta", "te"])
    parser.add_argument("--save_detailed", action="store_true", help="Save detailed outputs to JSON file")
    parser.add_argument("--prompt_template", type=str, default="xorqa_3shot", choices=["xorqa_0shot", "xorqa_3shot"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    
    args = parser.parse_args()
    
    logger.info(f"--- Standalone XOR-QA Test: {args.model_name} ---")
    
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
        
        results = evaluate_xorqa_in(**eval_args)
        
        print("\n" + "="*70)
        print("🏆 XOR-QA EVALUATION RESULTS (F1) 🏆")
        print("="*70)
        print(json.dumps(results, indent=2))
        
        # Print benchmark comparison
        print("\n📊 BENCHMARK COMPARISON:")
        print("Model Performance on XOR-QA (F1):")
        print("• Gemma-2-2B:   12.28")
        print("• Llama-3.2-3B: 18.37") 
        print("• Llama-3.1-8B: 24.00")
        print("• Sarvam-1:     25.27")
        if "XOR-QA" in results:
            print(f"• Your Model:   {results['XOR-QA']:.2f}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Failed to run XOR-QA evaluation: {e}")
        traceback.print_exc()
    
    logger.info("XOR-QA evaluation completed!")್ತರ:",
                "ml": "ചോദ്യം: {question}\nഉത্തരം:",
                "mr": "प्रश्न: {question}\nउत्तर:",
                "pa": "ਸਵਾਲ: {question}\nਜਵਾਬ:",
                "or": "ପ୍ରଶ୍ନ: {question}\nଉତ୍ତର:",
                "as": "প্ৰশ্ন: {question}\nউত্তৰ:",
                "ur": "سوال: {question}\nجواب:",
                "en": "Question: {question}\nAnswer:",
                "default": "Question: {question}\nAnswer:"
            }
        },
        "xorqa_3shot": {
            "template": "Question: {question}\nAnswer:",
            "description": "Few-shot open-domain question answering with examples",
            "use_few_shot": True,
            "num_few_shot_examples": 3,
            "language_specific_prompts": {
                "hi": "प्रश्न: {question}\nउत्तर:",
                "bn": "প্রশ্ন: {question}\nউত্তর:",
                "gu": "પ્રશ્ન: {question}\nજવાબ:",
                "ta": "கேள்வி: {question}\nபதில்:",
                "te": "ప্రశ্న: {question}\nసమাধানం:",
                "kn": "ಪ್ರಶ್ನೆ: {question}\nಉತ