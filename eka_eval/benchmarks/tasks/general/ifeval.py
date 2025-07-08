# eka_eval/benchmarks/tasks/general/ifeval.py

import torch
import re
import json
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import string

from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DATASET_NAME_IFEVAL = "google/IFEval"
DEFAULT_SPLIT_IFEVAL = "train"
DEFAULT_MAX_NEW_TOKENS_IFEVAL = 2048  # IFEval needs longer responses
DEFAULT_GENERATION_BATCH_SIZE_IFEVAL = 4
DEFAULT_PROMPT_TEMPLATE_KEY = "ifeval_basic"
PROMPT_FILE_BENCHMARK_KEY = "ifeval"
PROMPT_FILE_CATEGORY = "general"

def save_detailed_ifeval_results(
    results_data: List[Dict],
    model_name: str,
    dataset_name: str,
    accuracy: float,
    results_dir: str,
    process_id: int = 0
) -> str:
    """Save detailed IFEval results to JSON file."""
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ifeval_{model_clean}_{dataset_clean}_p{process_id}_{timestamp}.json"
    filepath = os.path.join(detailed_dir, filename)
    
    # Calculate constraint-wise statistics
    constraint_stats = {}
    for result in results_data:
        for constraint in result.get('constraints_checked', {}):
            if constraint not in constraint_stats:
                constraint_stats[constraint] = {'total': 0, 'passed': 0}
            constraint_stats[constraint]['total'] += 1
            if result['constraints_checked'][constraint]:
                constraint_stats[constraint]['passed'] += 1
    
    # Add success rates
    for constraint in constraint_stats:
        total = constraint_stats[constraint]['total']
        passed = constraint_stats[constraint]['passed']
        constraint_stats[constraint]['success_rate'] = (passed / total * 100) if total > 0 else 0.0
    
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_instructions": len(results_data),
        "instructions_followed": sum(1 for r in results_data if r["all_constraints_satisfied"]),
        "overall_accuracy": accuracy,
        "constraint_statistics": constraint_stats,
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
        logger.info(f"Detailed IFEval results saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save detailed IFEval results: {e}")
        return ""

class IFEvalConstraintChecker:
    """Class to check various IFEval constraints."""
    
    @staticmethod
    def check_word_count(text: str, min_words: int = None, max_words: int = None, exact_words: int = None) -> bool:
        """Check word count constraints."""
        words = text.strip().split()
        word_count = len(words)
        
        if exact_words is not None:
            return word_count == exact_words
        if min_words is not None and word_count < min_words:
            return False
        if max_words is not None and word_count > max_words:
            return False
        return True
    
    @staticmethod
    def check_keyword_presence(text: str, keywords: List[str], must_include: bool = True) -> bool:
        """Check if keywords are present or absent."""
        text_lower = text.lower()
        for keyword in keywords:
            keyword_present = keyword.lower() in text_lower
            if must_include and not keyword_present:
                return False
            if not must_include and keyword_present:
                return False
        return True
    
    @staticmethod
    def check_sentence_count(text: str, min_sentences: int = None, max_sentences: int = None, exact_sentences: int = None) -> bool:
        """Check sentence count constraints."""
        # Simple sentence splitting (can be improved)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        
        if exact_sentences is not None:
            return sentence_count == exact_sentences
        if min_sentences is not None and sentence_count < min_sentences:
            return False
        if max_sentences is not None and sentence_count > max_sentences:
            return False
        return True
    
    @staticmethod
    def check_paragraph_count(text: str, min_paragraphs: int = None, max_paragraphs: int = None, exact_paragraphs: int = None) -> bool:
        """Check paragraph count constraints."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        if exact_paragraphs is not None:
            return paragraph_count == exact_paragraphs
        if min_paragraphs is not None and paragraph_count < min_paragraphs:
            return False
        if max_paragraphs is not None and paragraph_count > max_paragraphs:
            return False
        return True
    
    @staticmethod
    def check_json_format(text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text.strip())
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def check_bullet_points(text: str, min_bullets: int = None) -> bool:
        """Check for bullet point format."""
        bullet_patterns = [r'^\s*[•\-\*]\s+', r'^\s*\d+\.\s+', r'^\s*[a-zA-Z]\.\s+']
        lines = text.split('\n')
        bullet_count = 0
        
        for line in lines:
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    bullet_count += 1
                    break
        
        if min_bullets is not None:
            return bullet_count >= min_bullets
        return bullet_count > 0
    
    @staticmethod
    def check_starts_with(text: str, prefix: str) -> bool:
        """Check if text starts with specific prefix."""
        return text.strip().lower().startswith(prefix.lower())
    
    @staticmethod
    def check_ends_with(text: str, suffix: str) -> bool:
        """Check if text ends with specific suffix."""
        return text.strip().lower().endswith(suffix.lower())
    
    @staticmethod
    def check_contains_pattern(text: str, pattern: str) -> bool:
        """Check if text contains a regex pattern."""
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False
    
    @staticmethod
    def check_no_punctuation(text: str) -> bool:
        """Check if text contains no punctuation."""
        return not any(char in string.punctuation for char in text)
    
    @staticmethod
    def check_all_caps(text: str) -> bool:
        """Check if text is all uppercase."""
        return text.isupper()
    
    @staticmethod
    def check_all_lowercase(text: str) -> bool:
        """Check if text is all lowercase."""
        return text.islower()

def parse_ifeval_instruction(instruction: str) -> Dict[str, Any]:
    """
    Parse IFEval instruction to extract constraints.
    This is a simplified parser - you may need to enhance it based on actual instruction formats.
    """
    constraints = {}
    
    # Word count patterns
    word_patterns = [
        (r'(?:at least|minimum of|min)\s+(\d+)\s+words?', 'min_words'),
        (r'(?:at most|maximum of|max)\s+(\d+)\s+words?', 'max_words'),
        (r'(?:exactly|precisely)\s+(\d+)\s+words?', 'exact_words'),
        (r'(\d+)\s+words?\s+(?:exactly|precisely)', 'exact_words'),
    ]
    
    for pattern, constraint_type in word_patterns:
        match = re.search(pattern, instruction, re.IGNORECASE)
        if match:
            constraints[constraint_type] = int(match.group(1))
    
    # Keyword patterns
    include_patterns = [
        r'(?:include|mention|use)\s+(?:the\s+)?(?:word|phrase|keyword)s?\s+["\']([^"\']+)["\']',
        r'(?:must\s+)?(?:contain|include)\s+["\']([^"\']+)["\']',
    ]
    
    exclude_patterns = [
        r'(?:do not|don\'t|avoid|exclude)\s+(?:use|mention|include)\s+["\']([^"\']+)["\']',
        r'(?:without|exclude|avoid)\s+(?:the\s+)?(?:word|phrase)s?\s+["\']([^"\']+)["\']',
    ]
    
    for pattern in include_patterns:
        matches = re.findall(pattern, instruction, re.IGNORECASE)
        if matches:
            constraints['include_keywords'] = matches
    
    for pattern in exclude_patterns:
        matches = re.findall(pattern, instruction, re.IGNORECASE)
        if matches:
            constraints['exclude_keywords'] = matches
    
    # Format constraints
    if re.search(r'(?:in\s+)?json\s+format', instruction, re.IGNORECASE):
        constraints['json_format'] = True
    
    if re.search(r'bullet\s+points?|bulleted\s+list', instruction, re.IGNORECASE):
        constraints['bullet_points'] = True
    
    # Sentence count
    sentence_patterns = [
        (r'(?:at least|minimum of)\s+(\d+)\s+sentences?', 'min_sentences'),
        (r'(?:at most|maximum of)\s+(\d+)\s+sentences?', 'max_sentences'),
        (r'(?:exactly|precisely)\s+(\d+)\s+sentences?', 'exact_sentences'),
    ]
    
    for pattern, constraint_type in sentence_patterns:
        match = re.search(pattern, instruction, re.IGNORECASE)
        if match:
            constraints[constraint_type] = int(match.group(1))
    
    # Start/end constraints
    start_match = re.search(r'(?:start|begin)\s+with\s+["\']([^"\']+)["\']', instruction, re.IGNORECASE)
    if start_match:
        constraints['starts_with'] = start_match.group(1)
    
    end_match = re.search(r'(?:end|finish)\s+with\s+["\']([^"\']+)["\']', instruction, re.IGNORECASE)
    if end_match:
        constraints['ends_with'] = end_match.group(1)
    
    return constraints

def evaluate_ifeval_response(response: str, constraints: Dict[str, Any]) -> Dict[str, bool]:
    """Evaluate if a response satisfies IFEval constraints."""
    checker = IFEvalConstraintChecker()
    results = {}
    
    # Word count checks
    if any(key in constraints for key in ['min_words', 'max_words', 'exact_words']):
        results['word_count'] = checker.check_word_count(
            response,
            constraints.get('min_words'),
            constraints.get('max_words'),
            constraints.get('exact_words')
        )
    
    # Keyword inclusion checks
    if 'include_keywords' in constraints:
        results['keyword_inclusion'] = checker.check_keyword_presence(
            response, constraints['include_keywords'], must_include=True
        )
    
    # Keyword exclusion checks
    if 'exclude_keywords' in constraints:
        results['keyword_exclusion'] = checker.check_keyword_presence(
            response, constraints['exclude_keywords'], must_include=False
        )
    
    # Sentence count checks
    if any(key in constraints for key in ['min_sentences', 'max_sentences', 'exact_sentences']):
        results['sentence_count'] = checker.check_sentence_count(
            response,
            constraints.get('min_sentences'),
            constraints.get('max_sentences'),
            constraints.get('exact_sentences')
        )
    
    # Format checks
    if constraints.get('json_format'):
        results['json_format'] = checker.check_json_format(response)
    
    if constraints.get('bullet_points'):
        results['bullet_points'] = checker.check_bullet_points(response)
    
    # Start/end checks
    if 'starts_with' in constraints:
        results['starts_with'] = checker.check_starts_with(response, constraints['starts_with'])
    
    if 'ends_with' in constraints:
        results['ends_with'] = checker.check_ends_with(response, constraints['ends_with'])
    
    return results

def evaluate_ifeval(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_IFEVAL,
    dataset_split: str = DEFAULT_SPLIT_IFEVAL,
    prompt_template_name: str = DEFAULT_PROMPT_TEMPLATE_KEY,
    prompt_file_benchmark_key: str = PROMPT_FILE_BENCHMARK_KEY,
    prompt_file_category: str = PROMPT_FILE_CATEGORY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_IFEVAL,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_IFEVAL,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_detailed: bool = True,
    **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting IFEval: {model_name_for_logging} on {dataset_name}")
    logger.info(f"Generation config: max_new_tokens={max_new_tokens}, batch_size={generation_batch_size}")

    # Load prompt template
    prompt_template_dict = get_prompt_template(
        benchmark_name=prompt_file_benchmark_key,
        template_name=prompt_template_name,
        specific_task_group=prompt_file_category
    )
    
    if not prompt_template_dict:
        logger.error(f"Prompt template '{prompt_template_name}' not found")
        return {"IFEval": 0.0, "error_message": f"PromptTemplate '{prompt_template_name}' NotFound"}

    # Load dataset
    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"IFEval": 0.0, "error_message": f"DatasetLoadFailed: {e}"}
    
    logger.info(f"P{process_id}: Loaded IFEval ({len(full_data)} examples).")

    # Handle multi-GPU data splitting
    if num_gpus > 1:
        total = len(full_data)
        per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1:
            end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    
    if len(subset_to_process) == 0:
        return {"IFEval": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} IFEval examples.")

    # Initialize tracking
    total_instructions = 0
    instructions_followed = 0
    prompts_for_batch, original_items_for_batch_info = [], []
    detailed_results = []

    # Main evaluation loop
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - IFEval Eval")):
        prompt_text = item_data.get("prompt")
        
        if not prompt_text:
            logger.warning(f"Missing prompt for item {item_idx}")
            continue

        # Parse constraints from the instruction
        constraints = parse_ifeval_instruction(prompt_text)
        
        # Generate prompt using template
        main_q_data = {"prompt": prompt_text}
        llm_prompt = format_prompt(prompt_template_dict, **main_q_data)
        
        prompts_for_batch.append(llm_prompt)
        original_items_for_batch_info.append({
            'llm_prompt': llm_prompt,
            'original_prompt': prompt_text,
            'constraints': constraints,
            'item_idx': item_idx
        })

        # Process batch
        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    orig_info = original_items_for_batch_info[k]
                    
                    # Extract generated text
                    if raw_out_list and len(raw_out_list) > 0 and 'generated_text' in raw_out_list[0]:
                        generated_full_text = raw_out_list[0]['generated_text']
                        # Extract only the new part
                        if generated_full_text.startswith(orig_info['llm_prompt']):
                            generated_response = generated_full_text[len(orig_info['llm_prompt']):].strip()
                        else:
                            generated_response = generated_full_text.strip()
                    else:
                        generated_response = "[NO_GENERATION]"
                    
                    # Evaluate constraints
                    constraints_checked = evaluate_ifeval_response(generated_response, orig_info['constraints'])
                    all_constraints_satisfied = all(constraints_checked.values()) if constraints_checked else True
                    
                    total_instructions += 1
                    if all_constraints_satisfied:
                        instructions_followed += 1
                    
                    # Store detailed result
                    if save_detailed:
                        detailed_result = {
                            "instruction_id": orig_info['item_idx'],
                            "original_instruction": orig_info['original_prompt'],
                            "constraints_parsed": orig_info['constraints'],
                            "generated_response": generated_response,
                            "constraints_checked": constraints_checked,
                            "all_constraints_satisfied": all_constraints_satisfied,
                            "prompt_used": orig_info['llm_prompt']
                        }
                        detailed_results.append(detailed_result)
                        
            except Exception as e_batch:
                logger.error(f"P{process_id}: IFEval generation batch error: {e_batch}", exc_info=True)
                # Add fallback for failed batch
                for info in original_items_for_batch_info:
                    total_instructions += 1
                    
                    if save_detailed:
                        detailed_result = {
                            "instruction_id": info['item_idx'],
                            "original_instruction": info['original_prompt'],
                            "constraints_parsed": info['constraints'],
                            "generated_response": "[GENERATION_ERROR]",
                            "constraints_checked": {},
                            "all_constraints_satisfied": False,
                            "prompt_used": info['llm_prompt']
                        }
                        detailed_results.append(detailed_result)
            
            # Reset batch
            prompts_for_batch, original_items_for_batch_info = [], []
    
    # Calculate accuracy
    if total_instructions == 0:
        return {"IFEval": 0.0}
    
    accuracy = (instructions_followed / total_instructions) * 100
    
    logger.info(f"P{process_id} - Final IFEval Accuracy: {accuracy:.2f}% ({instructions_followed}/{total_instructions})")
    
    # Save detailed results
    if save_detailed and detailed_results:
        saved_path = save_detailed_ifeval_results(
            detailed_results,
            model_name_for_logging,
            dataset_name,
            accuracy,
            results_dir,
            process_id
        )
        if saved_path:
            logger.info(f"Detailed IFEval results with {len(detailed_results)} examples saved to: {saved_path}")
    
    return {"IFEval": accuracy}
