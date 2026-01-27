# eka_eval/benchmarks/tasks/math/gsm8k.py
import torch
import re
from datasets import load_dataset
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
import json
import os
import logging
import sys
import argparse
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_GSM8K = "gsm8k"
DEFAULT_CONFIG_GSM8K = "main"
DEFAULT_SPLIT_GSM8K = "test[:5]"
DEFAULT_MAX_NEW_TOKENS_GSM8K = 512  # Increased to match lm-harness behavior
DEFAULT_GENERATION_BATCH_SIZE_GSM8K = 8
DEFAULT_NUM_FEWSHOT_GSM8K = 5

FEWSHOT_EXAMPLES_GSM8K_DEFAULT_SET = [
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15, and her grandparents gave her twice as much. How much more money does she need?",
        "answer": "Betty has 100 / 2 = $<<100/2=50>>50.\nGrandparents gave her 15 * 2 = $<<15*2=30>>30.\nTotal = 50 + 15 + 30 = <<50+15+30=95>>95.\n100 - 95 = $<<100-95=5>>5.\n#### 5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday she read 12 pages, and today twice as many. If she wants to read half the remaining pages tomorrow, how many pages is that?",
        "answer": "Today she read 12 x 2 = <<12*2=24>>24.\nTotal read so far = 12 + 24 = <<12+24=36>>36.\nRemaining = 120 - 36 = <<120-36=84>>84.\nHalf of 84 = <<84/2=42>>42.\n#### 42"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "3 pages x 2 friends = <<3*2=6>>6 pages per letter session.\nTwice a week = 6 x 2 = <<6*2=12>>12 pages per week.\nPer year = 12 x 52 = <<12*52=624>>624.\n#### 624"
    },
    {
        "question": "Mark planted 10 yellow flowers. He planted 80% more purple ones. Then 25% as many green flowers as the total of yellow and purple. How many total flowers?",
        "answer": "Purple = 10 x 0.8 = <<10*0.8=8>>8.\nTotal yellow + purple = 10 + 8 = <<10+8=18>>18.\nGreen = 0.25 x 18 = <<0.25*18=4.5>>4.5\nTotal = 10 + 8 + 4.5 = <<10+8+4.5=22.5>>22.5.\n#### 22.5"
    }
]


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for HuggingFace transformers pipeline"""
    
    def __init__(self, stop_sequences: List[str], tokenizer):
        """
        Args:
            stop_sequences: List of strings that should trigger stopping
            tokenizer: HuggingFace tokenizer to decode generated tokens
        """
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if any of the stop sequences appear in the generated text
        
        Args:
            input_ids: Generated token IDs so far
            scores: Model scores (not used but required by interface)
            
        Returns:
            True if generation should stop, False otherwise
        """

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
       
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                return True
                
        return False


def _format_gsm8k_prompt(question: str, fewshot_examples: List[Dict], num_few_shot: int) -> str:
    """
    Format prompt following lm-evaluation-harness template:
    doc_to_text: "Question: {{question}}\nAnswer:"
    """
    prompt = ""
    if num_few_shot > 0 and fewshot_examples:
        actual_fewshot = fewshot_examples[:num_few_shot]
        for ex in actual_fewshot:
           
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    
   
    prompt += f"Question: {question}\nAnswer:"
    return prompt

def _extract_gsm8k_final_answer_strict(text: Optional[str]) -> Optional[str]:
    if text is None: 
        return None
    
    # Fixed: single backslashes for regex
    match = re.search(r'####\s*(\-?[0-9\.\,]+)', text)
    if match:
        num_str = match.group(1).strip()
        num_str = num_str.replace(',', '')
        num_str = num_str.replace('$', '')
        num_str = num_str.rstrip('.')
        
        try:
            return str(float(num_str))
        except ValueError:
            logger.warning(f"GSM8K: Could not convert extracted answer '{num_str}' to float.")
            return num_str
    
    logger.debug(f"GSM8K: Strict-match failed, text: '{text[-100:]}'")
    return None


def _normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    
    answer = re.sub(r',', '', answer)
    answer = re.sub(r'\$', '', answer)  # Fixed
    answer = re.sub(r'(?s).*####\s*', '', answer)
    answer = answer.rstrip('.')
    answer = answer.lower().strip()
    
    return answer

def _extract_gsm8k_final_answer_flexible(text: Optional[str]) -> Optional[str]:
    """
    Extract answer using lm-harness "flexible-extract" filter:
    regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    group_select: -1 (take last match)
    
    Fallback when #### marker is not present
    """
    if text is None:
        return None
    
    # Find all numbers matching the pattern
    matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    
    if matches:
        # Take the last match (group_select: -1)
        last_match = matches[-1]
        # Get the non-empty group
        num_str = last_match[0] if last_match[0] else last_match[1]
        
        # Clean up
        num_str = num_str.replace(',', '')
        num_str = num_str.replace('$', '')
        num_str = num_str.rstrip('.')
        
        try:
            return str(float(num_str))
        except ValueError:
            logger.warning(f"GSM8K: Flexible extract failed to convert '{num_str}' to float.")
            return num_str
    
    return None


def _extract_gsm8k_answer(text: Optional[str]) -> Optional[str]:
    """
    Extract answer using lm-harness filter list logic:
    1. Try strict-match first (#### pattern)
    2. Fall back to flexible-extract if strict fails
    """
    # Try strict match first
    answer = _extract_gsm8k_final_answer_strict(text)
    
    # Fall back to flexible extract
    if answer is None:
        answer = _extract_gsm8k_final_answer_flexible(text)
    
    return answer


def _normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison following lm-harness rules:
    - ignore_case: true
    - ignore_punctuation: false (we keep punctuation)
    - regexes_to_ignore: [",", "\\$", "(?s).*#### ", "\\."]
    """
    if not answer:
        return ""
    
    # Apply regexes_to_ignore from lm-harness config
    answer = re.sub(r',', '', answer)           # Remove commas
    answer = re.sub(r'\\$', '', answer)         # Remove dollar signs
    answer = re.sub(r'(?s).*####\s*', '', answer)  # Remove everything before ####
    answer = answer.rstrip('.')                 # Remove trailing period
    
    # ignore_case: true
    answer = answer.lower()
    
    return answer.strip()


def evaluate_gsm8k(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_path: str = DEFAULT_DATASET_NAME_GSM8K,
    dataset_name: str = DEFAULT_CONFIG_GSM8K,
    dataset_split: str = DEFAULT_SPLIT_GSM8K,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_GSM8K,
    few_shot_examples_list: Optional[List[Dict]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_GSM8K,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_GSM8K,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output", 
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K following lm-evaluation-harness specifications
    
    Reference config from lm-harness:
    - task: gsm8k
    - dataset_path: gsm8k
    - dataset_name: main
    - output_type: generate_until
    - doc_to_text: "Question: {{question}}\\nAnswer:"
    - metric: exact_match with mean aggregation
    - generation_kwargs.until: ["Question:", "</s>", "<|im_end|>"]
    - generation_kwargs.do_sample: false
    - generation_kwargs.temperature: 0.0
    - num_fewshot: 5
    
    Args:
        pipe: HuggingFace text generation pipeline
        tokenizer: Model tokenizer
        model_name_for_logging: Name of model for logging
        device: Device to run on
        dataset_path: HuggingFace dataset path (default: "gsm8k")
        dataset_name: Dataset configuration (default: "main")
        dataset_split: Which split to evaluate on (default: "test")
        num_few_shot: Number of few-shot examples (default: 5)
        few_shot_examples_list: Custom few-shot examples (optional)
        max_new_tokens: Maximum tokens to generate (default: 512)
        generation_batch_size: Batch size for generation
        process_id: Process ID for multi-GPU setup
        gpu_id: GPU ID
        num_gpus: Total number of GPUs
        results_dir: Directory to save results
        
    Returns:
        Dictionary with GSM8K accuracy score
    """

    logger.info(f"Starting GSM8K ({num_few_shot}-shot): {model_name_for_logging} on {dataset_path}/{dataset_name}")

    if few_shot_examples_list is None:
        few_shot_examples_list = FEWSHOT_EXAMPLES_GSM8K_DEFAULT_SET

    # Load dataset
    try:
        full_data = load_dataset(dataset_path, dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load GSM8K dataset: {e}")
        return {"GSM8K": 0.0, "error_message": f"DatasetLoadFailed GSM8K: {e}"}
    
    logger.info(f"P{process_id}: Loaded GSM8K '{dataset_path}/{dataset_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    # Handle multi-GPU distribution
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
        return {"GSM8K": 0.0}
    
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} GSM8K examples.")

    correct_predictions = 0
    total_evaluated = 0
    
    prompts_for_batch, original_items_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - GSM8K Eval")):
        question = item_data.get('question')
        true_answer_text = item_data.get('answer')

        if not question or not true_answer_text:
            logger.warning(f"P{process_id}: Skipping GSM8K item due to missing data. Q: {str(question)[:50]}")
            continue
        
        # Extract ground truth answer using same logic as predictions
        true_final_answer_str = _extract_gsm8k_answer(true_answer_text)
        if true_final_answer_str is None:
            logger.warning(f"P{process_id}: Could not extract ground truth answer for GSM8K item. Q: {str(question)[:50]}, Ans: {str(true_answer_text)[:100]}")
            continue

        prompt_text = _format_gsm8k_prompt(question, few_shot_examples_list, num_few_shot)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append({
            'true_final_answer_str': true_final_answer_str,
            'question': question
        })

        # Process batch when full or at end
        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            # Create stopping criteria matching lm-harness "until" parameter
            # generation_kwargs.until: ["Question:", "</s>", "<|im_end|>"]
            stop_sequences = ["Question:", "</s>", "<|im_end|>"]
            stopping_criteria = StoppingCriteriaList([
                StopOnTokens(stop_sequences=stop_sequences, tokenizer=tokenizer)
            ])
            
            # Generation configuration matching lm-harness
            # do_sample: false, temperature: 0.0
            gen_config = {
                "do_sample": False,
                "temperature": 0.0,  # Explicitly set to 0.0 as per lm-harness
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "stopping_criteria": stopping_criteria,
                "return_full_text": False
            }
            
            try:
                with torch.no_grad(): 
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item_info = original_items_for_batch[k]
                    raw_gen_text = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else ""
                    
                    # Clean up stop sequences from output if present
                    for stop_seq in stop_sequences:
                        if stop_seq in raw_gen_text:
                            raw_gen_text = raw_gen_text.split(stop_seq)[0]
                    
                    # Extract predicted answer using lm-harness filter logic
                    pred_final_answer_str = _extract_gsm8k_answer(raw_gen_text)
                    
                    # Normalize both answers for comparison (ignore_case: true)
                    pred_normalized = _normalize_answer(pred_final_answer_str) if pred_final_answer_str else ""
                    true_normalized = _normalize_answer(original_item_info['true_final_answer_str'])
                    
                    logger.debug(
                        f"GSM8K Q: {original_item_info['question'][:50]}...\n"
                        f"Generated: {raw_gen_text[:150]}...\n"
                        f"Pred: {pred_normalized}, True: {true_normalized}"
                    )

                    # Exact match comparison (metric: exact_match)
                    if pred_normalized and pred_normalized == true_normalized:
                        correct_predictions += 1
                    
                    total_evaluated += 1

            except Exception as e_batch_gsm:
                logger.error(f"P{process_id}: Error in GSM8K gen batch: {e_batch_gsm}", exc_info=True)
                total_evaluated += len(prompts_for_batch)
            
            prompts_for_batch, original_items_for_batch = [], []

    # Compute accuracy (aggregation: mean)
    accuracy_score = (correct_predictions / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final GSM8K Accuracy: {accuracy_score:.2f}% ({correct_predictions}/{total_evaluated}).")
    
    return {"GSM8K": accuracy_score}


if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: 
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test GSM8K (lm-harness compatible)")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="test[:5]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=1)
    test_parser.add_argument("--num_few_shot_test", type=int, default=5)  # Changed to 5 (lm-harness default)
    
    gsm_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="GSM8KFileTest")
    logger.info(f"--- Standalone GSM8K Test (lm-harness compatible): {gsm_args.model_name_test} ({gsm_args.num_few_shot_test}-shot) ---")
    
    gsm_pipe, _ = initialize_model_pipeline(gsm_args.model_name_test, target_device_id=0)
    if gsm_pipe:
        gsm_eval_args = {
            "pipe": gsm_pipe, 
            "tokenizer": gsm_pipe.tokenizer, 
            "model_name_for_logging": gsm_args.model_name_test,
            "device": gsm_pipe.device, 
            "dataset_split": gsm_args.dataset_split_test,
            "num_few_shot": gsm_args.num_few_shot_test,
            "generation_batch_size": gsm_args.gen_batch_size_test,
            "process_id": 0, 
            "gpu_id": 0, 
            "num_gpus": 1
        }
        try: 
            print(json.dumps(evaluate_gsm8k(**gsm_eval_args), indent=2))
        finally: 
            cleanup_model_resources(gsm_pipe, getattr(gsm_pipe, 'model', None))