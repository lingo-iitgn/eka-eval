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

DEFAULT_DATASET_PATH_GSM8K = "openai/gsm8k"
DEFAULT_CONFIG_GSM8K = "main"
DEFAULT_SPLIT_GSM8K = "test"
DEFAULT_MAX_NEW_TOKENS_GSM8K = 512
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
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                return True
        return False


def _format_gsm8k_prompt(question: str, fewshot_examples: List[Dict], num_few_shot: int) -> str:
    logger.debug(f"[PROMPT] Formatting prompt with {num_few_shot} few-shot examples")
    prompt = ""
    if num_few_shot > 0 and fewshot_examples:
        actual_fewshot = fewshot_examples[:num_few_shot]
        for i, ex in enumerate(actual_fewshot):
            logger.debug(f"[PROMPT] Adding few-shot example {i+1}/{num_few_shot}")
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    logger.debug(f"[PROMPT] Final prompt length: {len(prompt)} chars")
    return prompt


def _extract_gsm8k_final_answer_strict(text: Optional[str]) -> Optional[str]:
    if text is None: 
        logger.debug("[EXTRACT] Text is None, returning None")
        return None
    
    logger.debug(f"[EXTRACT] Attempting strict match on text: '{text[:100]}...'")
    match = re.search(r'####\s*(\-?[0-9\.\,]+)', text)
    
    if match:
        num_str = match.group(1).strip()
        logger.debug(f"[EXTRACT] Strict match found: '{num_str}'")
        num_str = num_str.replace(',', '').replace('$', '').rstrip('.')
        logger.debug(f"[EXTRACT] After cleanup: '{num_str}'")
        try:
            result = str(float(num_str))
            logger.debug(f"[EXTRACT] Converted to float: '{result}'")
            return result
        except ValueError:
            logger.warning(f"[EXTRACT] Could not convert '{num_str}' to float")
            return num_str
    
    logger.debug("[EXTRACT] Strict match failed - no #### pattern found")
    return None


def _extract_gsm8k_final_answer_flexible(text: Optional[str]) -> Optional[str]:
    if text is None:
        logger.debug("[EXTRACT-FLEX] Text is None, returning None")
        return None
    
    logger.debug(f"[EXTRACT-FLEX] Attempting flexible extraction on: '{text[:100]}...'")
    matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    logger.debug(f"[EXTRACT-FLEX] Found {len(matches)} potential numbers")
    
    if matches:
        last_match = matches[-1]
        num_str = last_match[0] if last_match[0] else last_match[1]
        logger.debug(f"[EXTRACT-FLEX] Taking last match: '{num_str}'")
        num_str = num_str.replace(',', '').replace('$', '').rstrip('.')
        logger.debug(f"[EXTRACT-FLEX] After cleanup: '{num_str}'")
        try:
            result = str(float(num_str))
            logger.debug(f"[EXTRACT-FLEX] Converted to float: '{result}'")
            return result
        except ValueError:
            logger.warning(f"[EXTRACT-FLEX] Could not convert '{num_str}' to float")
            return num_str
    
    logger.debug("[EXTRACT-FLEX] No numbers found")
    return None


def _extract_gsm8k_answer(text: Optional[str]) -> Optional[str]:
    logger.debug("[EXTRACT] Starting answer extraction")
    answer = _extract_gsm8k_final_answer_strict(text)
    if answer is None:
        logger.debug("[EXTRACT] Falling back to flexible extraction")
        answer = _extract_gsm8k_final_answer_flexible(text)
    logger.debug(f"[EXTRACT] Final extracted answer: '{answer}'")
    return answer


def _normalize_answer(answer: str) -> str:
    if not answer:
        logger.debug("[NORMALIZE] Empty answer, returning empty string")
        return ""
    
    logger.debug(f"[NORMALIZE] Input: '{answer}'")
    answer = re.sub(r',', '', answer)
    answer = re.sub(r'\$', '', answer)
    answer = re.sub(r'(?s).*####\s*', '', answer)
    answer = answer.rstrip('.').lower().strip()
    logger.debug(f"[NORMALIZE] Output: '{answer}'")
    return answer


def evaluate_gsm8k(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_path: str = DEFAULT_DATASET_PATH_GSM8K,
    dataset_config: str = DEFAULT_CONFIG_GSM8K,
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
    
    Args:
        pipe: HuggingFace pipeline
        tokenizer: Tokenizer
        model_name_for_logging: Model name
        device: Device
        dataset_path: HF dataset path (default: "openai/gsm8k")
        dataset_config: Dataset config (default: "main")
        dataset_split: Split to evaluate (default: "test")
        num_few_shot: Number of few-shot examples (default: 5)
        few_shot_examples_list: Custom few-shot examples
        max_new_tokens: Max tokens to generate (default: 512)
        generation_batch_size: Batch size (default: 8)
        process_id: Process ID
        gpu_id: GPU ID
        num_gpus: Total GPUs
        results_dir: Results directory
    
    Returns:
        Dict with GSM8K accuracy
    """

    logger.info(f"[INIT] ========================================")
    logger.info(f"[INIT] Starting GSM8K Evaluation")
    logger.info(f"[INIT] ========================================")
    logger.info(f"[INIT] Model: {model_name_for_logging}")
    logger.info(f"[INIT] Dataset: {dataset_path}")
    logger.info(f"[INIT] Config: {dataset_config}")
    logger.info(f"[INIT] Split: {dataset_split}")
    logger.info(f"[INIT] Few-shot: {num_few_shot}")
    logger.info(f"[INIT] Max tokens: {max_new_tokens}")
    logger.info(f"[INIT] Batch size: {generation_batch_size}")
    logger.info(f"[INIT] Process ID: {process_id}, GPU: {gpu_id}")
    logger.info(f"[INIT] ========================================")

    if few_shot_examples_list is None:
        few_shot_examples_list = FEWSHOT_EXAMPLES_GSM8K_DEFAULT_SET
        logger.info(f"[INIT] Using default few-shot examples ({len(few_shot_examples_list)} examples)")

    try:
        logger.info(f"[DATASET] Loading dataset from '{dataset_path}' with config '{dataset_config}'...")
        full_data = load_dataset(dataset_path, dataset_config, split=dataset_split)
        logger.info(f"[DATASET] ✓ Successfully loaded {len(full_data)} examples from split '{dataset_split}'")
    except Exception as e:
        logger.error(f"[DATASET] ✗ Failed to load dataset: {e}", exc_info=True)
        return {"GSM8K": 0.0, "error_message": f"DatasetLoadFailed: {e}"}

    if num_gpus > 1:
        total = len(full_data)
        per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: 
            end = total
        subset_to_process = full_data.select(range(start, end))
        logger.info(f"[SHARD] Multi-GPU mode: GPU {gpu_id} processing examples {start}-{end} ({len(subset_to_process)} examples)")
    else:
        subset_to_process = full_data
        logger.info(f"[SHARD] Single-GPU mode: processing all {len(subset_to_process)} examples")
    
    if len(subset_to_process) == 0: 
        logger.warning("[SHARD] ✗ No examples to process after sharding")
        return {"GSM8K": 0.0}

    correct_predictions = 0
    total_evaluated = 0
    prompts_for_batch, original_items_for_batch = [], []

    logger.info(f"[EVAL] ========================================")
    logger.info(f"[EVAL] Starting evaluation loop...")
    logger.info(f"[EVAL] ========================================")
    
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - GSM8K Eval")):
        question = item_data.get('question')
        true_answer_text = item_data.get('answer')

        if not question or not true_answer_text:
            logger.warning(f"[ITEM-{item_idx}] ✗ Skipping - missing question or answer")
            continue
        
        logger.debug(f"[ITEM-{item_idx}] Question: {question[:80]}...")
        
        true_final_answer_str = _extract_gsm8k_answer(true_answer_text)
        if true_final_answer_str is None:
            logger.warning(f"[ITEM-{item_idx}] ✗ Could not extract ground truth answer from: {true_answer_text[:100]}")
            continue

        logger.debug(f"[ITEM-{item_idx}] Ground truth: '{true_final_answer_str}'")

        prompt_text = _format_gsm8k_prompt(question, few_shot_examples_list, num_few_shot)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append({
            'true_final_answer_str': true_final_answer_str,
            'question': question,
            'item_idx': item_idx
        })

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            logger.info(f"[BATCH] ========================================")
            logger.info(f"[BATCH] Processing batch of {len(prompts_for_batch)} prompts")
            logger.info(f"[BATCH] ========================================")
            
            stop_sequences = ["Question:", "</s>", "<|im_end|>"]
            stopping_criteria = StoppingCriteriaList([
                StopOnTokens(stop_sequences=stop_sequences, tokenizer=tokenizer)
            ])
            
            gen_config = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "stopping_criteria": stopping_criteria,
                "return_full_text": False
            }
            
            logger.debug(f"[BATCH] Generation config: do_sample=False, temp=0.0, max_tokens={max_new_tokens}")
            logger.info(f"[BATCH] Starting generation...")
            
            try:
                with torch.no_grad(): 
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                logger.info(f"[BATCH] ✓ Generation complete")
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item_info = original_items_for_batch[k]
                    item_idx_actual = original_item_info['item_idx']
                    raw_gen_text = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else ""
                    
                    logger.debug(f"[ITEM-{item_idx_actual}] Raw generation ({len(raw_gen_text)} chars): '{raw_gen_text[:150]}...'")
                    
                    for stop_seq in stop_sequences:
                        if stop_seq in raw_gen_text:
                            raw_gen_text = raw_gen_text.split(stop_seq)[0]
                            logger.debug(f"[ITEM-{item_idx_actual}] Truncated at stop sequence: '{stop_seq}'")
                    
                    pred_final_answer_str = _extract_gsm8k_answer(raw_gen_text)
                    pred_normalized = _normalize_answer(pred_final_answer_str) if pred_final_answer_str else ""
                    true_normalized = _normalize_answer(original_item_info['true_final_answer_str'])
                    
                    is_correct = pred_normalized and pred_normalized == true_normalized
                    
                    logger.info(f"[ITEM-{item_idx_actual}] {'='*60}")
                    logger.info(f"[ITEM-{item_idx_actual}] Q: {original_item_info['question'][:70]}...")
                    logger.info(f"[ITEM-{item_idx_actual}] Generated: {raw_gen_text[:100]}...")
                    logger.info(f"[ITEM-{item_idx_actual}] Predicted: '{pred_normalized}'")
                    logger.info(f"[ITEM-{item_idx_actual}] Expected:  '{true_normalized}'")
                    logger.info(f"[ITEM-{item_idx_actual}] Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                    logger.info(f"[ITEM-{item_idx_actual}] {'='*60}")

                    if is_correct:
                        correct_predictions += 1
                    total_evaluated += 1

            except Exception as e_batch_gsm:
                logger.error(f"[BATCH] ✗ Error during generation: {e_batch_gsm}", exc_info=True)
                total_evaluated += len(prompts_for_batch)
            
            prompts_for_batch, original_items_for_batch = [], []
            
            current_accuracy = (correct_predictions / total_evaluated * 100) if total_evaluated > 0 else 0
            logger.info(f"[PROGRESS] Evaluated: {total_evaluated} | Correct: {correct_predictions} | Accuracy: {current_accuracy:.2f}%")
            logger.info(f"[PROGRESS] ========================================")

    accuracy_score = (correct_predictions / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    
    logger.info(f"")
    logger.info(f"[FINAL] ========================================")
    logger.info(f"[FINAL] GSM8K EVALUATION COMPLETE")
    logger.info(f"[FINAL] ========================================")
    logger.info(f"[FINAL] Total Evaluated: {total_evaluated}")
    logger.info(f"[FINAL] Correct: {correct_predictions}")
    logger.info(f"[FINAL] Wrong: {total_evaluated - correct_predictions}")
    logger.info(f"[FINAL] Accuracy: {accuracy_score:.2f}%")
    logger.info(f"[FINAL] ========================================")
    
    return {"GSM8K": accuracy_score}


if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: 
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test GSM8K")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="test[:5]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=1)
    test_parser.add_argument("--num_few_shot_test", type=int, default=5)
    
    gsm_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="GSM8KTest")
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"STANDALONE GSM8K TEST")
    logger.info(f"Model: {gsm_args.model_name_test}")
    logger.info(f"Split: {gsm_args.dataset_split_test}")
    logger.info(f"Few-shot: {gsm_args.num_few_shot_test}")
    logger.info("="*70)
    logger.info("")
    
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
            result = evaluate_gsm8k(**gsm_eval_args)
            print("\n" + "="*70)
            print("FINAL RESULT:")
            print(json.dumps(result, indent=2))
            print("="*70)
        finally: 
            cleanup_model_resources(gsm_pipe, getattr(gsm_pipe, 'model', None))
    else:
        logger.error("Failed to initialize model")