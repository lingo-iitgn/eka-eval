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
            if generated_text.endswith(stop_seq):
                return True
        return False


def _format_gsm8k_prompt(question: str, fewshot_examples: List[Dict], num_few_shot: int) -> str:
    # logger.debug(f"[PROMPT] Formatting prompt with {num_few_shot} few-shot examples")
    prompt = ""
    if num_few_shot > 0 and fewshot_examples:
        actual_fewshot = fewshot_examples[:num_few_shot]
        for i, ex in enumerate(actual_fewshot):
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def _extract_gsm8k_final_answer_strict(text: Optional[str]) -> Optional[str]:
    if text is None: return None
    match = re.search(r'####\s*(\-?[0-9\.\,]+)', text)
    if match:
        num_str = match.group(1).strip()
        num_str = num_str.replace(',', '').replace('$', '').rstrip('.')
        try:
            return str(float(num_str))
        except ValueError:
            return num_str
    return None


def _extract_gsm8k_final_answer_flexible(text: Optional[str]) -> Optional[str]:
    if text is None: return None
    matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if matches:
        last_match = matches[-1]
        num_str = last_match[0] if last_match[0] else last_match[1]
        num_str = num_str.replace(',', '').replace('$', '').rstrip('.')
        try:
            return str(float(num_str))
        except ValueError:
            return num_str
    return None


def _extract_gsm8k_answer(text: Optional[str]) -> Optional[str]:
    answer = _extract_gsm8k_final_answer_strict(text)
    if answer is None:
        answer = _extract_gsm8k_final_answer_flexible(text)
    return answer


def _normalize_answer(answer: str) -> str:
    if not answer: return ""
    answer = re.sub(r',', '', answer)
    answer = re.sub(r'\$', '', answer)
    answer = re.sub(r'(?s).*####\s*', '', answer)
    answer = answer.rstrip('.').lower().strip()
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

    logger.info(f"[INIT] Model: {model_name_for_logging} | Split: {dataset_split} | GPU: {gpu_id}")

    if few_shot_examples_list is None:
        few_shot_examples_list = FEWSHOT_EXAMPLES_GSM8K_DEFAULT_SET

    try:
        full_data = load_dataset(dataset_path, dataset_config, split=dataset_split)
    except Exception as e:
        logger.error(f"[DATASET] ✗ Failed: {e}")
        return {"GSM8K": 0.0, "error_message": str(e)}

    if num_gpus > 1:
        total = len(full_data)
        per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    
    if len(subset_to_process) == 0: 
        return {"GSM8K": 0.0}

    correct_predictions = 0
    total_evaluated = 0
    prompts_for_batch, original_items_for_batch = [], []
    saved_results_list = []

    logger.info(f"[EVAL] Starting loop on {len(subset_to_process)} items...")
    
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - GSM8K")):
        question = item_data.get('question')
        true_answer_text = item_data.get('answer')

        if not question or not true_answer_text: continue
        
        true_final_answer_str = _extract_gsm8k_answer(true_answer_text)
        if true_final_answer_str is None: continue

        prompt_text = _format_gsm8k_prompt(question, few_shot_examples_list, num_few_shot)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append({
            'true_final_answer_str': true_final_answer_str,
            'question': question,
            'item_idx': item_idx
        })

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            
            # FIX: Separate active stopping tokens from truncation tokens
            # "Question:" is dangerous for StopOnTokens because it exists in the prompt.
            # We remove it from generation stopping, but keep it for truncation.
            active_stop_sequences = ["</s>", "<|im_end|>", "<|endoftext|>"]
            truncation_sequences = ["Question:", "</s>", "<|im_end|>", "<|endoftext|>"]
            
            stopping_criteria = StoppingCriteriaList([
                StopOnTokens(stop_sequences=active_stop_sequences, tokenizer=tokenizer)
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
            
            try:
                with torch.no_grad(): 
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item_info = original_items_for_batch[k]
                    raw_gen_text = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else ""
                    
                    # Truncate at "Question:" or other stop tokens here
                    for stop_seq in truncation_sequences:
                        if stop_seq in raw_gen_text:
                            raw_gen_text = raw_gen_text.split(stop_seq)[0]
                    
                    pred_final_answer_str = _extract_gsm8k_answer(raw_gen_text)
                    pred_normalized = _normalize_answer(pred_final_answer_str) if pred_final_answer_str else ""
                    true_normalized = _normalize_answer(original_item_info['true_final_answer_str'])
                    
                    is_correct = pred_normalized == true_normalized

                    result_entry = {
                        "question": original_item_info['question'],
                        "ground_truth_answer": true_normalized,
                        "model_output": raw_gen_text,
                        "predicted_answer": pred_normalized,
                        "is_correct": is_correct
                    }
                    saved_results_list.append(result_entry)
                    
                    if is_correct: correct_predictions += 1
                    total_evaluated += 1

            except Exception as e_batch_gsm:
                logger.error(f"[BATCH] ✗ Error: {e_batch_gsm}")
                total_evaluated += len(prompts_for_batch)
            
            prompts_for_batch, original_items_for_batch = [], []

    accuracy_score = (correct_predictions / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    
    # Save Results
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        safe_model_name = model_name_for_logging.replace('/', '_').replace('\\', '_')
        output_filename = f"gsm8k_results_{safe_model_name}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(saved_results_list, f, indent=4, ensure_ascii=False)
            
        logger.info(f"[SAVE] ✓ Saved results to: {output_path}")
    except Exception as e:
        logger.error(f"[SAVE] ✗ Failed to save: {e}")

    logger.info(f"[FINAL] Accuracy: {accuracy_score:.2f}%")
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
            "num_gpus": 1,
            "results_dir": "results_output"
        }
        try: 
            result = evaluate_gsm8k(**gsm_eval_args)
            print("\nFINAL RESULT:", json.dumps(result, indent=2))
        finally: 
            cleanup_model_resources(gsm_pipe, getattr(gsm_pipe, 'model', None))
    else:
        logger.error("Failed to initialize model")