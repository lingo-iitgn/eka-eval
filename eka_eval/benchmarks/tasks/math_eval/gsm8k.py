# eka_eval/benchmarks/tasks/math/gsm8k.py
import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import logging
import sys
import argparse
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_GSM8K = "openai/gsm8k"
DEFAULT_CONFIG_GSM8K = "main"
DEFAULT_SPLIT_GSM8K = "test"
DEFAULT_MAX_NEW_TOKENS_GSM8K = 256
DEFAULT_GENERATION_BATCH_SIZE_GSM8K = 8
DEFAULT_NUM_FEWSHOT_GSM8K = 5

FEWSHOT_EXAMPLES_GSM8K_DEFAULT_SET = [
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer_with_steps": "Step 1: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nStep 2: Working 50 minutes, she earned 0.2 x 50 = <<0.2*50=10>>10.\nStep 3: The final answer is #### 10."
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15, and her grandparents gave her twice as much. How much more money does she need?",
        "answer_with_steps": "Step 1: Betty has 100 / 2 = $<<100/2=50>>50.\nStep 2: Grandparents gave her 15 * 2 = $<<15*2=30>>30.\nStep 3: Total = 50 + 15 + 30 = <<50+15+30=95>>95.\nStep 4: 100 - 95 = $<<100-95=5>>5.\nStep 5: The final answer is #### 5."
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday she read 12 pages, and today twice as many. If she wants to read half the remaining pages tomorrow, how many pages is that?",
        "answer_with_steps": "Step 1: Today she read 12 x 2 = <<12*2=24>>24.\nStep 2: Total read so far = 12 + 24 = <<12+24=36>>36.\nStep 3: Remaining = 120 - 36 = <<120-36=84>>84.\nStep 4: Half of 84 = <<84/2=42>>42.\nStep 5: The final answer is #### 42."
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer_with_steps": "Step 1: 3 pages x 2 friends = <<3*2=6>>6 pages per letter session.\nStep 2: Twice a week = 6 x 2 = <<6*2=12>>12 pages per week.\nStep 3: Per year = 12 x 52 = <<12*52=624>>624.\nStep 4: The final answer is #### 624."
    },
    {
        "question": "Mark planted 10 yellow flowers. He planted 80% more purple ones. Then 25% as many green flowers as the total of yellow and purple. How many total flowers?",
        "answer_with_steps": "Step 1: Purple = 10 x 0.8 = <<10*0.8=8>>8.\nStep 2: Total yellow + purple = 10 + 8 = <<10+8=18>>18.\nStep 3: Green = 0.25 x 18 = <<0.25*18=4.5>>4.5\nStep 4: Total = 10 + 8 + 4.5 = <<10+8+4.5=22.5>>22.5.\nStep 5: The final answer is #### 22.5."
    }
]

def _format_gsm8k_prompt(question: str, fewshot_examples: List[Dict], num_few_shot: int) -> str:
    prompt = ""
    if num_few_shot > 0 and fewshot_examples:
        actual_fewshot = fewshot_examples[:num_few_shot]
        for ex in actual_fewshot:
            prompt += f"Q: {ex['question']}\nA: {ex['answer_with_steps']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt

def _extract_gsm8k_final_answer(text: Optional[str]) -> Optional[str]:
    if text is None: return None
    match = re.search(r'####\s*([0-9\-\.,/]+)', text)
    if match:
        num_str = match.group(1).strip().replace(',', '')
        try:
            return str(float(num_str))
        except ValueError:
            logger.warning(f"GSM8K: Could not convert extracted answer '{num_str}' to float.")
            return num_str
    logger.debug(f"GSM8K: Final answer marker '####' not found in text: '{text[-100:]}'")
    return None

def evaluate_gsm8k(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_GSM8K,
    dataset_config_name: str = DEFAULT_CONFIG_GSM8K,
    dataset_split: str = DEFAULT_SPLIT_GSM8K,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_GSM8K,
    few_shot_examples_list: Optional[List[Dict]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_GSM8K,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_GSM8K,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting GSM8K ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}/{dataset_config_name}")

    if few_shot_examples_list is None:
        few_shot_examples_list = FEWSHOT_EXAMPLES_GSM8K_DEFAULT_SET

    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"GSM8K": 0.0, "error_message": f"DatasetLoadFailed GSM8K: {e}"}
    logger.info(f"P{process_id}: Loaded GSM8K '{dataset_name}/{dataset_config_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    if num_gpus > 1:
        total = len(full_data); per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    if len(subset_to_process) == 0: return {"GSM8K": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} GSM8K examples.")

    correct_predictions = 0
    total_evaluated = 0
    
    prompts_for_batch, original_items_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - GSM8K Eval")):
        question = item_data.get('question')
        true_answer_text_with_steps = item_data.get('answer')

        if not question or not true_answer_text_with_steps:
            logger.warning(f"P{process_id}: Skipping GSM8K item due to missing data. Q: {str(question)[:50]}")
            continue
        
        true_final_answer_str = _extract_gsm8k_final_answer(true_answer_text_with_steps)
        if true_final_answer_str is None:
            logger.warning(f"P{process_id}: Could not extract ground truth answer for GSM8K item. Q: {str(question)[:50]}, Ans: {str(true_answer_text_with_steps)[:100]}")
            continue

        prompt_text = _format_gsm8k_prompt(question, few_shot_examples_list, num_few_shot)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append({'true_final_answer_str': true_final_answer_str})

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "stop_sequences": ["\nQ:", "####"],
                "return_full_text": False
            }
            try:
                with torch.no_grad(): batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item_info = original_items_for_batch[k]
                    raw_gen_steps = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else "#GenFail"
                    
                    pred_final_answer_str = _extract_gsm8k_final_answer("A: " + raw_gen_steps)
                    
                    logger.debug(f"GSM8K Q: ...{question[-50:]}\nPred Raw Steps: {raw_gen_steps[:100]}...\nPred Final Ans: {pred_final_answer_str}, True Final Ans: {original_item_info['true_final_answer_str']}")

                    if pred_final_answer_str is not None and pred_final_answer_str == original_item_info['true_final_answer_str']:
                        correct_predictions += 1
                    total_evaluated +=1

            except Exception as e_batch_gsm:
                logger.error(f"P{process_id}: Error in GSM8K gen batch: {e_batch_gsm}", exc_info=True)
                total_evaluated += len(prompts_for_batch)
            prompts_for_batch, original_items_for_batch = [], []

    accuracy_score = (correct_predictions / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final GSM8K Accuracy: {accuracy_score:.2f}% ({correct_predictions}/{total_evaluated}).")
    return {"GSM8K": accuracy_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser = argparse.ArgumentParser(description="Standalone Test GSM8K")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="test[:5]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=1)
    test_parser.add_argument("--num_few_shot_test", type=int, default=2)
    
    gsm_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="GSM8KFileTest")
    logger.info(f"--- Standalone GSM8K Test: {gsm_args.model_name_test} ({gsm_args.num_few_shot_test}-shot) ---")
    
    gsm_pipe, _ = initialize_model_pipeline(gsm_args.model_name_test, target_device_id=0)
    if gsm_pipe:
        gsm_eval_args = {
            "pipe": gsm_pipe, "tokenizer": gsm_pipe.tokenizer, "model_name_for_logging": gsm_args.model_name_test,
            "device": gsm_pipe.device, "dataset_split": gsm_args.dataset_split_test,
            "num_few_shot": gsm_args.num_few_shot_test,
            "generation_batch_size": gsm_args.gen_batch_size_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_gsm8k(**gsm_eval_args), indent=2))
        finally: cleanup_model_resources(gsm_pipe, getattr(gsm_pipe, 'model', None))
