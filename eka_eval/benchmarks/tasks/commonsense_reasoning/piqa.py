import torch
import re
import sys
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import hashlib
import logging
from typing import Dict, List, Any
import evaluate as hf_evaluate
import gc

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_PIQA = "piqa"
DEFAULT_SPLIT_PIQA = "validation"
DEFAULT_MAX_NEW_TOKENS_PIQA = 5
DEFAULT_CHECKPOINT_DIR_PIQA = "checkpoints/piqa_checkpoints"

try:
    piqa_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for PIQA loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for PIQA: {e}. PIQA will not run correctly.", exc_info=True)
    piqa_accuracy_metric = None


def _format_piqa_prompt(item: Dict) -> str:
    goal = item.get('goal', '')
    sol1 = item.get('sol1', '')
    sol2 = item.get('sol2', '')
    prompt = (
        "Select the most physically appropriate solution (0 or 1) to achieve the goal.\n\n"
        f"Goal: {goal}\n"
        f"0) {sol1}\n"
        f"1) {sol2}\n\n"
        "Your answer must be exactly 0 or 1.\nAnswer:"
    )
    return prompt


def _extract_piqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    match = re.search(r'^\s*\b(0|1)\b', completion_part)
    if match:
        return match.group(1)
    logger.debug(f"PIQA: Could not extract 0 or 1 from start of completion: '{completion_part[:20]}'")
    return "X"


def _safe_generate_piqa(
    pipe: Any,
    prompts: List[str],
    tokenizer: Any,
    max_new_tokens: int,
    num_retries: int = 2
) -> List[Dict[str, str]]:
    generation_params = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": True
    }
    all_outputs = []
    for attempt in range(num_retries):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(prompts) != 1:
                logger.warning(f"PIQA _safe_generate expected 1 prompt, got {len(prompts)}. Processing first only for now.")
            prompt_to_gen = prompts[0] if prompts else ""
            if not prompt_to_gen:
                all_outputs.append({"generated_text": ""})
                continue
            output_list_of_dicts = pipe(prompt_to_gen, **generation_params)
            if output_list_of_dicts and output_list_of_dicts[0].get('generated_text'):
                all_outputs.append(output_list_of_dicts[0])
            else:
                all_outputs.append({"generated_text": prompt_to_gen + "X"})
            return all_outputs
        except Exception as e:
            logger.error(f"PIQA Generation attempt {attempt + 1} failed: {e}", exc_info=True)
            if attempt < num_retries - 1:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("PIQA: Retrying generation...")
            else:
                logger.error("PIQA: Max retries reached for generation.")
                return [{"generated_text": p + "X_ERROR"} for p in prompts]
    return [{"generated_text": p + "X_ALL_RETRIES_FAILED"} for p in prompts]


def evaluate_piqa(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_PIQA,
    dataset_split: str = DEFAULT_SPLIT_PIQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_PIQA,
    generation_batch_size: int = 1,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    if piqa_accuracy_metric is None:
        return {"PIQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting PIQA evaluation for model: {model_name_for_logging} on dataset: {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Params: split='{dataset_split}', gen_batch_size={generation_batch_size}")

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"PIQA": 0.0, "error_message": f"DatasetLoadFailed PIQA: {e}"}

    logger.info(f"P{process_id}: Loaded PIQA '{dataset_name}' (split: '{dataset_split}') with {len(full_data_for_split)} examples.")

    if num_gpus > 1:
        total_examples = len(full_data_for_split)
        examples_per_instance = total_examples // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1:
            end_idx = total_examples
        dataset_subset_to_process = full_data_for_split.select(range(start_idx, end_idx))
    else:
        dataset_subset_to_process = full_data_for_split

    if len(dataset_subset_to_process) == 0:
        return {"PIQA": 0.0}

    predictions_numeric, true_labels_numeric = [], []
    prompts_for_batch, infos_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(dataset_subset_to_process, desc=f"P{process_id} - PIQA Eval")):
        prompt_text = _format_piqa_prompt(item_data)
        prompts_for_batch.append(prompt_text)
        infos_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(dataset_subset_to_process) - 1:
            generation_config_piqa = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **generation_config_piqa)
                for k, raw_output_list in enumerate(batch_raw_outputs):
                    original_item = infos_for_batch[k]
                    raw_generated_text = raw_output_list[0]['generated_text'] if raw_output_list and raw_output_list[0] else prompts_for_batch[k] + "X"
                    predicted_answer_str = _extract_piqa_answer(raw_generated_text, prompts_for_batch[k])
                    pred_numeric = int(predicted_answer_str) if predicted_answer_str in ["0", "1"] else -1
                    true_numeric = int(original_item['label'])
                    if pred_numeric == -1:
                        pred_numeric = 1 - true_numeric
                    predictions_numeric.append(pred_numeric)
                    true_labels_numeric.append(true_numeric)
            except Exception as e_batch_piqa:
                logger.error(f"P{process_id}: Error in PIQA generation batch: {e_batch_piqa}", exc_info=True)
                for _ in range(len(prompts_for_batch)):
                    predictions_numeric.append(0)
                    true_labels_numeric.append(1)
            prompts_for_batch, infos_for_batch = [], []

    if not true_labels_numeric:
        return {"PIQA": 0.0}

    accuracy_score = 0.0
    try:
        accuracy_results = piqa_accuracy_metric.compute(predictions=predictions_numeric, references=true_labels_numeric)
        accuracy_score = accuracy_results.get("accuracy", 0.0) * 100
    except Exception as e_metric:
        logger.error(f"P{process_id}: Error computing PIQA accuracy: {e_metric}", exc_info=True)

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final PIQA Accuracy: {accuracy_score:.2f}% on {len(true_labels_numeric)} examples.")
    return {"PIQA": accuracy_score}


if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser_piqa = argparse.ArgumentParser(description="Standalone Test PIQA")
    test_parser_piqa.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser_piqa.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser_piqa.add_argument("--gen_batch_size_test", type=int, default=2)

    pi_args = test_parser_piqa.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="PIQAFileTest")
    logger.info(f"--- Standalone PIQA Test: {pi_args.model_name_test} ---")
    pi_pipe, _ = initialize_model_pipeline(pi_args.model_name_test, target_device_id=0)
    if pi_pipe:
        pi_eval_args = {
            "pipe": pi_pipe,
            "tokenizer": pi_pipe.tokenizer,
            "model_name_for_logging": pi_args.model_name_test,
            "device": pi_pipe.device,
            "dataset_split": pi_args.dataset_split_test,
            "generation_batch_size": pi_args.gen_batch_size_test,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1
        }
        try:
            print(json.dumps(evaluate_piqa(**pi_eval_args), indent=2))
        finally:
            cleanup_model_resources(pi_pipe, getattr(pi_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {pi_args.model_name_test} for PIQA test.")
