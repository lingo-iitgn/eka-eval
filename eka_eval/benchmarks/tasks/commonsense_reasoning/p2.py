# eka_eval/benchmarks/tasks/reasoning/piqa.py

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
from typing import Dict, List, Any, Tuple
import evaluate as hf_evaluate
import torch.nn.functional as F
import gc

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_PIQA = "piqa"
DEFAULT_SPLIT_PIQA = "validation"
DEFAULT_MAX_NEW_TOKENS_PIQA = 10
DEFAULT_FEW_SHOT_COUNT_PIQA = 5

try:
    piqa_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for PIQA loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for PIQA: {e}. PIQA will not run correctly.", exc_info=True)
    piqa_accuracy_metric = None

# Balanced few-shot examples for PIQA (answers distributed between 0 and 1)
DEFAULT_FEW_SHOT_EXAMPLES_PIQA = [
    {
        "goal": "To remove a stain from clothing",
        "sol1": "Apply cold water immediately to the stain and gently blot with a clean cloth.",
        "sol2": "Set the clothing on fire to burn away the stain completely.",
        "label": 0  # Answer: sol1
    },
    {
        "goal": "To keep food fresh in the refrigerator",
        "sol1": "Leave all food uncovered and exposed to air.",
        "sol2": "Store food in airtight containers or wrap it properly.",
        "label": 1  # Answer: sol2
    },
    {
        "goal": "To light a candle safely",
        "sol1": "Pour gasoline on the wick and use a blowtorch.",
        "sol2": "Use a match or lighter to ignite the wick carefully.",
        "label": 1  # Answer: sol2
    },
    {
        "goal": "To water plants effectively",
        "sol1": "Water the soil around the roots, not the leaves directly.",
        "sol2": "Pour boiling water directly onto the plant leaves.",
        "label": 0  # Answer: sol1
    },
    {
        "goal": "To clean windows without streaks",
        "sol1": "Use newspaper or a lint-free cloth with glass cleaner.",
        "sol2": "Rub the windows with sandpaper to remove all dirt.",
        "label": 0  # Answer: sol1
    }
]

def doc_to_choice_piqa(doc):
    """Create choice completions following official format"""
    return [doc["sol1"], doc["sol2"]]

def _format_piqa_prompt_official(item: Dict, few_shot_examples: List[Dict]) -> Tuple[str, str]:
    """
    Format PIQA prompt following official format: "Question: {{goal}}\nAnswer:"
    Returns tuple of (choice1_completion, choice2_completion) for likelihood comparison.
    """
    goal = item.get('goal', '').strip()
    
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for ex_item in few_shot_examples:
            ex_goal = ex_item.get('goal', '').strip()
            ex_choices = doc_to_choice_piqa(ex_item)
            ex_label = ex_item.get('label', 0)
            correct_solution = ex_choices[ex_label]
            
            prompt += f"Question: {ex_goal}\nAnswer: {correct_solution}\n\n"
    
    # Create the two possible completions for the target item
    choices = doc_to_choice_piqa(item)
    choice1_completion = prompt + f"Question: {goal}\nAnswer: {choices[0]}"
    choice2_completion = prompt + f"Question: {goal}\nAnswer: {choices[1]}"
    
    return choice1_completion, choice2_completion

def _format_piqa_prompt_generation(item: Dict, few_shot_examples: List[Dict]) -> str:
    """Alternative: Format as generation task with few-shot examples."""
    goal = item.get('goal', '').strip()
    sol1 = item.get('sol1', '').strip()
    sol2 = item.get('sol2', '').strip()
    
    prompt = ""
    
    # Add few-shot examples
    if few_shot_examples:
        prompt += "Choose the most appropriate solution (0 or 1) to achieve the goal:\n\n"
        for ex_item in few_shot_examples:
            ex_goal = ex_item.get('goal', '').strip()
            ex_sol1 = ex_item.get('sol1', '').strip()
            ex_sol2 = ex_item.get('sol2', '').strip()
            ex_label = str(ex_item.get('label', 0))
            
            prompt += f"Question: {ex_goal}\n"
            prompt += f"0) {ex_sol1}\n"
            prompt += f"1) {ex_sol2}\n"
            prompt += f"Answer: {ex_label}\n\n"
    
    prompt += f"Question: {goal}\n"
    prompt += f"0) {sol1}\n"
    prompt += f"1) {sol2}\n"
    prompt += "Answer:"
    
    return prompt

def _extract_piqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    """Extract answer from generated text, looking for 0 or 1"""
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    
    # Look for 0 or 1 at the beginning
    match = re.search(r'^\s*\b(0|1)\b', completion_part)
    if match:
        return match.group(1)
    
    logger.debug(f"PIQA: Could not extract 0 or 1 from completion: '{completion_part[:20]}'")
    return "X"

def _compute_likelihood_score(pipe, tokenizer, text: str) -> float:
    """Compute the likelihood score for a given text completion."""
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(pipe.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = pipe.model(**inputs)
            logits = outputs.logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather the log probabilities of the actual tokens
            gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Sum log probabilities (log of product = sum of logs)
            total_log_prob = gathered_log_probs.sum().item()
            
            return total_log_prob
            
    except Exception as e:
        logger.error(f"Error computing likelihood: {e}")
        return float('-inf')

def evaluate_piqa(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_PIQA,
    dataset_split: str = DEFAULT_SPLIT_PIQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_PIQA,
    generation_batch_size: int = 8,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_PIQA,
    evaluation_method: str = "likelihood",  # "likelihood" or "generation"
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", 
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:

    if piqa_accuracy_metric is None:
        return {"PIQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting PIQA ({num_few_shot}-shot, {evaluation_method}): {model_name_for_logging} on {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): split='{dataset_split}', batch_size={generation_batch_size}")

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"PIQA": 0.0, "error_message": f"DatasetLoadFailed PIQA: {e}"}
    logger.info(f"P{process_id}: Loaded PIQA '{dataset_name}' ({len(full_data_for_split)} examples) for split '{dataset_split}'.")

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

    # Prepare few-shot examples
    few_shot_examples_list = DEFAULT_FEW_SHOT_EXAMPLES_PIQA[:num_few_shot] if num_few_shot > 0 else []

    predictions_numeric, true_labels_numeric = [], []
    outputs_dump = []

    if evaluation_method == "likelihood":
        # Likelihood-based evaluation (more accurate)
        for item_idx, item_data in enumerate(tqdm(dataset_subset_to_process, desc=f"P{process_id} - PIQA Likelihood Eval")):
            true_label = item_data.get('label', -1)
            if true_label not in [0, 1]:
                logger.warning(f"P{process_id}: Skipping invalid label '{true_label}'")
                continue

            try:
                # Get the two possible completions
                choice1_text, choice2_text = _format_piqa_prompt_official(item_data, few_shot_examples_list)
                
                # Compute likelihood scores
                score1 = _compute_likelihood_score(pipe, tokenizer, choice1_text)
                score2 = _compute_likelihood_score(pipe, tokenizer, choice2_text)
                
                # Choose the option with higher likelihood
                predicted_choice = 0 if score1 > score2 else 1
                true_choice = int(true_label)
                
                predictions_numeric.append(predicted_choice)
                true_labels_numeric.append(true_choice)
                
                if save_outputs:
                    outputs_dump.append({
                        "goal": item_data.get('goal', ''),
                        "sol1": item_data.get('sol1', ''),
                        "sol2": item_data.get('sol2', ''),
                        "correct_answer": true_choice,
                        "predicted_answer": predicted_choice,
                        "is_correct": predicted_choice == true_choice,
                        "choice1_completion": choice1_text,
                        "choice2_completion": choice2_text,
                        "choice1_score": score1,
                        "choice2_score": score2,
                        "evaluation_method": "likelihood"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing item {item_idx}: {e}")
                # Default to wrong prediction
                true_choice = int(item_data.get('label', 0))
                wrong_choice = 1 - true_choice
                predictions_numeric.append(wrong_choice)
                true_labels_numeric.append(true_choice)
                
    else:
        # Generation-based evaluation (fallback)
        prompts_for_batch, infos_for_batch = [], []
        
        for item_idx, item_data in enumerate(tqdm(dataset_subset_to_process, desc=f"P{process_id} - PIQA Generation Eval")):
            true_label = item_data.get('label', -1)
            if true_label not in [0, 1]:
                logger.warning(f"P{process_id}: Skipping invalid label '{true_label}'")
                continue

            prompt_text = _format_piqa_prompt_generation(item_data, few_shot_examples_list)
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
                        
                    for k, raw_output in enumerate(batch_raw_outputs):
                        original_item = infos_for_batch[k]
                        prompt = prompts_for_batch[k]
                        raw_generated_text = raw_output[0]['generated_text'] if isinstance(raw_output, list) else raw_output.get('generated_text', prompt + "X")
                        predicted_answer_str = _extract_piqa_answer(raw_generated_text, prompt)
                        pred_numeric = int(predicted_answer_str) if predicted_answer_str in ["0", "1"] else -1
                        true_numeric = int(original_item['label'])
                        
                        if pred_numeric == -1:
                            pred_numeric = 1 - true_numeric  # Choose opposite as wrong
                        
                        predictions_numeric.append(pred_numeric)
                        true_labels_numeric.append(true_numeric)
                        
                        if save_outputs:
                            outputs_dump.append({
                                "goal": original_item.get('goal', ''),
                                "sol1": original_item.get('sol1', ''),
                                "sol2": original_item.get('sol2', ''),
                                "correct_answer": true_numeric,
                                "predicted_answer": pred_numeric,
                                "is_correct": pred_numeric == true_numeric,
                                "prompt": prompt,
                                "raw_response": raw_generated_text,
                                "extracted_completion": raw_generated_text[len(prompt):].strip() if raw_generated_text.startswith(prompt) else raw_generated_text.strip(),
                                "evaluation_method": "generation"
                            })
                            
                except Exception as e_batch_piqa:
                    logger.error(f"P{process_id}: Error in PIQA gen batch: {e_batch_piqa}", exc_info=True)
                    for item_err in infos_for_batch:
                        true_choice = int(item_err.get('label', 0))
                        wrong_choice = 1 - true_choice
                        predictions_numeric.append(wrong_choice)
                        true_labels_numeric.append(true_choice)
                        
                prompts_for_batch, infos_for_batch = [], []

    if not true_labels_numeric:
        return {"PIQA": 0.0}

    accuracy_score = 0.0
    try:
        accuracy_results = piqa_accuracy_metric.compute(predictions=predictions_numeric, references=true_labels_numeric)
        accuracy_score = accuracy_results.get("accuracy", 0.0) * 100
    except Exception as e_metric:
        logger.error(f"P{process_id}: Error computing PIQA accuracy: {e_metric}", exc_info=True)

    # Save outputs to JSON file if requested
    if save_outputs and outputs_dump:
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"piqa_outputs_{model_name_for_logging.replace('/', '_')}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        summary_data = {
            "model_name": model_name_for_logging,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "num_few_shot": num_few_shot,
            "evaluation_method": evaluation_method,
            "total_examples": len(outputs_dump),
            "accuracy": accuracy_score,
            "correct_predictions": sum(1 for item in outputs_dump if item["is_correct"]),
            "process_id": process_id,
            "gpu_id": gpu_id,
            "examples": outputs_dump
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"P{process_id}: Saved {len(outputs_dump)} PIQA outputs to {output_path}")
        except Exception as e_save:
            logger.error(f"P{process_id}: Error saving PIQA outputs: {e_save}")

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final PIQA Accuracy: {accuracy_score:.2f}% on {len(true_labels_numeric)} examples.")
    return {"PIQA": accuracy_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser_piqa = argparse.ArgumentParser(description="Standalone Test PIQA")
    test_parser_piqa.add_argument("--model_name_test", type=str, default="google/gemma-2-2b")
    test_parser_piqa.add_argument("--dataset_split_test", type=str, default="validation")
    test_parser_piqa.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser_piqa.add_argument("--num_few_shot_test", type=int, default=3)
    test_parser_piqa.add_argument("--evaluation_method", type=str, default="likelihood", choices=["likelihood", "generation"])
    test_parser_piqa.add_argument("--save_outputs", action="store_true", help="Save detailed outputs to JSON file")

    pi_args = test_parser_piqa.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="PIQAFileTest")
    logger.info(f"--- Standalone PIQA Test: {pi_args.model_name_test} ({pi_args.evaluation_method}) ---")
    
    pi_pipe, _ = initialize_model_pipeline(pi_args.model_name_test, target_device_id=0)
    if pi_pipe:
        pi_eval_args = {
            "pipe": pi_pipe,
            "tokenizer": pi_pipe.tokenizer,
            "model_name_for_logging": pi_args.model_name_test,
            "device": pi_pipe.device,
            "dataset_split": pi_args.dataset_split_test,
            "generation_batch_size": pi_args.gen_batch_size_test,
            "num_few_shot": pi_args.num_few_shot_test,
            "evaluation_method": pi_args.evaluation_method,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": pi_args.save_outputs
        }
        try:
            print(json.dumps(evaluate_piqa(**pi_eval_args), indent=2))
        finally:
            cleanup_model_resources(pi_pipe, getattr(pi_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {pi_args.model_name_test} for PIQA test.")