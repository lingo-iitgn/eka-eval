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
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate
import torch.nn.functional as F
import gc
from eka_eval.utils.prompt_utils import get_prompt_template, format_prompt, format_few_shot_prompt, get_prompt_data

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_SIQA = "allenai/social_i_qa"
DEFAULT_SPLIT_SIQA = "validation"
DEFAULT_MAX_NEW_TOKENS_SIQA = 5
DEFAULT_GENERATION_BATCH_SIZE_SIQA = 8
DEFAULT_FEW_SHOT_COUNT_SIQA = 5
DEFAULT_PROMPT_TEMPLATE_KEY_LIKELIHOOD = "siqa_likelihood"
DEFAULT_PROMPT_TEMPLATE_KEY_GENERATION = "siqa_generation"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_LIKELIHOOD = "siqa_5shot_likelihood"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_GENERATION = "siqa_5shot_generation"
PROMPT_FILE_BENCHMARK_KEY = "siqa"
PROMPT_FILE_CATEGORY = "reasoning"

try:
    siqa_accuracy_metric = hf_evaluate.load("accuracy")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for SIQA: {e}")
    siqa_accuracy_metric = None

def _get_siqa_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    if num_few_shot <= 0:
        return []
    loaded_examples_list = get_prompt_data(PROMPT_FILE_BENCHMARK_KEY, data_key="default_few_shot_examples_siqa", specific_task_group=prompt_file_category)
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        return loaded_examples_list[:num_few_shot]
    return []

def doc_to_choice_siqa(doc):
    return [doc["answerA"], doc["answerB"], doc["answerC"]]

def _format_siqa_prompt_likelihood_with_template(
    item: Dict, 
    num_few_shot: int = 0,
    prompt_file_category: str = PROMPT_FILE_CATEGORY
) -> Tuple[str, str, str]:
    context = item.get('context', '').strip()
    question = item.get('question', '').strip()
    choices = doc_to_choice_siqa(item)
    
    if num_few_shot > 0:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_LIKELIHOOD
        few_shot_examples = _get_siqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        
        try:
            choice1_completion = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=few_shot_examples,
                context=context,
                question=question,
                answer_text=choices[0],
                specific_task_group=prompt_file_category
            )
            choice2_completion = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=few_shot_examples,
                context=context,
                question=question,
                answer_text=choices[1],
                specific_task_group=prompt_file_category
            )
            choice3_completion = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=few_shot_examples,
                context=context,
                question=question,
                answer_text=choices[2],
                specific_task_group=prompt_file_category
            )
            return choice1_completion, choice2_completion, choice3_completion
        except Exception as e:
            error_text = f"Error: Prompt formatting failed - {str(e)}"
            return error_text, error_text, error_text
    else:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_LIKELIHOOD
        try:
            choice1_completion = format_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                context=context,
                question=question,
                answer_text=choices[0],
                specific_task_group=prompt_file_category
            )
            choice2_completion = format_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                context=context,
                question=question,
                answer_text=choices[1],
                specific_task_group=prompt_file_category
            )
            choice3_completion = format_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                context=context,
                question=question,
                answer_text=choices[2],
                specific_task_group=prompt_file_category
            )
            return choice1_completion, choice2_completion, choice3_completion
        except Exception as e:
            error_text = f"Error: Prompt formatting failed - {str(e)}"
            return error_text, error_text, error_text

def _format_siqa_prompt_generation_with_template(
    item: Dict, 
    num_few_shot: int = 0,
    prompt_file_category: str = PROMPT_FILE_CATEGORY
) -> str:
    context = item.get('context', '').strip()
    question = item.get('question', '').strip()
    ans_a = item.get('answerA', '').strip()
    ans_b = item.get('answerB', '').strip()
    ans_c = item.get('answerC', '').strip()
    
    if num_few_shot > 0:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_GENERATION
        few_shot_examples = _get_siqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        
        try:
            formatted_prompt = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=few_shot_examples,
                context=context,
                question=question,
                answerA=ans_a,
                answerB=ans_b,
                answerC=ans_c,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"
    else:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_GENERATION
        try:
            formatted_prompt = format_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                context=context,
                question=question,
                answerA=ans_a,
                answerB=ans_b,
                answerC=ans_c,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"

def _extract_siqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    
    match = re.search(r'^\s*\b([1-3])\b', completion_part)
    if match:
        return match.group(1)
    
    return "X"

def _compute_likelihood_score(pipe, tokenizer, text: str) -> float:
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(pipe.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = pipe.model(**inputs)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            total_log_prob = gathered_log_probs.sum().item()
            
            return total_log_prob
            
    except Exception as e:
        return float('-inf')

def evaluate_siqa(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_SIQA,
    dataset_split: str = DEFAULT_SPLIT_SIQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_SIQA,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_SIQA,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_SIQA,
    evaluation_method: str = "generation",
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:

    if siqa_accuracy_metric is None:
        return {"SIQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"SIQA": 0.0, "error_message": f"DatasetLoadFailed SIQA: {e}"}

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
        return {"SIQA": 0.0}

    predictions_numeric, true_labels_numeric = [], []
    outputs_dump = []

    if evaluation_method == "likelihood":
        for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - SIQA Likelihood Eval")):
            true_label_str = str(item_data.get('label', '')).strip()
            if true_label_str not in ['1', '2', '3']:
                continue

            try:
                choice1_text, choice2_text, choice3_text = _format_siqa_prompt_likelihood_with_template(
                    item_data, 
                    num_few_shot=num_few_shot
                )
                
                score1 = _compute_likelihood_score(pipe, tokenizer, choice1_text)
                score2 = _compute_likelihood_score(pipe, tokenizer, choice2_text)
                score3 = _compute_likelihood_score(pipe, tokenizer, choice3_text)
                
                scores = [score1, score2, score3]
                predicted_choice = scores.index(max(scores)) + 1
                true_choice = int(true_label_str)
                
                predictions_numeric.append(predicted_choice)
                true_labels_numeric.append(true_choice)
                
                if save_outputs:
                    outputs_dump.append({
                        "context": item_data.get('context', ''),
                        "question": item_data.get('question', ''),
                        "answerA": item_data.get('answerA', ''),
                        "answerB": item_data.get('answerB', ''),
                        "answerC": item_data.get('answerC', ''),
                        "correct_answer": true_choice,
                        "predicted_answer": predicted_choice,
                        "is_correct": predicted_choice == true_choice,
                        "choice1_completion": choice1_text,
                        "choice2_completion": choice2_text,
                        "choice3_completion": choice3_text,
                        "choice1_score": score1,
                        "choice2_score": score2,
                        "choice3_score": score3,
                        "evaluation_method": "likelihood"
                    })
                    
            except Exception as e:
                true_choice = int(item_data.get('label', '1'))
                wrong_choice = (true_choice % 3) + 1
                predictions_numeric.append(wrong_choice)
                true_labels_numeric.append(true_choice)
                
    else:
        prompts_for_batch, original_items_for_batch = [], []
        
        for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - SIQA Generation Eval")):
            true_label_str = str(item_data.get('label', '')).strip()
            if true_label_str not in ['1', '2', '3']:
                continue

            prompt_text = _format_siqa_prompt_generation_with_template(
                item_data, 
                num_few_shot=num_few_shot
            )
            prompts_for_batch.append(prompt_text)
            original_items_for_batch.append(item_data)

            if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
                gen_config = {
                    "do_sample": False, 
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens, 
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "return_full_text": True
                }
                
                try:
                    with torch.no_grad():
                        batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                        
                    for k, raw_out_list in enumerate(batch_raw_outputs):
                        original_item = original_items_for_batch[k]
                        prompt = prompts_for_batch[k]
                        raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompt + "X"
                        pred_str = _extract_siqa_answer(raw_gen, prompt)
                        pred_num = int(pred_str) if pred_str in ["1", "2", "3"] else -1
                        true_num = int(original_item['label'])
                        
                        if pred_num == -1:
                            pred_num = (true_num % 3) + 1
                        
                        predictions_numeric.append(pred_num)
                        true_labels_numeric.append(true_num)
                        
                        if save_outputs:
                            outputs_dump.append({
                                "context": original_item.get('context', ''),
                                "question": original_item.get('question', ''),
                                "answerA": original_item.get('answerA', ''),
                                "answerB": original_item.get('answerB', ''),
                                "answerC": original_item.get('answerC', ''),
                                "correct_answer": true_num,
                                "predicted_answer": pred_num,
                                "is_correct": pred_num == true_num,
                                "prompt": prompt,
                                "raw_response": raw_gen,
                                "extracted_completion": raw_gen[len(prompt):].strip() if raw_gen.startswith(prompt) else raw_gen.strip(),
                                "evaluation_method": "generation"
                            })
                            
                except Exception as e_batch_siqa:
                    for item_err_info in original_items_for_batch:
                        true_num_err = int(item_err_info['label'])
                        predictions_numeric.append((true_num_err % 3) + 1)
                        true_labels_numeric.append(true_num_err)
                        
                prompts_for_batch, original_items_for_batch = [], []

    if not true_labels_numeric:
        return {"SIQA": 0.0}

    acc_score = 0.0
    try:
        valid_indices = [i for i, ref in enumerate(true_labels_numeric) if ref in [1, 2, 3]]
        if valid_indices:
            valid_preds = [predictions_numeric[i] for i in valid_indices]
            valid_refs = [true_labels_numeric[i] for i in valid_indices]
            if valid_preds and valid_refs:
                acc_results = siqa_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
                acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric:
        pass

    if save_outputs and outputs_dump:
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"siqa_outputs_{model_name_for_logging.replace('/', '_')}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        summary_data = {
            "model_name": model_name_for_logging,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "num_few_shot": num_few_shot,
            "evaluation_method": evaluation_method,
            "total_examples": len(outputs_dump),
            "accuracy": acc_score,
            "correct_predictions": sum(1 for item in outputs_dump if item["is_correct"]),
            "process_id": process_id,
            "gpu_id": gpu_id,
            "examples": outputs_dump
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
        except Exception as e_save:
            pass

    return {"SIQA": acc_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test SIQA")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser.add_argument("--num_few_shot_test", type=int, default=3)
    test_parser.add_argument("--evaluation_method", type=str, default="generation", choices=["likelihood", "generation"])
    test_parser.add_argument("--save_outputs", action="store_true")
    
    si_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="SIQAFileTest")
    
    si_pipe, _ = initialize_model_pipeline(si_args.model_name_test, target_device_id=0)
    if si_pipe:
        si_eval_args = {
            "pipe": si_pipe,
            "tokenizer": si_pipe.tokenizer,
            "model_name_for_logging": si_args.model_name_test,
            "device": si_pipe.device,
            "dataset_split": si_args.dataset_split_test,
            "generation_batch_size": si_args.gen_batch_size_test,
            "num_few_shot": si_args.num_few_shot_test,
            "evaluation_method": si_args.evaluation_method,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": si_args.save_outputs
        }
        try:
            print(json.dumps(evaluate_siqa(**si_eval_args), indent=2))
        finally:
            cleanup_model_resources(si_pipe, getattr(si_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {si_args.model_name_test} for SIQA test.")