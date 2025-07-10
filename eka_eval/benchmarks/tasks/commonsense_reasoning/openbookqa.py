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

DEFAULT_DATASET_NAME_OBQA = "allenai/openbookqa"
DEFAULT_CONFIG_OBQA = "main"
DEFAULT_SPLIT_OBQA = "validation"
DEFAULT_MAX_NEW_TOKENS_OBQA = 5
DEFAULT_GENERATION_BATCH_SIZE_OBQA = 8
DEFAULT_FEW_SHOT_COUNT_OBQA = 5
DEFAULT_PROMPT_TEMPLATE_KEY_LIKELIHOOD = "openbookqa_likelihood"
DEFAULT_PROMPT_TEMPLATE_KEY_GENERATION = "openbookqa_generation"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_LIKELIHOOD = "openbookqa_5shot_likelihood"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_GENERATION = "openbookqa_5shot_generation"
PROMPT_FILE_BENCHMARK_KEY = "openbookqa"
PROMPT_FILE_CATEGORY = "reasoning"

try:
    obqa_accuracy_metric = hf_evaluate.load("accuracy")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for OBQA: {e}")
    obqa_accuracy_metric = None

def _get_obqa_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    if num_few_shot <= 0:
        return []
    loaded_examples_list = get_prompt_data(PROMPT_FILE_BENCHMARK_KEY, data_key="default_few_shot_examples_openbookqa", specific_task_group=prompt_file_category)
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        return loaded_examples_list[:num_few_shot]
    return []

def doc_to_choice_obqa(doc):
    choices_dict = doc.get('choices', {})
    choice_texts = choices_dict.get('text', [])
    choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))])
    return choice_texts, choice_labels

def _format_choices_string(choice_texts: List[str], choice_labels: List[str]) -> str:
    choices_str = ""
    for label, text in zip(choice_labels, choice_texts):
        choices_str += f"{label}. {text}\n"
    return choices_str.strip()

def _format_obqa_prompt_likelihood_with_template(
    item: Dict, 
    num_few_shot: int = 0,
    prompt_file_category: str = PROMPT_FILE_CATEGORY
) -> Tuple[str, str, str, str]:
    question_stem = item.get('question_stem', '').strip()
    choice_texts, choice_labels = doc_to_choice_obqa(item)
    
    if num_few_shot > 0:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_LIKELIHOOD
        few_shot_examples = _get_obqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        
        try:
            completions = []
            for choice_text in choice_texts:
                completion = format_few_shot_prompt(
                    benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                    template_key=template_key,
                    few_shot_examples=few_shot_examples,
                    question_stem=question_stem,
                    choice_text=choice_text,
                    specific_task_group=prompt_file_category
                )
                completions.append(completion)
            
            while len(completions) < 4:
                completions.append(f"Error: Missing choice {len(completions)}")
            
            return completions[0], completions[1], completions[2], completions[3]
        except Exception as e:
            error_text = f"Error: Prompt formatting failed - {str(e)}"
            return error_text, error_text, error_text, error_text
    else:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_LIKELIHOOD
        try:
            completions = []
            for choice_text in choice_texts:
                completion = format_prompt(
                    benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                    template_key=template_key,
                    question_stem=question_stem,
                    choice_text=choice_text,
                    specific_task_group=prompt_file_category
                )
                completions.append(completion)
            
            while len(completions) < 4:
                completions.append(f"Error: Missing choice {len(completions)}")
            
            return completions[0], completions[1], completions[2], completions[3]
        except Exception as e:
            error_text = f"Error: Prompt formatting failed - {str(e)}"
            return error_text, error_text, error_text, error_text

def _format_obqa_prompt_generation_with_template(
    item: Dict, 
    num_few_shot: int = 0,
    prompt_file_category: str = PROMPT_FILE_CATEGORY
) -> str:
    question_stem = item.get('question_stem', '').strip()
    choice_texts, choice_labels = doc_to_choice_obqa(item)
    choices_str = _format_choices_string(choice_texts, choice_labels)
    
    if num_few_shot > 0:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_GENERATION
        few_shot_examples = _get_obqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        
        processed_examples = []
        for ex in few_shot_examples:
            ex_choices_str = _format_choices_string(ex.get('choices', []), ex.get('labels', ['A', 'B', 'C', 'D']))
            processed_ex = ex.copy()
            processed_ex['choices_str'] = ex_choices_str
            processed_examples.append(processed_ex)
        
        try:
            formatted_prompt = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=processed_examples,
                question_stem=question_stem,
                choices_str=choices_str,
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
                question_stem=question_stem,
                choices_str=choices_str,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"

def _extract_openbookqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    
    match = re.search(r'(?:[Aa]nswer[:\s]*)?\b([A-D])\b', completion_part)
    if match:
        return match.group(1).upper()
    
    simple_match = re.search(r'^[A-D]', completion_part, re.IGNORECASE)
    if simple_match:
        return simple_match.group(0).upper()
    
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

def evaluate_openbookqa(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_OBQA,
    dataset_config_name: str = DEFAULT_CONFIG_OBQA,
    dataset_split: str = DEFAULT_SPLIT_OBQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_OBQA,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_OBQA,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_OBQA,
    evaluation_method: str = "generation",
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:

    if obqa_accuracy_metric is None:
        return {"OpenBookQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"OpenBookQA": 0.0, "error_message": f"DatasetLoadFailed OBQA: {e}"}

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
        return {"OpenBookQA": 0.0}

    predictions_numeric, true_labels_numeric = [], []
    label_to_int = {chr(65 + i): i for i in range(4)}
    outputs_dump = []

    if evaluation_method == "likelihood":
        for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - OBQA Likelihood Eval")):
            true_answer_letter = item_data.get('answerKey', '').strip()
            if not true_answer_letter or true_answer_letter not in label_to_int:
                continue

            try:
                choice1_text, choice2_text, choice3_text, choice4_text = _format_obqa_prompt_likelihood_with_template(
                    item_data, 
                    num_few_shot=num_few_shot
                )
                
                score1 = _compute_likelihood_score(pipe, tokenizer, choice1_text)
                score2 = _compute_likelihood_score(pipe, tokenizer, choice2_text)
                score3 = _compute_likelihood_score(pipe, tokenizer, choice3_text)
                score4 = _compute_likelihood_score(pipe, tokenizer, choice4_text)
                
                scores = [score1, score2, score3, score4]
                predicted_choice_idx = scores.index(max(scores))
                predicted_choice_letter = chr(65 + predicted_choice_idx)
                true_choice_idx = label_to_int[true_answer_letter]
                
                predictions_numeric.append(predicted_choice_idx)
                true_labels_numeric.append(true_choice_idx)
                
                if save_outputs:
                    choice_texts, choice_labels = doc_to_choice_obqa(item_data)
                    outputs_dump.append({
                        "question_stem": item_data.get('question_stem', ''),
                        "choices": choice_texts,
                        "labels": choice_labels,
                        "correct_answer": true_answer_letter,
                        "predicted_answer": predicted_choice_letter,
                        "is_correct": predicted_choice_idx == true_choice_idx,
                        "choice_completions": [choice1_text, choice2_text, choice3_text, choice4_text],
                        "choice_scores": scores,
                        "evaluation_method": "likelihood"
                    })
                    
            except Exception as e:
                true_choice_idx = label_to_int.get(item_data.get('answerKey', 'A').strip(), 0)
                wrong_choice_idx = (true_choice_idx + 1) % 4
                predictions_numeric.append(wrong_choice_idx)
                true_labels_numeric.append(true_choice_idx)
                
    else:
        prompts_for_batch, original_items_for_batch = [], []
        
        for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - OBQA Generation Eval")):
            true_answer_letter = item_data.get('answerKey', '').strip()
            if not true_answer_letter or true_answer_letter not in label_to_int:
                continue

            prompt_text = _format_obqa_prompt_generation_with_template(
                item_data, 
                num_few_shot=num_few_shot
            )
            prompts_for_batch.append(prompt_text)
            original_items_for_batch.append(item_data)

            if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
                gen_config_obqa = {
                    "do_sample": False,
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "return_full_text": True
                }
                
                try:
                    with torch.no_grad():
                        batch_raw_outputs = pipe(prompts_for_batch, **gen_config_obqa)
                        
                    for k, raw_out_list in enumerate(batch_raw_outputs):
                        original_item = original_items_for_batch[k]
                        prompt = prompts_for_batch[k]
                        raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompt + "X"
                        pred_letter = _extract_openbookqa_answer(raw_gen, prompt)
                        pred_num = label_to_int.get(pred_letter, -1)
                        true_num = label_to_int.get(original_item['answerKey'].strip(), -1)
                        
                        if pred_num == -1 and true_num != -1:
                            pred_num = (true_num + 1) % 4
                        
                        predictions_numeric.append(pred_num)
                        true_labels_numeric.append(true_num)
                        
                        if save_outputs:
                            choice_texts, choice_labels = doc_to_choice_obqa(original_item)
                            outputs_dump.append({
                                "question_stem": original_item.get('question_stem', ''),
                                "choices": choice_texts,
                                "labels": choice_labels,
                                "correct_answer": original_item['answerKey'].strip(),
                                "predicted_answer": pred_letter,
                                "is_correct": pred_num == true_num,
                                "prompt": prompt,
                                "raw_response": raw_gen,
                                "extracted_completion": raw_gen[len(prompt):].strip() if raw_gen.startswith(prompt) else raw_gen.strip(),
                                "evaluation_method": "generation"
                            })
                            
                except Exception as e_batch_obqa:
                    for item_err_info in original_items_for_batch:
                        true_num_err = label_to_int.get(item_err_info['answerKey'].strip(), 0)
                        predictions_numeric.append((true_num_err + 1) % 4)
                        true_labels_numeric.append(true_num_err)
                        
                prompts_for_batch, original_items_for_batch = [], []

    if not true_labels_numeric:
        return {"OpenBookQA": 0.0}

    acc_score = 0.0
    try:
        valid_preds = [p for i, p in enumerate(predictions_numeric) if true_labels_numeric[i] != -1]
        valid_refs = [r for r in true_labels_numeric if r != -1]
        if valid_preds and valid_refs:
            acc_results = obqa_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
            acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric:
        pass

    if save_outputs and outputs_dump:
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"obqa_outputs_{model_name_for_logging.replace('/', '_')}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        summary_data = {
            "model_name": model_name_for_logging,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config_name,
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

    return {"OpenBookQA": acc_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test OpenBookQA")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser.add_argument("--num_few_shot_test", type=int, default=3)
    test_parser.add_argument("--evaluation_method", type=str, default="generation", choices=["likelihood", "generation"])
    test_parser.add_argument("--save_outputs", action="store_true")
    
    ob_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="OBQAFileTest")
    
    ob_pipe, _ = initialize_model_pipeline(ob_args.model_name_test, target_device_id=0)
    if ob_pipe:
        ob_eval_args = {
            "pipe": ob_pipe,
            "tokenizer": ob_pipe.tokenizer,
            "model_name_for_logging": ob_args.model_name_test,
            "device": ob_pipe.device,
            "dataset_split": ob_args.dataset_split_test,
            "generation_batch_size": ob_args.gen_batch_size_test,
            "num_few_shot": ob_args.num_few_shot_test,
            "evaluation_method": ob_args.evaluation_method,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": ob_args.save_outputs
        }
        try:
            print(json.dumps(evaluate_openbookqa(**ob_eval_args), indent=2))
        finally:
            cleanup_model_resources(ob_pipe, getattr(ob_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {ob_args.model_name_test} for OBQA test.")