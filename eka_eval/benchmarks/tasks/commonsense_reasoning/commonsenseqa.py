import torch
import sys
import argparse 
import re
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

DEFAULT_DATASET_NAME_CSQA = "commonsense_qa"
DEFAULT_SPLIT_CSQA = "validation"
DEFAULT_MAX_NEW_TOKENS_CSQA = 5
DEFAULT_FEW_SHOT_COUNT_CSQA = 7
DEFAULT_GENERATION_BATCH_SIZE_CSQA = 8
DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT = "commonsenseqa_0shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_5 = "commonsenseqa_5shot"
DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_7 = "commonsenseqa_7shot"
PROMPT_FILE_BENCHMARK_KEY = "commonsenseqa"
PROMPT_FILE_CATEGORY = "reasoning"

try:
    csqa_accuracy_metric = hf_evaluate.load("accuracy")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for CSQA: {e}")
    csqa_accuracy_metric = None

def _get_csqa_fewshot_examples_from_config(num_few_shot: int, prompt_file_category: str) -> List[Dict]:
    if num_few_shot <= 0:
        return []
    loaded_examples_list = get_prompt_data(PROMPT_FILE_BENCHMARK_KEY, data_key="default_few_shot_examples_commonsenseqa", specific_task_group=prompt_file_category)
    if loaded_examples_list and isinstance(loaded_examples_list, list):
        return loaded_examples_list[:num_few_shot]
    return []

def _format_choices_string_csqa(choice_texts: List[str], choice_labels: List[str]) -> str:
    choices_str = ""
    for label, text in zip(choice_labels, choice_texts):
        choices_str += f"{label}. {text}\n"
    return choices_str.strip()

def _format_csqa_prompt_with_template(
    item: Dict, 
    num_few_shot: int = 0,
    prompt_file_category: str = PROMPT_FILE_CATEGORY
) -> str:
    question = item.get('question', '').strip()
    choices_dict = item.get('choices', {})
    choice_texts = choices_dict.get('text', [])
    choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))])
    choices_str = _format_choices_string_csqa(choice_texts, choice_labels)
    
    if num_few_shot > 0:
        if num_few_shot <= 5:
            template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_5
        else:
            template_key = DEFAULT_PROMPT_TEMPLATE_KEY_FEW_SHOT_7
            
        few_shot_examples = _get_csqa_fewshot_examples_from_config(num_few_shot, prompt_file_category)
        
        processed_examples = []
        for ex in few_shot_examples:
            ex_choices = ex.get('choices', [])
            ex_labels = [chr(65 + i) for i in range(len(ex_choices))]
            ex_choices_str = _format_choices_string_csqa(ex_choices, ex_labels)
            processed_ex = ex.copy()
            processed_ex['choices_str'] = ex_choices_str
            processed_examples.append(processed_ex)
        
        try:
            formatted_prompt = format_few_shot_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                few_shot_examples=processed_examples,
                question=question,
                choices_str=choices_str,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"
    else:
        template_key = DEFAULT_PROMPT_TEMPLATE_KEY_ZERO_SHOT
        try:
            formatted_prompt = format_prompt(
                benchmark_key=PROMPT_FILE_BENCHMARK_KEY,
                template_key=template_key,
                question=question,
                choices_str=choices_str,
                specific_task_group=prompt_file_category
            )
            return formatted_prompt
        except Exception as e:
            return f"Error: Prompt formatting failed - {str(e)}"

def _extract_commonsenseqa_answer(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    completion_part = completion_part.strip()
    
    match = re.search(r'(?:[Aa]nswer[:\s]*)?\b([A-E])\b', completion_part)
    if match:
        return match.group(1).upper()
    
    simple_match = re.search(r'^[A-E]', completion_part, re.IGNORECASE)
    if simple_match:
        return simple_match.group(0).upper()
    
    return "X"

def evaluate_commonsenseqa(
    pipe: Any, 
    tokenizer: Any, 
    model_name_for_logging: str, 
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_CSQA,
    dataset_split: str = DEFAULT_SPLIT_CSQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_CSQA,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_CSQA,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_CSQA,
    process_id: int = 0, 
    gpu_id: int = 0, 
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:

    if csqa_accuracy_metric is None:
        return {"CommonSenseQA": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"CommonSenseQA": 0.0, "error_message": f"DatasetLoadFailed CSQA: {e}"}

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
        return {"CommonSenseQA": 0.0}

    predictions_numeric, true_labels_numeric = [], []
    label_to_int = {chr(65 + i): i for i in range(5)}
    outputs_dump = []

    prompts_for_batch, original_items_for_batch = [], []
    
    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - CSQA Eval")):
        true_answer_letter = item_data.get('answerKey')
        if not true_answer_letter or true_answer_letter not in label_to_int:
            continue
        
        prompt_text = _format_csqa_prompt_with_template(
            item_data, 
            num_few_shot=num_few_shot
        )
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append(item_data)

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config_csqa = {
                "do_sample": False, 
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens, 
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": True
            }
            
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config_csqa)
                    
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item = original_items_for_batch[k]
                    prompt = prompts_for_batch[k]
                    raw_gen = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else prompt + "X"
                    pred_letter = _extract_commonsenseqa_answer(raw_gen, prompt)
                    pred_num = label_to_int.get(pred_letter, -1)
                    true_num = label_to_int.get(original_item['answerKey'], -1)
                    
                    if pred_num == -1 and true_num != -1:
                        pred_num = (true_num + 1) % 5
                    
                    predictions_numeric.append(pred_num)
                    true_labels_numeric.append(true_num)
                    
                    if save_outputs:
                        choices_dict = original_item.get('choices', {})
                        choice_texts = choices_dict.get('text', [])
                        choice_labels = choices_dict.get('label', [chr(65 + i) for i in range(len(choice_texts))])
                        
                        outputs_dump.append({
                            "question": original_item.get('question', ''),
                            "choices": choice_texts,
                            "labels": choice_labels,
                            "correct_answer": original_item['answerKey'],
                            "predicted_answer": pred_letter,
                            "is_correct": pred_num == true_num,
                            "prompt": prompt,
                            "raw_response": raw_gen,
                            "extracted_completion": raw_gen[len(prompt):].strip() if raw_gen.startswith(prompt) else raw_gen.strip()
                        })
                        
            except Exception as e_batch_csqa:
                for item_err_info in original_items_for_batch:
                    true_num_err = label_to_int.get(item_err_info['answerKey'], 0)
                    predictions_numeric.append((true_num_err + 1) % 5)
                    true_labels_numeric.append(true_num_err)
                    
            prompts_for_batch, original_items_for_batch = [], []

    if not true_labels_numeric:
        return {"CommonSenseQA": 0.0}

    acc_score = 0.0
    try:
        valid_preds = [p for i, p in enumerate(predictions_numeric) if true_labels_numeric[i] != -1]
        valid_refs = [r for r in true_labels_numeric if r != -1]
        if valid_preds and valid_refs:
            acc_results = csqa_accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
            acc_score = acc_results.get("accuracy", 0.0) * 100
    except Exception as e_metric:
        pass

    if save_outputs and outputs_dump:
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"csqa_outputs_{model_name_for_logging.replace('/', '_')}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        summary_data = {
            "model_name": model_name_for_logging,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "num_few_shot": num_few_shot,
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

    return {"CommonSenseQA": acc_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test CommonsenseQA")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:10]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser.add_argument("--num_few_shot_test", type=int, default=3)
    test_parser.add_argument("--save_outputs", action="store_true")
    
    cs_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="CSQAFileTest")
    
    cs_pipe, _ = initialize_model_pipeline(cs_args.model_name_test, target_device_id=0)
    if cs_pipe:
        cs_eval_args = {
            "pipe": cs_pipe,
            "tokenizer": cs_pipe.tokenizer,
            "model_name_for_logging": cs_args.model_name_test,
            "device": cs_pipe.device,
            "dataset_split": cs_args.dataset_split_test,
            "generation_batch_size": cs_args.gen_batch_size_test,
            "num_few_shot": cs_args.num_few_shot_test,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": cs_args.save_outputs
        }
        try:
            print(json.dumps(evaluate_commonsenseqa(**cs_eval_args), indent=2))
        finally:
            cleanup_model_resources(cs_pipe, getattr(cs_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {cs_args.model_name_test} for CSQA test.")