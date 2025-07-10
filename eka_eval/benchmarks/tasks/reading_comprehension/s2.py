# eka_eval/benchmarks/tasks/reading_comprehension/squad.py

import torch
import sys
import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import string
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_SQUAD = "squad"
DEFAULT_SPLIT_SQUAD = "validation"
DEFAULT_MAX_NEW_TOKENS_SQUAD = 64
DEFAULT_CHECKPOINT_DIR_SQUAD = "checkpoints/squad_checkpoints"
DEFAULT_FEW_SHOT_COUNT_SQUAD = 3

try:
    squad_metric = hf_evaluate.load("squad")
    logger.info("SQuAD metric loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'squad' metric: {e}. SQuAD may not run correctly.", exc_info=True)
    squad_metric = None

# Few-shot examples for SQuAD (reading comprehension examples)
DEFAULT_FEW_SHOT_EXAMPLES_SQUAD = [
    {
        "context": "The Amazon rainforest, also known as the Amazon Jungle, is a moist broadleaf forest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest.",
        "question": "How many square kilometers is the Amazon basin?",
        "answer": "7,000,000 square kilometers"
    },
    {
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League for the 2015 season. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
        "question": "Where was Super Bowl 50 played?",
        "answer": "Levi's Stadium"
    },
    {
        "context": "Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist who is best known for his contributions to the design of the modern alternating current electricity supply system.",
        "question": "What nationality was Nikola Tesla?",
        "answer": "Serbian-American"
    }
]

def _normalize_answer_squad(s: str) -> str:
    """Normalize answer following SQuAD evaluation standards"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    
    if not isinstance(s, str): 
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _format_squad_prompt_improved(context: str, question: str, few_shot_examples: List[Dict]) -> str:
    """
    Improved SQuAD prompt format with few-shot examples and better instructions.
    """
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        prompt += "Read the passage carefully and answer the question based only on the information provided in the passage.\n\n"
        for i, ex_item in enumerate(few_shot_examples, 1):
            ex_context = ex_item.get('context', '').strip()
            ex_question = ex_item.get('question', '').strip()
            ex_answer = ex_item.get('answer', '').strip()
            
            prompt += f"Example {i}:\n"
            prompt += f"Passage: {ex_context}\n"
            prompt += f"Question: {ex_question}\n"
            prompt += f"Answer: {ex_answer}\n\n"
        
        prompt += "Now answer this question:\n"
    else:
        prompt += "Read the passage carefully and answer the question based only on the information provided.\n\n"
    
    prompt += f"Passage: {context}\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer:"
    
    return prompt

def _extract_answer_from_response(generated_text: str, prompt: str) -> str:
    """Extract answer from generated response, handling various formats"""
    # Remove the prompt part
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
    
    # Handle common response patterns
    # Remove "Answer:" prefix if present
    response = re.sub(r'^[Aa]nswer\s*:?\s*', '', response)
    
    # Take only the first sentence/line as answer (before any explanation)
    lines = response.split('\n')
    answer = lines[0].strip()
    
    # Remove common prefixes
    answer = re.sub(r'^(The answer is|It is|According to the passage,?)\s*', '', answer, flags=re.IGNORECASE)
    
    # Remove trailing punctuation that's not part of the answer
    answer = re.sub(r'\s*[.!?]*$', '', answer)
    
    return answer if answer else response

def _save_checkpoint_squad(checkpoint_filepath: str, predictions_so_far: List[Dict], references_so_far: List[Dict], processed_qas_ids: set):
    """Save checkpoint for resuming evaluation"""
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    try:
        data_to_save = {
            'predictions': predictions_so_far,
            'references': references_so_far,
            'processed_qas_ids': list(processed_qas_ids)
        }
        with open(checkpoint_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logger.info(f"SQuAD Checkpoint saved to {checkpoint_filepath}")
    except Exception as e:
        logger.error(f"Failed to save SQuAD checkpoint to {checkpoint_filepath}: {e}", exc_info=True)

def evaluate_squad(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_SQUAD,
    dataset_split: str = DEFAULT_SPLIT_SQUAD,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_SQUAD,
    generation_batch_size: int = 8,
    num_few_shot: int = DEFAULT_FEW_SHOT_COUNT_SQUAD,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR_SQUAD,
    resume: bool = False,
    checkpoint_save_interval_batches: int = 50,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:
    
    if squad_metric is None:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": "MetricLoadFailed"}

    logger.info(f"Starting SQuAD ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")
    logger.info(f"P{process_id}(GPU{gpu_id}): split='{dataset_split}', batch_size={generation_batch_size}")

    try:
        full_data_for_split = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}
    
    logger.info(f"P{process_id}: Loaded SQuAD '{dataset_name}' ({len(full_data_for_split)} examples) for split '{dataset_split}'.")

    if num_gpus > 1:
        total_examples = len(full_data_for_split)
        examples_per_instance = total_examples // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1: 
            end_idx = total_examples
        dataset_subset_to_process = full_data_for_split.select(range(start_idx, end_idx))
        logger.info(f"P{process_id}: Processing {len(dataset_subset_to_process)} examples (from {start_idx} to {end_idx-1}).")
    else:
        dataset_subset_to_process = full_data_for_split

    if len(dataset_subset_to_process) == 0:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": "NoSamplesAfterSplit"}

    # Prepare few-shot examples
    few_shot_examples_list = DEFAULT_FEW_SHOT_EXAMPLES_SQUAD[:num_few_shot] if num_few_shot > 0 else []

    # Setup checkpoint
    checkpoint_filename = f"squad_checkpoint_p{process_id}_gpu{gpu_id}.json"
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
    predictions_log: List[Dict[str, str]] = []
    references_log: List[Dict[str, Any]] = []
    processed_qas_ids_from_checkpoint: set = set()
    outputs_dump = []

    if resume and os.path.exists(checkpoint_filepath):
        logger.info(f"P{process_id}: Resuming SQuAD from checkpoint {checkpoint_filepath}...")
        try:
            with open(checkpoint_filepath, 'r') as f: 
                checkpoint_data = json.load(f)
            predictions_log = checkpoint_data.get('predictions', [])
            references_log = checkpoint_data.get('references', [])
            processed_qas_ids_from_checkpoint = set(checkpoint_data.get('processed_qas_ids', []))
            logger.info(f"P{process_id}: Loaded {len(predictions_log)} predictions from SQuAD checkpoint.")
        except Exception as e:
            logger.error(f"P{process_id}: Error reading SQuAD checkpoint: {e}. Starting fresh.", exc_info=True)
            predictions_log, references_log, processed_qas_ids_from_checkpoint = [], [], set()

    prompts_to_generate, current_batch_info_for_processing = [], []
    for example_data in tqdm(dataset_subset_to_process, desc=f"P{process_id} - Preparing SQuAD", disable=False):
        qas_id = example_data.get('id')
        if qas_id is None:
            logger.warning("SQuAD: Example missing 'id'. Skipping.")
            continue
        if qas_id in processed_qas_ids_from_checkpoint: 
            continue

        context = example_data.get('context', "")
        question = example_data.get('question', "")
        answers_dict = example_data.get('answers')

        if not context or not question or not answers_dict or not answers_dict.get('text'):
            logger.warning(f"SQuAD: Skipping QAS ID {qas_id} due to missing context, question, or answers text.")
            continue
        
        # Use improved prompt format
        prompt = _format_squad_prompt_improved(context, question, few_shot_examples_list)
        prompts_to_generate.append(prompt)
        current_batch_info_for_processing.append({
            'id': qas_id, 
            'answers_dict': answers_dict,
            'context': context,
            'question': question,
            'prompt': prompt
        })

    if not prompts_to_generate:
        logger.info(f"P{process_id}: No new SQuAD examples to process after filtering from checkpoint.")
        if predictions_log and references_log:
            try:
                norm_preds = [{'id': p['id'], 'prediction_text': _normalize_answer_squad(p['prediction_text'])} for p in predictions_log]
                norm_refs = [{'id': r['id'], 'answers': {'text': [_normalize_answer_squad(ans) for ans in r['answers']['text']], 'answer_start': r['answers']['answer_start']}} for r in references_log]
                if norm_preds and norm_refs:
                    final_results = squad_metric.compute(predictions=norm_preds, references=norm_refs)
                    f1, em = final_results.get('f1', 0.0), final_results.get('exact_match', 0.0)
                    return {"SQuAD": f1, "SQuAD_exact_match": em, "SQuAD_f1": f1}
            except Exception as e_metric:
                logger.error(f"P{process_id}: Error computing SQuAD metrics on resumed data: {e_metric}")
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0}

    logger.info(f"P{process_id}: Starting SQuAD batch inference for {len(prompts_to_generate)} prompts (batch_size={generation_batch_size}).")

    # Improved generation config
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,  # Deterministic generation
        "repetition_penalty": 1.1,  # Reduce repetition
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }

    for i in tqdm(range(0, len(prompts_to_generate), generation_batch_size), desc=f"P{process_id} - Generating SQuAD", unit="batch"):
        batch_prompts_slice = prompts_to_generate[i : i + generation_batch_size]
        batch_info_slice = current_batch_info_for_processing[i : i + generation_batch_size]
        
        try:
            with torch.no_grad():
                batch_outputs_raw = pipe(batch_prompts_slice, **generation_config)

            for j, output_list_item in enumerate(batch_outputs_raw):
                info_item = batch_info_slice[j]
                qas_id = info_item['id']
                answers_dict = info_item['answers_dict']
                prompt = info_item['prompt']
                
                # Extract and clean prediction
                if output_list_item and output_list_item[0] and 'generated_text' in output_list_item[0]:
                    raw_generated = output_list_item[0]['generated_text']
                    pred_text = _extract_answer_from_response(raw_generated, prompt)
                else:
                    raw_generated = "#GenFail"
                    pred_text = "#GenFail"
                
                predictions_log.append({'id': qas_id, 'prediction_text': pred_text})
                references_log.append({'id': qas_id, 'answers': answers_dict})
                processed_qas_ids_from_checkpoint.add(qas_id)
                
                # Save detailed output if requested
                if save_outputs:
                    # Normalize for comparison
                    pred_normalized = _normalize_answer_squad(pred_text)
                    ref_texts_normalized = [_normalize_answer_squad(ans) for ans in answers_dict['text']]
                    
                    # Check if prediction matches any reference (for is_correct)
                    is_correct = any(pred_normalized == ref_norm for ref_norm in ref_texts_normalized)
                    
                    outputs_dump.append({
                        "id": qas_id,
                        "context": info_item['context'],
                        "question": info_item['question'],
                        "reference_answers": answers_dict['text'],
                        "predicted_answer": pred_text,
                        "predicted_answer_normalized": pred_normalized,
                        "reference_answers_normalized": ref_texts_normalized,
                        "is_correct": is_correct,
                        "prompt": prompt,
                        "raw_response": raw_generated
                    })

        except Exception as e_batch_gen:
            logger.error(f"P{process_id}: Error during SQuAD generation batch {i//generation_batch_size}: {e_batch_gen}", exc_info=True)
            for info_item_err in batch_info_slice:
                qas_id, answers_dict = info_item_err['id'], info_item_err['answers_dict']
                if qas_id not in processed_qas_ids_from_checkpoint:
                    predictions_log.append({'id': qas_id, 'prediction_text': "#PipelineError"})
                    references_log.append({'id': qas_id, 'answers': answers_dict})
                    processed_qas_ids_from_checkpoint.add(qas_id)

        # Save checkpoint periodically
        current_batch_num = (i // generation_batch_size) + 1
        if current_batch_num % checkpoint_save_interval_batches == 0:
            _save_checkpoint_squad(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)

    # Save final checkpoint
    if prompts_to_generate:
        _save_checkpoint_squad(checkpoint_filepath, predictions_log, references_log, processed_qas_ids_from_checkpoint)
    
    logger.info(f"P{process_id}: SQuAD inference complete. Total items for metric: {len(predictions_log)}.")

    if not predictions_log or not references_log:
        return {"SQuAD": 0.0, "SQuAD_exact_match": 0.0, "SQuAD_f1": 0.0, "error_message": "NoPredsOrRefsForMetric"}

    # Compute final metrics
    final_metric_predictions = [{'id': p['id'], 'prediction_text': _normalize_answer_squad(p['prediction_text'])} for p in predictions_log]
    final_metric_references = []
    for r_item in references_log:
        final_metric_references.append({
            'id': r_item['id'],
            'answers': {
                'text': [_normalize_answer_squad(ans_text) for ans_text in r_item['answers']['text']],
                'answer_start': r_item['answers']['answer_start']
            }
        })

    em_score, f1_score = 0.0, 0.0
    try:
        if final_metric_predictions and final_metric_references:
            squad_eval_results = squad_metric.compute(predictions=final_metric_predictions, references=final_metric_references)
            em_score = squad_eval_results.get('exact_match', 0.0)
            f1_score = squad_eval_results.get('f1', 0.0)
        else:
            logger.warning(f"P{process_id}: Not enough data for SQuAD metric computation after normalization.")
    except Exception as e_metric_final:
        logger.error(f"P{process_id}: Error computing final SQuAD metrics: {e_metric_final}", exc_info=True)
        f1_score, em_score = 0.0, 0.0

    # Save outputs to JSON file if requested
    if save_outputs and outputs_dump:
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"squad_outputs_{model_name_for_logging.replace('/', '_')}_p{process_id}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        summary_data = {
            "model_name": model_name_for_logging,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "num_few_shot": num_few_shot,
            "total_examples": len(outputs_dump),
            "exact_match": em_score,
            "f1_score": f1_score,
            "correct_predictions": sum(1 for item in outputs_dump if item["is_correct"]),
            "process_id": process_id,
            "gpu_id": gpu_id,
            "examples": outputs_dump
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"P{process_id}: Saved {len(outputs_dump)} SQuAD outputs to {output_path}")
        except Exception as e_save:
            logger.error(f"P{process_id}: Error saving SQuAD outputs: {e_save}")

    logger.info(f"P{process_id}(GPU{gpu_id}) - Final SQuAD: EM={em_score:.2f}%, F1={f1_score:.2f}% on {len(final_metric_predictions)} examples.")
    
    return {"SQuAD": f1_score, "SQuAD_exact_match": em_score, "SQuAD_f1": f1_score}

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test SQuAD")
    test_parser.add_argument("--model_name_test", type=str, default="meta-llama/Meta-Llama-3-8B")
    test_parser.add_argument("--dataset_split_test", type=str, default="validation[:100]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=4)
    test_parser.add_argument("--num_few_shot_test", type=int, default=2)
    test_parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum new tokens to generate")
    test_parser.add_argument("--save_outputs", action="store_true", help="Save detailed outputs to JSON file")
    test_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    
    squad_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="SQuADFileTest")
    logger.info(f"--- Standalone SQuAD Test: {squad_args.model_name_test} ({squad_args.num_few_shot_test}-shot) ---")
    
    squad_pipe, _ = initialize_model_pipeline(squad_args.model_name_test, target_device_id=0)
    if squad_pipe:
        squad_eval_args = {
            "pipe": squad_pipe,
            "tokenizer": squad_pipe.tokenizer,
            "model_name_for_logging": squad_args.model_name_test,
            "device": squad_pipe.device,
            "dataset_split": squad_args.dataset_split_test,
            "generation_batch_size": squad_args.gen_batch_size_test,
            "num_few_shot": squad_args.num_few_shot_test,
            "max_new_tokens": squad_args.max_new_tokens,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": squad_args.save_outputs,
            "resume": squad_args.resume
        }
        try:
            print(json.dumps(evaluate_squad(**squad_eval_args), indent=2))
        finally:
            cleanup_model_resources(squad_pipe, getattr(squad_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {squad_args.model_name_test} for SQuAD test.")