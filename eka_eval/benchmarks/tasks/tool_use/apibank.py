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
from typing import Dict, List, Any, Optional
import evaluate as hf_evaluate
import gc
from collections import defaultdict
import ast
import requests
from functools import partial

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_APIBANK = "liminghao1630/API-Bank"
DEFAULT_SPLIT_APIBANK = "test"
DEFAULT_MAX_NEW_TOKENS_APIBANK = 512
DEFAULT_CHECKPOINT_DIR_APIBANK = "checkpoints/apibank_checkpoints"

try:
    apibank_accuracy_metric = hf_evaluate.load("accuracy")
    logger.info("Accuracy metric for API-Bank loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load 'accuracy' metric for API-Bank: {e}. API-Bank will not run correctly.", exc_info=True)
    apibank_accuracy_metric = None


def _format_apibank_prompt(item: Dict, num_few_shot: int = 0, few_shot_examples: List[Dict] = None) -> str:
    """Format the API-Bank prompt with optional few-shot examples."""
    # Extract relevant fields from the API-Bank dataset
    user_query = item.get('query', item.get('user_query', '')).strip()
    api_docs = item.get('api_docs', item.get('apis', '')).strip()
    
    if num_few_shot > 0 and few_shot_examples:
        # Build few-shot prompt
        prompt_parts = [
            "You are an expert at calling APIs based on user queries and API documentation.",
            "Given a user query and available APIs, determine which API to call and with what parameters.",
            "Here are some examples:\n"
        ]
        
        # Add few-shot examples
        for i, example in enumerate(few_shot_examples[:num_few_shot]):
            example_query = example.get('query', example.get('user_query', '')).strip()
            example_api_call = example.get('api_call', example.get('answer', '')).strip()
            example_apis = example.get('api_docs', example.get('apis', '')).strip()
            
            prompt_parts.append(f"Example {i+1}:")
            prompt_parts.append(f"User Query: {example_query}")
            if example_apis:
                prompt_parts.append(f"Available APIs: {example_apis}")
            prompt_parts.append(f"API Call: {example_api_call}\n")
        
        prompt_parts.extend([
            "Now determine the appropriate API call for the following:",
            f"User Query: {user_query}"
        ])
        
        if api_docs:
            prompt_parts.append(f"Available APIs: {api_docs}")
        
        prompt_parts.append("API Call:")
        
        return "\n".join(prompt_parts)
    else:
        # Zero-shot prompt
        prompt_parts = [
            "You are an expert at calling APIs based on user queries and API documentation.",
            "Given a user query and available APIs, determine which API to call and with what parameters.\n",
            f"User Query: {user_query}"
        ]
        
        if api_docs:
            prompt_parts.append(f"Available APIs: {api_docs}")
        
        prompt_parts.append("API Call:")
        
        return "\n".join(prompt_parts)


def _extract_api_call_apibank(generated_text: str, prompt_text_sent_to_llm: str) -> str:
    """Extract the API call from the generated text for API-Bank."""
    completion_part = generated_text
    if generated_text.startswith(prompt_text_sent_to_llm):
        completion_part = generated_text[len(prompt_text_sent_to_llm):]
    
    completion_part = completion_part.strip()
    
    # Try to find API call patterns
    lines = completion_part.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            # Look for function call patterns
            if '(' in line and ')' in line:
                # Clean up the line
                cleaned_line = re.sub(r'^.*?(?=\w+\s*\()', '', line)
                if cleaned_line and '(' in cleaned_line:
                    return cleaned_line.strip()
            # Look for JSON-like API calls
            elif line.startswith('{') and '}' in line:
                return line
            # Look for direct API names or endpoints
            elif any(keyword in line.lower() for keyword in ['api', 'call', 'function', 'method']):
                return line
    
    # If no clear API call found, return the first substantial line
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            return line
    
    logger.debug(f"API-Bank: Could not extract API call from completion: '{completion_part[:50]}'")
    return completion_part[:100] if completion_part else ""


def _normalize_api_call_apibank(api_call: str) -> str:
    """Normalize API call for comparison in API-Bank."""
    if not api_call:
        return ""
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', api_call.strip())
    
    # Handle different quote styles
    normalized = re.sub(r'"([^"]*)"', r"'\1'", normalized)
    
    # Normalize parameter order for function calls
    func_match = re.match(r'(\w+)\s*\((.*)\)', normalized)
    if func_match:
        func_name = func_match.group(1)
        params = func_match.group(2)
        
        # Try to parse and sort parameters
        try:
            # Simple parameter sorting for consistent comparison
            param_parts = []
            current_param = ""
            paren_count = 0
            
            for char in params:
                if char == '(' or char == '[' or char == '{':
                    paren_count += 1
                elif char == ')' or char == ']' or char == '}':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    if current_param.strip():
                        param_parts.append(current_param.strip())
                    current_param = ""
                    continue
                current_param += char
            
            if current_param.strip():
                param_parts.append(current_param.strip())
            
            # Sort parameters by key if they look like key=value pairs
            sorted_params = []
            for param in param_parts:
                if '=' in param:
                    sorted_params.append(param)
                else:
                    sorted_params.append(param)
            
            sorted_params.sort()
            normalized = f"{func_name}({', '.join(sorted_params)})"
            
        except:
            # If parsing fails, just use the original
            pass
    
    return normalized.lower()


def _calculate_apibank_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate various accuracy metrics for API-Bank."""
    if len(predictions) != len(references):
        logger.warning(f"Mismatch in predictions and references length: {len(predictions)} vs {len(references)}")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    exact_matches = 0
    normalized_matches = 0
    function_name_matches = 0
    semantic_matches = 0
    
    for pred, ref in zip(predictions, references):
        pred_str = str(pred).strip()
        ref_str = str(ref).strip()
        
        # Exact match
        if pred_str == ref_str:
            exact_matches += 1
            normalized_matches += 1
            function_name_matches += 1
            semantic_matches += 1
        else:
            # Normalized match
            pred_norm = _normalize_api_call_apibank(pred_str)
            ref_norm = _normalize_api_call_apibank(ref_str)
            
            if pred_norm == ref_norm:
                normalized_matches += 1
                function_name_matches += 1
                semantic_matches += 1
            else:
                # Function name match
                pred_func = re.search(r'^(\w+)\s*\(', pred_norm)
                ref_func = re.search(r'^(\w+)\s*\(', ref_norm)
                
                if pred_func and ref_func and pred_func.group(1) == ref_func.group(1):
                    function_name_matches += 1
                    
                    # Check if parameters are semantically similar
                    pred_params = pred_func.group(0) if pred_func else ""
                    ref_params = ref_func.group(0) if ref_func else ""
                    
                    # Simple semantic check - if most key parameters match
                    pred_keys = set(re.findall(r'(\w+)\s*=', pred_params))
                    ref_keys = set(re.findall(r'(\w+)\s*=', ref_params))
                    
                    if pred_keys and ref_keys:
                        overlap = len(pred_keys.intersection(ref_keys))
                        total = len(ref_keys)
                        if total > 0 and overlap / total >= 0.7:  # 70% parameter overlap
                            semantic_matches += 1
    
    total = len(predictions)
    return {
        "exact_match": (exact_matches / total * 100) if total > 0 else 0.0,
        "normalized_match": (normalized_matches / total * 100) if total > 0 else 0.0,
        "function_name_match": (function_name_matches / total * 100) if total > 0 else 0.0,
        "semantic_match": (semantic_matches / total * 100) if total > 0 else 0.0,
        "total_examples": total
    }


def _safe_generate_apibank(
    pipe: Any,
    prompts: List[str],
    tokenizer: Any,
    max_new_tokens: int,
    num_retries: int = 2
) -> List[Dict[str, str]]:
    """Safe generation with error handling for API-Bank."""
    generation_params = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": True
    }
    
    for attempt in range(num_retries):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs = pipe(prompts, **generation_params)
            
            # Handle both single and batch outputs
            if not isinstance(outputs[0], list):
                outputs = [outputs]
            
            results = []
            for output_list in outputs:
                if output_list and isinstance(output_list, list) and len(output_list) > 0:
                    results.append(output_list[0])
                else:
                    results.append({"generated_text": ""})
            
            return results
            
        except Exception as e:
            logger.error(f"API-Bank Generation attempt {attempt + 1} failed: {e}", exc_info=True)
            if attempt < num_retries - 1:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("API-Bank: Retrying generation...")
            else:
                logger.error("API-Bank: Max retries reached for generation.")
                return [{"generated_text": p + "\nERROR"} for p in prompts]
    
    return [{"generated_text": p + "\nALL_RETRIES_FAILED"} for p in prompts]


def evaluate_apibank(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_APIBANK,
    dataset_split: str = DEFAULT_SPLIT_APIBANK,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_APIBANK,
    generation_batch_size: int = 1,
    num_few_shot: int = 0,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    resume: bool = False,
    **kwargs
) -> Dict[str, float]:
    """Evaluate model on API-Bank dataset."""
    
    if apibank_accuracy_metric is None:
        return {"API_Bank": 0.0, "error_message": "AccuracyMetricLoadFailed"}

    logger.info(f"Starting API-Bank evaluation for model: {model_name_for_logging}")
    logger.info(f"P{process_id}(GPU{gpu_id}): Params: split='{dataset_split}', few_shot={num_few_shot}, batch_size={generation_batch_size}")

    # Setup checkpoint and output paths
    checkpoint_dir = os.path.join(results_dir, DEFAULT_CHECKPOINT_DIR_APIBANK)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_hash = hashlib.md5(model_name_for_logging.encode()).hexdigest()[:8]
    checkpoint_file = os.path.join(checkpoint_dir, f"apibank_{model_hash}_p{process_id}.json")
    output_file = os.path.join(results_dir, f"apibank_detailed_{model_hash}_p{process_id}.json")

    # Load dataset
    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load API-Bank dataset: {e}")
        return {"API_Bank": 0.0, "error_message": f"DatasetLoadFailed: {e}"}

    logger.info(f"P{process_id}: Loaded API-Bank '{dataset_name}' (split: '{dataset_split}') with {len(full_data)} examples.")

    # Split data across GPUs if needed
    if num_gpus > 1:
        total_examples = len(full_data)
        examples_per_instance = total_examples // num_gpus
        start_idx = process_id * examples_per_instance
        end_idx = start_idx + examples_per_instance
        if process_id == num_gpus - 1:
            end_idx = total_examples
        dataset_subset = full_data.select(range(start_idx, end_idx))
    else:
        dataset_subset = full_data

    if len(dataset_subset) == 0:
        return {"API_Bank": 0.0}

    # Prepare few-shot examples if needed
    few_shot_examples = []
    if num_few_shot > 0:
        # Use a separate split or first few examples for few-shot
        try:
            few_shot_data = load_dataset(dataset_name, split="train", trust_remote_code=True)
            few_shot_examples = list(few_shot_data.select(range(min(num_few_shot * 3, len(few_shot_data)))))
        except:
            logger.warning("Could not load few-shot examples, using zero-shot evaluation")
            num_few_shot = 0

    # Resume from checkpoint if available
    completed_indices = set()
    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed_indices = set(checkpoint_data.get('completed_indices', []))
                logger.info(f"Resuming from checkpoint: {len(completed_indices)} examples already completed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    # Evaluation variables
    predictions, references = [], []
    detailed_results = []
    prompts_batch, items_batch, indices_batch = [], [], []

    for item_idx, item in enumerate(tqdm(dataset_subset, desc=f"P{process_id} - API-Bank Eval")):
        if item_idx in completed_indices:
            continue

        prompt = _format_apibank_prompt(item, num_few_shot, few_shot_examples)
        prompts_batch.append(prompt)
        items_batch.append(item)
        indices_batch.append(item_idx)

        # Process batch when full or at end
        if len(prompts_batch) == generation_batch_size or item_idx == len(dataset_subset) - 1:
            try:
                batch_outputs = _safe_generate_apibank(
                    pipe, prompts_batch, tokenizer, max_new_tokens
                )

                for k, output_dict in enumerate(batch_outputs):
                    original_item = items_batch[k]
                    generated_text = output_dict.get('generated_text', '')
                    
                    predicted_api = _extract_api_call_apibank(generated_text, prompts_batch[k])
                    # Try different possible field names for the reference
                    reference_api = (original_item.get('api_call', '') or 
                                   original_item.get('answer', '') or 
                                   original_item.get('ground_truth', '') or 
                                   original_item.get('target', '')).strip()
                    
                    predictions.append(predicted_api)
                    references.append(reference_api)
                    
                    # Store detailed results
                    if save_outputs:
                        detailed_results.append({
                            "index": indices_batch[k],
                            "user_query": original_item.get('query', original_item.get('user_query', '')),
                            "api_docs": original_item.get('api_docs', original_item.get('apis', '')),
                            "reference_api": reference_api,
                            "predicted_api": predicted_api,
                            "generated_text": generated_text,
                            "prompt": prompts_batch[k]
                        })
                    
                    completed_indices.add(indices_batch[k])

                # Save checkpoint periodically
                if len(completed_indices) % 50 == 0:
                    checkpoint_data = {
                        'completed_indices': list(completed_indices),
                        'total_examples': len(dataset_subset)
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f)

            except Exception as e:
                logger.error(f"P{process_id}: Error in API-Bank batch processing: {e}", exc_info=True)
                # Add empty predictions for failed batch
                for i in range(len(prompts_batch)):
                    predictions.append("")
                    ref_api = (items_batch[i].get('api_call', '') or 
                             items_batch[i].get('answer', '') or 
                             items_batch[i].get('ground_truth', '') or 
                             items_batch[i].get('target', '')).strip()
                    references.append(ref_api)

            # Clear batch
            prompts_batch, items_batch, indices_batch = [], [], []

    # Calculate final metrics
    if not predictions:
        return {"API_Bank": 0.0}

    try:
        accuracy_results = _calculate_apibank_accuracy(predictions, references)
        
        # Primary metric is semantic match for API-Bank (closer to human evaluation)
        primary_score = accuracy_results['semantic_match']
        
        logger.info(f"P{process_id}(GPU{gpu_id}) - API-Bank Results:")
        logger.info(f"  Exact Match: {accuracy_results['exact_match']:.2f}%")
        logger.info(f"  Normalized Match: {accuracy_results['normalized_match']:.2f}%")
        logger.info(f"  Function Name Match: {accuracy_results['function_name_match']:.2f}%")
        logger.info(f"  Semantic Match: {accuracy_results['semantic_match']:.2f}%")
        logger.info(f"  Total Examples: {accuracy_results['total_examples']}")

        # Save detailed outputs if requested
        if save_outputs and detailed_results:
            with open(output_file, 'w') as f:
                json.dump({
                    'model': model_name_for_logging,
                    'metrics': accuracy_results,
                    'detailed_results': detailed_results
                }, f, indent=2)
            logger.info(f"Detailed results saved to: {output_file}")

        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return {
            "API_Bank": primary_score,
            "API_Bank_exact": accuracy_results['exact_match'],
            "API_Bank_normalized": accuracy_results['normalized_match'],
            "API_Bank_function_name": accuracy_results['function_name_match'],
            "API_Bank_semantic": accuracy_results['semantic_match'],
            "API_Bank_total": accuracy_results['total_examples']
        }

    except Exception as e:
        logger.error(f"P{process_id}: Error computing API-Bank metrics: {e}", exc_info=True)
        return {"API_Bank": 0.0, "error_message": f"MetricComputationFailed: {e}"}


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test API-Bank")
    test_parser.add_argument("--model_name_test", type=str, default="meta-llama/Meta-Llama-3-8B")
    test_parser.add_argument("--dataset_split_test", type=str, default="test[:50]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=2)
    test_parser.add_argument("--num_few_shot_test", type=int, default=0)
    test_parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    test_parser.add_argument("--save_outputs", action="store_true", help="Save detailed outputs to JSON file")
    test_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    
    apibank_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="APIBankFileTest")
    logger.info(f"--- Standalone API-Bank Test: {apibank_args.model_name_test} ({apibank_args.num_few_shot_test}-shot) ---")
    
    apibank_pipe, _ = initialize_model_pipeline(apibank_args.model_name_test, target_device_id=0)
    if apibank_pipe:
        apibank_eval_args = {
            "pipe": apibank_pipe,
            "tokenizer": apibank_pipe.tokenizer,
            "model_name_for_logging": apibank_args.model_name_test,
            "device": apibank_pipe.device,
            "dataset_split": apibank_args.dataset_split_test,
            "generation_batch_size": apibank_args.gen_batch_size_test,
            "num_few_shot": apibank_args.num_few_shot_test,
            "max_new_tokens": apibank_args.max_new_tokens,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": apibank_args.save_outputs,
            "resume": apibank_args.resume
        }
        try:
            print(json.dumps(evaluate_apibank(**apibank_eval_args), indent=2))
        finally:
            cleanup_model_resources(apibank_pipe, getattr(apibank_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {apibank_args.model_name_test} for API-Bank test.")