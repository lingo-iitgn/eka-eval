# eka_eval/benchmarks/tasks/indic/boolq_in/evaluator.py

import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate 
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from .prompts import format_prompt_for_boolq
from .normalizer import normalize_answer_to_bool_int

logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "sarvamai/boolq-indic"
DEFAULT_TARGET_LANGUAGES = ["en", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation'
DEFAULT_MAX_NEW_TOKENS = 10


def evaluate_boolq_in( 
    pipe: Any,
    tokenizer: Any, #
    model_name_for_logging: str,
    device: Any, 
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None, 
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = 1,
    do_sample: bool = False,
    process_id: int = 0,
    gpu_id: int = -1,    
    num_gpus: int = 1,   
    batch_size: int = 1, 
    **kwargs 
) -> Dict[str, float]:
    """
    Evaluates the model on the BoolQ-Indic benchmark.
    (Docstring content as before)
    """
    logger.info(f"Starting BoolQ-Indic evaluation for model: {model_name_for_logging} on dataset: {dataset_name} (Split: {dataset_split})")
    logger.debug(f"Task args received - Target Languages: {target_languages}, Max New Tokens: {max_new_tokens}, Num Beams: {num_beams}, Do Sample: {do_sample}")
    logger.debug(f"General args received (examples) - Process ID: {process_id}, GPU ID: {gpu_id}, Num GPUs: {num_gpus}, Batch Size (orchestrator): {batch_size}")


    if target_languages is None: # If not overridden by task_args
        target_languages = DEFAULT_TARGET_LANGUAGES
        logger.info(f"Using default target languages: {target_languages}")


    try:
        accuracy_metric = evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load 'accuracy' metric from Hugging Face evaluate: {e}", exc_info=True)
        return {"BOOLQ-IN": 0.0, "error_message": "MetricLoadFailed"}


    language_accuracies: Dict[str, float | None] = {}
    all_individual_accuracies_list: List[float] = []

    try:
        logger.info(f"Loading dataset '{dataset_name}' split '{dataset_split}'...")
        full_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"Dataset '{dataset_name}' loaded successfully with {len(full_dataset)} samples for split '{dataset_split}'.")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name} (split: {dataset_split}): {e}", exc_info=True)
        return {"BOOLQ-IN": 0.0, "error_message": f"DatasetLoadFailed: {dataset_name}"}


    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        predictions_normalized: List[int] = []
        references_normalized: List[int] = []

        try:
            lang_specific_dataset = full_dataset.filter(
                lambda example: example.get('language') == lang_code,
                load_from_cache_file=True
            )

            if not lang_specific_dataset or len(lang_specific_dataset) == 0:
                logger.warning(f"No samples found for language '{lang_code}' in dataset '{dataset_name}'. Skipping.")
                language_accuracies[lang_code] = None
                continue

            logger.info(f"Evaluating on {len(lang_specific_dataset)} samples for '{lang_code}'.")

            for example_idx, example in tqdm(enumerate(lang_specific_dataset), desc=f"Eval {lang_code.upper()}", total=len(lang_specific_dataset)):
                question = example.get("question", "")
                passage = example.get("passage", "")
                ground_truth_answer_str = example.get("answer", "")

                if not question or not passage:
                    logger.warning(f"Skipping example {example_idx} (lang '{lang_code}') due to missing question or passage.")
                    if not ground_truth_answer_str: # If GT is also missing, can't evaluate
                        logger.warning(f"  Ground truth also missing for example {example_idx} (lang '{lang_code}'). Sample fully skipped.")
                        continue
                
                    predictions_normalized.append(-1)
                    references_normalized.append(normalize_answer_to_bool_int(ground_truth_answer_str))
                    continue
                
                prompt = format_prompt_for_boolq(passage, question)
                if "Error: Invalid input" in prompt: # Check for error from prompt formatter
                    logger.error(f"Prompt formatting error for example {example_idx} (lang '{lang_code}'). Skipping generation.")
                    predictions_normalized.append(-1)
                    references_normalized.append(normalize_answer_to_bool_int(ground_truth_answer_str)) # Still need reference if GT exists
                    continue

                generated_text = None
                try:
                    with torch.no_grad():
                        # Using the passed Hugging Face pipeline `pipe`
                        pipeline_output = pipe(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                            do_sample=do_sample,
                            eos_token_id=tokenizer.eos_token_id, # Ensure tokenizer is from pipe or passed
                            pad_token_id=tokenizer.pad_token_id, # Ensure tokenizer is from pipe or passed
                            return_full_text=False # Get only the newly generated text
                        )
                    if pipeline_output and isinstance(pipeline_output, list) and pipeline_output[0] and isinstance(pipeline_output[0], dict):
                        generated_text = pipeline_output[0].get('generated_text', "").strip()
                        logger.debug(f"Lang {lang_code} Ex {example_idx} - Prompt: '{prompt[-100:]}', Raw Gen: '{generated_text}'")
                    else:
                        logger.warning(f"Unexpected output format from pipeline for example {example_idx}, lang '{lang_code}'. Output: {pipeline_output}")
                        generated_text = "[PIPELINE_OUTPUT_ERROR]"

                except Exception as e_gen:
                    logger.error(f"Error during generation for example {example_idx}, lang '{lang_code}': {e_gen}", exc_info=True)
                    generated_text = "[GENERATION_ERROR]"

                # Use the imported function
                predicted_normalized_answer = normalize_answer_to_bool_int(generated_text)
                reference_normalized_answer = normalize_answer_to_bool_int(ground_truth_answer_str)

                predictions_normalized.append(predicted_normalized_answer)
                references_normalized.append(reference_normalized_answer)

            # Calculate accuracy for the current language
            valid_pairs_for_metric: List[Tuple[int, int]] = []
            for pred_val, ref_val in zip(predictions_normalized, references_normalized):
                if ref_val != -1:
                    actual_pred_for_metric = pred_val if pred_val != -1 else -99 
                    valid_pairs_for_metric.append((actual_pred_for_metric, ref_val))

            lang_accuracy_value: float | None
            if valid_pairs_for_metric:
                valid_predictions = [p for p, r in valid_pairs_for_metric]
                valid_references = [r for p, r in valid_pairs_for_metric]
                lang_accuracy_value = accuracy_metric.compute(predictions=valid_predictions, references=valid_references)['accuracy']
            elif predictions_normalized: # Items were processed, but no valid reference answers found
                lang_accuracy_value = 0.0 # Or None, depending on how you want to treat this
                logger.warning(f"No valid reference answers found for language '{lang_code}' to compute accuracy (all refs were unparseable or samples skipped). Setting accuracy to 0.0.")
            else: # No items were processed at all for this language
                lang_accuracy_value = None
                logger.warning(f"No items processed for language '{lang_code}'. Accuracy not computed.")


            language_accuracies[lang_code] = lang_accuracy_value
            if lang_accuracy_value is not None:
                all_individual_accuracies_list.append(lang_accuracy_value)
            logger.info(f"  Accuracy for {lang_code.upper()}: {lang_accuracy_value:.4f} (on {len(valid_pairs_for_metric)} valid-reference samples)")

        except Exception as e_lang_processing: # Catch errors during processing of a specific language
            logger.error(f"CRITICAL error processing language {lang_code} for BoolQ-Indic: {e_lang_processing}", exc_info=True)
            language_accuracies[lang_code] = None

   
    valid_accuracies_for_avg = [acc for acc in all_individual_accuracies_list if acc is not None]
    overall_average_accuracy = np.mean(valid_accuracies_for_avg) if valid_accuracies_for_avg else 0.0 # Default to 0.0 if no valid accuracies

    logger.info(f"\nBoolQ-Indic Evaluation Summary for {model_name_for_logging}:")
    final_scores_dict: Dict[str, float] = {"BOOLQ-IN": float(overall_average_accuracy)} # Main score key, ensure float
    for lang, acc_val in language_accuracies.items():
        score_to_report = acc_val if acc_val is not None else 0.0 # Report 0.0 for errored/no-data languages
        logger.info(f"  - {lang.upper()}: {score_to_report:.4f}" + (" (Error or No Data)" if acc_val is None else ""))
        final_scores_dict[f"BOOLQ-IN_{lang}"] = float(score_to_report)

    logger.info(f"Overall Average Accuracy (BOOLQ-IN): {overall_average_accuracy:.4f} across {len(valid_accuracies_for_avg)} languages with valid scores.")

    return final_scores_dict

# Standalone test block (optional but good for debugging this specific benchmark)
if __name__ == '__main__':
    # This setup is only for when running this file directly for testing
    from eka_eval.utils.logging_setup import setup_logging
    # Need to adjust path if model_loader is one level up from eka_eval.core
    # Assuming this script is in eka_eval/benchmarks/tasks/indic/boolq_in/
    # We need to add the project root to sys.path for standalone testing to find eka_eval.core
    import sys
    import os
    # Determine the project root: goes up 4 levels from .../indic/boolq_in/evaluator.py to project root
    # eka_eval (project root) / eka_eval (package) / benchmarks / tasks / indic / boolq_in / evaluator.py
    # More robust way:
    current_script_path = os.path.abspath(__file__)
    # boolq_in_dir -> indic_dir -> tasks_dir -> benchmarks_dir -> eka_eval_pkg_dir -> project_root
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)


    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources

    setup_logging(level=logging.DEBUG, worker_id="BoolQIN-EvalTest") # Use DEBUG for more verbose test output
    TEST_MODEL_NAME = "google/gemma-2b" # Use a small, accessible model for testing
    logger.info(f"--- Standalone Test for boolq_in/evaluator.py with model: {TEST_MODEL_NAME} ---")

    # Mimic how the worker would call it, including task_args
    task_args_for_test = {
        "dataset_name": "sarvamai/boolq-indic",
        "target_languages": ["en", "hi"], # Short list for faster testing
        "dataset_split": "validation", # Consider "validation[:10]" if dataset supports it and for speed
        "max_new_tokens": 10,
        "num_beams": 1,
        "do_sample": False
    }

    # General args that would come from the worker
    general_args_for_test = {
        "model_name_for_logging": TEST_MODEL_NAME,
        "process_id": 999, # Dummy value
        "gpu_id": 0,       # Dummy value (if testing on GPU, ensure CUDA_VISIBLE_DEVICES is set)
        "num_gpus": 1,     # Dummy value
        "batch_size": 1    # Dummy value
    }


    pipe_instance, params_str = initialize_model_pipeline(TEST_MODEL_NAME, target_device_id=0) # Assumes GPU 0 or CPU
    if pipe_instance:
        logger.info(f"Model {TEST_MODEL_NAME} ({params_str}B) initialized for testing. Device: {pipe_instance.device}")
        try:
            # Combine general args and task-specific args for the call
            all_eval_args = {
                "pipe": pipe_instance,
                "tokenizer": pipe_instance.tokenizer, # Pass tokenizer explicitly
                "device": pipe_instance.device,
                **general_args_for_test,
                **task_args_for_test
            }
            results = evaluate_boolq_in(**all_eval_args)

            logger.info("\n--- Standalone Test Evaluation Results ---")
            if results:
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("Evaluation returned no results or an error occurred.")

        except Exception as e:
            logger.error(f"Error during standalone test evaluation run: {e}", exc_info=True)
        finally:
            logger.info("Cleaning up model resources after test.")
            model_ref = pipe_instance.model if hasattr(pipe_instance, 'model') else None
            cleanup_model_resources(pipe_instance, model_ref=model_ref)
    else:
        logger.error(f"Failed to initialize model {TEST_MODEL_NAME} for standalone test.")

    logger.info("--- Standalone test for boolq_in/evaluator.py finished ---")