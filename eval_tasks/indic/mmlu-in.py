# eka_eval/benchmarks/tasks/indic/boolq_in.py

import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate # Hugging Face evaluate library
import numpy as np
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

# --- Constants and Normalization Logic (can be kept within this module) ---
DEFAULT_DATASET_NAME = "sarvamai/boolq-indic" # Or your specific dataset if different
DEFAULT_TARGET_LANGUAGES = ["en", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
DEFAULT_SPLIT = 'validation'
DEFAULT_MAX_NEW_TOKENS = 10

# Normalization strings (as in your original script)
YES_STRINGS_ENGLISH = ["yes", "true", "correct", "affirmative"]
NO_STRINGS_ENGLISH = ["no", "false", "incorrect", "negative"]
YES_STRINGS_HINDI = ["हाँ", "हां", "सही", "सत्य"] # Ensure correct Hindi characters
NO_STRINGS_HINDI = ["नहीं", "नही", "गलत", "असत्य"] # Ensure correct Hindi characters

def format_prompt_for_boolq(passage: str, question: str) -> str:
    """Formats the prompt for the BoolQ task."""
    # Using the Hindi prompt from your example. This could be language-dependent if needed.
    return f"""निम्नलिखित गद्यांश को पढ़ें और प्रश्न का उत्तर 'हाँ' या 'नहीं' में दें।
गद्यांश:
{passage}

प्रश्न:
{question}

उत्तर (हाँ/नहीं):"""

def normalize_answer_to_bool_int(answer_text: Any) -> int:
    """
    Normalizes a given answer text (or boolean/int) to 1 (yes/true), 0 (no/false), or -1 (unknown/unparseable).
    """
    if answer_text is None:
        return -1
    if isinstance(answer_text, bool):
        return int(answer_text)
    if isinstance(answer_text, int):
        return 1 if answer_text == 1 else (0 if answer_text == 0 else -1)
    if not isinstance(answer_text, str):
        return -1

    text_original_case_stripped = answer_text.strip()
    text_lower_stripped = text_original_case_stripped.lower()

    # Check Hindi first due to its specific characters
    if text_original_case_stripped in YES_STRINGS_HINDI: return 1
    if text_original_case_stripped in NO_STRINGS_HINDI: return 0

    # Check English
    if text_lower_stripped in YES_STRINGS_ENGLISH: return 1
    if text_lower_stripped in NO_STRINGS_ENGLISH: return 0

    # Fallback: Check if any known Hindi substring is present (more lenient)
    for yes_hindi in YES_STRINGS_HINDI:
        if yes_hindi in text_original_case_stripped: return 1
    for no_hindi in NO_STRINGS_HINDI:
        if no_hindi in text_original_case_stripped: return 0

    # Fallback: Check if any known English substring is present
    for yes_eng in YES_STRINGS_ENGLISH:
        if yes_eng in text_lower_stripped: return 1
    for no_eng in NO_STRINGS_ENGLISH:
        if no_eng in text_lower_stripped: return 0

    return -1 # Unparseable


# --- Main Evaluation Function ---
def evaluate_boolq_in(
    pipe: Any, # The Hugging Face pipeline (model and tokenizer are within it)
    tokenizer: Any, # Tokenizer needed for decoding and prompt length
    model_name_for_logging: str, # For context in logs
    device: Any, # The device the model is on (e.g., torch.device('cuda:0'))
    # Task-specific parameters (can be passed from benchmark_config.py or have defaults)
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_languages: List[str] = None, # Allow override
    dataset_split: str = DEFAULT_SPLIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = 1, # From your original script
    do_sample: bool = False, # From your original script
    # Add other args passed by evaluation_worker if needed (e.g., process_id for checkpointing)
    **kwargs # To catch any other arguments from the worker
) -> Dict[str, float]:
    """
    Evaluates the model on the BoolQ-Indic benchmark.

    Args:
        pipe: The initialized Hugging Face text-generation pipeline.
        tokenizer: The tokenizer associated with the model.
        model_name_for_logging: Name of the model being evaluated.
        device: The torch device the model is on.
        dataset_name: Name of the BoolQ-Indic dataset on Hugging Face Hub.
        target_languages: List of language codes to evaluate.
        dataset_split: The dataset split to use (e.g., 'validation', 'test').
        max_new_tokens: Max new tokens to generate for the answer.
        num_beams: Number of beams for generation.
        do_sample: Whether to use sampling during generation.

    Returns:
        A dictionary containing scores, e.g.,
        {
            "BOOLQ-IN": overall_average_accuracy,
            "BOOLQ-IN_en": accuracy_en,
            "BOOLQ-IN_hi": accuracy_hi,
            ...
        }
    """
    logger.info(f"Starting BoolQ-Indic evaluation for model: {model_name_for_logging} on dataset: {dataset_name}")

    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    # Ensure the Hugging Face 'evaluate' metric is loaded
    try:
        accuracy_metric = evaluate.load("accuracy")
    except Exception as e:
        logger.error(f"Failed to load 'accuracy' metric from Hugging Face evaluate: {e}", exc_info=True)
        # Return a dictionary indicating failure or raise an error
        return {"BOOLQ-IN": 0.0, "error": "MetricLoadFailed"}


    language_accuracies: Dict[str, float | None] = {}
    all_individual_accuracies_list: List[float] = []
    # detailed_results_per_lang = {} # If you want to return raw predictions too

    try:
        # Load the full dataset for the specified split once
        logger.info(f"Loading dataset '{dataset_name}' split '{dataset_split}'...")
        full_dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True) # Added trust_remote_code
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}", exc_info=True)
        return {"BOOLQ-IN": 0.0, "error": "DatasetLoadFailed"}


    for lang_code in target_languages:
        logger.info(f"--- Evaluating Language: {lang_code.upper()} ---")
        predictions_normalized: List[int] = []
        references_normalized: List[int] = []
        # lang_detailed_results = []

        try:
            # Filter for the current language
            # The 'language' column name might vary; adjust if needed.
            # Your original script used example['language'] == lang_code
            lang_specific_dataset = full_dataset.filter(lambda example: example.get('language') == lang_code)

            if not lang_specific_dataset or len(lang_specific_dataset) == 0:
                logger.warning(f"No samples found for language '{lang_code}' in dataset '{dataset_name}'. Skipping.")
                language_accuracies[lang_code] = None # Or 0.0
                continue

            logger.info(f"Evaluating on {len(lang_specific_dataset)} samples for '{lang_code}'.")

            for example_idx, example in tqdm(enumerate(lang_specific_dataset), desc=f"Eval {lang_code.upper()}", total=len(lang_specific_dataset)):
                question = example.get("question", "")
                passage = example.get("passage", "")
                ground_truth_answer_str = example.get("answer", "") # This is 'yes'/'no' or bool

                if not question or not passage: # Ground truth can be missing for test sets sometimes
                    logger.warning(f"Skipping example {example_idx} for '{lang_code}' due to missing question or passage.")
                    # Decide how to handle: append -1 or skip? If skipped, counts change.
                    # For accuracy, if ground truth is also missing, it's problematic.
                    # Assuming ground_truth_answer_str is vital for reference:
                    if not ground_truth_answer_str:
                        logger.warning(f"  Also missing ground truth for example {example_idx}, lang '{lang_code}'.")
                        # Cannot compute accuracy for this sample.
                        continue # Skip this sample entirely from counts for this lang

                    predictions_normalized.append(-1) # Mark as unparseable prediction
                    references_normalized.append(normalize_answer_to_bool_int(ground_truth_answer_str))
                    # lang_detailed_results.append({"actual": ground_truth_answer_str, "predicted": None, "raw_generation": "[INPUT_ERROR]"})
                    continue

                prompt = format_prompt_for_boolq(passage, question)
                # Tokenize prompt to get its length for stripping later
                # The pipeline handles tokenization, but we need the prompt length
                # if we decode the full output.
                # Alternatively, set return_full_text=False in pipeline if available
                # and it only returns new tokens.

                # For `model.generate`, we need to tokenize manually.
                # If using `pipe`, it's more abstract.
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                prompt_token_length = inputs.input_ids.shape[1]
                generated_text = None

                try:
                    with torch.no_grad():
                        # Using pipe directly (preferred for abstraction)
                        # The `pipe` should be configured for text-generation
                        # If `pipe` is from AutoModelForCausalLM, it is not callable directly like a pipeline.
                        # We need to use `pipe.model.generate` if `pipe` is the model itself, or `pipe()` if it's a full pipeline.
                        # The `evaluation_worker` passes `model_pipeline` as `pipe`.
                        pipeline_output = pipe(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                            do_sample=do_sample,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            return_full_text=False # Important: only get generated part
                        )
                    if pipeline_output and isinstance(pipeline_output, list) and isinstance(pipeline_output[0], dict):
                        generated_text = pipeline_output[0].get('generated_text', "").strip()
                    else:
                        logger.warning(f"Unexpected output format from pipeline for example {example_idx}, lang '{lang_code}'. Output: {pipeline_output}")

                except Exception as e_gen:
                    logger.error(f"Error during generation for example {example_idx}, lang '{lang_code}': {e_gen}", exc_info=True)
                    generated_text = "[GENERATION_ERROR]" # Placeholder for failed generation

                predicted_normalized_answer = normalize_answer_to_bool_int(generated_text)
                reference_normalized_answer = normalize_answer_to_bool_int(ground_truth_answer_str)

                predictions_normalized.append(predicted_normalized_answer)
                references_normalized.append(reference_normalized_answer)
                # lang_detailed_results.append(...)

            # Calculate accuracy for the current language
            # Filter out pairs where reference is unparseable (-1), as accuracy cannot be computed.
            # Also filter where prediction is unparseable, as these are counted as incorrect by accuracy metric.
            valid_pairs_for_metric: List[Tuple[int, int]] = []
            for pred_val, ref_val in zip(predictions_normalized, references_normalized):
                if ref_val != -1: # Only consider if reference is valid (0 or 1)
                    # If prediction is -1 (unparseable), it will be treated as incorrect by accuracy metric
                    # when compared against a valid 0 or 1 reference.
                    valid_pairs_for_metric.append((pred_val if pred_val != -1 else -99, ref_val)) # Map -1 pred to a distinct wrong value

            if valid_pairs_for_metric:
                valid_predictions = [p for p, r in valid_pairs_for_metric]
                valid_references = [r for p, r in valid_pairs_for_metric]
                lang_accuracy = accuracy_metric.compute(predictions=valid_predictions, references=valid_references)['accuracy']
            elif predictions_normalized: # Some items processed but no valid pairs for metric (e.g., all refs were -1)
                lang_accuracy = 0.0
                logger.warning(f"No valid reference answers found for language '{lang_code}' to compute accuracy.")
            else: # No items processed for this language at all (should have been caught by len(lang_specific_dataset) == 0)
                lang_accuracy = None


            language_accuracies[lang_code] = lang_accuracy
            if lang_accuracy is not None:
                all_individual_accuracies_list.append(lang_accuracy)
            logger.info(f"  Accuracy for {lang_code.upper()}: {lang_accuracy:.4f} (on {len(valid_pairs_for_metric)} valid-reference samples)")
            # detailed_results_per_lang[lang_code] = {"accuracy": lang_accuracy, "results": lang_detailed_results}

        except Exception as e_lang:
            logger.error(f"CRITICAL error processing language {lang_code} for BoolQ-Indic: {e_lang}", exc_info=True)
            language_accuracies[lang_code] = None # Mark as error for this language

    # Calculate overall average accuracy
    valid_accuracies_for_avg = [acc for acc in all_individual_accuracies_list if acc is not None]
    overall_average_accuracy = np.mean(valid_accuracies_for_avg) if valid_accuracies_for_avg else 0.0

    logger.info(f"\nBoolQ-Indic Evaluation Summary for {model_name_for_logging}:")
    final_scores: Dict[str, float] = {"BOOLQ-IN": overall_average_accuracy} # Main score key
    for lang, acc in language_accuracies.items():
        logger.info(f"  - {lang.upper()}: {acc:.4f}" if acc is not None else f"  - {lang.upper()}: ERROR or No Data")
        final_scores[f"BOOLQ-IN_{lang}"] = acc if acc is not None else 0.0 # Store per-lang scores

    logger.info(f"Overall Average Accuracy (BOOLQ-IN): {overall_average_accuracy:.4f}")

    # The function should return a dictionary of scores.
    # The primary key should match the benchmark name in benchmark_config.py (e.g., "BOOLQ-IN")
    return final_scores

# Example of how this might be called (for standalone testing, not part of framework execution)
if __name__ == '__main__':
    # This block is for testing this script in isolation.
    # It won't be run when imported by the framework.
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources

    setup_logging(level=logging.INFO, worker_id="BoolQIN-Test")

    # --- Test Configuration (modify as needed) ---
    TEST_MODEL_NAME = "google/gemma-2b" # A smaller model for faster testing
    # TEST_MODEL_NAME = "sarvamai/sarvam-1" # Your original model

    logger.info(f"--- Standalone Test for boolq_in.py with model: {TEST_MODEL_NAME} ---")

    # 1. Initialize model and pipeline (mimicking evaluation_worker.py)
    # Assuming CUDA_VISIBLE_DEVICES is set if running on GPU, so logical_device_id is 0
    pipe_instance, params_str = initialize_model_pipeline(TEST_MODEL_NAME, target_device_id=0)

    if pipe_instance:
        logger.info(f"Model {TEST_MODEL_NAME} ({params_str}B) initialized for testing.")
        try:
            # Call the evaluation function
            results = evaluate_boolq_in(
                pipe=pipe_instance,
                tokenizer=pipe_instance.tokenizer,
                model_name_for_logging=TEST_MODEL_NAME,
                device=pipe_instance.device,
                # You can override other defaults for testing:
                # target_languages=["en", "hi"], # Test with fewer languages
                # dataset_split="validation[:10]" # If dataset supports slicing for quick test
            )
            logger.info("\n--- Test Evaluation Results ---")
            for key, value in results.items():
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error during standalone test evaluation: {e}", exc_info=True)
        finally:
            # 3. Cleanup
            logger.info("Cleaning up model resources after test.")
            model_ref = pipe_instance.model if hasattr(pipe_instance, 'model') else None
            cleanup_model_resources(pipe_instance, model_ref=model_ref)
    else:
        logger.error(f"Failed to initialize model {TEST_MODEL_NAME} for standalone test.")

    logger.info("--- Standalone test for boolq_in.py finished ---")