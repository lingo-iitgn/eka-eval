import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME_GPQA = "Idavidrein/gpqa"
DEFAULT_CONFIG_GPQA = "gpqa_main"
DEFAULT_SPLIT_GPQA = "test"
DEFAULT_MAX_NEW_TOKENS_GPQA = 256 #
DEFAULT_GENERATION_BATCH_SIZE_GPQA = 8
DEFAULT_NUM_FEWSHOT_GPQA = 3


FEWSHOT_EXAMPLES_GPQA = [
    {
        "question": "Which of the following is a primary function of mitochondria in eukaryotic cells?",
        "options": {
            "A": "Protein synthesis",
            "B": "Energy production through cellular respiration",
            "C": "Lipid storage",
            "D": "Waste elimination"
        },
        "answer_with_rationale": "Rationale: Mitochondria are known as the 'powerhouses' of the cell. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy, through the process of cellular respiration. Protein synthesis occurs in ribosomes, lipid storage in lipid droplets, and waste elimination in lysosomes. Therefore, energy production is the primary function. The correct answer is B."
    },
    {
        "question": "In Newtonian physics, what is the relationship between an object's mass (m), acceleration (a), and the net force (F) applied to it?",
        "options": {
            "A": "F = m / a",
            "B": "F = a / m",
            "C": "F = m * a",
            "D": "F = m * a^2"
        },
        "answer_with_rationale": "Rationale: Newton's Second Law of Motion states that the acceleration of an object is directly proportional to the net force acting upon the object and inversely proportional to its mass. This relationship is mathematically expressed as F = m * a. The other options represent incorrect formulas. The correct answer is C."
    },
    {
        "question": "What is the primary greenhouse gas responsible for the majority of Earth's natural greenhouse effect?",
        "options": {
            "A": "Methane (CH4)",
            "B": "Carbon Dioxide (CO2)",
            "C": "Nitrous Oxide (N2O)",
            "D": "Water Vapor (H2O)"
        },
        "answer_with_rationale": "Rationale: While CO2 and methane are significant greenhouse gases, water vapor is the most abundant and contributes the most to the natural greenhouse effect, accounting for about 60-70% of it. The other gases are less abundant but have higher global warming potentials per molecule. The question asks about the natural effect, where water vapor dominates. The correct answer is D."
    }
]

def _format_gpqa_prompt(question: str, options: Dict[str, str], fewshot_examples: List[Dict], num_few_shot: int) -> str:
    """Formats a multiple-choice prompt for the GPQA benchmark."""
    prompt = "Please answer the following multiple-choice question by providing a brief rationale and then stating the correct letter.\n\n"
    
    if num_few_shot > 0 and fewshot_examples:
        actual_fewshot = fewshot_examples[:num_few_shot]
        for ex in actual_fewshot:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"A) {ex['options']['A']}\n"
            prompt += f"B) {ex['options']['B']}\n"
            prompt += f"C) {ex['options']['C']}\n"
            prompt += f"D) {ex['options']['D']}\n"
            prompt += f"Answer: {ex['answer_with_rationale']}\n\n"
            
    prompt += f"Question: {question}\n"
    prompt += f"A) {options['A']}\n"
    prompt += f"B) {options['B']}\n"
    prompt += f"C) {options['C']}\n"
    prompt += f"D) {options['D']}\n"
    prompt += "Answer:"
    return prompt

def _extract_gpqa_answer(text: Optional[str]) -> Optional[str]:
    """Extracts the final letter answer (A, B, C, or D) from the generated text."""
    if text is None:
        return None

   
    match = re.search(r'The correct answer is\s+([A-D])|Answer:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return next((group for group in match.groups() if group is not None), None).upper()

    match_fallback = re.search(r'([A-D])[.\s]*$', text)
    if match_fallback:
        return match_fallback.group(1).upper()
        
    logger.debug(f"GPQA: Could not extract a valid answer (A, B, C, D) from text: '{text[-50:]}'")
    return None

def evaluate_gpqa(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_GPQA,
    dataset_config_name: str = DEFAULT_CONFIG_GPQA,
    dataset_split: str = DEFAULT_SPLIT_GPQA,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_GPQA,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_GPQA,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_GPQA,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting GPQA ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}/{dataset_config_name}")

    try:
        full_data = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"GPQA": 0.0, "error_message": f"DatasetLoadFailed GPQA: {e}"}
    logger.info(f"P{process_id}: Loaded GPQA '{dataset_name}/{dataset_config_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

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
        return {"GPQA": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} GPQA examples.")

    correct_predictions = 0
    total_evaluated = 0
    
    prompts_for_batch, original_items_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - GPQA Eval")):
        question = item_data.get('Question')
        options = {
            'A': item_data.get('A'), 'B': item_data.get('B'),
            'C': item_data.get('C'), 'D': item_data.get('D')
        }
        true_answer_letter = item_data.get('Correct Answer')

        if not all([question, true_answer_letter] + list(options.values())):
            logger.warning(f"P{process_id}: Skipping GPQA item due to missing data. Q: {str(question)[:50]}")
            continue

        prompt_text = _format_gpqa_prompt(question, options, FEWSHOT_EXAMPLES_GPQA, num_few_shot)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append({'true_answer_letter': true_answer_letter})

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": False
            }
            try:
                with torch.no_grad():
                    batch_raw_outputs = pipe(prompts_for_batch, **gen_config)

                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item_info = original_items_for_batch[k]
                    generated_text = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else ""
                    
                    pred_answer_letter = _extract_gpqa_answer(generated_text)
                    
                    logger.debug(f"GPQA Q: ...{question[-50:]}\nPred Raw: {generated_text[:100]}...\nPred Ans: {pred_answer_letter}, True Ans: {original_item_info['true_answer_letter']}")

                    if pred_answer_letter is not None and pred_answer_letter == original_item_info['true_answer_letter']:
                        correct_predictions += 1
                    total_evaluated += 1

            except Exception as e_batch_gpqa:
                logger.error(f"P{process_id}: Error in GPQA generation batch: {e_batch_gpqa}", exc_info=True)
                total_evaluated += len(prompts_for_batch)
            
            prompts_for_batch, original_items_for_batch = [], []

    accuracy_score = (correct_predictions / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final GPQA Accuracy: {accuracy_score:.2f}% ({correct_predictions}/{total_evaluated}).")
    return {"GPQA": accuracy_score}