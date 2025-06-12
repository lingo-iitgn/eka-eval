import torch
import re
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_MATH = "hendrycks/competition_math"
DEFAULT_SPLIT_MATH = "test"
DEFAULT_MAX_NEW_TOKENS_MATH = 384
DEFAULT_GENERATION_BATCH_SIZE_MATH = 4
DEFAULT_NUM_FEWSHOT_MATH = 4

FEWSHOT_EXAMPLES_MATH_DEFAULT_SET = [
    {
        "subject": "Prealgebra",
        "question": "If $5x - 3 = 12$, what is the value of $5x + 3$?",
        "solution": "Adding 6 to both sides of $5x - 3 = 12$ gives $5x - 3 + 6 = 12 + 6$. Simplifying both sides gives $5x + 3 = \\boxed{{18}}$.\nFinal answer: 18"
    },
    {
        "subject": "Prealgebra",
        "question": "Alice wants to buy $3$ pounds of veal at the grocery store, but the scales at the store only show weight in kilograms. If one kilogram is $2.20$ pounds, how many kilograms of veal should Alice buy? (You may use a calculator on this problem; answer to the nearest hundredth.)",
        "solution": "Since Alice wants to buy $3$ pounds of veal, we multiply the quantity of $3$ pounds by the conversion factor $\\frac{{1\\ \\text{{kg}}}}{{2.20\\ \\text{{lb}}}}$ to obtain $3\\ \\text{{lb}} \\cdot \\frac{{1\\ \\text{{kg}}}}{{2.20\\ \\text{{lb}}}} \\approx \\boxed{{1.36}}\\ \\text{{kg}}$.\nFinal answer: 1.36"
    },
    {
        "subject": "Number Theory",
        "question": "One morning each member of Angela's family drank an 8-ounce mixture of coffee with milk. The amounts of coffee and milk varied from cup to cup, but were never zero. Angela drank a quarter of the total amount of milk and a sixth of the total amount of coffee. How many people are in the family?",
        "solution": "Suppose that the whole family drank $x$ cups of milk and $y$ cups of coffee. Let $n$ denote the number of people in the family. The information given implies that $\\frac{{x}}{{4}} + \\frac{{y}}{{6}} = \\frac{{x + y}}{{n}}$. This leads to \n\\[\n3x(n - 4) = 2y(6 - n).\n\\]\nSince $x$ and $y$ are positive, the only positive integer $n$ for which both sides have the same sign is $n = \\boxed{{5}}$.\nFinal answer: 5"
    },
    {
        "subject": "Precalculus",
        "question": "Simplify\n\\[\n\\frac{{\\sin^4 x + \\cos^4 x - 1}}{{\\sin^6 x + \\cos^6 x - 1}}.\n\\]",
        "solution": "Let $p = \\sin x \\cos x$. We know that $\\sin^2 x + \\cos^2 x = 1$. Squaring both sides, we get\n\\[\n\\sin^4 x + 2 \\sin^2 x \\cos^2 x + \\cos^4 x = 1.\n\\]\nHence,\n\\[\n\\sin^4 x + \\cos^4 x = 1 - 2p^2.\n\\]\nThen, \n\\[\n(\\sin^2 x + \\cos^2 x)(\\sin^4 x + \\cos^4 x) = 1 - 2p^2.\n\\]\nExpanding, we get\n\\[\n\\sin^6 x + \\sin^2 x \\cos^4 x + \\cos^2 x \\sin^4 x + \\cos^6 x = 1 - 2p^2.\n\\]\nHence,\n\\[\n\\begin{{aligned}}\n\\sin^6 x + \\cos^6 x &= 1 - 2p^2 - (\\sin^2 x \\cos^4 x + \\cos^2 x \\sin^4 x) \\\\\n&= 1 - 2p^2 - \\sin^2 x \\cos^2 x (\\sin^2 x + \\cos^2 x) \\\\\n&= 1 - 3p^2.\n\\end{{aligned}}\n\\]\nTherefore,\n\\[\n\\frac{{\\sin^4 x + \\cos^4 x - 1}}{{\\sin^6 x + \\cos^6 x - 1}} = \\frac{{-2p^2}}{{-3p^2}} = \\boxed{{\\frac{{2}}{{3}}}}.\n\\]\nFinal answer: $\\frac{{2}}{{3}}$"
    }
]

def _format_math_prompt(question: str, subject: str, fewshot_examples: List[Dict], num_few_shot: int) -> str:
    prompt = f"Solve the following math problem from the subject: {subject}.\nThink stepwise, show your work, and put the final answer within \\boxed{{}} and also as 'Final answer: <answer>'.\n"
    if num_few_shot > 0 and fewshot_examples:
        prompt += "\nHere are some examples:\n"
        actual_fewshot = fewshot_examples[:num_few_shot]
        for i, ex in enumerate(actual_fewshot):
            prompt += f"\nExample {i+1}:\nsubject: {ex['subject']}\nquestion: {ex['question']}\nsolution: {ex['solution']}\n"
    prompt += f"\nNow, solve this problem:\nsubject: {subject}\nquestion: {question}\nsolution:"
    return prompt

def _extract_math_final_answer(text: Optional[str]) -> Optional[str]:
    if text is None: return None
    match_boxed = re.search(r'\\boxed\{([\s\S]*?)\}', text)
    if match_boxed:
        ans_str = match_boxed.group(1).strip()
        ans_str = ans_str.replace("\\%", "%").replace("\\$", "$")
        try: return str(float(ans_str))
        except ValueError: return ans_str
    match_final_answer = re.search(r'[Ff]inal\s*answer\s*:\s*([\s\S]+)', text)
    if match_final_answer:
        ans_str = match_final_answer.group(1).strip()
        if ans_str.endswith('.'): ans_str = ans_str[:-1].strip()
        try: return str(float(ans_str))
        except ValueError: return ans_str
    logger.debug(f"MATH: Could not extract final answer from: '{text[-100:]}'")
    return None

def evaluate_math(
    pipe: Any, tokenizer: Any, model_name_for_logging: str, device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_MATH,
    dataset_split: str = DEFAULT_SPLIT_MATH,
    num_few_shot: int = DEFAULT_NUM_FEWSHOT_MATH,
    few_shot_examples_list: Optional[List[Dict]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MATH,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE_MATH,
    process_id: int = 0, gpu_id: int = 0, num_gpus: int = 1,
    results_dir: str = "results_output", **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting Hendrycks MATH ({num_few_shot}-shot): {model_name_for_logging} on {dataset_name}")

    if few_shot_examples_list is None:
        few_shot_examples_list = FEWSHOT_EXAMPLES_MATH_DEFAULT_SET

    try:
        full_data = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    except Exception as e:
        return {"MATH": 0.0, "error_message": f"DatasetLoadFailed MATH: {e}"}
    logger.info(f"P{process_id}: Loaded MATH '{dataset_name}' ({len(full_data)} examples) for split '{dataset_split}'.")

    if num_gpus > 1:
        total = len(full_data); per_gpu = total // num_gpus
        start, end = process_id * per_gpu, (process_id + 1) * per_gpu
        if process_id == num_gpus - 1: end = total
        subset_to_process = full_data.select(range(start, end))
    else:
        subset_to_process = full_data
    if len(subset_to_process) == 0: return {"MATH": 0.0}
    logger.info(f"P{process_id}: Processing {len(subset_to_process)} MATH examples.")

    correct_predictions = 0
    total_evaluated = 0
    prompts_for_batch, original_items_for_batch = [], []

    for item_idx, item_data in enumerate(tqdm(subset_to_process, desc=f"P{process_id} - MATH Eval")):
        question = item_data.get('problem')
        subject = item_data.get('type', item_data.get('subject', 'Unknown'))
        true_answer_full_solution = item_data.get('solution')

        if not question or not true_answer_full_solution:
            logger.warning(f"P{process_id}: Skipping MATH item due to missing data. Q: {str(question)[:50]}")
            continue
        
        true_final_answer_str = _extract_math_final_answer(true_answer_full_solution)
        if true_final_answer_str is None:
            logger.warning(f"P{process_id}: Could not extract ground truth answer for MATH item. Q: {str(question)[:50]}, Sol: {str(true_answer_full_solution)[:100]}")
            continue

        prompt_text = _format_math_prompt(question, subject, few_shot_examples_list, num_few_shot)
        prompts_for_batch.append(prompt_text)
        original_items_for_batch.append({'true_final_answer_str': true_final_answer_str, 'question': question})

        if len(prompts_for_batch) == generation_batch_size or item_idx == len(subset_to_process) - 1:
            gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "stop_sequences": ["\n\nProblem:", "\n\nsubject:", "Final answer is often enough"],
                "return_full_text": False
            }
            try:
                with torch.no_grad(): batch_raw_outputs = pipe(prompts_for_batch, **gen_config)
                for k, raw_out_list in enumerate(batch_raw_outputs):
                    original_item_info = original_items_for_batch[k]
                    raw_gen_solution = raw_out_list[0]['generated_text'] if raw_out_list and raw_out_list[0] else "#GenFail"
                    pred_final_answer_str = _extract_math_final_answer("solution: " + raw_gen_solution)
                    
                    logger.debug(f"MATH Q: ...{original_item_info['question'][-50:]}\nPred Raw Sol: {raw_gen_solution[:100]}...\nPred Final Ans: {pred_final_answer_str}, True Final Ans: {original_item_info['true_final_answer_str']}")

                    if pred_final_answer_str is not None and pred_final_answer_str == original_item_info['true_final_answer_str']:
                        correct_predictions += 1
                    total_evaluated += 1
            except Exception as e_batch_math:
                logger.error(f"P{process_id}: Error in MATH gen batch: {e_batch_math}", exc_info=True)
                total_evaluated += len(prompts_for_batch)
            prompts_for_batch, original_items_for_batch = [], []

    accuracy_score = (correct_predictions / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    logger.info(f"P{process_id}(GPU{gpu_id}) - Final Hendrycks MATH Accuracy: {accuracy_score:.2f}% ({correct_predictions}/{total_evaluated}).")
    return {"MATH": accuracy_score}

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path: sys.path.insert(0, project_root_for_test)
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    test_parser = argparse.ArgumentParser(description="Standalone Test Hendrycks MATH")
    test_parser.add_argument("--model_name_test", type=str, default="gpt2")
    test_parser.add_argument("--dataset_split_test", type=str, default="test[:5]")
    test_parser.add_argument("--gen_batch_size_test", type=int, default=1)
    test_parser.add_argument("--num_few_shot_test", type=int, default=2)

    math_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="MATHFileTest")
    logger.info(f"--- Standalone MATH Test: {math_args.model_name_test} ({math_args.num_few_shot_test}-shot) ---")
    
    math_pipe, _ = initialize_model_pipeline(math_args.model_name_test, target_device_id=0)
    if math_pipe:
        math_eval_args = {
            "pipe": math_pipe, "tokenizer": math_pipe.tokenizer, "model_name_for_logging": math_args.model_name_test,
            "device": math_pipe.device, "dataset_split": math_args.dataset_split_test,
            "num_few_shot": math_args.num_few_shot_test,
            "generation_batch_size": math_args.gen_batch_size_test,
            "process_id": 0, "gpu_id": 0, "num_gpus": 1
        }
        try: print(json.dumps(evaluate_math(**math_eval_args), indent=2))
        finally: cleanup_model_resources(math_pipe, getattr(math_pipe, 'model', None))
