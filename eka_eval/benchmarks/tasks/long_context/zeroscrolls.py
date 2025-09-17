import torch
from datasets import load_dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Any
import numpy as np
from rouge_score import rouge_scorer
import re

logger = logging.getLogger(__name__)

def _format_prompt(task: str, item: Dict) -> str:
    """Formats a prompt based on the ZeroSCROLLS sub-task."""
    input_text = item.get('input', '')
    if task == "gov_report":
        return f"Please read the following government report and summarize its key findings in a concise paragraph.\n\nReport:\n{input_text}\n\nSummary:"
    elif task == "summ_screen_fd":
        return f"Based on the following transcript, provide a summary of the conversation.\n\nTranscript:\n{input_text}\n\nSummary:"
    # Add other task prompts here as needed
    return input_text

def evaluate_zeroscrolls(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    sub_tasks_to_run: List[str],
    num_samples_per_task: int,
    max_new_tokens: int,
    **kwargs
) -> Dict[str, float]:

    logger.info(f"Starting ZeroSCROLLS for {model_name_for_logging} on sub-tasks: {sub_tasks_to_run}")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    all_scores = []

    for task in sub_tasks_to_run:
        try:
            dataset = load_dataset("tau/zero_scrolls", name=task, split="test", trust_remote_code=True)
            subset = dataset.select(range(min(num_samples_per_task, len(dataset))))
            logger.info(f"Evaluating sub-task '{task}' on {len(subset)} samples.")

            task_scores = []
            for item in tqdm(subset, desc=f"Evaluating {task}"):
                prompt = _format_prompt(task, item)
                reference = item.get('output', '')
                if not prompt or not reference:
                    continue

                with torch.no_grad():
                    result = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False, do_sample=False)
                prediction = result[0]['generated_text']
                
                # Calculate ROUGE-L F-measure
                rouge_scores = scorer.score(reference, prediction)
                task_scores.append(rouge_scores['rougeL'].fmeasure)
            
            if task_scores:
                avg_task_score = np.mean(task_scores) * 100
                logger.info(f"Average ROUGE-L for {task}: {avg_task_score:.2f}")
                all_scores.append(avg_task_score)

        except Exception as e:
            logger.error(f"Failed to process ZeroSCROLLS sub-task '{task}': {e}")
    
    if not all_scores:
        return {"ZeroSCROLLS": 0.0}

    # The final score is the average of the scores from the evaluated sub-tasks
    final_score = np.mean(all_scores)
    logger.info(f"Overall ZeroSCROLLS Average Score: {final_score:.2f}")

    return {"ZeroSCROLLS": final_score}