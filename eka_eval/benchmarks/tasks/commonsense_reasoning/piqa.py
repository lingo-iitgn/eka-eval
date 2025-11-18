# eka_eval/benchmarks/tasks/commonsense_reasoning/piqa.py
# FIXED VERSION - Matches lm-evaluation-harness methodology

import torch
import logging
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Any
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME_PIQA = "baber/piqa"
DEFAULT_SPLIT_PIQA = "validation"

def _compute_conditional_loglikelihood(model, tokenizer, context: str, continuation: str, device) -> float:
    """
    Compute log P(continuation | context) exactly as lm-eval-harness does.
    
    This is the CORRECT way to evaluate multiple-choice questions:
    1. Concatenate context + continuation
    2. Get logits for the full sequence
    3. Calculate log probability only for continuation tokens
    4. Normalize by length (for acc_norm metric)
    """
    # Tokenize context and continuation separately to know boundaries
    context_enc = tokenizer(context, add_special_tokens=True, return_tensors="pt")
    full_enc = tokenizer(context + continuation, add_special_tokens=True, return_tensors="pt")
    
    context_ids = context_enc.input_ids.to(device)
    full_ids = full_enc.input_ids.to(device)
    
    # Context length (where continuation starts)
    ctx_len = context_ids.shape[1]
    
    # Get model logits
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
    
    # Compute log probabilities for continuation tokens only
    # logits[i] predicts token[i+1], so we need logits[ctx_len-1:] to predict continuation
    continuation_logits = logits[0, ctx_len-1:-1, :]  # Exclude last logit (no next token)
    continuation_ids = full_ids[0, ctx_len:]  # The actual continuation tokens
    
    # Compute log softmax
    log_probs = F.log_softmax(continuation_logits, dim=-1)
    
    # Get log prob for each actual token
    token_log_probs = log_probs[range(len(continuation_ids)), continuation_ids]
    
    # Sum log probs (this is log P(continuation | context))
    total_log_prob = token_log_probs.sum().item()
    
    # Normalize by length (for acc_norm metric)
    normalized_log_prob = total_log_prob / len(continuation)
    
    return total_log_prob, normalized_log_prob

def evaluate_piqa(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    dataset_name: str = DEFAULT_DATASET_NAME_PIQA,
    dataset_split: str = DEFAULT_SPLIT_PIQA,
    results_dir: str = "results_output",
    process_id: int = 0,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate PIQA using likelihood scoring (acc_norm method).
    
    This matches lm-evaluation-harness exactly:
    - doc_to_text: "Question: {goal}\nAnswer:"
    - doc_to_choice: [sol1, sol2]
    - Select choice with highest normalized log-likelihood
    """
    logger.info(f"[P{process_id}] Starting PIQA evaluation: {model_name_for_logging}")
    logger.info(f"[P{process_id}] Using likelihood scoring (acc_norm method)")
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        logger.info(f"[P{process_id}] Loaded {len(dataset)} examples from {dataset_name}/{dataset_split}")
    except Exception as e:
        logger.error(f"Failed to load PIQA dataset: {e}")
        return {"PIQA": 0.0}
    
    # Get the actual model (not pipeline)
    model = pipe.model if hasattr(pipe, 'model') else pipe
    model.eval()
    
    correct_acc = 0
    correct_acc_norm = 0
    total = 0
    
    # Process each example
    for item in tqdm(dataset, desc=f"P{process_id} - Evaluating PIQA"):
        # Format exactly as lm-eval-harness
        goal = item['goal'].strip()
        context = f"Question: {goal}\nAnswer:"
        
        # Get choices
        sol1 = item['sol1']
        sol2 = item['sol2']
        label = item['label']  # 0 or 1
        
        # Compute log-likelihoods for both choices
        try:
            # Choice 0 (sol1)
            raw_0, norm_0 = _compute_conditional_loglikelihood(
                model, tokenizer, context, sol1, device
            )
            
            # Choice 1 (sol2)
            raw_1, norm_1 = _compute_conditional_loglikelihood(
                model, tokenizer, context, sol2, device
            )
            
            # Predictions
            pred_acc = 0 if raw_0 > raw_1 else 1
            pred_acc_norm = 0 if norm_0 > norm_1 else 1
            
            # Check correctness
            if pred_acc == label:
                correct_acc += 1
            if pred_acc_norm == label:
                correct_acc_norm += 1
            
            total += 1
            
        except Exception as e:
            logger.error(f"Error processing PIQA item: {e}")
            continue
    
    # Calculate accuracies
    acc = (correct_acc / total * 100) if total > 0 else 0.0
    acc_norm = (correct_acc_norm / total * 100) if total > 0 else 0.0
    
    logger.info(f"[P{process_id}] PIQA Results:")
    logger.info(f"  - Accuracy (acc): {acc:.2f}% ({correct_acc}/{total})")
    logger.info(f"  - Accuracy Normalized (acc_norm): {acc_norm:.2f}% ({correct_acc_norm}/{total})")
    
    # Return acc_norm as main metric (this is what lm-eval reports)
    return {
        "PIQA": acc_norm,
        "PIQA_acc": acc,
        "PIQA_acc_norm": acc_norm
    }