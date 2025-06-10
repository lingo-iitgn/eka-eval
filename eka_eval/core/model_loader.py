# eka_eval/core/model_loader.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import logging
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)

def initialize_model_pipeline(
    model_name_or_path: str,
    target_device_id: int = 0,
    trust_remote_code: bool = True,
) -> Tuple[Optional[Any], str]:
    """
    Initialize Hugging Face model and pipeline for text generation.
    """
    logger.info(f"Initializing model: {model_name_or_path} on device_id: {target_device_id}")

    if torch.cuda.is_available():
        device_map_arg = {'': f'cuda:{target_device_id}'}
        target_dtype = torch.bfloat16
        logger.info(f"CUDA available. Using device_map: {device_map_arg}, dtype: {target_dtype}")
    else:
        device_map_arg = "cpu"
        target_dtype = torch.float32
        logger.info("CUDA not available. Using CPU.")

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side='left',
            trust_remote_code=trust_remote_code
        )
        logger.info(f"Tokenizer loaded for {model_name_or_path}.")
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}", exc_info=True)
        return None, 'N/A'

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
        else:
            logger.warning("No pad_token_id or eos_token_id. Adding default pad token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    num_added_tokens = 0
    special_tokens_to_add_internal = ["[END]"]
    if special_tokens_to_add_internal[0] not in tokenizer.get_vocab():
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add_internal})
        if num_added_tokens > 0:
            logger.info(f"Added special token(s): {special_tokens_to_add_internal}")

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=target_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit quantization configured for GPU.")

    model = None
    model_load_args = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map_arg,
        "torch_dtype": target_dtype,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True
    }
    if quantization_config:
        model_load_args["quantization_config"] = quantization_config
        logger.info(f"Loading model {model_name_or_path} with quantization.")
    elif not torch.cuda.is_available():
        logger.info(f"Loading model {model_name_or_path} on CPU.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_load_args
        )
        logger.info(f"Model {model_name_or_path} loaded.")
    except Exception as e:
        logger.warning(f"Model load failed: {e}", exc_info=True)
        if torch.cuda.is_available() and "quantization_config" in model_load_args:
            logger.info("Retrying model load without quantization_config.")
            del model_load_args["quantization_config"]
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    **model_load_args
                )
                logger.info(f"Model {model_name_or_path} loaded on retry.")
            except Exception as e2:
                logger.error(f"Model load failed on retry: {e2}", exc_info=True)
                return None, 'N/A'
        else:
            logger.error(f"Model load failed: {e}", exc_info=True)
            return None, 'N/A'

    if model and num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings: {len(tokenizer)}.")

    param_count_str = 'N/A'
    if model:
        try:
            total_params = sum(p.numel() for p in model.parameters())
            model_name_lower = model_name_or_path.lower()
            if "gemma-2b" in model_name_lower or "gemma_2b" in model_name_lower:
                param_count_str = "2.00"
            elif "llama-7b" in model_name_lower:
                param_count_str = "7.00"
            elif total_params > 0:
                param_count_str = f"{total_params / 1_000_000_000:.2f}"
            else:
                param_count_str = "0.00"
            logger.info(f"Model parameter count: {param_count_str}B")
        except Exception as e:
            logger.warning(f"Parameter count failed: {e}", exc_info=True)

    hf_pipeline = None
    if model and tokenizer:
        try:
            hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=target_dtype
            )
            logger.info(f"Pipeline created for {model_name_or_path}.")
        except Exception as e:
            logger.error(f"Pipeline creation failed: {e}", exc_info=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None, param_count_str

    return hf_pipeline, param_count_str

def cleanup_model_resources(pipeline_to_clean: Optional[Any], model_ref: Optional[Any] = None):
    """
    Clean up model and pipeline resources.
    """
    logger.info("Cleaning up model and pipeline resources...")
    cleaned_something = False
    try:
        model_to_delete = model_ref
        if hasattr(pipeline_to_clean, 'model') and pipeline_to_clean.model:
            if model_to_delete and model_to_delete is not pipeline_to_clean.model:
                logger.warning("Both pipeline.model and model_ref provided. Cleaning both.")
            if not model_to_delete:
                model_to_delete = pipeline_to_clean.model

        if model_to_delete:
            del model_to_delete
            logger.info("Model object deleted.")
            cleaned_something = True

        if pipeline_to_clean:
            del pipeline_to_clean
            logger.info("Pipeline object deleted.")
            cleaned_something = True

        if cleaned_something:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GC collected and CUDA cache emptied.")
            else:
                logger.info("GC collected (CPU mode).")
        else:
            logger.info("No model or pipeline to clean.")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
