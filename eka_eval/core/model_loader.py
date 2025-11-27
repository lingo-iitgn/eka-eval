# eka_eval/core/model_loader.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import logging
from typing import Tuple, Optional, Any, Union

from .api_loader import (
    initialize_api_model_pipeline, 
    cleanup_api_model_resources, 
    get_available_api_models,
    get_api_key_from_env,
    set_api_key_env
)

logger = logging.getLogger(__name__)

def initialize_model_pipeline(
    model_name_or_path: str,
    target_device_id: int = 0,
    trust_remote_code: bool = True,
    is_api_model: bool = False,
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """
    Initialize model pipeline for both local and API models.
    
    Args:
        model_name_or_path: Model name/path or API model identifier
        target_device_id: Target device ID for local models
        trust_remote_code: Whether to trust remote code for local models
        is_api_model: Whether this is an API model
        api_provider: API provider ('openai', 'gemini', 'claude')
        api_key: API key for authentication
    
    Returns:
        Tuple of (pipeline, parameter_count_string)
    """
    
    if is_api_model:
        return _initialize_api_model(model_name_or_path, api_provider, api_key)
    else:
        return _initialize_local_model(model_name_or_path, target_device_id, trust_remote_code)

def _initialize_api_model(
    model_name: str, 
    api_provider: str, 
    api_key: str
) -> Tuple[Optional[Any], str]:
    """Initialize API model pipeline"""
    logger.info(f"Initializing API model: {api_provider}/{model_name}")
    
    try:
        # Initialize API pipeline
        api_pipeline = initialize_api_model_pipeline(api_provider, model_name, api_key)
        
        if api_pipeline is None:
            logger.error(f"Failed to initialize API model: {api_provider}/{model_name}")
            return None, 'N/A'
        
        # Estimate parameter count based on model name (for display purposes)
        param_count_str = _estimate_api_model_size(model_name)
        
        logger.info(f"Successfully initialized API model: {api_provider}/{model_name}")
        return api_pipeline, param_count_str
        
    except Exception as e:
        logger.error(f"Error initializing API model {api_provider}/{model_name}: {e}", exc_info=True)
        return None, 'N/A'

def _initialize_local_model(
    model_name_or_path: str,
    target_device_id: int,
    trust_remote_code: bool
) -> Tuple[Optional[Any], str]:
    """Initialize local model pipeline (original implementation)"""
    logger.info(f"Initializing local model: {model_name_or_path} on device_id: {target_device_id}")

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
            # Calculate physical count (will be lower if quantized)
            total_params_physical = sum(p.numel() for p in model.parameters())
            
            model_name_lower = model_name_or_path.lower()
            
            # --- UPDATED HEURISTIC LOGIC ---
            if "gemma-2b" in model_name_lower or "2b" in model_name_lower:
                param_count_str = "2.00"
            elif "llama-7b" in model_name_lower or "7b" in model_name_lower:
                param_count_str = "7.00"
            elif "8b" in model_name_lower: 
                param_count_str = "8.00"
            elif "3b" in model_name_lower:  
                param_count_str = "3.00"
            elif "1b" in model_name_lower:  
                param_count_str = "1.00"
            elif "70b" in model_name_lower:
                param_count_str = "70.00"
          
            elif total_params_physical > 0:
                
                is_4bit = getattr(model, 'is_loaded_in_4bit', False)
                
                if is_4bit:
                     param_count_str = f"{total_params_physical / 1_000_000_000:.2f} (Quantized)"
                else:
                    param_count_str = f"{total_params_physical / 1_000_000_000:.2f}"
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

def _estimate_api_model_size(model_name: str) -> str:
    """Estimate parameter count for API models based on model name"""
    model_name_lower = model_name.lower()
    
    # OpenAI models
    if "gpt-4" in model_name_lower:
        if "turbo" in model_name_lower:
            return "175.00"  # Estimated
        return "1750.00"  # GPT-4 estimated
    elif "gpt-3.5" in model_name_lower:
        return "175.00"
    elif "gpt-4o" in model_name_lower:
        if "mini" in model_name_lower:
            return "8.00"
        return "200.00"  # Estimated
    
    # Gemini models
    elif "gemini-pro" in model_name_lower:
        return "540.00"  # Estimated
    elif "gemini-1.5" in model_name_lower:
        if "flash" in model_name_lower:
            return "20.00"  # Smaller variant
        return "1000.00"  # Pro variant estimated
    
    # Claude models
    elif "claude-3" in model_name_lower:
        if "haiku" in model_name_lower:
            return "20.00"  # Smallest Claude 3
        elif "sonnet" in model_name_lower:
            return "200.00"  # Medium Claude 3
        elif "opus" in model_name_lower:
            return "400.00"  # Largest Claude 3
    
    return "Unknown"

def cleanup_model_resources(pipeline_to_clean: Optional[Any], model_ref: Optional[Any] = None):
    """
    Clean up model and pipeline resources for both local and API models.
    """
    logger.info("Cleaning up model and pipeline resources...")
    
    # Check if this is an API model
    if hasattr(pipeline_to_clean, 'device') and pipeline_to_clean.device == "api":
        cleanup_api_model_resources(pipeline_to_clean)
        return
    
    # Original cleanup logic for local models
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

def get_model_selection_interface():
    """
    Interactive interface for model selection (local vs API)
    
    Returns:
        Tuple of (model_info_dict, is_api_model)
    """
    print("\n--- Model Selection ---")
    print("1. Local Model (Hugging Face/Local Path)")
    print("2. API Model (OpenAI/Gemini/Claude)")
    
    choice = input("Enter model type (1 or 2): ").strip()
    
    if choice == "1":
        # Local model selection (existing logic)
        source_choice = input("Enter model source ('1' for Hugging Face, '2' for Local Path): ").strip()
        
        if source_choice == '1':
            model_path = input("Enter Hugging Face model name (e.g., google/gemma-2b): ").strip()
        elif source_choice == '2':
            model_path = input("Enter the full local path to the model directory: ").strip()
        else:
            raise ValueError("Invalid model source choice")
        
        return {
            "model_name": model_path,
            "is_api": False
        }, False
        
    elif choice == "2":
        # API model selection
        available_models = get_available_api_models()
        
        print("\n--- Available API Providers ---")
        providers = list(available_models.keys())
        for i, provider in enumerate(providers):
            print(f"{i+1}. {provider}")
        
        provider_choice = input(f"Select provider (1-{len(providers)}): ").strip()
        try:
            selected_provider = providers[int(provider_choice) - 1]
        except (ValueError, IndexError):
            raise ValueError("Invalid provider choice")
        
        print(f"\n--- Available {selected_provider} Models ---")
        models = available_models[selected_provider]
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        
        model_choice = input(f"Select model (1-{len(models)}): ").strip()
        try:
            selected_model = models[int(model_choice) - 1]
        except (ValueError, IndexError):
            raise ValueError("Invalid model choice")
        
        # Get API key
        existing_key = get_api_key_from_env(selected_provider)
        if existing_key:
            use_existing = input(f"Found existing {selected_provider} API key. Use it? (y/n): ").strip().lower()
            if use_existing == 'y':
                api_key = existing_key
            else:
                api_key = input(f"Enter your {selected_provider} API key: ").strip()
        else:
            api_key = input(f"Enter your {selected_provider} API key: ").strip()
        
        if not api_key:
            raise ValueError("API key is required for API models")
        
        # Set the API key in environment for this session
        set_api_key_env(selected_provider, api_key)
        
        return {
            "model_name": selected_model,
            "provider": selected_provider,
            "api_key": api_key,
            "is_api": True
        }, True
    
    else:
        raise ValueError("Invalid model type choice")

# Helper function for backward compatibility
def initialize_model_pipeline_interactive(target_device_id: int = 0) -> Tuple[Optional[Any], str]:
    """
    Initialize model pipeline with interactive selection
    """
    try:
        model_info, is_api = get_model_selection_interface()
        
        if is_api:
            return initialize_model_pipeline(
                model_info["model_name"],
                target_device_id=target_device_id,
                is_api_model=True,
                api_provider=model_info["provider"],
                api_key=model_info["api_key"]
            )
        else:
            return initialize_model_pipeline(
                model_info["model_name"],
                target_device_id=target_device_id,
                is_api_model=False
            )
            
    except Exception as e:
        logger.error(f"Error in interactive model initialization: {e}")
        return None, 'N/A'