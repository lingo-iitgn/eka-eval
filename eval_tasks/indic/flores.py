from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import evaluate
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import re 
import json 
import os 
import traceback 
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Changed to Llama3 8B
DATASET_NAME = "google/IndicGenBench_flores_in"

TARGET_INDIC_LANG_CODES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
# TARGET_INDIC_LANG_CODES = ["bn", "hi"]  # For quicker testing

TRANSLATION_DIRECTION = "enxx"  # "enxx" (English to Indic) or "xxen" (Indic to English)

SPLIT_NAME_IN_FILENAME = "test" 
CUSTOM_SPLIT_KEY = "loaded_data_split" 

NUM_SAMPLES_PER_LANG = 100  # Increased for better evaluation
MAX_NEW_TOKENS_FOR_TRANSLATION = 128 
OUTPUT_DIR = "flores_indicgenbench_eval_llama3_8b_multi_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE_FULL_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
}

# Multi-GPU Configuration
BATCH_SIZE = 8  # Adjust based on GPU memory
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1

def setup_accelerator():
    """Setup Accelerator for multi-GPU training"""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",  # Use FP16 for memory efficiency
        kwargs_handlers=[ddp_kwargs]
    )
    return accelerator

def format_translation_prompt_llama3(source_text: str, source_lang_name: str, target_lang_name: str) -> str:
    """Format prompt specifically for Llama3 with chat template"""
    source_text_str = str(source_text) if source_text is not None else "[SOURCE TEXT MISSING]"
    source_lang_name_str = str(source_lang_name) if source_lang_name is not None else "[SOURCE LANG NAME MISSING]"
    target_lang_name_str = str(target_lang_name) if target_lang_name is not None else "[TARGET LANG NAME MISSING]"

    # Llama3 chat format
    messages = [
        {
            "role": "system", 
            "content": f"You are a professional translator. Translate the given text from {source_lang_name_str} to {target_lang_name_str}. Provide only the translation without any additional text or explanation."
        },
        {
            "role": "user", 
            "content": f"Translate this {source_lang_name_str} text to {target_lang_name_str}:\n\n{source_text_str}"
        }
    ]
    
    return messages

def batch_translate(model, tokenizer, prompts, accelerator, max_new_tokens=128):
    """Perform batch translation using multiple GPUs"""
    translations = []
    
    # Process in batches
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_translations = []
        
        for prompt in batch_prompts:
            try:
                # Apply chat template for Llama3
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Tokenize
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=0.1,
                        repetition_penalty=1.1
                    )
                
                # Decode only the new tokens (exclude input)
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Clean up translation (remove any remaining prompt artifacts)
                translation = clean_translation_output(translation)
                batch_translations.append(translation)
                
            except Exception as e:
                print(f"Error in translation: {e}")
                batch_translations.append("[TRANSLATION ERROR]")
        
        translations.extend(batch_translations)
        
        # Clear cache to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return translations

def clean_translation_output(text):
    """Clean the translation output to remove artifacts"""
    # Remove common artifacts
    text = text.strip()
    
    # Remove any remaining system/user prefixes
    patterns_to_remove = [
        r'^(Assistant|User|System):\s*',
        r'^Translation:\s*',
        r'^Output:\s*',
        r'^Result:\s*',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def evaluate_language_pair(accelerator, model, tokenizer, indic_lang_code, chrf_metric):
    """Evaluate a single language pair"""
    
    source_lang_short, target_lang_short = "", ""
    file_lang_component_for_filename = indic_lang_code 

    if TRANSLATION_DIRECTION == "enxx": 
        source_lang_short = "en"
        target_lang_short = indic_lang_code
    elif TRANSLATION_DIRECTION == "xxen": 
        source_lang_short = indic_lang_code
        target_lang_short = "en"
    else:
        return None, f"Unsupported TRANSLATION_DIRECTION: {TRANSLATION_DIRECTION}"

    source_lang_full_name = LANGUAGE_FULL_NAMES.get(source_lang_short, source_lang_short.upper())
    target_lang_full_name = LANGUAGE_FULL_NAMES.get(target_lang_short, target_lang_short.upper())
    display_config_name = f"flores_{source_lang_short}-{target_lang_short}" 
    
    if accelerator.is_main_process:
        print(f"\n--- Evaluating: {source_lang_full_name} to {target_lang_full_name} (as {display_config_name}) ---")
    
    try:
        # Load dataset
        data_file_to_load = f"flores_{file_lang_component_for_filename}_{TRANSLATION_DIRECTION}_{SPLIT_NAME_IN_FILENAME}.json"
        
        if accelerator.is_main_process:
            print(f"  Loading: {os.path.join(DATASET_NAME, data_file_to_load)}")
        
        ds_dict = load_dataset(
            DATASET_NAME, 
            data_files={CUSTOM_SPLIT_KEY: data_file_to_load}, 
            trust_remote_code=True
        )
        
        loaded_file_as_dataset = ds_dict[CUSTOM_SPLIT_KEY] 
        
        if not loaded_file_as_dataset or len(loaded_file_as_dataset) == 0:
            return None, f"No data loaded for file '{data_file_to_load}'"
            
        # Extract examples
        actual_translation_examples_list = []
        if 'examples' in loaded_file_as_dataset.features:
             actual_translation_examples_list = [
                 row['examples'] for row in loaded_file_as_dataset 
                 if 'examples' in row and isinstance(row['examples'], dict)
             ]
        else:
            return None, f"'examples' field not found in dataset features"
        
        if not actual_translation_examples_list:
            return None, f"No translation examples extracted from 'examples' field"

        current_dataset_for_eval = HFDataset.from_list(actual_translation_examples_list)
        
        # Sample subset
        num_to_sample = min(NUM_SAMPLES_PER_LANG, len(current_dataset_for_eval))
        if num_to_sample == 0:
            return {"chrf++": 0.0, "info": "0 samples evaluated"}, None
            
        subset_to_eval = current_dataset_for_eval.select(range(num_to_sample))
        
        if accelerator.is_main_process:
            print(f"  Evaluating on {len(subset_to_eval)} samples")

        # Prepare data for batch processing
        prompts = []
        references = []
        detailed_logs = []
        
        for example_idx, example_data in enumerate(subset_to_eval):
            source_text = example_data.get("source", "")
            reference_translation = example_data.get("target", "")
            
            # Skip invalid examples
            if not isinstance(source_text, str) or not isinstance(reference_translation, str):
                continue
            
            prompt = format_translation_prompt_llama3(
                source_text, 
                source_lang_full_name, 
                target_lang_full_name
            )
            
            prompts.append(prompt)
            references.append(reference_translation)
            
            detailed_logs.append({
                "pair_key": display_config_name,
                "example_index": example_idx,
                "source_text": source_text,
                "reference_translation": reference_translation,
                "model_translation": "",  # Will be filled later
            })
        
        if not prompts:
            return {"chrf++": 0.0, "info": "No valid examples"}, None
        
        # Batch translate
        if accelerator.is_main_process:
            print(f"  Starting batch translation for {len(prompts)} examples...")
        
        predictions = batch_translate(model, tokenizer, prompts, accelerator, MAX_NEW_TOKENS_FOR_TRANSLATION)
        
        # Update detailed logs
        for i, prediction in enumerate(predictions):
            if i < len(detailed_logs):
                detailed_logs[i]["model_translation"] = prediction
        
        # Compute metrics
        if predictions and references:
            # Prepare references in the format expected by sacrebleu
            references_list_of_lists = [[ref] for ref in references]
            
            metric_results = chrf_metric.compute(
                predictions=predictions, 
                references=references_list_of_lists,
                tokenize="flores200", 
                chrf_word_order=2
            )
            chrf_score = metric_results.get('score', 0.0)
            
            result = {'chrf++': chrf_score}
            
            if accelerator.is_main_process:
                print(f"  ✅ {display_config_name}: chrF++ = {chrf_score:.2f}")
            
            # Save detailed results
            if accelerator.is_main_process:
                lang_output_filename = os.path.join(OUTPUT_DIR, f"{display_config_name}_detailed_results.json")
                try:
                    with open(lang_output_filename, "w", encoding="utf-8") as f:
                        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
                    print(f"  Saved detailed results to {lang_output_filename}")
                except Exception as e:
                    print(f"  Error saving detailed results: {e}")
            
            return result, detailed_logs
        else:
            return {"chrf++": 0.0, "info": "No valid predictions"}, []
            
    except Exception as e:
        error_msg = f"Error processing {display_config_name}: {str(e)}"
        if accelerator.is_main_process:
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()
        return {"chrf++": 0.0, "error": error_msg}, []

def main():
    """Main evaluation function"""
    # Setup accelerator
    accelerator = setup_accelerator()
    
    if accelerator.is_main_process:
        print(f"--- Multi-GPU FLORES Evaluation Setup ---")
        print(f"Model: {MODEL_NAME}")
        print(f"Available GPUs: {WORLD_SIZE}")
        print(f"Translation Direction: {TRANSLATION_DIRECTION}")
        print(f"Target Languages: {', '.join(TARGET_INDIC_LANG_CODES)}")
        print(f"Samples per language: {NUM_SAMPLES_PER_LANG}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("="*70)
    
    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer and model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device_map for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    
    # Prepare model with accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    if accelerator.is_main_process:
        print("Model and tokenizer loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
    
    # Load metrics
    chrf_metric = evaluate.load("sacrebleu")
    
    # Results storage
    overall_scores = {}
    all_chrf_scores = []
    all_detailed_logs = []
    
    # Evaluate each language pair
    for indic_lang_code in TARGET_INDIC_LANG_CODES:
        display_name = f"flores_{('en' if TRANSLATION_DIRECTION == 'enxx' else indic_lang_code)}-{(indic_lang_code if TRANSLATION_DIRECTION == 'enxx' else 'en')}"
        
        result, detailed_logs = evaluate_language_pair(
            accelerator, model, tokenizer, indic_lang_code, chrf_metric
        )
        
        if result is not None:
            overall_scores[display_name] = result
            if "chrf++" in result and isinstance(result["chrf++"], (int, float)):
                all_chrf_scores.append(result["chrf++"])
            
            if detailed_logs:
                all_detailed_logs.extend(detailed_logs)
        
        # Synchronize processes
        accelerator.wait_for_everyone()
    
    # Save overall results (only on main process)
    if accelerator.is_main_process:
        # Save detailed logs
        overall_json_filename = os.path.join(
            OUTPUT_DIR, 
            f"all_lang_pairs_flores_eval_{MODEL_NAME.split('/')[-1]}_multi_gpu.json"
        )
        try:
            with open(overall_json_filename, "w", encoding="utf-8") as f:
                json.dump(all_detailed_logs, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Overall detailed log saved to: {overall_json_filename}")
        except Exception as e:
            print(f"Error saving overall detailed log: {e}")
        
        # Print final results
        print("\n" + "="*70)
        print("🏆 Final FLORES Evaluation Summary (chrF++) 🏆")
        print(f"Model: {MODEL_NAME}")
        print(f"Direction: {TRANSLATION_DIRECTION}")
        print("="*70)
        
        for pair_name, scores in overall_scores.items():
            if "error" not in scores:
                print(f"  - {pair_name}: chrF++ = {scores.get('chrf++', 0.0):.2f}")
            else:
                print(f"  - {pair_name}: ERROR ({scores.get('error', 'Unknown error')})")
        
        if all_chrf_scores:
            avg_chrf = np.mean(all_chrf_scores)
            print(f"\n📈 Overall Average chrF++: {avg_chrf:.2f} (across {len(all_chrf_scores)} language pairs)")
        else:
            print("\n⚠️ No valid chrF++ scores computed.")
        
        print("\nEvaluation complete!")

if __name__ == "__main__":
    main()