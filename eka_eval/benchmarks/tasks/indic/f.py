from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import evaluate
import torch
import torch.distributed as dist
from sacrebleu import corpus_chrf
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import re 
import json 
import os 
import traceback 
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs

MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "google/IndicGenBench_flores_in"

TARGET_INDIC_LANG_CODES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
TRANSLATION_DIRECTION = "enxx"  

SPLIT_NAME_IN_FILENAME = "test" 
CUSTOM_SPLIT_KEY = "loaded_data_split" 

NUM_SAMPLES_PER_LANG = 100
MAX_NEW_TOKENS_FOR_TRANSLATION = 128 
OUTPUT_DIR = "flores_indicgenbench_eval_sarvam_multi_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE_FULL_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
}

BATCH_SIZE = 4 
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1

def setup_accelerator():
    """Setup Accelerator for multi-GPU training"""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )
    return accelerator

def format_translation_prompt_sarvam(source_text: str, source_lang_name: str, target_lang_name: str) -> str:
    """Format prompt for Sarvam base model - using completion-style format"""
    source_text_str = str(source_text) if source_text is not None else "[SOURCE TEXT MISSING]"
    lang_codes = {
        "English": "en", "Hindi": "hi", "Bengali": "bn", "Gujarati": "gu", 
        "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml",
        "Marathi": "mr", "Punjabi": "pa", "Odia": "or", "Assamese": "as", "Urdu": "ur"
    }
    
    source_code = lang_codes.get(source_lang_name, source_lang_name.lower()[:2])
    target_code = lang_codes.get(target_lang_name, target_lang_name.lower()[:2])
    
    # Base completion format - provide examples in context
    prompt = f"""English: The quick brown fox jumps over the lazy dog.
Hindi: तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।

English: I love learning new languages.
Bengali: আমি নতুন ভাষা শিখতে ভালোবাসি।

English: Technology is advancing rapidly.
Gujarati: ટેકનોલોજી ઝડપથી આગળ વધી રહી છે।

{source_lang_name}: {source_text_str}
{target_lang_name}:"""
    
    return prompt

def load_flores_data_robust(indic_lang_code, translation_direction):
    """Robust data loading that handles cache issues"""
    import shutil
    from datasets import load_dataset, Dataset
    try:
        import datasets
        cache_dir = datasets.config.HF_DATASETS_CACHE
        possible_dirs = [
            "google___indic_gen_bench_flores_in",
            "google___IndicGenBench_flores_in"
        ]
        for dir_name in possible_dirs:
            dataset_cache_dir = os.path.join(cache_dir, dir_name)
            if os.path.exists(dataset_cache_dir):
                shutil.rmtree(dataset_cache_dir)
    except:
        pass
    
    data_file = f"flores_{indic_lang_code}_{translation_direction}_test.json"
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            return Dataset.from_list(examples)
        except Exception as e:
            print(f"Failed to load local file {data_file}: {e}")

    methods = [
        lambda: load_dataset(DATASET_NAME, data_files={CUSTOM_SPLIT_KEY: data_file}, trust_remote_code=True),
        lambda: load_dataset(DATASET_NAME, trust_remote_code=True, download_mode="force_redownload"),
        lambda: load_dataset(DATASET_NAME, trust_remote_code=True)
    ]
    
    dataset = None
    for i, method in enumerate(methods):
        try:
            print(f"    Trying loading method {i+1}...")
            dataset = method()
            print(f"    ✅ Method {i+1} successful!")
            break
        except Exception as e:
            print(f"    ❌ Method {i+1} failed: {e}")
            continue
    
    if dataset is None:
        raise ValueError("All dataset loading methods failed")
    
    extracted_examples = []
    
    if isinstance(dataset, dict) and CUSTOM_SPLIT_KEY in dataset:
        for item in dataset[CUSTOM_SPLIT_KEY]:
            if 'examples' in item and isinstance(item['examples'], dict):
                examples = item['examples']
                if (examples.get('lang') == indic_lang_code and 
                    examples.get('translation_direction') == translation_direction):
                    extracted_examples.append(examples)
    else:
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                if 'examples' in item and isinstance(item['examples'], dict):
                    examples = item['examples']
                    if (examples.get('lang') == indic_lang_code and 
                        examples.get('translation_direction') == translation_direction):
                        extracted_examples.append(examples)
    
    if not extracted_examples:
        raise ValueError(f"No examples found for {indic_lang_code}-{translation_direction}")
    
    print(f"    Found {len(extracted_examples)} examples for {indic_lang_code}")
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_examples, f, ensure_ascii=False, indent=2)
    
    return Dataset.from_list(extracted_examples)

def batch_translate(model, tokenizer, prompts, accelerator, max_new_tokens=128):
    """Perform batch translation using multiple GPUs - optimized for Sarvam"""
    translations = []
    
    with tqdm(total=len(prompts), desc="Translating", disable=not accelerator.is_main_process) as pbar:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_translations = []
            
            for prompt in batch_prompts:
                try:
                    
                    formatted_prompt = prompt
                    inputs = tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,  
                            temperature=0.7,  
                            top_p=0.95, 
                            num_beams=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.05, 
                            early_stopping=True,
                            use_cache=True
                        )
                    
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    translation = clean_translation_output(translation)
                    batch_translations.append(translation)
                    
                except Exception as e:
                    print(f"Error in translation: {e}")
                    batch_translations.append("[TRANSLATION ERROR]")
            
            translations.extend(batch_translations)
            pbar.update(len(batch_prompts))  
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return translations

def clean_translation_output(text):
    """Enhanced cleaning for Sarvam base model output"""
    text = text.strip()

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        text = lines[0]
    
    patterns_to_remove = [
        r'^(English|Hindi|Bengali|Gujarati|Tamil|Telugu|Kannada|Malayalam|Marathi|Punjabi|Odia|Assamese|Urdu):\s*',
        r'^Translation:\s*',
        r'^Output:\s*',
        r'^Result:\s*',
        r'^Answer:\s*',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text)
    stop_patterns = [
        r'\n\n',
        r'\n[A-Z][a-z]+:', 
        r'\nEnglish:',
        r'\nHindi:',
        r'\nBengali:',
        r'\nGujarati:',
        
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    
    return text.strip()

def evaluate_language_pair(accelerator, model, tokenizer, indic_lang_code):
    
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
       
        if accelerator.is_main_process:
            print(f"  Loading dataset for {indic_lang_code}...")
        current_dataset_for_eval = load_flores_data_robust(indic_lang_code, TRANSLATION_DIRECTION)
        
        num_to_sample = min(NUM_SAMPLES_PER_LANG, len(current_dataset_for_eval))
        if num_to_sample == 0:
            return {"chrf++": 0.0, "info": "0 samples evaluated"}, None
            
        subset_to_eval = current_dataset_for_eval.select(range(num_to_sample))
        
        if accelerator.is_main_process:
            print(f"  Evaluating on {len(subset_to_eval)} samples")

        prompts = []
        references = []
        detailed_logs = []
        
        for example_idx, example_data in tqdm(
            enumerate(subset_to_eval),
            total=len(subset_to_eval),
            desc=f"Preparing {indic_lang_code} examples",
            disable=not accelerator.is_main_process
        ):
            source_text = example_data.get("source", "")
            reference_translation = example_data.get("target", "")
            
            if not isinstance(source_text, str) or not isinstance(reference_translation, str):
                continue
            
            prompt = format_translation_prompt_sarvam(
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
                "model_translation": "", 
                "prompt_used": prompt  
            })
        
        if not prompts:
            return {"chrf++": 0.0, "info": "No valid examples"}, None
        
        if accelerator.is_main_process:
            print(f"  Starting batch translation for {len(prompts)} examples...")
        
        predictions = batch_translate(model, tokenizer, prompts, accelerator, MAX_NEW_TOKENS_FOR_TRANSLATION)
        
        for i, prediction in enumerate(predictions):
            if i < len(detailed_logs):
                detailed_logs[i]["model_translation"] = prediction
    
        if accelerator.is_main_process and len(detailed_logs) > 0:
            print(f"\n  Debug - First translation example for {indic_lang_code}:")
            print(f"  Source: {detailed_logs[0]['source_text'][:100]}...")
            print(f"  Reference: {detailed_logs[0]['reference_translation'][:100]}...")
            print(f"  Model output: {detailed_logs[0]['model_translation'][:100]}...")
            print(f"  Prompt preview: {detailed_logs[0]['prompt_used'][-200:]}...")  

        if predictions and references:
            valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
            
            if not valid_pairs:
                return {"chrf++": 0.0, "info": "No valid predictions after filtering"}, detailed_logs
            
            valid_predictions, valid_references = zip(*valid_pairs)
            
            references_list = [[ref] for ref in valid_references]
            
            chrf_score = corpus_chrf(
                valid_predictions, 
                references_list,
                word_order=2  
            ).score
            
            result = {
                'chrf++': chrf_score,
                'valid_samples': len(valid_pairs),
                'total_samples': len(predictions)
            }
            
            if accelerator.is_main_process:
                print(f"  ✅ {display_config_name}: chrF++ = {chrf_score:.2f} ({len(valid_pairs)}/{len(predictions)} valid)")
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
    accelerator = setup_accelerator()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.85, device=i)
    with tqdm(total=3, desc="Loading Sarvam model", disable=not accelerator.is_main_process) as pbar:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        pbar.update(1)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pbar.update(1)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={i: "25GB" for i in range(torch.cuda.device_count())} # Reduced memory
        )
        pbar.update(1)
    
    model.eval()
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    if accelerator.is_main_process:
        print("Sarvam model and tokenizer loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
    
    overall_scores = {}
    all_chrf_scores = []
    all_detailed_logs = []
    
    lang_iter = tqdm(
        TARGET_INDIC_LANG_CODES,
        desc="Evaluating languages",
        disable=not accelerator.is_main_process
    )
    
    for indic_lang_code in lang_iter:
        display_name = f"flores_{('en' if TRANSLATION_DIRECTION == 'enxx' else indic_lang_code)}-{(indic_lang_code if TRANSLATION_DIRECTION == 'enxx' else 'en')}"
        lang_iter.set_postfix({"lang": indic_lang_code})
        
        result, detailed_logs = evaluate_language_pair(
            accelerator, model, tokenizer, indic_lang_code
        )
        
        if result is not None:
            overall_scores[display_name] = result
            if "chrf++" in result and isinstance(result["chrf++"], (int, float)):
                all_chrf_scores.append(result["chrf++"])
            
            if detailed_logs:
                all_detailed_logs.extend(detailed_logs)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        overall_json_filename = os.path.join(
            OUTPUT_DIR, 
            f"all_lang_pairs_flores_eval_sarvam_multi_gpu.json"
        )
        try:
            with open(overall_json_filename, "w", encoding="utf-8") as f:
                json.dump(all_detailed_logs, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Overall detailed log saved to: {overall_json_filename}")
        except Exception as e:
            print(f"Error saving overall detailed log: {e}")
        
        print("\n" + "="*70)
        print("🏆 Final FLORES Evaluation Summary (chrF++) - SARVAM MODEL 🏆")
        print(f"Model: {MODEL_NAME}")
        print(f"Direction: {TRANSLATION_DIRECTION}")
        print("="*70)
        
        for pair_name, scores in overall_scores.items():
            if "error" not in scores:
                valid_info = f" ({scores.get('valid_samples', 0)}/{scores.get('total_samples', 0)} valid)" if 'valid_samples' in scores else ""
                print(f"  - {pair_name}: chrF++ = {scores.get('chrf++', 0.0):.2f}{valid_info}")
            else:
                print(f"  - {pair_name}: ERROR ({scores.get('error', 'Unknown error')})")
        
        if all_chrf_scores:
            avg_chrf = np.mean(all_chrf_scores)
            print(f"\n📈 Overall Average chrF++: {avg_chrf:.2f} (across {len(all_chrf_scores)} language pairs)")
            print(f"📊 Expected Sarvam paper score: ~41.0")
            
            if avg_chrf < 10:
                print("\n⚠️  WARNING: Scores are significantly lower than expected!")
                print("   Potential issues to check:")
                print("   1. Prompt format may not match Sarvam's training")
                print("   2. Generation parameters might need tuning")
                print("   3. Dataset preprocessing might differ from paper")
                print("   4. Model version or fine-tuning differences")
        else:
            print("\n⚠️ No valid chrF++ scores computed.")
        
        print("\nEvaluation complete!")

if __name__ == "__main__":
    main()