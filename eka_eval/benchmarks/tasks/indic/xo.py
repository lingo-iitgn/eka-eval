from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
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
from collections import Counter
import string

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "google/IndicGenBench_xorqa_in"

TARGET_INDIC_LANG_CODES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]

SPLIT_NAME_IN_FILENAME = "test" 
CUSTOM_SPLIT_KEY = "loaded_data_split" 

NUM_SAMPLES_PER_LANG = 100
MAX_NEW_TOKENS_FOR_QA = 256  
OUTPUT_DIR = "xorqa_indicgenbench_eval_sarvam_multi_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE_FULL_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
}

BATCH_SIZE = 2  
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1

def setup_accelerator():
    """Setup Accelerator for multi-GPU training"""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )
    return accelerator

def format_xorqa_prompt_sarvam(question: str, lang_name: str) -> str:
    """Format prompt for Sarvam base model for XOR-QA (open-domain QA)"""
    question_str = str(question) if question is not None else "[QUESTION MISSING]"
    
    if lang_name == "English":
        prompt = f"""Question: What is the capital of France?
Answer: Paris

Question: Who wrote the novel "Pride and Prejudice"?
Answer: Jane Austen

Question: What is the largest planet in our solar system?
Answer: Jupiter

Question: In which year did World War II end?
Answer: 1945

Question: {question_str}
Answer:"""
    else:
        if lang_name == "Hindi":
            prompt = f"""Question: फ्रांस की राजधानी क्या है?
Answer: पेरिस

Question: "प्राइड एंड प्रेज्यूडिस" उपन्यास किसने लिखा?
Answer: जेन ऑस्टन

Question: हमारे सौर मंडल का सबसे बड़ा ग्रह कौन सा है?
Answer: बृहस्पति

Question: {question_str}
Answer:"""
        elif lang_name == "Bengali":
            prompt = f"""Question: ফ্রান্সের রাজধানী কী?
Answer: প্যারিস

Question: "প্রাইড অ্যান্ড প্রেজুডিস" উপন্যাসটি কে লিখেছেন?
Answer: জেন অস্টেন

Question: আমাদের সৌরজগতের বৃহত্তম গ্রহ কোনটি?
Answer: বৃহস্পতি

Question: {question_str}
Answer:"""
        elif lang_name == "Tamil":
            prompt = f"""Question: பிரான்சின் தலைநகரம் என்ன?
Answer: பாரிஸ்

Question: "பிரைட் அண்ட் பிரெஜுடிஸ்" நாவலை யார் எழுதினார்?
Answer: ஜேன் ஆஸ்டன்

Question: நமது சூரிய குடும்பத்தில் மிகப்பெரிய கிரகம் எது?
Answer: வியாழன்

Question: {question_str}
Answer:"""
        elif lang_name == "Telugu":
            prompt = f"""Question: ఫ్రాన్స్ రాజధాని ఏమిటి?
Answer: పారిస్

Question: "ప్రైడ్ అండ్ ప్రెజుడీస్" నవల ఎవరు రాశారు?
Answer: జేన్ ఆస్టెన్

Question: మన సౌర వ్యవస్థలో అతిపెద్ద గ్రహం ఏది?
Answer: బృహస్పతి

Question: {question_str}
Answer:"""
        else:
           
            prompt = f"""Question: What is the capital of France?
Answer: Paris

Question: Who wrote "Pride and Prejudice"?
Answer: Jane Austen

Question: What is the largest planet in our solar system?
Answer: Jupiter

Question: {question_str}
Answer:"""
    
    return prompt

def load_xorqa_data_robust(indic_lang_code):
    """Robust data loading for XOR-QA dataset"""
    import shutil
    from datasets import load_dataset, Dataset

    try:
        import datasets
        cache_dir = datasets.config.HF_DATASETS_CACHE
        possible_dirs = [
            "google___indic_gen_bench_xorqa_in",
            "google___IndicGenBench_xorqa_in"
        ]
        for dir_name in possible_dirs:
            dataset_cache_dir = os.path.join(cache_dir, dir_name)
            if os.path.exists(dataset_cache_dir):
                shutil.rmtree(dataset_cache_dir)
    except:
        pass
    
    data_file = f"xorqa_{indic_lang_code}_test.json"
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
                if examples.get('lang') == indic_lang_code:
                    extracted_examples.append(examples)
    else:
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                if 'examples' in item and isinstance(item['examples'], dict):
                    examples = item['examples']
                    if examples.get('lang') == indic_lang_code:
                        extracted_examples.append(examples)
    
    if not extracted_examples:
        raise ValueError(f"No examples found for {indic_lang_code}")
    
    print(f"    Found {len(extracted_examples)} examples for {indic_lang_code}")

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_examples, f, ensure_ascii=False, indent=2)
    
    return Dataset.from_list(extracted_examples)

def batch_xorqa_generate(model, tokenizer, prompts, accelerator, max_new_tokens=256):
    """Perform batch open-domain QA using multiple GPUs"""
    answers = []
    
    with tqdm(total=len(prompts), desc="Generating answers", disable=not accelerator.is_main_process) as pbar:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_answers = []
            
            for prompt in batch_prompts:
                try:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048 
                    )
                    
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,  
                            temperature=0.7, 
                            top_p=0.9,
                            top_k=50,
                            num_beams=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            early_stopping=True,
                            use_cache=True
                        )
                    
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    answer = clean_xorqa_output(answer)
                    batch_answers.append(answer)
                    
                except Exception as e:
                    print(f"Error in XOR-QA generation: {e}")
                    batch_answers.append("[XORQA ERROR]")
            
            answers.extend(batch_answers)
            pbar.update(len(batch_prompts))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return answers

def clean_xorqa_output(text):
    """Clean XOR-QA model output"""
    text = text.strip()
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        for line in lines:
            if len(line) > 10:  
                text = line
                break
        else:
            text = lines[0]
    patterns_to_remove = [
        r'^Answer:\s*',
        r'^Response:\s*',
        r'^Output:\s*',
        r'^Result:\s*',
        r'^Question:.*',
        r'^Context:.*',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text)
    
    stop_patterns = [
        r'\n\n',
        r'\nQuestion:',
        r'\nAnswer:',
        r'\nNote:',
        r'\nExplanation:',
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    
    if len(text) > 500:
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            text = sentences[0] + '.'
        else:
            text = text[:500] + "..."
    
    return text.strip()

def normalize_answer(s):
    """Normalize answer for evaluation (from SQuAD evaluation script)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0
    
    precision = 1.0 * num_common / len(prediction_tokens)
    recall = 1.0 * num_common / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_xorqa_metrics(predictions, references):
    """Evaluate XOR-QA predictions using F1 and EM metrics"""
    if not predictions or not references or len(predictions) != len(references):
        return {"f1": 0.0, "exact_match": 0.0}
    
    f1_scores = []
    em_scores = []
    
    for pred, ref in zip(predictions, references):
        if isinstance(ref, list):
            max_f1 = max(f1_score(pred, r) for r in ref)
            max_em = max(exact_match_score(pred, r) for r in ref)
        else:
            max_f1 = f1_score(pred, ref)
            max_em = exact_match_score(pred, ref)
        
        f1_scores.append(max_f1)
        em_scores.append(max_em)
    
    return {
        "f1": np.mean(f1_scores) * 100, 
        "exact_match": np.mean(em_scores) * 100 
    }

def evaluate_language_xorqa(accelerator, model, tokenizer, indic_lang_code):
    """Evaluate XOR-QA for a specific language"""
    
    lang_full_name = LANGUAGE_FULL_NAMES.get(indic_lang_code, indic_lang_code.upper())
    display_config_name = f"xorqa_{indic_lang_code}"
    
    if accelerator.is_main_process:
        print(f"\n--- Evaluating: {lang_full_name} XOR-QA (as {display_config_name}) ---")
    
    try:
       
        if accelerator.is_main_process:
            print(f"  Loading XOR-QA dataset for {indic_lang_code}...")
        current_dataset_for_eval = load_xorqa_data_robust(indic_lang_code)
        
        num_to_sample = min(NUM_SAMPLES_PER_LANG, len(current_dataset_for_eval))
        if num_to_sample == 0:
            return {"f1": 0.0, "exact_match": 0.0, "info": "0 samples evaluated"}, None
            
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
            question = example_data.get("question", "")
            answers = example_data.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = []
            
            if not isinstance(question, str) or not answers:
                continue
            
            prompt = format_xorqa_prompt_sarvam(question, lang_full_name)
            
            prompts.append(prompt)
            references.append(answers) 
            
            detailed_logs.append({
                "pair_key": display_config_name,
                "example_index": example_idx,
                "question": question,
                "reference_answers": answers,
                "model_answer": "",  
                "prompt_used": prompt
            })
        
        if not prompts:
            return {"f1": 0.0, "exact_match": 0.0, "info": "No valid examples"}, None
        
        if accelerator.is_main_process:
            print(f"  Starting batch XOR-QA generation for {len(prompts)} examples...")
        
        predictions = batch_xorqa_generate(model, tokenizer, prompts, accelerator, MAX_NEW_TOKENS_FOR_QA)
        
        for i, prediction in enumerate(predictions):
            if i < len(detailed_logs):
                detailed_logs[i]["model_answer"] = prediction

        if accelerator.is_main_process and len(detailed_logs) > 0:
            print(f"\n  Debug - First XOR-QA example for {indic_lang_code}:")
            print(f"  Question: {detailed_logs[0]['question']}")
            print(f"  Reference answers: {detailed_logs[0]['reference_answers']}")
            print(f"  Model answer: {detailed_logs[0]['model_answer']}")

        if predictions and references:
            valid_pairs = []
            for p, r in zip(predictions, references):
                if isinstance(p, str) and p.strip() and isinstance(r, list) and r:
                    valid_refs = [str(ref).strip() for ref in r if str(ref).strip()]
                    if valid_refs:
                        valid_pairs.append((p.strip(), valid_refs))
            
            if not valid_pairs:
                if accelerator.is_main_process:
                    print(f"  ⚠️ {display_config_name}: No valid predictions after filtering")
                    print(f"     Total predictions: {len(predictions)}, Total references: {len(references)}")
                    print(f"     Sample predictions: {predictions[:3] if predictions else 'None'}")
                    print(f"     Sample references: {references[:3] if references else 'None'}")
                return {"f1": 0.0, "exact_match": 0.0, "info": "No valid predictions after filtering"}, detailed_logs
            
            valid_predictions, valid_references = zip(*valid_pairs)
            metrics = evaluate_xorqa_metrics(valid_predictions, valid_references)
            for i, (pred, refs) in enumerate(zip(valid_predictions, valid_references)):
                if i < len(detailed_logs):
                    best_f1 = max(f1_score(pred, ref) for ref in refs) * 100
                    best_em = max(exact_match_score(pred, ref) for ref in refs) * 100
                    detailed_logs[i]["f1_score"] = best_f1
                    detailed_logs[i]["em_score"] = best_em
            
            result = {
                'f1': metrics['f1'],
                'exact_match': metrics['exact_match'],
                'valid_samples': len(valid_pairs),
                'total_samples': len(predictions)
            }
            
            if accelerator.is_main_process:
                print(f"  ✅ {display_config_name}: F1 = {metrics['f1']:.2f}, EM = {metrics['exact_match']:.2f} ({len(valid_pairs)}/{len(predictions)} valid)")
                
               
                print(f"  📊 Sample scores for {display_config_name}:")
                for i in range(min(3, len(detailed_logs))):
                    if 'f1_score' in detailed_logs[i]:
                        print(f"    Example {i+1}: F1={detailed_logs[i]['f1_score']:.1f}, EM={detailed_logs[i]['em_score']:.1f}")
                        print(f"      Q: {detailed_logs[i]['question'][:80]}...")
                        print(f"      Refs: {[ref[:30] + '...' if len(ref) > 30 else ref for ref in detailed_logs[i]['reference_answers'][:2]]}")
                        print(f"      Pred: {detailed_logs[i]['model_answer'][:50]}...")
            
            if accelerator.is_main_process:
                lang_output_filename = os.path.join(OUTPUT_DIR, f"{display_config_name}_detailed_results.json")
                try:
                   
                    results_to_save = {
                        "summary": {
                            "language": indic_lang_code,
                            "language_full_name": lang_full_name,
                            "total_samples": len(predictions),
                            "valid_samples": len(valid_pairs),
                            "average_f1": metrics['f1'],
                            "average_em": metrics['exact_match']
                        },
                        "examples": detailed_logs
                    }
                    
                    with open(lang_output_filename, "w", encoding="utf-8") as f:
                        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
                    print(f"  💾 Saved detailed results to {lang_output_filename}")
                except Exception as e:
                    print(f"  ❌ Error saving detailed results: {e}")
            
            return result, detailed_logs
        else:
            return {"f1": 0.0, "exact_match": 0.0, "info": "No valid predictions"}, []
            
    except Exception as e:
        error_msg = f"Error processing {display_config_name}: {str(e)}"
        if accelerator.is_main_process:
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()
        return {"f1": 0.0, "exact_match": 0.0, "error": error_msg}, []

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
            max_memory={i: "25GB" for i in range(torch.cuda.device_count())}
        )
        pbar.update(1)
    
    model.eval()
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    if accelerator.is_main_process:
        print("Sarvam model and tokenizer loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
    
    overall_scores = {}
    all_f1_scores = []
    all_em_scores = []
    all_detailed_logs = []
    
    lang_iter = tqdm(
        TARGET_INDIC_LANG_CODES,
        desc="Evaluating languages",
        disable=not accelerator.is_main_process
    )
    
    for indic_lang_code in lang_iter:
        display_name = f"xorqa_{indic_lang_code}"
        lang_iter.set_postfix({"lang": indic_lang_code})
        
        result, detailed_logs = evaluate_language_xorqa(
            accelerator, model, tokenizer, indic_lang_code
        )
        
        if result is not None:
            overall_scores[display_name] = result
            if "f1" in result and isinstance(result["f1"], (int, float)):
                all_f1_scores.append(result["f1"])
            if "exact_match" in result and isinstance(result["exact_match"], (int, float)):
                all_em_scores.append(result["exact_match"])
            
            if detailed_logs:
                all_detailed_logs.extend(detailed_logs)

        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        overall_json_filename = os.path.join(
            OUTPUT_DIR, 
            f"all_lang_xorqa_eval_sarvam_multi_gpu.json"
        )
        try:
            with open(overall_json_filename, "w", encoding="utf-8") as f:
                json.dump(all_detailed_logs, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Overall detailed log saved to: {overall_json_filename}")
        except Exception as e:
            print(f"Error saving overall detailed log: {e}")

        summary_filename = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
        try:
            summary_data = {
                "model_name": MODEL_NAME,
                "dataset": "XOR-QA-IN",
                "num_samples_per_lang": NUM_SAMPLES_PER_LANG,
                "languages_evaluated": len(overall_scores),
                "overall_scores": overall_scores,
                "averages": {
                    "f1": np.mean(all_f1_scores) if all_f1_scores else 0.0,
                    "exact_match": np.mean(all_em_scores) if all_em_scores else 0.0
                }
            }
            
            with open(summary_filename, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            print(f"📊 Summary saved to: {summary_filename}")
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        print("\n" + "="*70)
        print("🏆 Final XOR-QA-IN Evaluation Summary - SARVAM MODEL 🏆")
        print(f"Model: {MODEL_NAME}")
        print("="*70)
        
        for pair_name, scores in overall_scores.items():
            if "error" not in scores:
                valid_info = f" ({scores.get('valid_samples', 0)}/{scores.get('total_samples', 0)} valid)" if 'valid_samples' in scores else ""
                print(f"  - {pair_name}: F1 = {scores.get('f1', 0.0):.2f}, EM = {scores.get('exact_match', 0.0):.2f}{valid_info}")
            else:
                print(f"  - {pair_name}: ERROR ({scores.get('error', 'Unknown error')})")
        
        if all_f1_scores and all_em_scores:
            avg_f1 = np.mean(all_f1_scores)
            avg_em = np.mean(all_em_scores)
            print(f"\n📈 Overall Average F1: {avg_f1:.2f} (across {len(all_f1_scores)} languages)")
            print(f"📈 Overall Average EM: {avg_em:.2f} (across {len(all_em_scores)} languages)")
        else:
            print("\n⚠️ No valid F1/EM scores computed.")
        
        print("\nEvaluation complete!")

if __name__ == "__main__":
    main()