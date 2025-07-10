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
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "google/IndicGenBench_xquad_in"

TARGET_INDIC_LANG_CODES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]

SPLIT_NAME_IN_FILENAME = "test" 
CUSTOM_SPLIT_KEY = "loaded_data_split" 

NUM_SAMPLES_PER_LANG = 100
MAX_NEW_TOKENS_FOR_QA = 128 
OUTPUT_DIR = "xquad_indicgenbench_eval_sarvam_multi_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE_FULL_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
}

# Multi-GPU Configuration
BATCH_SIZE = 4  # Reduced for Sarvam model
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1

def setup_accelerator():
    """Setup Accelerator for multi-GPU training"""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )
    return accelerator

def format_qa_prompt_sarvam(context: str, question: str, lang_name: str) -> str:
    """Format prompt for Sarvam base model for Question Answering"""
    context_str = str(context) if context is not None else "[CONTEXT MISSING]"
    question_str = str(question) if question is not None else "[QUESTION MISSING]"
    
    # Base completion format with examples for QA task
    if lang_name == "English":
        prompt = f"""Context: The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest.
Question: How many square kilometers does the Amazon basin cover?
Answer: 7,000,000 square kilometers

Context: Basketball is a team sport in which two teams, most commonly of five players each, opposing one another on a rectangular court, compete with the primary objective of shooting a basketball through the defender's hoop.
Question: How many players are typically on each basketball team?
Answer: five players

Context: {context_str}
Question: {question_str}
Answer:"""
    else:
        # For Indic languages, provide examples in the target language when possible
        if lang_name == "Hindi":
            prompt = f"""Context: अमेज़न वर्षावन एक नम चौड़ी पत्ती वाला जंगल है जो दक्षिण अमेरिका के अमेज़न बेसिन के अधिकांश हिस्से को कवर करता है। यह बेसिन 7,000,000 वर्ग किलोमीटर में फैला है।
Question: अमेज़न बेसिन कितने वर्ग किलोमीटर में फैला है?
Answer: 7,000,000 वर्ग किलोमीटर

Context: {context_str}
Question: {question_str}
Answer:"""
        elif lang_name == "Bengali":
            prompt = f"""Context: আমাজন রেইনফরেস্ট একটি আর্দ্র বিস্তৃত পাতার বন যা দক্ষিণ আমেরিকার আমাজন অববাহিকার বেশিরভাগ অংশ জুড়ে রয়েছে। এই অববাহিকা ৭,০০০,০০০ বর্গ কিলোমিটার জুড়ে বিস্তৃত।
Question: আমাজন অববাহিকা কত বর্গ কিলোমিটার জুড়ে বিস্তৃত?
Answer: ৭,০০০,০০০ বর্গ কিলোমিটার

Context: {context_str}
Question: {question_str}
Answer:"""
        else:
            # Generic format for other Indic languages
            prompt = f"""Context: The Amazon rainforest covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers.
Question: How many square kilometers does the Amazon basin cover?
Answer: 7,000,000 square kilometers

Context: {context_str}
Question: {question_str}
Answer:"""
    
    return prompt

def extract_answer_from_xquad_item(item, debug=False):
    """
    Extract answer from XQuAD item with multiple fallback strategies
    """
    if debug:
        print(f"Debug: Processing item keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
        print(f"Debug: Item type: {type(item)}")
    
    # Strategy 1: Direct answer field
    if isinstance(item, dict):
        if 'answer' in item and item['answer']:
            answer = item['answer']
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            elif isinstance(answer, dict) and 'text' in answer:
                return answer['text'].strip() if answer['text'] else ""
        
        # Strategy 2: answers array (standard XQuAD format)
        if 'answers' in item:
            answers = item['answers']
            if isinstance(answers, list) and len(answers) > 0:
                first_answer = answers[0]
                if isinstance(first_answer, dict) and 'text' in first_answer:
                    return first_answer['text'].strip() if first_answer['text'] else ""
                elif isinstance(first_answer, str):
                    return first_answer.strip()
        
        # Strategy 3: Check nested structures
        for key in ['examples', 'data', 'item']:
            if key in item and isinstance(item[key], dict):
                nested_answer = extract_answer_from_xquad_item(item[key], debug)
                if nested_answer:
                    return nested_answer
    
    if debug:
        print(f"Debug: No answer found in item: {item}")
    
    return ""

def load_xquad_data_robust(indic_lang_code):
    """Robust data loading for XQuAD dataset with improved answer extraction"""
    import shutil
    from datasets import load_dataset, Dataset
    
    print(f"Loading XQuAD data for {indic_lang_code}...")
    
    # Clear problematic cache
    try:
        import datasets
        cache_dir = datasets.config.HF_DATASETS_CACHE
        possible_dirs = [
            "google___indic_gen_bench_xquad_in",
            "google___IndicGenBench_xquad_in"
        ]
        for dir_name in possible_dirs:
            dataset_cache_dir = os.path.join(cache_dir, dir_name)
            if os.path.exists(dataset_cache_dir):
                shutil.rmtree(dataset_cache_dir)
    except:
        pass
    
    # Try loading from local files first
    data_file = f"xquad_{indic_lang_code}_test.json"
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            print(f"Loaded {len(examples)} examples from local file")
            return Dataset.from_list(examples)
        except Exception as e:
            print(f"Failed to load local file {data_file}: {e}")
    
    # Try multiple loading methods
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
    
    # Debug: Print dataset structure
    print(f"Dataset structure: {dataset}")
    if isinstance(dataset, dict):
        for split_name, split_data in dataset.items():
            print(f"Split '{split_name}': {len(split_data)} items")
            if len(split_data) > 0:
                print(f"First item keys: {list(split_data[0].keys())}")
                print(f"First item sample: {split_data[0]}")
                break
    
    # Extract examples for the specific language
    extracted_examples = []
    
    # Process all splits in the dataset
    splits_to_check = dataset.keys() if isinstance(dataset, dict) else ['train']
    
    for split_name in splits_to_check:
        current_split = dataset[split_name] if isinstance(dataset, dict) else dataset
        
        print(f"Processing split '{split_name}' with {len(current_split)} items")
        
        for idx, item in enumerate(current_split):
            if idx < 3:  # Debug first few items
                print(f"Debug item {idx}: {item}")
            
            # Check if this item is for our target language
            item_lang = None
            
            # Method 1: Direct lang field
            if 'lang' in item:
                item_lang = item['lang']
            elif 'language' in item:
                item_lang = item['language']
            elif 'examples' in item and isinstance(item['examples'], dict):
                if 'lang' in item['examples']:
                    item_lang = item['examples']['lang']
                elif 'language' in item['examples']:
                    item_lang = item['examples']['language']
            
            # Method 2: Check nested structure
            if not item_lang and 'examples' in item:
                examples_data = item['examples']
                if isinstance(examples_data, dict):
                    for key in ['lang', 'language']:
                        if key in examples_data:
                            item_lang = examples_data[key]
                            break
            
            if item_lang == indic_lang_code:
                # Extract the actual data
                if 'examples' in item and isinstance(item['examples'], dict):
                    example_data = item['examples']
                else:
                    example_data = item
                
                # Extract context, question, and answer
                context = example_data.get('context', '')
                question = example_data.get('question', '')
                
                # Extract answer using robust method
                answer = extract_answer_from_xquad_item(example_data, debug=(idx < 3))
                
                if context and question:  # Only add if we have context and question
                    extracted_examples.append({
                        'context': context,
                        'question': question,
                        'answer': answer,
                        'lang': indic_lang_code
                    })
                    
                    if len(extracted_examples) <= 3:
                        print(f"Example {len(extracted_examples)}: Q='{question[:50]}...', A='{answer}'")
    
    if not extracted_examples:
        raise ValueError(f"No examples found for {indic_lang_code}")
    
    print(f"    Found {len(extracted_examples)} examples for {indic_lang_code}")
    
    # Validate that we have non-empty answers
    non_empty_answers = sum(1 for ex in extracted_examples if ex['answer'].strip())
    print(f"    Examples with non-empty answers: {non_empty_answers}/{len(extracted_examples)}")
    
    if non_empty_answers == 0:
        print("    ⚠️  WARNING: No examples have non-empty answers!")
        # Print first few examples for debugging
        for i, ex in enumerate(extracted_examples[:3]):
            print(f"    Example {i+1}: Q='{ex['question'][:50]}...', A='{ex['answer']}'")
    
    # Save to local file for future use
    try:
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_examples, f, ensure_ascii=False, indent=2)
        print(f"    Saved to {data_file}")
    except Exception as e:
        print(f"    Warning: Could not save to {data_file}: {e}")
    
    return Dataset.from_list(extracted_examples)

def batch_qa_generate(model, tokenizer, prompts, accelerator, max_new_tokens=128):
    """Perform batch question answering using multiple GPUs"""
    answers = []
    
    with tqdm(total=len(prompts), desc="Generating answers", disable=not accelerator.is_main_process) as pbar:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_answers = []
            
            for prompt in batch_prompts:
                try:
                    # Tokenize
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024  # Increased for QA context
                    )
                    
                    # Move to device
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,  
                            temperature=0.3,  # Lower temperature for factual QA
                            top_p=0.9,
                            num_beams=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            early_stopping=True,
                            use_cache=True
                        )
                    
                    # Decode only the new tokens (exclude input)
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    # Clean up answer
                    answer = clean_qa_output(answer)
                    batch_answers.append(answer)
                    
                except Exception as e:
                    print(f"Error in QA generation: {e}")
                    batch_answers.append("[QA ERROR]")
            
            answers.extend(batch_answers)
            pbar.update(len(batch_prompts))
            
            # Clear cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return answers

def clean_qa_output(text):
    """Clean QA model output"""
    text = text.strip()
    
    # Split by newlines and take the first non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        text = lines[0]
    
    # Remove common QA artifacts
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
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Stop at common continuation patterns
    stop_patterns = [
        r'\n\n',
        r'\nContext:',
        r'\nQuestion:',
        r'\nAnswer:',
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    
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

def evaluate_qa_metrics(predictions, references):
    """Evaluate QA predictions using F1 and EM metrics"""
    if not predictions or not references or len(predictions) != len(references):
        return {"f1": 0.0, "exact_match": 0.0}
    
    f1_scores = []
    em_scores = []
    
    for pred, ref in zip(predictions, references):
        # Handle multiple reference answers (XQuAD typically has one answer)
        if isinstance(ref, list):
            # Take the best score across all reference answers
            max_f1 = max(f1_score(pred, r) for r in ref)
            max_em = max(exact_match_score(pred, r) for r in ref)
        else:
            max_f1 = f1_score(pred, ref)
            max_em = exact_match_score(pred, ref)
        
        f1_scores.append(max_f1)
        em_scores.append(max_em)
    
    return {
        "f1": np.mean(f1_scores) * 100,  # Convert to percentage
        "exact_match": np.mean(em_scores) * 100  # Convert to percentage
    }

def evaluate_language_qa(accelerator, model, tokenizer, indic_lang_code):
    """Evaluate QA for a specific language"""
    
    lang_full_name = LANGUAGE_FULL_NAMES.get(indic_lang_code, indic_lang_code.upper())
    display_config_name = f"xquad_{indic_lang_code}"
    
    if accelerator.is_main_process:
        print(f"\n--- Evaluating: {lang_full_name} QA (as {display_config_name}) ---")
    
    try:
        # Load dataset using robust method
        if accelerator.is_main_process:
            print(f"  Loading XQuAD dataset for {indic_lang_code}...")
        current_dataset_for_eval = load_xquad_data_robust(indic_lang_code)
        
        # Sample subset
        num_to_sample = min(NUM_SAMPLES_PER_LANG, len(current_dataset_for_eval))
        if num_to_sample == 0:
            return {"f1": 0.0, "exact_match": 0.0, "info": "0 samples evaluated"}, None
            
        subset_to_eval = current_dataset_for_eval.select(range(num_to_sample))
        
        if accelerator.is_main_process:
            print(f"  Evaluating on {len(subset_to_eval)} samples")

        # Prepare data for batch processing
        prompts = []
        references = []
        detailed_logs = []
        
        for example_idx, example_data in tqdm(
            enumerate(subset_to_eval),
            total=len(subset_to_eval),
            desc=f"Preparing {indic_lang_code} examples",
            disable=not accelerator.is_main_process
        ):
            context = example_data.get("context", "")
            question = example_data.get("question", "")
            answer = example_data.get("answer", "")
            
            if not isinstance(context, str) or not isinstance(question, str):
                continue
            
            # Ensure answer is a string
            if not isinstance(answer, str):
                answer = str(answer) if answer is not None else ""
            
            # Skip examples with empty essential fields
            if not context.strip() or not question.strip():
                continue
            
            # Format prompt for Sarvam
            prompt = format_qa_prompt_sarvam(context, question, lang_full_name)
            
            prompts.append(prompt)
            references.append(answer)
            
            detailed_logs.append({
                "pair_key": display_config_name,
                "example_index": example_idx,
                "context": context,
                "question": question,
                "reference_answer": answer,
                "model_answer": "",  # Will be filled later
                "prompt_used": prompt
            })
        
        if not prompts:
            return {"f1": 0.0, "exact_match": 0.0, "info": "No valid examples"}, None
        
        # Check reference quality
        non_empty_refs = sum(1 for ref in references if ref.strip())
        if accelerator.is_main_process:
            print(f"  References: {non_empty_refs}/{len(references)} non-empty")
            if non_empty_refs < len(references) * 0.5:
                print("  ⚠️  WARNING: Many empty reference answers detected!")
        
        # Batch generate answers
        if accelerator.is_main_process:
            print(f"  Starting batch QA generation for {len(prompts)} examples...")
        
        predictions = batch_qa_generate(model, tokenizer, prompts, accelerator, MAX_NEW_TOKENS_FOR_QA)
        
        # Update detailed logs
        for i, prediction in enumerate(predictions):
            if i < len(detailed_logs):
                detailed_logs[i]["model_answer"] = prediction
        
        # Debug: Print first few examples
        if accelerator.is_main_process and len(detailed_logs) > 0:
            print(f"\n  Debug - First QA example for {indic_lang_code}:")
            print(f"  Context: {detailed_logs[0]['context'][:100]}...")
            print(f"  Question: {detailed_logs[0]['question']}")
            print(f"  Reference: '{detailed_logs[0]['reference_answer']}'")
            print(f"  Model answer: '{detailed_logs[0]['model_answer']}'")

        # Compute metrics
        if predictions and references:
            # Filter out pairs where BOTH prediction and reference are empty
            valid_pairs = []
            for p, r in zip(predictions, references):
                if isinstance(p, str) and isinstance(r, str):
                    # Keep pairs where either prediction or reference is non-empty
                    if p.strip() or r.strip():
                        valid_pairs.append((p.strip(), r.strip()))
            
            if not valid_pairs:
                if accelerator.is_main_process:
                    print(f"  ⚠️ {display_config_name}: No valid predictions after filtering")
                    print(f"     Total predictions: {len(predictions)}, Total references: {len(references)}")
                    print(f"     Non-empty predictions: {sum(1 for p in predictions if str(p).strip())}")
                    print(f"     Non-empty references: {sum(1 for r in references if str(r).strip())}")
                return {"f1": 0.0, "exact_match": 0.0, "info": "No valid predictions after filtering"}, detailed_logs
            
            valid_predictions, valid_references = zip(*valid_pairs)
            
            # Calculate QA metrics
            metrics = evaluate_qa_metrics(valid_predictions, valid_references)
            
            # Add individual scores to detailed logs
            for i, (pred, ref) in enumerate(zip(valid_predictions, valid_references)):
                if i < len(detailed_logs):
                    detailed_logs[i]["f1_score"] = f1_score(pred, ref) * 100
                    detailed_logs[i]["em_score"] = exact_match_score(pred, ref) * 100
            
            result = {
                'f1': metrics['f1'],
                'exact_match': metrics['exact_match'],
                'valid_samples': len(valid_pairs),
                'total_samples': len(predictions),
                'non_empty_references': sum(1 for r in valid_references if r)
            }
            
            if accelerator.is_main_process:
                print(f"  ✅ {display_config_name}: F1 = {metrics['f1']:.2f}, EM = {metrics['exact_match']:.2f}")
                print(f"     Valid samples: {len(valid_pairs)}/{len(predictions)}")
                print(f"     Non-empty refs: {result['non_empty_references']}/{len(valid_pairs)}")
                
                # Show some individual examples with scores
                print(f"  📊 Sample scores for {display_config_name}:")
                for i in range(min(3, len(detailed_logs))):
                    if 'f1_score' in detailed_logs[i]:
                        print(f"    Example {i+1}: F1={detailed_logs[i]['f1_score']:.1f}, EM={detailed_logs[i]['em_score']:.1f}")
                        print(f"      Q: {detailed_logs[i]['question'][:80]}...")
                        print(f"      Ref: '{detailed_logs[i]['reference_answer'][:50]}...'")
                        print(f"      Pred: '{detailed_logs[i]['model_answer'][:50]}...'")
            
            # Save detailed results with individual scores
            if accelerator.is_main_process:
                lang_output_filename = os.path.join(OUTPUT_DIR, f"{display_config_name}_detailed_results.json")
                try:
                    # Add summary to the JSON file
                    results_to_save = {
                        "summary": {
                            "language": indic_lang_code,
                            "language_full_name": lang_full_name,
                            "total_samples": len(predictions),
                            "valid_samples": len(valid_pairs),
                            "non_empty_references": result['non_empty_references'],
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
    # Setup accelerator
    accelerator = setup_accelerator()
    
    # Set 85% memory limit on all GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.85, device=i)
    
    # Load Sarvam model
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
    
    # Prepare model for distributed training
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
        display_name = f"xquad_{indic_lang_code}"
        lang_iter.set_postfix({"lang": indic_lang_code})
        
        result, detailed_logs = evaluate_language_qa(
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
            f"all_lang_xquad_eval_sarvam_multi_gpu.json"
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
                "dataset": "XQuAD-IN",
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
        print("🏆 Final XQuAD-IN Evaluation Summary - SARVAM MODEL 🏆")
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
    try:
        main()
    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        traceback.print_exc()
        # Ensure all processes are properly terminated
        if dist.is_initialized():
            dist.destroy_process_group()