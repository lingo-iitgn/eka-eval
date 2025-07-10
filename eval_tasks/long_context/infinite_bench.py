from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import json
import re
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# Model setup
model_name = "meta-llama/Meta-Llama-3-8B"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,  # Use float16 for better memory efficiency
    low_cpu_mem_usage=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    max_new_tokens=200,
    do_sample=False,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

# Available splits in InfiniteBench
available_splits = [
    'passkey',
    'kv_retrieval', 
    'number_string',
    'code_run',
    'longdialogue_qa_eng',
    'longbook_qa_eng',
    'longbook_choice_eng',
    'longbook_sum_eng',
    'math_calc',
    'math_find'
]

def evaluate_split(split_name):
    """Evaluate a specific split of InfiniteBench"""
    print(f"\n=== Evaluating {split_name} ===")
    
    try:
        # Load specific split
        dataset = load_dataset(
            'xinrongzhang2022/InfiniteBench',
            split=split_name,
            trust_remote_code=True
        )
        
        if len(dataset) == 0:
            print(f"Split {split_name} is empty, skipping...")
            return None
            
        print(f"Loaded {len(dataset)} examples")
        
    except Exception as e:
        print(f"Error loading split {split_name}: {e}")
        return None
    
    # Initialize metrics based on task type
    if 'qa' in split_name or 'choice' in split_name:
        qa_metric = evaluate.load("squad")
    else:
        qa_metric = None
    
    def extract_answer(output, task_type):
        """Extract answer based on task type"""
        output = output.strip()
        
        if task_type == 'passkey':
            # Extract number from output
            match = re.search(r'\b\d+\b', output)
            return match.group() if match else output.split()[0] if output.split() else ""
        
        elif task_type == 'kv_retrieval':
            # Extract the value after "The value is" or similar patterns
            patterns = [r'value is (.+?)(?:\.|$)', r'answer is (.+?)(?:\.|$)', r'^(.+?)(?:\.|$)']
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return output.split()[0] if output.split() else ""
        
        elif task_type == 'number_string':
            # Extract number from output
            match = re.search(r'\b\d+\b', output)
            return match.group() if match else ""
        
        elif task_type == 'code_run':
            # Extract the final result/output
            lines = output.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
            return output.strip()
        
        elif 'qa' in task_type:
            # Extract first complete sentence or first line
            match = re.search(r"^(.*?[.!?])", output)
            return match.group(1) if match else output.split('\n')[0]
        
        elif 'choice' in task_type:
            # Extract A, B, C, D choice
            match = re.search(r'\b([A-D])\b', output)
            return match.group(1) if match else output.strip()[:1]
        
        elif 'sum' in task_type:
            # Return first paragraph as summary
            return output.split('\n\n')[0] if '\n\n' in output else output[:500]
        
        else:
            # Default: return first line or sentence
            match = re.search(r"^(.*?[.!?\n])", output)
            return match.group(1).strip() if match else output.strip()
    
    def create_prompt(example, split_name):
        """Create appropriate prompt based on task type"""
        
        if split_name == 'passkey':
            return f"{example['input']}\nWhat is the pass key? The pass key is"
        
        elif split_name == 'kv_retrieval':
            return f"{example['input']}\nAnswer:"
        
        elif split_name == 'number_string':
            return f"{example['input']}\nAnswer:"
        
        elif split_name == 'code_run':
            return f"Execute this code and provide the output:\n{example['input']}\nOutput:"
        
        elif 'qa' in split_name:
            # Parse context and question from input
            if "Question: " in example['input']:
                input_parts = example['input'].split("\nQuestion: ")
                context = input_parts[0].strip()
                question = input_parts[1].split("\nAnswer:")[0].strip()
                return f"Read the following text and answer the question.\n\nText: {context}\n\nQuestion: {question}\nAnswer:"
            else:
                return f"{example['input']}\nAnswer:"
        
        elif 'choice' in split_name:
            return f"{example['input']}\nAnswer:"
        
        elif 'sum' in split_name:
            return f"Summarize the following text:\n{example['input']}\nSummary:"
        
        elif split_name in ['math_calc', 'math_find']:
            return f"{example['input']}\nAnswer:"
        
        else:
            return f"{example['input']}\nAnswer:"
    
    results = []
    correct = 0
    total = 0
    
    # Limit evaluation to first 100 examples for faster testing
    eval_dataset = dataset.select(range(min(100, len(dataset))))
    
    for example in tqdm(eval_dataset, desc=f"Evaluating {split_name}"):
        prompt = create_prompt(example, split_name)
        reference = example['output'].strip()
        
        try:
            # Generate response
            output = pipe(prompt)[0]['generated_text'].strip()
            prediction = extract_answer(output, split_name)
            
            # Simple exact match for most tasks
            is_correct = prediction.lower().strip() == reference.lower().strip()
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error on example: {e}")
            prediction = ""
            is_correct = False
            total += 1
        
        results.append({
            "input": example['input'][:200] + "..." if len(example['input']) > 200 else example['input'],
            "prediction": prediction,
            "reference": reference,
            "correct": is_correct,
            "raw_output": output[:100] + "..." if len(output) > 100 else output
        })
    
    # Calculate scores
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # For QA tasks, also calculate F1 and EM using SQuAD metric
    squad_scores = None
    if qa_metric and 'qa' in split_name:
        try:
            predictions_squad = [{"prediction_text": r["prediction"], "id": str(i)} for i, r in enumerate(results)]
            references_squad = [{"answers": {"text": [r["reference"]], "answer_start": [0]}, "id": str(i)} for i, r in enumerate(results)]
            squad_scores = qa_metric.compute(predictions=predictions_squad, references=references_squad)
        except Exception as e:
            print(f"Error computing SQuAD scores: {e}")
    
    # Save results
    output_data = {
        "split": split_name,
        "total_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }
    
    if squad_scores:
        output_data["squad_scores"] = squad_scores
    
    filename = f"infinitebench_{split_name}_results.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print results
    print(f"\nResults for {split_name}:")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if squad_scores:
        print(f"F1 Score: {squad_scores['f1']:.2f}")
        print(f"Exact Match: {squad_scores['exact_match']:.2f}")
    
    # Show sample results
    print(f"\nSample results from {split_name}:")
    for i, r in enumerate(results[:3]):
        print(f"Example {i+1}:")
        print(f"  Input: {r['input']}")
        print(f"  Prediction: {r['prediction']}")
        print(f"  Reference: {r['reference']}")
        print(f"  Correct: {r['correct']}")
        print()
    
    return output_data

# Main evaluation
if __name__ == "__main__":
    print("Starting InfiniteBench evaluation...")
    
    # You can specify which splits to evaluate here
    # Comment out splits you don't want to evaluate
    splits_to_evaluate = [
        'passkey',
        'kv_retrieval',
        'number_string', 
        'longdialogue_qa_eng',
        # 'code_run',  # Uncomment if you want to evaluate code tasks
        # 'longbook_qa_eng',  # These are very long, uncomment if needed
        # 'math_calc',
    ]
    
    all_results = {}
    
    for split in splits_to_evaluate:
        try:
            result = evaluate_split(split)
            if result:
                all_results[split] = {
                    "accuracy": result["accuracy"],
                    "total_examples": result["total_examples"],
                    "correct": result["correct"]
                }
        except Exception as e:
            print(f"Failed to evaluate {split}: {e}")
            continue
    
    # Save summary
    with open("infinitebench_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    for split, scores in all_results.items():
        print(f"{split:25}: {scores['accuracy']:6.2f}% ({scores['correct']}/{scores['total_examples']})")
    
    if all_results:
        avg_accuracy = sum(scores['accuracy'] for scores in all_results.values()) / len(all_results)
        print(f"{'Average Accuracy':25}: {avg_accuracy:6.2f}%")
    
    print("\nEvaluation completed! Check individual JSON files for detailed results.")