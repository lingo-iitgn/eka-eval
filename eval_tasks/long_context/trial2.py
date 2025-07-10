from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import torch
from tqdm import tqdm
import json
import sys
import os# For path joining if saving outputs

# --- Model Setup ---
# " # Keep if you have access
model_name="meta-llama/Meta-Llama-3-8B"# Using Gemma for broader testability
print(f"Loading tokenizer for: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    device_map="auto", 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
)
print(f"Model loaded on device: {model.device}")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    # device_map="auto", # Already handled by model's device_map
    trust_remote_code=True, 
    max_new_tokens=1000, # Default max new tokens
    do_sample=False,    # Greedy decoding by default
    # temperature=0.6,    # Only if do_sample=True
    # top_p=0.9,          # Only if do_sample=True
    return_full_text=False # Get only the generated part
)
print("Pipeline created.")

# --- Metrics and Prompt Templates ---
task_metrics = {
    "squality": evaluate.load("rouge"),  # SQuality is extractive QA, ROUGE can be a proxy, F1/EM better
    "qasper": evaluate.load("squad"),    # QASPER is QA, SQuAD metric (EM/F1) is appropriate
    "quality": evaluate.load("accuracy"),# QUALITY is MCQA
    "gov_report": evaluate.load("rouge"),
    # Add other ZeroScrolls tasks here
}

def get_input_text(example):
    # ZeroScrolls loader should provide 'input'. Fallback for safety.
    return example.get('input', example.get('article', example.get('context', '')))

def get_question_text(example):
    # For QA, question might be in 'query', 'question', or part of 'input'
    q = example.get('query', example.get('question'))
    if q: return q
    # If 'input' contains "Question:", try to extract it (very basic)
    if isinstance(example.get('input'), str) and "Question:" in example['input']:
        parts = example['input'].split("Question:")
        if len(parts) > 1: return parts[1].split("Answer:")[0].strip()
    return "" # No clear question found

prompt_templates = {
    "summarization": lambda x: f"Summarize the following text:\n\n{get_input_text(x)}\n\nSummary:",
    "qa": lambda x: f"Answer the question based on the following context.\n\nContext:\n{get_input_text(x)}\n\nQuestion:\n{get_question_text(x)}\n\nAnswer:",
    # For QASPER, the 'input' from ZeroScrolls loader often IS the prompt.
    "qasper_direct": lambda x: f"{get_input_text(x)}", # Assumes 'input' ends with 'Answer:'
    "mc_qa": lambda x: f"Read the following and choose the correct answer (A, B, C, or D).\n\nContext:\n{get_input_text(x)}\n\nQuestion:\n{get_question_text(x)}\n\nOptions:\n{x.get('options', 'No options provided')}\n\nAnswer (letter only):",
}

# Datasets to load (using a subset for faster testing)
datasets_info = {
    "squality": ("tau/zero_scrolls", "squality", "qa"), # Using generic QA for now
    "qasper": ("tau/zero_scrolls", "qasper", "qasper_direct"), # Use direct input for qasper
    "quality": ("tau/zero_scrolls", "quality", "mc_qa"),
    "gov_report": ("tau/zero_scrolls", "gov_report", "summarization"),
}

results_summary = {}
MAX_EXAMPLES_PER_TASK_FOR_DEBUG = 3 # Limit examples for faster debugging

for task_name, (dataset_id, config_name, task_type) in datasets_info.items():
    print(f"\n--- Running evaluation for: {task_name} (Task Type: {task_type}) ---")
    try:
        # Load only a slice for debugging
        dataset = load_dataset(dataset_id, name=config_name, split=f"validation[:{MAX_EXAMPLES_PER_TASK_FOR_DEBUG}]", trust_remote_code=True)
        # For some ZS configs, split might be "test" or "train"
        # dataset = load_dataset(dataset_id, name=config_name, split=f"test[:{MAX_EXAMPLES_PER_TASK_FOR_DEBUG}]", trust_remote_code=True) 
        print(f"Loaded {len(dataset)} examples for {task_name}.")
    except Exception as e:
        print(f"ERROR: Could not load dataset {dataset_id}/{config_name}. Error: {e}")
        results_summary[task_name] = {"error": str(e)}
        continue

    metric = task_metrics.get(task_name) # Get metric, might be None if not defined
    prompt_fn = prompt_templates.get(task_type)

    if not prompt_fn:
        print(f"ERROR: No prompt function for task_type '{task_type}' (task: {task_name}). Skipping.")
        results_summary[task_name] = {"error": f"No prompt_fn for {task_type}"}
        continue

    # For SQuAD-like metrics (F1/EM)
    predictions_for_metric = [] # List of {'id': str, 'prediction_text': str}
    references_for_metric = []  # List of {'id': str, 'answers': {'text': List[str], 'answer_start': List[int]}}
    
    # For ROUGE/Accuracy (simpler lists)
    simple_preds, simple_refs = [], []

    sample_output_data = {} # To store one example's details

    print(f"Generating predictions for {task_name}...")
    for i, example in enumerate(tqdm(dataset, desc=f"Eval {task_name}", file=sys.stdout)):
        if i >= MAX_EXAMPLES_PER_TASK_FOR_DEBUG: break # Redundant if slice is in load_dataset

        # --- DEBUG: Print example keys and relevant content ---
        if i == 0: 
            print(f"\nDEBUG: First example keys for {task_name}: {list(example.keys())}")
            print(f"DEBUG: Example 'id': {example.get('id')}")
            print(f"DEBUG: Example 'input' preview: {str(example.get('input'))[:200]}...")
            print(f"DEBUG: Example 'output' preview: {str(example.get('output', example.get('outputs')) )[:200]}...")
            if task_type == "qa" or task_type == "mc_qa":
                 print(f"DEBUG: Example 'query' (if exists): {example.get('query')}")
                 print(f"DEBUG: Example 'question' (if exists): {example.get('question')}")
            if task_type == "mc_qa":
                 print(f"DEBUG: Example 'options' (if exists): {example.get('options')}")
        # ---

        try:
            prompt = prompt_fn(example)
        except KeyError as e:
            print(f"\nKeyError formatting prompt for {task_name}, example {i} (ID: {example.get('id','N/A')}). Missing key: {e}")
            print(f"Available keys: {list(example.keys())}")
            # Add dummy/error prediction to maintain list lengths if needed, or skip
            if task_name == "qasper": # QASPER/SQuAD metric expects entries for all IDs
                 predictions_for_metric.append({'id': example.get('id', str(i)), 'prediction_text': "#PROMPT_ERROR"})
                 # Construct a dummy reference to match
                 raw_ref_ans = example.get('output', example.get('outputs', {"text":["#REF_ERROR"], "answer_start":[-1]}))
                 if isinstance(raw_ref_ans, dict) and 'text' in raw_ref_ans:
                     references_for_metric.append({'id': example.get('id', str(i)), 'answers': raw_ref_ans})
                 else: # Fallback if output is not in expected dict format for SQuAD
                     ref_text_list = [str(r) for r in raw_ref_ans] if isinstance(raw_ref_ans, list) else [str(raw_ref_ans)]
                     references_for_metric.append({'id': example.get('id', str(i)), 'answers': {'text': ref_text_list, 'answer_start': [-1]*len(ref_text_list)}})
            else:
                simple_preds.append("#PROMPT_ERROR")
                raw_ref_ans = example.get('output', example.get('outputs', ["#REF_ERROR"]))
                ref_text_list = [str(r) for r in raw_ref_ans] if isinstance(raw_ref_ans, list) else [str(raw_ref_ans)]
                simple_refs.append(ref_text_list[0] if len(ref_text_list)==1 else ref_text_list)

            continue # Skip to next example


        # Generate (this part seems to have worked before the KeyError)
        # Assuming pipe is configured with return_full_text=False
        try:
            # print(f"DEBUG: Prompt for {task_name} ex {i}: {prompt[:300]}...") # Verbose
            outputs = pipe(prompt, pad_token_id=tokenizer.eos_token_id) # Added pad_token_id
            prediction = outputs[0]['generated_text'].strip()
        except Exception as e_pipe:
            print(f"ERROR during pipeline generation for {task_name} ex {i}: {e_pipe}")
            prediction = "#PIPE_ERROR"
        
        # Store predictions and references based on task type for metric
        if task_name == "qasper": # SQuAD-like metric
            predictions_for_metric.append({'id': example['id'], 'prediction_text': prediction})
            # QASPER references from ZeroScrolls might be directly in 'output' or need parsing from 'answers'
            # The tau/zero_scrolls loader for qasper SHOULD provide 'answers' in squad format in example['output']
            # If example['output'] is already {'text': [...], 'answer_start': [...]}, use it.
            # If example['output'] is a string, we need to adapt.
            qasper_ref_ans = example.get('output') 
            #if isinstance(qasper_ref_ans, dict) and 'text' in qasper_ans_struct and 'answer_start' in qasper_ans_struct:
                 #references_for_metric.append({'id': example['id'], 'answers': qasper_ref_ans })
            #elif isinstance(qasper_ref_ans, str): # If output is just a string (less likely for QASPER from ZS)
                 #references_for_metric.append({'id': example['id'], 'answers': {'text': [qasper_ref_ans], 'answer_start': [-1]}})
            #else: # Fallback if output is not in expected format
                 #references_for_metric.append({'id': example['id'], 'answers': {'text': ["#REF_FORMAT_ERROR"], 'answer_start': [-1]}})

        elif task_name == "quality": # Accuracy, simple lists
            simple_preds.append(prediction) # Model should output A, B, C, or D
            simple_refs.append(example['output']) # Ground truth is A, B, C, or D
        
        else: # Default to ROUGE-like tasks (squality, gov_report)
            simple_preds.append(prediction)
            raw_ref = example.get('output', example.get('outputs', [""]))
            ref_list = [str(r) for r in raw_ref] if isinstance(raw_ref, list) else [str(raw_ref)]
            simple_refs.append(ref_list[0] if len(ref_list)==1 else ref_list) # ROUGE takes list of strings or list of lists of strings

        if i == 0: # Save first example details
            sample_output_data = {
                "task": task_name, "id": example.get('id', 'N/A'), "prompt": prompt,
                "generated": prediction, 
                "reference": references_for_metric[-1] if task_name == "qasper" else simple_refs[-1]
            }

    # Compute metric
    score_result = "N/A"
    if metric:
        try:
            if task_name == "qasper":
                if predictions_for_metric and references_for_metric:
                    score_result = metric.compute(predictions=predictions_for_metric, references=references_for_metric)
                else: score_result = {"f1": 0.0, "exact_match": 0.0, "info": "No valid preds/refs for qasper"}
            elif task_name == "quality": # Accuracy
                 # For quality, model should predict A,B,C,D. Refs are A,B,C,D.
                 # Need to parse model output if it's like "(A)"
                parsed_preds_quality = []
                for p_text in simple_preds:
                    match = re.search(r'\b([A-D])\b', p_text.upper())
                    parsed_preds_quality.append(match.group(1) if match else "Z_FAIL") # Fail if not A,B,C,D
                if parsed_preds_quality and simple_refs:
                    score_result = metric.compute(predictions=parsed_preds_quality, references=simple_refs)
                else: score_result = {"accuracy": 0.0, "info": "No valid preds/refs for quality"}
            else: # ROUGE tasks
                if simple_preds and simple_refs:
                    score_result = metric.compute(predictions=simple_preds, references=simple_refs)
                else: score_result = {"rougeL": 0.0, "info": "No valid preds/refs for rouge task"}
            results_summary[task_name] = score_result
            print(f"Score for {task_name}: {score_result}")
        except Exception as e_metric:
            print(f"ERROR computing metric for {task_name}: {e_metric}")
            results_summary[task_name] = {"error": str(e_metric)}
    else:
        print(f"No metric defined for task: {task_name}")
        results_summary[task_name] = "No metric assigned"

    # Save sample output
    output_sample_filename = f"{task_name.replace('/', '_')}_sample_output.json"
    try:
        with open(output_sample_filename, "w") as f:
            json.dump(sample_output_data, f, indent=2)
        print(f"Saved sample output to {output_sample_filename}")
    except Exception as e_json:
        print(f"Error saving sample output for {task_name}: {e_json}")


print("\n--- Summary of Evaluation Scores ---")
for task, result in results_summary.items():
    print(f"{task}: {json.dumps(result, indent=2)}")