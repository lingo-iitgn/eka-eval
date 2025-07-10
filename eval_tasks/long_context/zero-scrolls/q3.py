import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

# Model setup
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    max_new_tokens=20,
    do_sample=False,
    return_full_text=False
)

# Load dataset
dataset = load_dataset('tau/zero_scrolls', 'quality', split='validation', trust_remote_code=True)
em_metric = load_metric("exact_match")

results = []
total_em = 0

for example in tqdm(dataset, desc="Evaluating"):
    system_message="You are an expert at reading and understaning context and answering multiple choice question with a single letter"
    prompt = example['input']
    ref_answer = example['output'].strip()
    
    # Generate response
    try:
        output = pipe(prompt)[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error generating for example: {e}")
        output = ""
    
    # Extract prediction
    match = re.search(r"\b([A-D])\b", output)
    pred_answer = match.group(1) if match else ""
    
    # Calculate EM
    em_score = em_metric.compute(predictions=[pred_answer], references=[ref_answer])['exact_match']
    total_em += em_score
    
    # Store results
    results.append({
        "prompt": prompt,
        "model_output": output,
        "prediction": pred_answer,
        "reference": ref_answer,
        "em_score": bool(em_score)  
    })

# Save results
with open("quality_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Calculate final metrics
avg_em = total_em / len(dataset)
print(f"\nAverage Exact Match Score: {avg_em:.2%}")
print(f"Results saved to quality_evaluation_results.json")
