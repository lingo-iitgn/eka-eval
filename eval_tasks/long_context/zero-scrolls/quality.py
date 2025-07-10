import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset

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
    max_new_tokens=128,      # Allow for reasoning + answer
    do_sample=False,
    temperature=0.0,
    return_full_text=False
)

# Load dataset
dataset = load_dataset('tau/zero_scrolls', 'quality', split='validation', trust_remote_code=True)

# Prompt template as you specified
prompt_template = lambda example: (
    f"You are a multiple-choice question expert who provides accurate answers. "
    f"For each question, give a brief reasoning followed by your final answer as a single letter (A, B, C, or D) on a new line.\n"
    f"Context: {example['input']}\n"
    f"Answer:"
)

def extract_letter(output):
    # Look for a single letter A-D on its own line, preferably at the end
    lines = output.strip().split('\n')
    for line in reversed(lines):
        match = re.match(r"^\(?([A-Da-d])[\).]?\)?\.?$", line.strip())
        if match:
            return match.group(1).upper()
    # Fallback: search anywhere
    match = re.search(r"\b([A-Da-d])\b", output)
    if match:
        return match.group(1).upper()
    return ""

results = []
total_em = 0

for example in tqdm(dataset, desc="Evaluating"):
    prompt = prompt_template(example)
    ref_answer = example['output'].strip().upper()
    output = pipe(prompt)[0]['generated_text'].strip()
    pred_answer = extract_letter(output)
    em_score = int(pred_answer == ref_answer)
    total_em += em_score
    results.append({
        "prompt": prompt,
        "model_output": output,
        "prediction": pred_answer,
        "reference": ref_answer,
        "em_score": bool(em_score)
    })

with open("quality_evaluation_results_roleprompt.json", "w") as f:
    json.dump(results, f, indent=2)


# Optionally, print a few failed cases for debugging
for ex in results:
    if not ex['em_score']:
        print(f"Prompt: {ex['prompt']}\nModel Output: {ex['model_output']}\nPrediction: {ex['prediction']}\nReference: {ex['reference']}\n")
        break  # Remove break to see more
    
avg_em = total_em / len(dataset)
print(f"\nAverage Exact Match Score: {avg_em:.2%}")
