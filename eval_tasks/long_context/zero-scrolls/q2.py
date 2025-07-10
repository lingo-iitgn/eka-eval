from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from evaluate import load as load_metric

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
    max_new_tokens=20,  # Only need the answer letter
    do_sample=False,
    return_full_text=False
)

# Load one example
dataset = load_dataset('tau/zero_scrolls', 'quality', split='validation', trust_remote_code=True)
example = dataset[1]

# Use the full input as prompt (includes story, question, and options)
prompt = example['input']

# Generate answer
output = pipe(prompt)[0]['generated_text'].strip()

# Extract the predicted letter (A, B, C, or D)
import re
match = re.search(r"\b([A-D])\b", output)
pred = match.group(1) if match else ""

# Reference answer (single letter)
ref = example['output'].strip()

# Exact match metric
em_metric = load_metric("exact_match")
em_score = em_metric.compute(predictions=[pred], references=[ref])['exact_match']

print(f"Prompt:\n{prompt}\n")
print(f"Model output: {output}")
print(f"Predicted: {pred}, Reference: {ref}")
print(f"Exact Match Score: {em_score}")
