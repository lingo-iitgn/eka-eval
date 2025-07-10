import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import evaluate
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
    max_new_tokens=250,      # Allow for detailed summaries
    do_sample=False,
    temperature=0.0,
    return_full_text=False
)

# Load squality dataset
squality_dataset = load_dataset('tau/zero_scrolls', 'squality', split='validation', trust_remote_code=True)

# Load ROUGE metric
rouge_metric = evaluate.load("rouge")

# Prompt template for summarization
prompt_template = lambda example: (
    "You are an expert at providing summaries of stories based on the guiding question. "
    "You provide relevant and accurate summaries.\n"
    f"Context: {example['input']}\n"
    "Answer:"
)

results = []
predictions = []
references = []

for example in tqdm(squality_dataset, desc="Evaluating"):
    prompt = prompt_template(example)
    ref_answer = example['output'].strip()
    output = pipe(prompt)[0]['generated_text'].strip()
    
    predictions.append(output)
    references.append(ref_answer)
    
    results.append({
        "prompt": prompt,
        "model_output": output,
        "reference": ref_answer
    })

# Compute ROUGE scores
rouge_scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)

# Save results
with open("squality_evaluation_results.json", "w") as f:
    json.dump({"results": results, "rouge_scores": rouge_scores}, f, indent=2)

# Print ROUGE scores
# Print a sample failure case
print("\nSample failure case:")
for ex in results:
    if ex['model_output'] != ex['reference']:
        print(f"Prompt: {ex['prompt']}")
        print(f"Model Output: {ex['model_output']}")
        print(f"Reference: {ex['reference']}")
        break

print("\nROUGE Scores for squality dataset:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.2%}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.2%}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.2%}")
print(f"ROUGE-Lsum: {rouge_scores['rougeLsum']:.2%}")

