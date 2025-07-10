from datasets import load_dataset
import evaluate
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import evaluate

# Load FLORES-IN dataset
ds = load_dataset('google/IndicGenBench_flores_in', split='validation')

# Load Sarvam-1 model and tokenizer
model_id = "sarvamai/sarvam-1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()

# ChrF evaluator
chrf = evaluate.load("chrf")

# Map FLORES ISO-639 language codes to full language names
def get_indic_language(lang_code):
    mapping = {
        "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
        "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"
    }
    return mapping.get(lang_code, "Unknown")

# Prompt templates
def build_prompt(example):
    direction = example["translation_direction"]
    lang_code = example["lang"]
    source = example["source"]
    if direction == "enxx":
        target_language = get_indic_language(lang_code)
        return f"""Translate the following:
{source} to {target_language}.

### Translation:"""
    else:
        return f"""Translate the following:
{source} to english.

### Translation:"""

# Run inference and evaluate on N examples
num_examples = 5  # adjust as needed
predictions = []
references = []

for i in range(num_examples):
    example = ds[i]['examples']
    prompt = build_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract generated translation after the prompt
    prediction = decoded.split("### Translation:")[-1].strip()

    predictions.append(prediction)
    references.append(example["target"])

    print(f"\n--- Example {i+1} ---")
    print("Prompt:", prompt)
    print("Prediction:", prediction)
    print("Reference:", example["target"])

# Compute ChrF score
results = chrf.compute(predictions=predictions, references=references)
print("\nChrF Score:", results["score"])
