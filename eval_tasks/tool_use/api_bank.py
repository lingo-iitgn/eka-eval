import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import evaluate

# Load input and reference files
with open("api_bank_data/level-1-api.json") as f1, open("api_bank_data/level-1-response.json") as f2:
    inputs = json.load(f1)
    refs = json.load(f2)

ref_by_id = {ex["id"]: ex for ex in refs}
paired_data = [(inp, ref_by_id[inp["id"]]) for inp in inputs if inp["id"] in ref_by_id]

print(f"Matched {len(paired_data)} examples")
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
    max_new_tokens=200,
    do_sample=False,
    return_full_text=False
)
exact_match_metric = evaluate.load("exact_match")

def normalize_api_call(text):
    """Normalize API call string for fair comparison:
    - lowercase
    - strip whitespace
    - collapse multiple spaces
    - strip trailing punctuation
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".;")
    return text

predictions = []
references = []

for inp, ref in tqdm(paired_data):
    prompt = inp["instruction"] + "\n" + inp["input"]
    generated = pipe(prompt)[0]["generated_text"]
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()

    gen_norm = normalize_api_call(generated)
    ref_norm = normalize_api_call(ref["expected_output"])

    predictions.append(gen_norm)
    references.append(ref_norm)
results = exact_match_metric.compute(predictions=predictions, references=references)
print(f"Exact Match Accuracy: {results['exact_match']:.2f}%")
