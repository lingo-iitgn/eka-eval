import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import evaluate
import torch

# ========== Force to use GPU 1 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ========== Model setup ==========
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "30GB"}
)

# ========== Chat Template Prompt ==========
def qasper_prompt_template(example):
    input_text = example['input']
    parts = input_text.split('\n\n', 1)
    if len(parts) == 2:
        question, article = parts
    else:
        question, article = input_text, ""
    article_tokens = tokenizer.encode(article, max_length=3500, truncation=True)
    article_trunc = tokenizer.decode(article_tokens)
    return (
        "You are an expert at answering questions about scientific research papers. "
        "Given the article and the question, answer as concisely as possible using a single phrase or sentence. "
        "If the question cannot be answered based on the article, reply with \"unanswerable\". "
        "If it is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\n"
        f"Article:\n{article_trunc.strip()}\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


# ========== Output Extraction ==========
def extract_answer(output):
    # Llama-3 chat models typically end completions with <|eot_id|> or EOS
    answer = output.split(tokenizer.eos_token, 1)[0]
    answer = answer.split("<|eot_id|>", 1)[0]
    # Take only the first non-empty line
    for line in answer.strip().split('\n'):
        line = line.strip()
        if line:
            return line
    return ""

# ========== Load QASPER dataset ==========
qasper_dataset = load_dataset('tau/zero_scrolls', 'qasper', split='validation', trust_remote_code=True)

# ========== Load SQuAD-style QA metric ==========
qa_metric = evaluate.load("squad")

# ========== Pipeline ==========
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    max_new_tokens=60,     # Allow a bit more room for answers
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    return_full_text=False,
    batch_size=1,
    pad_token_id=tokenizer.eos_token_id
)

# ========== Evaluation Loop ==========
results = []
predictions = []
references = []

for idx, example in enumerate(tqdm(qasper_dataset, desc="Evaluating")):
    prompt = qasper_prompt_template(example)
    ref_answer = example['output'].strip()
    try:
        output = pipe(prompt)[0]['generated_text'].strip()
        pred_answer = extract_answer(output)
    except Exception as e:
        print(f"Error at example {idx}: {e}")
        pred_answer = ""
        output = ""
    predictions.append({"prediction_text": pred_answer, "id": str(idx)})
    references.append({"answers": {"text": [ref_answer], "answer_start": [0]}, "id": str(idx)})
    results.append({
        "prompt": prompt,
        "model_output": output,
        "prediction": pred_answer,
        "reference": ref_answer
    })

# ========== Compute SQuAD-style metrics ==========
scores = qa_metric.compute(predictions=predictions, references=references)

# ========== Save results ==========
with open("qasper_evaluation_results_optimised.json", "w") as f:
    json.dump({"results": results, "qa_scores": scores}, f, indent=2)

# ========== Print F1 and EM scores ==========
print(f"\nSQuAD-style QA Metrics for QASPER dataset:")
print(f"Exact Match: {scores['exact_match']:.2f}%")
print(f"F1 Score: {scores['f1']:.2f}%")
