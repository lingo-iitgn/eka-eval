import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import evaluate
import random # For potentially selecting random needles
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
# Choose a dataset with a context length Llama-3-8B handles well (default 8k)
# and a config with multiple needles.
DATASET_PATH = "ccdv/multi-needle-haystack-en-8k" # Context up to 8k tokens
DATASET_CONFIG_NAME = "5-needles" # This config inserts 5 needles

# How many needles to target for retrieval in each example
# This fulfills the "retrieve 2 needles" requirement.
N_TARGET_NEEDLES_TO_RETRIEVE = 2

# Number of examples from the dataset to process (set to None for all)
NUM_EXAMPLES_TO_TEST = 5 # Small number for quick testing
# For full evaluation, set to: len(dataset) or None

# SQuAD F1 threshold for considering a needle "retrieved"
F1_THRESHOLD_FOR_RECALL = 50.0 # SQuAD F1 scores are 0-100

# --- Prompt Template (based on your SIMPLE_TEMPLATE) ---
PROMPT_TEMPLATE = """You are a helpful AI bot that answers questions for a user. Keep your responses short and direct.
The following is a set of context and a question that will relate to the context.
#CONTEXT
{context}
#ENDCONTEXT

#QUESTION
{question} Don’t give information outside the document or repeat your findings. If the information is not available in the context respond UNANSWERABLE.
#ANSWER:"""

# --- Model and Tokenizer Setup ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

print(f"Loading model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto" # uses bfloat16 if available, float16 otherwise
)
print("Model loaded.")

print("Creating text-generation pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    max_new_tokens=150,      # Max length of the generated answer
    do_sample=False,         # For deterministic output
    # temperature=0.0,       # Effectively set by do_sample=False
    return_full_text=False   # Only get the generated part
)
print("Pipeline created.")

# --- Load Dataset ---
print(f"Loading dataset {DATASET_PATH}, config: {DATASET_CONFIG_NAME}...")
try:
    dataset = load_dataset(DATASET_PATH, name=DATASET_CONFIG_NAME, split='test', trust_remote_code=True)
    print(f"Dataset loaded. Number of examples: {len(dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the dataset path and config name are correct and you have internet access.")
    print(f"Available configs for {DATASET_PATH} might need checking on Hugging Face Hub.")
    exit()

# --- Load SQuAD Metric ---
qa_metric = evaluate.load("squad")

# --- Evaluation Loop ---
results_log = []
total_questions_asked_overall = 0
total_needles_retrieved_em_overall = 0
total_needles_retrieved_f1_overall = 0
sum_of_all_f1_scores = 0.0

# Determine the number of examples to run
if NUM_EXAMPLES_TO_TEST is None or NUM_EXAMPLES_TO_TEST > len(dataset):
    num_to_run = len(dataset)
else:
    num_to_run = NUM_EXAMPLES_TO_TEST

print(f"Starting evaluation on {num_to_run} examples, targeting {N_TARGET_NEEDLES_TO_RETRIEVE} needles per example...")

for i, example in tqdm(enumerate(dataset.select(range(num_to_run))), total=num_to_run, desc="Evaluating Examples"):
    context = example['context']
    all_retrieval_questions = example['retrieval_questions'] # List of questions
    all_true_answers = example['true_answers']           # List of lists of answer strings
    num_needles_available_in_example = example['num_needles']

    if num_needles_available_in_example < N_TARGET_NEEDLES_TO_RETRIEVE:
        print(f"Warning: Example {i} has only {num_needles_available_in_example} needles, "
              f"but {N_TARGET_NEEDLES_TO_RETRIEVE} are targeted. Skipping this example.")
        continue

    # Select a subset of N_TARGET_NEEDLES_TO_RETRIEVE questions/answers
    # For simplicity, let's pick the first N_TARGET_NEEDLES_TO_RETRIEVE.
    # You could also use random.sample(range(num_needles_available_in_example), N_TARGET_NEEDLES_TO_RETRIEVE)
    # to pick random needles to target each time.
    indices_to_target = list(range(N_TARGET_NEEDLES_TO_RETRIEVE))

    example_details = {
        "example_id": i,
        "num_needles_available": num_needles_available_in_example,
        "num_needles_targeted": N_TARGET_NEEDLES_TO_RETRIEVE,
        "questions": []
    }

    current_example_retrieved_em = 0
    current_example_retrieved_f1_count = 0
    current_example_f1_sum = 0.0

    for needle_idx_in_target_set, original_needle_idx in enumerate(indices_to_target):
        question = all_retrieval_questions[original_needle_idx]
        # SQuAD metric expects answers in the format: {'text': ['ans1', 'ans2'], 'answer_start': [0, 0]}
        # The dataset 'true_answers' is already a list of strings for each question.
        true_answer_texts_for_squad = all_true_answers[original_needle_idx]

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            model_output_list = pipe(prompt)
            if model_output_list and isinstance(model_output_list, list) and 'generated_text' in model_output_list[0]:
                model_answer = model_output_list[0]['generated_text'].strip()
            else:
                model_answer = "[Pipeline Error - No Text]"
                print(f"Warning: Pipeline output error for example {i}, needle {original_needle_idx}")

        except Exception as e:
            print(f"Error during text generation for example {i}, needle {original_needle_idx}: {e}")
            model_answer = "[Generation Error]"

        # Prepare for SQuAD metric
        # Each call to compute needs unique IDs if you were to batch them, but here we do one by one.
        squad_prediction = [{"prediction_text": model_answer, "id": str(original_needle_idx)}]
        squad_reference = [{"answers": {"text": true_answer_texts_for_squad,
                                        "answer_start": [0] * len(true_answer_texts_for_squad)},
                            "id": str(original_needle_idx)}]

        scores = qa_metric.compute(predictions=squad_prediction, references=squad_reference)
        em_score = scores['exact_match'] # This is 0 or 100
        f1_score = scores['f1']         # This is 0-100

        example_details["questions"].append({
            "question_text": question,
            "true_answer": true_answer_texts_for_squad[0] if true_answer_texts_for_squad else "",
            "model_answer": model_answer,
            "em": em_score,
            "f1": f1_score
        })

        if em_score == 100.0:
            current_example_retrieved_em += 1
        if f1_score >= F1_THRESHOLD_FOR_RECALL:
            current_example_retrieved_f1_count += 1
        current_example_f1_sum += f1_score

    results_log.append(example_details)

    total_questions_asked_overall += N_TARGET_NEEDLES_TO_RETRIEVE
    total_needles_retrieved_em_overall += current_example_retrieved_em
    total_needles_retrieved_f1_overall += current_example_retrieved_f1_count
    sum_of_all_f1_scores += (current_example_f1_sum / N_TARGET_NEEDLES_TO_RETRIEVE) # Avg F1 for this example's targeted needles

# --- Calculate and Print Overall Results ---
if total_questions_asked_overall > 0:
    avg_recall_em = (total_needles_retrieved_em_overall / total_questions_asked_overall) * 100
    avg_recall_f1_threshold = (total_needles_retrieved_f1_overall / total_questions_asked_overall) * 100
    # Average F1 score across all *individual questions* that were asked
    # total_f1_sum_across_all_questions = sum(q['f1'] for ex_log in results_log for q in ex_log['questions'])
    # overall_avg_f1 = total_f1_sum_across_all_questions / total_questions_asked_overall
    # The sum_of_all_f1_scores is sum of per-example average F1s. So divide by num_to_run (or actual processed examples)
    num_examples_processed = len(results_log)
    overall_avg_f1 = (sum_of_all_f1_scores / num_examples_processed) if num_examples_processed > 0 else 0.0


    print("\n--- Overall Evaluation Results ---")
    print(f"Dataset: {DATASET_PATH} (Config: {DATASET_CONFIG_NAME})")
    print(f"Model: {MODEL_NAME}")
    print(f"Number of Examples Processed: {num_examples_processed}")
    print(f"Needles Targeted per Example: {N_TARGET_NEEDLES_TO_RETRIEVE}")
    print(f"Total Questions (Needles) Evaluated: {total_questions_asked_overall}")
    print(f"Overall Average F1 Score: {overall_avg_f1:.2f}")
    print(f"Recall (Exact Match): {avg_recall_em:.2f}%")
    print(f"Recall (F1 >= {F1_THRESHOLD_FOR_RECALL}): {avg_recall_f1_threshold:.2f}%")
else:
    print("No examples were processed. Check dataset or NUM_EXAMPLES_TO_TEST.")

# --- Save detailed results ---
output_filename = f"multi_needle_results_{MODEL_NAME.split('/')[-1]}_{DATASET_PATH.split('/')[-1]}_{DATASET_CONFIG_NAME}.json"
report = {
    "model_name": MODEL_NAME,
    "dataset_path": DATASET_PATH,
    "dataset_config_name": DATASET_CONFIG_NAME,
    "num_examples_tested": num_to_run, # Intended, not necessarily processed if skips
    "num_examples_processed": len(results_log),
    "n_target_needles_to_retrieve": N_TARGET_NEEDLES_TO_RETRIEVE,
    "f1_threshold_for_recall": F1_THRESHOLD_FOR_RECALL,
    "overall_metrics": {
        "avg_f1_score": overall_avg_f1 if total_questions_asked_overall > 0 else 0.0,
        "recall_em_percent": avg_recall_em if total_questions_asked_overall > 0 else 0.0,
        "recall_f1_threshold_percent": avg_recall_f1_threshold if total_questions_asked_overall > 0 else 0.0,
    },
    "individual_example_logs": results_log
}
with open(output_filename, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nDetailed results saved to {output_filename}")

# Print a sample from the logs
if results_log:
    print("\n--- Sample Logged Result (First Example) ---")
    print(json.dumps(results_log[0], indent=2))