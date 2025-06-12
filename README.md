<div align="center">
  <!-- Eka Logo -->
  <img width="118" alt="eka-eval logo" src="https://github.com/user-attachments/assets/2822b114-39bb-4c19-8808-accd8b415b3a" style="margin-bottom: 10px;" />

  <h1>eka-eval.</h1>
  <h2>The Unified LLM Evaluation Framework for India and the World.</h2>

  <!-- Badges Row -->
  <p>
    <img src="https://img.shields.io/badge/release-v1.0-blue.svg" alt="Release v1.0" />
    <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License Apache 2.0" />
    <a href="https://colab.research.google.com/github/your_repo/eka-eval/blob/main/notebooks/quick_start.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
    </a>
    <a href="https://discord.gg/yourcommunitylink">
      <img src="https://img.shields.io/discord/308323056592486420?label=Join%20Community&logo=discord&colorB=7289DA" alt="Discord Community" />
    </a>
  </p>

  <!-- Navigation Links -->
  <p>
    <a href="#key-features">Key Features</a> •
    <a href="#supported-benchmarks-and-metrics">Supported Benchmarks</a> •
    <a href="#script-usage">Getting Started</a> •
    <a href="#reporting-and-results">Reporting</a> •
    <a href="#project-ethos">Project Ethos</a>
  </p>
</div>



---

# eka-eval: Benchmarking Pipeline Documentation

---

## **Overview**

**eka-eval** is the official evaluation pipeline for the EKA project ([eka.soket.ai](https://eka.soket.ai)), designed to benchmark large language models (LLMs). It supports a wide range of global and Indic benchmarks, ensuring rigorous, fair, and transparent evaluation for both English and Indian languages.

---
## **Key Features**

*   **Comprehensive Benchmark Suite:** Supports a wide range of global (MMLU, BBH, GSM8K, HumanEval, etc.) and India-centric (MMLU-IN, IndicGenBench, Flores-IN, etc.) benchmarks.
*   **Multi-Lingual Support:** Designed for evaluating models in English and multiple Indian languages.
*   **Modular Design:** Easily extendable with new benchmarks, tasks, models, and metrics.
*   **Hugging Face Integration:** Seamlessly works with models and datasets from the Hugging Face Hub.
*   **Quantization Support:** Built-in support for 4-bit/8-bit quantization (via `bitsandbytes`) for efficient evaluation of large models.
*   **Batching & Parallelism:** Supports batched inference and multi-GPU evaluation for speed and scalability.
*   **Few-Shot & Zero-Shot:** Configurable few-shot prompting as per benchmark protocols.
*   **Detailed Reporting:** Generates comprehensive results in CSV format, including per-instance scores and detailed logs for error analysis (JSONL for some tasks).
*   **Reproducibility:** Aims for reproducible evaluation runs through clear configuration and versioning.
*   **Customizable:** Allows users to add custom benchmarks and evaluation logic.

---
## **Installation**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-org/eka-eval.git # Replace with your actual repo URL
    cd eka-eval
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure your `requirements.txt` includes:
    ```
    torch
    transformers
    datasets
    evaluate
    pandas
    tqdm
    accelerate
    bitsandbytes # For quantization
    # Add other specific dependencies your tasks might need (e.g., sentencepiece, protobuf for certain tokenizers/models)
    ```
    For CUDA support with `bitsandbytes` and PyTorch, ensure your CUDA toolkit is compatible with the versions you install.

4.  **Hugging Face Authentication (Recommended for accessing private/gated models or some datasets):**
    *   **CLI Login (Recommended):**
        ```bash
        huggingface-cli login
        ```
        Paste your Hugging Face access token (with read permissions, generate from [HF Settings](https://huggingface.co/settings/tokens)).
    *   **Environment Variable:**
        ```bash
        export HF_TOKEN="your_hf_read_token_here"
        ```

---

## **🚀 Quick Start**

To quickly evaluate a model (e.g., `google/gemma-2b`) on a small subset of a benchmark (e.g., MMLU):
```bash
# Ensure you are in the root directory of the eka-eval project
# (the directory containing the 'scripts' and 'eka_eval' folders)

python3 scripts/run_benchmarks.py
```

--- 

## 🔧 Directory Structure

```
eka-eval/
├── eka_eval/                     # Main library package
│   ├── __init__.py
│   ├── benchmarks/               # Benchmark-specific logic
│   │   ├── __init__.py
│   │   ├── benchmark_registry.py
│   │   └── tasks/
│   │       ├── __init__.py
│   │       ├── code/             # Code generation tasks (e.g., HumanEval, MBPP)
│   │       ├── indic/            # Indic language tasks
│   │       ├── reasoning/        # Reasoning tasks
│   │       └── general/          # General tasks (e.g., MMLU, BBH)
│   ├── config/
│   │   ├── __init__.py
│   │   └── benchmark_config.py   # All supported benchmarks and their settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Loads models/tokenizers
│   │   └── evaluator.py (optional)
│   ├── results/
│   │   ├── __init__.py
│   │   └── result_manager.py     # For managing result aggregation/output
│   └── utils/
│       ├── __init__.py
│       ├── constants.py
│       ├── file_utils.py
│       ├── gpu_utils.py
│       └── logging_setup.py
├── scripts/                      # Executables to run evaluations
│   ├── __init__.py
│   ├── run_benchmarks.py   # Main orchestrator
│   └── evaluation_worker.py      # Worker process logic
├── results_output/               # Default result directory
│   └── calculated.csv            # Aggregated results
├── checkpoints/                  # Task-specific checkpoints
├── tests/                        # Tests (TODO)
├── .github/                      # GitHub configs
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 How to Run Evaluations

Use:

```bash
python3 scripts/run_benchmarks.py
```

You’ll be guided through the following interactive steps:

### 🔹 Model Selection

* **Prompt**: `Enter model source ('1' for Hugging Face, '2' for Local Path):`
* Examples:

  * Hugging Face: `google/gemma-2b`
  * Local path: `/path/to/your/model`

### 🔹 Custom Benchmarks

* Prompt: `Do you want to add any custom/internal benchmarks for this session? (yes/no)`
* Select `no` for quick start or `yes` to register a custom benchmark (see below).

### 🔹 Task Group Selection

* Choose from available groups (e.g., CODE GENERATION, MMLU, INDIC BENCHMARKS).
* Enter numbers (e.g., `1 3`) or `ALL`.

### 🔹 Benchmark Selection

* For each selected group, select benchmarks or use `ALL` to include all in that group.
* Single-benchmark groups (e.g., MMLU) are auto-selected.

### 🔹 Code Task Parameters (if applicable)

* Prompt: `Enter comma-separated k values for pass@k (e.g., 1,5,10) [Default: 1]`
* Prompt: `Enter generation batch size [Default: 1]`

---

## ➕ Adding Custom Benchmarks

### ✨ Interactive Mode

If you answered `yes` when asked about custom benchmarks, you'll be prompted:

1. **Task Group Name**: e.g., `MY_INTERNAL_EVALS`
2. **Benchmark Display Name**: e.g., `LogicTest`
3. **Python Module Path**: e.g., `my_custom_evals.logic_test`
4. **Function Name**: e.g., `evaluate_logic_test`

The module must be importable (i.e., in your `PYTHONPATH` or part of the project).

---

### ✍️ Manual Addition (Optional)

Edit `eka_eval/config/benchmark_config.py`:

```python
BENCHMARK_CONFIG = {
    "MY_CUSTOM_TASKS": {
        "MyLogicTest": {
            "description": "Evaluates logic skills.",
            "evaluation_function": "my_project.custom_evals.logic_test.evaluate_logic_test",
            "is_custom": True,
            "task_args": {
                "dataset_path": "/path/to/data.jsonl",
                "num_few_shot": 3,
                "max_new_tokens": 128
            }
        }
    }
}
```

---

## 📊 Results & Reporting

### ✅ Aggregated CSV Output

* Stored in `results_output/calculated.csv`
* Columns:

  * `Model`, `Size (B)`, `Task`, `Benchmark`, `Score`, `Timestamp`, `Status`

### 📄 Detailed Logs

* Some benchmarks (e.g., HumanEval) write JSONL logs in:

  ```
  results_output/<benchmark_name>_detailed/
  ```

### 🖥️ Console Output

* Final results are printed as a **Markdown table**.
* Worker logs are prefixed (e.g., `[Worker 0 (GPU 1)]`).

---

## ⚠️ Troubleshooting

| Issue                                             | Fix                                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'eka_eval'` | Ensure you're running from the root directory.                                    |
| Worker can't import your custom task              | Ensure correct `evaluation_function` path and `__init__.py` files exist.          |
| Hugging Face 404                                  | Verify the model/dataset name and check authentication (`huggingface-cli login`). |
| CUDA OOM errors                                   | Reduce `generation_batch_size`, use quantized models, or adjust config.           |
| `code_eval` metric errors                         | Set `HF_ALLOW_CODE_EVAL=1` in your environment.                                   |


---

## **Supported Benchmarks and Metrics**

### **English Benchmarks**

| Category                | Benchmark(s)                                 | Metric(s)                     |
|-------------------------|----------------------------------------------|-------------------------------|
| General                 | MMLU, MMLU-Pro, IFEval, BBH (3-shot), AGIEval (3–5 shot) | Accuracy                      |
| Math & Reasoning        | GSM8K, MATH, GPQA, ARC-Challenge             | Accuracy                      |
| Code                    | HumanEval, MBPP, HumanEval+, MBPP EvalPlus, MultiPL-E | Pass@1 (accuracy)             |
| Multilinguality         | MGSM, Multilingual MMLU (internal)           | Accuracy                      |
| Tool-use                | Nexus, API-Bank, API-Bench, BFCL             | Success Rate, Task-specific   |
| Long Context            | ZeroSCROLLS, Needle-in-a-Haystack, InfiniteBench | Task-specific (Accuracy, F1, EM, Recall) |
| Commonsense Reasoning   | PIQA, SIQA, HellaSwag, ARC (Easy/Chall), WinoGrande, CommonsenseQA (7-shot), OpenBookQA | Accuracy                      |
| World Knowledge         | TriviaQA (5-shot), NaturalQuestions (5-shot) | Accuracy                      |
| Reading Comprehension   | SQuAD (F1, EM), QuAC (F1, EM), BoolQ (Accuracy) | F1, Exact Match, Accuracy     |

### **Indic Language Benchmarks**

| Benchmark            | Metric(s)               |
|----------------------|-------------------------|
| MMLU-IN              | Accuracy                |
| TriviaQA-IN          | Accuracy                |
| MILU                 | Accuracy                |
| GSM-8K-IN            | Accuracy                |
| IndicGenBench        | Task Specific           |
| Flores-IN            | BLEU, ChrF             |
| XQuAD-IN             | F1, Exact Match         |
| XorQA-IN             | F1, Exact Match         |

---
 

## 📊 **Reporting and Results**

The `eka-eval` framework generates detailed and transparent results to support both high-level benchmarking and fine-grained error analysis.

---

### ✅ **Aggregated Results (CSV)**

A primary CSV file named `calculated.csv` is generated in the directory specified via the `--results_dir` flag (default: `results_output/`). Each row represents a single benchmark evaluation for a model.

**CSV Columns:**

| Column      | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `Model`     | Name or path of the evaluated model (e.g., `google/gemma-2b`)       |
| `Size (B)`  | Approximate model parameter size (in billions), if determinable     |
| `Task`      | Task group (e.g., `CODE GENERATION`, `MMLU`, `INDIC BENCHMARKS`)    |
| `Benchmark` | Specific benchmark name (e.g., `HumanEval`, `GSM8K`, `BOOLQ-IN`)    |
| `Score`     | Primary metric (e.g., `accuracy`, `pass@1`, `F1`)                   |
| `Timestamp` | Date and time of the evaluation run                                 |
| `Status`    | (Optional) Evaluation status (e.g., `Completed`, `EvaluationError`) |

---

### 📄 **Detailed Task Logs (JSONL)**

For selected tasks that involve generation or require detailed analysis (e.g., `HumanEval`, `MBPP`, `IndicGenBench`), a JSONL file is created per benchmark in:

```
results_output/<benchmark_name>_detailed/
```

Each JSONL line typically includes:

* Input prompt
* Model output (raw prediction)
* Ground truth / reference
* Evaluation metrics (e.g., exact match, pass\@k)
* Metadata (e.g., instance ID, language, model config)

This format supports post-hoc error analysis and reproducibility.

---

### 🖥️ **Console Output**

At the end of an evaluation run, results are printed to the terminal in a **Markdown-style table** showing model performance across all selected benchmarks.

Example:

```
| Model           | Task Group       | Benchmark  | Score   |
|----------------|------------------|------------|---------|
| gemma-2b        | MMLU             | MMLU       | 44.7%   |
| gemma-2b        | CODE GENERATION  | HumanEval  | 18.2%   |
```

---

### 🧵 **Worker Logs**

Each GPU worker process outputs logs in real time, with each log message prefixed by worker ID and GPU ID for easy debugging:

```
[Worker 0 (GPU 0)] Loading model: google/gemma-2b
[Worker 0 (GPU 0)] Running benchmark: MMLU
```

These logs include:

* Model/tokenizer loading status
* Dataset loading and preprocessing
* Evaluation progress per benchmark
* Any encountered warnings or errors

---

### 🗃️ **Raw Scores**

All reported results are **raw and per-task**, with no manual aggregation. Metrics include:

* **Accuracy**: For multiple-choice and classification tasks
* **F1 / Exact Match (EM)**: For reading comprehension and QA tasks (e.g., XQuAD-IN, BoolQ)
* **Pass\@1 / Pass\@k**: For code generation tasks (e.g., HumanEval, MBPP)
* **BLEU / ChrF**: For translation tasks (e.g., Flores-IN)
* **Task-specific**: Metrics defined by custom benchmark logic (e.g., NeedleRecall for Multi-Needle)

---

---
## 🛠️ Contribute

### 🧪 Testing and Feedback

We welcome contributions from the community to help test and improve this library.  
If you encounter any issues or have suggestions for enhancements, please feel free to [open an issue](https://github.com/your-org/eka-eval/issues) on the GitHub repository.

Your feedback helps make **eka-eval** better for everyone!

---

## **References**

- [EKA Project](https://eka.soket.ai/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [BBH Official](https://github.com/suzgunmirac/BIG-Bench-Hard)
- [MMLU Official](https://github.com/hendrycks/test)
- [AGIEval Official](https://github.com/THUDM/AGIEval)
- [MILU Benchmark](https://github.com/AI4Bharat/MILU)
- [IndicGenBench](https://github.com/AI4Bharat/IndicGenBench)

---

## **Project Ethos**

- **Open-source**: All code and model weights are freely available.
- **Ethical**: Prioritizes fairness, transparency, and privacy.
- **India-first**: Benchmarks and models for all Indian languages and use-cases.
- **Community-driven**: Contributions and feedback are welcome.

---

**eka-eval: The open, ethical, and comprehensive LLM benchmarking suite.**

---

