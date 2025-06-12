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

**eka-eval** is the official evaluation pipeline for the EKA project ([eka.soket.ai](https://eka.soket.ai)), designed to benchmark large language models (LLMs) for India and the world. It supports a wide range of global and Indic benchmarks, ensuring rigorous, fair, and transparent evaluation for both English and Indian languages.

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

## **Vision & Principles**

- **Open, Ethical, and Inclusive:** Built to support India’s linguistic and cultural diversity, with open-source code and datasets.
- **Global Standards, Local Relevance:** Benchmarks include both international and India-centric tasks.
- **Transparent & Reproducible:** All evaluation scripts, metrics, and results are open and reproducible.
- **Climate-Conscious (Aspirational):** We aim to optimize for efficient, large-scale evaluation with minimal energy use where possible.

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
│   ├── run_evaluation_suite.py   # Main orchestrator
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
python3 scripts/run_evaluation_suite.py
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

## 🧪 Supported Benchmarks and Metrics

Refer to your `benchmark_config.py` or project documentation for up-to-date supported benchmarks. Common metrics include:

* **Accuracy**
* **F1**
* **Exact Match**
* **pass\@k (for code generation)**

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

## **Script Usage**

The main script should take the following options:

```bash
python eka_eval.py \
    --model_name_or_path  \
    --benchmark  \
    --language  \
    --batch_size  \
    --shots  \
    --output_file 
```

**Arguments:**
- `--model_name_or_path`: Hugging Face model identifier or path.
- `--benchmark`: Name of the benchmark to run (e.g., mmlu, bbh, gsm8k, piqa, mmlu-in, milu).
- `--language`: Language code (for Indic benchmarks).
- `--batch_size`: Batch size for inference.
- `--shots`: Number of few-shot examples (0 for zero-shot, 3/5/7 as per protocol).
- `--output_file`: Where to save the evaluation results.

---

## **Example Usage**

---

## **Reporting and Results**

- **All results are reported as raw scores** (no aggregation), per benchmark and per language.
- Results include per-task accuracy, F1, EM, pass@1, BLEU, etc., as appropriate.
- Output JSON includes detailed per-example outputs for transparency and error analysis.

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

**eka-eval: The open, ethical, and comprehensive LLM benchmarking suite for India and the world.**

---

