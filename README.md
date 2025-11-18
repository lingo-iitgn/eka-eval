<p align="center"> <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" /> <img src="https://img.shields.io/badge/benchmarks-English%20%2B%20Indic-orange?style=for-the-badge" /> </p> <p align="center"> <img src="https://img.shields.io/github/stars/lingo-iitgn/eka-eval?style=flat-square" /> <img src="https://img.shields.io/github/forks/lingo-iitgn/eka-eval?style=flat-square" /> <img src="https://img.shields.io/github/contributors/lingo-iitgn/eka-eval?style=flat-square" /> <img src="https://img.shields.io/github/last-commit/lingo-iitgn/eka-eval?style=flat-square" /> </p>


# **Eka-Eval**

<div align="center">
  <a href="https://eka.soket.ai/">
    <img width="118" src="https://github.com/user-attachments/assets/2822b114-39bb-4c19-8808-accd8b415b3a" alt="eka-eval logo"/>
  </a>
  <h3><strong>Comprehensive Evaluation Framework for Large Language Models with an India-First Lens</strong></h3>
</div>

---

# **📌 Table of Contents**

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Supported Benchmarks](#supported-benchmarks)

   * Global Benchmarks
   * Indic Benchmarks
   * Supported Languages
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Advanced Usage](#advanced-usage)

   * Custom Benchmarks
   * Quantization
   * Multi-GPU
   * Debug Mode
8. [Results & Reporting](#results--reporting)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [References](#references)
12. [Citation](#citation)
13. [License](#license)

---

# **Overview**

**Eka-Eval** is the official evaluation pipeline for the **EKA project**, designed to provide **reliable, reproducible, and India-centric evaluation** of LLMs.

It combines:

* Global benchmarks
* Indian-language benchmarks
* Long-context evaluation
* Code, math, reasoning, QA

Eka-Eval provides a **uniform interface**, **structured results**, and **production-ready performance features**.

---

# **Key Features**

## ✔️ Benchmark Coverage

* **17+ Global benchmarks**: MMLU, GSM8K, ARC-Challenge, HumanEval, HellaSwag, etc.
* **12+ Indic benchmarks**: MMLU-IN, BoolQ-IN, ARC-IN, MILU, Flores-IN, etc.
* **Long-context**: ZeroSCROLLS, InfiniteBench, Multi-Needle
* **Code generation** with pass@k
* **Math & logical reasoning**
* **Multilingual evaluation** across 11 Indian languages

## ✔️ Multilingual Support

* 11 Indic languages + English
* Smart Hindi-English transliteration
* Per-language scores
* Unified prompt templates

## ✔️ Performance & Scalability

* Multi-GPU distributed evaluation
* 4-bit / 8-bit quantization
* Efficient batching
* Automatic CUDA memory cleanup

## ✔️ Developer Friendly

* Modular task registry
* Easy custom-benchmark integration
* JSON-based configs
* Clear logging + progress tracking

## ✔️ Reporting & Analysis

* CSV summary
* JSONL detailed results
* Per-language metrics
* Error analysis
* Full reproducibility with configuration dump

---

# **Supported Benchmarks**

## 🌍 Global Benchmarks

| Category     | Benchmarks                          | Metrics          |
| ------------ | ----------------------------------- | ---------------- |
| Knowledge    | MMLU, MMLU-Pro, TriviaQA            | Accuracy         |
| Math         | GSM8K, MATH, GPQA                   | Accuracy         |
| Code         | HumanEval, MBPP                     | pass@1, pass@k   |
| Reasoning    | AGIEval, BBH, WinoGrande, HellaSwag | Accuracy         |
| Reading      | SQuAD, QuAC, BoolQ                  | EM, F1, Accuracy |
| Long Context | ZeroSCROLLS, InfiniteBench          | Multiple         |

---

## 🇮🇳 Indic Benchmarks

| Benchmark        | Description                      | Metric     |
| ---------------- | -------------------------------- | ---------- |
| MMLU-IN          | Indian-subject knowledge         | Accuracy   |
| ARC-Challenge-IN | Indian science reasoning         | Accuracy   |
| BoolQ-IN         | Indic yes/no QA                  | Accuracy   |
| MILU             | Multilingual Indic understanding | Accuracy   |
| Flores-IN        | Translation                      | BLEU, ChrF |
| XQuAD-IN         | Reading Comprehension            | F1, EM     |

---

## 🗣️ Supported Languages

Hindi (hi), Bengali (bn), Gujarati (gu), Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), Telugu (te), English (en)

---

# **Installation**

## 1. Clone Repo

```bash
git clone https://github.com/lingo-iitgn/eka-eval.git
cd eka-eval
```

## 2. Create Environment (macOS/Linux)

```bash
python3 -m venv eka-env
source eka-env/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Create Environment (Windows)

```cmd
python -m venv eka-env
eka-env\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
```

## 3. (Optional) HuggingFace Login

```bash
huggingface-cli login
```

---

# **Quick Start**

### Run the Interactive Evaluator

```bash
python3 scripts/run_benchmarks.py
```

### Video Demonstration 
---
https://github.com/user-attachments/assets/44192d82-0cf8-499c-9ae9-750e0a00e415
---

---

## Running Benchmarks (Interactive Wizard Mode)

`eka-eval` includes a fully interactive CLI wizard for evaluating models across English and Indic benchmark suites.

To start the wizard, simply run:

```bash
python scripts/run_benchmarks.py
```

This will launch a guided, step-by-step interface.

---

## 🧩 1. **Select Model Source**

```plaintext
--- Model Selection ---

1. Hugging Face / Local Model
2. API Model (OpenAI, Anthropic, etc.)

Enter choice: 1
Enter model name: google/gemma-2-2b
```

---

## 📚 2. **Select Task Groups**

```plaintext
--- Available Benchmark Task Groups ---

1. CODE GENERATION           9. INDIC BENCHMARKS
2. MATH AND REASONING       10. ALL Task Groups
...

Select task group #(s): 2 9
```

You can select multiple groups by entering space-separated numbers (e.g., `2 9`).

---

## 🎯 3. **Select Specific Benchmarks**

```plaintext
--- Select benchmarks for INDIC BENCHMARKS ---

1. MMLU-IN                 4. ARC-Challenge-IN
2. BoolQ-IN                5. ALL
3. Flores-IN               6. SKIP

Select benchmark #(s): 4 5
```

Again, multiple selections are supported.

---

## 📊 4. **View Results & Visualize**

```plaintext
... Evaluation Complete ...

| Model             | Task             | Benchmark | Score |
|-------------------|------------------|-----------|-------|
| google/gemma-2-2b | INDIC BENCHMARKS | ARC-IN    | 33.5% |
```

When prompted:

```plaintext
Do you want to create visualizations for the results? (yes/no): yes
```

The system will generate plots:

```
✅ Visualizations created successfully! 
Saved to: results_output/visualizations
```

---


# **Project Structure**

```
eka-eval/
├─ eka_eval/
│  ├─ benchmarks/
│  │  ├─ tasks/
│  │  │  ├─ code/
│  │  │  ├─ math/
│  │  │  ├─ indic/
│  │  │  ├─ reasoning/
│  │  │  ├─ long_context/
│  │  │  └─ general/
│  │  └─ benchmark_registry.py
│  ├─ core/
│  ├─ utils/
│  └─ config/
├─ prompts/
├─ scripts/
│  ├─ run_benchmarks.py
│  └─ evaluation_worker.py
└─ results_output/
```

---

# **Advanced Usage**

## 🔧 Add a Custom Benchmark

### 1. Write evaluator

```python
def evaluate_my_task(pipe, tokenizer, model_name_for_logging, device, **kwargs):
    return {"MyTask": score}
```

### 2. Add prompt

`prompts/custom/my_task.json`

```json
{
  "my_task_0shot": {
    "template": "Question: {question}\nAnswer:"
  }
}
```

### 3. Register benchmark

```python
"MyTask": {
  "evaluation_function": "my_project.my_benchmark.evaluate_my_task",
  "task_args": {...}
}
```

---

## ⚙️ Quantization

```bash
python3 scripts/run_benchmarks.py \
    --model llama-2-70b \
    --quantization 4bit
```

## 🧠 Multi-GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 scripts/run_benchmarks.py --num_gpus 3
```

## 🐞 Debug Mode

```bash
python3 scripts/run_benchmarks.py --log_level DEBUG
```

---

# **Results & Reporting**

### CSV Summary

`results_output/calculated.csv`

```
model,task,benchmark,score
gemma-2b,MATH,GSM8K,42.3
```

### JSONL Detailed Results

`results_output/detailed_results/*.jsonl`

```json
{
  "id": 123,
  "question": "...",
  "predicted": "4",
  "correct": true
}
```

### Per-Language Metrics

```json
{
  "BoolQ-IN_hi": 65.2,
  "BoolQ-IN_bn": 70.1
}
```

---

# **Troubleshooting**

| Issue                   | Fix                                 |
| ----------------------- | ----------------------------------- |
| CUDA OOM                | Reduce batch size, use quantization |
| HF 404                  | Wrong model name or missing token   |
| Missing prompt template | Check prompts folder                |
| Code evaluator error    | Set `export HF_ALLOW_CODE_EVAL=1`   |

---

# **Contributing**

We welcome contributions!

* Report issues
* Add new benchmarks
* Improve documentation
* Submit PRs

---

# **References**

* MMLU
* GSM8K
* HumanEval
* BBH
* AGIEval
* AI4Bharat Indic datasets

---

# **Citation**

```bibtex
@misc{sinha2025ekaevalcomprehensiveevaluation,
      title={Eka-Eval : A Comprehensive Evaluation Framework for Large Language Models in Indian Languages}, 
      author={Samridhi Raj Sinha and Rajvee Sheth and Abhishek Upperwal and Mayank Singh},
      year={2025},
      eprint={2507.01853},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

# **License**

MIT License – see `LICENSE`.

---
