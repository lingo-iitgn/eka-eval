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

![My Image](architecture.png)


---

# **Key Features**

## ✔️ Benchmark Coverage

* **25+ Global benchmarks**: MMLU, GSM8K, ARC-Challenge, HumanEval, HellaSwag, etc.
* **20+ Indic benchmarks**: MMLU-IN, BoolQ-IN, ARC-IN, MILU, Flores-IN, etc.
* **Long-context**: ZeroSCROLLS, InfiniteBench, Multi-Needle
* **Code generation** with pass@k
* **Math & logical reasoning**
* **Multilingual evaluation** across 11 languages

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


| Category                 | Count | Benchmarks                                                                                                                                                               | Metrics                        |
|--------------------------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| 🇮🇳 Indic Ecosystem       | 18    | Knowledge: MMLU-IN, MILU, TriviaQA-IN<br>Reasoning: ARC-Challenge-IN, ARC-Easy-IN, HellaSwag-IN, IndicCOPA<br>NLU: IndicXNLI, IndicSentiment, IndicXParaphrase<br>QA & Reading: BoolQ-IN, XQuAD-IN, XorQA-IN<br>Math: GSM8K-IN<br>Generation: Flores-IN, CrossSum-IN, IndicNLG, Indic-Toxic | Accuracy, F1, BLEU, chrF++, ROUGE-L |
| 🧠 Reasoning             | 10    | ARC-Challenge, ARC-Easy, HellaSwag, PIQA, SIQA, WinoGrande, OpenBookQA, CommonSenseQA, BBH, AGI-Eval                                                                     | Accuracy, Normalized Accuracy  |
| 📚 Knowledge              | 4     | MMLU, MMLU-Pro, TriviaQA, NaturalQuestions                                                                                                                               | Accuracy, Exact Match          |
| 🧮 Math & Code            | 7     | Math: GSM8K, MATH, GPQA<br>Code: HumanEval, MBPP, HumanEval+, MBPP+                                                                                                      | Accuracy, pass@1               |
| 📖 Reading                | 3     | SQuAD, QuAC, BoolQ                                                                                                                                                       | F1, Exact Match                |
| 🛠️ Tool & Context         | 6     | Long Context: InfiniteBench, ZeroSCROLLS, NeedleInAHaystack<br>Tool Use: API-Bank, API-Bench, ToolBench                                                                  | Retrieval Acc, Success Rate    |


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

## 2. Create Environment (Conda)

We use Conda to manage Python 3.10 environments to ensure compatibility across macOS, Linux, and Windows.

### **Step 1: Create and Activate**

Run this on **any system**:

```bash
# Create environment with Python 3.10
conda create -n eka-env python=3.10 pip -y

# Activate the environment
conda activate eka-env
```

### **Step 2: Install Dependencies**

Choose the option that matches your hardware:

#### **Option A — macOS (M1/M2/M3) or CPU-only**

Uses the clean file without NVIDIA/CUDA packages.

```bash
pip install -r requirements-cpu.txt
```

#### **Option B — GPU Server (Linux + NVIDIA)**

Uses the file with `bitsandbytes`, CUDA extensions, and quantization support.

```bash
pip install -r requirements-gpu.txt
```

### **Step 3: Install Project**

Install the project in editable (`-e`) mode:

```bash
pip install -e .
```


## 3. (Optional) HuggingFace Login

Some models require authentication (e.g., Llama 3, gemma).
Create a token at **Hugging Face → Settings → Access Tokens** (usually “Read” or “Write”).
Log in:

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
<video src="https://github.com/user-attachments/assets/44192d82-0cf8-499c-9ae9-750e0a00e415" controls width="600"></video>

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

1. CODE GENERATION
2. Tool use
3. MATH
4. READING COMPREHENSION
5. COMMONSENSE REASONING
6. WORLD KNOWLEDGE
7. LONG CONTEXT
8. General
9. INDIC BENCHMARKS
10. ALL Task Groups

Select task group #(s) (e.g., '1', '1 3', 'ALL'): 2 12
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

# 🎯 Advanced Configuration & Usage

Eka-Eval provides extensive customization for Indic languages, few-shot settings, prompt templates, and even fully custom benchmarks.

---

## 1. Configuring Indic Languages & Splits

Benchmarks like **MILU** or **ARC-Challenge-Indic** can be restricted to specific languages by modifying:

`eka_eval/config/benchmark_config.py`

### **Example: Run MILU only for Bengali**

```python
"MILU": {
    "description": "Accuracy on the Massive Indic Language Understanding benchmark",
    "evaluation_function": "indic.milu_in.evaluate_milu_in",
    "task_args": {
        "dataset_name": "ai4bharat/MILU",
        "target_languages": ["Bengali"],   # restrict to one language
        "dataset_split": "test",
        "max_new_tokens": 5,
        "save_detailed": False,
        "prompt_file_benchmark_key": "milu_in"
    }
}
```

---

## 2. Customizing Few-Shot & Zero-Shot Settings

Control the number of demonstration examples and batch sizes directly via `task_args`.

### **Example: Zero-Shot ARC-Challenge-Indic**

```python
"ARC-Challenge-Indic": {
    "description": "Zero-shot ARC-Challenge-Indic evaluation across 11 languages",
    "evaluation_function": "indic.arc_c_in.evaluate_arc_c_in",
    "task_args": {
        "dataset_name": "sarvamai/arc-challenge-indic",
        "target_languages": ["bn"],     # only Bengali
        "dataset_split": "validation",
        "num_few_shot": 0,              # Zero-shot; set >0 for few-shot
        "max_new_tokens": 10,
        "generation_batch_size": 8,

        # switch prompt templates
        "prompt_template_name_zeroshot": "arc_c_in_0shot",
        "prompt_template_name_fewshot": "arc_c_in_5shot",

        "prompt_file_benchmark_key": "arc_c_in",
        "prompt_file_category": "indic"
    }
}
```

---

## 3. Modifying Prompt Templates & Few-Shot Examples

Prompts are stored under the **`prompts/`** directory and can be fully customized.

### **Example File:** `prompts/indic/boolq_in.json`

```json
{
  "boolq_in_0shot": {
    "template": "Passage: {passage}\nQuestion: {question}\nAnswer (Yes/No):",
    "description": "Standard zero-shot prompt"
  },
  "default_few_shot_examples_boolq_in": [
    {
      "passage": "भारत दक्षिण एशिया में स्थित एक देश है। यह दुनिया का सातवां सबसे बड़ा देश है।",
      "question": "क्या भारत एशिया में है?",
      "answer": "हाँ"
    },
    {
      "passage": "सूर्य पृथ्वी के चारों ओर घूमता है। यह हमारे सौर मंडल का केंद्र है।",
      "question": "क्या सूर्य पृथ्वी के चारों ओर घूमता है?",
      "answer": "नहीं"
    }
  ]
}
```

You can edit:

* the **template**
* instructions
* placeholders `{question}`, `{context}`, `{choices}`
* the few-shot examples list

---

## 4. Adding a Completely Custom Benchmark

You can add entirely new datasets and evaluators.

---

### **Step 1: Create Evaluator Logic**

File: `eka_eval/benchmarks/tasks/custom/my_task.py`

```python
def evaluate_my_task(pipe, tokenizer, model_name_for_logging, device, **kwargs):
    score = 85.5  # your logic here
    return {"MyTask": score}
```

### **Step 2: Add Prompt Templates**

File: `prompts/custom/my_task.json`

```json
{
  "my_task_0shot": {
    "template": "Question: {question}\nAnswer:"
  }
}
```

### **Step 3: Register in Configuration**

Add to `benchmark_config.py`:

```python
"MyTask": {
  "evaluation_function": "custom.my_task.evaluate_my_task",
  "task_args": {
      "dataset_name": "my_org/custom_dataset",
      "prompt_file_category": "custom"
  }
}
```

---

# ⚙️ Hardware Optimization

Eka-Eval supports quantization and multi-GPU evaluation out of the box.

---

## **4-bit / 8-bit Quantization**

Useful for running 33B–70B models on consumer GPUs.

```bash
python scripts/run_benchmarks.py \
    --model_name "meta-llama/Llama-2-70b-hf" \
    --quantization "4bit"
```

---

## **Multi-GPU Parallel Evaluation**

Distribute workload across multiple GPUs.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
python scripts/run_benchmarks.py --num_gpus 3
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

`results_output/detailed_results/*.json`

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
