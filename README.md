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

## **Vision & Principles**

- **Open, Ethical, and Inclusive:** Built to support India’s linguistic and cultural diversity, with open-source code and datasets.
- **Global Standards, Local Relevance:** Benchmarks include both international (MMLU, BBH, AGIEval, etc.) and India-centric (MMLU-IN, MILU, etc.) tasks.
- **Transparent & Reproducible:** All evaluation scripts, metrics, and results are open and reproducible.
- **Climate-Conscious:** Optimized for efficient, large-scale evaluation with minimal energy use.

---

## **Pipeline Theory**

The **eka-eval** pipeline is modular and extensible:

1. **Task Loader:** Loads benchmarks from Hugging Face Datasets or local files.
2. **Prompt Generator:** Formats prompts for zero-shot or few-shot (as per benchmark protocol).
3. **Model Inference:** Supports any Hugging Face-compatible LLM, with batching and quantization.
4. **Postprocessing:** Extracts and normalizes model outputs.
5. **Metric Computation:** Computes benchmark-specific metrics (Accuracy, F1, EM, pass@1, BLEU, etc.).
6. **Reporting:** Aggregates and logs raw scores, detailed outputs, and saves results in JSON/CSV.

---
## **Directory Structure**

eka_eval/
├── core/
│ ├── model_loader.py
│ └── evaluator.py
├── benchmarks/
│ ├── benchmark_registry.py
│ ├── base_benchmark.py
│ └── tasks/
│ ├── general/
│ │ ├── mmlu.py
│ │ └── ...
│ ├── code/
│ │ ├── humaneval.py
│ │ └── mbpp.py
│ ├── math/
│ │ ├── gsm8k.py
│ │ └── math_eval.py
│ ├── reading_comprehension/
│ │ ├── boolq.py
│ │ ├── squad.py
│ │ └── quac.py
│ ├── indic/
│ │ ├── init.py
│ │ ├── boolq_in/
│ │ │ ├── init.py
│ │ │ ├── prompts.py
│ │ │ ├── normalizers.py
│ │ │ └── evaluator.py
│ │ └── ... (other Indic benchmarks)
│ └── custom_loader.py
├── results/
│ └── result_manager.py
├── utils/
│ ├── logging_setup.py
│ └── gpu_utils.py
├── scripts/
│ ├── run_evaluation_suite.py
│ └── evaluation_worker.py
├── config/
│ └── benchmark_definitions.json
├── results_output/
├── tests/
├── .github/
│ └── workflows/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt

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

**Evaluate Falcon-7B on MMLU (English, zero-shot):**
```bash
python eka_eval.py --model_name_or_path tiiuae/falcon-7b --benchmark mmlu --language en --batch_size 8 --shots 0 --output_file results_mmlu_falcon7b.json
```

**Evaluate EKA model on MILU (Hindi, zero-shot):**
```bash
python eka_eval.py --model_name_or_path eka-ai/eka-7b --benchmark milu --language hi --batch_size 8 --shots 0 --output_file results_milu_eka7b.json
```

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

