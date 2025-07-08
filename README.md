<div align="center">
  <!-- Eka Logo -->
  <a href="https://eka.soket.ai/" target="_blank">
    <img 
      width="118" 
      alt="eka-eval logo" 
      src="https://github.com/user-attachments/assets/2822b114-39bb-4c19-8808-accd8b415b3a" 
      style="margin-bottom: 10px;" 
    />
  </a>
  <h1>Eka-Eval</h1>
  <h2>A Comprehensive Evaluation Framework for Large Language Models in Indian Languages.</h2>

  <!-- Badges Row -->
  <p>
    <img src="https://img.shields.io/badge/release-v1.0-blue.svg" alt="Release v1.0" />
    <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License Apache 2.0" />
    <a href="https://colab.research.google.com/github/your_repo/eka-eval/blob/main/notebooks/quick_start.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
    </a>
    <a href="https://discord.gg/pQaFJ857">
      <img src="https://img.shields.io/discord/308323056592486420?label=Join%20Community&logo=discord&colorB=7289DA" alt="Discord Community" />
    </a>
  </p>

  <!-- Navigation Links -->
  <p>
    <a href="#key-features">Key Features</a> •
    <a href="#supported-benchmarks">Supported Benchmarks</a> •
    <a href="#quick-start">Getting Started</a> •
    <a href="#results-and-reporting">Reporting</a> •
    <a href="#project-ethos">Project Ethos</a>
  </p>
</div>

---

## **Overview**

**Eka-Eval** is the official evaluation pipeline for the EKA project ([eka.soket.ai](https://eka.soket.ai)), designed to provide comprehensive, fair, and transparent benchmarking for large language models (LLMs). Our framework supports both global and India-centric evaluations, with special emphasis on multilingual capabilities across Indian languages.

### 🎯 **Why Eka-Eval?**

- **🌏 Global + India-First**: Combines international benchmarks with India-specific evaluations
- **🔬 Rigorous & Reproducible**: Standardized evaluation protocols with detailed logging
- **🚀 Production-Ready**: Optimized for efficiency with quantization and multi-GPU support
- **🔧 Extensible**: Easy integration of custom benchmarks and evaluation logic
- **📊 Transparent**: Comprehensive reporting with detailed error analysis

---

## **Key Features**

### 🎯 **Comprehensive Benchmark Coverage**
- **17+ English Benchmarks**: MMLU, GSM8K, HumanEval, ARC-Challenge, and more
- **12+ Indic Benchmarks**: MMLU-IN, BoolQ-IN, ARC-Challenge-IN, MILU, and others
- **Specialized Tasks**: Code generation, mathematical reasoning, long-context understanding
- **Multi-modal Support**: Text, code, and multilingual evaluation capabilities

### 🌐 **Multilingual Excellence**
- **11 Indian Languages**: Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu
- **Smart Language Handling**: Automatic script recognition and Hindi-English letter mapping
- **Per-Language Metrics**: Detailed breakdown of performance across languages

### ⚡ **Performance & Scalability**
- **Multi-GPU Support**: Distributed evaluation across multiple GPUs
- **Quantization Ready**: 4-bit/8-bit quantization for efficient large model evaluation
- **Batched Inference**: Optimized throughput with configurable batch sizes
- **Memory Management**: Smart resource cleanup and CUDA cache management

### 🔧 **Developer Experience**
- **Modular Architecture**: Clean separation of concerns with extensible design
- **Prompt System**: Template-based prompts with language-specific customization
- **Rich Configuration**: JSON-based benchmark configs with validation
- **Detailed Logging**: Comprehensive debug information and progress tracking

### 📊 **Advanced Reporting**
- **Multiple Output Formats**: CSV summaries, JSONL details, console tables
- **Error Analysis**: Per-instance results for debugging and improvement
- **Reproducibility**: Timestamped results with full configuration tracking
- **Flexible Metrics**: Accuracy, F1, BLEU, pass@k, and custom metrics

---

## **Installation**

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-org/eka-eval.git
cd eka-eval
```

### 2. **Environment Setup**
```bash
# Create virtual environment
python3 -m venv eka-env
source eka-env/bin/activate  # On Windows: eka-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Required Dependencies**
```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
evaluate>=0.4.0
accelerate>=0.24.0
bitsandbytes>=0.41.0  # For quantization
pandas>=1.5.0
tqdm>=4.64.0
numpy>=1.24.0
```

### 4. **Authentication (Optional)**
For private models or gated datasets:
```bash
huggingface-cli login
# OR
export HF_TOKEN="your_hf_token_here"
```

---

## **🚀 Quick Start**

### **Basic Evaluation**
```bash
# Run interactive evaluation
python3 scripts/run_benchmarks.py
```

### **Command Line Examples**
```bash
# Evaluate specific model on math benchmarks
python3 scripts/run_benchmarks.py \
    --model "google/gemma-2b" \
    --task_groups "MATH AND REASONING" \
    --benchmarks "GSM8K,MATH"

# Multi-language evaluation
python3 scripts/run_benchmarks.py \
    --model "sarvamai/sarvam-1" \
    --task_groups "INDIC BENCHMARKS" \
    --languages "hi,bn,gu"

# Code generation evaluation
python3 scripts/run_benchmarks.py \
    --model "microsoft/CodeT5-large" \
    --task_groups "CODE GENERATION" \
    --pass_k "1,5,10"
```

### **Standalone Benchmark Testing**
```bash
# Test individual benchmarks
python eka_eval/benchmarks/tasks/math/gsm8k.py --model_name_test gpt2
python eka_eval/benchmarks/tasks/indic/boolq_in.py --target_languages_test hi en
python eka_eval/benchmarks/tasks/long_context/infinitebench.py --dataset_split_test longdialogue_qa_eng
```

---

## **🏗️ Project Structure**

```
eka-eval/
├── 📦 eka_eval/                    # Core library
│   ├── 🧪 benchmarks/              # Evaluation logic
│   │   ├── tasks/
│   │   │   ├── 💻 code/            # HumanEval, MBPP, etc.
│   │   │   ├── 🧮 math/            # GSM8K, MATH, etc.
│   │   │   ├── 🌏 indic/           # Indic language benchmarks
│   │   │   ├── 🧠 reasoning/       # ARC, HellaSwag, etc.
│   │   │   ├── 📚 long_context/    # InfiniteBench, etc.
│   │   │   └── 🎯 general/         # MMLU, AGIEval, etc.
│   │   └── benchmark_registry.py
│   ├── ⚙️ core/                    # Model loading & evaluation
│   ├── 🔧 utils/                   # Utilities & helpers
│   └── 📋 config/                  # Benchmark configurations
├── 🚀 scripts/                     # Execution scripts
│   ├── run_benchmarks.py          # Main orchestrator
│   └── evaluation_worker.py       # Worker process logic
├── 📊 results_output/              # Evaluation results
├── 🎯 prompts/                     # Prompt templates
│   ├── math/                      # Math benchmark prompts
│   ├── indic/                     # Indic benchmark prompts
│   ├── general/                   # General benchmark prompts
│   └── long_context/              # Long context prompts
└── 📝 requirements.txt
```

---

## **Supported Benchmarks**

### 🌍 **Global Benchmarks**

| **Category** | **Benchmarks** | **Languages** | **Metrics** |
|--------------|---------------|---------------|-------------|
| **📚 Knowledge** | MMLU, MMLU-Pro, TriviaQA, NaturalQuestions | English | Accuracy |
| **🧮 Mathematics** | GSM8K, MATH, GPQA, ARC-Challenge | English | Accuracy |
| **💻 Code Generation** | HumanEval, MBPP, HumanEval+, MBPP+ | Python, Multi-PL | pass@1, pass@k |
| **🧠 Reasoning** | BBH, AGIEval, HellaSwag, WinoGrande | English | Accuracy |
| **📖 Reading** | SQuAD, QuAC, BoolQ, XQuAD | English + Others | F1, EM, Accuracy |
| **📏 Long Context** | InfiniteBench, ZeroSCROLLS, Needle-in-Haystack | English | Task-specific |

### 🇮🇳 **India-Centric Benchmarks**

| **Benchmark** | **Languages** | **Description** | **Metrics** |
|--------------|---------------|----------------|-------------|
| **MMLU-IN** | 11 Indic + EN | Knowledge understanding across subjects | Accuracy |
| **BoolQ-IN** | 11 Indic + EN | Yes/No question answering | Accuracy |
| **ARC-Challenge-IN** | 11 Indic + EN | Science reasoning questions | Accuracy |
| **MILU** | 11 Indic + EN | AI4Bharat's multilingual understanding | Accuracy |
| **GSM8K-IN** | Hindi, Others | Math word problems in Indian languages | Accuracy |
| **IndicGenBench** | Multiple | Generation tasks for Indic languages | Task-specific |
| **Flores-IN** | 22 Languages | Translation quality assessment | BLEU, ChrF |
| **XQuAD-IN** | 11 Languages | Cross-lingual reading comprehension | F1, EM |

### **Supported Languages**
- **English**: Primary evaluation language
- **Hindi (hi)**: देवनागरी script with smart character mapping
- **Bengali (bn)**: বাংলা script
- **Gujarati (gu)**: ગુજરાતી script  
- **Kannada (kn)**: ಕನ್ನಡ script
- **Malayalam (ml)**: മലയാളം script
- **Marathi (mr)**: मराठी script
- **Odia (or)**: ଓଡ଼ିଆ script
- **Punjabi (pa)**: ਪੰਜਾਬੀ script
- **Tamil (ta)**: தமிழ் script
- **Telugu (te)**: తెలుగు script

---

## **🔧 Interactive Evaluation Workflow**

### **1. Model Selection**
```
Enter model source ('1' for Hugging Face, '2' for Local Path): 1
Enter Hugging Face model name: google/gemma-2b
```

### **2. Task Group Selection**
```
--- Available Benchmark Task Groups ---
1. CODE GENERATION          7. MMLU
2. MATH AND REASONING       8. MMLU-Pro  
3. READING COMPREHENSION    9. IFEval
4. COMMONSENSE REASONING   10. BBH
5. WORLD KNOWLEDGE         11. AGIEval
6. LONG CONTEXT           12. INDIC BENCHMARKS

Select task group #(s): 2 12
```

### **3. Benchmark Selection**
```
--- Select benchmarks for MATH AND REASONING ---
1. GSM8K                    4. ARC-Challenge
2. MATH                     5. ALL
3. GPQA                     6. SKIP

Select benchmark #(s): 1 2
```

### **4. Execution & Results**
```
[Worker 0 (GPU 0)] Loading model: google/gemma-2b (2.0B parameters)
[Worker 0 (GPU 0)] Running GSM8K evaluation...
[Worker 0 (GPU 0)] GSM8K Accuracy: 42.3% (527/1247)
[Worker 0 (GPU 0)] Running MATH evaluation...
[Worker 0 (GPU 0)] MATH Accuracy: 12.1% (601/5000)

Results saved to: results_output/calculated.csv
```

---

## **🎯 Advanced Usage**

### **Custom Benchmark Integration**

#### **1. Create Evaluation Function**
```python
# my_benchmark.py
def evaluate_my_task(pipe, tokenizer, model_name_for_logging, device, **kwargs):
    # Your evaluation logic here
    results = {"MyTask": accuracy_score}
    return results
```

#### **2. Add Prompt Configuration**
```
prompts/custom/my_task.json

{
  "my_task_0shot": {
    "template": "Question: {question}\nAnswer:",
    "description": "Zero-shot prompt for my task"
  },
  "default_few_shot_examples": [
    {"question": "Example question", "answer": "Example answer"}
  ]
}
```

#### **3. Register in Config**
```python
# Add to benchmark_config.py
"MyTask": {
    "description": "My custom evaluation task",
    "evaluation_function": "my_project.my_benchmark.evaluate_my_task",
    "task_args": {
        "prompt_template_name_zeroshot": "my_task_0shot",
        "prompt_file_benchmark_key": "my_task",
        "prompt_file_category": "custom"
    }
}
```

### **Quantization & Optimization**
```python
# Automatic 4-bit quantization for large models
python3 scripts/run_benchmarks.py \
    --model "meta-llama/Llama-2-70b-hf" \
    --quantization "4bit" \
    --batch_size 1
```

### **Multi-GPU Evaluation**
```python
# Distributed evaluation across GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/run_benchmarks.py \
    --model "microsoft/DialoGPT-large" \
    --task_groups "ALL" \
    --num_gpus 4
```

---

## **📊 Results and Reporting**

### **📈 Aggregated Results (CSV)**
Located at `results_output/calculated.csv`:

| Model | Size (B) | Task | Benchmark | Score | Timestamp | Status |
|-------|----------|------|-----------|-------|-----------|---------|
| gemma-2b | 2.00 | MATH AND REASONING | GSM8K | 42.3% | 2024-01-15T10:30:45 | Completed |
| gemma-2b | 2.00 | INDIC BENCHMARKS | BoolQ-IN | 67.8% | 2024-01-15T11:15:20 | Completed |

### **📋 Detailed Analysis (JSONL)**
Per-benchmark detailed results in `results_output/detailed_results/`:
```json
{
  "question_id": 123,
  "question": "What is 2+2?",
  "correct_answer": "4",
  "predicted_answer": "4", 
  "is_correct": true,
  "generated_text": "The answer is 4.",
  "prompt_used": "Question: What is 2+2?\nAnswer:"
}
```

### **🖥️ Console Output**
```markdown
| Model      | Task                | Benchmark      | Score   |
|------------|--------------------|--------------------|---------|
| gemma-2b   | MATH AND REASONING | GSM8K             | 42.3%   |
| gemma-2b   | MATH AND REASONING | MATH              | 12.1%   |
| gemma-2b   | INDIC BENCHMARKS   | BoolQ-IN          | 67.8%   |
| gemma-2b   | INDIC BENCHMARKS   | MMLU-IN           | 39.2%   |
```

### **📊 Language-Specific Metrics**
```json
{
  "BoolQ-IN": 67.8,
  "BoolQ-IN_hi": 65.2,
  "BoolQ-IN_bn": 70.1,
  "BoolQ-IN_en": 74.5,
  "BoolQ-IN_gu": 63.8
}
```

---

## **⚠️ Troubleshooting**

### **Common Issues & Solutions**

| **Issue** | **Solution** |
|-----------|-------------|
| 🔴 `ModuleNotFoundError: eka_eval` | Run from project root directory |
| 🔴 CUDA Out of Memory | Reduce `generation_batch_size` or use quantization |
| 🔴 Hugging Face 404 Error | Check model name and authentication |
| 🔴 `code_eval` metric error | Set `HF_ALLOW_CODE_EVAL=1` environment variable |
| 🔴 Prompt template not found | Check prompt file exists in correct category folder |
| 🔴 Dataset loading failure | Verify dataset name and internet connection |

### **Performance Optimization**

```bash
# For large models (>7B parameters)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 scripts/run_benchmarks.py \
    --model "meta-llama/Llama-2-70b-hf" \
    --quantization "4bit" \
    --batch_size 1 \
    --max_new_tokens 256

# For faster evaluation
python3 scripts/run_benchmarks.py \
    --model "google/gemma-2b" \
    --batch_size 16 \
    --max_examples 100  # Limit dataset size for testing
```

### **Debug Mode**
```bash
# Enable detailed logging
python3 scripts/run_benchmarks.py \
    --model "google/gemma-2b" \
    --log_level DEBUG \
    --save_detailed true
```

---

## **🤝 Contributing**

We welcome contributions from the community! Here's how you can help:

### **🐛 Bug Reports**
- Use our [issue template](https://github.com/your-org/eka-eval/issues/new?template=bug_report.md)
- Include error logs, model names, and reproduction steps
- Test with minimal examples when possible

### **✨ Feature Requests**
- Propose new benchmarks or evaluation metrics
- Suggest performance improvements
- Request additional language support

### **🔧 Development**
```bash
# Fork the repository
git clone https://github.com/your-username/eka-eval.git
cd eka-eval

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python3 -m pytest tests/

# Submit pull request
```

### **📚 Documentation**
- Improve README examples
- Add benchmark documentation
- Create tutorial notebooks

---

## **🔗 References & Resources**

### **Official Benchmark Papers**
- [MMLU](https://arxiv.org/abs/2009.03300) - Hendrycks et al., ICLR 2021
- [GSM8K](https://arxiv.org/abs/2110.14168) - Cobbe et al., 2021
- [HumanEval](https://arxiv.org/abs/2107.03374) - Chen et al., 2021
- [BBH](https://arxiv.org/abs/2210.09261) - Suzgun et al., 2022
- [AGIEval](https://arxiv.org/abs/2304.06364) - Zhong et al., 2023

### **Indic Language Resources**
- [AI4Bharat](https://ai4bharat.org/) - IndicNLP toolkit and datasets
- [MILU](https://github.com/AI4Bharat/MILU) - Multilingual Indic understanding
- [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/) - Indic language evaluation

### **Related Projects**
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate) - Evaluation library
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Alternative framework
- [OpenCompass](https://github.com/InternLM/opencompass) - Comprehensive LLM evaluation

---

## **🌟 Project Ethos**

### **🔓 Open Source**
- All code and evaluation protocols are freely available
- Transparent methodology with detailed documentation
- Community-driven development and improvement

### **⚖️ Ethical AI**
- Fair and unbiased evaluation practices
- Privacy-preserving evaluation methods
- Responsible AI development guidelines

### **🇮🇳 India-First Approach**
- Comprehensive coverage of Indian languages
- Cultural and linguistic sensitivity in evaluation
- Supporting the growth of Indic AI capabilities

### **🔬 Scientific Rigor**
- Reproducible evaluation protocols
- Standardized metrics and reporting
- Peer-reviewed benchmark implementations

---
## **📄 License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## **📚 Citation**

If you use Eka-Eval in your research, please cite:

```bibtex
@misc{sinha2025ekaevalcomprehensiveevaluation,
      title={Eka-Eval : A Comprehensive Evaluation Framework for Large Language Models in Indian Languages}, 
      author={Samridhi Raj Sinha and Rajvee Sheth and Abhishek Upperwal and Mayank Singh},
      year={2025},
      eprint={2507.01853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.01853}, 
}
```

---

<div align="center">
  <h3>🚀 <strong>Eka-Eval: Powering the Future of AI Evaluation</strong> 🚀</h3>
  <p><em>Open • Ethical • Comprehensive • India-First</em></p>
  
  <p>
    <a href="https://eka.soket.ai">🌐 Website</a> •
<!--     <a href="https://discord.gg/pQaFJ857">💬 Discord</a> •
    <a href="https://github.com/your-org/eka-eval/issues">🐛 Issues</a> •
    <a href="https://github.com/your-org/eka-eval/discussions">💡 Discussions</a> -->
  </p>
</div>
