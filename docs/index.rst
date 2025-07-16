============================================
Eka-Eval Documentation
============================================

.. image:: _static/eka-eval-logo.png
   :width: 200px
   :align: center
   :alt: Eka-Eval Logo

A Comprehensive Evaluation Framework for Large Language Models in Indian Languages.

Overview
========

Eka-Eval is the official evaluation pipeline for the EKA project (`eka.soket.ai <https://eka.soket.ai>`_), designed to provide comprehensive, fair, and transparent benchmarking for large language models (LLMs). Our framework supports both global and India-centric evaluations, with special emphasis on multilingual capabilities across Indian languages.

🎯 Why Eka-Eval?
================

* 🌏 **Global + India-First**: Combines international benchmarks with India-specific evaluations
* 🔬 **Rigorous & Reproducible**: Standardized evaluation protocols with detailed logging
* 🚀 **Production-Ready**: Optimized for efficiency with quantization and multi-GPU support
* 🔧 **Extensible**: Easy integration of custom benchmarks and evaluation logic
* 📊 **Transparent**: Comprehensive reporting with detailed error analysis

Key Features
============

🎯 Comprehensive Benchmark Coverage
-----------------------------------

* **17+ English Benchmarks**: MMLU, GSM8K, HumanEval, ARC-Challenge, and more
* **12+ Indic Benchmarks**: MMLU-IN, BoolQ-IN, ARC-Challenge-IN, MILU, and others
* **Specialized Tasks**: Code generation, mathematical reasoning, long-context understanding
* **Multi-modal Support**: Text, code, and multilingual evaluation capabilities

🌐 Multilingual Excellence
--------------------------

* **11 Indian Languages**: Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu
* **Smart Language Handling**: Automatic script recognition and Hindi-English letter mapping
* **Per-Language Metrics**: Detailed breakdown of performance across languages

⚡ Performance & Scalability
----------------------------

* **Multi-GPU Support**: Distributed evaluation across multiple GPUs
* **Quantization Ready**: 4-bit/8-bit quantization for efficient large model evaluation
* **Batched Inference**: Optimized throughput with configurable batch sizes
* **Memory Management**: Smart resource cleanup and CUDA cache management

🔧 Developer Experience
-----------------------

* **Modular Architecture**: Clean separation of concerns with extensible design
* **Prompt System**: Template-based prompts with language-specific customization
* **Rich Configuration**: JSON-based benchmark configs with validation
* **Detailed Logging**: Comprehensive debug information and progress tracking

📊 Advanced Reporting
---------------------

* **Multiple Output Formats**: CSV summaries, JSONL details, console tables
* **Error Analysis**: Per-instance results for debugging and improvement
* **Reproducibility**: Timestamped results with full configuration tracking
* **Flexible Metrics**: Accuracy, F1, BLEU, pass@k, and custom metrics

Quick Start
===========

Basic Evaluation
----------------

.. code-block:: bash

   # Run interactive evaluation
   python3 scripts/run_benchmarks.py

Command Line Examples
--------------------

.. code-block:: bash

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

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   project_structure

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   benchmarks
   supported_languages
   evaluation_workflow
   advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: Development:

   api_reference
   contributing
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Resources:

   examples
   results_reporting
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`