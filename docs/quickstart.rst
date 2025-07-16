===========
Quick Start
===========

🚀 Basic Evaluation
===================

The simplest way to get started with Eka-Eval is using the interactive evaluation script:

.. code-block:: bash

   # Run interactive evaluation
   python3 scripts/run_benchmarks.py

This will launch an interactive session where you can:

1. Select your model (Hugging Face or local path)
2. Choose benchmark task groups
3. Configure evaluation parameters
4. View real-time results

📺 Video Demonstration
======================

.. raw:: html

   <video width="100%" controls>
     <source src="_static/demonstration.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

⚡ Command Line Examples
========================

Evaluate Math Benchmarks
-------------------------

.. code-block:: bash

   # Evaluate specific model on math benchmarks
   python3 scripts/run_benchmarks.py \
       --model "google/gemma-2b" \
       --task_groups "MATH AND REASONING" \
       --benchmarks "GSM8K,MATH"

Multi-language Evaluation
--------------------------

.. code-block:: bash

   # Multi-language evaluation
   python3 scripts/run_benchmarks.py \
       --model "sarvamai/sarvam-1" \
       --task_groups "INDIC BENCHMARKS" \
       --languages "hi,bn,gu"

Code Generation Evaluation
---------------------------

.. code-block:: bash

   # Code generation evaluation
   python3 scripts/run_benchmarks.py \
       --model "microsoft/CodeT5-large" \
       --task_groups "CODE GENERATION" \
       --pass_k "1,5,10"

🔧 Interactive Evaluation Workflow
==================================

1. Model Selection
------------------

.. code-block:: text

   Enter model source ('1' for Hugging Face, '2' for Local Path): 1
   Enter Hugging Face model name: google/gemma-2b

2. Task Group Selection
-----------------------

.. code-block:: text

   --- Available Benchmark Task Groups ---
   1. CODE GENERATION          7. MMLU
   2. MATH AND REASONING       8. MMLU-Pro  
   3. READING COMPREHENSION    9. IFEval
   4. COMMONSENSE REASONING   10. BBH
   5. WORLD KNOWLEDGE         11. AGIEval
   6. LONG CONTEXT           12. INDIC BENCHMARKS

   Select task group #(s): 2 12

3. Benchmark Selection
----------------------

.. code-block:: text

   --- Select benchmarks for MATH AND REASONING ---
   1. GSM8K                    4. ARC-Challenge
   2. MATH                     5. ALL
   3. GPQA                     6. SKIP

   Select benchmark #(s): 1 2

4. Execution & Results
----------------------

.. code-block:: text

   [Worker 0 (GPU 0)] Loading model: google/gemma-2b (2.0B parameters)
   [Worker 0 (GPU 0)] Running GSM8K evaluation...
   [Worker 0 (GPU 0)] GSM8K Accuracy: 42.3% (527/1247)
   [Worker 0 (GPU 0)] Running MATH evaluation...
   [Worker 0 (GPU 0)] MATH Accuracy: 12.1% (601/5000)

   Results saved to: results_output/calculated.csv

🧪 Standalone Benchmark Testing
===============================

Test individual benchmarks for development and debugging:

.. code-block:: bash

   # Test individual benchmarks
   python eka_eval/benchmarks/tasks/math/gsm8k.py --model_name_test gpt2
   python eka_eval/benchmarks/tasks/indic/boolq_in.py --target_languages_test hi en
   python eka_eval/benchmarks/tasks/long_context/infinitebench.py --dataset_split_test longdialogue_qa_eng

📊 Understanding Results
========================

Console Output
--------------

Results are displayed in a formatted table:

.. code-block:: text

   | Model      | Task                | Benchmark      | Score   |
   |------------|--------------------|--------------------|---------|
   | gemma-2b   | MATH AND REASONING | GSM8K             | 42.3%   |
   | gemma-2b   | MATH AND REASONING | MATH              | 12.1%   |
   | gemma-2b   | INDIC BENCHMARKS   | BoolQ-IN          | 67.8%   |
   | gemma-2b   | INDIC BENCHMARKS   | MMLU-IN           | 39.2%   |

CSV Results
-----------

Aggregated results are saved to ``results_output/calculated.csv``:

.. list-table::
   :header-rows: 1

   * - Model
     - Size (B)
     - Task
     - Benchmark
     - Score
     - Timestamp
     - Status
   * - gemma-2b
     - 2.00
     - MATH AND REASONING
     - GSM8K
     - 42.3%
     - 2024-01-15T10:30:45
     - Completed
   * - gemma-2b
     - 2.00
     - INDIC BENCHMARKS
     - BoolQ-IN
     - 67.8%
     - 2024-01-15T11:15:20
     - Completed

Language-Specific Metrics
-------------------------

For multilingual benchmarks, results are broken down by language:

.. code-block:: json

   {
     "BoolQ-IN": 67.8,
     "BoolQ-IN_hi": 65.2,
     "BoolQ-IN_bn": 70.1,
     "BoolQ-IN_en": 74.5,
     "BoolQ-IN_gu": 63.8
   }

🎯 Common Use Cases
===================

Model Comparison
----------------

.. code-block:: bash

   # Compare multiple models on the same benchmarks
   for model in "google/gemma-2b" "microsoft/DialoGPT-medium" "meta-llama/Llama-2-7b-hf"
   do
     python3 scripts/run_benchmarks.py \
       --model "$model" \
       --task_groups "MATH AND REASONING" \
       --benchmarks "GSM8K"
   done

Quick Testing
-------------

.. code-block:: bash

   # Quick test with limited examples
   python3 scripts/run_benchmarks.py \
       --model "google/gemma-2b" \
       --task_groups "MATH AND REASONING" \
       --max_examples 100

Large Model Evaluation
----------------------

.. code-block:: bash

   # Evaluate large models with quantization
   python3 scripts/run_benchmarks.py \
       --model "meta-llama/Llama-2-70b-hf" \
       --quantization "4bit" \
       --batch_size 1 \
       --task_groups "INDIC BENCHMARKS"

⚙️ Configuration Options
========================

Key command-line options:

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``--model``
     - Model name or path
     - ``"google/gemma-2b"``
   * - ``--task_groups``
     - Benchmark categories
     - ``"MATH AND REASONING"``
   * - ``--benchmarks``
     - Specific benchmarks
     - ``"GSM8K,MATH"``
   * - ``--languages``
     - Target languages
     - ``"hi,bn,en"``
   * - ``--quantization``
     - Model quantization
     - ``"4bit"``
   * - ``--batch_size``
     - Inference batch size
     - ``8``
   * - ``--max_examples``
     - Limit dataset size
     - ``1000``
   * - ``--save_detailed``
     - Save detailed results
     - ``true``

🚨 First Run Checklist
======================

Before running your first evaluation:

✅ **Environment Setup**
   - Python 3.8+ installed
   - Virtual environment activated
   - All dependencies installed

✅ **GPU Configuration**
   - CUDA drivers installed (if using GPU)
   - Sufficient GPU memory available
   - Set ``CUDA_VISIBLE_DEVICES`` if needed

✅ **Model Access**
   - Hugging Face token configured (if needed)
   - Internet connection for model downloads
   - Sufficient disk space for model cache

✅ **Output Directory**
   - ``results_output/`` directory exists
   - Write permissions available

Next Steps
==========

- Explore :doc:`advanced_usage` for performance optimization
- Check :doc:`benchmarks` for detailed benchmark information
- See :doc:`troubleshooting` if you encounter issues
- Visit :doc:`examples` for more usage patterns