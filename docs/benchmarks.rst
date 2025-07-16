=====================
Supported Benchmarks
=====================

🌍 Global Benchmarks
====================

+-------------------+----------------------------------------+------------+----------------+
| Category          | Benchmarks                             | Languages  | Metrics        |
+===================+========================================+============+================+
| 📚 Knowledge      | MMLU, MMLU-Pro, TriviaQA,             | English    | Accuracy       |
|                   | NaturalQuestions                       |            |                |
+-------------------+----------------------------------------+------------+----------------+
| 🧮 Mathematics   | GSM8K, MATH, GPQA, ARC-Challenge      | English    | Accuracy       |
+-------------------+----------------------------------------+------------+----------------+
| 💻 Code Generation| HumanEval, MBPP, HumanEval+, MBPP+    | Python,    | pass@1,        |
|                   |                                        | Multi-PL   | pass@k         |
+-------------------+----------------------------------------+------------+----------------+
| 🧠 Reasoning     | BBH, AGIEval, HellaSwag, WinoGrande   | English    | Accuracy       |
+-------------------+----------------------------------------+------------+----------------+
| 📖 Reading       | SQuAD, QuAC, BoolQ, XQuAD             | English +  | F1, EM,        |
|                   |                                        | Others     | Accuracy       |
+-------------------+----------------------------------------+------------+----------------+
| 📏 Long Context  | InfiniteBench, ZeroSCROLLS,           | English    | Task-specific  |
|                   | Needle-in-Haystack                    |            |                |
+-------------------+----------------------------------------+------------+----------------+

🇮🇳 India-Centric Benchmarks
=============================

+-------------------+----------------+-------------------------------------------+----------------+
| Benchmark         | Languages      | Description                               | Metrics        |
+===================+================+===========================================+================+
| MMLU-IN           | 11 Indic + EN  | Knowledge understanding across subjects   | Accuracy       |
+-------------------+----------------+-------------------------------------------+----------------+
| BoolQ-IN          | 11 Indic + EN  | Yes/No question answering                 | Accuracy       |
+-------------------+----------------+-------------------------------------------+----------------+
| ARC-Challenge-IN  | 11 Indic + EN  | Science reasoning questions               | Accuracy       |
+-------------------+----------------+-------------------------------------------+----------------+
| MILU              | 11 Indic + EN  | AI4Bharat's multilingual understanding   | Accuracy       |
+-------------------+----------------+-------------------------------------------+----------------+
| GSM8K-IN          | Hindi, Others  | Math word problems in Indian languages    | Accuracy       |
+-------------------+----------------+-------------------------------------------+----------------+
| IndicGenBench     | Multiple       | Generation tasks for Indic languages     | Task-specific  |
+-------------------+----------------+-------------------------------------------+----------------+
| Flores-IN         | 22 Languages   | Translation quality assessment            | BLEU, ChrF     |
+-------------------+----------------+-------------------------------------------+----------------+
| XQuAD-IN          | 11 Languages   | Cross-lingual reading comprehension      | F1, EM         |
+-------------------+----------------+-------------------------------------------+----------------+

Benchmark Categories
====================

📚 Knowledge & Understanding
----------------------------

**MMLU (Massive Multitask Language Understanding)**
   Measures a model's world knowledge and problem-solving ability across 57 academic subjects.

**MMLU-Pro**
   Enhanced version of MMLU with more challenging questions and improved evaluation.

**TriviaQA**
   Reading comprehension dataset featuring trivia questions paired with evidence documents.

🧮 Mathematical Reasoning
-------------------------

**GSM8K**
   Grade school math word problems requiring multi-step reasoning.

**MATH**
   Competition mathematics problems covering algebra, geometry, number theory, and more.

**GPQA**
   Graduate-level Google-proof Q&A benchmark in biology, physics, and chemistry.

💻 Code Generation
------------------

**HumanEval**
   Hand-written programming problems for measuring functional correctness.

**MBPP (Mostly Basic Python Problems)**
   Crowd-sourced Python programming problems.

**HumanEval+ / MBPP+**
   Enhanced versions with additional test cases for more robust evaluation.

🧠 Reasoning & Logic
--------------------

**BBH (Big Bench Hard)**
   Challenging tasks from BigBench requiring complex reasoning.

**ARC-Challenge**
   Science questions designed for grade-school level AI systems.

**HellaSwag**
   Commonsense inference about physical situations.

**WinoGrande**
   Large-scale dataset for commonsense reasoning.

📖 Reading Comprehension
------------------------

**SQuAD**
   Stanford Question Answering Dataset for machine reading comprehension.

**BoolQ**
   Yes/no questions that require difficult entailment-like inference.

**QuAC**
   Question Answering in Context dataset for information-seeking dialogue.

📏 Long Context Understanding
-----------------------------

**InfiniteBench**
   Comprehensive benchmark for evaluating long context understanding.

**ZeroSCROLLS**
   Zero-shot evaluation suite for long text understanding.

**Needle-in-Haystack**
   Tests ability to retrieve information from very long contexts.

🌐 Indic Language Benchmarks
============================

**MMLU-IN**
   Translated and adapted version of MMLU for 11 Indian languages, maintaining cultural and linguistic appropriateness.

**BoolQ-IN**
   Indian language adaptation of BoolQ with culturally relevant questions and contexts.

**ARC-Challenge-IN**
   Science reasoning questions translated to Indian languages with appropriate cultural context.

**MILU (Multilingual Indic Language Understanding)**
   Comprehensive benchmark developed by AI4Bharat covering various NLU tasks across Indian languages.

**GSM8K-IN**
   Mathematical word problems adapted for Indian contexts and languages, including currency, cultural references, and naming conventions.

**IndicGenBench**
   Generation tasks specifically designed for Indic languages, including text summarization, dialogue generation, and creative writing.

Evaluation Metrics
==================

**Accuracy**
   Percentage of correct predictions out of total predictions.

**F1 Score**
   Harmonic mean of precision and recall, useful for imbalanced datasets.

**Exact Match (EM)**
   Percentage of predictions that match the reference answer exactly.

**BLEU Score**
   Measures quality of text generation by comparing n-gram overlap with references.

**pass@k**
   Percentage of problems where at least one out of k generated solutions is correct.

**ChrF**
   Character-level F-score for translation evaluation.

Custom Metrics
==============

Eka-Eval supports custom metrics for specialized evaluation needs:

.. code-block:: python

   def custom_metric(predictions, references):
       # Your custom evaluation logic
       score = calculate_score(predictions, references)
       return score
   
   # Register the metric
   evaluator.add_custom_metric('my_metric', custom_metric)