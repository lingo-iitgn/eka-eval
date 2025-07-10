eka-eval/
├── eka_eval/                     # Main library code (will be importable)
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Logic for initializing models, tokenizers, pipelines
│   │   └── evaluator.py          # Core logic for running a single benchmark evaluation (was in main.py)
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── benchmark_registry.py # Manages BENCHMARK_CONFIG, loads definitions, discovers tasks
│   │   ├── base_benchmark.py     # (Optional) An abstract base class for all benchmark tasks
│   │   └── tasks/                # Directory for actual benchmark evaluation logic
│   │       ├── __init__.py
│   │       ├── code/
│   │       │   ├── __init__.py
│   │       │   ├── humaneval.py
│   │       │   └── mbpp.py
│   │       ├── math/
│   │       │   ├── __init__.py
│   │       │   ├── gsm8k.py
│   │       │   └── math_eval.py  # Renamed from 'math.py' to avoid conflict with stdlib
│   │       ├── reading_comprehension/
│   │       │   ├── __init__.py
│   │       │   ├── boolq.py
│   │       │   ├── squad.py
│   │       │   └── quac.py
│   │       ├── indic/
│   │       │   ├── __init__.py
│   │       │   └── mmlu_in.py    # Corrected name from your example
│   │       ├── general/          # For MMLU, BBH, AGIEval etc.
│   │       │   ├── __init__.py
│   │       │   ├── mmlu.py
│   │       │   └── ...
│   │       └── custom_loader.py  # Helper for loading dynamically defined custom tasks
│   ├── results/
│   │   ├── __init__.py
│   │   └── result_manager.py     # Handles loading, saving, and querying evaluation results (CSV)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gpu_utils.py
│   │   └── logging_setup.py      # Centralized logging configuration
│   └── constants.py              # (Optional) For global constants like default paths, special token names
├── scripts/                      # User-facing executable scripts
│   ├── run_evaluation_suite.py   # New name for run_benchmarks.py (the orchestrator)
│   └── evaluation_worker.py      # New name for main.py (the script run by each process)
├── config/                       # Configuration files
│   └── benchmark_definitions.json # BENCHMARK_CONFIG will live here as a JSON/YAML file
├── results_output/               # Default directory for generated CSV results
├── tests/                        # For unit and integration tests
│   └── ...
├── .github/
│   └── workflows/
│       └── ...
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml                # Important for packaging
└── requirements.txt