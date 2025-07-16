# eka_eval/utils/constants.py

import os

# --- Directory Paths ---
# It's often good to define these relative to some known project root if possible,
# or assume they are created relative to where the script is run or a configured base path.
# For now, these are just default names. Your scripts (orchestrator/worker) will
# typically construct full paths using these defaults and a base results/config directory.

# Default directory name for storing evaluation results CSV files.
DEFAULT_RESULTS_DIRNAME = "results_output"

# Default directory name for storing benchmark configuration files (like benchmark_config.py).
DEFAULT_CONFIG_DIRNAME = "config"

# Default directory name for storing model or evaluation checkpoints.
DEFAULT_CHECKPOINTS_DIRNAME = "checkpoints"


# --- File Names ---
# Default name for the main benchmark configuration Python file.
BENCHMARK_CONFIG_FILENAME = "benchmark_config.py"

# Default name for the CSV file storing aggregated results.
# You might make this model-specific in your scripts, e.g., f"{model_name}_results.csv"
# Or have a single file as you originally did.
DEFAULT_RESULTS_CSV_FILENAME = "calculated_results.csv"


# --- Special Strings / Keys ---
# Common key used for average scores in results DataFrames or dictionaries.
AVERAGE_SCORE_KEY = "Average"

# Default status messages that might be used in results logging
STATUS_PENDING = "Pending"
STATUS_COMPLETED = "Completed"
STATUS_PRECALCULATED = "PreCalculated"
STATUS_EVALUATION_ERROR = "EvaluationError"
STATUS_MODEL_LOAD_FAILED = "ModelLoadFailed"
STATUS_EVAL_FUNCTION_NOT_FOUND = "EvalFunctionNotFound"
STATUS_INVALID_RETURN_FORMAT = "InvalidReturnFormat"
STATUS_AGGREGATED = "Aggregated" # For average scores

# Special tokens, if you find them used in multiple modules (e.g., model_loader and specific tasks)
# For example, if your model_loader hardcodes "[END]" but a task also needs to know about it.
# SPECIAL_END_TOKEN = "[END]"
# PAD_TOKEN = "[PAD]" # If you add a default one in model_loader

# --- Task-Specific Constants (Optional) ---
# If some benchmarks have specific, fixed configurations that are better as constants
# than repeated strings/numbers in their task_args.
# Example (illustrative, you might not need this for BoolQ if defaults are in the task file):
# BOOLQ_DEFAULT_MAX_NEW_TOKENS = 10
# HUMANEVAL_DEFAULT_K_VALUES = [1]


# --- Other Potential Constants ---
# Default logging levels (though usually set via argparse)
# DEFAULT_LOG_LEVEL = "INFO"

# Default batch size (if you want a central default, though often an argparse default)
# DEFAULT_BATCH_SIZE = 1


# --- Example of constructing a full path using these constants ---
# This is how you might use them in other scripts, not part of the constants file itself.
# def get_default_results_csv_path(base_dir="."):
#     return os.path.join(base_dir, DEFAULT_RESULTS_DIRNAME, DEFAULT_RESULTS_CSV_FILENAME)