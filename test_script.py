# TOP_LEVEL_eka_eval/test_import.py
import sys
import os

# Add the current directory to sys.path just to be absolutely sure
# (though it should be there by default when running `python test_import.py`)
# CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# if CURRENT_DIR not in sys.path:
#    sys.path.insert(0, CURRENT_DIR)

print("--- sys.path (from test_import.py) ---")
for p in sys.path:
    print(p)
print("--------------------------------------")

try:
    print("Attempting to import eka_eval...")
    import eka_eval # Just try to import the top-level package
    print("SUCCESS: Imported eka_eval")
    print(f"eka_eval found at: {eka_eval.__file__}")

    print("Attempting to import eka_eval.benchmarks...")
    import eka_eval.benchmarks
    print("SUCCESS: Imported eka_eval.benchmarks")
    print(f"eka_eval.benchmarks found at: {eka_eval.benchmarks.__file__}")


    print("Attempting to import eka_eval.benchmarks.benchmark_registry...")
    from eka_eval.benchmarks.benchmark_registry import BenchmarkRegistry
    print("SUCCESS: Imported BenchmarkRegistry from eka_eval.benchmarks.benchmark_registry")
    print(BenchmarkRegistry)

except ModuleNotFoundError as e:
    print(f"FAIL: ModuleNotFoundError: {e}")
except ImportError as e:
    print(f"FAIL: ImportError: {e}")
except Exception as e:
    print(f"FAIL: An unexpected error occurred: {e}")