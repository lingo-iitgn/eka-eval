import importlib
import copy
import logging

logger = logging.getLogger(__name__)

class BenchmarkRegistry:
    """Registry for managing and resolving benchmark definitions."""

    def __init__(self, config_module_path="eka_eval.config.benchmark_config"):
        """Initialize registry and load benchmark config."""
        try:
            config_module = importlib.import_module(config_module_path)
            self.benchmarks = copy.deepcopy(config_module.BENCHMARK_CONFIG)
            logger.info(f"Benchmark configuration loaded from {config_module_path}.")
        except ImportError:
            logger.error(f"Failed to import config from: {config_module_path}", exc_info=True)
            self.benchmarks = {}
        except AttributeError:
            logger.error(f"BENCHMARK_CONFIG not found in: {config_module_path}", exc_info=True)
            self.benchmarks = {}
        self._validate_config()

    def _validate_config(self):
        """Validate structure of loaded benchmark config."""
        if not isinstance(self.benchmarks, dict):
            logger.error("BENCHMARK_CONFIG is not a dictionary.")
            self.benchmarks = {}
            return
        for task_group_name, task_group_data in self.benchmarks.items():
            if not isinstance(task_group_data, dict):
                logger.error(f"Task group '{task_group_name}' data is not a dictionary.")
                continue
            if "evaluation_function" in task_group_data:
                if not isinstance(task_group_data["evaluation_function"], str):
                    logger.error(f"Task group '{task_group_name}' has invalid 'evaluation_function'.")
            else:
                for bm_name, bm_details in task_group_data.items():
                    if not isinstance(bm_details, dict):
                        logger.error(f"Benchmark '{bm_name}' in group '{task_group_name}' is not a dictionary.")
                        continue
                    if "evaluation_function" not in bm_details:
                        logger.error(f"Benchmark '{bm_name}' in group '{task_group_name}' missing 'evaluation_function'.")
                    elif not isinstance(bm_details["evaluation_function"], str):
                        logger.error(f"Benchmark '{bm_name}' in group '{task_group_name}' has invalid 'evaluation_function'.")
        logger.debug("Benchmark configuration validation complete.")

    def get_task_groups(self) -> list:
        """Return list of all top-level task group names."""
        return list(self.benchmarks.keys())

    def get_benchmarks_for_group(self, task_group_name: str) -> list:
        """Return list of benchmark names in a task group."""
        if task_group_name not in self.benchmarks:
            logger.warning(f"Task group '{task_group_name}' not found.")
            return []
        group_content = self.benchmarks[task_group_name]
        if "evaluation_function" in group_content:
            return [task_group_name]
        return list(group_content.keys())

    def get_benchmark_details(self, task_group_name: str, benchmark_name: str) -> dict:
        """Return configuration dictionary for a specific benchmark."""
        if task_group_name not in self.benchmarks:
            logger.warning(f"Task group '{task_group_name}' not found for details.")
            return {}
        group_content = self.benchmarks[task_group_name]
        if task_group_name == benchmark_name and "evaluation_function" in group_content:
            return group_content
        if isinstance(group_content, dict):
            return group_content.get(benchmark_name, {})
        return {}

    def get_evaluation_function_path_str(self, task_group_name: str, benchmark_name: str) -> str:
        """Get the evaluation function string path for a benchmark."""
        details = self.get_benchmark_details(task_group_name, benchmark_name)
        return details.get("evaluation_function") if isinstance(details, dict) else None

    def resolve_evaluation_function(self, task_group_name: str, benchmark_name: str):
        """Resolve evaluation function string to a callable."""
        details = self.get_benchmark_details(task_group_name, benchmark_name)
        if not details:
            logger.error(f"No details for {task_group_name} - {benchmark_name}.")
            return None
        
        eval_function_path_str = details.get("evaluation_function")
        if not eval_function_path_str:
            logger.error(f"No 'evaluation_function' path for {task_group_name} - {benchmark_name}.")
            return None
        
        logger.debug(f"Raw evaluation_function string: '{eval_function_path_str}'")
        
        try:
            if details.get("is_custom"):
                # Custom benchmark - use path as-is
                module_name, function_name = eval_function_path_str.rsplit('.', 1)
                logger.debug(f"Loading custom function: {module_name}.{function_name}")
            else:
                # Standard benchmark - check if it already includes function name
                if eval_function_path_str.startswith("eka_eval.benchmarks.tasks."):
                    # Old format: "eka_eval.benchmarks.tasks.code.humaneval" (missing function name)
                    # Need to add the function name
                    parts = eval_function_path_str.split('.')
                    if len(parts) >= 5:  # eka_eval.benchmarks.tasks.category.module
                        module_file = parts[-1]  # e.g., "humaneval"
                        function_name = f"evaluate_{module_file}"  # e.g., "evaluate_humaneval"
                        module_name = eval_function_path_str
                    else:
                        logger.error(f"Invalid full path format: {eval_function_path_str}")
                        return None
                else:
                    # New format: "code.humaneval.evaluate_humaneval" (relative path with function)
                    full_path = f"eka_eval.benchmarks.tasks.{eval_function_path_str}"
                    module_name, function_name = full_path.rsplit('.', 1)
                
                logger.debug(f"Loading standard function: {module_name}.{function_name}")
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the function
            if not hasattr(module, function_name):
                logger.error(f"Function '{function_name}' not found in module '{module_name}'")
                logger.debug(f"Available functions in module: {[name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]}")
                return None
            
            function_callable = getattr(module, function_name)
            logger.info(f"Successfully resolved: {module_name}.{function_name}")
            return function_callable
            
        except ImportError as e:
            logger.error(f"ImportError loading module '{module_name}': {e}")
            return None
        except (AttributeError, ValueError) as e:
            logger.error(f"Error resolving function from '{eval_function_path_str}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error resolving '{eval_function_path_str}': {e}")
            return None

    def add_custom_benchmark_definition(self, task_group: str, bm_name: str,
                                        custom_module_path: str, custom_function_name: str,
                                        description: str = "Custom benchmark"):
        """Add a custom benchmark definition to the registry."""
        if not all([task_group, bm_name, custom_module_path, custom_function_name]):
            logger.error("All fields required for custom benchmark. Skipping.")
            return False
        if task_group not in self.benchmarks:
            self.benchmarks[task_group] = {}
            logger.info(f"Created new task group: '{task_group}'")
        eval_func_identifier = f"{custom_module_path}.{custom_function_name}"
        self.benchmarks[task_group][bm_name] = {
            "description": description,
            "evaluation_function": eval_func_identifier,
            "is_custom": True,
        }
        logger.info(f"Added custom benchmark '{bm_name}' under '{task_group}' with eval function: {eval_func_identifier}")
        return True
