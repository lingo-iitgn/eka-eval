import json
import os
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")

_prompt_cache: Dict[str, Dict] = {}

def _load_prompt_file(file_path: str) -> Optional[Dict]:
    if file_path in _prompt_cache:
        logger.debug(f"Using cached prompt file: {file_path}")
        return _prompt_cache[file_path]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _prompt_cache[file_path] = data
            logger.debug(f"Successfully loaded and cached prompt file: {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from prompt file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading prompt file {file_path}: {e}")
    return None

def get_available_prompts(benchmark_name: str, category: Optional[str] = None) -> List[str]:
    file_paths = _get_prompt_file_paths(benchmark_name, category)
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            prompt_config = _load_prompt_file(file_path)
            if prompt_config:
                template_keys = [
                    key for key in prompt_config.keys() 
                    if isinstance(prompt_config[key], dict) and 
                    ('template' in prompt_config[key] or 'template_prefix' in prompt_config[key])
                ]
                return template_keys
    
    logger.warning(f"No prompt templates found for benchmark '{benchmark_name}' in category '{category}'")
    return []

def _get_prompt_file_paths(benchmark_name: str, category: Optional[str] = None) -> List[str]:
    paths = []
    
    if category:
        paths.extend([
            os.path.join(PROMPT_DIR, category, f"{benchmark_name}.json"),
            os.path.join(PROMPT_DIR, category, f"{benchmark_name}_prompts.json"),
        ])
    
    paths.extend([
        os.path.join(PROMPT_DIR, f"{benchmark_name}.json"),
        os.path.join(PROMPT_DIR, f"{benchmark_name}_prompts.json"),
        os.path.join(PROMPT_DIR, "general", f"{benchmark_name}.json"),
        os.path.join(PROMPT_DIR, "general", f"{benchmark_name}_prompts.json"),
    ])
    
    return paths

def get_prompt_template(
    benchmark_name: str, 
    template_name: str, 
    specific_task_group: Optional[str] = None 
) -> Optional[Dict[str, Any]]:
    file_paths = _get_prompt_file_paths(benchmark_name, specific_task_group)
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            prompt_config = _load_prompt_file(file_path)
            if prompt_config and template_name in prompt_config:
                logger.debug(f"Found template '{template_name}' in {file_path}")
                return prompt_config[template_name]
    
    logger.warning(f"Prompt template '{template_name}' not found for benchmark '{benchmark_name}' in category '{specific_task_group}'")
    logger.debug(f"Searched paths: {file_paths}")
    return None

def get_prompt_data(
    benchmark_name: str,
    data_key: str,
    specific_task_group: Optional[str] = None
) -> Optional[Any]:
    file_paths = _get_prompt_file_paths(benchmark_name, specific_task_group)
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            prompt_config = _load_prompt_file(file_path)
            if prompt_config and data_key in prompt_config:
                logger.debug(f"Found data '{data_key}' in {file_path}")
                return prompt_config[data_key]
    
    logger.warning(f"Data '{data_key}' not found for benchmark '{benchmark_name}' in category '{specific_task_group}'")
    return None

def format_prompt(template_dict: Dict[str, Any], **kwargs: Any) -> str:
    template_str = template_dict.get("template")
    if not template_str:
        logger.error(f"Prompt template dictionary is missing 'template' key: {template_dict}")
        return "Error: Prompt template string not found."

    try:
        formatted = template_str.format(**kwargs)
        logger.debug(f"Successfully formatted prompt template")
        return formatted
    except KeyError as e:
        logger.error(f"Missing key '{e}' for prompt formatting. Available keys: {list(kwargs.keys())}")
        logger.error(f"Template: {template_str}")
        return f"Error: Prompt formatting failed due to missing key {e}."
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        return f"Error: Prompt formatting failed: {e}"

def format_few_shot_prompt(
    template_dict: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]], 
    main_question_data: Dict[str, Any]       
) -> str:
    prefix = template_dict.get("template_prefix", "")
    example_template = template_dict.get("few_shot_example_template")
    suffix_template = template_dict.get("template_suffix")
    separator = template_dict.get("few_shot_separator", "\n\n")

    if not example_template or not suffix_template:
        logger.error("Few-shot prompt template dictionary is missing required keys (few_shot_example_template, template_suffix).")
        logger.error(f"Available keys: {list(template_dict.keys())}")
        return "Error: Invalid few-shot template."

    try:
        formatted_prefix = prefix.format(**main_question_data) if prefix else ""
    except KeyError as e:
        logger.error(f"Missing key '{e}' in main_question_data for prefix formatting")
        formatted_prefix = prefix

    formatted_examples = []
    for i, ex_data in enumerate(few_shot_examples):
        try:
            formatted_example = example_template.format(**ex_data)
            formatted_examples.append(formatted_example)
        except KeyError as e:
            logger.error(f"Missing key '{e}' in few-shot example {i}: {ex_data}. Skipping example.")
            continue
        except Exception as e:
            logger.error(f"Error formatting few-shot example {i}: {e}. Skipping example.")
            continue

    try:
        formatted_suffix = suffix_template.format(**main_question_data)
    except KeyError as e:
        logger.error(f"Missing key '{e}' in main_question_data for suffix formatting")
        formatted_suffix = suffix_template

    examples_str = separator.join(formatted_examples)
    if formatted_examples and examples_str:
        full_prompt = formatted_prefix + examples_str + separator + formatted_suffix
    else:
        logger.warning("No valid few-shot examples formatted, using suffix only")
        full_prompt = formatted_prefix + formatted_suffix

    logger.debug(f"Successfully formatted few-shot prompt with {len(formatted_examples)} examples")
    return full_prompt

def validate_prompt_template(template_dict: Dict[str, Any], template_type: str = "simple") -> bool:
    if not isinstance(template_dict, dict):
        logger.error(f"Template is not a dictionary: {type(template_dict)}")
        return False

    if template_type == "simple":
        required_keys = ["template"]
    elif template_type == "few_shot":
        required_keys = ["few_shot_example_template", "template_suffix"]
    else:
        logger.error(f"Unknown template type: {template_type}")
        return False

    missing_keys = [key for key in required_keys if key not in template_dict]
    if missing_keys:
        logger.error(f"Template missing required keys: {missing_keys}")
        logger.error(f"Available keys: {list(template_dict.keys())}")
        return False

    return True

def list_all_available_prompts() -> Dict[str, Dict[str, List[str]]]:
    available = {}
    
    if not os.path.exists(PROMPT_DIR):
        logger.warning(f"Prompts directory not found: {PROMPT_DIR}")
        return available

    for root, dirs, files in os.walk(PROMPT_DIR):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, PROMPT_DIR)
                category = os.path.dirname(relative_path) if os.path.dirname(relative_path) else "root"
                benchmark = os.path.splitext(os.path.basename(file))[0]
                
                if benchmark.endswith('_prompts'):
                    benchmark = benchmark[:-8]
                
                prompts = get_available_prompts(benchmark, category if category != "root" else None)
                
                if prompts:
                    if category not in available:
                        available[category] = {}
                    available[category][benchmark] = prompts

    return available

def get_few_shot_examples(benchmark_name: str, num_examples: int = 5, category: Optional[str] = None) -> List[Dict]:
    examples_key = f"default_few_shot_examples_{benchmark_name}"
    examples = get_prompt_data(benchmark_name, examples_key, category)
    
    if examples and isinstance(examples, list):
        return examples[:num_examples]
    
    logger.warning(f"No few-shot examples found for {benchmark_name}")
    return []

def test_prompt_utils():
    print("Testing Prompt Utils...")
    
    all_prompts = list_all_available_prompts()
    print(f"Available prompts: {all_prompts}")
    
    mmlu_templates = get_available_prompts("mmlu", "general")
    print(f"MMLU templates: {mmlu_templates}")
    
    template = get_prompt_template("mmlu", "mmlu_0shot", "general")
    print(f"MMLU 0-shot template: {template}")
    
    if template:
        formatted = format_prompt(template, subject="Mathematics", question="What is 2+2?", choices_str="A. 3\nB. 4\nC. 5\nD. 6")
        print(f"Formatted prompt: {formatted}")

if __name__ == "__main__":
    test_prompt_utils()
