#!/usr/bin/env python3
"""
Test script for MultiPL-E evaluation system.
This script sets up the prompt configuration and runs a basic test.
"""

import os
import sys
import json
from pathlib import Path

def setup_prompts():
    """Set up the prompt configuration file."""
    # Get the current working directory (should be eka_eval root)
    root_dir = Path.cwd()
    
    # Create the prompts directory structure
    prompts_dir = root_dir / "prompts"
    code_prompts_dir = prompts_dir / "code"
    code_prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple prompt configuration for testing
    config = {
        "multiple_0shot": {
            "template": "Complete the following {language} function by implementing the function body. Only provide the implementation code that goes inside the function.\n\n{problem_prompt}",
            "description": "Zero-shot MultiPL-E prompt for code completion in any language"
        },
        "multiple_3shot": {
            "template_prefix": "You are an expert {language} programmer. Complete the following functions based on their documentation and requirements. Here are some examples:\n\n",
            "few_shot_example_template": "{problem_prompt}{solution}",
            "few_shot_separator": "\n\n",
            "template_suffix": "Now complete this function:\n\n{problem_prompt}",
            "description": "Few-shot MultiPL-E prompt template"
        },
        "default_few_shot_examples_cpp": [
            {
                "problem_prompt": "bool has_close_elements(vector<float> numbers, float threshold){\n",
                "solution": "    for (int i = 0; i < numbers.size(); i++) {\n        for (int j = i + 1; j < numbers.size(); j++) {\n            if (abs(numbers[i] - numbers[j]) < threshold) {\n                return true;\n            }\n        }\n    }\n    return false;\n}\n"
            }
        ],
        "evaluation_settings": {
            "default_max_new_tokens": 512,
            "default_batch_size": 1,
            "default_samples_per_task": 1,
            "default_k_values": [1],
            "generation_params": {
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.95
            },
            "timeout_seconds": 30
        }
    }
    
    # Save the configuration
    config_path = code_prompts_dir / "multiple.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created prompt configuration at: {config_path}")
    return config_path

def main():
    print("🚀 Setting up MultiPL-E test environment...")
    
    # Setup prompts
    config_path = setup_prompts()
    
    if not config_path.exists():
        print("❌ Failed to create prompt configuration")
        return False
    
    print("✅ Prompt configuration created successfully")
    
    # Test the configuration can be loaded
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded: {len(config)} keys found")
        
        # Check required keys
        required_keys = ["multiple_0shot", "multiple_3shot"]
        for key in required_keys:
            if key in config:
                print(f"  ✅ Found template: {key}")
            else:
                print(f"  ❌ Missing template: {key}")
                return False
        
        print("\n🎉 Setup completed successfully!")
        print("\nNow you can run:")
        print("  python3 -m eka_eval.benchmarks.tasks.code.multiple")
        print("\nOr run this simple test first:")
        print("  cd /path/to/eka_eval && python3 setup_multiple.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)