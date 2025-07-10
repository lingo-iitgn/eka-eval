#!/usr/bin/env python3
"""
Simple function to load the extracted FLORES data
"""
import json
import os
from datasets import Dataset

def load_flores_data(lang_code, direction="enxx"):
    """Load FLORES data for a specific language and direction"""
    data_file = f"flores_{lang_code}_{direction}_test.json"
    data_path = os.path.join("/home/samridhiraj/eka_eval/eka_eval/benchmarks/tasks/indic/flores_data_extracted_method1", data_file)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    return Dataset.from_list(examples)

# Example usage:
# dataset = load_flores_data("hi", "enxx")
# print(f"Loaded {len(dataset)} examples")
