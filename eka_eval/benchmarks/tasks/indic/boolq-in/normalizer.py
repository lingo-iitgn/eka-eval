
# eka_eval/benchmarks/tasks/indic/boolq_in.py

import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate # Hugging Face evaluate library
import numpy as np
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)



YES_STRINGS_ENGLISH = ["yes", "true", "correct", "affirmative"]
NO_STRINGS_ENGLISH = ["no", "false", "incorrect", "negative"]
YES_STRINGS_HINDI = ["हाँ", "हां", "सही", "सत्य"] 
NO_STRINGS_HINDI = ["नहीं", "नही", "गलत", "असत्य"] 

def normalize_answer_to_bool_int(answer_text: Any) -> int:
    """
    Normalizes a given answer text (or boolean/int) to 1 (yes/true), 0 (no/false), or -1 (unknown/unparseable).
    """
    if answer_text is None:
        return -1
    if isinstance(answer_text, bool):
        return int(answer_text)
    if isinstance(answer_text, int):
        return 1 if answer_text == 1 else (0 if answer_text == 0 else -1)
    if not isinstance(answer_text, str):
        return -1

    text_original_case_stripped = answer_text.strip()
    text_lower_stripped = text_original_case_stripped.lower()


    if text_original_case_stripped in YES_STRINGS_HINDI: return 1
    if text_original_case_stripped in NO_STRINGS_HINDI: return 0

    if text_lower_stripped in YES_STRINGS_ENGLISH: return 1
    if text_lower_stripped in NO_STRINGS_ENGLISH: return 0

 
    for yes_hindi in YES_STRINGS_HINDI:
        if yes_hindi in text_original_case_stripped: return 1
    for no_hindi in NO_STRINGS_HINDI:
        if no_hindi in text_original_case_stripped: return 0

    for yes_eng in YES_STRINGS_ENGLISH:
        if yes_eng in text_lower_stripped: return 1
    for no_eng in NO_STRINGS_ENGLISH:
        if no_eng in text_lower_stripped: return 0

    return -1