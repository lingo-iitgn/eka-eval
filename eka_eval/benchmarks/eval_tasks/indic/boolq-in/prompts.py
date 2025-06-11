# eka_eval/benchmarks/tasks/indic/boolq_in.py

import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate # Hugging Face evaluate library
import numpy as np
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


def format_prompt_for_boolq(passage: str, question: str) -> str:
    """Formats the prompt for the BoolQ task."""
   
    return f"""निम्नलिखित गद्यांश को पढ़ें और प्रश्न का उत्तर 'हाँ' या 'नहीं' में दें।
गद्यांश:
{passage}

प्रश्न:
{question}

उत्तर (हाँ/नहीं):"""

