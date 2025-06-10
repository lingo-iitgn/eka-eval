# eka_eval/utils/gpu_utils.py
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_available_gpus() -> List[int]:
    """
    Checks for available CUDA-enabled GPUs and returns their physical IDs.
    Returns an empty list if CUDA is not available or no GPUs are found.
    """
    available_gpu_ids = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            available_gpu_ids = list(range(num_gpus))
            logger.info(f"Found {num_gpus} CUDA-enabled GPU(s): {available_gpu_ids}")
        else:
            logger.warning("torch.cuda.is_available() is True, but torch.cuda.device_count() is 0. No GPUs found.")
    else:
        logger.info("CUDA is not available on this system.")
    return available_gpu_ids

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Available GPUs:", get_available_gpus())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")