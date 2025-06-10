# eka_eval/utils/file_utils.py
import os
import json
import logging

logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}", exc_info=True)
            raise 


def read_json_file(file_path: str) -> dict | list | None:
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}", exc_info=True)
        return None

def write_json_file(data: dict | list, file_path: str, indent: int = 2) -> bool:
    """Writes data to a JSON file."""
    try:
        ensure_dir_exists(os.path.dirname(file_path)) 
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Successfully wrote JSON data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}", exc_info=True)
        return False

