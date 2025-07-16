import logging
import sys

def setup_logging(level=logging.INFO, log_to_console=True, log_file_path=None, process_id=None):
    """Set up logging configuration for the application."""
    handlers = []
    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file_path:
        try:
            handlers.append(logging.FileHandler(log_file_path, mode='a'))
        except Exception as e:
            print(f"Warning: Could not set up file logging to '{log_file_path}': {e}", file=sys.stderr)
    log_format = '%(asctime)s %(process)d [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    logging.info("Logging setup complete.")
    if log_file_path and any(isinstance(h, logging.FileHandler) for h in handlers):
        logging.info(f"Logging to console and file: {log_file_path}")
    elif log_file_path:
        logging.warning(f"File logging requested to '{log_file_path}' but handler was not set up.")
    else:
        logging.info("Logging to console.")
