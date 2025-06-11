# eka_eval/utils/logging_setup.py

import logging
import sys
import os
# Change 'process_id' to 'worker_id' here
def setup_logging(level=logging.INFO,
                  log_to_console: bool = True,
                  log_file_path: str = None,
                  worker_id: str = None): # <--- CHANGED PARAMETER NAME
    """Set up logging configuration for the application."""
    handlers = []
    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_file_path:
        # If you want unique log files per worker:
        if worker_id and '%' in log_file_path: # e.g. "app_worker_%s.log"
             actual_log_file_path = log_file_path % worker_id
        elif worker_id and log_file_path: # e.g. "app.log" -> "app_Orchestrator.log"
             base, ext = os.path.splitext(log_file_path)
             actual_log_file_path = f"{base}_{worker_id}{ext}"
        else:
             actual_log_file_path = log_file_path

        try:
            # Use actual_log_file_path here
            file_handler = logging.FileHandler(actual_log_file_path, mode='a')
            handlers.append(file_handler)
        except Exception as e:
            # Use actual_log_file_path in error message
            print(f"Warning: Could not set up file logging to '{actual_log_file_path}': {e}", file=sys.stderr)
            actual_log_file_path = None # Ensure it's None if setup failed

    # --- Adjusting log format to use worker_id if provided ---
    log_format_parts = ['%(asctime)s']
    if worker_id:
        log_format_parts.append(f'[{worker_id}]') # Add worker_id to the log line
    # %(process)d gives the OS PID, which is also useful
    log_format_parts.extend(['P%(process)d', '[%(levelname)s]', '%(name)s:%(lineno)d', '-', '%(message)s'])
    log_format = ' '.join(log_format_parts)
    date_format = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Override any existing basicConfig
    )
    logging.info("Logging setup complete.") # This line itself will now include the worker_id if provided

    # Check if file logging was actually set up
    file_handler_present = any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)

    if log_file_path and file_handler_present:
        # Use actual_log_file_path in the confirmation message
        logging.info(f"Logging to console and file: {actual_log_file_path}")
    elif log_file_path and not file_handler_present:
        logging.warning(f"File logging requested to '{log_file_path}' (intended: '{actual_log_file_path if 'actual_log_file_path' in locals() else 'N/A'}') but handler was not set up.")
    else:
        logging.info("Logging to console.")