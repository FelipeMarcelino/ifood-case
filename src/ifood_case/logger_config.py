"""Centralized logging configuration for the iFood case study project.

This module provides a setup function to configure a root logger that outputs
to both the console and a file, ensuring consistent logging across all modules.
"""
import logging
import sys
from pathlib import Path


def setup_logging():
    """Sets up the root logger for the application.

    Configures a logger to stream to both the console (stdout) and a log file
    (logs/pipeline.log) with a standardized format.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    log_file = log_dir / "ifood_case_pipeline.log"
    # --- END OF LOGIC ---

    # Create the logs directory using the absolute path.
    # exist_ok=True prevents an error if the directory already exists.
    log_dir.mkdir(parents=True, exist_ok=True)


    # Define the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    print("Logging configured to output to console and logs/ifood_case_pipeline.log")
