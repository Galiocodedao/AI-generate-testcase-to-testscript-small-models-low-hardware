"""
Logging utility for the AI Test Script Generator
"""

import logging
import sys
from typing import Union

def setup_logger(log_level: Union[int, str] = logging.INFO) -> None:
    """
    Set up the logger for the application
    
    Args:
        log_level: Logging level (e.g., logging.INFO)
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.addHandler(console_handler)
