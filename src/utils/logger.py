"""
Logging configuration using loguru
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from .config import get_config


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "500 MB",
    retention: str = "10 days",
) -> None:
    """
    Setup loguru logger with file and console output

    Args:
        log_file: Optional log file path. If None, uses config log directory
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files
        retention: How long to keep old logs
    """
    # Remove default logger
    logger.remove()

    # Add console handler with nice formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file handler
    if log_file is None:
        config = get_config()
        log_file = config.log_dir / "pinn_finance.log"

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    logger.info(f"Logger initialized. Logging to {log_file}")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance

    Args:
        name: Optional logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Global flag to track if logger has been initialized
_logger_initialized = False


def ensure_logger_initialized():
    """
    Ensure logger is initialized exactly once per process
    """
    global _logger_initialized
    if not _logger_initialized:
        setup_logger()
        _logger_initialized = True
