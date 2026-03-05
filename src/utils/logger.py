"""
Logging helpers with loguru when available, otherwise a lightweight stdlib fallback.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from .config import get_config

# Set TRAINING_DEBUG=1 in environment for DEBUG level logging
DEBUG_MODE = os.environ.get("TRAINING_DEBUG", "0") == "1"
DEFAULT_LOG_LEVEL = "DEBUG" if DEBUG_MODE else "INFO"

# Prefer loguru if installed; otherwise provide a minimal compatible shim.
try:  # pragma: no cover - tiny import guard
    from loguru import logger as _loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:  # Fallback to stdlib logging
    LOGURU_AVAILABLE = False

    class _StdLoggerAdapter:
        """Minimal loguru-like wrapper around stdlib logging."""

        def __init__(self, name: Optional[str] = None):
            self._logger = logging.getLogger(name or "pinn")

        def bind(self, **kwargs):
            # loguru's bind returns a new logger; here we just return self.
            return self

        def success(self, *args, **kwargs):
            return self._logger.info(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(self._logger, item)

    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        stream=sys.stderr,
    )
    _loguru_logger = _StdLoggerAdapter()


def setup_logger(
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    rotation: str = "500 MB",
    retention: str = "10 days",
) -> None:
    """
    Setup logger with file and console output.

    Args:
        log_file: Optional log file path. If None, uses config log directory
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to DEBUG if TRAINING_DEBUG=1, else INFO.
        rotation: When to rotate log files
        retention: How long to keep old logs
    """
    # Use default level based on TRAINING_DEBUG environment variable
    if level is None:
        level = DEFAULT_LOG_LEVEL

    # Remove default logger
    if LOGURU_AVAILABLE:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True,
        )

        if log_file is None:
            config = get_config()
            log_file = config.log_dir / "pinn_finance.log"

        _loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
        _loguru_logger.info(f"Logger initialized. Logging to {log_file}")
    else:
        # stdlib fallback: ensure file handler if requested
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance

    Args:
        name: Optional logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    if name:
        return _loguru_logger.bind(name=name)
    return _loguru_logger


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
