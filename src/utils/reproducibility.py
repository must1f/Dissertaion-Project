"""
Reproducibility utilities - ensuring deterministic behavior across runs
"""

import os
import random
import numpy as np
import torch
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries

    Args:
        seed: Random seed value
    """
    logger.info(f"Setting random seed to {seed} for reproducibility")

    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Random seeds set successfully")


def log_system_info():
    """Log system information for reproducibility"""
    import platform
    import torch

    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)

    # Python
    logger.info(f"Python version: {platform.python_version()}")

    # PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # NumPy
    logger.info(f"NumPy version: {np.__version__}")

    # System
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")

    logger.info("=" * 80)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for PyTorch

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


class ReproducibilityContext:
    """Context manager for reproducible code blocks"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random_state = None
        self.np_state = None
        self.torch_state = None

    def __enter__(self):
        # Save current states
        self.random_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()

        # Set seed
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore states
        random.setstate(self.random_state)
        np.random.set_state(self.np_state)
        torch.random.set_rng_state(self.torch_state)
