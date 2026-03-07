"""
Split management utilities for reproducible train/val/test splits.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Callable, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SplitStrategy(Enum):
    """Data splitting strategies"""
    TEMPORAL = "temporal"           # Simple temporal split
    EXPANDING = "expanding"         # Walk-forward with expanding window
    ROLLING = "rolling"             # Walk-forward with rolling window
    COMBINATORIAL = "combinatorial" # CPCV


@dataclass
class SplitConfig:
    """Configuration for data splitting"""
    strategy: SplitStrategy = SplitStrategy.TEMPORAL
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    n_folds: int = 5
    min_train_size: int = 252
    min_test_size: int = 63
    embargo_size: int = 5


class SplitManager:
    """
    Manages data splits for evaluation.

    Ensures reproducible, non-leaking splits across all models.
    """

    def __init__(self, config: SplitConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self._split_hash: Optional[str] = None

    def compute_split_hash(self, n_samples: int) -> str:
        """Compute hash of split configuration for versioning"""
        config_str = f"{self.config.strategy.value}_{self.config.train_ratio}_{n_samples}_{self.seed}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def create_temporal_split(
        self,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal train/val/test split indices.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))

        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end, val_end)
        test_indices = np.arange(val_end, n_samples)

        self._split_hash = self.compute_split_hash(n_samples)

        logger.info(f"Temporal split created: train={len(train_indices)}, "
                   f"val={len(val_indices)}, test={len(test_indices)}")

        return train_indices, val_indices, test_indices

    def validate_sequence_boundaries(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        sequence_length: int,
        forecast_horizon: int,
    ) -> bool:
        """Ensure sequences/windows do not cross split boundaries."""
        window = sequence_length + forecast_horizon
        if len(train_indices) < window or len(val_indices) < window or len(test_indices) < window:
            raise ValueError("Window (sequence_length + forecast_horizon) exceeds split length; adjust splits or window.")
        max_train = train_indices.max() if len(train_indices) else -1
        min_val = val_indices.min() if len(val_indices) else 0
        min_test = test_indices.min() if len(test_indices) else 0

        # Last index used by a train window
        last_train_usage = max_train
        if last_train_usage >= min_val:
            raise ValueError("Train sequences overlap validation boundary; adjust sequence_length or split ratios.")

        last_val_usage = val_indices.max() if len(val_indices) else min_test
        if last_val_usage >= min_test:
            raise ValueError("Validation sequences overlap test boundary; adjust sequence_length or split ratios.")

        return True

    def create_walk_forward_splits(
        self,
        n_samples: int,
        timestamps: Optional[pd.DatetimeIndex] = None,
        validator_factory: Optional[Callable[..., Any]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create walk-forward validation splits.

        Returns:
            List of (train_indices, test_indices) tuples
        """
        method = 'anchored' if self.config.strategy == SplitStrategy.EXPANDING else 'rolling'

        if validator_factory is None:
            raise ValueError("validator_factory must be provided for walk-forward splits")

        validator = validator_factory(
            method=method,
            n_folds=self.config.n_folds,
            min_train_size=self.config.min_train_size,
            min_test_size=self.config.min_test_size,
            embargo_size=self.config.embargo_size
        )

        folds = validator.generate_folds(n_samples, timestamps)
        splits = []
        for fold in folds:
            train_idx = np.arange(fold.train_start, fold.train_end)
            test_idx = np.arange(fold.test_start, fold.test_end)
            splits.append((train_idx, test_idx))

        self._split_hash = self.compute_split_hash(n_samples)
        logger.info(f"Created {len(splits)} walk-forward splits using {method} strategy")
        return splits

    @property
    def split_hash(self) -> Optional[str]:
        return self._split_hash
