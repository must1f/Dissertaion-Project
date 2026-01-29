"""
Walk-Forward Validation for Time Series Models

Implements expanding window and rolling window validation
to avoid look-ahead bias in financial time series
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation"""
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    fold_num: int


class WalkForwardValidator:
    """
    Walk-Forward Cross-Validation

    Simulates realistic trading scenario where model is trained on historical data
    and validated on future unseen data
    """

    def __init__(
        self,
        n_samples: int,
        initial_train_size: int,
        validation_size: int,
        step_size: Optional[int] = None,
        mode: str = 'expanding',  # 'expanding' or 'rolling'
        max_train_size: Optional[int] = None
    ):
        """
        Args:
            n_samples: Total number of samples
            initial_train_size: Initial training window size
            validation_size: Validation window size
            step_size: Step size for moving window (default: validation_size)
            mode: 'expanding' (increasing train size) or 'rolling' (fixed train size)
            max_train_size: Maximum training size for expanding mode
        """
        self.n_samples = n_samples
        self.initial_train_size = initial_train_size
        self.validation_size = validation_size
        self.step_size = step_size or validation_size
        self.mode = mode
        self.max_train_size = max_train_size

        logger.info(f"WalkForwardValidator initialized: mode={mode}, "
                   f"train={initial_train_size}, val={validation_size}, step={self.step_size}")

    def split(self) -> List[WalkForwardFold]:
        """
        Generate walk-forward splits

        Returns:
            List of WalkForwardFold objects
        """
        folds = []
        fold_num = 0

        train_start = 0
        train_end = self.initial_train_size
        val_start = train_end
        val_end = val_start + self.validation_size

        while val_end <= self.n_samples:
            # Create fold
            fold = WalkForwardFold(
                train_start_idx=train_start,
                train_end_idx=train_end,
                val_start_idx=val_start,
                val_end_idx=val_end,
                fold_num=fold_num
            )
            folds.append(fold)

            logger.debug(f"Fold {fold_num}: train=[{train_start}:{train_end}], "
                        f"val=[{val_start}:{val_end}]")

            # Move window forward
            if self.mode == 'expanding':
                # Expand training window
                train_end = val_end
                if self.max_train_size is not None:
                    train_start = max(0, train_end - self.max_train_size)
            else:  # rolling
                # Keep training window size fixed
                train_size = train_end - train_start
                train_start += self.step_size
                train_end = train_start + train_size

            val_start += self.step_size
            val_end = val_start + self.validation_size
            fold_num += 1

        logger.info(f"Generated {len(folds)} walk-forward folds")

        return folds

    def get_train_val_indices(
        self,
        fold: WalkForwardFold
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get train and validation indices for a fold

        Args:
            fold: WalkForwardFold object

        Returns:
            Tuple of (train_indices, val_indices)
        """
        train_indices = np.arange(fold.train_start_idx, fold.train_end_idx)
        val_indices = np.arange(fold.val_start_idx, fold.val_end_idx)

        return train_indices, val_indices


def create_walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 5,
    validation_months: int = 3,
    step_months: int = 1,
    mode: str = 'expanding',
    date_column: str = 'time'
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward splits based on dates

    Args:
        df: DataFrame with date column
        initial_train_years: Initial training period in years
        validation_months: Validation period in months
        step_months: Step size in months
        mode: 'expanding' or 'rolling'
        date_column: Name of date column

    Returns:
        List of (train_df, val_df) tuples
    """
    # Sort by date
    df = df.sort_values(date_column).reset_index(drop=True)

    # Get date range
    start_date = df[date_column].min()
    end_date = df[date_column].max()

    # Calculate split points
    train_end = start_date + pd.DateOffset(years=initial_train_years)
    val_end = train_end + pd.DateOffset(months=validation_months)

    splits = []

    while val_end <= end_date:
        if mode == 'expanding':
            train_df = df[df[date_column] < train_end]
        else:  # rolling
            train_start = train_end - pd.DateOffset(years=initial_train_years)
            train_df = df[(df[date_column] >= train_start) & (df[date_column] < train_end)]

        val_df = df[(df[date_column] >= train_end) & (df[date_column] < val_end)]

        if len(train_df) > 0 and len(val_df) > 0:
            splits.append((train_df, val_df))
            logger.debug(f"Split {len(splits)}: train={len(train_df)}, val={len(val_df)}")

        # Move forward
        train_end += pd.DateOffset(months=step_months)
        val_end = train_end + pd.DateOffset(months=validation_months)

    logger.info(f"Created {len(splits)} date-based walk-forward splits")

    return splits


class TimeSeriesCrossValidator:
    """
    Time series cross-validation with multiple strategies
    """

    @staticmethod
    def blocked_cv(
        n_samples: int,
        n_splits: int = 5,
        test_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Blocked time series cross-validation

        Data is split into blocks, each block is used as test set once

        Args:
            n_samples: Number of samples
            n_splits: Number of splits
            test_size: Size of test set (default: n_samples // n_splits)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if test_size is None:
            test_size = n_samples // n_splits

        splits = []

        for i in range(n_splits):
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)

            if test_end > n_samples:
                break

            # Train on all data before test
            train_indices = np.arange(0, test_start)

            # Test on current block
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    @staticmethod
    def anchored_cv(
        n_samples: int,
        min_train_size: int,
        test_size: int,
        step_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Anchored walk-forward validation

        Training set starts at index 0 and expands,
        test set moves forward

        Args:
            n_samples: Number of samples
            min_train_size: Minimum training size
            test_size: Test set size
            step_size: Step size (default: test_size)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if step_size is None:
            step_size = test_size

        splits = []

        train_end = min_train_size
        test_start = train_end
        test_end = test_start + test_size

        while test_end <= n_samples:
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

            # Expand training, move test forward
            train_end = test_end
            test_start = train_end
            test_end = test_start + test_size

        return splits
