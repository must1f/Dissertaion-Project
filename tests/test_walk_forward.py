"""
Unit tests for Walk-Forward Validation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.walk_forward import (
    WalkForwardValidator,
    WalkForwardFold,
    TimeSeriesCrossValidator,
    create_walk_forward_splits
)


class TestWalkForwardValidator:
    """Test WalkForwardValidator class"""

    def test_basic_split(self):
        """Test basic walk-forward split"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=500,
            validation_size=100,
            step_size=100,
            mode='expanding'
        )

        folds = validator.split()

        # Should generate multiple folds
        assert len(folds) > 0

        # First fold should have correct sizes
        first_fold = folds[0]
        assert first_fold.train_end_idx - first_fold.train_start_idx == 500
        assert first_fold.val_end_idx - first_fold.val_start_idx == 100

    def test_expanding_mode(self):
        """Test expanding window mode"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=300,
            validation_size=100,
            step_size=100,
            mode='expanding'
        )

        folds = validator.split()

        # Training size should increase
        for i in range(1, len(folds)):
            prev_train_size = folds[i-1].train_end_idx - folds[i-1].train_start_idx
            curr_train_size = folds[i].train_end_idx - folds[i].train_start_idx

            assert curr_train_size > prev_train_size

    def test_rolling_mode(self):
        """Test rolling window mode"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=300,
            validation_size=100,
            step_size=100,
            mode='rolling'
        )

        folds = validator.split()

        # Training size should stay constant
        train_sizes = [
            fold.train_end_idx - fold.train_start_idx
            for fold in folds
        ]

        assert all(size == train_sizes[0] for size in train_sizes)

    def test_no_overlap(self):
        """Test that train and validation don't overlap"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=500,
            validation_size=100,
            mode='expanding'
        )

        folds = validator.split()

        for fold in folds:
            # Training should end before validation starts
            assert fold.train_end_idx <= fold.val_start_idx

    def test_no_lookahead(self):
        """Test that validation always comes after training"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=500,
            validation_size=100,
            mode='expanding'
        )

        folds = validator.split()

        for fold in folds:
            # All training indices should be before validation indices
            assert fold.train_end_idx <= fold.val_start_idx
            assert fold.train_start_idx < fold.val_start_idx

    def test_fold_progression(self):
        """Test that folds progress forward in time"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=300,
            validation_size=100,
            step_size=100,
            mode='expanding'
        )

        folds = validator.split()

        for i in range(1, len(folds)):
            # Each fold should start after the previous
            assert folds[i].val_start_idx > folds[i-1].val_start_idx

    def test_get_train_val_indices(self):
        """Test getting train and validation indices"""
        validator = WalkForwardValidator(
            n_samples=1000,
            initial_train_size=500,
            validation_size=100,
            mode='expanding'
        )

        folds = validator.split()
        first_fold = folds[0]

        train_indices, val_indices = validator.get_train_val_indices(first_fold)

        assert len(train_indices) == 500
        assert len(val_indices) == 100
        assert train_indices[-1] < val_indices[0]


class TestWalkForwardFold:
    """Test WalkForwardFold dataclass"""

    def test_fold_creation(self):
        """Test fold creation"""
        fold = WalkForwardFold(
            train_start_idx=0,
            train_end_idx=500,
            val_start_idx=500,
            val_end_idx=600,
            fold_num=0
        )

        assert fold.train_start_idx == 0
        assert fold.train_end_idx == 500
        assert fold.val_start_idx == 500
        assert fold.val_end_idx == 600
        assert fold.fold_num == 0


class TestTimeSeriesCrossValidator:
    """Test TimeSeriesCrossValidator"""

    def test_blocked_cv(self):
        """Test blocked time series CV"""
        splits = TimeSeriesCrossValidator.blocked_cv(
            n_samples=1000,
            n_splits=5
        )

        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Training should come before test
            assert train_idx[-1] < test_idx[0]

    def test_anchored_cv(self):
        """Test anchored walk-forward CV"""
        splits = TimeSeriesCrossValidator.anchored_cv(
            n_samples=1000,
            min_train_size=500,
            test_size=100,
            step_size=100
        )

        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Training starts at 0
            assert train_idx[0] == 0
            # Training before test
            assert train_idx[-1] < test_idx[0]

    def test_anchored_expanding(self):
        """Test that anchored CV expands training set"""
        splits = TimeSeriesCrossValidator.anchored_cv(
            n_samples=1000,
            min_train_size=300,
            test_size=100,
            step_size=100
        )

        train_sizes = [len(train_idx) for train_idx, _ in splits]

        # Each split should have larger training set
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1]


class TestDateBasedSplits:
    """Test date-based walk-forward splits"""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe with dates"""
        dates = pd.date_range(start='2014-01-01', end='2024-01-01', freq='D')
        n = len(dates)

        return pd.DataFrame({
            'time': dates,
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'volume': np.random.uniform(1e6, 1e7, n)
        })

    def test_date_splits(self, sample_df):
        """Test date-based splitting"""
        splits = create_walk_forward_splits(
            sample_df,
            initial_train_years=5,
            validation_months=3,
            step_months=1,
            mode='expanding',
            date_column='time'
        )

        assert len(splits) > 0

        for train_df, val_df in splits:
            # Validation should come after training
            assert train_df['time'].max() < val_df['time'].min()

    def test_date_splits_rolling(self, sample_df):
        """Test rolling date-based splits"""
        splits = create_walk_forward_splits(
            sample_df,
            initial_train_years=5,
            validation_months=3,
            step_months=3,
            mode='rolling',
            date_column='time'
        )

        assert len(splits) > 0


class TestEdgeCases:
    """Test edge cases"""

    def test_small_dataset(self):
        """Test with dataset smaller than initial train size"""
        validator = WalkForwardValidator(
            n_samples=100,
            initial_train_size=500,  # Larger than dataset
            validation_size=100,
            mode='expanding'
        )

        folds = validator.split()

        # Should return empty or handle gracefully
        assert len(folds) == 0

    def test_minimum_splits(self):
        """Test getting minimum number of splits"""
        validator = WalkForwardValidator(
            n_samples=700,
            initial_train_size=500,
            validation_size=100,
            step_size=100,
            mode='expanding'
        )

        folds = validator.split()

        # Should get at least 1 fold
        assert len(folds) >= 1

    def test_max_train_size(self):
        """Test max training size constraint"""
        validator = WalkForwardValidator(
            n_samples=2000,
            initial_train_size=300,
            validation_size=100,
            step_size=100,
            mode='expanding',
            max_train_size=500  # Cap training size
        )

        folds = validator.split()

        for fold in folds:
            train_size = fold.train_end_idx - fold.train_start_idx
            assert train_size <= 500


class TestValidatorMetrics:
    """Test computing metrics across walk-forward folds"""

    def test_sharpe_stability(self):
        """Test computing Sharpe ratio stability across folds"""
        np.random.seed(42)

        # Simulate Sharpe ratios across folds
        sharpe_ratios = np.random.normal(1.0, 0.5, 10)

        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else np.inf

        # Coefficient of variation should be reasonable
        assert cv > 0

    def test_consistency_metric(self):
        """Test computing consistency metric"""
        # Simulate returns across folds
        fold_returns = [0.05, 0.02, -0.01, 0.03, 0.04, -0.02, 0.01, 0.02]

        profitable_folds = sum(1 for r in fold_returns if r > 0)
        consistency = profitable_folds / len(fold_returns)

        assert 0 <= consistency <= 1
        assert consistency == 6/8  # 6 profitable out of 8

    def test_average_drawdown(self):
        """Test computing average drawdown across folds"""
        # Simulate drawdowns
        drawdowns = [-0.05, -0.08, -0.03, -0.12, -0.06]

        avg_drawdown = np.mean(drawdowns)
        max_drawdown = np.min(drawdowns)

        assert avg_drawdown < 0
        assert max_drawdown == -0.12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
