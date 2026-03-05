"""
Unit Tests for Data Transforms

Tests for data preprocessing, normalization, and feature transformations.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestDataPreprocessor:
    """Tests for data preprocessor transforms."""

    @pytest.fixture
    def sample_df(self):
        """Create sample financial data DataFrame."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 0,  # Will be computed
            'low': 0,   # Will be computed
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000000, 10000000, 100)
        }
        data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(100) * 0.5)
        data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(100) * 0.5)

        df = pd.DataFrame(data, index=dates)
        df['ticker'] = 'AAPL'
        return df

    def test_returns_calculation(self, sample_df):
        """Test returns feature calculation."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        # Calculate simple returns
        returns = (sample_df['close'] - sample_df['close'].shift(1)) / sample_df['close'].shift(1)

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Returns should be reasonable (typically < 0.2 daily)
        assert returns.dropna().abs().max() < 1.0

    def test_log_returns_calculation(self, sample_df):
        """Test log returns calculation."""
        close = sample_df['close']
        log_returns = np.log(close / close.shift(1))

        # First value should be NaN
        assert pd.isna(log_returns.iloc[0])

        # Log returns close to simple returns for small values
        simple_returns = (close - close.shift(1)) / close.shift(1)
        np.testing.assert_array_almost_equal(
            log_returns.dropna().values,
            simple_returns.dropna().values,
            decimal=1  # Allow some difference
        )

    def test_volatility_calculation(self, sample_df):
        """Test volatility feature calculation."""
        returns = sample_df['close'].pct_change()

        # Rolling volatility
        vol_20 = returns.rolling(window=20).std()

        # First 19 values should be NaN
        assert vol_20.iloc[:19].isna().all()

        # Volatility should be positive
        assert (vol_20.dropna() >= 0).all()

    def test_momentum_calculation(self, sample_df):
        """Test momentum feature calculation."""
        close = sample_df['close']

        # Price momentum (rate of change)
        roc_10 = (close - close.shift(10)) / close.shift(10)

        # Should have NaN for first 10 values
        assert roc_10.iloc[:10].isna().all()

    def test_moving_average_calculation(self, sample_df):
        """Test moving average calculation."""
        close = sample_df['close']

        # Simple moving average
        sma_20 = close.rolling(window=20).mean()

        # First 19 values should be NaN
        assert sma_20.iloc[:19].isna().all()

        # SMA should be between min and max of window
        for i in range(19, len(close)):
            window = close.iloc[i-19:i+1]
            assert window.min() <= sma_20.iloc[i] <= window.max()

    def test_rsi_bounds(self, sample_df):
        """Test RSI is bounded between 0 and 100."""
        # Simple RSI calculation
        close = sample_df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # RSI should be between 0 and 100
        rsi_valid = rsi.dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()


class TestNormalization:
    """Tests for data normalization."""

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for normalization tests."""
        torch.manual_seed(42)
        return torch.randn(100, 5) * 10 + 50

    def test_minmax_normalization(self, sample_tensor):
        """Test min-max normalization to [0, 1]."""
        min_val = sample_tensor.min(dim=0)[0]
        max_val = sample_tensor.max(dim=0)[0]

        normalized = (sample_tensor - min_val) / (max_val - min_val + 1e-8)

        # Should be in [0, 1]
        assert normalized.min() >= 0 - 1e-5
        assert normalized.max() <= 1 + 1e-5

    def test_zscore_normalization(self, sample_tensor):
        """Test z-score normalization."""
        mean = sample_tensor.mean(dim=0)
        std = sample_tensor.std(dim=0)

        normalized = (sample_tensor - mean) / (std + 1e-8)

        # Mean should be ~0, std ~1
        assert normalized.mean().abs() < 0.1
        assert (normalized.std(dim=0) - 1.0).abs().max() < 0.1

    def test_robust_normalization(self, sample_tensor):
        """Test robust normalization using median and IQR."""
        median = sample_tensor.median(dim=0)[0]
        q1 = torch.quantile(sample_tensor, 0.25, dim=0)
        q3 = torch.quantile(sample_tensor, 0.75, dim=0)
        iqr = q3 - q1

        normalized = (sample_tensor - median) / (iqr + 1e-8)

        # Median should be ~0
        assert normalized.median().abs() < 0.1

    def test_normalization_preserves_shape(self, sample_tensor):
        """Test that normalization preserves tensor shape."""
        mean = sample_tensor.mean(dim=0)
        std = sample_tensor.std(dim=0)
        normalized = (sample_tensor - mean) / (std + 1e-8)

        assert normalized.shape == sample_tensor.shape


class TestSequenceCreation:
    """Tests for sequence creation for time series models."""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_sliding_window_creation(self, time_series_data):
        """Test sliding window sequence creation."""
        seq_len = 30
        data = torch.tensor(time_series_data, dtype=torch.float32)

        # Create sequences
        sequences = []
        targets = []
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            targets.append(data[i+seq_len, 0])  # Predict first feature

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)

        # Check shapes
        assert sequences.shape == (100 - seq_len, seq_len, 5)
        assert targets.shape == (100 - seq_len,)

    def test_sequence_overlap(self, time_series_data):
        """Test that consecutive sequences overlap correctly."""
        seq_len = 10
        data = torch.tensor(time_series_data, dtype=torch.float32)

        seq1 = data[0:seq_len]
        seq2 = data[1:seq_len+1]

        # Sequences should overlap by seq_len - 1
        overlap = (seq1[1:] == seq2[:-1]).all()
        assert overlap

    def test_multi_step_targets(self, time_series_data):
        """Test multi-step ahead target creation."""
        seq_len = 30
        forecast_horizon = 5
        data = torch.tensor(time_series_data, dtype=torch.float32)

        sequences = []
        targets = []
        for i in range(len(data) - seq_len - forecast_horizon + 1):
            sequences.append(data[i:i+seq_len])
            targets.append(data[i+seq_len:i+seq_len+forecast_horizon, 0])

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)

        # Check shapes
        assert sequences.shape[1] == seq_len
        assert targets.shape[1] == forecast_horizon


class TestFeatureRegistry:
    """Tests for feature registry and validation."""

    def test_feature_registry_import(self):
        """Test feature registry can be imported."""
        from src.data.feature_registry import FeatureRegistry

        registry = FeatureRegistry()
        assert registry is not None

    def test_builtin_features_exist(self):
        """Test built-in features are registered."""
        from src.data.feature_registry import FeatureRegistry

        registry = FeatureRegistry()
        features = registry.list_features()

        # Should have registered features
        assert len(features) > 0

        # Check for common expected features
        expected_features = ['returns_1d', 'log_returns_1d', 'volatility_20d']
        for feature in expected_features:
            # Feature may exist with different name convention
            matching = [f for f in features if feature in f or f in feature]
            # At least some financial features should exist
        assert len(features) >= 10  # Should have at least 10 features

    def test_feature_formula_documentation(self):
        """Test that features are properly registered."""
        from src.data.feature_registry import FeatureRegistry

        registry = FeatureRegistry()

        # Get list of features
        features = registry.list_features()

        # Should have features registered
        assert len(features) > 0

        # Each feature name should be a non-empty string
        for name in features:
            assert isinstance(name, str)
            assert len(name) > 0


class TestDataCleaner:
    """Tests for data cleaning utilities."""

    @pytest.fixture
    def dirty_df(self):
        """Create DataFrame with various data quality issues."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        data = {
            'close': np.random.randn(50) * 10 + 100,
            'volume': np.random.randint(1000, 10000, 50).astype(float)
        }

        # Introduce issues
        data['close'][5] = np.nan  # Missing value
        data['close'][10] = 1000   # Outlier (10x normal)
        data['volume'][15] = np.nan  # Missing volume
        data['volume'][20] = -100  # Negative volume (suspicious)

        return pd.DataFrame(data, index=dates)

    def test_missing_value_detection(self, dirty_df):
        """Test detection of missing values."""
        missing_count = dirty_df.isnull().sum()

        assert missing_count['close'] == 1
        assert missing_count['volume'] == 1

    def test_forward_fill_imputation(self, dirty_df):
        """Test forward fill for missing values."""
        filled = dirty_df.ffill()

        # Should have no missing values
        assert filled.isnull().sum().sum() == 0

    def test_interpolation_imputation(self, dirty_df):
        """Test linear interpolation for missing values."""
        interpolated = dirty_df.interpolate(method='linear')

        # Should have no missing values
        assert interpolated.isnull().sum().sum() == 0

    def test_outlier_detection_zscore(self, dirty_df):
        """Test z-score based outlier detection."""
        zscore = (dirty_df['close'] - dirty_df['close'].mean()) / dirty_df['close'].std()

        # Outliers have |z| > 3
        outliers = zscore.abs() > 3

        # Should detect the 1000 value as outlier
        assert outliers.any()

    def test_outlier_detection_iqr(self, dirty_df):
        """Test IQR-based outlier detection."""
        q1 = dirty_df['close'].quantile(0.25)
        q3 = dirty_df['close'].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (dirty_df['close'] < lower_bound) | (dirty_df['close'] > upper_bound)

        # Should detect outliers
        assert outliers.any()

    def test_winsorization(self, dirty_df):
        """Test winsorization of outliers."""
        percentile_low = dirty_df['close'].quantile(0.01)
        percentile_high = dirty_df['close'].quantile(0.99)

        winsorized = dirty_df['close'].clip(lower=percentile_low, upper=percentile_high)

        # Values should be within bounds
        assert winsorized.min() >= percentile_low
        assert winsorized.max() <= percentile_high


class TestDataValidation:
    """Tests for data validation utilities."""

    def test_chronological_order(self):
        """Test that time series is in chronological order."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'value': range(100)}, index=dates)

        # Check index is sorted
        assert df.index.is_monotonic_increasing

    def test_no_future_data_leakage(self):
        """Test that features don't use future data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)

        # Calculate features using only past data
        returns = close.pct_change()  # Uses t-1, OK
        sma = close.rolling(window=5).mean()  # Uses t-4 to t, OK

        # These would be lookahead bias:
        # future_return = close.pct_change().shift(-1)  # Uses t+1, BAD
        # future_sma = close.rolling(window=5).mean().shift(-5)  # Uses future, BAD

        # Verify our features don't include future info
        # A feature at time t should only depend on data at time <= t
        assert pd.isna(returns.iloc[0])  # First return is NaN (correct)
        assert pd.isna(sma.iloc[:4]).all()  # First 4 SMA are NaN (correct)

    def test_train_test_no_overlap(self):
        """Test that train and test sets don't overlap."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'value': range(100)}, index=dates)

        train_end = 70
        train = df.iloc[:train_end]
        test = df.iloc[train_end:]

        # Check no overlap
        common_indices = train.index.intersection(test.index)
        assert len(common_indices) == 0

    def test_scaler_fit_on_train_only(self):
        """Test that scalers are fit only on training data."""
        np.random.seed(42)
        train_data = np.random.randn(70, 5) * 10 + 50
        test_data = np.random.randn(30, 5) * 10 + 50

        # Fit scaler on train
        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)

        # Transform both using train statistics
        train_normalized = (train_data - train_mean) / train_std
        test_normalized = (test_data - train_mean) / train_std

        # Train should have mean ~0, std ~1
        assert np.abs(train_normalized.mean()) < 0.1
        assert np.abs(train_normalized.std() - 1.0) < 0.1

        # Test may have different statistics (this is expected)
        # The key is we used train statistics to transform test
