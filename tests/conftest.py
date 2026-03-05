"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@pytest.fixture
def sample_sequences():
    """Create sample sequences for model testing"""
    batch_size = 16
    seq_len = 60
    n_features = 10

    X = torch.randn(batch_size, seq_len, n_features)
    y = torch.randn(batch_size, 1)

    return X, y


@pytest.fixture
def sample_returns():
    """Create sample returns for financial metrics testing"""
    np.random.seed(42)
    n_days = 252  # One year of trading days
    returns = np.random.normal(0.0005, 0.02, n_days)
    return returns


@pytest.fixture
def sample_price_data():
    """Create sample price data"""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n_days)))

    return pd.DataFrame({
        'time': dates,
        'ticker': ['AAPL'] * n_days,
        'open': price * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': price * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': price * (1 - np.random.uniform(0, 0.02, n_days)),
        'close': price,
        'volume': np.random.uniform(1e6, 1e7, n_days)
    })


@pytest.fixture
def mock_config():
    """Create mock configuration object"""
    config = Mock()
    config.data = Mock()
    config.data.sequence_length = 60
    config.data.feature_cols = ['close', 'volume', 'log_return']
    config.data.tickers = ['AAPL', 'MSFT', 'GOOGL']
    config.data.start_date = '2014-01-01'
    config.data.end_date = '2024-01-01'

    config.model = Mock()
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.dropout = 0.1

    config.training = Mock()
    config.training.batch_size = 32
    config.training.epochs = 100
    config.training.learning_rate = 0.001
    config.training.early_stopping_patience = 10
    config.training.random_seed = 42
    config.training.device = 'cpu'

    config.project_root = Path(__file__).parent.parent
    config.checkpoint_dir = config.project_root / 'checkpoints'

    return config


@pytest.fixture
def sample_predictions_targets():
    """Create sample predictions and targets for metrics testing"""
    np.random.seed(42)
    n_samples = 1000

    # Create correlated predictions and targets
    targets = np.random.randn(n_samples)
    noise = np.random.randn(n_samples) * 0.3
    predictions = targets + noise

    return predictions, targets


@pytest.fixture
def sample_signals():
    """Create sample trading signals"""
    n_signals = 50
    dates = pd.date_range(start='2024-01-01', periods=n_signals, freq='D')

    return pd.DataFrame({
        'timestamp': dates,
        'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_signals),
        'action': np.random.choice(['BUY', 'SELL', 'HOLD'], n_signals),
        'confidence': np.random.uniform(0.5, 1.0, n_signals),
        'predicted_price': np.random.uniform(100, 200, n_signals),
        'current_price': np.random.uniform(100, 200, n_signals)
    })


# Markers for slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


# Skip GPU tests if not available
def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
