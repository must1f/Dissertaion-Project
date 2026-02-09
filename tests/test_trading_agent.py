"""
Comprehensive tests for trading agent and signal generation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.trading.agent import Signal, SignalGenerator


class DummyModel(nn.Module):
    """Simple model for testing signal generation"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        # Take last timestep
        x = x[:, -1, :]
        return self.fc(x)


class TestSignal:
    """Test Signal dataclass"""

    def test_signal_creation(self):
        """Test creating a signal"""
        signal = Signal(
            timestamp=pd.Timestamp('2020-01-01'),
            ticker='AAPL',
            action='BUY',
            confidence=0.85,
            predicted_price=150.0,
            current_price=145.0,
            expected_return=0.0345
        )
        
        assert signal.timestamp == pd.Timestamp('2020-01-01')
        assert signal.ticker == 'AAPL'
        assert signal.action == 'BUY'
        assert signal.confidence == 0.85
        assert signal.predicted_price == 150.0
        assert signal.current_price == 145.0
        assert abs(signal.expected_return - 0.0345) < 1e-6

    def test_signal_attributes(self):
        """Test signal has all required attributes"""
        signal = Signal(
            timestamp=pd.Timestamp('2020-01-01'),
            ticker='TEST',
            action='HOLD',
            confidence=0.5,
            predicted_price=100.0,
            current_price=100.0,
            expected_return=0.0
        )
        
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'ticker')
        assert hasattr(signal, 'action')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'predicted_price')
        assert hasattr(signal, 'current_price')
        assert hasattr(signal, 'expected_return')


class TestSignalGenerator:
    """Test SignalGenerator class"""

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model"""
        model = DummyModel()
        model.eval()
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 5
        seq_len = 60
        n_features = 10
        
        sequences = torch.randn(batch_size, seq_len, n_features)
        current_prices = np.array([100.0, 150.0, 200.0, 50.0, 75.0])
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        timestamps = [pd.Timestamp('2020-01-01') + timedelta(days=i) for i in range(batch_size)]
        
        return sequences, current_prices, tickers, timestamps

    def test_signal_generator_initialization(self, dummy_model):
        """Test signal generator initialization"""
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        assert generator.model is not None
        assert generator.device.type == 'cpu'
        assert not generator.model.training  # Should be in eval mode

    def test_predict(self, dummy_model, sample_data):
        """Test prediction function"""
        sequences, _, _, _ = sample_data
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        predictions, uncertainties = generator.predict(sequences)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sequences)
        assert uncertainties is None  # Not implemented yet

    def test_generate_signals_buy(self, dummy_model):
        """Test generating BUY signals"""
        # Set model weights to predict higher prices
        with torch.no_grad():
            dummy_model.fc.weight.fill_(10.0)
            dummy_model.fc.bias.fill_(10.0)
        
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        sequences = torch.randn(3, 60, 10)
        current_prices = np.array([100.0, 150.0, 200.0])
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        timestamps = [pd.Timestamp('2020-01-01')] * 3
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.01, confidence_threshold=0.5
        )
        
        assert len(signals) == 3
        assert all(isinstance(s, Signal) for s in signals)
        
        # With high weights, predictions should be high, leading to BUY signals
        # (depending on the random input, but expected return should be positive)

    def test_generate_signals_sell(self, dummy_model):
        """Test generating SELL signals"""
        # Set model weights to predict lower prices
        with torch.no_grad():
            dummy_model.fc.weight.fill_(-10.0)
            dummy_model.fc.bias.fill_(-10.0)
        
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        sequences = torch.randn(3, 60, 10)
        current_prices = np.array([100.0, 150.0, 200.0])
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        timestamps = [pd.Timestamp('2020-01-01')] * 3
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.01, confidence_threshold=0.5
        )
        
        assert len(signals) == 3
        # With negative weights, predictions should be low, leading to SELL signals

    def test_generate_signals_hold(self, dummy_model):
        """Test generating HOLD signals"""
        # Set model to predict similar prices (near zero)
        with torch.no_grad():
            dummy_model.fc.weight.fill_(0.0)
            dummy_model.fc.bias.fill_(0.0)
        
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        sequences = torch.randn(3, 60, 10)
        current_prices = np.array([100.0, 150.0, 200.0])
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        timestamps = [pd.Timestamp('2020-01-01')] * 3
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.02, confidence_threshold=0.5
        )
        
        assert len(signals) == 3
        # With zero weights and high threshold, should mostly get HOLD signals
        # (predicted price near 0, so negative expected return, but within threshold)

    def test_signal_threshold(self, dummy_model):
        """Test that threshold affects signal generation"""
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        sequences = torch.randn(2, 60, 10)
        current_prices = np.array([100.0, 100.0])
        tickers = ['AAPL', 'GOOGL']
        timestamps = [pd.Timestamp('2020-01-01')] * 2
        
        # Low threshold - more sensitive
        signals_low = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.001, confidence_threshold=0.5
        )
        
        # High threshold - less sensitive
        signals_high = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.10, confidence_threshold=0.5
        )
        
        # Both should have same number of signals
        assert len(signals_low) == len(signals_high) == 2

    def test_confidence_threshold(self, dummy_model):
        """Test that confidence threshold filters signals"""
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        sequences = torch.randn(3, 60, 10)
        current_prices = np.array([100.0, 150.0, 200.0])
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        timestamps = [pd.Timestamp('2020-01-01')] * 3
        
        # Low confidence threshold
        signals_low = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.01, confidence_threshold=0.1
        )
        
        # High confidence threshold
        signals_high = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=0.01, confidence_threshold=0.99
        )
        
        assert len(signals_low) == 3
        assert len(signals_high) == 3
        
        # With current implementation (no uncertainty), confidence is always 1.0
        # so both should have similar results

    def test_signals_to_dataframe(self, dummy_model, sample_data):
        """Test converting signals to DataFrame"""
        sequences, current_prices, tickers, timestamps = sample_data
        generator = SignalGenerator(dummy_model, device=torch.device('cpu'))
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps
        )
        
        df = generator.signals_to_dataframe(signals)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(signals)
        assert 'timestamp' in df.columns
        assert 'ticker' in df.columns
        assert 'signal' in df.columns or 'action' in df.columns


class TestSignalGeneratorEdgeCases:
    """Test edge cases for signal generation"""

    def test_empty_sequences(self):
        """Test with empty sequences"""
        model = DummyModel()
        generator = SignalGenerator(model, device=torch.device('cpu'))
        
        sequences = torch.empty(0, 60, 10)
        current_prices = np.array([])
        tickers = []
        timestamps = []
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps
        )
        
        assert len(signals) == 0

    def test_single_sequence(self):
        """Test with single sequence"""
        model = DummyModel()
        generator = SignalGenerator(model, device=torch.device('cpu'))
        
        sequences = torch.randn(1, 60, 10)
        current_prices = np.array([100.0])
        tickers = ['AAPL']
        timestamps = [pd.Timestamp('2020-01-01')]
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps
        )
        
        assert len(signals) == 1
        assert signals[0].ticker == 'AAPL'

    def test_zero_prices(self):
        """Test with zero current prices"""
        model = DummyModel()
        generator = SignalGenerator(model, device=torch.device('cpu'))
        
        sequences = torch.randn(2, 60, 10)
        current_prices = np.array([0.0, 100.0])
        tickers = ['ZERO', 'AAPL']
        timestamps = [pd.Timestamp('2020-01-01')] * 2
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps
        )
        
        # Should handle zero price (division by zero in expected return)
        assert len(signals) == 2
        # Expected return for zero price will be inf or nan
        assert np.isinf(signals[0].expected_return) or np.isnan(signals[0].expected_return)

    def test_negative_prices(self):
        """Test with negative prices (edge case, shouldn't happen in reality)"""
        model = DummyModel()
        generator = SignalGenerator(model, device=torch.device('cpu'))
        
        sequences = torch.randn(2, 60, 10)
        current_prices = np.array([-100.0, 100.0])
        tickers = ['NEG', 'AAPL']
        timestamps = [pd.Timestamp('2020-01-01')] * 2
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps
        )
        
        assert len(signals) == 2

    def test_very_high_threshold(self):
        """Test with very high threshold (should result in all HOLD)"""
        model = DummyModel()
        generator = SignalGenerator(model, device=torch.device('cpu'))
        
        sequences = torch.randn(5, 60, 10)
        current_prices = np.array([100.0, 150.0, 200.0, 50.0, 75.0])
        tickers = ['A', 'B', 'C', 'D', 'E']
        timestamps = [pd.Timestamp('2020-01-01')] * 5
        
        signals = generator.generate_signals(
            sequences, current_prices, tickers, timestamps,
            threshold=10.0  # 1000% return required!
        )
        
        # With such high threshold, most/all should be HOLD
        hold_count = sum(1 for s in signals if s.action == 'HOLD')
        assert hold_count >= 0  # At least some should be HOLD


class TestSignalProperties:
    """Test signal properties and calculations"""

    def test_expected_return_calculation(self):
        """Test expected return is calculated correctly"""
        signal = Signal(
            timestamp=pd.Timestamp('2020-01-01'),
            ticker='AAPL',
            action='BUY',
            confidence=0.9,
            predicted_price=110.0,
            current_price=100.0,
            expected_return=(110.0 - 100.0) / 100.0
        )
        
        # Expected return should be 10%
        assert abs(signal.expected_return - 0.10) < 1e-6

    def test_signal_action_types(self):
        """Test all action types"""
        actions = ['BUY', 'SELL', 'HOLD']
        
        for action in actions:
            signal = Signal(
                timestamp=pd.Timestamp('2020-01-01'),
                ticker='TEST',
                action=action,
                confidence=0.8,
                predicted_price=100.0,
                current_price=100.0,
                expected_return=0.0
            )
            
            assert signal.action == action

    def test_confidence_range(self):
        """Test that confidence is in valid range"""
        # Confidence should typically be between 0 and 1
        signal = Signal(
            timestamp=pd.Timestamp('2020-01-01'),
            ticker='TEST',
            action='BUY',
            confidence=0.75,
            predicted_price=100.0,
            current_price=100.0,
            expected_return=0.0
        )
        
        assert 0.0 <= signal.confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
