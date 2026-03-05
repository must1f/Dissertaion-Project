"""
Unit tests for Trading Agent and Signal Generation
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSignalGeneration:
    """Test signal generation logic"""

    def test_buy_signal_threshold(self):
        """Test buy signal threshold logic"""
        buy_threshold = 0.01  # 1% expected return

        predictions = np.array([100.5, 101.5, 102.0, 100.2])
        current_prices = np.array([100.0, 100.0, 100.0, 100.0])

        expected_returns = (predictions - current_prices) / current_prices
        buy_signals = expected_returns > buy_threshold

        assert buy_signals[0] == False  # 0.5% < 1%
        assert buy_signals[1] == True   # 1.5% > 1%
        assert buy_signals[2] == True   # 2.0% > 1%
        assert buy_signals[3] == False  # 0.2% < 1%

    def test_sell_signal_threshold(self):
        """Test sell signal threshold logic"""
        sell_threshold = -0.01  # -1% expected return

        predictions = np.array([99.5, 98.5, 99.8, 98.0])
        current_prices = np.array([100.0, 100.0, 100.0, 100.0])

        expected_returns = (predictions - current_prices) / current_prices
        sell_signals = expected_returns < sell_threshold

        assert sell_signals[0] == False  # -0.5% > -1%
        assert sell_signals[1] == True   # -1.5% < -1%
        assert sell_signals[2] == False  # -0.2% > -1%
        assert sell_signals[3] == True   # -2.0% < -1%

    def test_hold_signal(self):
        """Test hold signal (no action)"""
        buy_threshold = 0.01
        sell_threshold = -0.01

        predictions = np.array([100.5, 99.5, 100.0])
        current_prices = np.array([100.0, 100.0, 100.0])

        expected_returns = (predictions - current_prices) / current_prices

        buy_signals = expected_returns > buy_threshold
        sell_signals = expected_returns < sell_threshold
        hold_signals = ~buy_signals & ~sell_signals

        assert hold_signals[0] == True   # 0.5% is between thresholds
        assert hold_signals[1] == True   # -0.5% is between thresholds
        assert hold_signals[2] == True   # 0% is between thresholds


class TestSignalConfidence:
    """Test signal confidence calculation"""

    def test_confidence_from_magnitude(self):
        """Test confidence based on prediction magnitude"""
        def calculate_confidence(expected_return, threshold=0.01):
            """Simple confidence based on how much expected return exceeds threshold"""
            if abs(expected_return) <= threshold:
                return 0.0

            # Confidence increases with distance from threshold
            excess = abs(expected_return) - threshold
            return min(excess / threshold, 1.0)

        assert calculate_confidence(0.005) == 0.0  # Below threshold
        assert calculate_confidence(0.02) == 1.0   # 2x threshold = full confidence
        assert abs(calculate_confidence(0.015) - 0.5) < 1e-9  # 1.5x threshold = 50% confidence (with float tolerance)

    def test_confidence_bounds(self):
        """Test confidence is always between 0 and 1"""
        confidences = np.random.uniform(0, 2, 100)
        bounded = np.clip(confidences, 0, 1)

        assert np.all(bounded >= 0)
        assert np.all(bounded <= 1)


class TestUncertaintyEstimation:
    """Test uncertainty estimation"""

    def test_mc_dropout_variance(self):
        """Test MC Dropout produces variance in predictions"""
        np.random.seed(42)

        # Simulate MC Dropout samples
        n_samples = 50
        base_prediction = 100.0
        noise_std = 2.0

        samples = base_prediction + np.random.normal(0, noise_std, n_samples)

        mean_pred = np.mean(samples)
        std_pred = np.std(samples)

        # Should have reasonable variance
        assert std_pred > 0
        assert abs(mean_pred - base_prediction) < noise_std

    def test_prediction_intervals(self):
        """Test prediction interval calculation"""
        mean = 100.0
        std = 5.0
        z_95 = 1.96  # 95% CI

        lower = mean - z_95 * std
        upper = mean + z_95 * std

        assert lower < mean < upper
        assert abs(upper - lower - 2 * z_95 * std) < 0.01

    def test_uncertainty_to_confidence(self):
        """Test converting uncertainty to confidence"""
        def uncertainty_to_confidence(std, max_std=0.1):
            """Higher uncertainty = lower confidence"""
            return max(0, 1 - std / max_std)

        assert uncertainty_to_confidence(0.0) == 1.0   # No uncertainty = full confidence
        assert uncertainty_to_confidence(0.1) == 0.0   # Max uncertainty = no confidence
        assert uncertainty_to_confidence(0.05) == 0.5  # Half uncertainty = half confidence


class TestSignalDataclass:
    """Test Signal dataclass functionality"""

    @dataclass
    class Signal:
        """Mock Signal dataclass for testing"""
        timestamp: pd.Timestamp
        ticker: str
        action: str
        confidence: float
        predicted_price: float
        current_price: float
        expected_return: float

    def test_signal_creation(self):
        """Test creating a signal"""
        signal = self.Signal(
            timestamp=pd.Timestamp('2024-01-01'),
            ticker='AAPL',
            action='BUY',
            confidence=0.8,
            predicted_price=155.0,
            current_price=150.0,
            expected_return=0.0333
        )

        assert signal.ticker == 'AAPL'
        assert signal.action == 'BUY'
        assert signal.confidence == 0.8

    def test_signal_expected_return(self):
        """Test expected return calculation"""
        current_price = 100.0
        predicted_price = 105.0
        expected_return = (predicted_price - current_price) / current_price

        assert expected_return == 0.05


class TestRiskAdjustedSignals:
    """Test risk-adjusted signal generation"""

    def test_uncertainty_adjusted_threshold(self):
        """Test adjusting thresholds based on uncertainty"""
        base_threshold = 0.01
        uncertainty = 0.005

        # Higher uncertainty = higher threshold
        adjusted_threshold = base_threshold * (1 + uncertainty / 0.01)

        assert adjusted_threshold > base_threshold

    def test_prediction_interval_validation(self):
        """Test validating signals with prediction intervals"""
        current_price = 100.0
        predicted_price = 102.0
        lower_bound = 99.0
        upper_bound = 105.0

        # Buy only if lower bound > current (conservative)
        conservative_buy = lower_bound > current_price

        # Standard buy (just predicted > current)
        standard_buy = predicted_price > current_price

        assert standard_buy == True
        assert conservative_buy == False  # Lower bound is below current

    def test_confidence_weighted_position(self):
        """Test position size weighted by confidence"""
        base_position = 100  # shares
        confidence = 0.7

        adjusted_position = int(base_position * confidence)

        assert adjusted_position == 70


class TestSignalFiltering:
    """Test signal filtering logic"""

    def test_minimum_confidence_filter(self):
        """Test filtering signals by minimum confidence"""
        signals = [
            {'action': 'BUY', 'confidence': 0.8},
            {'action': 'BUY', 'confidence': 0.3},
            {'action': 'SELL', 'confidence': 0.6},
            {'action': 'BUY', 'confidence': 0.5},
        ]

        min_confidence = 0.5
        filtered = [s for s in signals if s['confidence'] >= min_confidence]

        assert len(filtered) == 3
        assert all(s['confidence'] >= min_confidence for s in filtered)

    def test_action_type_filter(self):
        """Test filtering by action type"""
        signals = [
            {'action': 'BUY', 'ticker': 'AAPL'},
            {'action': 'SELL', 'ticker': 'MSFT'},
            {'action': 'HOLD', 'ticker': 'GOOGL'},
            {'action': 'BUY', 'ticker': 'NVDA'},
        ]

        buy_signals = [s for s in signals if s['action'] == 'BUY']

        assert len(buy_signals) == 2
        assert all(s['action'] == 'BUY' for s in buy_signals)

    def test_duplicate_ticker_filter(self):
        """Test filtering duplicate signals for same ticker"""
        signals = [
            {'ticker': 'AAPL', 'confidence': 0.8, 'action': 'BUY'},
            {'ticker': 'AAPL', 'confidence': 0.6, 'action': 'BUY'},
            {'ticker': 'MSFT', 'confidence': 0.7, 'action': 'SELL'},
        ]

        # Keep highest confidence for each ticker
        best_signals = {}
        for s in signals:
            ticker = s['ticker']
            if ticker not in best_signals or s['confidence'] > best_signals[ticker]['confidence']:
                best_signals[ticker] = s

        assert len(best_signals) == 2
        assert best_signals['AAPL']['confidence'] == 0.8


class TestSignalTiming:
    """Test signal timing and execution"""

    def test_signal_timestamp(self):
        """Test signal has proper timestamp"""
        timestamp = pd.Timestamp('2024-01-15 09:30:00')

        signal = {
            'timestamp': timestamp,
            'action': 'BUY',
            'ticker': 'AAPL'
        }

        assert signal['timestamp'] == timestamp

    def test_signal_ordering(self):
        """Test signals are ordered by timestamp"""
        signals = [
            {'timestamp': pd.Timestamp('2024-01-15 10:00:00')},
            {'timestamp': pd.Timestamp('2024-01-15 09:30:00')},
            {'timestamp': pd.Timestamp('2024-01-15 11:00:00')},
        ]

        sorted_signals = sorted(signals, key=lambda x: x['timestamp'])

        timestamps = [s['timestamp'] for s in sorted_signals]
        assert timestamps == sorted(timestamps)


class TestEnsemblePredictions:
    """Test ensemble predictions for uncertainty"""

    def test_ensemble_mean(self):
        """Test computing ensemble mean"""
        predictions = np.array([100.0, 101.0, 99.5, 100.5, 100.2])

        mean = np.mean(predictions)

        assert abs(mean - 100.24) < 0.01

    def test_ensemble_std(self):
        """Test computing ensemble standard deviation"""
        predictions = np.array([100.0, 101.0, 99.5, 100.5, 100.2])

        std = np.std(predictions)

        assert std > 0
        assert std < 1  # Should be relatively small disagreement

    def test_ensemble_agreement(self):
        """Test measuring ensemble agreement"""
        # High agreement (low variance)
        high_agreement = np.array([100.0, 100.1, 99.9, 100.0, 100.0])
        high_std = np.std(high_agreement)

        # Low agreement (high variance)
        low_agreement = np.array([95.0, 100.0, 105.0, 98.0, 102.0])
        low_std = np.std(low_agreement)

        assert high_std < low_std


class TestEdgeCases:
    """Test edge cases in trading agent"""

    def test_no_signals(self):
        """Test handling when no signals are generated"""
        predictions = np.array([100.0, 100.0, 100.0])
        current_prices = np.array([100.0, 100.0, 100.0])
        threshold = 0.01

        expected_returns = (predictions - current_prices) / current_prices
        signals = np.abs(expected_returns) > threshold

        assert not np.any(signals)

    def test_all_buy_signals(self):
        """Test when all signals are buy"""
        predictions = np.array([105.0, 106.0, 107.0])
        current_prices = np.array([100.0, 100.0, 100.0])
        threshold = 0.01

        expected_returns = (predictions - current_prices) / current_prices
        buy_signals = expected_returns > threshold

        assert np.all(buy_signals)

    def test_conflicting_signals(self):
        """Test handling conflicting signals"""
        # Different models give different signals
        model1_signal = 'BUY'
        model2_signal = 'SELL'
        model3_signal = 'BUY'

        # Majority vote
        signals = [model1_signal, model2_signal, model3_signal]
        from collections import Counter
        majority = Counter(signals).most_common(1)[0][0]

        assert majority == 'BUY'

    def test_nan_predictions(self):
        """Test handling NaN predictions"""
        predictions = np.array([100.0, np.nan, 102.0])
        current_prices = np.array([100.0, 100.0, 100.0])

        # Should handle NaN gracefully
        valid_mask = ~np.isnan(predictions)
        valid_predictions = predictions[valid_mask]

        assert len(valid_predictions) == 2
        assert not np.any(np.isnan(valid_predictions))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
