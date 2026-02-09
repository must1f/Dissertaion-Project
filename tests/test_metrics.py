"""
Comprehensive tests for evaluation metrics
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test MetricsCalculator class"""

    def test_rmse(self):
        """Test Root Mean Squared Error"""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        
        # Manual calculation
        expected = np.sqrt(np.mean([(3-2.5)**2, (-0.5-0)**2, (2-2)**2, (7-8)**2]))
        np.testing.assert_allclose(rmse, expected, rtol=1e-5)
        
        assert rmse > 0

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        np.testing.assert_allclose(rmse, 0.0, atol=1e-10)

    def test_mae(self):
        """Test Mean Absolute Error"""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        mae = MetricsCalculator.mae(y_true, y_pred)
        
        # Manual calculation
        expected = np.mean([abs(3-2.5), abs(-0.5-0), abs(2-2), abs(7-8)])
        np.testing.assert_allclose(mae, expected, rtol=1e-5)

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mae = MetricsCalculator.mae(y_true, y_pred)
        np.testing.assert_allclose(mae, 0.0, atol=1e-10)

    def test_mape(self):
        """Test Mean Absolute Percentage Error"""
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([110, 190, 310, 380])
        
        mape = MetricsCalculator.mape(y_true, y_pred)
        
        # Manual calculation
        expected = np.mean([abs((100-110)/100), abs((200-190)/200), 
                           abs((300-310)/300), abs((400-380)/400)]) * 100
        np.testing.assert_allclose(mape, expected, rtol=1e-5)

    def test_mape_with_epsilon(self):
        """Test MAPE with near-zero values (uses epsilon)"""
        y_true = np.array([0.0, 0.001, 0.01, 1.0])
        y_pred = np.array([0.001, 0.002, 0.02, 1.1])
        
        mape = MetricsCalculator.mape(y_true, y_pred, epsilon=1e-10)
        
        # Should not divide by zero
        assert not np.isnan(mape)
        assert not np.isinf(mape)

    def test_r2_score(self):
        """Test R-squared score"""
        y_true = np.array([3, -0.5, 2, 7, 4.2])
        y_pred = np.array([2.5, 0.0, 2, 8, 4.1])
        
        r2 = MetricsCalculator.r2(y_true, y_pred)
        
        # R² should be between -inf and 1
        assert r2 <= 1.0

    def test_r2_perfect_prediction(self):
        """Test R² with perfect predictions"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = MetricsCalculator.r2(y_true, y_pred)
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)

    def test_r2_constant_prediction(self):
        """Test R² with constant predictions (should be 0 if pred = mean)"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # Mean of y_true
        
        r2 = MetricsCalculator.r2(y_true, y_pred)
        np.testing.assert_allclose(r2, 0.0, atol=1e-10)


class TestDirectionalAccuracy:
    """Test directional accuracy metric"""

    def test_directional_accuracy_with_prices(self):
        """Test directional accuracy with price data"""
        # Prices going up, then down
        y_true = np.array([100, 105, 110, 108, 106, 109])
        # Predictions mostly correct direction
        y_pred = np.array([100, 104, 112, 107, 105, 110])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=False
        )
        
        # Check it's a percentage
        assert 0 <= accuracy <= 100
        
        # Should be high since we mostly got direction right
        assert accuracy > 50

    def test_directional_accuracy_with_returns(self):
        """Test directional accuracy with return data"""
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.015, -0.018, 0.025, -0.008, 0.022])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=True
        )
        
        # All signs match, so should be 100%
        assert accuracy == 100.0

    def test_directional_accuracy_wrong_direction(self):
        """Test when all directions are wrong"""
        y_true = np.array([0.01, 0.02, 0.03, 0.04])
        y_pred = np.array([-0.01, -0.02, -0.03, -0.04])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=True
        )
        
        # All wrong, should be 0%
        assert accuracy == 0.0

    def test_directional_accuracy_with_threshold(self):
        """Test directional accuracy with threshold"""
        # Very small changes should be filtered out
        y_true = np.array([0.0001, -0.0001, 0.01, -0.01])
        y_pred = np.array([-0.0001, 0.0001, 0.015, -0.015])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=True, threshold=0.001
        )
        
        # Only the last 2 movements are significant, both correct
        assert accuracy == 100.0

    def test_directional_accuracy_no_significant_movements(self):
        """Test when there are no significant movements"""
        y_true = np.array([1e-10, -1e-10, 1e-11])
        y_pred = np.array([-1e-10, 1e-10, -1e-11])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=True, threshold=1e-8
        )
        
        # No significant movements, should return baseline (50%)
        assert accuracy == 50.0

    def test_directional_accuracy_insufficient_data(self):
        """Test with insufficient data"""
        y_true = np.array([100])
        y_pred = np.array([105])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=False
        )
        
        # Can't calculate direction with single point
        assert accuracy == 0.0

    def test_directional_accuracy_mixed_results(self):
        """Test with mixed correct/incorrect directions"""
        # Prices: up, down, up, down, up
        y_true = np.array([100, 105, 100, 105, 100, 105])
        # Predictions: up(✓), down(✓), down(✗), up(✗), up(✓) = 3/5 = 60%
        y_pred = np.array([100, 106, 105, 104, 105, 110])
        
        accuracy = MetricsCalculator.directional_accuracy(
            y_true, y_pred, are_returns=False
        )
        
        assert 50 < accuracy < 80  # Should be around 60%


class TestSharpeRatio:
    """Test Sharpe ratio calculation"""

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns"""
        # Consistent positive returns
        returns = np.array([0.01, 0.015, 0.012, 0.018, 0.014])
        
        sharpe = MetricsCalculator.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should be positive (returns > risk-free rate)
        assert sharpe > 0

    def test_sharpe_ratio_negative_returns(self):
        """Test Sharpe ratio with negative returns"""
        # Consistent negative returns
        returns = np.array([-0.01, -0.015, -0.012, -0.018, -0.014])
        
        sharpe = MetricsCalculator.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should be negative
        assert sharpe < 0

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation"""
        # All returns are the same
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = MetricsCalculator.sharpe_ratio(
            returns, risk_free_rate=0.0, periods_per_year=252
        )
        
        # Should handle zero std gracefully (return 0 or inf)
        assert not np.isnan(sharpe)

    def test_sharpe_ratio_different_frequencies(self):
        """Test Sharpe ratio with different time frequencies"""
        daily_returns = np.random.randn(252) * 0.01 + 0.001  # Daily
        
        sharpe_daily = MetricsCalculator.sharpe_ratio(
            daily_returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should work without errors
        assert isinstance(sharpe_daily, (int, float))
        assert not np.isnan(sharpe_daily)


class TestMetricsEdgeCases:
    """Test edge cases for all metrics"""

    def test_empty_arrays(self):
        """Test with empty arrays"""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Most metrics should handle empty arrays gracefully or raise appropriate errors
        with pytest.raises((ValueError, IndexError)):
            MetricsCalculator.rmse(y_true, y_pred)

    def test_nan_values(self):
        """Test with NaN values"""
        y_true = np.array([1, 2, np.nan, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Should propagate NaN
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        assert np.isnan(rmse)

    def test_inf_values(self):
        """Test with infinite values"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, np.inf, 4, 5])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        assert np.isinf(rmse)

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            MetricsCalculator.rmse(y_true, y_pred)

    def test_large_values(self):
        """Test with very large values"""
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1.1e10, 2.1e10, 3.1e10])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        
        # Should handle large values without overflow
        assert not np.isinf(rmse)
        assert not np.isnan(rmse)
        assert rmse > 0

    def test_small_values(self):
        """Test with very small values"""
        y_true = np.array([1e-10, 2e-10, 3e-10])
        y_pred = np.array([1.1e-10, 2.1e-10, 3.1e-10])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        
        # Should handle small values
        assert rmse > 0
        assert not np.isnan(rmse)

    def test_negative_values(self):
        """Test all metrics work with negative values"""
        y_true = np.array([-10, -5, -3, -8, -6])
        y_pred = np.array([-9, -6, -2, -7, -5])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        mae = MetricsCalculator.mae(y_true, y_pred)
        r2 = MetricsCalculator.r2(y_true, y_pred)
        
        assert rmse > 0
        assert mae > 0
        assert isinstance(r2, (int, float))

    def test_single_value(self):
        """Test with single value arrays"""
        y_true = np.array([5.0])
        y_pred = np.array([5.5])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        mae = MetricsCalculator.mae(y_true, y_pred)
        
        assert rmse == 0.5
        assert mae == 0.5

    def test_all_zeros(self):
        """Test with all zero values"""
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        mae = MetricsCalculator.mae(y_true, y_pred)
        r2 = MetricsCalculator.r2(y_true, y_pred)
        
        assert rmse == 0.0
        assert mae == 0.0
        # R² is undefined for constant y_true, but sklearn returns 0
        assert np.isnan(r2) or r2 == 0.0


class TestMetricsConsistency:
    """Test consistency between different metrics"""

    def test_rmse_vs_mae(self):
        """RMSE should be >= MAE due to squaring"""
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        mae = MetricsCalculator.mae(y_true, y_pred)
        
        # RMSE >= MAE always holds
        assert rmse >= mae

    def test_perfect_prediction_all_metrics(self):
        """All metrics should show perfect score with perfect prediction"""
        y_true = np.random.randn(100)
        y_pred = y_true.copy()
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        mae = MetricsCalculator.mae(y_true, y_pred)
        mape = MetricsCalculator.mape(y_true + 100, y_pred + 100)  # Offset to avoid division issues
        r2 = MetricsCalculator.r2(y_true, y_pred)
        
        np.testing.assert_allclose(rmse, 0.0, atol=1e-10)
        np.testing.assert_allclose(mae, 0.0, atol=1e-10)
        np.testing.assert_allclose(mape, 0.0, atol=1e-10)
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
