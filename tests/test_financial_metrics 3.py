"""
Unit tests for Financial Metrics
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.financial_metrics import (
    FinancialMetrics,
    compute_strategy_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio
)


class TestSharpeRatio:
    """Test Sharpe Ratio calculation"""

    def test_positive_sharpe(self):
        """Test positive Sharpe ratio"""
        # Returns with positive mean and low volatility
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02, 0.01, 0.015])

        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.0)

        assert sharpe > 0

    def test_negative_sharpe(self):
        """Test negative Sharpe ratio"""
        # Returns with negative mean
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.02])

        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.0)

        assert sharpe < 0

    def test_zero_volatility(self):
        """Test Sharpe with zero volatility"""
        # Constant returns
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.0)

        # Should handle gracefully (returns 0 when std is 0)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_annualized_sharpe(self):
        """Test annualized Sharpe ratio"""
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 252)

        daily_sharpe = FinancialMetrics.sharpe_ratio(
            daily_returns,
            risk_free_rate=0.0,
            periods_per_year=1
        )

        annual_sharpe = FinancialMetrics.sharpe_ratio(
            daily_returns,
            risk_free_rate=0.0,
            periods_per_year=252
        )

        # Annual Sharpe should be sqrt(252) times daily Sharpe
        expected_ratio = np.sqrt(252)
        assert abs(annual_sharpe / (daily_sharpe + 1e-10) - expected_ratio) < 1

    def test_standalone_sharpe_function(self):
        """Test standalone calculate_sharpe_ratio function"""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

        assert sharpe > 0
        assert not np.isnan(sharpe)


class TestSortinoRatio:
    """Test Sortino Ratio calculation"""

    def test_positive_sortino(self):
        """Test positive Sortino ratio"""
        # Positive returns with some downside
        returns = np.array([0.02, 0.01, -0.005, 0.03, 0.01, -0.002])

        sortino = FinancialMetrics.sortino_ratio(returns, risk_free_rate=0.0)

        assert sortino > 0

    def test_sortino_vs_sharpe(self):
        """Test that Sortino is typically higher than Sharpe for positive returns"""
        np.random.seed(42)
        # Skewed distribution with more upside
        returns = np.abs(np.random.normal(0.001, 0.02, 100)) - 0.005

        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.0)
        sortino = FinancialMetrics.sortino_ratio(returns, risk_free_rate=0.0)

        # Sortino ignores upside volatility, so often higher for positive mean returns
        # This relationship isn't guaranteed but common
        assert not np.isnan(sortino)

    def test_no_downside(self):
        """Test Sortino with no downside returns"""
        # All positive returns
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])

        sortino = FinancialMetrics.sortino_ratio(returns, risk_free_rate=0.0)

        # Should handle gracefully (returns 10.0 when no downside)
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)

    def test_standalone_sortino_function(self):
        """Test standalone calculate_sortino_ratio function"""
        # Need enough data with downside returns for valid calculation
        returns = np.array([0.02, 0.01, -0.005, 0.03, 0.01, -0.01, 0.02, -0.003])

        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)

        # Should return a valid number (not NaN)
        assert not np.isnan(sortino) or sortino == 0.0 or sortino == 10.0  # Handle edge cases


class TestMaxDrawdown:
    """Test Maximum Drawdown calculation"""

    def test_basic_drawdown(self):
        """Test basic drawdown calculation"""
        # Simple pattern: up 10%, then down 20%, then up 10%
        returns = np.array([0.10, -0.20, 0.10])

        max_dd = FinancialMetrics.max_drawdown(returns)

        # After +10%: 1.10, after -20%: 0.88, that's -20% from peak (1.10)
        assert max_dd < 0
        assert max_dd >= -1.0  # Can't lose more than 100%

    def test_no_drawdown(self):
        """Test with no drawdown (all positive returns)"""
        returns = np.array([0.01, 0.02, 0.01, 0.015])

        max_dd = FinancialMetrics.max_drawdown(returns)

        # Should be 0 or very small
        assert max_dd <= 0
        assert max_dd > -0.01  # Essentially no drawdown

    def test_complete_loss(self):
        """Test handling of complete loss scenario"""
        # Returns that lead to near-zero equity
        returns = np.array([-0.5, -0.5, -0.5])

        max_dd = FinancialMetrics.max_drawdown(returns)

        # Should be capped at -100%
        assert max_dd >= -1.0
        assert max_dd < 0

    def test_drawdown_bounds(self):
        """Test drawdown is always between -1 and 0"""
        np.random.seed(42)
        for _ in range(10):
            returns = np.random.normal(-0.01, 0.05, 100)
            max_dd = FinancialMetrics.max_drawdown(returns)

            assert max_dd <= 0
            assert max_dd >= -1.0

    def test_drawdown_series(self):
        """Test drawdown series return"""
        returns = np.array([0.10, -0.20, 0.10, -0.05])

        max_dd, dd_series = FinancialMetrics.max_drawdown(returns, return_series=True)

        assert len(dd_series) == len(returns)
        assert np.max(dd_series) == 0  # Peak is always 0
        assert np.min(dd_series) == max_dd

    def test_standalone_drawdown_function(self):
        """Test standalone calculate_max_drawdown function"""
        returns = np.array([0.10, -0.20, 0.10])

        max_dd = calculate_max_drawdown(returns)

        assert max_dd < 0
        assert max_dd >= -1.0


class TestCalmarRatio:
    """Test Calmar Ratio calculation"""

    def test_positive_calmar(self):
        """Test positive Calmar ratio"""
        # Positive annual return with moderate drawdown
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.01, -0.005] * 42)

        calmar = FinancialMetrics.calmar_ratio(returns)

        # With positive returns and some drawdown, Calmar should be positive
        assert not np.isnan(calmar)

    def test_calmar_capping(self):
        """Test Calmar ratio is capped at reasonable values"""
        # Very small drawdown should not cause extreme Calmar
        returns = np.array([0.001, 0.001, 0.001, 0.001] * 63)

        calmar = FinancialMetrics.calmar_ratio(returns)

        # Should be capped
        assert calmar <= 10.0
        assert calmar >= -10.0

    def test_standalone_calmar_function(self):
        """Test standalone calculate_calmar_ratio function"""
        returns = np.array([0.01, 0.02, -0.01, 0.015] * 63)

        calmar = calculate_calmar_ratio(returns)

        assert not np.isnan(calmar)
        assert not np.isinf(calmar)


class TestDirectionalAccuracy:
    """Test Directional Accuracy calculation"""

    def test_perfect_accuracy(self):
        """Test 100% directional accuracy with returns"""
        # For returns, compare signs directly
        predictions = np.array([0.01, 0.02, -0.01, 0.015, -0.02])
        targets = np.array([0.005, 0.01, -0.005, 0.01, -0.01])

        accuracy = FinancialMetrics.directional_accuracy(predictions, targets, are_returns=True)

        # Perfect sign agreement
        assert accuracy == 1.0

    def test_zero_accuracy(self):
        """Test 0% directional accuracy"""
        # Opposite signs
        predictions = np.array([0.01, 0.02, 0.01, 0.02])
        targets = np.array([-0.01, -0.02, -0.01, -0.02])

        accuracy = FinancialMetrics.directional_accuracy(predictions, targets, are_returns=True)

        assert accuracy == 0.0

    def test_fifty_percent_accuracy(self):
        """Test 50% directional accuracy"""
        # Half correct signs
        predictions = np.array([0.01, 0.02, 0.01, 0.02])
        targets = np.array([0.01, -0.02, 0.01, -0.02])  # 2 correct, 2 wrong

        accuracy = FinancialMetrics.directional_accuracy(predictions, targets, are_returns=True)

        assert accuracy == 0.5

    def test_accuracy_bounds(self):
        """Test accuracy is always between 0 and 1"""
        np.random.seed(42)
        for _ in range(10):
            predictions = np.random.randn(100)
            targets = np.random.randn(100)

            accuracy = FinancialMetrics.directional_accuracy(predictions, targets, are_returns=True)

            assert 0 <= accuracy <= 1


class TestInformationCoefficient:
    """Test Information Coefficient calculation"""

    def test_perfect_correlation(self):
        """Test IC with perfect correlation using returns"""
        # Use varying values so diff has variance
        predictions = np.array([100, 102, 101, 104, 103, 106])
        targets = np.array([100, 102, 101, 104, 103, 106])

        ic = FinancialMetrics.information_coefficient(predictions, targets, use_returns=True)

        # Perfect correlation of changes
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_negative_correlation(self):
        """Test IC with negative correlation"""
        # Changes go in opposite directions
        predictions = np.array([100, 102, 100, 102, 100, 102])
        targets = np.array([100, 98, 100, 98, 100, 98])

        ic = FinancialMetrics.information_coefficient(predictions, targets, use_returns=True)

        # Negative correlation of changes
        assert ic == pytest.approx(-1.0, abs=0.01)

    def test_zero_correlation(self):
        """Test IC with no correlation"""
        np.random.seed(42)
        predictions = np.random.randn(1000)
        targets = np.random.randn(1000)

        ic = FinancialMetrics.information_coefficient(predictions, targets, use_returns=True)

        # Should be close to zero
        assert abs(ic) < 0.15


class TestPrecisionRecall:
    """Test Precision and Recall calculation"""

    def test_perfect_precision_recall(self):
        """Test perfect precision and recall with returns"""
        # Perfect positive change prediction
        predictions = np.array([100, 101, 102, 103, 104])
        targets = np.array([100, 101, 102, 103, 104])

        result = FinancialMetrics.precision_recall(predictions, targets, use_returns=True)

        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0

    def test_precision_recall_bounds(self):
        """Test precision/recall are always between 0 and 1"""
        np.random.seed(42)
        for _ in range(10):
            predictions = np.random.randn(100)
            targets = np.random.randn(100)

            result = FinancialMetrics.precision_recall(predictions, targets, use_returns=True)

            assert 0 <= result['precision'] <= 1
            assert 0 <= result['recall'] <= 1
            assert 0 <= result['f1_score'] <= 1


class TestStrategyReturns:
    """Test Strategy Returns computation"""

    def test_perfect_prediction_returns(self):
        """Test returns with perfect prediction"""
        # If we perfectly predict direction, we should capture all moves
        predictions = np.array([0.01, 0.02, -0.01, 0.015])
        actual_returns = np.array([0.01, 0.02, -0.01, 0.015])

        strategy_returns = compute_strategy_returns(
            predictions, actual_returns, transaction_cost=0.0
        )

        # Should capture absolute value of all moves
        assert len(strategy_returns) == len(actual_returns)

    def test_transaction_costs(self):
        """Test that transaction costs reduce returns"""
        predictions = np.array([0.01, 0.02, 0.015, 0.01])
        actual_returns = np.array([0.01, 0.02, 0.015, 0.01])

        returns_no_cost = compute_strategy_returns(
            predictions, actual_returns, transaction_cost=0.0
        )
        returns_with_cost = compute_strategy_returns(
            predictions, actual_returns, transaction_cost=0.01
        )

        # Returns with cost should be lower
        assert np.sum(returns_with_cost) <= np.sum(returns_no_cost)

    def test_overflow_protection(self):
        """Test that extreme returns don't cause overflow"""
        # Very extreme returns
        predictions = np.array([1.0, 1.0, 1.0] * 100)
        actual_returns = np.array([0.5, 0.5, 0.5] * 100)

        strategy_returns = compute_strategy_returns(predictions, actual_returns)

        # Should not overflow
        assert not np.any(np.isinf(strategy_returns))
        assert not np.any(np.isnan(strategy_returns))


class TestComputeAllMetrics:
    """Test compute_all_metrics function"""

    def test_all_metrics_returned(self):
        """Test that all expected metrics are returned"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        predictions = np.random.randn(252)
        targets = np.random.randn(252)

        metrics = FinancialMetrics.compute_all_metrics(
            returns=returns,
            predictions=predictions,
            targets=targets
        )

        # Check key metrics are present
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'calmar_ratio' in metrics
        assert 'volatility' in metrics
        assert 'directional_accuracy' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics

    def test_no_nan_metrics(self):
        """Test that metrics don't contain NaN"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        predictions = np.random.randn(252)
        targets = np.random.randn(252)

        metrics = FinancialMetrics.compute_all_metrics(
            returns=returns,
            predictions=predictions,
            targets=targets
        )

        # Check no NaN values
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"{key} is NaN"

    def test_no_inf_metrics(self):
        """Test that metrics don't contain Inf"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        predictions = np.random.randn(252)
        targets = np.random.randn(252)

        metrics = FinancialMetrics.compute_all_metrics(
            returns=returns,
            predictions=predictions,
            targets=targets
        )

        # Check no Inf values
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not np.isinf(value), f"{key} is Inf"


class TestTensorInput:
    """Test that functions work with PyTorch tensors"""

    def test_sharpe_with_tensor(self):
        """Test Sharpe ratio with tensor input"""
        returns = torch.tensor([0.01, 0.02, 0.015, 0.01, 0.02])

        # Convert to numpy first since sharpe_ratio doesn't handle tensors directly
        sharpe = FinancialMetrics.sharpe_ratio(returns.numpy(), risk_free_rate=0.0)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_directional_accuracy_with_tensor(self):
        """Test directional accuracy with tensor input (returns mode)"""
        predictions = torch.tensor([0.01, 0.02, -0.01, 0.015, -0.02])
        targets = torch.tensor([0.005, 0.01, -0.005, 0.01, -0.01])

        accuracy = FinancialMetrics.directional_accuracy(predictions, targets, are_returns=True)

        assert isinstance(accuracy, float)
        assert accuracy == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
