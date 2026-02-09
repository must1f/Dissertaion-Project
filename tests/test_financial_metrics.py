"""
Comprehensive tests for financial metrics
"""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.financial_metrics import FinancialMetrics


class TestSharpeRatio:
    """Test Sharpe ratio calculation"""

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns"""
        # Consistent positive returns above risk-free rate
        returns = np.array([0.01, 0.012, 0.015, 0.011, 0.013] * 20)
        
        sharpe = FinancialMetrics.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should be positive
        assert sharpe > 0
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_negative_returns(self):
        """Test Sharpe ratio with negative returns"""
        returns = np.array([-0.01, -0.012, -0.015, -0.011, -0.013] * 20)
        
        sharpe = FinancialMetrics.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should be negative
        assert sharpe < 0

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation"""
        returns = np.array([0.01] * 100)  # All the same
        
        sharpe = FinancialMetrics.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should return 0 when std is 0
        assert sharpe == 0.0

    def test_sharpe_ratio_with_pandas_series(self):
        """Test Sharpe ratio with pandas Series input"""
        returns = pd.Series([0.01, 0.012, 0.015, 0.011, 0.013] * 20)
        
        sharpe = FinancialMetrics.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_sharpe_ratio_with_nans(self):
        """Test Sharpe ratio with NaN values"""
        returns = np.array([0.01, np.nan, 0.015, np.nan, 0.013] * 20)
        
        sharpe = FinancialMetrics.sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should filter out NaNs
        assert not np.isnan(sharpe)
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_empty_after_filtering(self):
        """Test Sharpe ratio when all values are NaN"""
        returns = np.array([np.nan, np.nan, np.nan])
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        
        # Should return 0 for empty array
        assert sharpe == 0.0

    def test_sharpe_ratio_different_frequencies(self):
        """Test Sharpe ratio with different time frequencies"""
        daily_returns = np.random.randn(252) * 0.01 + 0.0005
        
        sharpe_daily = FinancialMetrics.sharpe_ratio(
            daily_returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Monthly returns (21 trading days per month)
        monthly_returns = np.random.randn(12) * 0.05 + 0.01
        
        sharpe_monthly = FinancialMetrics.sharpe_ratio(
            monthly_returns, risk_free_rate=0.02, periods_per_year=12
        )
        
        assert isinstance(sharpe_daily, float)
        assert isinstance(sharpe_monthly, float)


class TestSortinoRatio:
    """Test Sortino ratio calculation"""

    def test_sortino_ratio_basic(self):
        """Test basic Sortino ratio calculation"""
        # Mixed returns
        returns = np.array([0.02, 0.01, -0.01, 0.03, -0.005, 0.015, -0.02, 0.025])
        
        sortino = FinancialMetrics.sortino_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio when all returns are positive"""
        returns = np.array([0.01, 0.02, 0.015, 0.018, 0.012])
        
        sortino = FinancialMetrics.sortino_ratio(
            returns, risk_free_rate=0.0, periods_per_year=252
        )
        
        # Should be inf or 0 when no downside
        assert np.isinf(sortino) or sortino == 0.0

    def test_sortino_ratio_all_downside(self):
        """Test Sortino ratio when all returns are negative"""
        returns = np.array([-0.01, -0.02, -0.015, -0.018, -0.012])
        
        sortino = FinancialMetrics.sortino_ratio(
            returns, risk_free_rate=0.02, periods_per_year=252
        )
        
        # Should be negative
        assert sortino < 0

    def test_sortino_vs_sharpe(self):
        """Test that Sortino is typically higher than Sharpe (penalizes downside only)"""
        # Returns with more upside volatility than downside
        returns = np.array([0.05, 0.04, -0.01, 0.06, -0.005, 0.07, -0.01, 0.08])
        
        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        sortino = FinancialMetrics.sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        
        # Sortino should be >= Sharpe (less penalty for upside volatility)
        assert sortino >= sharpe

    def test_sortino_ratio_with_nans(self):
        """Test Sortino ratio with NaN values"""
        returns = np.array([0.01, np.nan, -0.01, np.nan, 0.015])
        
        sortino = FinancialMetrics.sortino_ratio(returns)
        
        # Should filter NaNs
        assert not np.isnan(sortino)


class TestMaxDrawdown:
    """Test maximum drawdown calculation"""

    def test_max_drawdown_basic(self):
        """Test basic maximum drawdown calculation"""
        # Returns that lead to a drawdown
        returns = np.array([0.1, 0.05, -0.15, -0.10, 0.05, 0.08])
        
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should be negative (it's a drawdown)
        assert max_dd < 0
        assert isinstance(max_dd, float)

    def test_max_drawdown_no_losses(self):
        """Test max drawdown with only positive returns"""
        returns = np.array([0.01, 0.02, 0.015, 0.018, 0.012])
        
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should be 0 or very small (no drawdown)
        assert max_dd >= -1e-10

    def test_max_drawdown_continuous_loss(self):
        """Test max drawdown with continuous losses"""
        returns = np.array([-0.1, -0.1, -0.1, -0.1])
        
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should be significant negative value
        assert max_dd < -0.3

    def test_max_drawdown_with_series(self):
        """Test max drawdown returning drawdown series"""
        returns = np.array([0.1, 0.05, -0.15, -0.10, 0.05, 0.08])
        
        max_dd, dd_series = FinancialMetrics.max_drawdown(returns, return_series=True)
        
        assert max_dd < 0
        assert isinstance(dd_series, np.ndarray)
        assert len(dd_series) == len(returns)
        
        # All drawdowns should be <= 0
        assert (dd_series <= 0).all()
        
        # Maximum should match max_dd
        assert np.min(dd_series) == max_dd

    def test_max_drawdown_recovery(self):
        """Test max drawdown with recovery"""
        # Draw down then recover
        returns = np.array([0.1, 0.1, -0.2, -0.1, 0.15, 0.15, 0.15])
        
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should capture the maximum drawdown even if recovered
        assert max_dd < -0.25

    def test_max_drawdown_empty_array(self):
        """Test max drawdown with empty array"""
        returns = np.array([])
        
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        assert max_dd == 0.0

    def test_max_drawdown_with_nans(self):
        """Test max drawdown with NaN values"""
        returns = np.array([0.1, np.nan, -0.15, np.nan, 0.05])
        
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should filter NaNs
        assert not np.isnan(max_dd)


class TestCalmarRatio:
    """Test Calmar ratio calculation"""

    def test_calmar_ratio_basic(self):
        """Test basic Calmar ratio calculation"""
        # Returns with some drawdown
        returns = np.array([0.01, 0.02, -0.05, 0.03, -0.02, 0.04] * 10)
        
        calmar = FinancialMetrics.calmar_ratio(returns, periods_per_year=252)
        
        assert isinstance(calmar, float)
        assert not np.isnan(calmar)

    def test_calmar_ratio_no_drawdown(self):
        """Test Calmar ratio with no drawdown"""
        returns = np.array([0.01, 0.02, 0.015, 0.018] * 10)
        
        calmar = FinancialMetrics.calmar_ratio(returns, periods_per_year=252)
        
        # Should be inf or very large when no drawdown
        assert calmar >= 0 or np.isinf(calmar)

    def test_calmar_ratio_negative_returns(self):
        """Test Calmar ratio with negative average returns"""
        returns = np.array([-0.01, -0.02, -0.015, -0.018] * 10)
        
        calmar = FinancialMetrics.calmar_ratio(returns, periods_per_year=252)
        
        # Should be negative
        assert calmar < 0


class TestCumulativeReturns:
    """Test cumulative returns calculation"""

    def test_cumulative_returns_basic(self):
        """Test basic cumulative returns"""
        returns = np.array([0.1, 0.05, -0.05, 0.1])
        
        cum_returns = FinancialMetrics.cumulative_returns(returns)
        
        assert isinstance(cum_returns, np.ndarray)
        assert len(cum_returns) == len(returns)
        
        # Should be cumulative product
        expected = np.array([1.1, 1.1*1.05, 1.1*1.05*0.95, 1.1*1.05*0.95*1.1]) - 1
        np.testing.assert_allclose(cum_returns, expected, rtol=1e-5)

    def test_cumulative_returns_positive(self):
        """Test cumulative returns with all positive"""
        returns = np.array([0.01, 0.02, 0.015])
        
        cum_returns = FinancialMetrics.cumulative_returns(returns)
        
        # Should be monotonically increasing
        assert (np.diff(cum_returns) > 0).all()

    def test_cumulative_returns_negative(self):
        """Test cumulative returns with losses"""
        returns = np.array([0.1, -0.2, 0.1])
        
        cum_returns = FinancialMetrics.cumulative_returns(returns)
        
        # Final value should reflect compounding
        final_value = (1 + 0.1) * (1 - 0.2) * (1 + 0.1) - 1
        np.testing.assert_allclose(cum_returns[-1], final_value, rtol=1e-5)


class TestFinancialMetricsEdgeCases:
    """Test edge cases for financial metrics"""

    def test_all_zeros(self):
        """Test with all zero returns"""
        returns = np.zeros(100)
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        sortino = FinancialMetrics.sortino_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        assert sharpe == 0.0
        assert not np.isnan(sortino)
        assert max_dd >= -1e-10  # Should be ~0

    def test_single_return(self):
        """Test with single return value"""
        returns = np.array([0.01])
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        
        # Should handle single value (std will be 0)
        assert sharpe == 0.0

    def test_extreme_values(self):
        """Test with extreme return values"""
        returns = np.array([10.0, -0.9, 5.0, -0.8])  # 1000%, -90%, 500%, -80%
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should handle extreme values
        assert isinstance(sharpe, float)
        assert isinstance(max_dd, float)
        assert not np.isnan(sharpe)
        assert not np.isnan(max_dd)

    def test_very_long_series(self):
        """Test with very long return series"""
        np.random.seed(42)
        returns = np.random.randn(10000) * 0.01 + 0.0005
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        sortino = FinancialMetrics.sortino_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Should handle large datasets
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(max_dd, float)

    def test_high_volatility(self):
        """Test with high volatility returns"""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.5  # 50% daily volatility!
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        
        # Should handle high volatility
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_consistent_small_losses(self):
        """Test with consistent small losses"""
        returns = np.array([-0.001] * 100)
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Sharpe should be negative (consistent losses)
        assert sharpe < 0
        
        # Max drawdown should be significant
        assert max_dd < -0.05


class TestFinancialMetricsIntegration:
    """Integration tests for financial metrics"""

    def test_realistic_trading_scenario(self):
        """Test with realistic trading returns"""
        np.random.seed(42)
        # Simulate realistic daily returns: 12% annual return, 20% annual volatility
        daily_mean = 0.12 / 252
        daily_std = 0.20 / np.sqrt(252)
        returns = np.random.randn(252) * daily_std + daily_mean
        
        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        sortino = FinancialMetrics.sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        max_dd = FinancialMetrics.max_drawdown(returns)
        calmar = FinancialMetrics.calmar_ratio(returns, periods_per_year=252)
        
        # All metrics should be reasonable
        assert -5 < sharpe < 5  # Typical range
        assert sortino >= sharpe  # Sortino usually >= Sharpe
        assert -0.5 < max_dd < 0  # Max drawdown should be negative but not too extreme
        assert not np.isnan(calmar)

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other"""
        returns = np.array([0.02, 0.01, -0.01, 0.03, -0.02, 0.015, 0.02, -0.01])
        
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        sortino = FinancialMetrics.sortino_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Sortino should generally be >= Sharpe
        assert sortino >= sharpe - 0.1  # Allow small numerical differences
        
        # Max drawdown should be negative or zero
        assert max_dd <= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
