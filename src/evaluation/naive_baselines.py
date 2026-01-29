"""
Naive Trading Baselines for Benchmarking

Implements simple trading strategies to establish baseline performance:
- Buy-and-Hold: Always long
- Random Walk: Predict no change (naive forecast)
- Moving Average Crossover: Classic technical analysis
- Momentum: Follow recent trends
- Mean Reversion: Trade against recent trends

These baselines address the audit finding about missing naive strategy comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass

from .financial_metrics import FinancialMetrics, compute_strategy_returns
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaselineResults:
    """Results from baseline strategy evaluation"""
    strategy_name: str
    predictions: np.ndarray
    positions: np.ndarray
    strategy_returns: np.ndarray
    metrics: Dict[str, float]


class NaiveBaselines:
    """
    Collection of naive trading strategies for benchmarking

    These strategies require no training and serve as important
    baselines for evaluating learned models.
    """

    def __init__(
        self,
        transaction_cost: float = 0.003,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Args:
            transaction_cost: Cost per trade (default 0.3%)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def buy_and_hold(
        self,
        prices: np.ndarray
    ) -> BaselineResults:
        """
        Buy-and-Hold strategy: Always maintain long position

        This is the simplest baseline - buy at start, hold until end.
        Any forecasting model should ideally beat this.

        Args:
            prices: Price array [n_samples]

        Returns:
            BaselineResults
        """
        prices = prices.flatten()
        n = len(prices)

        # Always predict "up" (positive return)
        predictions = np.ones(n) * 0.01  # Small positive prediction

        # Always long
        positions = np.ones(n)

        # Compute actual returns
        actual_returns = np.diff(prices) / prices[:-1]
        actual_returns = np.append(actual_returns, 0)  # Pad to match length

        # Strategy returns (only transaction cost at entry)
        strategy_returns = actual_returns.copy()
        strategy_returns[0] -= self.transaction_cost  # Entry cost

        # Compute metrics
        metrics = FinancialMetrics.compute_all_metrics(
            returns=strategy_returns,
            predictions=predictions,
            targets=actual_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            predictions_are_returns=True
        )

        return BaselineResults(
            strategy_name="Buy-and-Hold",
            predictions=predictions,
            positions=positions,
            strategy_returns=strategy_returns,
            metrics=metrics
        )

    def random_walk(
        self,
        prices: np.ndarray
    ) -> BaselineResults:
        """
        Random Walk baseline: Predict no change

        The efficient market hypothesis suggests prices follow a random walk,
        meaning the best prediction for tomorrow's price is today's price.
        This baseline predicts zero returns (no change).

        Args:
            prices: Price array [n_samples]

        Returns:
            BaselineResults
        """
        prices = prices.flatten()
        n = len(prices)

        # Predict no change (zero return)
        predictions = np.zeros(n)

        # Position based on prediction: zero means hold/flat
        positions = np.zeros(n)

        # Actual returns
        actual_returns = np.diff(prices) / prices[:-1]
        actual_returns = np.append(actual_returns, 0)

        # No positions means no strategy returns
        strategy_returns = np.zeros(n)

        # Compute metrics
        metrics = FinancialMetrics.compute_all_metrics(
            returns=strategy_returns,
            predictions=predictions,
            targets=actual_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            predictions_are_returns=True
        )

        return BaselineResults(
            strategy_name="Random Walk (No Change)",
            predictions=predictions,
            positions=positions,
            strategy_returns=strategy_returns,
            metrics=metrics
        )

    def moving_average_crossover(
        self,
        prices: np.ndarray,
        short_window: int = 50,
        long_window: int = 200
    ) -> BaselineResults:
        """
        Moving Average Crossover strategy

        Classic technical analysis: go long when short MA > long MA,
        short when short MA < long MA.

        Args:
            prices: Price array [n_samples]
            short_window: Short moving average window (default 50 days)
            long_window: Long moving average window (default 200 days)

        Returns:
            BaselineResults
        """
        prices = prices.flatten()
        n = len(prices)

        # Compute moving averages
        short_ma = pd.Series(prices).rolling(window=short_window, min_periods=1).mean().values
        long_ma = pd.Series(prices).rolling(window=long_window, min_periods=1).mean().values

        # Generate signals
        # Long when short MA > long MA, flat otherwise
        positions = np.where(short_ma > long_ma, 1.0, 0.0)

        # Predictions: predict positive return when long, negative when short
        predictions = np.where(short_ma > long_ma, 0.01, -0.01)

        # Actual returns
        actual_returns = np.diff(prices) / prices[:-1]
        actual_returns = np.append(actual_returns, 0)

        # Strategy returns with transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        strategy_returns = positions * actual_returns - position_changes * self.transaction_cost

        # Compute metrics
        metrics = FinancialMetrics.compute_all_metrics(
            returns=strategy_returns,
            predictions=predictions,
            targets=actual_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            predictions_are_returns=True
        )

        return BaselineResults(
            strategy_name=f"MA Crossover ({short_window}/{long_window})",
            predictions=predictions,
            positions=positions,
            strategy_returns=strategy_returns,
            metrics=metrics
        )

    def momentum(
        self,
        prices: np.ndarray,
        lookback: int = 20
    ) -> BaselineResults:
        """
        Momentum strategy: Follow recent trend

        Go long if recent returns are positive, short if negative.
        This captures the idea that trends tend to persist.

        Args:
            prices: Price array [n_samples]
            lookback: Lookback period for momentum calculation (default 20 days)

        Returns:
            BaselineResults
        """
        prices = prices.flatten()
        n = len(prices)

        # Compute momentum (cumulative return over lookback period)
        momentum = pd.Series(prices).pct_change(lookback).values

        # Position: long if momentum > 0, flat otherwise
        positions = np.where(momentum > 0, 1.0, 0.0)
        positions = np.nan_to_num(positions, nan=0.0)

        # Predictions based on momentum
        predictions = np.sign(momentum) * 0.01
        predictions = np.nan_to_num(predictions, nan=0.0)

        # Actual returns
        actual_returns = np.diff(prices) / prices[:-1]
        actual_returns = np.append(actual_returns, 0)

        # Strategy returns with transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        strategy_returns = positions * actual_returns - position_changes * self.transaction_cost

        # Compute metrics
        metrics = FinancialMetrics.compute_all_metrics(
            returns=strategy_returns,
            predictions=predictions,
            targets=actual_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            predictions_are_returns=True
        )

        return BaselineResults(
            strategy_name=f"Momentum ({lookback}-day)",
            predictions=predictions,
            positions=positions,
            strategy_returns=strategy_returns,
            metrics=metrics
        )

    def mean_reversion(
        self,
        prices: np.ndarray,
        lookback: int = 20,
        threshold: float = 1.5
    ) -> BaselineResults:
        """
        Mean Reversion strategy: Trade against extremes

        Go long when price is below moving average (expecting reversion up),
        short when above (expecting reversion down).

        Args:
            prices: Price array [n_samples]
            lookback: Lookback period for mean calculation
            threshold: Number of standard deviations to trigger signal

        Returns:
            BaselineResults
        """
        prices = prices.flatten()
        n = len(prices)

        # Compute rolling mean and std
        rolling_mean = pd.Series(prices).rolling(window=lookback, min_periods=1).mean().values
        rolling_std = pd.Series(prices).rolling(window=lookback, min_periods=1).std().values
        rolling_std = np.maximum(rolling_std, 1e-8)  # Avoid division by zero

        # Z-score
        z_score = (prices - rolling_mean) / rolling_std

        # Position: long when oversold (z < -threshold), short when overbought (z > threshold)
        positions = np.zeros(n)
        positions[z_score < -threshold] = 1.0   # Oversold: buy
        positions[z_score > threshold] = -1.0   # Overbought: sell (if short selling allowed)

        # For long-only version:
        positions = np.maximum(positions, 0)

        # Predictions: expect mean reversion
        predictions = -np.sign(z_score) * 0.01
        predictions = np.nan_to_num(predictions, nan=0.0)

        # Actual returns
        actual_returns = np.diff(prices) / prices[:-1]
        actual_returns = np.append(actual_returns, 0)

        # Strategy returns with transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        strategy_returns = positions * actual_returns - position_changes * self.transaction_cost

        # Compute metrics
        metrics = FinancialMetrics.compute_all_metrics(
            returns=strategy_returns,
            predictions=predictions,
            targets=actual_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            predictions_are_returns=True
        )

        return BaselineResults(
            strategy_name=f"Mean Reversion ({lookback}-day, {threshold}σ)",
            predictions=predictions,
            positions=positions,
            strategy_returns=strategy_returns,
            metrics=metrics
        )

    def evaluate_all(
        self,
        prices: np.ndarray
    ) -> Dict[str, BaselineResults]:
        """
        Evaluate all baseline strategies

        Args:
            prices: Price array

        Returns:
            Dict of {strategy_name: BaselineResults}
        """
        logger.info("=" * 80)
        logger.info("NAIVE BASELINE EVALUATION")
        logger.info("=" * 80)

        results = {}

        # Buy-and-Hold
        results['buy_and_hold'] = self.buy_and_hold(prices)
        logger.info(f"Buy-and-Hold Sharpe: {results['buy_and_hold'].metrics.get('sharpe_ratio', 0):.3f}")

        # Random Walk
        results['random_walk'] = self.random_walk(prices)
        logger.info(f"Random Walk Sharpe: {results['random_walk'].metrics.get('sharpe_ratio', 0):.3f}")

        # MA Crossover (50/200)
        results['ma_crossover'] = self.moving_average_crossover(prices, 50, 200)
        logger.info(f"MA Crossover Sharpe: {results['ma_crossover'].metrics.get('sharpe_ratio', 0):.3f}")

        # Momentum (20-day)
        results['momentum_20'] = self.momentum(prices, 20)
        logger.info(f"Momentum-20 Sharpe: {results['momentum_20'].metrics.get('sharpe_ratio', 0):.3f}")

        # Mean Reversion
        results['mean_reversion'] = self.mean_reversion(prices, 20, 1.5)
        logger.info(f"Mean Reversion Sharpe: {results['mean_reversion'].metrics.get('sharpe_ratio', 0):.3f}")

        logger.info("=" * 80)

        return results

    def compare_with_model(
        self,
        prices: np.ndarray,
        model_predictions: np.ndarray,
        model_name: str = "PINN Model"
    ) -> pd.DataFrame:
        """
        Compare model performance against all baselines

        Args:
            prices: Price array
            model_predictions: Model's price predictions
            model_name: Name of the model for display

        Returns:
            DataFrame comparing all strategies
        """
        # Evaluate baselines
        baseline_results = self.evaluate_all(prices)

        # Evaluate model
        actual_returns = np.diff(prices.flatten()) / prices.flatten()[:-1]
        actual_returns = np.append(actual_returns, 0)

        model_strategy_returns = compute_strategy_returns(
            model_predictions, prices.flatten(), self.transaction_cost
        )

        model_metrics = FinancialMetrics.compute_all_metrics(
            returns=model_strategy_returns,
            predictions=model_predictions.flatten(),
            targets=prices.flatten(),
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            predictions_are_returns=False  # Model predicts prices, not returns
        )

        # Build comparison table
        rows = []

        for name, result in baseline_results.items():
            rows.append({
                'Strategy': result.strategy_name,
                'Sharpe': result.metrics.get('sharpe_ratio', 0),
                'Sortino': result.metrics.get('sortino_ratio', 0),
                'Max_DD_%': result.metrics.get('max_drawdown', 0) * 100,
                'Annual_Ret_%': result.metrics.get('annualized_return', 0) * 100,
                'Dir_Acc_%': result.metrics.get('directional_accuracy', 0) * 100,
                'Win_Rate_%': result.metrics.get('win_rate', 0) * 100,
                'Profit_Factor': result.metrics.get('profit_factor', 0)
            })

        # Add model
        rows.append({
            'Strategy': model_name,
            'Sharpe': model_metrics.get('sharpe_ratio', 0),
            'Sortino': model_metrics.get('sortino_ratio', 0),
            'Max_DD_%': model_metrics.get('max_drawdown', 0) * 100,
            'Annual_Ret_%': model_metrics.get('annualized_return', 0) * 100,
            'Dir_Acc_%': model_metrics.get('directional_accuracy', 0) * 100,
            'Win_Rate_%': model_metrics.get('win_rate', 0) * 100,
            'Profit_Factor': model_metrics.get('profit_factor', 0)
        })

        df = pd.DataFrame(rows)
        df = df.sort_values('Sharpe', ascending=False)

        logger.info("\nModel vs Baseline Comparison:")
        logger.info(df.to_string(index=False))

        return df


def evaluate_naive_baselines(
    prices: np.ndarray,
    transaction_cost: float = 0.003
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to evaluate all naive baselines

    Args:
        prices: Price array
        transaction_cost: Transaction cost per trade

    Returns:
        Dict of {strategy_name: metrics_dict}
    """
    evaluator = NaiveBaselines(transaction_cost=transaction_cost)
    results = evaluator.evaluate_all(prices)

    return {name: result.metrics for name, result in results.items()}
