"""
Benchmark Trading Strategies

Provides baseline strategies for comparison:
- Buy and Hold
- Naive Last Value (persistence)
- Market Index (SPY)
- Random Walk
- Moving Average Crossover

These benchmarks are essential for demonstrating that
model-based strategies provide genuine value.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

from .strategy_evaluator import (
    TradingStrategy,
    StrategyConfig,
    StrategyResult,
    TradeSignal,
    PositionType,
    StrategyEvaluator
)
from ..evaluation.financial_metrics import FinancialMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkStrategy(TradingStrategy):
    """Base class for benchmark strategies"""

    def __init__(self, name: str = "Benchmark"):
        config = StrategyConfig(
            name=name,
            transaction_cost=0.0,  # Assume no costs for benchmarks
            max_position_size=1.0
        )
        super().__init__(config)


class BuyAndHoldStrategy(BenchmarkStrategy):
    """
    Buy and Hold benchmark.

    Always maintains a 100% long position.
    The simplest possible strategy.
    """

    def __init__(self, name: str = "Buy & Hold"):
        super().__init__(name)

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        """Always long with full position"""
        return [
            TradeSignal(
                timestamp=ts,
                ticker="",
                position=PositionType.LONG,
                confidence=1.0,
                target_weight=1.0,
                predicted_return=0.0
            )
            for ts in timestamps
        ]


class NaiveLastValueStrategy(BenchmarkStrategy):
    """
    Naive Last Value (Persistence) benchmark.

    Predicts that tomorrow's return equals today's return.
    Goes long if last return was positive.
    """

    def __init__(self, name: str = "Naive Persistence"):
        super().__init__(name)

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        # Calculate actual returns
        returns = np.diff(np.log(prices))
        returns = np.concatenate([[0], returns])

        signals = []
        for i, ts in enumerate(timestamps):
            # Use previous return as prediction
            if i > 0 and returns[i] > 0:
                position = PositionType.LONG
                weight = 1.0
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=0.5,  # No real confidence
                target_weight=weight,
                predicted_return=returns[i] if i > 0 else 0
            ))

        return signals


class MomentumBenchmark(BenchmarkStrategy):
    """
    Simple momentum benchmark.

    Goes long if recent returns are positive.
    """

    def __init__(
        self,
        lookback: int = 20,
        name: str = "Momentum (20d)"
    ):
        super().__init__(name)
        self.lookback = lookback

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        # Calculate momentum (rate of change)
        momentum = np.zeros(len(prices))
        for i in range(self.lookback, len(prices)):
            momentum[i] = (prices[i] / prices[i - self.lookback]) - 1

        signals = []
        for i, ts in enumerate(timestamps):
            if momentum[i] > 0:
                position = PositionType.LONG
                weight = 1.0
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=min(1.0, abs(momentum[i]) * 10),
                target_weight=weight,
                predicted_return=momentum[i]
            ))

        return signals


class MeanReversionBenchmark(BenchmarkStrategy):
    """
    Simple mean reversion benchmark.

    Goes long if price is below moving average.
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.02,
        name: str = "Mean Reversion (20d)"
    ):
        super().__init__(name)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        # Calculate moving average
        ma = pd.Series(prices).rolling(window=self.lookback, min_periods=1).mean().values

        # Calculate deviation from MA
        deviation = (prices - ma) / ma

        signals = []
        for i, ts in enumerate(timestamps):
            if deviation[i] < -self.threshold:
                # Price below MA - expect reversion up
                position = PositionType.LONG
                weight = 1.0
            elif deviation[i] > self.threshold:
                # Price above MA - expect reversion down (stay flat if no shorts)
                position = PositionType.FLAT
                weight = 0.0
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=min(1.0, abs(deviation[i]) / self.threshold),
                target_weight=weight,
                predicted_return=-deviation[i]  # Expect reversion
            ))

        return signals


class MACrossoverBenchmark(BenchmarkStrategy):
    """
    Moving Average Crossover benchmark.

    Classic technical analysis strategy.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        name: str = "MA Crossover (10/30)"
    ):
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        # Calculate MAs
        fast_ma = pd.Series(prices).rolling(
            window=self.fast_period, min_periods=1
        ).mean().values
        slow_ma = pd.Series(prices).rolling(
            window=self.slow_period, min_periods=1
        ).mean().values

        signals = []
        for i, ts in enumerate(timestamps):
            if fast_ma[i] > slow_ma[i]:
                position = PositionType.LONG
                weight = 1.0
            else:
                position = PositionType.FLAT
                weight = 0.0

            ma_diff = (fast_ma[i] - slow_ma[i]) / slow_ma[i] if slow_ma[i] > 0 else 0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=min(1.0, abs(ma_diff) * 20),
                target_weight=weight,
                predicted_return=ma_diff
            ))

        return signals


class RandomStrategy(BenchmarkStrategy):
    """
    Random benchmark.

    Randomly goes long/flat. Useful for statistical testing.
    """

    def __init__(
        self,
        seed: int = 42,
        long_probability: float = 0.5,
        name: str = "Random"
    ):
        super().__init__(name)
        self.seed = seed
        self.long_probability = long_probability

    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime]
    ) -> List[TradeSignal]:
        np.random.seed(self.seed)
        random_signals = np.random.random(len(timestamps))

        signals = []
        for i, ts in enumerate(timestamps):
            if random_signals[i] < self.long_probability:
                position = PositionType.LONG
                weight = 1.0
            else:
                position = PositionType.FLAT
                weight = 0.0

            signals.append(TradeSignal(
                timestamp=ts,
                ticker="",
                position=position,
                confidence=0.5,
                target_weight=weight,
                predicted_return=0.0
            ))

        return signals


@dataclass
class BenchmarkComparison:
    """Results of comparing model strategy to benchmarks"""
    model_result: StrategyResult
    benchmark_results: Dict[str, StrategyResult]
    outperformance: Dict[str, float]  # vs each benchmark
    statistical_significance: Dict[str, float]  # p-values
    summary: str


class BenchmarkEvaluator:
    """
    Evaluates model strategies against multiple benchmarks.

    Provides comprehensive comparison and statistical testing.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02
    ):
        self.evaluator = StrategyEvaluator(
            transaction_cost=transaction_cost,
            risk_free_rate=risk_free_rate
        )

    def get_standard_benchmarks(self) -> List[BenchmarkStrategy]:
        """Get standard set of benchmark strategies"""
        return [
            BuyAndHoldStrategy(),
            NaiveLastValueStrategy(),
            MomentumBenchmark(lookback=20),
            MeanReversionBenchmark(lookback=20),
            MACrossoverBenchmark(fast_period=10, slow_period=30),
        ]

    def compare_to_benchmarks(
        self,
        model_strategy: TradingStrategy,
        predictions: np.ndarray,
        prices: np.ndarray,
        timestamps: List[datetime],
        actual_returns: Optional[np.ndarray] = None,
        benchmarks: Optional[List[BenchmarkStrategy]] = None
    ) -> BenchmarkComparison:
        """
        Compare model strategy to benchmarks.

        Args:
            model_strategy: Strategy based on model predictions
            predictions: Model predictions
            prices: Asset prices
            timestamps: Timestamps
            actual_returns: Actual returns
            benchmarks: List of benchmarks (default: standard set)

        Returns:
            BenchmarkComparison with all results
        """
        benchmarks = benchmarks or self.get_standard_benchmarks()

        # Evaluate model strategy
        model_result = self.evaluator.evaluate_strategy(
            model_strategy, predictions, prices, timestamps, actual_returns
        )

        # Evaluate benchmarks
        benchmark_results = {}
        for benchmark in benchmarks:
            result = self.evaluator.evaluate_strategy(
                benchmark, predictions, prices, timestamps, actual_returns
            )
            benchmark_results[benchmark.config.name] = result

        # Calculate outperformance
        outperformance = {}
        for name, bench_result in benchmark_results.items():
            outperformance[name] = model_result.sharpe_ratio - bench_result.sharpe_ratio

        # Statistical significance (simplified - bootstrap would be better)
        significance = {}
        for name, bench_result in benchmark_results.items():
            # Use return difference and standard error for simple t-stat
            diff = model_result.daily_returns - bench_result.daily_returns
            t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))
            # Approximate p-value (two-tailed)
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diff)-1))
            significance[name] = p_value

        # Generate summary
        summary = self._generate_summary(
            model_result, benchmark_results, outperformance, significance
        )

        return BenchmarkComparison(
            model_result=model_result,
            benchmark_results=benchmark_results,
            outperformance=outperformance,
            statistical_significance=significance,
            summary=summary
        )

    def _generate_summary(
        self,
        model_result: StrategyResult,
        benchmark_results: Dict[str, StrategyResult],
        outperformance: Dict[str, float],
        significance: Dict[str, float]
    ) -> str:
        """Generate text summary of comparison"""
        lines = [
            f"Model Strategy: {model_result.strategy_name}",
            f"  Sharpe: {model_result.sharpe_ratio:.2f}",
            f"  Return: {model_result.annualized_return:.1%}",
            "",
            "Benchmark Comparison:",
        ]

        for name, bench_result in benchmark_results.items():
            outperf = outperformance[name]
            sig = significance[name]
            sig_str = "*" if sig < 0.05 else ""

            lines.append(
                f"  vs {name}: Sharpe diff = {outperf:+.2f}{sig_str} "
                f"(p={sig:.3f})"
            )

        # Overall assessment
        beats_all = all(o > 0 for o in outperformance.values())
        beats_bh = outperformance.get("Buy & Hold", 0) > 0

        if beats_all:
            assessment = "OUTPERFORMS all benchmarks"
        elif beats_bh:
            assessment = "OUTPERFORMS buy & hold"
        else:
            assessment = "UNDERPERFORMS buy & hold"

        lines.append("")
        lines.append(f"Assessment: {assessment}")

        return "\n".join(lines)

    def generate_comparison_table(
        self,
        comparison: BenchmarkComparison
    ) -> pd.DataFrame:
        """Generate comparison table for reporting"""
        rows = []

        # Model row
        model = comparison.model_result
        rows.append({
            'Strategy': f"{model.strategy_name} (Model)",
            'Total Return': model.total_return,
            'Ann. Return': model.annualized_return,
            'Volatility': model.volatility,
            'Sharpe': model.sharpe_ratio,
            'Sortino': model.sortino_ratio,
            'Max DD': model.max_drawdown,
            'Win Rate': model.win_rate,
            'Trades': model.n_trades
        })

        # Benchmark rows
        for name, result in comparison.benchmark_results.items():
            rows.append({
                'Strategy': name,
                'Total Return': result.total_return,
                'Ann. Return': result.annualized_return,
                'Volatility': result.volatility,
                'Sharpe': result.sharpe_ratio,
                'Sortino': result.sortino_ratio,
                'Max DD': result.max_drawdown,
                'Win Rate': result.win_rate,
                'Trades': result.n_trades
            })

        df = pd.DataFrame(rows)

        # Format columns
        pct_cols = ['Total Return', 'Ann. Return', 'Volatility', 'Max DD', 'Win Rate']
        for col in pct_cols:
            df[col] = df[col].apply(lambda x: f"{x:.1%}")

        float_cols = ['Sharpe', 'Sortino']
        for col in float_cols:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")

        return df


def evaluate_against_benchmarks(
    model_strategy: TradingStrategy,
    predictions: np.ndarray,
    prices: np.ndarray,
    timestamps: List[datetime],
    actual_returns: Optional[np.ndarray] = None
) -> BenchmarkComparison:
    """
    Convenience function to evaluate strategy against standard benchmarks.

    Args:
        model_strategy: Strategy to evaluate
        predictions: Model predictions
        prices: Asset prices
        timestamps: Timestamps
        actual_returns: Actual returns

    Returns:
        BenchmarkComparison
    """
    evaluator = BenchmarkEvaluator()
    return evaluator.compare_to_benchmarks(
        model_strategy, predictions, prices, timestamps, actual_returns
    )
