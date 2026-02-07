"""Evaluation modules"""

from .metrics import (
    calculate_metrics,
    calculate_financial_metrics,
    MetricsCalculator
)
from .backtester import Backtester, BacktestResults
from .backtesting_platform import (
    BacktestingPlatform,
    BacktestConfig,
    StrategyResult,
    Strategy,
    BuyAndHoldStrategy,
    SMACrossoverStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    ModelBasedStrategy,
    run_comprehensive_backtest
)
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults,
    compute_var_cvar,
    monte_carlo_price_path
)
from .naive_baselines import (
    NaiveBaselines,
    BaselineResults,
    evaluate_naive_baselines
)
from .financial_metrics import (
    FinancialMetrics,
    compute_strategy_returns,
    validate_metrics
)

__all__ = [
    "calculate_metrics",
    "calculate_financial_metrics",
    "MetricsCalculator",
    "Backtester",
    "BacktestResults",
    "BacktestingPlatform",
    "BacktestConfig",
    "StrategyResult",
    "Strategy",
    "BuyAndHoldStrategy",
    "SMACrossoverStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "ModelBasedStrategy",
    "run_comprehensive_backtest",
    "MonteCarloSimulator",
    "MonteCarloResults",
    "compute_var_cvar",
    "monte_carlo_price_path",
    "NaiveBaselines",
    "BaselineResults",
    "evaluate_naive_baselines",
    "FinancialMetrics",
    "compute_strategy_returns",
    "validate_metrics",
]
