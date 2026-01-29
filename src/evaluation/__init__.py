"""Evaluation modules"""

from .metrics import (
    calculate_metrics,
    calculate_financial_metrics,
    MetricsCalculator
)
from .backtester import Backtester, BacktestResults
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

__all__ = [
    "calculate_metrics",
    "calculate_financial_metrics",
    "MetricsCalculator",
    "Backtester",
    "BacktestResults",
    "MonteCarloSimulator",
    "MonteCarloResults",
    "compute_var_cvar",
    "monte_carlo_price_path",
    "NaiveBaselines",
    "BaselineResults",
    "evaluate_naive_baselines",
]
