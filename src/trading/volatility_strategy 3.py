"""
Volatility Targeting Strategy

Implements volatility-targeting portfolio construction for backtesting
volatility forecasts. The strategy scales position size inversely to
predicted volatility to maintain constant portfolio risk.

Strategy:
    w_t = σ_target / σ̂_t

This approach:
    1. Reduces exposure when volatility is high (risk management)
    2. Increases exposure when volatility is low (return enhancement)
    3. Maintains approximately constant portfolio volatility

References:
    - Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios."
      Journal of Finance.
    - Fleming, J., Kirby, C., & Ostdiek, B. (2001). "The Economic Value
      of Volatility Timing." Journal of Finance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from ..constants import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

logger = get_logger(__name__)


@dataclass
class StrategyResult:
    """Container for strategy backtest results."""
    equity_curve: np.ndarray
    returns: np.ndarray
    weights: np.ndarray
    dates: Optional[pd.DatetimeIndex]

    # Performance metrics
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Comparison metrics
    benchmark_sharpe: Optional[float] = None
    benchmark_return: Optional[float] = None
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    # Strategy characteristics
    avg_leverage: float = 1.0
    leverage_std: float = 0.0
    turnover: float = 0.0
    realized_vol: float = 0.0
    vol_tracking_error: float = 0.0


class VolatilityTargetingStrategy:
    """
    Volatility-Targeting Portfolio Strategy.

    Scales position size to maintain constant portfolio volatility:
        w_t = σ_target / σ̂_t

    With constraints:
        - Minimum weight (e.g., 0.25 = max 4x reduction)
        - Maximum weight (e.g., 2.0 = max 2x leverage)
        - Optional rebalancing frequency (daily, weekly, monthly)
    """

    def __init__(
        self,
        target_vol: float = 0.15,  # 15% annual target volatility
        min_weight: float = 0.25,  # Minimum position (25%)
        max_weight: float = 2.0,   # Maximum position (200%, 2x leverage)
        transaction_cost: float = 0.001,  # 10 bps per trade
        rebalance_freq: str = 'daily',  # 'daily', 'weekly', 'monthly'
        use_trailing_vol: bool = False,  # Use trailing realized vol as backup
        trailing_window: int = 20,
    ):
        """
        Initialize volatility targeting strategy.

        Args:
            target_vol: Target annual volatility
            min_weight: Minimum position weight
            max_weight: Maximum position weight (leverage limit)
            transaction_cost: Transaction cost per trade
            rebalance_freq: Rebalancing frequency
            use_trailing_vol: Whether to use trailing realized vol as backup
            trailing_window: Window for trailing volatility calculation
        """
        self.target_vol = target_vol
        self.target_vol_daily = target_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.use_trailing_vol = use_trailing_vol
        self.trailing_window = trailing_window

        logger.info(f"VolatilityTargetingStrategy initialized: "
                   f"target={target_vol:.1%}, limits=[{min_weight:.2f}, {max_weight:.2f}]")

    def compute_weights(
        self,
        predicted_vol: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute position weights based on predicted volatility.

        Args:
            predicted_vol: Predicted volatility (daily or annualized)
            returns: Historical returns (for trailing vol backup)

        Returns:
            Array of position weights
        """
        predicted_vol = np.asarray(predicted_vol).flatten()

        # Determine if volatility is daily or annualized
        # (heuristic: daily vol is typically < 0.05)
        if np.nanmean(predicted_vol) > 0.1:
            # Likely annualized, convert to daily
            vol_daily = predicted_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            vol_daily = predicted_vol

        # Handle missing/invalid predictions
        if self.use_trailing_vol and returns is not None:
            # Use trailing realized vol as backup
            trailing_vol = self._compute_trailing_vol(returns)
            invalid_mask = np.isnan(vol_daily) | (vol_daily <= 0)
            vol_daily = np.where(invalid_mask, trailing_vol, vol_daily)

        # Compute target weights
        weights = self.target_vol_daily / np.maximum(vol_daily, 1e-6)

        # Apply constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Handle any remaining NaN
        weights = np.nan_to_num(weights, nan=1.0)

        return weights

    def _compute_trailing_vol(self, returns: np.ndarray) -> np.ndarray:
        """Compute trailing realized volatility."""
        returns = np.asarray(returns).flatten()
        vol = pd.Series(returns).rolling(
            window=self.trailing_window,
            min_periods=5
        ).std().values
        vol[0] = np.nanstd(returns[:self.trailing_window])
        return np.nan_to_num(vol, nan=np.nanstd(returns))

    def _get_rebalance_mask(self, n: int) -> np.ndarray:
        """Get mask for rebalancing periods."""
        mask = np.ones(n, dtype=bool)

        if self.rebalance_freq == 'daily':
            return mask
        elif self.rebalance_freq == 'weekly':
            # Rebalance every 5 days
            mask[1:] = np.arange(1, n) % 5 == 0
        elif self.rebalance_freq == 'monthly':
            # Rebalance every 22 days
            mask[1:] = np.arange(1, n) % 22 == 0

        return mask

    def backtest(
        self,
        returns: np.ndarray,
        predicted_vol: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> StrategyResult:
        """
        Full backtest of volatility-targeting strategy.

        Args:
            returns: Asset returns
            predicted_vol: Predicted volatility
            benchmark_returns: Benchmark returns (e.g., buy-and-hold)
            dates: Date index

        Returns:
            StrategyResult with full backtest results
        """
        returns = np.asarray(returns).flatten()
        predicted_vol = np.asarray(predicted_vol).flatten()

        n = min(len(returns), len(predicted_vol))
        returns = returns[:n]
        predicted_vol = predicted_vol[:n]

        if benchmark_returns is None:
            benchmark_returns = returns.copy()
        else:
            benchmark_returns = np.asarray(benchmark_returns).flatten()[:n]

        # Compute weights
        weights = self.compute_weights(predicted_vol, returns)

        # Apply rebalancing frequency
        rebal_mask = self._get_rebalance_mask(n)

        # Forward-fill weights between rebalancing dates
        final_weights = np.zeros(n)
        current_weight = weights[0]

        for i in range(n):
            if rebal_mask[i]:
                current_weight = weights[i]
            final_weights[i] = current_weight

        weights = final_weights

        # Strategy returns (before costs)
        strategy_returns = weights * returns

        # Transaction costs from weight changes
        weight_changes = np.abs(np.diff(weights, prepend=weights[0]))
        costs = weight_changes * self.transaction_cost

        # Net returns
        strategy_returns_net = strategy_returns - costs

        # Equity curve
        equity_curve = np.cumprod(1 + strategy_returns_net)

        # Performance metrics
        metrics = self._compute_metrics(
            strategy_returns_net,
            weights,
            benchmark_returns,
        )

        return StrategyResult(
            equity_curve=equity_curve,
            returns=strategy_returns_net,
            weights=weights,
            dates=dates,
            **metrics
        )

    def _compute_metrics(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict:
        """Compute comprehensive performance metrics."""
        n = len(returns)

        # Total and annualized return
        total_return = np.prod(1 + returns) - 1
        n_years = n / TRADING_DAYS_PER_YEAR

        if n_years > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_return = -1.0

        # Volatility
        annual_vol = np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe ratio
        if annual_vol > 1e-6:
            sharpe = (annual_return - RISK_FREE_RATE) / annual_vol
        else:
            sharpe = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if downside_vol > 1e-6:
                sortino = (annual_return - RISK_FREE_RATE) / downside_vol
            else:
                sortino = 10.0
        else:
            sortino = 10.0

        # Maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Calmar ratio
        if abs(max_drawdown) > 0.001:
            calmar = annual_return / abs(max_drawdown)
        else:
            calmar = 10.0 if annual_return > 0 else 0.0

        # Strategy characteristics
        avg_leverage = np.mean(weights)
        leverage_std = np.std(weights)
        turnover = np.sum(np.abs(np.diff(weights))) * TRADING_DAYS_PER_YEAR / n

        # Realized volatility
        realized_vol = annual_vol
        vol_tracking_error = abs(realized_vol - self.target_vol)

        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            bench_total = np.prod(1 + benchmark_returns) - 1
            bench_annual = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 and bench_total > -1 else -1.0
            bench_vol = np.std(benchmark_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
            bench_sharpe = (bench_annual - RISK_FREE_RATE) / bench_vol if bench_vol > 1e-6 else 0.0

            # Alpha and beta (CAPM regression)
            if np.std(benchmark_returns) > 1e-6:
                cov = np.cov(returns, benchmark_returns)[0, 1]
                var_bench = np.var(benchmark_returns)
                beta = cov / var_bench if var_bench > 1e-10 else 1.0
                alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
                alpha_annual = alpha * TRADING_DAYS_PER_YEAR
            else:
                beta = 1.0
                alpha_annual = 0.0

            # Information ratio
            active_returns = returns - benchmark_returns
            if np.std(active_returns) > 1e-6:
                ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                ir = 0.0
        else:
            bench_sharpe = None
            bench_annual = None
            ir = None
            alpha_annual = None
            beta = None

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(np.clip(sharpe, -5, 5)),
            'sortino_ratio': float(np.clip(sortino, -10, 10)),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(np.clip(calmar, -10, 10)),
            'benchmark_sharpe': float(bench_sharpe) if bench_sharpe is not None else None,
            'benchmark_return': float(bench_annual) if bench_annual is not None else None,
            'information_ratio': float(ir) if ir is not None else None,
            'alpha': float(alpha_annual) if alpha_annual is not None else None,
            'beta': float(beta) if beta is not None else None,
            'avg_leverage': float(avg_leverage),
            'leverage_std': float(leverage_std),
            'turnover': float(turnover),
            'realized_vol': float(realized_vol),
            'vol_tracking_error': float(vol_tracking_error),
        }

    def compare_to_benchmark(
        self,
        strategy_result: StrategyResult,
        benchmark_result: StrategyResult,
    ) -> Dict[str, float]:
        """
        Compare strategy performance to benchmark.

        Args:
            strategy_result: Result from volatility-targeting strategy
            benchmark_result: Result from benchmark (e.g., buy-and-hold)

        Returns:
            Dictionary with comparison metrics
        """
        return {
            'sharpe_improvement': strategy_result.sharpe_ratio - (benchmark_result.sharpe_ratio or 0),
            'return_improvement': strategy_result.annual_return - (benchmark_result.annual_return or 0),
            'vol_reduction': (benchmark_result.annual_volatility or 0) - strategy_result.annual_volatility,
            'drawdown_improvement': strategy_result.max_drawdown - (benchmark_result.max_drawdown or 0),
            'sharpe_ratio_strategy': strategy_result.sharpe_ratio,
            'sharpe_ratio_benchmark': benchmark_result.sharpe_ratio,
        }


class RiskParityStrategy:
    """
    Risk Parity Strategy for multi-asset volatility targeting.

    Allocates portfolio weights to equalize risk contribution from each asset.
    Each asset contributes equally to total portfolio risk.

    w_i ∝ 1 / σ_i  (simplified equal risk contribution)

    Reference:
        Maillard, S., Roncalli, T., & Teiletche, J. (2010). "The Properties
        of Equally Weighted Risk Contribution Portfolios."
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        min_weight: float = 0.05,
        max_weight: float = 0.5,
        transaction_cost: float = 0.001,
    ):
        """
        Initialize risk parity strategy.

        Args:
            target_vol: Target portfolio volatility
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            transaction_cost: Transaction cost
        """
        self.target_vol = target_vol
        self.target_vol_daily = target_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.transaction_cost = transaction_cost

    def compute_weights(
        self,
        predicted_vols: np.ndarray,
    ) -> np.ndarray:
        """
        Compute risk parity weights.

        Args:
            predicted_vols: Predicted volatilities [n_periods, n_assets]

        Returns:
            Weights [n_periods, n_assets]
        """
        predicted_vols = np.asarray(predicted_vols)

        if predicted_vols.ndim == 1:
            predicted_vols = predicted_vols.reshape(-1, 1)

        n_periods, n_assets = predicted_vols.shape

        # Risk parity: weight inversely proportional to volatility
        # Then scale to sum to 1
        weights = 1.0 / np.maximum(predicted_vols, 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Apply constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Renormalize
        weights = weights / weights.sum(axis=1, keepdims=True)

        return weights

    def backtest(
        self,
        returns: np.ndarray,
        predicted_vols: np.ndarray,
    ) -> StrategyResult:
        """
        Backtest risk parity strategy.

        Args:
            returns: Asset returns [n_periods, n_assets]
            predicted_vols: Predicted volatilities [n_periods, n_assets]

        Returns:
            StrategyResult
        """
        returns = np.asarray(returns)
        predicted_vols = np.asarray(predicted_vols)

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        if predicted_vols.ndim == 1:
            predicted_vols = predicted_vols.reshape(-1, 1)

        n_periods = min(len(returns), len(predicted_vols))
        returns = returns[:n_periods]
        predicted_vols = predicted_vols[:n_periods]

        # Compute weights
        weights = self.compute_weights(predicted_vols)

        # Portfolio returns
        portfolio_returns = np.sum(weights * returns, axis=1)

        # Transaction costs
        weight_changes = np.sum(np.abs(np.diff(weights, axis=0, prepend=weights[:1])), axis=1)
        costs = weight_changes * self.transaction_cost

        # Net returns
        net_returns = portfolio_returns - costs

        # Equity curve
        equity_curve = np.cumprod(1 + net_returns)

        # Average weights across periods
        avg_weights = weights.mean(axis=0)

        # Compute metrics
        vol_strategy = VolatilityTargetingStrategy(target_vol=self.target_vol)
        metrics = vol_strategy._compute_metrics(
            net_returns,
            weights.mean(axis=1),  # Use average weight per period
            portfolio_returns,
        )

        return StrategyResult(
            equity_curve=equity_curve,
            returns=net_returns,
            weights=weights,
            dates=None,
            **metrics
        )


def plot_strategy_results(
    strategy_result: StrategyResult,
    benchmark_equity: Optional[np.ndarray] = None,
    title: str = "Volatility Targeting Strategy",
    save_path: Optional[str] = None,
):
    """
    Create visualization of strategy results.

    Args:
        strategy_result: Strategy backtest result
        benchmark_equity: Benchmark equity curve
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    dates = strategy_result.dates
    if dates is None:
        dates = np.arange(len(strategy_result.equity_curve))

    # Panel 1: Equity curves
    ax1 = axes[0]
    ax1.plot(dates, strategy_result.equity_curve, label='Vol-Target Strategy', linewidth=1.5)
    if benchmark_equity is not None:
        ax1.plot(dates, benchmark_equity, label='Benchmark', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Position weights
    ax2 = axes[1]
    ax2.fill_between(dates, strategy_result.weights, alpha=0.5, label='Position Weight')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full Investment')
    ax2.set_ylabel('Weight')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Rolling Sharpe
    ax3 = axes[2]
    rolling_sharpe = pd.Series(strategy_result.returns).rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    ax3.plot(dates, rolling_sharpe, label='Rolling 1Y Sharpe')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Drawdown
    ax4 = axes[3]
    cum_returns = strategy_result.equity_curve
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max * 100
    ax4.fill_between(dates, drawdown, 0, alpha=0.5, color='red')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy plot saved to {save_path}")

    return fig


def generate_strategy_report(
    strategy_result: StrategyResult,
    benchmark_result: Optional[StrategyResult] = None,
) -> str:
    """
    Generate text report of strategy performance.

    Args:
        strategy_result: Strategy backtest result
        benchmark_result: Optional benchmark result

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "VOLATILITY TARGETING STRATEGY REPORT",
        "=" * 60,
        "",
        "PERFORMANCE SUMMARY",
        "-" * 40,
        f"Total Return:        {strategy_result.total_return:>10.2%}",
        f"Annual Return:       {strategy_result.annual_return:>10.2%}",
        f"Annual Volatility:   {strategy_result.annual_volatility:>10.2%}",
        f"Sharpe Ratio:        {strategy_result.sharpe_ratio:>10.2f}",
        f"Sortino Ratio:       {strategy_result.sortino_ratio:>10.2f}",
        f"Max Drawdown:        {strategy_result.max_drawdown:>10.2%}",
        f"Calmar Ratio:        {strategy_result.calmar_ratio:>10.2f}",
        "",
        "STRATEGY CHARACTERISTICS",
        "-" * 40,
        f"Average Leverage:    {strategy_result.avg_leverage:>10.2f}x",
        f"Leverage Std Dev:    {strategy_result.leverage_std:>10.2f}",
        f"Annual Turnover:     {strategy_result.turnover:>10.2%}",
        f"Realized Volatility: {strategy_result.realized_vol:>10.2%}",
        f"Vol Tracking Error:  {strategy_result.vol_tracking_error:>10.2%}",
    ]

    if benchmark_result is not None:
        lines.extend([
            "",
            "BENCHMARK COMPARISON",
            "-" * 40,
            f"Benchmark Sharpe:    {benchmark_result.sharpe_ratio:>10.2f}",
            f"Sharpe Improvement:  {strategy_result.sharpe_ratio - benchmark_result.sharpe_ratio:>10.2f}",
            f"Information Ratio:   {strategy_result.information_ratio or 0:>10.2f}",
            f"Alpha (annual):      {(strategy_result.alpha or 0) * 100:>10.2f}%",
            f"Beta:                {strategy_result.beta or 1:>10.2f}",
        ])

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
