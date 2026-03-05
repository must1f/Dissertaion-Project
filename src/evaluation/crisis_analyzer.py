"""
Crisis Period Performance Analyzer

Provides detailed analysis of strategy performance during crisis periods:
- Predefined historical crises (2008, COVID, etc.)
- Custom crisis period definition
- Recovery analysis
- Comparison with benchmark

References:
    - Taleb, N. N. (2007). "The Black Swan."
    - Reinhart, C. & Rogoff, K. (2009). "This Time Is Different."
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .financial_metrics import FinancialMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Crisis Period Definitions
# ============================================================================

@dataclass
class CrisisDefinition:
    """Definition of a crisis period"""
    name: str
    start_date: str  # YYYY-MM-DD
    end_date: str
    description: str
    category: str  # 'market_crash', 'volatility', 'recession', 'sector'
    severity: str  # 'mild', 'moderate', 'severe'


# Predefined historical crisis periods
HISTORICAL_CRISES = [
    CrisisDefinition(
        name="Asian Financial Crisis",
        start_date="1997-07-01",
        end_date="1998-01-31",
        description="Currency crisis starting in Thailand spreading across Asia",
        category="market_crash",
        severity="severe"
    ),
    CrisisDefinition(
        name="LTCM Crisis",
        start_date="1998-08-01",
        end_date="1998-10-31",
        description="Long-Term Capital Management collapse",
        category="volatility",
        severity="moderate"
    ),
    CrisisDefinition(
        name="Dot-com Crash",
        start_date="2000-03-10",
        end_date="2002-10-09",
        description="Tech bubble burst, NASDAQ fell 78%",
        category="market_crash",
        severity="severe"
    ),
    CrisisDefinition(
        name="9/11 Shock",
        start_date="2001-09-10",
        end_date="2001-09-21",
        description="Market closure and recovery after terrorist attacks",
        category="volatility",
        severity="moderate"
    ),
    CrisisDefinition(
        name="Global Financial Crisis",
        start_date="2007-10-09",
        end_date="2009-03-09",
        description="Subprime mortgage crisis, Lehman collapse",
        category="market_crash",
        severity="severe"
    ),
    CrisisDefinition(
        name="Flash Crash 2010",
        start_date="2010-05-06",
        end_date="2010-05-07",
        description="Dow dropped ~1000 points in minutes",
        category="volatility",
        severity="mild"
    ),
    CrisisDefinition(
        name="European Debt Crisis",
        start_date="2010-04-27",
        end_date="2012-07-26",
        description="Greek debt crisis, Euro stability concerns",
        category="recession",
        severity="moderate"
    ),
    CrisisDefinition(
        name="US Debt Ceiling 2011",
        start_date="2011-07-22",
        end_date="2011-08-22",
        description="S&P downgraded US debt rating",
        category="volatility",
        severity="moderate"
    ),
    CrisisDefinition(
        name="China Devaluation",
        start_date="2015-08-11",
        end_date="2016-02-11",
        description="Yuan devaluation, China growth concerns",
        category="market_crash",
        severity="moderate"
    ),
    CrisisDefinition(
        name="Brexit Shock",
        start_date="2016-06-23",
        end_date="2016-06-27",
        description="UK votes to leave EU",
        category="volatility",
        severity="mild"
    ),
    CrisisDefinition(
        name="Volmageddon",
        start_date="2018-02-02",
        end_date="2018-02-09",
        description="VIX spike, volatility ETF collapse",
        category="volatility",
        severity="moderate"
    ),
    CrisisDefinition(
        name="Q4 2018 Selloff",
        start_date="2018-10-03",
        end_date="2018-12-24",
        description="Fed tightening fears, trade war concerns",
        category="market_crash",
        severity="moderate"
    ),
    CrisisDefinition(
        name="COVID-19 Crash",
        start_date="2020-02-19",
        end_date="2020-03-23",
        description="Pandemic-induced fastest bear market in history",
        category="market_crash",
        severity="severe"
    ),
    CrisisDefinition(
        name="2022 Inflation Crisis",
        start_date="2022-01-03",
        end_date="2022-10-12",
        description="Fed tightening, inflation, tech selloff",
        category="market_crash",
        severity="moderate"
    ),
    CrisisDefinition(
        name="SVB Banking Crisis",
        start_date="2023-03-08",
        end_date="2023-03-20",
        description="Silicon Valley Bank collapse, regional bank stress",
        category="sector",
        severity="moderate"
    ),
]


@dataclass
class CrisisAnalysisResult:
    """Results from analyzing a single crisis period"""
    crisis: CrisisDefinition
    in_sample: bool  # Whether crisis falls within the data period
    duration_days: int
    strategy_return: float
    benchmark_return: Optional[float]
    alpha: Optional[float]  # strategy - benchmark
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    worst_day: float
    best_day: float
    days_to_recovery: Optional[int]
    recovery_date: Optional[datetime]


@dataclass
class CrisisComparisonResult:
    """Comparison of strategy vs benchmark across crises"""
    total_crises_analyzed: int
    crises_outperformed: int  # Where strategy > benchmark
    avg_alpha: float
    avg_strategy_return: float
    avg_benchmark_return: float
    avg_max_drawdown: float
    worst_crisis: str
    best_crisis: str


# ============================================================================
# Crisis Analyzer
# ============================================================================

class CrisisAnalyzer:
    """
    Analyze strategy performance during crisis periods.

    Usage:
        analyzer = CrisisAnalyzer()
        results = analyzer.analyze(strategy_returns, benchmark_returns, dates)
    """

    def __init__(
        self,
        crisis_definitions: Optional[List[CrisisDefinition]] = None,
        include_predefined: bool = True,
    ):
        """
        Initialize crisis analyzer.

        Args:
            crisis_definitions: Custom crisis definitions
            include_predefined: Include predefined historical crises
        """
        self.crises: List[CrisisDefinition] = []

        if include_predefined:
            self.crises.extend(HISTORICAL_CRISES)

        if crisis_definitions:
            self.crises.extend(crisis_definitions)

        logger.info(f"CrisisAnalyzer initialized with {len(self.crises)} crisis periods")

    def add_crisis(self, crisis: CrisisDefinition):
        """Add a custom crisis definition."""
        self.crises.append(crisis)

    def analyze(
        self,
        returns: Union[np.ndarray, pd.Series],
        timestamps: pd.DatetimeIndex,
        benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
        include_recovery: bool = True,
    ) -> List[CrisisAnalysisResult]:
        """
        Analyze strategy performance during all defined crisis periods.

        Args:
            returns: Strategy returns
            timestamps: Return timestamps
            benchmark_returns: Optional benchmark returns
            include_recovery: Calculate recovery times

        Returns:
            List of CrisisAnalysisResult for each crisis
        """
        # Convert to pandas for date indexing
        if isinstance(returns, np.ndarray):
            returns_series = pd.Series(returns, index=timestamps)
        else:
            returns_series = returns

        if benchmark_returns is not None:
            if isinstance(benchmark_returns, np.ndarray):
                bench_series = pd.Series(benchmark_returns, index=timestamps)
            else:
                bench_series = benchmark_returns
        else:
            bench_series = None

        results = []

        for crisis in self.crises:
            result = self._analyze_single_crisis(
                crisis=crisis,
                returns=returns_series,
                benchmark=bench_series,
                include_recovery=include_recovery,
            )
            results.append(result)

        return results

    def _analyze_single_crisis(
        self,
        crisis: CrisisDefinition,
        returns: pd.Series,
        benchmark: Optional[pd.Series],
        include_recovery: bool,
    ) -> CrisisAnalysisResult:
        """Analyze a single crisis period."""
        try:
            start = pd.to_datetime(crisis.start_date)
            end = pd.to_datetime(crisis.end_date)
        except (ValueError, TypeError, pd.errors.ParserError) as e:
            logger.debug(f"Failed to parse crisis dates: {e}")
            return self._empty_result(crisis, False)

        # Check if crisis falls within data period
        data_start = returns.index.min()
        data_end = returns.index.max()

        if end < data_start or start > data_end:
            return self._empty_result(crisis, False)

        # Filter to crisis period
        mask = (returns.index >= start) & (returns.index <= end)
        crisis_returns = returns[mask]

        if len(crisis_returns) < 3:
            return self._empty_result(crisis, True)

        # Calculate strategy metrics
        strategy_return = float(np.prod(1 + crisis_returns) - 1)
        max_dd = FinancialMetrics.max_drawdown(crisis_returns.values)
        sharpe = FinancialMetrics.sharpe_ratio(crisis_returns.values)
        sortino = FinancialMetrics.sortino_ratio(crisis_returns.values)
        volatility = float(np.std(crisis_returns.values, ddof=1) * np.sqrt(252))
        worst_day = float(crisis_returns.min())
        best_day = float(crisis_returns.max())

        # Benchmark comparison
        benchmark_return = None
        alpha = None

        if benchmark is not None:
            bench_crisis = benchmark[mask]
            if len(bench_crisis) > 0:
                benchmark_return = float(np.prod(1 + bench_crisis) - 1)
                alpha = strategy_return - benchmark_return

        # Recovery analysis
        days_to_recovery = None
        recovery_date = None

        if include_recovery and end < data_end:
            days_to_recovery, recovery_date = self._calculate_recovery(
                returns, end, crisis_returns
            )

        return CrisisAnalysisResult(
            crisis=crisis,
            in_sample=True,
            duration_days=(end - start).days,
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            volatility=volatility,
            worst_day=worst_day,
            best_day=best_day,
            days_to_recovery=days_to_recovery,
            recovery_date=recovery_date,
        )

    def _empty_result(
        self,
        crisis: CrisisDefinition,
        in_sample: bool
    ) -> CrisisAnalysisResult:
        """Create empty result for out-of-sample crisis."""
        return CrisisAnalysisResult(
            crisis=crisis,
            in_sample=in_sample,
            duration_days=0,
            strategy_return=np.nan,
            benchmark_return=None,
            alpha=None,
            max_drawdown=np.nan,
            sharpe_ratio=np.nan,
            sortino_ratio=np.nan,
            volatility=np.nan,
            worst_day=np.nan,
            best_day=np.nan,
            days_to_recovery=None,
            recovery_date=None,
        )

    def _calculate_recovery(
        self,
        returns: pd.Series,
        crisis_end: datetime,
        crisis_returns: pd.Series,
        max_days: int = 500
    ) -> Tuple[Optional[int], Optional[datetime]]:
        """Calculate days to recover to pre-crisis level."""
        # Get post-crisis returns
        post_crisis = returns[returns.index > crisis_end]

        if len(post_crisis) == 0:
            return None, None

        # Calculate cumulative loss during crisis
        crisis_cum_return = np.prod(1 + crisis_returns) - 1

        # Need to recover this loss
        target_recovery = -crisis_cum_return / (1 + crisis_cum_return)

        # Calculate post-crisis cumulative returns
        post_cum = np.cumprod(1 + post_crisis.values) - 1

        # Find first day where we exceed target
        recovery_mask = post_cum >= target_recovery

        if np.any(recovery_mask):
            recovery_idx = np.argmax(recovery_mask)
            days_to_recovery = recovery_idx + 1
            recovery_date = post_crisis.index[recovery_idx]
            return days_to_recovery, recovery_date

        return None, None

    def compare_vs_benchmark(
        self,
        results: List[CrisisAnalysisResult]
    ) -> CrisisComparisonResult:
        """
        Generate comparison summary vs benchmark.

        Args:
            results: List of crisis analysis results

        Returns:
            CrisisComparisonResult with aggregated statistics
        """
        # Filter to in-sample crises with valid data
        valid_results = [
            r for r in results
            if r.in_sample and not np.isnan(r.strategy_return)
        ]

        if len(valid_results) == 0:
            return CrisisComparisonResult(
                total_crises_analyzed=0,
                crises_outperformed=0,
                avg_alpha=0,
                avg_strategy_return=0,
                avg_benchmark_return=0,
                avg_max_drawdown=0,
                worst_crisis="N/A",
                best_crisis="N/A",
            )

        # Calculate statistics
        strategy_returns = [r.strategy_return for r in valid_results]
        benchmark_returns = [r.benchmark_return for r in valid_results if r.benchmark_return is not None]
        alphas = [r.alpha for r in valid_results if r.alpha is not None]
        drawdowns = [r.max_drawdown for r in valid_results]

        # Find best/worst
        worst_idx = np.argmin(strategy_returns)
        best_idx = np.argmax(strategy_returns)

        crises_outperformed = sum(
            1 for r in valid_results
            if r.alpha is not None and r.alpha > 0
        )

        return CrisisComparisonResult(
            total_crises_analyzed=len(valid_results),
            crises_outperformed=crises_outperformed,
            avg_alpha=float(np.mean(alphas)) if alphas else 0,
            avg_strategy_return=float(np.mean(strategy_returns)),
            avg_benchmark_return=float(np.mean(benchmark_returns)) if benchmark_returns else 0,
            avg_max_drawdown=float(np.mean(drawdowns)),
            worst_crisis=valid_results[worst_idx].crisis.name,
            best_crisis=valid_results[best_idx].crisis.name,
        )

    def get_crises_by_category(
        self,
        category: str
    ) -> List[CrisisDefinition]:
        """Get crises filtered by category."""
        return [c for c in self.crises if c.category == category]

    def get_crises_by_severity(
        self,
        severity: str
    ) -> List[CrisisDefinition]:
        """Get crises filtered by severity."""
        return [c for c in self.crises if c.severity == severity]


def crisis_results_to_dataframe(
    results: List[CrisisAnalysisResult]
) -> pd.DataFrame:
    """Convert crisis analysis results to DataFrame."""
    records = []

    for r in results:
        if not r.in_sample:
            continue

        records.append({
            'Crisis': r.crisis.name,
            'Category': r.crisis.category,
            'Severity': r.crisis.severity,
            'Duration': f"{r.duration_days}d",
            'Return': f"{r.strategy_return:.1%}" if not np.isnan(r.strategy_return) else "N/A",
            'Max DD': f"{r.max_drawdown:.1%}" if not np.isnan(r.max_drawdown) else "N/A",
            'Sharpe': f"{r.sharpe_ratio:.2f}" if not np.isnan(r.sharpe_ratio) else "N/A",
            'Alpha': f"{r.alpha:.1%}" if r.alpha is not None else "N/A",
            'Recovery': f"{r.days_to_recovery}d" if r.days_to_recovery else "N/A",
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Crisis Analyzer Demo")
    print("=" * 60)

    # Generate synthetic returns covering 2015-2024
    np.random.seed(42)

    dates = pd.date_range(start='2015-01-01', end='2024-01-01', freq='B')
    n_days = len(dates)

    # Base returns with some regime structure
    base_returns = np.random.randn(n_days) * 0.01 + 0.0002

    # Add crisis-like behavior
    for crisis in HISTORICAL_CRISES:
        try:
            start = pd.to_datetime(crisis.start_date)
            end = pd.to_datetime(crisis.end_date)
            mask = (dates >= start) & (dates <= end)
            if np.sum(mask) > 0:
                # Higher volatility and negative drift during crises
                base_returns[mask] = np.random.randn(np.sum(mask)) * 0.03 - 0.001
        except (ValueError, TypeError, pd.errors.ParserError) as e:
            logger.debug(f"Failed to parse crisis dates for synthetic data: {e}")

    returns = pd.Series(base_returns, index=dates)

    # Generate benchmark (S&P 500 proxy)
    benchmark = np.random.randn(n_days) * 0.012 + 0.0003
    benchmark = pd.Series(benchmark, index=dates)

    print(f"\nGenerated {n_days} days of returns (2015-2024)")

    # Run analysis
    analyzer = CrisisAnalyzer()
    results = analyzer.analyze(
        returns=returns,
        timestamps=dates,
        benchmark_returns=benchmark,
    )

    # Show results
    print("\n" + "-" * 40)
    print("Crisis Performance Analysis:")
    df = crisis_results_to_dataframe(results)
    print(df.to_string(index=False))

    # Summary
    print("\n" + "-" * 40)
    print("Benchmark Comparison:")
    comparison = analyzer.compare_vs_benchmark(results)
    print(f"  Total crises analyzed: {comparison.total_crises_analyzed}")
    print(f"  Crises outperformed: {comparison.crises_outperformed}")
    print(f"  Average alpha: {comparison.avg_alpha:.1%}")
    print(f"  Average strategy return: {comparison.avg_strategy_return:.1%}")
    print(f"  Average max drawdown: {comparison.avg_max_drawdown:.1%}")
    print(f"  Worst crisis: {comparison.worst_crisis}")
    print(f"  Best crisis: {comparison.best_crisis}")

    print("\n" + "=" * 60)
    print("Demo complete!")
