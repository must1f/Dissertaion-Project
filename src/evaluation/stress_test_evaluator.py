"""
Stress Test Evaluator

Evaluates model performance during crisis periods and stress events.
Critical for demonstrating model robustness in dissertation.

Features:
- Pre-defined crisis period calendar
- Custom stress window definitions
- Regime-stratified evaluation
- Shock impact analysis
- Tail risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Optional YAML for custom stress windows
try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:  # pragma: no cover
    HAS_YAML = False

from .financial_metrics import FinancialMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CrisisType(Enum):
    """Types of market crises"""
    FINANCIAL = "financial"
    PANDEMIC = "pandemic"
    GEOPOLITICAL = "geopolitical"
    POLICY = "policy"
    SECTOR = "sector"
    FLASH_CRASH = "flash_crash"


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class CrisisPeriod:
    """Definition of a crisis period"""
    name: str
    start_date: datetime
    end_date: datetime
    crisis_type: CrisisType
    description: str
    peak_date: Optional[datetime] = None
    recovery_date: Optional[datetime] = None
    max_drawdown: Optional[float] = None


@dataclass
class StressTestResult:
    """Results for a single stress period"""
    period_name: str
    start_date: datetime
    end_date: datetime
    n_days: int

    # Performance during stress
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # Comparison to baseline
    return_vs_baseline: float
    sharpe_vs_baseline: float
    drawdown_vs_baseline: float

    # Prediction quality
    mse: Optional[float] = None
    mae: Optional[float] = None
    directional_accuracy: Optional[float] = None

    # Risk metrics
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None


@dataclass
class RegimeAnalysis:
    """Analysis for a specific volatility regime"""
    regime: VolatilityRegime
    n_observations: int
    pct_of_total: float

    # Performance
    avg_return: float
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Prediction quality
    mse: Optional[float] = None
    directional_accuracy: Optional[float] = None


@dataclass
class StressTestReport:
    """Complete stress test report"""
    model_name: str
    baseline_metrics: Dict[str, float]
    crisis_results: Dict[str, StressTestResult]
    regime_analysis: Dict[VolatilityRegime, RegimeAnalysis]
    tail_risk_metrics: Dict[str, float]
    recommendations: List[str]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CrisisCalendar:
    """
    Pre-defined calendar of market crisis events.

    Includes major market events for stress testing.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.crises: List[CrisisPeriod] = []
        self._load_default_crises()
        if config_path:
            self._load_from_config(config_path)

    def _load_default_crises(self) -> None:
        """Load default crisis periods"""
        self.crises = [
            # Global Financial Crisis
            CrisisPeriod(
                name="GFC_2008",
                start_date=datetime(2008, 9, 1),
                end_date=datetime(2009, 3, 31),
                crisis_type=CrisisType.FINANCIAL,
                description="Global Financial Crisis - Lehman collapse",
                peak_date=datetime(2008, 10, 10),
                max_drawdown=-0.57
            ),

            # European Debt Crisis
            CrisisPeriod(
                name="Euro_Debt_2011",
                start_date=datetime(2011, 7, 1),
                end_date=datetime(2011, 10, 31),
                crisis_type=CrisisType.FINANCIAL,
                description="European sovereign debt crisis",
                max_drawdown=-0.21
            ),

            # Flash Crash 2010
            CrisisPeriod(
                name="Flash_Crash_2010",
                start_date=datetime(2010, 5, 6),
                end_date=datetime(2010, 5, 7),
                crisis_type=CrisisType.FLASH_CRASH,
                description="Flash Crash - DOW drops 1000 points",
                max_drawdown=-0.09
            ),

            # China Devaluation 2015
            CrisisPeriod(
                name="China_2015",
                start_date=datetime(2015, 8, 11),
                end_date=datetime(2015, 9, 30),
                crisis_type=CrisisType.GEOPOLITICAL,
                description="China yuan devaluation shock",
                max_drawdown=-0.12
            ),

            # COVID-19 Crash
            CrisisPeriod(
                name="COVID_2020",
                start_date=datetime(2020, 2, 20),
                end_date=datetime(2020, 4, 30),
                crisis_type=CrisisType.PANDEMIC,
                description="COVID-19 pandemic market crash",
                peak_date=datetime(2020, 3, 23),
                recovery_date=datetime(2020, 8, 18),
                max_drawdown=-0.34
            ),

            # 2022 Rate Hike Selloff
            CrisisPeriod(
                name="Rate_Hike_2022",
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 10, 12),
                crisis_type=CrisisType.POLICY,
                description="Fed rate hiking cycle selloff",
                max_drawdown=-0.27
            ),

            # SVB Banking Crisis
            CrisisPeriod(
                name="SVB_2023",
                start_date=datetime(2023, 3, 8),
                end_date=datetime(2023, 3, 20),
                crisis_type=CrisisType.FINANCIAL,
                description="Silicon Valley Bank collapse",
                max_drawdown=-0.05
            ),
        ]

    def _load_from_config(self, config_path: Union[str, Path]) -> None:
        """Extend crises from YAML config if present."""
        path = Path(config_path)
        if not path.exists() or not HAS_YAML:
            return
        try:
            data = yaml.safe_load(path.read_text()) or {}
            for item in data.get("windows", []):
                try:
                    self.crises.append(
                        CrisisPeriod(
                            name=item["name"],
                            start_date=datetime.fromisoformat(item["start"]),
                            end_date=datetime.fromisoformat(item["end"]),
                            crisis_type=CrisisType.FINANCIAL,
                            description=f"Config stress window ({item.get('type', 'crisis')})",
                        )
                    )
                except Exception:
                    continue
        except Exception:
            return

    def get_crises_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[CrisisPeriod]:
        """Get crises that overlap with date range"""
        return [
            c for c in self.crises
            if c.start_date <= end and c.end_date >= start
        ]

    def get_by_type(self, crisis_type: CrisisType) -> List[CrisisPeriod]:
        """Get crises by type"""
        return [c for c in self.crises if c.crisis_type == crisis_type]

    def add_crisis(self, crisis: CrisisPeriod) -> None:
        """Add a custom crisis period"""
        self.crises.append(crisis)


class StressTestEvaluator:
    """
    Evaluates model performance during stress periods.

    Provides comprehensive stress testing for dissertation-grade analysis.
    """

    def __init__(
        self,
        vol_thresholds: Tuple[float, float, float, float] = (0.10, 0.15, 0.25, 0.40),
        crisis_calendar: Optional[CrisisCalendar] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize stress test evaluator.

        Args:
            vol_thresholds: (low, normal, elevated, high) boundaries for annualized vol
            crisis_calendar: Optional custom crisis calendar
        """
        self.vol_thresholds = vol_thresholds
        self.crisis_calendar = crisis_calendar or CrisisCalendar(config_path=config_path)

    def classify_volatility_regime(
        self,
        returns: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Classify volatility regime for each observation.

        Args:
            returns: Return series
            window: Rolling window for volatility

        Returns:
            Array of regime labels
        """
        # Calculate rolling volatility (annualized)
        vol = pd.Series(returns).rolling(window, min_periods=5).std() * np.sqrt(252)
        vol = vol.fillna(vol.iloc[window] if len(vol) > window else 0.15).values

        regimes = np.array([VolatilityRegime.NORMAL.value] * len(returns))

        low, normal, elevated, high = self.vol_thresholds

        regimes[vol < low] = VolatilityRegime.LOW.value
        regimes[(vol >= low) & (vol < normal)] = VolatilityRegime.NORMAL.value
        regimes[(vol >= normal) & (vol < elevated)] = VolatilityRegime.ELEVATED.value
        regimes[(vol >= elevated) & (vol < high)] = VolatilityRegime.HIGH.value
        regimes[vol >= high] = VolatilityRegime.EXTREME.value

        return regimes

    def evaluate_crisis_period(
        self,
        crisis: CrisisPeriod,
        returns: np.ndarray,
        dates: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
        baseline_sharpe: float = 0.0
    ) -> StressTestResult:
        """
        Evaluate performance during a specific crisis period.

        Args:
            crisis: Crisis period definition
            returns: Strategy returns
            dates: Dates
            predictions: Optional predictions
            actuals: Optional actuals
            baseline_sharpe: Baseline Sharpe for comparison

        Returns:
            StressTestResult
        """
        dates = pd.to_datetime(dates)

        # Find observations in crisis period
        mask = (dates >= crisis.start_date) & (dates <= crisis.end_date)
        crisis_returns = returns[mask]

        if len(crisis_returns) == 0:
            logger.warning(f"No data for crisis period: {crisis.name}")
            return StressTestResult(
                period_name=crisis.name,
                start_date=crisis.start_date,
                end_date=crisis.end_date,
                n_days=0,
                total_return=0,
                annualized_return=0,
                volatility=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                return_vs_baseline=0,
                sharpe_vs_baseline=0,
                drawdown_vs_baseline=0
            )

        # Calculate metrics
        total_return = np.prod(1 + crisis_returns) - 1
        n_days = len(crisis_returns)
        ann_factor = 252 / n_days if n_days > 0 else 1
        ann_return = (1 + total_return) ** ann_factor - 1
        volatility = np.std(crisis_returns) * np.sqrt(252)
        sharpe = FinancialMetrics.sharpe_ratio(crisis_returns)
        sortino = FinancialMetrics.sortino_ratio(crisis_returns)
        max_dd = FinancialMetrics.max_drawdown(crisis_returns)

        # Prediction metrics
        mse, mae, dir_acc = None, None, None
        if predictions is not None and actuals is not None:
            pred_crisis = predictions[mask]
            act_crisis = actuals[mask]
            if len(pred_crisis) > 0:
                mse = float(np.mean((pred_crisis - act_crisis) ** 2))
                mae = float(np.mean(np.abs(pred_crisis - act_crisis)))
                pred_dir = np.sign(np.diff(pred_crisis))
                act_dir = np.sign(np.diff(act_crisis))
                dir_acc = float(np.mean(pred_dir == act_dir)) if len(pred_dir) > 0 else None

        # VaR and CVaR
        var_95 = np.percentile(crisis_returns, 5) if len(crisis_returns) > 5 else None
        cvar_95 = np.mean(crisis_returns[crisis_returns <= var_95]) if var_95 is not None else None

        # Baseline comparison
        baseline_return = np.mean(returns) * n_days  # Approximate baseline return
        return_vs_baseline = total_return - baseline_return

        return StressTestResult(
            period_name=crisis.name,
            start_date=crisis.start_date,
            end_date=crisis.end_date,
            n_days=n_days,
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            return_vs_baseline=return_vs_baseline,
            sharpe_vs_baseline=sharpe - baseline_sharpe,
            drawdown_vs_baseline=max_dd - FinancialMetrics.max_drawdown(returns),
            mse=mse,
            mae=mae,
            directional_accuracy=dir_acc,
            var_95=var_95,
            cvar_95=cvar_95
        )

    def evaluate_by_regime(
        self,
        returns: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None
    ) -> Dict[VolatilityRegime, RegimeAnalysis]:
        """
        Evaluate performance by volatility regime.

        Args:
            returns: Strategy returns
            predictions: Optional predictions
            actuals: Optional actuals

        Returns:
            Dictionary of regime -> analysis
        """
        regimes = self.classify_volatility_regime(returns)
        total_obs = len(returns)

        results = {}

        for regime in VolatilityRegime:
            mask = regimes == regime.value
            regime_returns = returns[mask]

            if len(regime_returns) == 0:
                continue

            # Calculate metrics
            avg_return = np.mean(regime_returns)
            total_return = np.prod(1 + regime_returns) - 1
            volatility = np.std(regime_returns) * np.sqrt(252)
            sharpe = FinancialMetrics.sharpe_ratio(regime_returns)
            max_dd = FinancialMetrics.max_drawdown(regime_returns)

            # Prediction metrics
            mse, dir_acc = None, None
            if predictions is not None and actuals is not None:
                pred_regime = predictions[mask]
                act_regime = actuals[mask]
                if len(pred_regime) > 0:
                    mse = float(np.mean((pred_regime - act_regime) ** 2))
                    pred_dir = np.sign(np.diff(pred_regime))
                    act_dir = np.sign(np.diff(act_regime))
                    dir_acc = float(np.mean(pred_dir == act_dir)) if len(pred_dir) > 0 else None

            results[regime] = RegimeAnalysis(
                regime=regime,
                n_observations=len(regime_returns),
                pct_of_total=len(regime_returns) / total_obs,
                avg_return=avg_return,
                total_return=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                mse=mse,
                directional_accuracy=dir_acc
            )

        return results

    def compute_tail_risk_metrics(
        self,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute tail risk metrics.

        Args:
            returns: Return series

        Returns:
            Dictionary of tail risk metrics
        """
        return {
            'var_95': float(np.percentile(returns, 5)),
            'var_99': float(np.percentile(returns, 1)),
            'cvar_95': float(np.mean(returns[returns <= np.percentile(returns, 5)])),
            'cvar_99': float(np.mean(returns[returns <= np.percentile(returns, 1)])),
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurtosis()),
            'worst_day': float(np.min(returns)),
            'worst_week': float(pd.Series(returns).rolling(5).sum().min()),
            'worst_month': float(pd.Series(returns).rolling(21).sum().min()),
            'tail_ratio': float(
                np.abs(np.percentile(returns, 95)) /
                np.abs(np.percentile(returns, 5))
            ) if np.percentile(returns, 5) != 0 else 0,
        }

    def generate_recommendations(
        self,
        crisis_results: Dict[str, StressTestResult],
        regime_analysis: Dict[VolatilityRegime, RegimeAnalysis],
        tail_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []

        # Analyze crisis performance
        worst_crisis = min(
            crisis_results.values(),
            key=lambda x: x.sharpe_ratio
        ) if crisis_results else None

        if worst_crisis and worst_crisis.sharpe_ratio < -0.5:
            recommendations.append(
                f"Poor performance during {worst_crisis.period_name} "
                f"(Sharpe: {worst_crisis.sharpe_ratio:.2f}). "
                f"Consider adding crisis-detection features or regime-switching."
            )

        # Analyze regime performance
        if VolatilityRegime.HIGH in regime_analysis:
            high_vol = regime_analysis[VolatilityRegime.HIGH]
            if high_vol.sharpe_ratio < 0:
                recommendations.append(
                    f"Negative Sharpe ({high_vol.sharpe_ratio:.2f}) in high volatility regime. "
                    f"Model may be overfit to calm markets."
                )

        if VolatilityRegime.EXTREME in regime_analysis:
            extreme_vol = regime_analysis[VolatilityRegime.EXTREME]
            if extreme_vol.pct_of_total > 0.1:
                recommendations.append(
                    f"Significant exposure ({extreme_vol.pct_of_total:.1%}) to extreme volatility. "
                    f"Consider volatility targeting or position sizing."
                )

        # Analyze tail risk
        if tail_metrics.get('kurtosis', 0) > 5:
            recommendations.append(
                f"High kurtosis ({tail_metrics['kurtosis']:.1f}) indicates fat tails. "
                f"Standard risk metrics may underestimate risk."
            )

        if tail_metrics.get('skewness', 0) < -1:
            recommendations.append(
                f"Negative skewness ({tail_metrics['skewness']:.2f}) indicates "
                f"asymmetric downside risk."
            )

        if not recommendations:
            recommendations.append(
                "Model shows reasonable stress resilience. "
                "Continue monitoring during live trading."
            )

        return recommendations

    def run_stress_tests(
        self,
        returns: np.ndarray,
        dates: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
        model_name: str = "Unknown"
    ) -> StressTestReport:
        """
        Run comprehensive stress tests.

        Args:
            returns: Strategy returns
            dates: Dates
            predictions: Optional predictions
            actuals: Optional actuals
            model_name: Model name

        Returns:
            StressTestReport
        """
        dates = pd.to_datetime(dates)
        data_start = dates.min()
        data_end = dates.max()

        # Baseline metrics
        baseline_metrics = {
            'total_return': float(np.prod(1 + returns) - 1),
            'sharpe_ratio': float(FinancialMetrics.sharpe_ratio(returns)),
            'sortino_ratio': float(FinancialMetrics.sortino_ratio(returns)),
            'max_drawdown': float(FinancialMetrics.max_drawdown(returns)),
            'volatility': float(np.std(returns) * np.sqrt(252))
        }

        baseline_sharpe = baseline_metrics['sharpe_ratio']

        # Evaluate crisis periods
        applicable_crises = self.crisis_calendar.get_crises_in_range(data_start, data_end)
        crisis_results = {}

        for crisis in applicable_crises:
            result = self.evaluate_crisis_period(
                crisis, returns, dates, predictions, actuals, baseline_sharpe
            )
            if result.n_days > 0:
                crisis_results[crisis.name] = result

        # Evaluate by regime
        regime_analysis = self.evaluate_by_regime(returns, predictions, actuals)

        # Compute tail risk
        tail_metrics = self.compute_tail_risk_metrics(returns)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            crisis_results, regime_analysis, tail_metrics
        )

        return StressTestReport(
            model_name=model_name,
            baseline_metrics=baseline_metrics,
            crisis_results=crisis_results,
            regime_analysis=regime_analysis,
            tail_risk_metrics=tail_metrics,
            recommendations=recommendations
        )

    def to_dataframe(self, report: StressTestReport) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert stress test report to DataFrames.

        Returns:
            Tuple of (crisis_df, regime_df)
        """
        # Crisis results
        crisis_records = []
        for name, result in report.crisis_results.items():
            crisis_records.append({
                'Period': name,
                'Days': result.n_days,
                'Return': result.total_return,
                'Sharpe': result.sharpe_ratio,
                'Max DD': result.max_drawdown,
                'vs Baseline': result.sharpe_vs_baseline,
                'Dir Acc': result.directional_accuracy
            })
        crisis_df = pd.DataFrame(crisis_records)

        # Regime analysis
        regime_records = []
        for regime, analysis in report.regime_analysis.items():
            regime_records.append({
                'Regime': regime.value,
                'Observations': analysis.n_observations,
                '% of Total': analysis.pct_of_total,
                'Sharpe': analysis.sharpe_ratio,
                'Return': analysis.total_return,
                'Max DD': analysis.max_drawdown,
                'Dir Acc': analysis.directional_accuracy
            })
        regime_df = pd.DataFrame(regime_records)

        return crisis_df, regime_df


def run_stress_tests(
    returns: np.ndarray,
    dates: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    model_name: str = "Unknown"
) -> StressTestReport:
    """
    Convenience function for stress testing.

    Args:
        returns: Strategy returns
        dates: Dates
        predictions: Optional predictions
        actuals: Optional actuals
        model_name: Model name

    Returns:
        StressTestReport
    """
    evaluator = StressTestEvaluator()
    return evaluator.run_stress_tests(
        returns, dates, predictions, actuals, model_name
    )
