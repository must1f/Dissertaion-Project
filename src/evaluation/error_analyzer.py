"""
Error Analysis Module

Provides comprehensive error analysis for understanding model failure modes:
- Regime-stratified error analysis
- Event-based error analysis (earnings, Fed, crises)
- Residual vs forecast error correlation
- Temporal error patterns
- Feature importance for errors

Critical for dissertation insights into when/why models fail.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ErrorMetric(Enum):
    """Error metrics for analysis"""
    ABSOLUTE = "absolute"
    SQUARED = "squared"
    PERCENTAGE = "percentage"
    DIRECTIONAL = "directional"


class MarketRegime(Enum):
    """Market regime classifications"""
    LOW_VOL = "low_volatility"
    MEDIUM_VOL = "medium_volatility"
    HIGH_VOL = "high_volatility"
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways"


@dataclass
class ErrorStatistics:
    """Statistics about forecast errors"""
    mean: float
    std: float
    median: float
    mad: float  # Median absolute deviation
    skewness: float
    kurtosis: float
    min_error: float
    max_error: float
    percentile_5: float
    percentile_95: float
    n_samples: int


@dataclass
class RegimeErrorAnalysis:
    """Error analysis for a specific regime"""
    regime: str
    error_stats: ErrorStatistics
    n_observations: int
    pct_of_total: float
    worst_errors: List[Tuple[datetime, float]]  # Top 5 worst errors
    best_predictions: List[Tuple[datetime, float]]  # Top 5 best


@dataclass
class EventErrorAnalysis:
    """Error analysis around specific events"""
    event_type: str
    pre_event_error: float  # Average error before event
    event_error: float  # Average error during event
    post_event_error: float  # Average error after event
    error_spike_ratio: float  # event_error / baseline_error
    n_events: int
    significant: bool  # Statistically significant difference


@dataclass
class ErrorReport:
    """Complete error analysis report"""
    overall_stats: ErrorStatistics
    regime_analysis: Dict[str, RegimeErrorAnalysis]
    event_analysis: Dict[str, EventErrorAnalysis]
    temporal_patterns: Dict[str, Any]
    residual_correlation: Optional[Dict[str, float]]
    recommendations: List[str]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ErrorAnalyzer:
    """
    Comprehensive error analysis for forecast evaluation.

    Analyzes errors across:
    - Market regimes (volatility, trend)
    - Events (earnings, Fed, crises)
    - Time patterns (day of week, month)
    - Physics residual correlation
    """

    def __init__(
        self,
        vol_thresholds: Tuple[float, float] = (0.15, 0.25),
        trend_window: int = 20,
        event_window: int = 5
    ):
        """
        Initialize error analyzer.

        Args:
            vol_thresholds: (low_high_boundary, high_boundary) for vol regimes
            trend_window: Window for trend detection
            event_window: Days before/after event for analysis
        """
        self.vol_thresholds = vol_thresholds
        self.trend_window = trend_window
        self.event_window = event_window

        # Known crisis periods
        self.crisis_periods = [
            ("GFC", datetime(2008, 9, 1), datetime(2009, 3, 31)),
            ("COVID", datetime(2020, 2, 20), datetime(2020, 4, 30)),
            ("2022_Selloff", datetime(2022, 1, 1), datetime(2022, 10, 31)),
        ]

    def compute_error_statistics(
        self,
        errors: np.ndarray
    ) -> ErrorStatistics:
        """
        Compute comprehensive error statistics.

        Args:
            errors: Array of prediction errors

        Returns:
            ErrorStatistics object
        """
        errors = np.asarray(errors).flatten()
        errors = errors[~np.isnan(errors)]

        if len(errors) == 0:
            return ErrorStatistics(
                mean=np.nan, std=np.nan, median=np.nan, mad=np.nan,
                skewness=np.nan, kurtosis=np.nan, min_error=np.nan,
                max_error=np.nan, percentile_5=np.nan, percentile_95=np.nan,
                n_samples=0
            )

        return ErrorStatistics(
            mean=np.mean(errors),
            std=np.std(errors),
            median=np.median(errors),
            mad=np.median(np.abs(errors - np.median(errors))),
            skewness=stats.skew(errors) if len(errors) > 2 else 0,
            kurtosis=stats.kurtosis(errors) if len(errors) > 3 else 0,
            min_error=np.min(errors),
            max_error=np.max(errors),
            percentile_5=np.percentile(errors, 5),
            percentile_95=np.percentile(errors, 95),
            n_samples=len(errors)
        )

    def detect_volatility_regime(
        self,
        returns: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Detect volatility regime for each observation.

        Args:
            returns: Array of returns
            window: Rolling window for volatility

        Returns:
            Array of regime labels
        """
        # Calculate rolling volatility (annualized)
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)

        regimes = np.array([MarketRegime.MEDIUM_VOL.value] * len(returns))

        regimes[rolling_vol < self.vol_thresholds[0]] = MarketRegime.LOW_VOL.value
        regimes[rolling_vol > self.vol_thresholds[1]] = MarketRegime.HIGH_VOL.value

        return regimes

    def detect_trend_regime(
        self,
        prices: np.ndarray,
        window: int = None
    ) -> np.ndarray:
        """
        Detect trend regime (bull/bear/sideways).

        Args:
            prices: Price array
            window: Trend detection window

        Returns:
            Array of regime labels
        """
        window = window or self.trend_window

        # Calculate cumulative return over window
        returns = pd.Series(prices).pct_change(window)

        regimes = np.array([MarketRegime.SIDEWAYS.value] * len(prices))

        regimes[returns > 0.05] = MarketRegime.BULL.value
        regimes[returns < -0.05] = MarketRegime.BEAR.value

        return regimes

    def analyze_by_regime(
        self,
        errors: np.ndarray,
        regimes: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> Dict[str, RegimeErrorAnalysis]:
        """
        Analyze errors stratified by regime.

        Args:
            errors: Prediction errors
            regimes: Regime labels for each error
            dates: Optional dates for worst error identification

        Returns:
            Dictionary of regime -> RegimeErrorAnalysis
        """
        results = {}
        total_samples = len(errors)

        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            mask = regimes == regime
            regime_errors = errors[mask]

            if len(regime_errors) == 0:
                continue

            # Get worst and best predictions
            abs_errors = np.abs(regime_errors)
            worst_indices = np.argsort(abs_errors)[-5:][::-1]
            best_indices = np.argsort(abs_errors)[:5]

            if dates is not None:
                regime_dates = dates[mask]
                worst_errors = [(regime_dates[i], regime_errors[i]) for i in worst_indices]
                best_predictions = [(regime_dates[i], regime_errors[i]) for i in best_indices]
            else:
                worst_errors = [(None, regime_errors[i]) for i in worst_indices]
                best_predictions = [(None, regime_errors[i]) for i in best_indices]

            results[regime] = RegimeErrorAnalysis(
                regime=regime,
                error_stats=self.compute_error_statistics(regime_errors),
                n_observations=len(regime_errors),
                pct_of_total=len(regime_errors) / total_samples,
                worst_errors=worst_errors,
                best_predictions=best_predictions
            )

        return results

    def analyze_by_event(
        self,
        errors: np.ndarray,
        dates: np.ndarray,
        event_dates: Dict[str, List[datetime]],
        window: int = None
    ) -> Dict[str, EventErrorAnalysis]:
        """
        Analyze errors around specific events.

        Args:
            errors: Prediction errors
            dates: Dates for each error
            event_dates: Dictionary of event_type -> list of event dates
            window: Days before/after to analyze

        Returns:
            Dictionary of event_type -> EventErrorAnalysis
        """
        window = window or self.event_window
        dates = pd.to_datetime(dates)
        baseline_error = np.mean(np.abs(errors))

        results = {}

        for event_type, events in event_dates.items():
            pre_errors = []
            event_errors = []
            post_errors = []

            for event_date in events:
                event_date = pd.to_datetime(event_date)

                # Pre-event window
                pre_mask = (dates >= event_date - timedelta(days=window)) & \
                           (dates < event_date)
                pre_errors.extend(errors[pre_mask])

                # Event day
                event_mask = (dates >= event_date) & \
                            (dates <= event_date + timedelta(days=1))
                event_errors.extend(errors[event_mask])

                # Post-event window
                post_mask = (dates > event_date + timedelta(days=1)) & \
                           (dates <= event_date + timedelta(days=window + 1))
                post_errors.extend(errors[post_mask])

            if not event_errors:
                continue

            pre_error = np.mean(np.abs(pre_errors)) if pre_errors else baseline_error
            event_error = np.mean(np.abs(event_errors))
            post_error = np.mean(np.abs(post_errors)) if post_errors else baseline_error

            # Test significance
            if len(event_errors) > 2 and len(pre_errors) > 2:
                _, p_value = stats.mannwhitneyu(
                    np.abs(event_errors),
                    np.abs(pre_errors),
                    alternative='greater'
                )
                significant = p_value < 0.05
            else:
                significant = False

            results[event_type] = EventErrorAnalysis(
                event_type=event_type,
                pre_event_error=pre_error,
                event_error=event_error,
                post_event_error=post_error,
                error_spike_ratio=event_error / baseline_error if baseline_error > 0 else 1.0,
                n_events=len(events),
                significant=significant
            )

        return results

    def analyze_temporal_patterns(
        self,
        errors: np.ndarray,
        dates: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in errors.

        Args:
            errors: Prediction errors
            dates: Dates for each error

        Returns:
            Dictionary of temporal patterns
        """
        df = pd.DataFrame({
            'error': np.abs(errors),
            'date': pd.to_datetime(dates)
        })
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year

        patterns = {}

        # Day of week analysis
        dow_errors = df.groupby('day_of_week')['error'].agg(['mean', 'std', 'count'])
        dow_errors.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(dow_errors)]
        patterns['day_of_week'] = dow_errors.to_dict()

        # Monthly analysis
        monthly_errors = df.groupby('month')['error'].agg(['mean', 'std', 'count'])
        patterns['month'] = monthly_errors.to_dict()

        # Quarterly analysis
        quarterly_errors = df.groupby('quarter')['error'].agg(['mean', 'std', 'count'])
        patterns['quarter'] = quarterly_errors.to_dict()

        # Year over year
        if df['year'].nunique() > 1:
            yearly_errors = df.groupby('year')['error'].agg(['mean', 'std', 'count'])
            patterns['yearly'] = yearly_errors.to_dict()

        # Rolling error trend
        df_sorted = df.sort_values('date')
        rolling_error = df_sorted['error'].rolling(window=20, min_periods=5).mean()
        patterns['error_trend'] = {
            'start': rolling_error.iloc[20] if len(rolling_error) > 20 else rolling_error.iloc[0],
            'end': rolling_error.iloc[-1],
            'trend_direction': 'increasing' if rolling_error.iloc[-1] > rolling_error.iloc[20] else 'decreasing'
        } if len(rolling_error) > 20 else {}

        return patterns

    def analyze_residual_correlation(
        self,
        forecast_errors: np.ndarray,
        pde_residuals: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Analyze correlation between PDE residuals and forecast errors.

        Args:
            forecast_errors: Forecast prediction errors
            pde_residuals: Dictionary of PDE name -> residual values

        Returns:
            Dictionary of PDE name -> correlation coefficient
        """
        correlations = {}

        for pde_name, residuals in pde_residuals.items():
            # Ensure same length
            min_len = min(len(forecast_errors), len(residuals))
            errors = forecast_errors[:min_len]
            resids = residuals[:min_len]

            # Remove NaN values
            mask = ~(np.isnan(errors) | np.isnan(resids))
            if mask.sum() < 10:
                correlations[pde_name] = np.nan
                continue

            # Compute correlation
            corr, p_value = stats.pearsonr(
                np.abs(errors[mask]),
                np.abs(resids[mask])
            )

            correlations[pde_name] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_correlation(corr)
            }

        return correlations

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(corr)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"

        direction = "positive" if corr > 0 else "negative"
        return f"{strength} {direction}"

    def generate_recommendations(
        self,
        regime_analysis: Dict[str, RegimeErrorAnalysis],
        event_analysis: Dict[str, EventErrorAnalysis],
        temporal_patterns: Dict[str, Any],
        residual_correlation: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Regime-based recommendations
        if regime_analysis:
            # Find worst regime
            worst_regime = max(
                regime_analysis.values(),
                key=lambda x: x.error_stats.mean
            )
            best_regime = min(
                regime_analysis.values(),
                key=lambda x: x.error_stats.mean
            )

            if worst_regime.error_stats.mean > 1.5 * best_regime.error_stats.mean:
                recommendations.append(
                    f"Model struggles in {worst_regime.regime} regime "
                    f"(error {worst_regime.error_stats.mean:.4f} vs {best_regime.error_stats.mean:.4f}). "
                    f"Consider regime-specific models or adaptive strategies."
                )

        # Event-based recommendations
        if event_analysis:
            high_spike_events = [
                e for e in event_analysis.values()
                if e.error_spike_ratio > 1.5 and e.significant
            ]
            if high_spike_events:
                event_names = [e.event_type for e in high_spike_events]
                recommendations.append(
                    f"Significant error spikes around: {', '.join(event_names)}. "
                    f"Consider excluding these periods or using event-aware features."
                )

        # Temporal recommendations
        if temporal_patterns and 'day_of_week' in temporal_patterns:
            dow_means = temporal_patterns['day_of_week'].get('mean', {})
            if dow_means:
                worst_day = max(dow_means.items(), key=lambda x: x[1])
                best_day = min(dow_means.items(), key=lambda x: x[1])
                if worst_day[1] > 1.3 * best_day[1]:
                    recommendations.append(
                        f"Higher errors on {worst_day[0]} ({worst_day[1]:.4f}) vs "
                        f"{best_day[0]} ({best_day[1]:.4f}). Check for calendar effects."
                    )

        # Residual correlation recommendations
        if residual_correlation:
            strong_correlations = [
                (name, data) for name, data in residual_correlation.items()
                if isinstance(data, dict) and abs(data.get('correlation', 0)) > 0.3
            ]
            if strong_correlations:
                for pde_name, data in strong_correlations:
                    recommendations.append(
                        f"PDE '{pde_name}' residuals correlate with errors "
                        f"(r={data['correlation']:.3f}). "
                        f"Better physics compliance may improve predictions."
                    )

        if not recommendations:
            recommendations.append(
                "No major systematic error patterns detected. "
                "Consider hyperparameter tuning for general improvement."
            )

        return recommendations

    def analyze(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: np.ndarray,
        returns: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        pde_residuals: Optional[Dict[str, np.ndarray]] = None,
        event_dates: Optional[Dict[str, List[datetime]]] = None
    ) -> ErrorReport:
        """
        Perform comprehensive error analysis.

        Args:
            predictions: Model predictions
            actuals: Actual values
            dates: Dates for each observation
            returns: Optional returns for volatility regime
            prices: Optional prices for trend regime
            pde_residuals: Optional PDE residuals
            event_dates: Optional event calendar

        Returns:
            Complete ErrorReport
        """
        errors = predictions - actuals
        abs_errors = np.abs(errors)

        # Overall statistics
        overall_stats = self.compute_error_statistics(abs_errors)

        # Regime analysis
        regime_analysis = {}

        if returns is not None:
            vol_regimes = self.detect_volatility_regime(returns)
            regime_analysis.update(
                self.analyze_by_regime(abs_errors, vol_regimes, dates)
            )

        if prices is not None:
            trend_regimes = self.detect_trend_regime(prices)
            regime_analysis.update(
                self.analyze_by_regime(abs_errors, trend_regimes, dates)
            )

        # Event analysis
        event_analysis = {}
        if event_dates:
            event_analysis = self.analyze_by_event(abs_errors, dates, event_dates)

        # Add crisis period analysis
        crisis_events = {}
        for name, start, end in self.crisis_periods:
            crisis_events[f"crisis_{name}"] = [start + timedelta(days=i)
                                                for i in range((end - start).days)]
        if crisis_events:
            event_analysis.update(
                self.analyze_by_event(abs_errors, dates, crisis_events)
            )

        # Temporal patterns
        temporal_patterns = self.analyze_temporal_patterns(abs_errors, dates)

        # Residual correlation
        residual_correlation = None
        if pde_residuals:
            residual_correlation = self.analyze_residual_correlation(
                abs_errors, pde_residuals
            )

        # Generate recommendations
        recommendations = self.generate_recommendations(
            regime_analysis, event_analysis, temporal_patterns, residual_correlation
        )

        return ErrorReport(
            overall_stats=overall_stats,
            regime_analysis=regime_analysis,
            event_analysis=event_analysis,
            temporal_patterns=temporal_patterns,
            residual_correlation=residual_correlation,
            recommendations=recommendations
        )

    def to_dataframe(self, report: ErrorReport) -> pd.DataFrame:
        """
        Convert error report to DataFrame for easy viewing.

        Args:
            report: ErrorReport object

        Returns:
            Summary DataFrame
        """
        rows = []

        # Overall stats
        rows.append({
            'category': 'Overall',
            'subcategory': 'All',
            'mean_error': report.overall_stats.mean,
            'std_error': report.overall_stats.std,
            'median_error': report.overall_stats.median,
            'n_samples': report.overall_stats.n_samples
        })

        # Regime stats
        for regime, analysis in report.regime_analysis.items():
            rows.append({
                'category': 'Regime',
                'subcategory': regime,
                'mean_error': analysis.error_stats.mean,
                'std_error': analysis.error_stats.std,
                'median_error': analysis.error_stats.median,
                'n_samples': analysis.n_observations,
                'pct_of_total': analysis.pct_of_total
            })

        # Event stats
        for event_type, analysis in report.event_analysis.items():
            rows.append({
                'category': 'Event',
                'subcategory': event_type,
                'mean_error': analysis.event_error,
                'error_spike_ratio': analysis.error_spike_ratio,
                'significant': analysis.significant,
                'n_samples': analysis.n_events
            })

        return pd.DataFrame(rows)


def analyze_forecast_errors(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    **kwargs
) -> ErrorReport:
    """
    Convenience function for error analysis.

    Args:
        predictions: Model predictions
        actuals: Actual values
        dates: Observation dates
        **kwargs: Additional arguments for ErrorAnalyzer

    Returns:
        ErrorReport
    """
    analyzer = ErrorAnalyzer()
    return analyzer.analyze(predictions, actuals, dates, **kwargs)
