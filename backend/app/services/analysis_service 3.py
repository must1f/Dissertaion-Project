"""Service for advanced analysis functionality."""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.data_service import DataService
from backend.app.schemas.analysis import (
    RegimeMetrics,
    CurrentRegimeState,
    RegimeLabel,
    RegimeHistoryPoint,
    ExposureSnapshot,
    ExposureHistoryPoint,
    BenchmarkPoint,
    RollingMetricsPoint,
    RobustnessTestResult,
    CrisisPerformance,
)

# Import src modules
try:
    from src.evaluation.regime_detector import (
        get_regime_detector,
        RollingVolatilityDetector,
        RegimeLabel as SrcRegimeLabel,
        load_stress_windows,
        label_stress_windows,
    )
    from src.evaluation.regime_analysis import (
        compute_regime_metrics as src_compute_regime_metrics,
        compute_full_regime_analysis,
    )
    from src.evaluation.financial_metrics import FinancialMetrics
    from src.evaluation.robustness_tests import run_full_robustness_suite
    from src.evaluation.crisis_analyzer import CrisisAnalyzer
    from src.evaluation.stress_test_evaluator import StressTestEvaluator
    from src.trading.exposure_manager import ExposureManager
    HAS_SRC_MODULES = True
except ImportError:
    HAS_SRC_MODULES = False


class AnalysisService:
    """Service for advanced analysis functionality."""

    def __init__(self):
        """Initialize analysis service."""
        self._data_service = DataService()
        self._exposure_manager = None
        self._exposure_history: List[ExposureSnapshot] = []
        self._stress_windows = load_stress_windows(Path("configs/stress_windows.yaml")) if HAS_SRC_MODULES else []

    # =========================================================================
    # Regime Analysis
    # =========================================================================

    def get_current_regime(
        self,
        ticker: str = "^GSPC",
        method: str = "rolling",
        lookback_days: int = 252,
    ) -> CurrentRegimeState:
        """Get current market regime."""
        # Fetch data — convert lookback_days to start_date
        start_date = (datetime.now() - timedelta(days=int(lookback_days * 1.5))).strftime("%Y-%m-%d")
        stock_data = self._data_service.get_stock_data(
            ticker=ticker,
            start_date=start_date,
        )

        if len(stock_data.data) < 50:
            raise ValueError("Insufficient data for regime detection")

        # Calculate returns
        prices = np.array([d.close for d in stock_data.data])
        returns = np.diff(prices) / prices[:-1]

        if not HAS_SRC_MODULES:
            # Fallback: simple volatility-based regime
            vol = np.std(returns[-21:]) * np.sqrt(252)
            if vol < 0.12:
                regime = RegimeLabel.LOW_VOL
            elif vol < 0.25:
                regime = RegimeLabel.NORMAL
            else:
                regime = RegimeLabel.HIGH_VOL

            return CurrentRegimeState(
                regime=regime,
                probability=0.8,
                regime_probabilities={
                    'low_volatility': 0.1 if regime != RegimeLabel.LOW_VOL else 0.8,
                    'normal': 0.1 if regime != RegimeLabel.NORMAL else 0.8,
                    'high_volatility': 0.1 if regime != RegimeLabel.HIGH_VOL else 0.8,
                },
                volatility=vol,
                transition_probability=0.1,
            )

        # Use src module
        detector = get_regime_detector(method=method)
        detector.fit(returns)
        regimes = detector.predict(returns)

        current_regime_id = regimes[-1]
        regime_map = {0: RegimeLabel.LOW_VOL, 1: RegimeLabel.NORMAL, 2: RegimeLabel.HIGH_VOL}
        current_regime = regime_map.get(current_regime_id, RegimeLabel.NORMAL)

        # Calculate volatility
        vol = np.std(returns[-21:]) * np.sqrt(252)

        # Calculate regime probabilities (simplified)
        probs = {
            'low_volatility': float(np.mean(regimes == 0)),
            'normal': float(np.mean(regimes == 1)),
            'high_volatility': float(np.mean(regimes == 2)),
        }

        return CurrentRegimeState(
            regime=current_regime,
            probability=probs[current_regime.value],
            regime_probabilities=probs,
            volatility=vol,
            transition_probability=0.1,  # Simplified
        )

    def get_regime_history(
        self,
        ticker: str = "^GSPC",
        method: str = "rolling",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[List[RegimeHistoryPoint], Dict[str, RegimeMetrics]]:
        """Get regime history and summary metrics."""
        # Fetch data
        stock_data = self._data_service.get_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if len(stock_data.data) < 50:
            return [], {}

        # Prepare data
        prices = np.array([d.close for d in stock_data.data])
        returns = np.diff(prices) / prices[:-1]
        timestamps = [d.timestamp for d in stock_data.data[1:]]

        if not HAS_SRC_MODULES:
            # Simple fallback
            history = []
            rolling_vol = pd.Series(returns).rolling(21).std() * np.sqrt(252)

            for i, (ts, vol) in enumerate(zip(timestamps, rolling_vol)):
                if pd.isna(vol):
                    continue
                if vol < 0.12:
                    regime = RegimeLabel.LOW_VOL
                elif vol < 0.25:
                    regime = RegimeLabel.NORMAL
                else:
                    regime = RegimeLabel.HIGH_VOL

                history.append(RegimeHistoryPoint(
                    timestamp=ts,
                    regime=regime,
                    probability=0.8,
                    volatility=vol,
                ))

            return history, {}

        # Use src module
        analysis = compute_full_regime_analysis(
            returns=returns,
            timestamps=pd.DatetimeIndex(timestamps),
            regime_method=method,
        )

        regimes = analysis['regimes']
        rolling_vol = pd.Series(returns).rolling(21).std() * np.sqrt(252)

        regime_map = {0: RegimeLabel.LOW_VOL, 1: RegimeLabel.NORMAL, 2: RegimeLabel.HIGH_VOL}

        history = []
        for i, (ts, regime_id) in enumerate(zip(timestamps, regimes)):
            vol = rolling_vol.iloc[i] if i < len(rolling_vol) else 0.15
            history.append(RegimeHistoryPoint(
                timestamp=ts,
                regime=regime_map.get(regime_id, RegimeLabel.NORMAL),
                probability=0.8,
                volatility=vol if not pd.isna(vol) else 0.15,
            ))

        # Convert regime metrics
        src_metrics = analysis['regime_metrics']
        regime_metrics = {}
        for name, m in src_metrics.items():
            regime_metrics[name] = RegimeMetrics(
                regime=name,
                count=m.count,
                proportion=m.proportion,
                mean_return=m.mean_return if not np.isnan(m.mean_return) else None,
                volatility=m.volatility if not np.isnan(m.volatility) else None,
                sharpe_ratio=m.sharpe_ratio if not np.isnan(m.sharpe_ratio) else None,
                sortino_ratio=m.sortino_ratio if not np.isnan(m.sortino_ratio) else None,
                max_drawdown=m.max_drawdown if not np.isnan(m.max_drawdown) else None,
                win_rate=m.win_rate if not np.isnan(m.win_rate) else None,
                avg_duration=m.avg_duration if not np.isnan(m.avg_duration) else None,
            )

        # Optional stress window labeling
        if self._stress_windows:
            labels = label_stress_windows(pd.DatetimeIndex(timestamps), self._stress_windows)
            for p, label in zip(history, labels):
                if label:
                    p.stress_window = label  # type: ignore[attr-defined]

        return history, regime_metrics

    # =========================================================================
    # Returns fetcher (for frontend stress testing)
    # =========================================================================

    def get_returns_series(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_points: int = 2000,
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """Fetch daily returns for a ticker within date range, optionally downsampled."""
        stock_data = self._data_service.get_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if len(stock_data.data) < 2:
            raise ValueError("Insufficient data for returns series")

        prices = np.array([d.close for d in stock_data.data])
        returns = np.diff(prices) / prices[:-1]
        timestamps = pd.DatetimeIndex([d.timestamp for d in stock_data.data[1:]])

        if len(returns) > max_points:
            step = len(returns) // max_points + 1
            returns = returns[::step]
            timestamps = timestamps[::step]

        return returns, timestamps

    # =========================================================================
    # Stress Testing
    # =========================================================================

    def run_stress_tests(
        self,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ):
        """Run stress testing for a given return series."""
        if not HAS_SRC_MODULES:
            raise RuntimeError("Stress testing requires src evaluation modules.")
        evaluator = StressTestEvaluator(config_path=Path("configs/stress_windows.yaml"))
        return evaluator.evaluate_returns(returns=returns, timestamps=timestamps)

    # =========================================================================
    # Exposure Analysis
    # =========================================================================

    def get_current_exposure(
        self,
        returns: np.ndarray,
        target_volatility: float = 0.15,
        regime_aware: bool = True,
    ) -> ExposureSnapshot:
        """Calculate current exposure recommendation."""
        if not HAS_SRC_MODULES:
            # Fallback
            vol = np.std(returns[-21:]) * np.sqrt(252) if len(returns) >= 21 else 0.15
            vol_scalar = target_volatility / max(vol, 0.05)
            exposure = min(2.0, max(0.1, vol_scalar))

            return ExposureSnapshot(
                target_exposure=exposure,
                gross_exposure=exposure,
                net_exposure=exposure,
                leverage_ratio=exposure,
                volatility_scalar=vol_scalar,
                regime_scalar=1.0,
                regime='normal',
                realized_volatility=vol,
                target_volatility=target_volatility,
                turnover_cost=0.0,
                position_weights={},
            )

        # Use src module
        if self._exposure_manager is None:
            self._exposure_manager = ExposureManager(
                target_volatility=target_volatility,
                regime_aware=regime_aware,
            )
            self._exposure_manager.calibrate(returns)

        result = self._exposure_manager.calculate_exposure(returns)

        snapshot = ExposureSnapshot(
            target_exposure=result.target_exposure,
            gross_exposure=result.gross_exposure,
            net_exposure=result.net_exposure,
            leverage_ratio=result.leverage_ratio,
            volatility_scalar=result.volatility_scalar,
            regime_scalar=result.regime_scalar,
            regime=result.metadata.get('regime', 'normal'),
            realized_volatility=result.metadata.get('realized_vol', 0.15),
            target_volatility=target_volatility,
            turnover_cost=result.turnover_cost,
            position_weights=result.position_weights,
        )

        self._exposure_history.append(snapshot)

        return snapshot

    def get_exposure_history(self) -> List[ExposureHistoryPoint]:
        """Get exposure calculation history."""
        return [
            ExposureHistoryPoint(
                timestamp=e.timestamp,
                target_exposure=e.target_exposure,
                gross_exposure=e.gross_exposure,
                volatility_scalar=e.volatility_scalar,
                regime=e.regime,
            )
            for e in self._exposure_history
        ]

    # =========================================================================
    # Benchmark Comparison
    # =========================================================================

    def compare_with_benchmark(
        self,
        strategy_returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        benchmark_ticker: str = "^GSPC",
        initial_capital: float = 100000.0,
        include_regimes: bool = True,
    ) -> Dict[str, Any]:
        """Compare strategy performance with benchmark."""
        # Fetch benchmark data
        benchmark_data = self._data_service.get_stock_data(
            ticker=benchmark_ticker,
            start_date=timestamps[0].strftime('%Y-%m-%d'),
            end_date=timestamps[-1].strftime('%Y-%m-%d'),
        )

        # Calculate benchmark returns
        bench_prices = np.array([d.close for d in benchmark_data.data])
        bench_returns = np.diff(bench_prices) / bench_prices[:-1]

        # Align lengths
        min_len = min(len(strategy_returns), len(bench_returns))
        strategy_returns = strategy_returns[:min_len]
        bench_returns = bench_returns[:min_len]
        timestamps = timestamps[:min_len]

        # Calculate cumulative values
        strategy_cum = initial_capital * np.cumprod(1 + strategy_returns)
        bench_cum = initial_capital * np.cumprod(1 + bench_returns)

        # Calculate metrics
        strategy_total = float(strategy_cum[-1] / initial_capital - 1)
        bench_total = float(bench_cum[-1] / initial_capital - 1)

        if HAS_SRC_MODULES:
            strategy_sharpe = FinancialMetrics.sharpe_ratio(strategy_returns)
            bench_sharpe = FinancialMetrics.sharpe_ratio(bench_returns)
            strategy_dd = FinancialMetrics.max_drawdown(strategy_returns)
            bench_dd = FinancialMetrics.max_drawdown(bench_returns)
        else:
            strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            bench_sharpe = np.mean(bench_returns) / np.std(bench_returns) * np.sqrt(252)
            strategy_dd = -0.1
            bench_dd = -0.1

        # Calculate beta and alpha
        correlation = np.corrcoef(strategy_returns, bench_returns)[0, 1]
        beta = np.cov(strategy_returns, bench_returns)[0, 1] / np.var(bench_returns)
        alpha = np.mean(strategy_returns) * 252 - beta * np.mean(bench_returns) * 252

        # Tracking error
        tracking_diff = strategy_returns - bench_returns
        tracking_error = np.std(tracking_diff) * np.sqrt(252)

        # Information ratio
        ir = np.mean(tracking_diff) * 252 / tracking_error if tracking_error > 0 else 0

        # Generate comparison data
        comparison_data = []
        for i, ts in enumerate(timestamps):
            point = BenchmarkPoint(
                timestamp=ts,
                strategy_value=strategy_cum[i],
                benchmark_value=bench_cum[i],
                strategy_return=strategy_returns[i],
                benchmark_return=bench_returns[i],
                alpha=strategy_returns[i] - bench_returns[i],
            )
            comparison_data.append(point)

        return {
            'strategy_final_value': float(strategy_cum[-1]),
            'benchmark_final_value': float(bench_cum[-1]),
            'strategy_total_return': strategy_total,
            'benchmark_total_return': bench_total,
            'alpha': float(alpha),
            'beta': float(beta),
            'correlation': float(correlation),
            'tracking_error': float(tracking_error),
            'information_ratio': float(ir),
            'strategy_sharpe': float(strategy_sharpe),
            'benchmark_sharpe': float(bench_sharpe),
            'strategy_max_dd': float(strategy_dd),
            'benchmark_max_dd': float(bench_dd),
            'comparison_data': comparison_data,
        }

    # =========================================================================
    # Rolling Metrics
    # =========================================================================

    def calculate_rolling_metrics(
        self,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        window: int = 126,
        metrics: List[str] = None,
    ) -> List[RollingMetricsPoint]:
        """Calculate rolling metrics."""
        if metrics is None:
            metrics = ['sharpe', 'volatility', 'sortino']

        returns_series = pd.Series(returns, index=timestamps)
        result_points = []

        for i in range(window, len(returns_series)):
            window_returns = returns_series.iloc[i-window:i].values
            ts = returns_series.index[i]

            point = RollingMetricsPoint(timestamp=ts)

            if HAS_SRC_MODULES:
                if 'sharpe' in metrics:
                    point.sharpe_ratio = FinancialMetrics.sharpe_ratio(window_returns)
                if 'sortino' in metrics:
                    point.sortino_ratio = FinancialMetrics.sortino_ratio(window_returns)
                if 'volatility' in metrics:
                    point.volatility = float(np.std(window_returns, ddof=1) * np.sqrt(252))
                if 'return' in metrics:
                    point.return_value = float(np.mean(window_returns) * 252)
                if 'max_drawdown' in metrics:
                    point.max_drawdown = FinancialMetrics.max_drawdown(window_returns)
            else:
                if 'sharpe' in metrics:
                    point.sharpe_ratio = float(np.mean(window_returns) / np.std(window_returns) * np.sqrt(252))
                if 'volatility' in metrics:
                    point.volatility = float(np.std(window_returns, ddof=1) * np.sqrt(252))

            result_points.append(point)

        return result_points

    # =========================================================================
    # Robustness Testing
    # =========================================================================

    def run_robustness_tests(
        self,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        benchmark_returns: Optional[np.ndarray] = None,
        tests: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run robustness testing suite."""
        if not HAS_SRC_MODULES:
            # Return basic pass/fail
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            return {
                'overall_score': 0.5 if sharpe > 0 else 0.2,
                'tests_passed': 1 if sharpe > 0.3 else 0,
                'tests_failed': 0 if sharpe > 0.3 else 1,
                'is_robust': sharpe > 0.3,
                'test_results': [
                    RobustnessTestResult(
                        test_name='Basic Sharpe',
                        passed=sharpe > 0.3,
                        score=min(1.0, sharpe / 1.5),
                        details={'sharpe': sharpe},
                        recommendations=[] if sharpe > 0.3 else ['Improve risk-adjusted returns'],
                    )
                ],
                'recommendations': [],
            }

        # Use full robustness suite
        report = run_full_robustness_suite(
            returns=returns,
            timestamps=timestamps,
            benchmark_returns=benchmark_returns,
            tests=tests,
        )

        return {
            'overall_score': report.overall_score,
            'tests_passed': report.tests_passed,
            'tests_failed': report.tests_failed,
            'is_robust': report.is_robust,
            'test_results': [
                RobustnessTestResult(
                    test_name=r.test_name,
                    passed=r.passed,
                    score=r.score,
                    details=r.details,
                    recommendations=r.recommendations,
                )
                for r in report.test_results
            ],
            'recommendations': report.recommendations,
        }

    # =========================================================================
    # Crisis Analysis
    # =========================================================================

    def analyze_crisis_performance(
        self,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Analyze strategy performance during crisis periods."""
        if not HAS_SRC_MODULES:
            return {
                'crises_analyzed': 0,
                'crises_outperformed': 0,
                'avg_alpha': None,
                'avg_crisis_return': 0,
                'avg_max_drawdown': 0,
                'worst_crisis': 'N/A',
                'best_crisis': 'N/A',
                'crisis_results': [],
            }

        analyzer = CrisisAnalyzer()
        results = analyzer.analyze(
            returns=pd.Series(returns, index=timestamps),
            timestamps=timestamps,
            benchmark_returns=pd.Series(benchmark_returns, index=timestamps) if benchmark_returns is not None else None,
        )

        # Filter to in-sample crises
        valid_results = [r for r in results if r.in_sample and not np.isnan(r.strategy_return)]

        if not valid_results:
            return {
                'crises_analyzed': 0,
                'crises_outperformed': 0,
                'avg_alpha': None,
                'avg_crisis_return': 0,
                'avg_max_drawdown': 0,
                'worst_crisis': 'N/A',
                'best_crisis': 'N/A',
                'crisis_results': [],
            }

        comparison = analyzer.compare_vs_benchmark(results)

        crisis_performances = []
        for r in valid_results:
            crisis_performances.append(CrisisPerformance(
                crisis_name=r.crisis.name,
                start_date=r.crisis.start_date,
                end_date=r.crisis.end_date,
                duration_days=r.duration_days,
                strategy_return=r.strategy_return,
                benchmark_return=r.benchmark_return,
                alpha=r.alpha,
                max_drawdown=r.max_drawdown,
                sharpe_ratio=r.sharpe_ratio,
                days_to_recovery=r.days_to_recovery,
            ))

        return {
            'crises_analyzed': comparison.total_crises_analyzed,
            'crises_outperformed': comparison.crises_outperformed,
            'avg_alpha': comparison.avg_alpha if comparison.avg_alpha else None,
            'avg_crisis_return': comparison.avg_strategy_return,
            'avg_max_drawdown': comparison.avg_max_drawdown,
            'worst_crisis': comparison.worst_crisis,
            'best_crisis': comparison.best_crisis,
            'crisis_results': crisis_performances,
        }
