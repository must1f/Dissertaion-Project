"""
Evaluation Pipeline — End-to-End Model-Agnostic Evaluation

Chains together:
    Predictions → Signal → Position → Costs → Net Returns → Metrics → Plots

This is the single entry point for dissertation-quality evaluation.
It integrates with the web-app via ``training_service.py`` (called after
training completes) and the existing ``/api/metrics`` endpoints.

Usage::

    from src.evaluation.pipeline import EvaluationPipeline, PipelineConfig

    pipe = EvaluationPipeline(PipelineConfig(strategy="sign"))
    result = pipe.evaluate_model(predictions, actual_returns, "LSTM")

    # Cost sensitivity sweep
    sweep = pipe.run_cost_sensitivity(predictions, actual_returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from .strategy_engine import StrategyEngine, StrategyConfig
from .financial_metrics import FinancialMetrics
from .plot_diagnostics import DiagnosticPlotter
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Full configuration for the evaluation pipeline."""
    # Strategy
    strategy: str = "sign"                # sign | threshold | vol_scaled | probability
    transaction_cost: float = 0.001       # 10 bps
    threshold: float = 0.0005             # for threshold strategy
    w_max: float = 1.0                    # for vol_scaled strategy

    # Financial metrics
    risk_free_rate: float = 0.02          # 2% annual
    periods_per_year: int = 252

    # Plots
    rolling_sharpe_window: int = 63       # ~3 months
    n_quantiles: int = 10
    generate_plots: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/evaluation"))


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""
    model_name: str
    strategy_name: str
    config: Dict[str, Any]

    # Core arrays
    net_returns: np.ndarray
    positions: np.ndarray
    predictions: np.ndarray
    actual_returns: np.ndarray

    # Computed metrics
    metrics: Dict[str, float]
    trading_stats: Dict[str, float]

    # Plot paths
    plot_paths: List[Path] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        """Return a single-row DataFrame of key metrics."""
        row = {
            "Model": self.model_name,
            "Strategy": self.strategy_name,
            "Sharpe": self.metrics.get("sharpe_ratio", 0),
            "Sortino": self.metrics.get("sortino_ratio", 0),
            "Calmar": self.metrics.get("calmar_ratio", 0),
            "Total Return (%)": self.metrics.get("total_return", 0) * 100,
            "Max DD (%)": self.metrics.get("max_drawdown", 0) * 100,
            "Dir. Acc.": self.metrics.get("directional_accuracy", 0),
            "Turnover/day": self.trading_stats.get("avg_daily_turnover", 0),
            "Exposure (%)": self.trading_stats.get("exposure_pct", 0),
        }
        return pd.DataFrame([row])

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary (for API responses)."""
        return {
            "model_name": self.model_name,
            "strategy_name": self.strategy_name,
            "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in self.metrics.items()},
            "trading_stats": self.trading_stats,
            "plot_paths": [str(p) for p in self.plot_paths],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EvaluationPipeline:
    """
    End-to-end evaluation pipeline.

    Accepts a generic predictions vector from any model and applies the same
    signal mapping, lagging rule, cost model, metrics, and plots — ensuring
    fair, model-agnostic comparison.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.plotter = DiagnosticPlotter(
            periods_per_year=self.config.periods_per_year,
            risk_free_rate=self.config.risk_free_rate,
        )

    # === Single-model evaluation ==========================================

    def evaluate_model(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        model_name: str = "Model",
        timestamps: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """Run full evaluation for a single model.

        Parameters
        ----------
        predictions : array
            Model predictions (returns or probabilities).
        actual_returns : array
            Realised simple returns r_t = (P_t - P_{t-1}) / P_{t-1}.
        model_name : str
            Display name for plots and tables.
        timestamps : array, optional
            Date index for x-axes.

        Returns
        -------
        PipelineResult
        """
        predictions = np.asarray(predictions).flatten()
        actual_returns = np.asarray(actual_returns).flatten()

        # 1. Signal → Position → Net Returns
        strat_config = StrategyConfig(
            strategy=self.config.strategy,
            transaction_cost=self.config.transaction_cost,
            threshold=self.config.threshold,
            w_max=self.config.w_max,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.config.periods_per_year,
        )
        net_returns, positions, trading_stats = StrategyEngine.run(
            predictions, actual_returns, strat_config
        )

        # 2. Compute financial metrics from net returns
        metrics = FinancialMetrics.compute_all_metrics(
            returns=net_returns,
            predictions=predictions,
            targets=actual_returns,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.config.periods_per_year,
            predictions_are_returns=True,
        )

        # 3. Generate plots
        plot_paths: List[Path] = []
        if self.config.generate_plots:
            out_dir = self.config.output_dir / model_name.lower().replace(" ", "_")
            plot_paths = self.plotter.plot_all(
                net_returns=net_returns,
                positions=positions,
                predictions=predictions,
                actuals=actual_returns,
                model_name=model_name,
                output_dir=out_dir,
                timestamps=timestamps,
                rolling_window=self.config.rolling_sharpe_window,
                n_quantiles=self.config.n_quantiles,
            )

        result = PipelineResult(
            model_name=model_name,
            strategy_name=self.config.strategy,
            config=asdict(self.config),
            net_returns=net_returns,
            positions=positions,
            predictions=predictions,
            actual_returns=actual_returns,
            metrics=metrics,
            trading_stats=trading_stats,
            plot_paths=plot_paths,
        )

        # Log summary
        logger.info(
            f"[Pipeline] {model_name} ({self.config.strategy}): "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
            f"Sortino={metrics.get('sortino_ratio', 0):.3f}, "
            f"Return={metrics.get('total_return', 0)*100:.1f}%, "
            f"MaxDD={metrics.get('max_drawdown', 0)*100:.1f}%, "
            f"Turnover/day={trading_stats.get('avg_daily_turnover', 0):.4f}"
        )

        return result

    # === Multi-model comparison ===========================================

    def compare_models(
        self,
        model_predictions: Dict[str, np.ndarray],
        actual_returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, PipelineResult]]:
        """Compare multiple models head-to-head.

        Parameters
        ----------
        model_predictions : dict
            {model_name: predictions_array}
        actual_returns : array
            Shared realised returns.

        Returns
        -------
        comparison_df : DataFrame
            Side-by-side metrics table.
        results : dict
            {model_name: PipelineResult}
        """
        results = {}
        rows = []

        for name, preds in model_predictions.items():
            result = self.evaluate_model(preds, actual_returns, name, timestamps)
            results[name] = result
            rows.append(result.summary_table())

        comparison_df = pd.concat(rows, ignore_index=True)
        return comparison_df, results

    # === Sensitivity sweeps ===============================================

    def run_cost_sensitivity(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        costs: Optional[List[float]] = None,
        model_name: str = "Model",
    ) -> pd.DataFrame:
        """Sweep transaction costs and report metrics for each.

        Parameters
        ----------
        costs : list[float]
            Transaction costs to sweep (default: [0, 1bp, 5bp, 10bp, 30bp]).

        Returns
        -------
        DataFrame with one row per cost level.
        """
        if costs is None:
            costs = [0.0, 0.0001, 0.0005, 0.001, 0.003]

        rows = []
        for c in costs:
            cfg = PipelineConfig(
                strategy=self.config.strategy,
                transaction_cost=c,
                threshold=self.config.threshold,
                w_max=self.config.w_max,
                risk_free_rate=self.config.risk_free_rate,
                periods_per_year=self.config.periods_per_year,
                generate_plots=False,  # no plots during sweep
            )
            pipe = EvaluationPipeline(cfg)
            result = pipe.evaluate_model(predictions, actual_returns, model_name)
            rows.append({
                "cost_bps": c * 10000,
                "sharpe": result.metrics.get("sharpe_ratio", 0),
                "sortino": result.metrics.get("sortino_ratio", 0),
                "total_return_pct": result.metrics.get("total_return", 0) * 100,
                "max_drawdown_pct": result.metrics.get("max_drawdown", 0) * 100,
                "avg_turnover": result.trading_stats.get("avg_daily_turnover", 0),
            })

        return pd.DataFrame(rows)

    def run_threshold_sensitivity(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        thresholds: Optional[List[float]] = None,
        model_name: str = "Model",
    ) -> pd.DataFrame:
        """Sweep signal thresholds and report metrics for each.

        Only meaningful for the ``threshold`` strategy.

        Parameters
        ----------
        thresholds : list[float]
            Threshold values to sweep.

        Returns
        -------
        DataFrame with one row per threshold.
        """
        if thresholds is None:
            thresholds = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005]

        rows = []
        for tau in thresholds:
            cfg = PipelineConfig(
                strategy="threshold",
                transaction_cost=self.config.transaction_cost,
                threshold=tau,
                risk_free_rate=self.config.risk_free_rate,
                periods_per_year=self.config.periods_per_year,
                generate_plots=False,
            )
            pipe = EvaluationPipeline(cfg)
            result = pipe.evaluate_model(predictions, actual_returns, model_name)

            # Exposure % helps understand how often the model trades
            rows.append({
                "threshold": tau,
                "sharpe": result.metrics.get("sharpe_ratio", 0),
                "sortino": result.metrics.get("sortino_ratio", 0),
                "total_return_pct": result.metrics.get("total_return", 0) * 100,
                "exposure_pct": result.trading_stats.get("exposure_pct", 0),
                "avg_turnover": result.trading_stats.get("avg_daily_turnover", 0),
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience function for integration with training_service.py
# ---------------------------------------------------------------------------

def run_pipeline_evaluation(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    model_name: str,
    strategy: str = "sign",
    transaction_cost: float = 0.001,
    generate_plots: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for the web-app training service.

    Returns a JSON-serialisable dictionary of metrics, trading stats,
    and plot paths — ready to be stored in ``*_results.json`` and served
    by the FastAPI backend.
    """
    config = PipelineConfig(
        strategy=strategy,
        transaction_cost=transaction_cost,
        generate_plots=generate_plots,
        output_dir=Path(output_dir) if output_dir else Path("results/evaluation"),
    )
    pipe = EvaluationPipeline(config)
    result = pipe.evaluate_model(predictions, actual_returns, model_name)
    return result.to_dict()
