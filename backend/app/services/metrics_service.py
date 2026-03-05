"""Service for metrics calculation wrapping src/evaluation/."""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.schemas.metrics import (
    MLMetrics,
    FinancialMetrics,
    PhysicsMetrics,
    ModelMetricsResponse,
    MetricsComparisonResponse,
)
from backend.app.services.model_service import ModelService

# Import from existing src/
try:
    from src.evaluation.financial_metrics import FinancialMetrics as SrcFinancialMetrics
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.leaderboard import (
        ResultsDatabase,
        LeaderboardGenerator,
        RankingMetric,
    )
    HAS_SRC = True
except ImportError:
    HAS_SRC = False
    SrcFinancialMetrics = None
    MetricsCalculator = None
    ResultsDatabase = None
    LeaderboardGenerator = None
    RankingMetric = None


class MetricsService:
    """Service for calculating and retrieving metrics."""

    def __init__(self):
        """Initialize metrics service."""
        self._model_service = ModelService()
        self._cached_metrics: Dict[str, ModelMetricsResponse] = {}
        self._results_db = ResultsDatabase(settings.results_db_path) if HAS_SRC else None

    def calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> MLMetrics:
        """Calculate ML prediction metrics."""
        if HAS_SRC:
            return MLMetrics(
                rmse=MetricsCalculator.rmse(y_true, y_pred),
                mae=MetricsCalculator.mae(y_true, y_pred),
                mape=MetricsCalculator.mape(y_true, y_pred),
                r2=MetricsCalculator.r2(y_true, y_pred),
                # MetricsCalculator now returns 0–1; API contract expects percentage
                directional_accuracy=MetricsCalculator.directional_accuracy(y_true, y_pred) * 100,
            )
        else:
            # Basic implementation
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            # Directional accuracy
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            da = np.mean(true_direction == pred_direction) * 100

            return MLMetrics(
                rmse=float(rmse),
                mae=float(mae),
                mape=float(mape),
                r2=float(r2),
                directional_accuracy=float(da),
                mse=float(mse),
            )

    def calculate_financial_metrics(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> FinancialMetrics:
        """Calculate financial performance metrics."""
        returns = np.array(returns)

        if HAS_SRC:
            metrics_class = SrcFinancialMetrics

            sharpe = metrics_class.sharpe_ratio(returns, risk_free_rate, periods_per_year)
            sortino = metrics_class.sortino_ratio(returns, risk_free_rate, periods_per_year)
            max_dd = metrics_class.max_drawdown(returns)
            win_rate = metrics_class.win_rate(returns)

            # Calculate returns
            total_return = np.prod(1 + returns) - 1
            annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

            info_ratio = None
            if benchmark_returns is not None:
                info_ratio = metrics_class.information_ratio(returns, benchmark_returns)

            calmar = None
            if max_dd != 0:
                calmar = annual_return / abs(max_dd)

        else:
            # Basic implementation
            daily_rf = risk_free_rate / periods_per_year
            excess_returns = returns - daily_rf

            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Sharpe
            sharpe = (mean_return - daily_rf) / (std_return + 1e-8) * np.sqrt(periods_per_year)

            # Sortino
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
            sortino = (mean_return - daily_rf) / (downside_std + 1e-8) * np.sqrt(periods_per_year)

            # Max Drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_dd = float(np.min(drawdowns))

            # Win rate
            win_rate = float(np.mean(returns > 0) * 100)

            # Returns
            total_return = float(np.prod(1 + returns) - 1)
            annual_return = float((1 + total_return) ** (periods_per_year / max(len(returns), 1)) - 1)

            # Calmar
            calmar = annual_return / abs(max_dd) if max_dd != 0 else None

            # Information ratio
            info_ratio = None
            if benchmark_returns is not None:
                active_returns = returns - benchmark_returns
                info_ratio = float(np.mean(active_returns) / (np.std(active_returns) + 1e-8) * np.sqrt(periods_per_year))

        # Count trades
        winning = np.sum(returns > 0)
        losing = np.sum(returns < 0)
        total_trades = len(returns)

        # Average win/loss
        avg_win = float(np.mean(returns[returns > 0])) if winning > 0 else 0.0
        avg_loss = float(np.mean(returns[returns < 0])) if losing > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else None

        return FinancialMetrics(
            total_return=float(total_return),
            annual_return=float(annual_return),
            daily_return_mean=float(mean_return) if not HAS_SRC else float(np.mean(returns)),
            daily_return_std=float(std_return) if not HAS_SRC else float(np.std(returns)),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar) if calmar is not None else None,
            information_ratio=float(info_ratio) if info_ratio is not None else None,
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=int(winning),
            losing_trades=int(losing),
        )

    def get_leaderboard(self, metric: str = "sharpe_ratio", top_n: int = 10) -> Dict[str, Any]:
        """Return a leaderboard of experiments from the results DB."""
        if not HAS_SRC or self._results_db is None or LeaderboardGenerator is None:
            raise RuntimeError("Leaderboard requires src.evaluation to be available")

        # Map string to RankingMetric; default to SHARPE when unknown.
        metric_enum = next((m for m in RankingMetric if m.value == metric), RankingMetric.SHARPE)
        generator = LeaderboardGenerator(self._results_db)
        lb = generator.generate_leaderboard(metric_enum, top_n=top_n)

        return {
            "metric": lb.metric.value,
            "generated_at": lb.generated_at,
            "n_experiments": lb.n_experiments,
            "entries": [
                {
                    "rank": e.rank,
                    "experiment_id": e.experiment_id,
                    "model_name": e.model_name,
                    "metric_value": e.metric_value,
                    "metric_name": e.metric_name,
                    "other_metrics": e.other_metrics,
                }
                for e in lb.entries
            ],
        }

    def get_physics_metrics(self, model_key: str) -> Optional[PhysicsMetrics]:
        """Get physics constraint metrics for a PINN model."""
        import logging
        logger = logging.getLogger(__name__)

        # Define default physics configurations based on model key
        default_physics_config = {
            'baseline': {'lambda_gbm': 0.0, 'lambda_ou': 0.0, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'pinn_baseline': {'lambda_gbm': 0.0, 'lambda_ou': 0.0, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'gbm': {'lambda_gbm': 0.1, 'lambda_ou': 0.0, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'pinn_gbm': {'lambda_gbm': 0.1, 'lambda_ou': 0.0, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'ou': {'lambda_gbm': 0.0, 'lambda_ou': 0.1, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'pinn_ou': {'lambda_gbm': 0.0, 'lambda_ou': 0.1, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'black_scholes': {'lambda_gbm': 0.0, 'lambda_ou': 0.0, 'lambda_bs': 0.1, 'lambda_langevin': 0.0},
            'pinn_black_scholes': {'lambda_gbm': 0.0, 'lambda_ou': 0.0, 'lambda_bs': 0.1, 'lambda_langevin': 0.0},
            'gbm_ou': {'lambda_gbm': 0.05, 'lambda_ou': 0.05, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'pinn_gbm_ou': {'lambda_gbm': 0.05, 'lambda_ou': 0.05, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'global': {'lambda_gbm': 0.05, 'lambda_ou': 0.05, 'lambda_bs': 0.03, 'lambda_langevin': 0.02},
            'pinn_global': {'lambda_gbm': 0.05, 'lambda_ou': 0.05, 'lambda_bs': 0.03, 'lambda_langevin': 0.02},
            'stacked': {'lambda_gbm': 0.1, 'lambda_ou': 0.1, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'stacked_pinn': {'lambda_gbm': 0.1, 'lambda_ou': 0.1, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'residual': {'lambda_gbm': 0.1, 'lambda_ou': 0.1, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
            'residual_pinn': {'lambda_gbm': 0.1, 'lambda_ou': 0.1, 'lambda_bs': 0.0, 'lambda_langevin': 0.0},
        }

        # First try to load from saved results (faster, no model loading needed)
        saved_metrics = self.load_saved_metrics(model_key)
        config = {}
        fin_metrics = {}
        total_physics_loss = 0.0

        if saved_metrics:
            # Check if we have physics-related data in saved metrics
            config = saved_metrics.get("configuration", {})
            fin_metrics = saved_metrics.get("financial_metrics", {}) or saved_metrics.get("test_metrics", {})

            # Get history for physics loss if available
            history = saved_metrics.get("history", {})
            train_physics_loss = history.get("train_physics_loss", [])
            total_physics_loss = train_physics_loss[-1] if train_physics_loss else 0.0

        # If no config from saved metrics, use defaults based on model key
        if not config:
            config = default_physics_config.get(model_key, {})

        # Get physics weights from configuration
        lambda_gbm = config.get("lambda_gbm", 0.0)
        lambda_ou = config.get("lambda_ou", 0.0)
        lambda_bs = config.get("lambda_bs", 0.0)
        lambda_langevin = config.get("lambda_langevin", 0.0)

        # Check if this is a PINN model (has physics constraints or enable_physics flag)
        is_pinn = (
            config.get("enable_physics", False) or
            any([lambda_gbm, lambda_ou, lambda_bs, lambda_langevin]) or
            'pinn' in model_key.lower()
        )

        # Get physics params from saved metrics if available
        physics_params = saved_metrics.get("physics_params", {}) if saved_metrics else {}

        if is_pinn:
            return PhysicsMetrics(
                total_physics_loss=float(total_physics_loss),
                gbm_loss=float(lambda_gbm) if lambda_gbm else None,
                ou_loss=float(lambda_ou) if lambda_ou else None,
                black_scholes_loss=float(lambda_bs) if lambda_bs else None,
                langevin_loss=float(lambda_langevin) if lambda_langevin else None,
                # Learned parameters - Only return if their respective physics loss was active
                theta=physics_params.get("theta") if lambda_ou > 0 else None,
                gamma=physics_params.get("gamma") if lambda_langevin > 0 else None,
                temperature=(physics_params.get("temperature") or physics_params.get("T")) if lambda_langevin > 0 else None,
                mu=physics_params.get("mu") or (fin_metrics.get("annualized_return") / 100.0 if fin_metrics and "annualized_return" in fin_metrics else None) if (lambda_gbm > 0 or lambda_bs > 0) else None,
                sigma=physics_params.get("sigma") or (fin_metrics.get("volatility") if fin_metrics else None) if (lambda_gbm > 0 or lambda_bs > 0) else None,
            )

        # Fallback: try to load the model directly
        try:
            model = self._model_service.load_model(model_key)

            if model is None:
                logger.warning(f"Model {model_key} could not be loaded")
                return None

            if not hasattr(model, "get_learned_physics_params"):
                logger.warning(f"Model {model_key} does not have get_learned_physics_params method")
                return None

            params = model.get_learned_physics_params()
            logger.info(f"Got physics params for {model_key}: {params}")

            return PhysicsMetrics(
                total_physics_loss=params.get("total_physics_loss", 0.0),
                gbm_loss=params.get("gbm_loss"),
                ou_loss=params.get("ou_loss"),
                black_scholes_loss=params.get("black_scholes_loss"),
                langevin_loss=params.get("langevin_loss"),
                theta=params.get("theta"),
                gamma=params.get("gamma"),
                temperature=params.get("temperature") or params.get("T"),
                mu=params.get("mu"),
                sigma=params.get("sigma"),
            )
        except Exception as e:
            logger.error(f"Error getting physics metrics for {model_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def get_model_metrics(
        self,
        model_key: str,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
    ) -> ModelMetricsResponse:
        """Get complete metrics for a model."""
        model_info = self._model_service.get_model_info(model_key)

        # Try to load saved metrics as fallback
        saved_metrics = self.load_saved_metrics(model_key)

        # ML metrics
        ml_metrics = None
        if y_true is not None and y_pred is not None:
            ml_metrics = self.calculate_ml_metrics(y_true, y_pred)
        elif saved_metrics:
            # Try to load from saved metrics
            test_data = saved_metrics.get("test_metrics", {})
            ml_data = saved_metrics.get("ml_metrics") or saved_metrics.get("metrics", {}) or test_data
            if ml_data and any(k in ml_data for k in ["rmse", "mae", "r2"]):
                ml_metrics = MLMetrics(
                    rmse=float(ml_data.get("rmse", 0)),
                    mae=float(ml_data.get("mae", 0)),
                    mape=float(ml_data.get("mape", 0)),
                    r2=float(ml_data.get("r2", 0)),
                    directional_accuracy=float(ml_data.get("directional_accuracy", 0)),
                )

        # Financial metrics
        financial_metrics = None
        if returns is not None:
            financial_metrics = self.calculate_financial_metrics(returns)
        elif saved_metrics:
            # Try to load from saved metrics
            test_data = saved_metrics.get("test_metrics", {})
            fin_data = saved_metrics.get("financial_metrics") or test_data or saved_metrics
            if fin_data and ("sharpe_ratio" in fin_data or "max_drawdown" in fin_data):
                # Handle both field name conventions
                annual_ret = fin_data.get("annual_return") or fin_data.get("annualized_return", 0)
                financial_metrics = FinancialMetrics(
                    total_return=float(fin_data.get("total_return", 0)),
                    annual_return=float(annual_ret),
                    daily_return_mean=float(fin_data.get("daily_return_mean", fin_data.get("return_mean", 0))),
                    daily_return_std=float(fin_data.get("daily_return_std") or fin_data.get("volatility", 0) or fin_data.get("return_std", 0)),
                    sharpe_ratio=float(fin_data.get("sharpe_ratio", 0)),
                    sortino_ratio=float(fin_data.get("sortino_ratio", 0)),
                    max_drawdown=float(fin_data.get("max_drawdown", 0)),
                    win_rate=float(fin_data.get("win_rate", 0)),
                    calmar_ratio=float(fin_data["calmar_ratio"]) if fin_data.get("calmar_ratio") else None,
                    information_ratio=float(fin_data.get("information_coefficient") or fin_data.get("information_ratio")) if fin_data.get("information_coefficient") or fin_data.get("information_ratio") else None,
                    profit_factor=float(fin_data["profit_factor"]) if fin_data.get("profit_factor") else None,
                )

        # Physics metrics
        physics_metrics = None
        if model_info.is_pinn:
            physics_metrics = self.get_physics_metrics(model_key)

        return ModelMetricsResponse(
            model_key=model_key,
            model_name=model_info.display_name,
            is_pinn=model_info.is_pinn,
            ml_metrics=ml_metrics or MLMetrics(
                rmse=0.0, mae=0.0, mape=0.0, r2=0.0, directional_accuracy=0.0
            ),
            financial_metrics=financial_metrics,
            physics_metrics=physics_metrics,
            evaluation_date=datetime.now(),
        )

    def compare_models(
        self,
        model_keys: List[str],
        metrics_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> MetricsComparisonResponse:
        """Compare metrics across multiple models."""
        models = []
        metric_summary = {}
        rankings = {}

        for model_key in model_keys:
            try:
                if metrics_data and model_key in metrics_data:
                    data = metrics_data[model_key]
                    model_metrics = self.get_model_metrics(
                        model_key=model_key,
                        y_true=np.array(data.get("y_true", [])) if data.get("y_true") else None,
                        y_pred=np.array(data.get("y_pred", [])) if data.get("y_pred") else None,
                        returns=np.array(data.get("returns", [])) if data.get("returns") else None,
                    )
                else:
                    model_metrics = self.get_model_metrics(model_key)

                models.append(model_metrics)

                # Collect metric values for summary
                if model_metrics.ml_metrics:
                    for metric_name in ["rmse", "mae", "r2", "directional_accuracy"]:
                        value = getattr(model_metrics.ml_metrics, metric_name, None)
                        if value is not None:
                            if metric_name not in metric_summary:
                                metric_summary[metric_name] = {}
                            metric_summary[metric_name][model_key] = value

                if model_metrics.financial_metrics:
                    for metric_name in ["sharpe_ratio", "sortino_ratio", "max_drawdown", "total_return"]:
                        value = getattr(model_metrics.financial_metrics, metric_name, None)
                        if value is not None:
                            if metric_name not in metric_summary:
                                metric_summary[metric_name] = {}
                            metric_summary[metric_name][model_key] = value

            except Exception as e:
                print(f"Error getting metrics for {model_key}: {e}")
                continue

        # Determine best by metric and rankings
        best_by_metric = {}
        for metric_name, values in metric_summary.items():
            if not values:
                continue

            # Determine if lower is better
            lower_is_better = any(
                x in metric_name.lower()
                for x in ["loss", "error", "rmse", "mae", "mape", "drawdown"]
            )

            sorted_models = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=not lower_is_better
            )
            rankings[metric_name] = [m[0] for m in sorted_models]
            best_by_metric[metric_name] = sorted_models[0][0]

        return MetricsComparisonResponse(
            models=models,
            metric_summary=metric_summary,
            best_by_metric=best_by_metric,
            rankings=rankings,
        )

    def load_saved_metrics(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Load saved metrics from results directory."""
        results_dir = settings.results_path

        # Try multiple file patterns to find results
        patterns = [
            results_dir / f"{model_key}_results.json",
            results_dir / f"pinn_{model_key}_results.json",
            results_dir / f"rigorous_{model_key}_results.json",
            results_dir / f"rigorous_pinn_{model_key}_results.json",
            results_dir / f"{model_key}_metrics.json",
        ]

        for metrics_file in patterns:
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

        # Also try loading from detailed_results.json for PINN models
        detailed_path = results_dir / 'pinn_comparison' / 'detailed_results.json'
        if detailed_path.exists():
            try:
                with open(detailed_path, 'r') as f:
                    detailed_results = json.load(f)
                    # Search for matching variant
                    for variant in detailed_results:
                        variant_key = variant.get('variant_key', '')
                        # Match by variant_key directly or with pinn_ prefix
                        if variant_key == model_key or f"pinn_{variant_key}" == model_key or variant_key == model_key.replace('pinn_', ''):
                            return variant
            except (json.JSONDecodeError, IOError):
                pass

        return None

    def save_metrics(self, model_key: str, metrics: Dict[str, Any]):
        """Save metrics to results directory."""
        results_dir = settings.results_path
        results_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = results_dir / f"{model_key}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
