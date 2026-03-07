"""
Dissertation experiment audit/orchestration pipeline.

Phase-1 objective:
- Validate the 25-model experiment specification and grouping.
- Audit available result artifacts against a strict evaluation contract.
- Produce reproducible tables/figures split by model groups and causal/oracle.
- Persist leaderboard-ready rows into the experiments database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .leaderboard import ExperimentEntry, ResultsDatabase
from ..utils.logger import get_logger

logger = get_logger(__name__)


LOWER_IS_BETTER = {"rmse", "mae", "mse", "mape", "max_drawdown"}

DEFAULT_RESULT_ALIASES: Dict[str, List[str]] = {
    "baseline_pinn": ["baseline_pinn", "pinn_baseline", "baseline"],
    "gbm": ["gbm", "pinn_gbm"],
    "ou": ["ou", "pinn_ou"],
    "black_scholes": ["black_scholes", "pinn_black_scholes"],
    "gbm_ou": ["gbm_ou", "pinn_gbm_ou"],
    "global": ["global", "pinn_global"],
    "stacked": ["stacked", "stacked_pinn", "pinn_stacked"],
    "residual": ["residual", "residual_pinn", "pinn_residual"],
}


@dataclass
class DissertationPipelineConfig:
    name: str
    expected_model_count: int
    groups: Dict[str, List[str]]
    causal_models: List[str]
    oracle_models: List[str]
    training_contract: Dict[str, Any] = field(default_factory=dict)
    evaluation_contract: Dict[str, Any] = field(default_factory=dict)
    ranking_metric: str = "sharpe_ratio"
    comparison_metrics: List[str] = field(
        default_factory=lambda: ["sharpe_ratio", "rmse", "directional_accuracy"]
    )
    artifacts: Dict[str, Any] = field(default_factory=dict)
    result_aliases: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def all_models(self) -> List[str]:
        models: List[str] = []
        for keys in self.groups.values():
            models.extend(keys)
        return models

    @property
    def unique_models(self) -> List[str]:
        return sorted(set(self.all_models))


@dataclass
class ModelAudit:
    model_key: str
    group: str
    is_causal: bool
    has_results: bool
    result_path: Optional[str]
    prediction_path: Optional[str]
    epochs_trained: Optional[int]
    target_epochs: Optional[int]
    reached_target_epochs: bool
    metrics_valid: bool
    strict_contract_pass: bool
    has_scaler_params: bool
    has_position_lag_flag: bool
    has_raw_metrics: bool
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    mse: Optional[float] = None
    r2: Optional[float] = None
    directional_accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    total_return: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None
    annualized_return: Optional[float] = None
    sharpe_ratio_raw: Optional[float] = None
    sortino_ratio_raw: Optional[float] = None
    total_return_raw: Optional[float] = None
    annualized_return_raw: Optional[float] = None
    best_val_loss: Optional[float] = None
    notes: str = ""


def load_dissertation_config(path: Path | str) -> DissertationPipelineConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text()) or {}
    cfg = DissertationPipelineConfig(
        name=str(data.get("name", "dissertation_pipeline")),
        expected_model_count=int(data.get("expected_model_count", 25)),
        groups={k: list(v) for k, v in (data.get("groups") or {}).items()},
        causal_models=list(data.get("causal_models") or []),
        oracle_models=list(data.get("oracle_models") or []),
        training_contract=dict(data.get("training_contract") or {}),
        evaluation_contract=dict(data.get("evaluation_contract") or {}),
        ranking_metric=str(data.get("ranking_metric", "sharpe_ratio")),
        comparison_metrics=list(
            data.get("comparison_metrics")
            or ["sharpe_ratio", "rmse", "directional_accuracy"]
        ),
        artifacts=dict(data.get("artifacts") or {}),
        result_aliases={k: list(v) for k, v in (data.get("result_aliases") or {}).items()},
    )
    return cfg


def validate_dissertation_config(config: DissertationPipelineConfig) -> List[str]:
    issues: List[str] = []

    if not config.groups:
        issues.append("Config contains no model groups.")
        return issues

    all_models = config.all_models
    unique_models = set(all_models)
    if len(unique_models) != config.expected_model_count:
        issues.append(
            f"Expected {config.expected_model_count} unique models but found {len(unique_models)}."
        )

    duplicate_models = sorted({m for m in all_models if all_models.count(m) > 1})
    if duplicate_models:
        issues.append(f"Duplicate models across groups: {duplicate_models}")

    causal_set = set(config.causal_models)
    oracle_set = set(config.oracle_models)
    overlap = causal_set & oracle_set
    if overlap:
        issues.append(f"Models marked as both causal and oracle: {sorted(overlap)}")

    missing_classification = unique_models - (causal_set | oracle_set)
    if missing_classification:
        issues.append(
            "Models missing causal/oracle classification: "
            f"{sorted(missing_classification)}"
        )

    unknown_classification = (causal_set | oracle_set) - unique_models
    if unknown_classification:
        issues.append(
            "Causal/oracle list contains unknown models: "
            f"{sorted(unknown_classification)}"
        )

    if config.ranking_metric not in (
        set(config.comparison_metrics)
        | {
            "rmse",
            "mae",
            "mse",
            "mape",
            "r2",
            "directional_accuracy",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "total_return",
            "volatility",
            "win_rate",
            "annualized_return",
        }
    ):
        issues.append(f"Unknown ranking metric: {config.ranking_metric}")

    return issues


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        fv = _safe_float(v)
        if fv is not None:
            out.append(fv)
    return out


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


class DissertationPipelineRunner:
    """Audit and reporting runner for dissertation experiment outputs."""

    def __init__(self, config: DissertationPipelineConfig, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.results_dir = Path(self.config.artifacts.get("results_dir", "results"))
        self.output_root = Path(
            self.config.artifacts.get("output_root", "results/dissertation_pipeline")
        )
        self.db_path = Path(self.config.artifacts.get("db_path", "results/experiments.db"))
        self.top_n = int(self.config.artifacts.get("top_n_per_plot", 6))
        self.histories: Dict[str, Dict[str, Any]] = {}

    def run(self, output_dir: Optional[Path] = None, persist_db: bool = True) -> Dict[str, Any]:
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = output_dir or (self.output_root / run_ts)
        out_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        model_rows = self._collect_model_rows()
        df = pd.DataFrame([row.__dict__ for row in model_rows])
        if not df.empty:
            df = df.sort_values(["group", "model_key"]).reset_index(drop=True)

        self._write_outputs(df, out_dir)
        self._write_group_leaderboards(df, out_dir)
        self._write_causal_oracle_leaderboards(df, out_dir)
        self._make_group_plots(df, figures_dir)

        if persist_db:
            self._persist_to_results_db(df)

        summary = self._build_summary(df, out_dir)
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(
            f"Dissertation pipeline complete: {summary.get('total_models')} models, "
            f"{summary.get('models_with_results')} with results, output={out_dir}"
        )
        return summary

    def _result_aliases_for(self, model_key: str) -> List[str]:
        merged = dict(DEFAULT_RESULT_ALIASES)
        merged.update(self.config.result_aliases)
        aliases = merged.get(model_key, [model_key])
        if model_key not in aliases:
            aliases = [model_key] + aliases
        seen = set()
        uniq: List[str] = []
        for alias in aliases:
            if alias not in seen:
                uniq.append(alias)
                seen.add(alias)
        return uniq

    def _find_result_path(self, model_key: str) -> Optional[Path]:
        for stem in self._result_aliases_for(model_key):
            candidate = self.results_dir / f"{stem}_results.json"
            if candidate.exists():
                return candidate
        return None

    def _find_prediction_path(self, model_key: str) -> Optional[Path]:
        for stem in self._result_aliases_for(model_key):
            candidate = self.results_dir / f"{stem}_predictions.npz"
            if candidate.exists():
                return candidate
        return None

    def _load_result_payload(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {}

    def _extract_metrics(self, payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
        metrics = payload.get("test_metrics") or payload.get("metrics")
        if metrics is None:
            # Support evaluation_harness shape
            if isinstance(payload.get("ml_metrics"), dict) or isinstance(payload.get("financial_metrics"), dict):
                metrics = {}
                metrics.update(payload.get("ml_metrics") or {})
                metrics.update(payload.get("financial_metrics") or {})
            else:
                metrics = {}

        out: Dict[str, Optional[float]] = {}
        for key in [
            "rmse",
            "mae",
            "mape",
            "mse",
            "r2",
            "directional_accuracy",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "total_return",
            "volatility",
            "win_rate",
            "annualized_return",
            "sharpe_ratio_raw",
            "sortino_ratio_raw",
            "total_return_raw",
            "annualized_return_raw",
        ]:
            out[key] = _safe_float(metrics.get(key))
        return out

    def _extract_epochs(self, payload: Dict[str, Any]) -> Optional[int]:
        direct = payload.get("epochs_trained")
        if isinstance(direct, int):
            return direct

        history = payload.get("history") or {}
        if isinstance(history, dict):
            direct_hist = history.get("epochs_trained")
            if isinstance(direct_hist, int):
                return direct_hist

            train_loss = history.get("train_loss")
            if isinstance(train_loss, list) and train_loss:
                return len(train_loss)

            epochs = history.get("epochs")
            if isinstance(epochs, list) and epochs:
                return len(epochs)

        return None

    def _extract_best_val_loss(self, payload: Dict[str, Any]) -> Optional[float]:
        direct = _safe_float(payload.get("best_val_loss"))
        if direct is not None:
            return direct

        history = payload.get("history") or {}
        val_loss = history.get("val_loss") if isinstance(history, dict) else None
        if not isinstance(val_loss, list):
            return None
        finite = _finite_values(val_loss)
        if not finite:
            return None
        return float(min(finite))

    def _history(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        history = payload.get("history")
        return history if isinstance(history, dict) else {}

    def _contract_flags(
        self,
        payload: Dict[str, Any],
        metrics: Dict[str, Optional[float]],
        metrics_valid: bool,
    ) -> Tuple[bool, bool, bool, bool]:
        scaler_map = payload.get("scaler_params")
        has_scaler_params = bool(scaler_map)
        if not has_scaler_params:
            has_scaler_params = (
                payload.get("scaler_mean") is not None
                and payload.get("scaler_std") is not None
            )

        execution = payload.get("execution_assumptions") or {}
        has_position_lag = execution.get("position_lag") == 1

        has_raw_metrics = (
            metrics.get("sharpe_ratio_raw") is not None
            and metrics.get("sortino_ratio_raw") is not None
            and metrics.get("total_return_raw") is not None
            and metrics.get("annualized_return_raw") is not None
        )

        strict_pass = metrics_valid
        if self.config.evaluation_contract.get("require_destandardization", True):
            strict_pass = strict_pass and has_scaler_params
        if self.config.evaluation_contract.get("require_position_lag", True):
            strict_pass = strict_pass and has_position_lag
        if self.config.evaluation_contract.get("require_clipped_and_unclipped_metrics", True):
            strict_pass = strict_pass and has_raw_metrics

        return strict_pass, has_scaler_params, has_position_lag, has_raw_metrics

    def _collect_model_rows(self) -> List[ModelAudit]:
        rows: List[ModelAudit] = []
        target_epochs = self.config.training_contract.get("epochs")
        target_epochs_int = int(target_epochs) if target_epochs is not None else None

        model_to_group: Dict[str, str] = {}
        for group, models in self.config.groups.items():
            for model in models:
                model_to_group[model] = group

        for model_key in self.config.unique_models:
            group = model_to_group.get(model_key, "unassigned")
            is_causal = model_key in set(self.config.causal_models)
            result_path = self._find_result_path(model_key)
            pred_path = self._find_prediction_path(model_key)

            if result_path is None:
                rows.append(
                    ModelAudit(
                        model_key=model_key,
                        group=group,
                        is_causal=is_causal,
                        has_results=False,
                        result_path=None,
                        prediction_path=str(pred_path) if pred_path else None,
                        epochs_trained=None,
                        target_epochs=target_epochs_int,
                        reached_target_epochs=False,
                        metrics_valid=False,
                        strict_contract_pass=False,
                        has_scaler_params=False,
                        has_position_lag_flag=False,
                        has_raw_metrics=False,
                        notes="missing_result_artifact",
                    )
                )
                continue

            payload = self._load_result_payload(result_path)
            metrics = self._extract_metrics(payload)
            finite_metric_values = [v for v in metrics.values() if v is not None]
            metrics_valid = len(finite_metric_values) > 0

            epochs_trained = self._extract_epochs(payload)
            reached_target = (
                target_epochs_int is not None
                and epochs_trained is not None
                and epochs_trained >= target_epochs_int
            )
            best_val_loss = self._extract_best_val_loss(payload)

            strict_pass, has_scaler, has_lag, has_raw = self._contract_flags(
                payload, metrics, metrics_valid
            )

            history = self._history(payload)
            if history:
                self.histories[model_key] = history

            notes: List[str] = []
            if target_epochs_int is not None and not reached_target:
                notes.append("undertrained_vs_contract")
            if not strict_pass:
                notes.append("strict_contract_failed")
            if not metrics_valid:
                notes.append("metrics_missing_or_non_finite")

            rows.append(
                ModelAudit(
                    model_key=model_key,
                    group=group,
                    is_causal=is_causal,
                    has_results=True,
                    result_path=str(result_path),
                    prediction_path=str(pred_path) if pred_path else None,
                    epochs_trained=epochs_trained,
                    target_epochs=target_epochs_int,
                    reached_target_epochs=reached_target,
                    metrics_valid=metrics_valid,
                    strict_contract_pass=strict_pass,
                    has_scaler_params=has_scaler,
                    has_position_lag_flag=has_lag,
                    has_raw_metrics=has_raw,
                    rmse=metrics.get("rmse"),
                    mae=metrics.get("mae"),
                    mape=metrics.get("mape"),
                    mse=metrics.get("mse"),
                    r2=metrics.get("r2"),
                    directional_accuracy=metrics.get("directional_accuracy"),
                    sharpe_ratio=metrics.get("sharpe_ratio"),
                    sortino_ratio=metrics.get("sortino_ratio"),
                    max_drawdown=metrics.get("max_drawdown"),
                    calmar_ratio=metrics.get("calmar_ratio"),
                    total_return=metrics.get("total_return"),
                    volatility=metrics.get("volatility"),
                    win_rate=metrics.get("win_rate"),
                    annualized_return=metrics.get("annualized_return"),
                    sharpe_ratio_raw=metrics.get("sharpe_ratio_raw"),
                    sortino_ratio_raw=metrics.get("sortino_ratio_raw"),
                    total_return_raw=metrics.get("total_return_raw"),
                    annualized_return_raw=metrics.get("annualized_return_raw"),
                    best_val_loss=best_val_loss,
                    notes=";".join(notes),
                )
            )

        return rows

    def _write_outputs(self, df: pd.DataFrame, out_dir: Path) -> None:
        df.to_csv(out_dir / "model_audit.csv", index=False)
        (out_dir / "model_audit.json").write_text(
            json.dumps(df.to_dict(orient="records"), indent=2, default=str)
        )

        group_summary = (
            df.groupby("group", dropna=False)
            .agg(
                total_models=("model_key", "count"),
                models_with_results=("has_results", "sum"),
                strict_contract_pass=("strict_contract_pass", "sum"),
                reached_target_epochs=("reached_target_epochs", "sum"),
            )
            .reset_index()
        )
        group_summary.to_csv(out_dir / "group_summary.csv", index=False)

        missing = df.loc[~df["has_results"], ["model_key", "group"]]
        missing.to_csv(out_dir / "missing_models.csv", index=False)

    def _sorted_metric_frame(
        self, frame: pd.DataFrame, metric: str, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        valid = frame.loc[frame[metric].notna()].copy()
        if valid.empty:
            return valid
        ascending = metric in LOWER_IS_BETTER
        valid = valid.sort_values(metric, ascending=ascending)
        if top_n is not None:
            valid = valid.head(top_n)
        return valid

    def _write_group_leaderboards(self, df: pd.DataFrame, out_dir: Path) -> None:
        for group_name in sorted(df["group"].dropna().unique()):
            group_df = df.loc[df["group"] == group_name]
            lb = self._sorted_metric_frame(group_df, self.config.ranking_metric)
            lb.to_csv(
                out_dir / f"leaderboard_{_slug(group_name)}_{self.config.ranking_metric}.csv",
                index=False,
            )

    def _write_causal_oracle_leaderboards(self, df: pd.DataFrame, out_dir: Path) -> None:
        for key, mask in [
            ("causal", df["is_causal"] == True),
            ("oracle", df["is_causal"] == False),
        ]:
            subset = df.loc[mask & df["has_results"] & df["metrics_valid"]].copy()
            ranked = self._sorted_metric_frame(subset, self.config.ranking_metric)
            ranked.to_csv(
                out_dir / f"{key}_leaderboard_{self.config.ranking_metric}.csv",
                index=False,
            )

    def _make_group_plots(self, df: pd.DataFrame, figures_dir: Path) -> None:
        self._plot_group_metric_bars(df, figures_dir)
        self._plot_training_curves(df, figures_dir)
        self._plot_data_vs_physics(df, figures_dir)

    def _plot_group_metric_bars(self, df: pd.DataFrame, figures_dir: Path) -> None:
        for group_name in sorted(df["group"].dropna().unique()):
            group_df = df.loc[df["group"] == group_name].copy()
            if group_df.empty:
                continue
            for metric in self.config.comparison_metrics:
                if metric not in group_df.columns:
                    continue
                ranked = self._sorted_metric_frame(group_df, metric, top_n=self.top_n)
                if ranked.empty:
                    continue
                plot_df = ranked.sort_values(metric, ascending=True)
                fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * len(plot_df)))
                ax.barh(plot_df["model_key"], plot_df[metric], color="#336699", alpha=0.85)
                ax.set_title(f"{group_name}: {metric} (top {len(plot_df)})")
                ax.set_xlabel(metric)
                ax.set_ylabel("model")
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                out_path = figures_dir / f"{_slug(group_name)}_{metric}_top.png"
                fig.savefig(out_path, dpi=180, bbox_inches="tight")
                plt.close(fig)

    def _plot_training_curves(self, df: pd.DataFrame, figures_dir: Path) -> None:
        for group_name in sorted(df["group"].dropna().unique()):
            group_df = df.loc[df["group"] == group_name].copy()
            if group_df.empty:
                continue
            candidates = self._sorted_metric_frame(
                group_df, self.config.ranking_metric, top_n=self.top_n
            )
            if candidates.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            plotted = 0
            for model_key in candidates["model_key"]:
                history = self.histories.get(model_key) or {}
                train_loss = _finite_values(history.get("train_loss") or [])
                val_loss = _finite_values(history.get("val_loss") or [])
                if not train_loss:
                    continue
                x_train = np.arange(1, len(train_loss) + 1)
                ax.plot(x_train, train_loss, label=f"{model_key}:train", alpha=0.8)
                if val_loss:
                    x_val = np.arange(1, len(val_loss) + 1)
                    ax.plot(
                        x_val,
                        val_loss,
                        linestyle="--",
                        label=f"{model_key}:val",
                        alpha=0.7,
                    )
                plotted += 1

            if plotted == 0:
                plt.close(fig)
                continue

            ax.set_title(f"{group_name}: training curves")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            out_path = figures_dir / f"{_slug(group_name)}_training_curves.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _plot_data_vs_physics(self, df: pd.DataFrame, figures_dir: Path) -> None:
        for group_name in sorted(df["group"].dropna().unique()):
            group_df = df.loc[df["group"] == group_name].copy()
            if group_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            plotted = 0

            for model_key in group_df["model_key"]:
                history = self.histories.get(model_key) or {}
                data_loss = _finite_values(history.get("data_loss") or [])
                physics_loss = _finite_values(history.get("physics_loss") or [])
                if not data_loss or not physics_loss:
                    continue

                n = min(len(data_loss), len(physics_loss))
                x = np.arange(1, n + 1)
                ax.plot(x, data_loss[:n], label=f"{model_key}:data", alpha=0.75)
                ax.plot(
                    x,
                    physics_loss[:n],
                    linestyle="--",
                    label=f"{model_key}:physics",
                    alpha=0.75,
                )
                plotted += 1
                if plotted >= self.top_n:
                    break

            if plotted == 0:
                plt.close(fig)
                continue

            ax.set_title(f"{group_name}: data loss vs physics loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            out_path = figures_dir / f"{_slug(group_name)}_data_vs_physics.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _persist_to_results_db(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        db = ResultsDatabase(self.db_path)
        config_hash = self._config_hash()

        for _, row in df.iterrows():
            if not bool(row.get("has_results")):
                continue
            if not bool(row.get("metrics_valid")):
                continue

            sharpe = _safe_float(row.get("sharpe_ratio")) or 0.0
            sortino = _safe_float(row.get("sortino_ratio")) or 0.0
            total_return = _safe_float(row.get("total_return")) or 0.0
            annualized_return = _safe_float(row.get("annualized_return")) or 0.0
            volatility = _safe_float(row.get("volatility")) or 0.0
            max_drawdown = _safe_float(row.get("max_drawdown")) or 0.0
            calmar_ratio = _safe_float(row.get("calmar_ratio")) or 0.0

            model_key = str(row["model_key"])
            entry = ExperimentEntry(
                experiment_id=f"{model_key}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                model_name=model_key,
                model_type=str(row["group"]),
                config_hash=config_hash,
                timestamp=datetime.now(timezone.utc).isoformat(),
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                mse=_safe_float(row.get("mse")),
                mae=_safe_float(row.get("mae")),
                rmse=_safe_float(row.get("rmse")),
                mape=_safe_float(row.get("mape")),
                directional_accuracy=_safe_float(row.get("directional_accuracy")),
                n_epochs=int(row["epochs_trained"]) if pd.notna(row.get("epochs_trained")) else None,
                training_time=None,
                n_parameters=None,
                seed=self._training_seed(),
                is_causal=bool(row["is_causal"]),
                model_category="forecasting" if bool(row["is_causal"]) else "oracle",
                sharpe_ratio_raw=_safe_float(row.get("sharpe_ratio_raw")),
                sortino_ratio_raw=_safe_float(row.get("sortino_ratio_raw")),
                notes=str(row.get("notes", "")),
            )
            db.save_experiment(entry, config=self._metadata_blob())

    def _training_seed(self) -> Optional[int]:
        seed = self.config.training_contract.get("seed")
        return int(seed) if seed is not None else None

    def _config_hash(self) -> str:
        payload = self._metadata_blob()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:12]

    def _metadata_blob(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "name": self.config.name,
            "expected_model_count": self.config.expected_model_count,
            "groups": self.config.groups,
            "causal_models": self.config.causal_models,
            "oracle_models": self.config.oracle_models,
            "training_contract": self.config.training_contract,
            "evaluation_contract": self.config.evaluation_contract,
            "ranking_metric": self.config.ranking_metric,
            "comparison_metrics": self.config.comparison_metrics,
            "artifacts": self.config.artifacts,
        }

    def _build_summary(self, df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
        strict_fail = int((~df["strict_contract_pass"]).sum()) if not df.empty else 0
        missing = int((~df["has_results"]).sum()) if not df.empty else 0
        models_with_results = int(df["has_results"].sum()) if not df.empty else 0

        summary = {
            "name": self.config.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(self.config_path),
            "config_hash": self._config_hash(),
            "output_dir": str(out_dir),
            "total_models": int(len(df)),
            "models_with_results": models_with_results,
            "missing_models": missing,
            "strict_contract_failures": strict_fail,
            "ranking_metric": self.config.ranking_metric,
            "groups": list(self.config.groups.keys()),
        }
        return summary
