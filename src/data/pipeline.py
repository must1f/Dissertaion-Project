"""
End-to-end leakage-safe multi-asset data pipeline for the benchmark task.

Steps:
1) Fetch multi-asset data with cache + QA
2) Align to master calendar
3) Build adjusted-return features and cross-asset signals
4) Construct target (SPY next-day adjusted log return by default)
5) Chronological split with optional gap
6) Fit scaler on train only and apply to val/test
7) Build identical sequence windows for all models
8) Persist reproducibility artifacts (metadata, QA, leakage report)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .fetcher import DataFetcher
from .universe import universe_from_config, UniverseDefinition
from .calendar import build_master_calendar, align_to_calendar
from .quality import run_qa
from .cache import CacheManager
from .features import (
    add_adjusted_returns,
    add_rolling_volatility,
    add_momentum_trend,
    add_cross_asset_spreads,
    add_regime_markers,
)
from .targets import add_next_day_log_return, add_realized_vol, add_joint_return_and_vol
from .split import temporal_split_with_gap
from .scaling import fit_scaler, apply_scaler, save_scaler
from .sequence import build_sequences
from .leakage_auditor import LeakageAuditor, LeakageSeverity, LeakageWarning


QA_POLICY_DEFAULT: Dict[str, Any] = {
    "max_duplicates": 0,
    "max_bad_prices": 0,
    "max_missing_fraction": 0.30,
    "max_extreme_jump_fraction": 0.05,
}

# Core benchmark features required for a scientifically comparable baseline.
REQUIRED_CORE_FEATURES = [
    "adj_return_1d",
    "adj_return_5d",
    "rolling_vol_10",
    "rolling_vol_20",
    "momentum_20_z",
    "momentum_60_z",
    "cross_spy_qqq_spread",
    "cross_spy_iwm_spread",
]

FAIRNESS_CONTRACT_VERSION = "1.0"


def dataset_fingerprint(
    universe: UniverseDefinition,
    feature_cols: List[str],
    target: str,
    lookback: int,
    horizon: int,
    start_date: str,
    end_date: str,
) -> str:
    """Deterministic fingerprint for caching/tracking."""
    payload = (
        f"{universe.hash()}|{','.join(sorted(feature_cols))}|{target}|"
        f"{lookback}|{horizon}|{start_date}|{end_date}"
    )
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _build_fairness_contract(
    *,
    meta: Dict[str, Any],
    feature_columns: List[str],
    required_features: List[str],
    cache_keys: Dict[str, str],
    scaler_path: str,
    sequence_shapes: Dict[str, Any],
    split_ratios: Dict[str, float],
) -> Dict[str, Any]:
    """Construct the strict benchmark contract used across models."""

    return {
        "version": FAIRNESS_CONTRACT_VERSION,
        "fingerprint": meta.get("fingerprint"),
        "universe": meta.get("universe"),
        "cache_keys": cache_keys,
        "dataset": {
            "target_symbol": meta.get("target_symbol"),
            "target_type": meta.get("target_type"),
            "target_columns": meta.get("target_columns"),
            "lookback": meta.get("lookback"),
            "horizon": meta.get("horizon"),
            "start_date": meta.get("start_date"),
            "end_date": meta.get("end_date"),
            "vol_window": meta.get("target_vol_window"),
        },
        "features": {
            "required_core": required_features,
            "effective": feature_columns,
        },
        "splits": {
            "boundaries": meta.get("split_boundaries", {}),
            "ratios": split_ratios,
        },
        "scaling": {
            "type": "standard",
            "fit_on": "train",
            "scaler_path": scaler_path,
        },
        "sequences": sequence_shapes,
    }


def _safe_log_return(series: pd.Series) -> pd.Series:
    out = np.log(series / series.shift(1))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _enforce_qa_gate(report: Dict[str, Any], policy: Dict[str, Any], stage: str) -> None:
    if report.get("error"):
        raise ValueError(f"QA failed at {stage}: {report['error']}")

    dup_count = int(report.get("duplicates", {}).get("count", 0))
    bad_price_count = int(report.get("bad_prices", {}).get("count", 0))
    row_count = int(report.get("coverage", {}).get("rows", 0))
    jump_count = int(report.get("extreme_jumps", {}).get("count", 0))

    if dup_count > int(policy.get("max_duplicates", 0)):
        raise ValueError(f"QA failed at {stage}: duplicate rows={dup_count}")

    if bad_price_count > int(policy.get("max_bad_prices", 0)):
        raise ValueError(f"QA failed at {stage}: bad price rows={bad_price_count}")

    if row_count <= 0:
        raise ValueError(f"QA failed at {stage}: no rows available")

    jump_fraction = jump_count / max(row_count, 1)
    if jump_fraction > float(policy.get("max_extreme_jump_fraction", 1.0)):
        raise ValueError(
            f"QA failed at {stage}: extreme jump fraction={jump_fraction:.4f}"
        )


def _resolve_feature_columns(
    frame: pd.DataFrame,
    configured_cols: List[str],
) -> tuple[List[str], List[str], List[str]]:
    """
    Resolve configured feature columns into:
    - effective feature columns used for training
    - missing optional feature columns
    - missing required core columns (fatal)
    """
    missing_required = [c for c in REQUIRED_CORE_FEATURES if c not in frame.columns]

    optional = [c for c in configured_cols if c not in REQUIRED_CORE_FEATURES]
    optional_available = [c for c in optional if c in frame.columns and frame[c].notna().any()]
    missing_optional = [c for c in optional if c not in optional_available]

    effective = [c for c in REQUIRED_CORE_FEATURES if c in frame.columns] + optional_available

    return effective, missing_optional, missing_required


def _compute_leakage_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_X: np.ndarray,
    sequence_length: int,
    horizon: int,
    feature_cols: List[str],
    scaler_fit_df: pd.DataFrame,
    scaling_policy: str,
) -> Dict[str, Any]:
    auditor = LeakageAuditor()

    # Track train-only fit for scaler.
    if len(scaler_fit_df) > 0:
        auditor.register_scaler_fit(
            scaler_id="benchmark_scaler",
            fit_data=scaler_fit_df[feature_cols].to_numpy(),
            fit_indices=(0, len(scaler_fit_df) - 1),
            fit_dates=(str(scaler_fit_df["time"].min()), str(scaler_fit_df["time"].max())),
            feature_names=feature_cols,
        )

    train_idx = np.arange(0, len(train_df))
    val_idx = np.arange(len(train_df), len(train_df) + len(val_df))
    test_idx = np.arange(len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df))
    timestamps = pd.DatetimeIndex(pd.concat([train_df["time"], val_df["time"], test_df["time"]], ignore_index=True))

    split_warnings = auditor.check_split_leakage(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        timestamps=timestamps,
    )

    if scaling_policy == "train_only":
        scaler_warnings = auditor.audit_scaler_splits(
            scaler_id="benchmark_scaler",
            train_indices=(0, len(train_df) - 1),
            val_indices=(len(train_df), len(train_df) + len(val_df) - 1) if len(val_df) else None,
            test_indices=(
                len(train_df) + len(val_df),
                len(train_df) + len(val_df) + len(test_df) - 1,
            )
            if len(test_df)
            else None,
        )
    else:
        scaler_warnings = [
            LeakageWarning(
                severity=LeakageSeverity.INFO,
                category="scaling_policy",
                message="Scaler fit uses full dataset (ablation/improper preprocessing)",
                details={"scaling_policy": scaling_policy},
                suggested_fix="Use scaling_policy=train_only for fair benchmark",
            )
        ]

    sequence_warnings = auditor.check_sequence_leakage(
        sequences=train_X,
        train_end_idx=len(train_df) - 1,
        sequence_length=sequence_length,
        forecast_horizon=horizon,
    )

    warnings = split_warnings + scaler_warnings + sequence_warnings
    critical = [w for w in warnings if w.severity == LeakageSeverity.CRITICAL]

    return {
        "passed": len(critical) == 0,
        "critical_count": len(critical),
        "warning_count": len([w for w in warnings if w.severity == LeakageSeverity.WARNING]),
        "info_count": len([w for w in warnings if w.severity == LeakageSeverity.INFO]),
        "warnings": [w.to_dict() for w in warnings],
    }


def build_benchmark_dataset(config) -> Dict[str, Any]:
    """
    Build the benchmark dataset for SPY next-day adjusted log return with a
    shared multi-asset feature set.

    Returns dict with splits, scaler info, sequences, and metadata.
    """

    data_cfg = config.data
    universe = universe_from_config(data_cfg)
    qa_policy = dict(QA_POLICY_DEFAULT)
    price_col = getattr(data_cfg, "price_column", "adjusted_close")
    missing_policy = getattr(data_cfg, "missing_policy", "leakage_safe")
    scaling_policy = getattr(data_cfg, "scaling_policy", "train_only")
    target_type = getattr(data_cfg, "target_type", "next_day_log_return")
    target_vol_window = int(getattr(data_cfg, "target_vol_window", 5))
    default_fill_limit = int(getattr(data_cfg, "default_forward_fill_limit", 1))
    per_symbol_limit = dict(getattr(data_cfg, "per_symbol_forward_fill_limit", {}) or {})
    disable_qa = getattr(data_cfg, "disable_qa", False)

    fetcher = DataFetcher(config)
    processed_cache = CacheManager(Path(config.data_dir) / data_cfg.cache_dir)
    raw_cache_paths = fetcher.cache.paths(
        universe=universe,
        start_date=data_cfg.start_date,
        end_date=data_cfg.end_date,
        interval=data_cfg.interval,
        dataset_tag="raw_cache",
    )
    raw = fetcher.fetch_multi_asset_cached(
        start_date=data_cfg.start_date,
        end_date=data_cfg.end_date,
        interval=data_cfg.interval,
        force_refresh=data_cfg.force_refresh,
        dataset_tag="raw_cache",
    )
    if raw.empty:
        raise ValueError("Benchmark dataset build failed: raw multi-asset fetch returned no rows")

    master_calendar = build_master_calendar(
        data_cfg.start_date,
        data_cfg.end_date,
        data_cfg.calendar,
        data_cfg.master_calendar_holidays,
    )

    qa_raw = run_qa(raw, expected_calendar=master_calendar)
    if not disable_qa:
        _enforce_qa_gate(qa_raw, qa_policy, stage="raw")

    aligned, alignment_report = align_to_calendar(
        raw,
        master_calendar,
        default_forward_fill_limit=default_fill_limit,
        per_symbol_forward_fill_limit=per_symbol_limit,
        return_report=True,
    )
    qa_aligned = run_qa(aligned, expected_calendar=master_calendar)
    if not disable_qa:
        _enforce_qa_gate(qa_aligned, qa_policy, stage="aligned")

    # Feature engineering
    feats = aligned.copy()
    feats = add_adjusted_returns(feats, price_col=price_col, horizons=[1, 5])
    feats = add_rolling_volatility(feats, return_col="adj_return_1d", windows=[10, 20])
    feats = add_momentum_trend(feats, return_col="adj_return_1d", windows=[20, 60])
    feats = add_cross_asset_spreads(
        feats,
        pairs=[("SPY", "QQQ"), ("SPY", "IWM")],
        price_col=price_col,
    )
    feats = add_regime_markers(feats, vol_col="rolling_vol_20", threshold=0.02)

    # Cross-asset context (levels/changes) merged on time.
    pivot = feats.pivot_table(index="time", columns="ticker", values=price_col)
    context = pd.DataFrame(index=pivot.index)

    if "^VIX" in pivot.columns:
        context["vix_level"] = pivot["^VIX"]
        context["vix_change"] = _safe_log_return(pivot["^VIX"])

    if "^TNX" in pivot.columns:
        context["tnx_yield"] = pivot["^TNX"] / 100.0

    if "GC=F" in pivot.columns:
        context["commodity_gc_ret"] = _safe_log_return(pivot["GC=F"])

    if "CL=F" in pivot.columns:
        context["commodity_cl_ret"] = _safe_log_return(pivot["CL=F"])

    context = context.reset_index()
    feats = feats.merge(context, on="time", how="left")

    featurized_full = feats.copy()

    # Target branching
    target_cols: List[str] = []
    if target_type == "next_day_log_return":
        feats = add_next_day_log_return(
            feats,
            price_col=price_col,
            target_col=data_cfg.target_column,
            group_col="ticker",
        )
        target_cols = [data_cfg.target_column]
    elif target_type == "realized_vol":
        vol_col = data_cfg.target_column or "target_vol"
        feats = add_realized_vol(
            feats,
            price_col=price_col,
            window=target_vol_window,
            target_col=vol_col,
            group_col="ticker",
        )
        target_cols = [vol_col]
    elif target_type == "joint_return_vol":
        return_col = data_cfg.target_column or "target_return"
        vol_col = f"{return_col}_vol"
        feats = add_joint_return_and_vol(
            feats,
            price_col=price_col,
            return_col=return_col,
            vol_col=vol_col,
            vol_window=target_vol_window,
            group_col="ticker",
        )
        target_cols = [return_col, vol_col]
    else:
        raise ValueError(f"Unsupported target_type={target_type}")

    feats = feats[feats["ticker"] == data_cfg.target_symbol].copy()
    feats = feats.sort_values("time")

    # Resolve robust feature set (required + optional available).
    effective_feature_cols, missing_optional, missing_required = _resolve_feature_columns(
        feats,
        list(getattr(data_cfg, "base_feature_columns", [])),
    )

    if missing_required:
        raise ValueError(
            "Benchmark dataset missing required core features: "
            f"{missing_required}. Check universe symbols and upstream data quality."
        )

    # Missing data policy (defaults to leakage-safe forward fill + dropna).
    if missing_policy == "improper_backfill":
        feats[effective_feature_cols] = feats[effective_feature_cols].ffill().bfill()
    else:
        feats[effective_feature_cols] = feats[effective_feature_cols].ffill()

    # Drop rows with missing required features or target unless policy allows carry.
    if missing_policy != "allow_missing":
        feats = feats.dropna(subset=effective_feature_cols + target_cols)
    if feats.empty:
        raise ValueError("Benchmark dataset has no rows after feature/target filtering")

    qa_model = run_qa(feats[["time", "ticker", price_col]].copy(), expected_calendar=master_calendar)

    # Split
    train_df, val_df, test_df = temporal_split_with_gap(
        feats,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        test_ratio=data_cfg.test_ratio,
        gap=0,
    )

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced empty partition(s): "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    # Scaling (train-only by default; ablations may override)
    if scaling_policy == "full_dataset":
        scaler_fit_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    else:
        scaler_fit_df = train_df

    scaler = fit_scaler(scaler_fit_df, effective_feature_cols)
    scaler_path = Path(config.data_dir) / data_cfg.cache_dir / "scalers" / f"scaler_{data_cfg.target_symbol}.json"
    save_scaler(
        scaler,
        scaler_path,
        effective_feature_cols,
        metadata={
            "fit_start": str(scaler_fit_df["time"].min()) if not scaler_fit_df.empty else None,
            "fit_end": str(scaler_fit_df["time"].max()) if not scaler_fit_df.empty else None,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "scaling_policy": scaling_policy,
        },
    )

    train_df_scaled = apply_scaler(train_df, scaler, effective_feature_cols)
    val_df_scaled = apply_scaler(val_df, scaler, effective_feature_cols)
    test_df_scaled = apply_scaler(test_df, scaler, effective_feature_cols)

    # Sequences
    lookback = int(data_cfg.sequence_length)
    horizon = int(data_cfg.forecast_horizon)

    sequences: Dict[str, Any] = {}

    if target_type == "joint_return_vol":
        return_col, vol_col = target_cols
        X_train, y_train_ret, _ = build_sequences(
            train_df_scaled,
            effective_feature_cols,
            return_col,
            lookback,
            horizon,
            group_col="ticker",
        )
        X_val, y_val_ret, _ = build_sequences(
            val_df_scaled,
            effective_feature_cols,
            return_col,
            lookback,
            horizon,
            group_col="ticker",
        )
        X_test, y_test_ret, _ = build_sequences(
            test_df_scaled,
            effective_feature_cols,
            return_col,
            lookback,
            horizon,
            group_col="ticker",
        )

        _, y_train_vol, _ = build_sequences(
            train_df_scaled,
            effective_feature_cols,
            vol_col,
            lookback,
            horizon,
            group_col="ticker",
        )
        _, y_val_vol, _ = build_sequences(
            val_df_scaled,
            effective_feature_cols,
            vol_col,
            lookback,
            horizon,
            group_col="ticker",
        )
        _, y_test_vol, _ = build_sequences(
            test_df_scaled,
            effective_feature_cols,
            vol_col,
            lookback,
            horizon,
            group_col="ticker",
        )

        sequences = {
            "return": {"train": (X_train, y_train_ret), "val": (X_val, y_val_ret), "test": (X_test, y_test_ret)},
            "vol": {"train": (X_train, y_train_vol), "val": (X_val, y_val_vol), "test": (X_test, y_test_vol)},
        }

        if len(X_train) == 0 or len(y_train_vol) == 0:
            raise ValueError("Sequence generation returned empty arrays for joint target")
    else:
        X_train, y_train, _ = build_sequences(
            train_df_scaled,
            effective_feature_cols,
            target_cols[0],
            lookback,
            horizon,
            group_col="ticker",
        )
        X_val, y_val, _ = build_sequences(
            val_df_scaled,
            effective_feature_cols,
            target_cols[0],
            lookback,
            horizon,
            group_col="ticker",
        )
        X_test, y_test, _ = build_sequences(
            test_df_scaled,
            effective_feature_cols,
            target_cols[0],
            lookback,
            horizon,
            group_col="ticker",
        )

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError(
                "Sequence generation returned empty arrays. "
                f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

        sequences = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    leakage_report = _compute_leakage_report(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_X=X_train,
        sequence_length=lookback,
        horizon=horizon,
        feature_cols=effective_feature_cols,
        scaler_fit_df=scaler_fit_df,
        scaling_policy=scaling_policy,
    )

    if not leakage_report["passed"]:
        critical_messages = [
            w["message"]
            for w in leakage_report["warnings"]
            if w.get("severity") == LeakageSeverity.CRITICAL.value
        ]
        raise ValueError(
            "Leakage audit failed with critical issues: "
            + "; ".join(critical_messages)
        )

    fp = dataset_fingerprint(
        universe,
        effective_feature_cols,
        data_cfg.target_column,
        lookback,
        horizon,
        data_cfg.start_date,
        data_cfg.end_date,
    )

    processed_paths = processed_cache.paths(
        universe=universe,
        start_date=data_cfg.start_date,
        end_date=data_cfg.end_date,
        interval=data_cfg.interval,
        dataset_tag=f"processed_{fp}",
    )
    processed_cache.save(
        processed_paths,
        featurized_full,
        metadata={
            "fingerprint": fp,
            "universe": asdict(universe),
            "features": effective_feature_cols,
            "target_symbol": data_cfg.target_symbol,
            "start_date": data_cfg.start_date,
            "end_date": data_cfg.end_date,
            "interval": data_cfg.interval,
        },
        qa_report=qa_aligned,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(config.project_root) / "results" / "dataset_artifacts" / f"{fp}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    split_boundaries = {
        "train": {
            "start": str(train_df["time"].min()) if not train_df.empty else None,
            "end": str(train_df["time"].max()) if not train_df.empty else None,
        },
        "val": {
            "start": str(val_df["time"].min()) if not val_df.empty else None,
            "end": str(val_df["time"].max()) if not val_df.empty else None,
        },
        "test": {
            "start": str(test_df["time"].min()) if not test_df.empty else None,
            "end": str(test_df["time"].max()) if not test_df.empty else None,
        },
    }

    split_ratios = {
        "train": data_cfg.train_ratio,
        "val": data_cfg.val_ratio,
        "test": data_cfg.test_ratio,
    }

    sequence_shapes = {
        "lookback": lookback,
        "horizon": horizon,
        "feature_count": len(effective_feature_cols),
        "samples": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
    }

    cache_keys = {
        "raw": raw_cache_paths.cache_key(),
        "processed": processed_paths.cache_key(),
    }

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": asdict(universe),
        "target_symbol": data_cfg.target_symbol,
        "target_type": data_cfg.target_type,
        "target_columns": target_cols,
        "target_vol_window": target_vol_window if "vol" in target_type else None,
        "feature_columns": effective_feature_cols,
        "missing_optional_feature_columns": missing_optional,
        "target_column": data_cfg.target_column,
        "lookback": lookback,
        "horizon": horizon,
        "start_date": data_cfg.start_date,
        "end_date": data_cfg.end_date,
        "fingerprint": fp,
        "alignment_report": alignment_report,
        "split_sizes": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "test_sequences": len(X_test),
        },
        "split_boundaries": split_boundaries,
        "qa_policy": qa_policy,
        "qa_disabled": disable_qa,
        "qa_raw": qa_raw,
        "qa_aligned": qa_aligned,
        "qa_model": qa_model,
        "leakage_report": leakage_report,
        "scaler_path": str(scaler_path),
        "price_column": price_col,
        "missing_policy": missing_policy,
        "scaling_policy": scaling_policy,
        "artifact_dir": str(artifact_dir),
        "cache_keys": cache_keys,
    }

    fairness_contract = _build_fairness_contract(
        meta=meta,
        feature_columns=effective_feature_cols,
        required_features=REQUIRED_CORE_FEATURES,
        cache_keys=cache_keys,
        scaler_path=str(scaler_path),
        sequence_shapes=sequence_shapes,
        split_ratios=split_ratios,
    )
    meta["fairness_contract_path"] = str(artifact_dir / "fairness_contract.json")
    meta["fairness_contract"] = fairness_contract

    _write_json(artifact_dir / "dataset_metadata.json", meta)
    _write_json(artifact_dir / "qa_raw.json", qa_raw)
    _write_json(artifact_dir / "qa_aligned.json", qa_aligned)
    _write_json(artifact_dir / "qa_model.json", qa_model)
    _write_json(artifact_dir / "leakage_report.json", leakage_report)
    _write_json(artifact_dir / "fairness_contract.json", fairness_contract)

    suffix = target_type

    train_df.to_parquet(artifact_dir / f"train_unscaled_{suffix}.parquet", index=False)
    val_df.to_parquet(artifact_dir / f"val_unscaled_{suffix}.parquet", index=False)
    test_df.to_parquet(artifact_dir / f"test_unscaled_{suffix}.parquet", index=False)

    train_df_scaled.to_parquet(artifact_dir / f"train_scaled_{suffix}.parquet", index=False)
    val_df_scaled.to_parquet(artifact_dir / f"val_scaled_{suffix}.parquet", index=False)
    test_df_scaled.to_parquet(artifact_dir / f"test_scaled_{suffix}.parquet", index=False)

    return {
        "train_df": train_df_scaled,
        "val_df": val_df_scaled,
        "test_df": test_df_scaled,
        "train_df_unscaled": train_df,
        "val_df_unscaled": val_df,
        "test_df_unscaled": test_df,
        "scaler_path": str(scaler_path),
        "sequences": sequences,
        "metadata": meta,
        "artifact_dir": str(artifact_dir),
        "fairness_contract": fairness_contract,
    }
