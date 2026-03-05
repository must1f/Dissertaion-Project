"""
CLI entry point for running evaluation harness with a config.

Minimal usage (placeholder until full training/eval wiring):
    python -m src.evaluation --config configs/eval.yaml

Currently this CLI validates the config file and prints the parsed options.
Hook your training/prediction pipeline here to generate predictions, then call
`EvaluationHarness.evaluate` and `ResultLogger` will persist results.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch

from .evaluation_harness import EvaluationHarness
from ..models.model_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation harness.")
    parser.add_argument("--config", type=Path, required=True, help="Path to eval config YAML")
    parser.add_argument("--predictions", type=Path, help="Path to predictions (npy or csv single column)")
    parser.add_argument("--targets", type=Path, help="Path to targets (npy or csv single column)")
    parser.add_argument("--returns", type=Path, help="Optional realized returns vector (npy or csv)")
    parser.add_argument("--model-key", type=str, default="model", help="Model key/name")
    parser.add_argument("--timestamps", type=Path, help="Optional timestamps CSV column 'time'")
    parser.add_argument("--persist-windows", action=argparse.BooleanOptionalAction, default=True,
                        help="Persist walk-forward window metrics (default: true)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval_runs"), help="Output directory for logs/DBs")
    parser.add_argument("--window-db", type=Path, help="Optional explicit path for window DB (sqlite)")
    parser.add_argument("--save-predictions", action="store_true", help="Persist predictions/targets in results DB")
    # Optional: load checkpoint + input CSV to generate predictions on the fly
    parser.add_argument("--checkpoint", type=Path, help="Path to model checkpoint (torch save).")
    parser.add_argument("--model-type", type=str, help="Model type key for ModelRegistry (e.g., lstm, transformer, stacked).")
    parser.add_argument("--registry-key", type=str, help="Lookup checkpoint from Models/registry.json (e.g., stacked_pinn_seed42).")
    parser.add_argument("--input-csv", type=Path, help="CSV with features (and target column if evaluating).")
    parser.add_argument("--target-col", type=str, help="Name of target column in input CSV.")
    parser.add_argument("--feature-cols", type=str, help="Comma-separated feature columns (defaults to all except target).")
    parser.add_argument("--dataloader", type=str, choices=["csv"], default="csv",
                        help="Data loader to use when running inference from checkpoint (default: csv).")
    args = parser.parse_args()

    cfg_path = args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(cfg_path.read_text())
    print("Loaded evaluation config:")
    print(yaml.safe_dump(config, sort_keys=False))

    # Instantiate harness
    harness = EvaluationHarness(
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        window_db_path=args.window_db
    )

    # If predictions/targets provided, run a single evaluation
    if args.predictions and args.targets:
        preds = _load_vector(args.predictions)
        targs = _load_vector(args.targets)
        rets = _load_vector(args.returns) if args.returns else None
        timestamps = None
        if args.timestamps:
            ts_df = pd.read_csv(args.timestamps)
            col = ts_df.columns[0]
            timestamps = pd.to_datetime(ts_df[col])

        result = harness.evaluate(
            predictions=preds,
            targets=targs,
            model_key=args.model_key,
            model_name=args.model_key,
            timestamps=timestamps,
            persist_windows=args.persist_windows,
            returns=rets
        )
        print("Evaluation complete. Metrics:")
        print(json.dumps({**result.ml_metrics, **result.financial_metrics}, indent=2))
    # Alternatively, load checkpoint + CSV to produce predictions directly
    elif (args.checkpoint or args.registry_key) and args.input_csv and args.target_col:
        checkpoint_path = args.checkpoint
        model_type = args.model_type

        if args.registry_key:
            registry = ModelRegistry(Path("."))
            entry = registry.get(args.registry_key)
            if entry is None:
                print(f"Registry key not found: {args.registry_key}", file=sys.stderr)
                sys.exit(1)
            checkpoint_path = Path(entry.path)
            if not model_type:
                model_type = args.registry_key.split("_")[0]

        if checkpoint_path is None or model_type is None:
            print("Checkpoint path or model type missing for inference run.", file=sys.stderr)
            sys.exit(1)

        preds, targs, timestamps = _infer_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            csv_path=args.input_csv,
            target_col=args.target_col,
            feature_cols=args.feature_cols.split(",") if args.feature_cols else None,
        )
        result = harness.evaluate(
            predictions=preds,
            targets=targs,
            model_key=args.model_key,
            model_name=args.model_key,
            timestamps=timestamps,
            persist_windows=args.persist_windows,
            returns=None
        )
        print("Evaluation complete (from checkpoint). Metrics:")
        print(json.dumps({**result.ml_metrics, **result.financial_metrics}, indent=2))
    else:
        print("EvaluationHarness is ready. Provide --predictions/--targets or --checkpoint+--input-csv to run eval.")


def _load_vector(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    # assume CSV
    df = pd.read_csv(path)
    col = df.columns[0]
    return df[col].to_numpy()


def _infer_from_checkpoint(
    checkpoint_path: Path,
    model_type: str,
    csv_path: Path,
    target_col: str,
    feature_cols: Optional[list[str]] = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[pd.DatetimeIndex]]:
    """Load model from checkpoint and run inference on CSV."""
    df = pd.read_csv(csv_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col and c.lower() not in {"timestamp", "time", "date"}]

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in CSV")

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    timestamps = None
    for ts_col in ["timestamp", "time", "date"]:
        if ts_col in df.columns:
            timestamps = pd.to_datetime(df[ts_col])
            break

    registry = ModelRegistry(Path("."))
    model = registry.create_model(model_type=model_type, input_dim=X.shape[1])
    if model is None:
        raise ValueError(f"Could not create model for type {model_type}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32))
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.squeeze().cpu().numpy()

    return preds, y, timestamps


if __name__ == "__main__":
    main()
