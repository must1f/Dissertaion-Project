#!/usr/bin/env python3
"""
Run ablation configs and emit merged YAMLs.

This is a thin wrapper; it does not train models. It produces per-variant
config files you can feed into your training/evaluation pipeline.

Example:
    python scripts/run_ablations.py --ablation curriculum --base configs/experiments/pinn_gbm.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import json

from datetime import datetime

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ablation_runner import AblationRunner  # noqa: E402
from scripts import train_models  # noqa: E402


class DummyDataset:
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):  # pragma: no cover (unused in smoke mode)
        raise IndexError


def _smoke_fairness_contract(target_type: str) -> Dict[str, Any]:
    return {
        "dataset": {
            "fingerprint": f"smoke-{target_type}",
            "target_symbol": "SPY",
            "target_type": target_type,
            "target_column": "target" if not target_type.startswith("realized") else "realized_vol",
            "price_column": "adjusted_close",
            "lookback": 5,
            "horizon": 1,
            "start_date": "2020-01-01",
            "end_date": "2020-01-10",
        },
        "features": {
            "required_core": [],
            "effective": [],
        },
    }


def run_ablation_smoke(ablation_name: str, output_dir: Path, model: str = "lstm", target_type: str = "next_day_log_return") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_ds = DummyDataset()
    treatment_ds = DummyDataset()
    fairness_contract = _smoke_fairness_contract(target_type)
    train_models._CORE_CONTRACT_SIGNATURE = None

    baseline_result = train_models.train_single_model(
        model_type=model,
        train_dataset=baseline_ds,
        val_dataset=baseline_ds,
        test_dataset=baseline_ds,
        input_dim=4,
        epochs=1,
        research_mode=False,
        scalers=None,
        feature_cols=[],
        target_type=target_type,
        fairness_contract=fairness_contract,
        dataset_meta={"target_type": target_type, "fingerprint": fairness_contract["dataset"]["fingerprint"]},
        run_dir=output_dir,
        regime_context=None,
        allow_target_mismatch=True,
        smoke_test=True,
    )

    train_models._CORE_CONTRACT_SIGNATURE = None
    treatment_result = train_models.train_single_model(
        model_type=model,
        train_dataset=treatment_ds,
        val_dataset=treatment_ds,
        test_dataset=treatment_ds,
        input_dim=4,
        epochs=1,
        research_mode=False,
        scalers=None,
        feature_cols=[],
        target_type=target_type,
        fairness_contract=fairness_contract,
        dataset_meta={"target_type": target_type, "fingerprint": fairness_contract["dataset"]["fingerprint"]},
        run_dir=output_dir,
        regime_context=None,
        allow_target_mismatch=True,
        smoke_test=True,
    )

    rows = []
    for label, result in [("baseline", baseline_result), ("treatment", treatment_result)]:
        metrics = result.get("test_metrics", {})
        rows.append({
            "ablation": ablation_name,
            "variant": label,
            "model": model,
            "rmse": metrics.get("rmse"),
            "sharpe": metrics.get("sharpe_ratio"),
        })

    comp_path = output_dir / f"{ablation_name}_comparison.csv"
    comp_path.write_text("ablation,variant,model,rmse,sharpe\n" + "\n".join([
        f"{r['ablation']},{r['variant']},{r['model']},{r['rmse']},{r['sharpe']}" for r in rows
    ]))
    return comp_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation configs.")
    parser.add_argument("--ablation", required=True, help="Ablation name (as defined in configs/ablations.yaml)")
    parser.add_argument("--base", required=True, type=Path, help="Path to base experiment YAML")
    parser.add_argument("--ablations-file", default=Path("configs/ablations.yaml"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/ablations"), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print merged configs to stdout instead of writing files")
    parser.add_argument("--execute", action="store_true", help="Run baseline+treatment (smoke) instead of just emitting configs")
    parser.add_argument("--smoke", action="store_true", help="Use smoke-test path for execution (fast)")
    parser.add_argument("--model", default="lstm", help="Model key to train for execution mode")
    parser.add_argument("--target-type", default="next_day_log_return", help="Target type for execution mode")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.base.read_text()) or {}
    runner = AblationRunner(args.ablations_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.execute:
        comp_path = run_ablation_smoke(
            ablation_name=args.ablation,
            output_dir=args.output_dir,
            model=args.model,
            target_type=args.target_type,
        )
        print(f"Smoke execution complete -> {comp_path}")
    else:
        for variant in runner.generate(args.ablation, base_cfg):
            if args.dry_run:
                print(f"# {variant.name}")
                print(yaml.safe_dump(variant.config, sort_keys=False))
                print()
            else:
                out_path = args.output_dir / f"{variant.name}.yaml"
                out_path.write_text(yaml.safe_dump(variant.config, sort_keys=False))
                print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
