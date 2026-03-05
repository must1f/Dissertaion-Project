#!/usr/bin/env python3
"""
Promote a checkpoint to the registry.

Usage:
    python scripts/promote_model.py --key pinn_gbm_v2 --checkpoint Models/checkpoints/pinn_gbm_v2_seed42.pt --metric sharpe_ratio 1.23
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

# Ensure src is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model_registry import ModelRegistry, RegistryEntry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model checkpoint to the registry.")
    parser.add_argument("--key", required=True, help="Registry key (e.g., pinn_gbm_v2_seed42)")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint file")
    parser.add_argument("--metric", action="append", nargs=2, metavar=("name", "value"), help="Metric name/value pairs")
    parser.add_argument("--regime", help="Optional regime tag")
    parser.add_argument("--model-name", help="Model display name", default=None)
    parser.add_argument("--git-commit", help="Git commit hash", default=None)
    parser.add_argument("--registry-path", type=Path, default=Path("Models/registry.json"))
    args = parser.parse_args()

    metrics = {}
    if args.metric:
        for name, val in args.metric:
            try:
                metrics[name] = float(val)
            except ValueError:
                metrics[name] = val

    entry = RegistryEntry(
        model_name=args.model_name or args.key,
        path=str(args.checkpoint),
        metrics=metrics,
        regime=args.regime,
        git_commit=args.git_commit,
    )

    registry = ModelRegistry(args.registry_path)
    registry.register(args.key, entry)
    print(f"Registered {args.key} -> {args.registry_path}")
    print(json.dumps(entry.__dict__, indent=2))


if __name__ == "__main__":
    main()
