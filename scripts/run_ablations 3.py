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
import yaml

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ablation_runner import AblationRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation configs.")
    parser.add_argument("--ablation", required=True, help="Ablation name (as defined in configs/ablations.yaml)")
    parser.add_argument("--base", required=True, type=Path, help="Path to base experiment YAML")
    parser.add_argument("--ablations-file", default=Path("configs/ablations.yaml"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/ablations"), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print merged configs to stdout instead of writing files")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.base.read_text()) or {}
    runner = AblationRunner(args.ablations_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

