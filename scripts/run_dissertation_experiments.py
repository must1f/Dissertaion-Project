#!/usr/bin/env python3
"""
Run dissertation experiment audit pipeline.

This script audits model artifacts, enforces the experiment contract, and
generates grouped outputs suitable for dissertation reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dissertation_pipeline import (  # noqa: E402
    DissertationPipelineRunner,
    load_dissertation_config,
    validate_dissertation_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dissertation experiment pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/dissertation.yaml"),
        help="Path to dissertation pipeline YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--no-persist-db",
        action="store_true",
        help="Do not write experiment rows to results database.",
    )
    args = parser.parse_args()

    config = load_dissertation_config(args.config)
    issues = validate_dissertation_config(config)
    if issues:
        for issue in issues:
            print(f"[CONFIG ERROR] {issue}")
        raise SystemExit(1)

    runner = DissertationPipelineRunner(config=config, config_path=args.config)
    summary = runner.run(output_dir=args.output_dir, persist_db=not args.no_persist_db)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
