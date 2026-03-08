"""
Lightweight experiment tracking utilities for datasets/configs/metrics.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def save_run(output_dir: Path, run_metadata: Dict[str, Any], metrics: Dict[str, Any], artifacts: Optional[Dict[str, Any]] = None) -> Path:
    """Persist run metadata and metrics to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": run_metadata,
        "metrics": metrics,
    }
    if artifacts:
        payload["artifacts"] = artifacts

    out_path = output_dir / "run_summary.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
