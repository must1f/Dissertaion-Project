"""Structured result logging for evaluations.

Persists results to both JSON (for quick inspection) and SQLite (via
ResultsDatabase). Designed to be a thin wrapper that can be called from the
evaluation harness or CLI entrypoints without coupling to training code.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import shutil
from typing import Any, Dict, Optional

from .leaderboard import ResultsDatabase, ExperimentEntry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResultLogger:
    """Handles dual logging: JSON artifact + SQLite persistence."""

    def __init__(self, db_path: Path | str = "results/experiments.db", artifacts_dir: Path | str = "results/artifacts"):
        db_path = Path(db_path)
        try:
            self.db = ResultsDatabase(db_path)
        except sqlite3.OperationalError as exc:  # pragma: no cover - safety for schema drift
            backup = db_path.with_suffix(".bak")
            shutil.move(db_path, backup)
            logger.warning(f"Corrupt or old schema at {db_path}, backed up to {backup}. Recreating fresh DB.")
            self.db = ResultsDatabase(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        entry: ExperimentEntry,
        config: Optional[Dict[str, Any]] = None,
        predictions: Optional[Dict[str, Any]] = None,
        stat_tests: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist an experiment run and return artifact path."""
        self.db.save_experiment(entry, config=config)
        if entry.regime_sharpe:
            self.db.save_regime_results(entry.experiment_id, entry.regime_sharpe)

        artifact_path = self._save_json(entry, config, predictions, stat_tests)
        logger.info(f"Logged experiment {entry.experiment_id} → {artifact_path}")
        return artifact_path

    def _save_json(
        self,
        entry: ExperimentEntry,
        config: Optional[Dict[str, Any]],
        predictions: Optional[Dict[str, Any]],
        stat_tests: Optional[Dict[str, Any]],
    ) -> Path:
        payload = {
            "experiment": entry.__dict__,
            "config": config,
            "predictions": predictions,
            "stat_tests": stat_tests,
        }
        path = self.artifacts_dir / f"{entry.experiment_id}.json"
        with path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path
