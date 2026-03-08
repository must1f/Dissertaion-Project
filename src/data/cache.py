"""
Local Parquet-based cache for raw and processed datasets.

Design goals:
- File-system only (no DB dependency) using Parquet + JSON metadata
- Deterministic paths keyed by universe hash, date range, interval, and feature/target hashes
- Force-refresh support to bypass cached artifacts
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
from datetime import datetime, timezone

from .universe import UniverseDefinition


@dataclass
class CachePaths:
    root: Path
    universe: UniverseDefinition
    start_date: str
    end_date: str
    interval: str
    dataset_tag: str = "raw"

    def base_dir(self) -> Path:
        return self.root / self.dataset_tag / self.universe.hash() / f"{self.start_date}_{self.end_date}_{self.interval}"

    def parquet_path(self) -> Path:
        return self.base_dir() / "data.parquet"

    def metadata_path(self) -> Path:
        return self.base_dir() / "metadata.json"

    def qa_path(self) -> Path:
        return self.base_dir() / "qa_report.json"

    def actions_path(self) -> Path:
        return self.base_dir() / "actions.parquet"

    def cache_key(self) -> str:
        return f"{self.dataset_tag}::{self.universe.hash()}::{self.start_date}_{self.end_date}_{self.interval}"


class CacheManager:
    """File-based cache for datasets and metadata."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def paths(self, universe: UniverseDefinition, start_date: str, end_date: str, interval: str, dataset_tag: str = "raw") -> CachePaths:
        return CachePaths(
            root=self.root,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            dataset_tag=dataset_tag,
        )

    def load(self, paths: CachePaths, ttl_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        df, _, _ = self.load_with_meta(paths, ttl_days=ttl_days)
        return df

    def load_with_meta(self, paths: CachePaths, ttl_days: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        parquet_path = paths.parquet_path()
        if not parquet_path.exists():
            return None, None, None

        meta = None
        if paths.metadata_path().exists():
            with open(paths.metadata_path()) as f:
                meta = json.load(f)

        qa = None
        if paths.qa_path().exists():
            with open(paths.qa_path()) as f:
                qa = json.load(f)

        if ttl_days is not None and meta is not None:
            saved_at = meta.get("saved_at")
            try:
                if saved_at:
                    saved_dt = datetime.fromisoformat(saved_at)
                    age = datetime.now(timezone.utc) - saved_dt
                    if age.total_seconds() > ttl_days * 86400:
                        return None, meta, qa
            except Exception:
                return None, meta, qa

        return pd.read_parquet(parquet_path), meta, qa

    def save(self, paths: CachePaths, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None, qa_report: Optional[Dict[str, Any]] = None, actions: Optional[pd.DataFrame] = None) -> None:
        paths.base_dir().mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths.parquet_path(), index=False)

        meta = metadata or {}
        meta.setdefault("universe", paths.universe.to_dict())
        meta.setdefault("start_date", paths.start_date)
        meta.setdefault("end_date", paths.end_date)
        meta.setdefault("interval", paths.interval)
        meta.setdefault("dataset_tag", paths.dataset_tag)
        meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        meta.setdefault("cache_key", paths.cache_key())

        with open(paths.metadata_path(), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        if qa_report is not None:
            with open(paths.qa_path(), "w") as f:
                json.dump(qa_report, f, indent=2, default=str)

        if actions is not None and not actions.empty:
            actions.to_parquet(paths.actions_path(), index=False)
