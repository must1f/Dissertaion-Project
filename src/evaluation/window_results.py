"""
Window-level result storage and aggregation for walk-forward validation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class WindowMetrics:
    """Complete metrics for a single validation window"""
    window_id: int
    start_date: Optional[str]
    end_date: Optional[str]
    start_idx: int
    end_idx: int
    n_samples: int

    # Performance metrics
    sharpe: float
    sortino: float
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    calmar_ratio: float

    # Prediction metrics (if available)
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    directional_accuracy: Optional[float] = None

    # Regime information
    regime: Optional[str] = None
    avg_volatility: Optional[float] = None

    # Metadata
    model_name: Optional[str] = None
    experiment_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WindowAggregation:
    """Aggregated statistics across windows"""
    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    count: int


@dataclass
class ExtendedWalkForwardResult:
    """Extended results with full window-level detail"""
    summary: Any
    window_metrics: List[WindowMetrics]
    aggregations: Dict[str, WindowAggregation]
    regime_breakdown: Dict[str, Dict[str, float]]
    experiment_id: str
    model_name: str
    config: Dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class WindowResultsDatabase:
    """
    SQLite database for persisting window-level results.
    """

    def __init__(self, db_path: Union[str, Path] = "results/walk_forward_results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    config_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS window_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    window_id INTEGER NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    start_idx INTEGER,
                    end_idx INTEGER,
                    n_samples INTEGER,
                    sharpe REAL,
                    sortino REAL,
                    total_return REAL,
                    annualized_return REAL,
                    volatility REAL,
                    max_drawdown REAL,
                    calmar_ratio REAL,
                    mse REAL,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    directional_accuracy REAL,
                    regime TEXT,
                    avg_volatility REAL,
                    metrics_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS aggregations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    mean REAL,
                    std REAL,
                    min_val REAL,
                    max_val REAL,
                    median REAL,
                    q25 REAL,
                    q75 REAL,
                    count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_window_experiment
                ON window_results(experiment_id);

                CREATE INDEX IF NOT EXISTS idx_window_regime
                ON window_results(regime);
            """)

    def save_experiment(self, experiment_id: str, model_name: str, config: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments (id, model_name, config_json) VALUES (?, ?, ?)",
                (experiment_id, model_name, json.dumps(config))
            )

    def save_window_results(self, results: List[WindowMetrics], experiment_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for r in results:
                conn.execute("""
                    INSERT INTO window_results
                    (experiment_id, window_id, start_date, end_date, start_idx, end_idx,
                     n_samples, sharpe, sortino, total_return, annualized_return,
                     volatility, max_drawdown, calmar_ratio, mse, mae, rmse, mape,
                     directional_accuracy, regime, avg_volatility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, r.window_id, r.start_date, r.end_date,
                    r.start_idx, r.end_idx, r.n_samples, r.sharpe, r.sortino,
                    r.total_return, r.annualized_return, r.volatility,
                    r.max_drawdown, r.calmar_ratio, r.mse, r.mae, r.rmse,
                    r.mape, r.directional_accuracy, r.regime, r.avg_volatility
                ))

    def save_aggregations(self, aggregations: Dict[str, WindowAggregation], experiment_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for name, agg in aggregations.items():
                conn.execute("""
                    INSERT INTO aggregations
                    (experiment_id, metric_name, mean, std, min_val, max_val,
                     median, q25, q75, count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, name, agg.mean, agg.std, agg.min, agg.max,
                    agg.median, agg.q25, agg.q75, agg.count
                ))

    def load_window_results(self, experiment_id: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM window_results WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )


def compute_window_aggregations(window_metrics: List[WindowMetrics]) -> Dict[str, WindowAggregation]:
    """Compute aggregated statistics across windows."""
    if not window_metrics:
        return {}

    records = []
    for w in window_metrics:
        records.append({
            'sharpe': w.sharpe,
            'sortino': w.sortino,
            'total_return': w.total_return,
            'volatility': w.volatility,
            'max_drawdown': w.max_drawdown,
            'calmar_ratio': w.calmar_ratio,
            'mse': w.mse,
            'mae': w.mae,
        })

    df = pd.DataFrame(records)
    aggregations: Dict[str, WindowAggregation] = {}
    for col in df.columns:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        aggregations[col] = WindowAggregation(
            metric_name=col,
            mean=float(values.mean()),
            std=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            median=float(values.median()),
            q25=float(values.quantile(0.25)),
            q75=float(values.quantile(0.75)),
            count=len(values)
        )

    return aggregations
