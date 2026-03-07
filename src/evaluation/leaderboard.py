"""
Leaderboard and Results Database

Provides persistent tracking of experiment results and auto-generation
of comparison tables for dissertation reporting.

Features:
- SQLite-backed results storage
- Auto-ranking by multiple metrics
- Regime-stratified leaderboards
- LaTeX/Markdown table export
- Historical comparison
"""

import sqlite3
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RankingMetric(Enum):
    """Metrics available for ranking"""
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR = "calmar_ratio"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    DIRECTIONAL_ACCURACY = "directional_accuracy"


@dataclass
class ExperimentEntry:
    """A single experiment entry in the leaderboard"""
    experiment_id: str
    model_name: str
    model_type: str
    config_hash: str
    timestamp: str

    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    calmar_ratio: float

    # Prediction metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    directional_accuracy: Optional[float] = None

    # Training info
    n_epochs: Optional[int] = None
    training_time: Optional[float] = None
    n_parameters: Optional[int] = None
    seed: Optional[int] = None

    # Regime performance
    regime_sharpe: Optional[Dict[str, float]] = None

    # Metadata
    dataset_version: Optional[str] = None
    notes: Optional[str] = None

    # ===== CAUSAL VS ORACLE CLASSIFICATION =====
    # Scientific validity requires separating causal (forecasting) from oracle models
    is_causal: bool = True  # True = causal forecasting, False = oracle (uses future info)
    model_category: str = "forecasting"  # "forecasting" or "oracle"

    # ===== RAW METRICS (unclipped) =====
    # For research analysis - may exceed display bounds
    sharpe_ratio_raw: Optional[float] = None
    sortino_ratio_raw: Optional[float] = None


@dataclass
class LeaderboardEntry:
    """Ranked entry in a leaderboard"""
    rank: int
    experiment_id: str
    model_name: str
    metric_value: float
    metric_name: str
    other_metrics: Dict[str, float]


@dataclass
class Leaderboard:
    """A complete leaderboard for a specific metric"""
    metric: RankingMetric
    entries: List[LeaderboardEntry]
    generated_at: str
    n_experiments: int
    best_entry: LeaderboardEntry
    description: str = ""


class ResultsDatabase:
    """
    SQLite database for experiment results.

    Provides persistent storage and querying of experiment results.
    """

    def __init__(self, db_path: Union[str, Path] = "results/experiments.db"):
        """
        Initialize results database.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT,
                    config_hash TEXT,
                    config_json TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,

                    -- Performance metrics
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    total_return REAL,
                    annualized_return REAL,
                    volatility REAL,
                    max_drawdown REAL,
                    calmar_ratio REAL,

                    -- Raw (unclipped) metrics for research
                    sharpe_ratio_raw REAL,
                    sortino_ratio_raw REAL,

                    -- Prediction metrics
                    mse REAL,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    directional_accuracy REAL,

                    -- Training info
                    n_epochs INTEGER,
                    training_time REAL,
                    n_parameters INTEGER,
                    seed INTEGER,

                    -- Causal vs Oracle classification
                    is_causal INTEGER DEFAULT 1,  -- 1 = causal, 0 = oracle
                    model_category TEXT DEFAULT 'forecasting',  -- 'forecasting' or 'oracle'

                    -- Metadata
                    dataset_version TEXT,
                    notes TEXT,
                    regime_sharpe_json TEXT
                );

                CREATE TABLE IF NOT EXISTS regime_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    sharpe_ratio REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    n_samples INTEGER,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS leaderboards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    metric_value REAL,
                    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_exp_model ON experiments(model_name);
                CREATE INDEX IF NOT EXISTS idx_exp_type ON experiments(model_type);
                CREATE INDEX IF NOT EXISTS idx_regime_exp ON regime_results(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_lb_metric ON leaderboards(metric);
            """)

    def save_experiment(
        self,
        entry: ExperimentEntry,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save experiment entry to database"""
        config_json = json.dumps(config) if config else None
        regime_json = json.dumps(entry.regime_sharpe) if entry.regime_sharpe else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, model_name, model_type, config_hash, config_json, timestamp,
                 sharpe_ratio, sortino_ratio, total_return, annualized_return,
                 volatility, max_drawdown, calmar_ratio,
                 sharpe_ratio_raw, sortino_ratio_raw,
                 mse, mae, rmse, mape, directional_accuracy,
                 n_epochs, training_time, n_parameters, seed,
                 is_causal, model_category,
                 dataset_version, notes, regime_sharpe_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.experiment_id, entry.model_name, entry.model_type,
                entry.config_hash, config_json, entry.timestamp,
                entry.sharpe_ratio, entry.sortino_ratio, entry.total_return,
                entry.annualized_return, entry.volatility, entry.max_drawdown,
                entry.calmar_ratio,
                entry.sharpe_ratio_raw, entry.sortino_ratio_raw,
                entry.mse, entry.mae, entry.rmse, entry.mape,
                entry.directional_accuracy,
                entry.n_epochs, entry.training_time, entry.n_parameters, entry.seed,
                1 if entry.is_causal else 0, entry.model_category,
                entry.dataset_version, entry.notes, regime_json
            ))

        logger.info(f"Saved experiment: {entry.experiment_id}")

    def save_regime_results(
        self,
        experiment_id: str,
        regime_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Save regime-stratified results"""
        with sqlite3.connect(self.db_path) as conn:
            for regime, metrics in regime_results.items():
                conn.execute("""
                    INSERT INTO regime_results
                    (experiment_id, regime, sharpe_ratio, total_return, max_drawdown, n_samples)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, regime,
                    metrics.get('sharpe_ratio'),
                    metrics.get('total_return'),
                    metrics.get('max_drawdown'),
                    metrics.get('n_samples')
                ))

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentEntry]:
        """Get a single experiment by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,)
            ).fetchone()

            if row is None:
                return None

            regime_sharpe = json.loads(row['regime_sharpe_json']) if row['regime_sharpe_json'] else None

            return ExperimentEntry(
                experiment_id=row['id'],
                model_name=row['model_name'],
                model_type=row['model_type'],
                config_hash=row['config_hash'],
                timestamp=row['timestamp'],
                sharpe_ratio=row['sharpe_ratio'],
                sortino_ratio=row['sortino_ratio'],
                total_return=row['total_return'],
                annualized_return=row['annualized_return'],
                volatility=row['volatility'],
                max_drawdown=row['max_drawdown'],
                calmar_ratio=row['calmar_ratio'],
                mse=row['mse'],
                mae=row['mae'],
                rmse=row['rmse'],
                mape=row['mape'],
                directional_accuracy=row['directional_accuracy'],
                n_epochs=row['n_epochs'],
                training_time=row['training_time'],
                n_parameters=row['n_parameters'],
                seed=row['seed'],
                dataset_version=row['dataset_version'],
                notes=row['notes'],
                regime_sharpe=regime_sharpe
            )

    def get_all_experiments(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get all experiments as DataFrame"""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_ranked(
        self,
        metric: RankingMetric,
        ascending: bool = None,
        limit: int = 20,
        include_oracle: bool = False
    ) -> pd.DataFrame:
        """
        Get experiments ranked by a metric.

        Args:
            metric: Metric to rank by
            ascending: Sort order (auto-determined if None)
            limit: Number of results

        Returns:
            Ranked DataFrame
        """
        # Determine sort order
        if ascending is None:
            # Lower is better for error metrics
            ascending = metric in [
                RankingMetric.MSE, RankingMetric.MAE,
                RankingMetric.RMSE, RankingMetric.MAX_DRAWDOWN
            ]

        order = "ASC" if ascending else "DESC"
        metric_col = metric.value

        causal_filter = "" if include_oracle else "AND is_causal = 1"

        query = f"""
            SELECT * FROM experiments
            WHERE {metric_col} IS NOT NULL
            {causal_filter}
            ORDER BY {metric_col} {order}
            LIMIT {limit}
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_best_by_model_type(
        self,
        metric: RankingMetric = RankingMetric.SHARPE
    ) -> pd.DataFrame:
        """Get best experiment for each model type"""
        ascending = metric in [
            RankingMetric.MSE, RankingMetric.MAE,
            RankingMetric.RMSE, RankingMetric.MAX_DRAWDOWN
        ]
        func = "MIN" if ascending else "MAX"
        metric_col = metric.value

        query = f"""
            SELECT e.*
            FROM experiments e
            INNER JOIN (
                SELECT model_type, {func}({metric_col}) as best_value
                FROM experiments
                WHERE {metric_col} IS NOT NULL
                GROUP BY model_type
            ) best ON e.model_type = best.model_type
                   AND e.{metric_col} = best.best_value
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_causal_ranked(
        self,
        metric: RankingMetric,
        ascending: bool = None,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Get ONLY causal (forecasting) models ranked by a metric.

        For scientific validity, causal models should be compared separately
        from oracle models that use future information.
        """
        if ascending is None:
            ascending = metric in [
                RankingMetric.MSE, RankingMetric.MAE,
                RankingMetric.RMSE, RankingMetric.MAX_DRAWDOWN
            ]

        order = "ASC" if ascending else "DESC"
        metric_col = metric.value

        query = f"""
            SELECT * FROM experiments
            WHERE {metric_col} IS NOT NULL
              AND (is_causal = 1 OR is_causal IS NULL)
            ORDER BY {metric_col} {order}
            LIMIT {limit}
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_oracle_ranked(
        self,
        metric: RankingMetric,
        ascending: bool = None,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Get ONLY oracle (non-causal) models ranked by a metric.

        Oracle models use future information and should not be compared
        directly with causal forecasting models.
        """
        if ascending is None:
            ascending = metric in [
                RankingMetric.MSE, RankingMetric.MAE,
                RankingMetric.RMSE, RankingMetric.MAX_DRAWDOWN
            ]

        order = "ASC" if ascending else "DESC"
        metric_col = metric.value

        query = f"""
            SELECT * FROM experiments
            WHERE {metric_col} IS NOT NULL
              AND is_causal = 0
            ORDER BY {metric_col} {order}
            LIMIT {limit}
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)


class LeaderboardGenerator:
    """
    Generates leaderboards and comparison tables.

    Supports multiple output formats for dissertation use.
    """

    def __init__(self, db: ResultsDatabase):
        """
        Initialize generator.

        Args:
            db: Results database
        """
        self.db = db

    def generate_leaderboard(
        self,
        metric: RankingMetric,
        top_n: int = 10,
        include_regime: bool = False,
        include_oracle: bool = False
    ) -> Leaderboard:
        """
        Generate a leaderboard for a specific metric.

        Args:
            metric: Ranking metric
            top_n: Number of entries to include
            include_regime: Include regime breakdown

        Returns:
            Leaderboard object
        """
        df = self.db.get_ranked(metric, limit=top_n, include_oracle=include_oracle)

        entries = []
        for idx, row in df.iterrows():
            other_metrics = {
                'sharpe': row['sharpe_ratio'],
                'sortino': row['sortino_ratio'],
                'return': row['total_return'],
                'max_dd': row['max_drawdown'],
                'mse': row['mse'],
                'mae': row['mae']
            }

            entries.append(LeaderboardEntry(
                rank=idx + 1,
                experiment_id=row['id'],
                model_name=row['model_name'],
                metric_value=row[metric.value],
                metric_name=metric.value,
                other_metrics=other_metrics
            ))

        best = entries[0] if entries else None

        return Leaderboard(
            metric=metric,
            entries=entries,
            generated_at=datetime.now().isoformat(),
            n_experiments=len(df),
            best_entry=best,
            description=f"Top {top_n} models by {metric.value} ({'all' if include_oracle else 'causal-only'})"
        )

    def generate_comparison_table(
        self,
        experiment_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comparison table for selected experiments.

        Args:
            experiment_ids: Specific experiments to include
            model_types: Model types to include
            metrics: Metrics to show

        Returns:
            Comparison DataFrame
        """
        if experiment_ids:
            placeholders = ','.join(['?'] * len(experiment_ids))
            query = f"SELECT * FROM experiments WHERE id IN ({placeholders})"
            with sqlite3.connect(self.db.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=experiment_ids)
        elif model_types:
            placeholders = ','.join(['?'] * len(model_types))
            query = f"""
                SELECT * FROM experiments
                WHERE model_type IN ({placeholders})
                ORDER BY sharpe_ratio DESC
            """
            with sqlite3.connect(self.db.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=model_types)
        else:
            df = self.db.get_all_experiments(limit=50)

        if df.empty:
            return df

        # Select and format columns
        if metrics is None:
            metrics = [
                'sharpe_ratio', 'sortino_ratio', 'total_return',
                'max_drawdown', 'mse', 'mae', 'directional_accuracy'
            ]

        columns = ['model_name', 'model_type'] + metrics
        columns = [c for c in columns if c in df.columns]

        result = df[columns].copy()

        # Format
        result = result.rename(columns={
            'model_name': 'Model',
            'model_type': 'Type',
            'sharpe_ratio': 'Sharpe',
            'sortino_ratio': 'Sortino',
            'total_return': 'Return',
            'max_drawdown': 'Max DD',
            'mse': 'MSE',
            'mae': 'MAE',
            'directional_accuracy': 'Dir. Acc.'
        })

        return result

    def export_latex(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        highlight_best: bool = True
    ) -> str:
        """
        Export DataFrame to LaTeX table.

        Args:
            df: DataFrame to export
            caption: Table caption
            label: LaTeX label
            highlight_best: Highlight best values

        Returns:
            LaTeX table string
        """
        # Format numeric columns
        df_formatted = df.copy()

        for col in df_formatted.columns:
            if df_formatted[col].dtype in [np.float64, np.float32]:
                # Determine if lower is better
                lower_better = col in ['Max DD', 'MSE', 'MAE', 'RMSE', 'MAPE']

                if highlight_best:
                    if lower_better:
                        best_idx = df_formatted[col].idxmin()
                    else:
                        best_idx = df_formatted[col].idxmax()

                    # Format values
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                    )

                    if pd.notna(best_idx):
                        val = df_formatted.loc[best_idx, col]
                        df_formatted.loc[best_idx, col] = f"\\textbf{{{val}}}"
                else:
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                    )

        # Build LaTeX
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{tab:{label}}}",
        ]

        col_format = "l" + "r" * (len(df.columns) - 1)
        lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        lines.append("\\toprule")

        # Header
        header = " & ".join(df_formatted.columns)
        lines.append(f"{header} \\\\")
        lines.append("\\midrule")

        # Data
        for _, row in df_formatted.iterrows():
            row_str = " & ".join(str(v) for v in row.values)
            lines.append(f"{row_str} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def export_markdown(
        self,
        df: pd.DataFrame,
        title: str
    ) -> str:
        """
        Export DataFrame to Markdown table.

        Args:
            df: DataFrame to export
            title: Table title

        Returns:
            Markdown table string
        """
        lines = [f"### {title}", ""]

        # Format DataFrame
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if df_formatted[col].dtype in [np.float64, np.float32]:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )

        # Header
        lines.append("| " + " | ".join(df_formatted.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(df_formatted.columns)) + "|")

        # Data
        for _, row in df_formatted.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

        return "\n".join(lines)

    def generate_model_type_summary(self) -> pd.DataFrame:
        """Generate summary by model type"""
        query = """
            SELECT
                model_type as "Model Type",
                COUNT(*) as "N Experiments",
                AVG(sharpe_ratio) as "Avg Sharpe",
                MAX(sharpe_ratio) as "Best Sharpe",
                AVG(total_return) as "Avg Return",
                AVG(max_drawdown) as "Avg Max DD"
            FROM experiments
            GROUP BY model_type
            ORDER BY AVG(sharpe_ratio) DESC
        """

        with sqlite3.connect(self.db.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def generate_full_report(
        self,
        output_dir: Path,
        format: str = "latex"
    ) -> Dict[str, Path]:
        """
        Generate full leaderboard report.

        Args:
            output_dir: Output directory
            format: Output format (latex, markdown)

        Returns:
            Dictionary of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # Main leaderboard
        leaderboard = self.generate_leaderboard(RankingMetric.SHARPE, top_n=20)
        comparison = self.generate_comparison_table()

        if not comparison.empty:
            if format == "latex":
                content = self.export_latex(
                    comparison,
                    "Model Performance Comparison",
                    "model_comparison"
                )
                path = output_dir / "leaderboard.tex"
            else:
                content = self.export_markdown(comparison, "Model Performance Comparison")
                path = output_dir / "leaderboard.md"

            path.write_text(content)
            files['main_leaderboard'] = path

        # Model type summary
        summary = self.generate_model_type_summary()
        if not summary.empty:
            if format == "latex":
                content = self.export_latex(
                    summary,
                    "Performance by Model Type",
                    "model_type_summary"
                )
                path = output_dir / "model_type_summary.tex"
            else:
                content = self.export_markdown(summary, "Performance by Model Type")
                path = output_dir / "model_type_summary.md"

            path.write_text(content)
            files['model_type_summary'] = path

        logger.info(f"Generated {len(files)} report files in {output_dir}")
        return files


def create_experiment_entry(
    experiment_id: str,
    model_name: str,
    model_type: str,
    metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None
) -> ExperimentEntry:
    """
    Convenience function to create an experiment entry.

    Args:
        experiment_id: Unique experiment ID
        model_name: Model name
        model_type: Model type (lstm, pinn_gbm, etc.)
        metrics: Dictionary of metric values
        config: Optional configuration dictionary
        training_info: Optional training information

    Returns:
        ExperimentEntry
    """
    config_hash = hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8] if config else ""

    training_info = training_info or {}

    return ExperimentEntry(
        experiment_id=experiment_id,
        model_name=model_name,
        model_type=model_type,
        config_hash=config_hash,
        timestamp=datetime.now().isoformat(),
        sharpe_ratio=metrics.get('sharpe_ratio', 0),
        sortino_ratio=metrics.get('sortino_ratio', 0),
        total_return=metrics.get('total_return', 0),
        annualized_return=metrics.get('annualized_return', 0),
        volatility=metrics.get('volatility', 0),
        max_drawdown=metrics.get('max_drawdown', 0),
        calmar_ratio=metrics.get('calmar_ratio', 0),
        mse=metrics.get('mse'),
        mae=metrics.get('mae'),
        rmse=metrics.get('rmse'),
        mape=metrics.get('mape'),
        directional_accuracy=metrics.get('directional_accuracy'),
        n_epochs=training_info.get('n_epochs'),
        training_time=training_info.get('training_time'),
        n_parameters=training_info.get('n_parameters'),
        seed=training_info.get('seed')
    )
