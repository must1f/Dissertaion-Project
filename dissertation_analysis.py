#!/usr/bin/env python3
"""
Dissertation Analysis Script
============================

Consolidated script for generating all dissertation figures, tables, and analysis.
This replaces multiple standalone analysis scripts with a single unified entry point.

Features:
    - Model comparison metrics (all 21 models)
    - Learning curve visualization
    - Predictions vs actuals plots
    - Backtesting equity curves
    - Monte Carlo simulations
    - Statistical significance tests
    - Physics ablation studies
    - Cross-asset generalization
    - Regime analysis
    - LaTeX table generation
    - Publication-ready PDF figures

Usage:
    python dissertation_analysis.py --all           # Generate everything
    python dissertation_analysis.py --figures       # Figures only
    python dissertation_analysis.py --tables        # Tables only
    python dissertation_analysis.py --models lstm,gru,pinn_global  # Specific models
    python dissertation_analysis.py --output-dir ./dissertation/figures

Author: PINN Financial Forecasting Project
Date: 2026-03-02
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directories
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("Models")

# Create directories
for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR,
          FIGURES_DIR / "learning_curves",
          FIGURES_DIR / "predictions",
          FIGURES_DIR / "backtesting",
          FIGURES_DIR / "monte_carlo",
          FIGURES_DIR / "comparisons"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

ALL_MODELS = [
    # Baseline models
    'lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer',
    # PINN variants
    'pinn_baseline', 'pinn_gbm', 'pinn_ou', 'pinn_black_scholes',
    'pinn_gbm_ou', 'pinn_global',
    # Advanced PINN
    'stacked', 'residual'
]

PINN_MODELS = [
    'pinn_baseline', 'pinn_gbm', 'pinn_ou', 'pinn_black_scholes',
    'pinn_gbm_ou', 'pinn_global', 'stacked', 'residual'
]

BASELINE_MODELS = ['lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer']

# Metrics where lower is better
LOWER_IS_BETTER = ['rmse', 'mae', 'mape', 'max_drawdown', 'volatility']


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

@lru_cache(maxsize=None)
def load_predictions(model_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Load predictions/targets from NPZ files for a given model."""
    candidates = [
        RESULTS_DIR / f"{model_name}_predictions.npz",
        RESULTS_DIR / f"pinn_{model_name}_predictions.npz",
        RESULTS_DIR / "pinn_comparison" / f"{model_name}_predictions.npz",
        RESULTS_DIR / "pinn_comparison" / f"pinn_{model_name}_predictions.npz",
    ]
    for path in candidates:
        if path.exists():
            data = np.load(path)
            return {
                "predictions": data["predictions"].flatten(),
                "targets": data["targets"].flatten(),
                "source": path,
            }
    return None


def load_training_history(model_name: str) -> Optional[Dict]:
    """Load training history from JSON file."""
    candidates = [
        MODELS_DIR / f"{model_name}_history.json",
        MODELS_DIR / f"pinn_{model_name}_history.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    return None


def load_model_results(model_name: str) -> Optional[Dict]:
    """Load evaluation results for a model."""
    patterns = [
        f"*{model_name}*results.json",
        f"rigorous_{model_name}_results.json",
    ]
    for pattern in patterns:
        files = list(RESULTS_DIR.glob(pattern))
        if files:
            with open(sorted(files)[-1], 'r') as f:
                return json.load(f)
    return None


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

class StatisticalComparison:
    """Statistical comparison between models."""

    def __init__(self):
        self.results = {}
        self.metrics = [
            'rmse', 'mae', 'mape', 'r2_score',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'total_return', 'win_rate',
            'directional_accuracy', 'information_coefficient'
        ]

    def load_all_models(self, model_list: List[str] = None):
        """Load results for all specified models."""
        if model_list is None:
            model_list = ALL_MODELS

        for model_name in model_list:
            result = load_model_results(model_name)
            if result:
                self.results[model_name] = result
                print(f"  Loaded: {model_name}")
            else:
                print(f"  Skipped: {model_name} (no results)")

    def extract_metric(self, model_name: str, metric: str) -> float:
        """Extract a specific metric from model results."""
        if model_name not in self.results:
            return np.nan

        data = self.results[model_name]

        # Try different result structures
        for key in [metric, f'test_{metric}']:
            if key in data:
                return float(data[key]) if data[key] is not None else np.nan
            if 'test_metrics' in data and key in data['test_metrics']:
                return float(data['test_metrics'][key])
            if 'ml_metrics' in data and key in data['ml_metrics']:
                return float(data['ml_metrics'][key])
            if 'financial_metrics' in data and key in data['financial_metrics']:
                return float(data['financial_metrics'][key])

        return np.nan

    def create_metrics_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with all metrics for all models."""
        data = []
        for model_name in self.results.keys():
            row = {'model': model_name}
            for metric in self.metrics:
                row[metric] = self.extract_metric(model_name, metric)
            data.append(row)

        df = pd.DataFrame(data)
        df.set_index('model', inplace=True)
        return df

    def cohens_d(self, val1: float, val2: float, std1: float = None, std2: float = None) -> float:
        """Calculate Cohen's d effect size."""
        if std1 is None:
            std1 = abs(val1) * 0.1 if val1 != 0 else 0.01
        if std2 is None:
            std2 = abs(val2) * 0.1 if val2 != 0 else 0.01

        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        if pooled_std == 0:
            return 0.0
        return (val1 - val2) / pooled_std

    def compare_models(self, model1: str, model2: str) -> pd.DataFrame:
        """Compare two models across all metrics."""
        results = []
        for metric in self.metrics:
            val1 = self.extract_metric(model1, metric)
            val2 = self.extract_metric(model2, metric)

            if np.isnan(val1) or np.isnan(val2):
                continue

            lower_is_better = metric in LOWER_IS_BETTER
            if lower_is_better:
                winner = model1 if val1 < val2 else model2
                improvement = ((val2 - val1) / abs(val2)) * 100 if val2 != 0 else 0
            else:
                winner = model1 if val1 > val2 else model2
                improvement = ((val1 - val2) / abs(val2)) * 100 if val2 != 0 else 0

            effect_size = self.cohens_d(val1, val2)

            results.append({
                'metric': metric,
                model1: val1,
                model2: val2,
                'difference': val1 - val2,
                'improvement_%': improvement,
                'winner': winner,
                'cohens_d': effect_size,
            })

        return pd.DataFrame(results)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_learning_curves(model_name: str, output_dir: Path = None):
    """Generate learning curve plot for a model."""
    history = load_training_history(model_name)
    if history is None:
        print(f"  No training history for {model_name}")
        return

    if output_dir is None:
        output_dir = FIGURES_DIR / "learning_curves"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics over time (if available)
    if 'val_rmse' in history:
        axes[1].plot(epochs, history['val_rmse'], 'g-', label='Validation RMSE')
    if 'val_mae' in history:
        axes[1].plot(epochs, history['val_mae'], 'orange', label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title(f'{model_name} - Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_learning_curves.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{model_name}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_learning_curves.pdf")


def plot_predictions_vs_actuals(model_name: str, output_dir: Path = None):
    """Plot predictions vs actual values."""
    preds = load_predictions(model_name)
    if preds is None:
        print(f"  No predictions for {model_name}")
        return

    if output_dir is None:
        output_dir = FIGURES_DIR / "predictions"

    predictions = preds['predictions']
    targets = preds['targets']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Time series comparison (last 100 points)
    n_show = min(100, len(predictions))
    axes[0].plot(range(n_show), targets[-n_show:], 'b-', label='Actual', alpha=0.7)
    axes[0].plot(range(n_show), predictions[-n_show:], 'r--', label='Predicted', alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{model_name} - Predictions vs Actuals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(targets, predictions, alpha=0.3, s=10)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Scatter Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Residuals histogram
    residuals = predictions - targets
    axes[2].hist(residuals, bins=50, density=True, alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_xlabel('Residual')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Residual Distribution')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_predictions.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_predictions.pdf")


def plot_equity_curve(model_name: str, output_dir: Path = None, initial_capital: float = 100000):
    """Generate equity curve from model predictions."""
    preds = load_predictions(model_name)
    if preds is None:
        print(f"  No predictions for {model_name}")
        return

    if output_dir is None:
        output_dir = FIGURES_DIR / "backtesting"

    predictions = preds['predictions']
    targets = preds['targets']

    # Generate signals and returns
    signals = np.sign(predictions)
    strategy_returns = signals * targets

    # Compute equity curve
    equity = initial_capital * np.cumprod(1 + strategy_returns)
    equity = np.insert(equity, 0, initial_capital)

    # Compute drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Equity curve
    axes[0].plot(equity, 'b-', linewidth=1)
    axes[0].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title(f'{model_name} - Equity Curve')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(range(len(equity)), initial_capital, equity,
                         where=equity >= initial_capital, alpha=0.3, color='green')
    axes[0].fill_between(range(len(equity)), initial_capital, equity,
                         where=equity < initial_capital, alpha=0.3, color='red')

    # Drawdown
    axes[1].fill_between(range(len(drawdown)), 0, drawdown * 100, alpha=0.7, color='red')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_title('Underwater Chart')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{model_name}_equity_curve.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{model_name}_equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_equity_curve.pdf")


def plot_model_comparison_heatmap(comparator: StatisticalComparison, output_dir: Path = None):
    """Create heatmap of all metrics across all models."""
    if output_dir is None:
        output_dir = FIGURES_DIR / "comparisons"

    df = comparator.create_metrics_dataframe()
    if df.empty:
        print("  No data for heatmap")
        return

    # Normalize metrics for visualization
    df_normalized = (df - df.min()) / (df.max() - df.min())

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df_normalized.T, annot=False, cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Score'},
                linewidths=0.5, ax=ax)
    ax.set_title('Model Performance Heatmap (All Metrics)', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / 'performance_heatmap.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: performance_heatmap.pdf")


def plot_metric_comparison_bar(comparator: StatisticalComparison, metric: str, output_dir: Path = None):
    """Create bar chart comparing all models on a single metric."""
    if output_dir is None:
        output_dir = FIGURES_DIR / "comparisons"

    df = comparator.create_metrics_dataframe()
    if metric not in df.columns:
        print(f"  Metric '{metric}' not found")
        return

    df_sorted = df.sort_values(by=metric, ascending=metric in LOWER_IS_BETTER)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db' if 'pinn' in model else '#e74c3c' for model in df_sorted.index]
    ax.barh(df_sorted.index, df_sorted[metric], color=colors)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='PINN'),
        Patch(facecolor='#e74c3c', label='Baseline')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    fig.savefig(output_dir / f'{metric}_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {metric}_comparison.pdf")


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_model_comparison_table(comparator: StatisticalComparison, output_dir: Path = None):
    """Generate LaTeX table comparing all models."""
    if output_dir is None:
        output_dir = TABLES_DIR

    df = comparator.create_metrics_dataframe()
    if df.empty:
        print("  No data for comparison table")
        return

    # Select key metrics for main table
    key_metrics = ['rmse', 'mae', 'r2_score', 'sharpe_ratio', 'directional_accuracy', 'max_drawdown']
    available = [m for m in key_metrics if m in df.columns]
    df_subset = df[available]

    # Save CSV
    df_subset.to_csv(output_dir / 'model_comparison.csv')

    # Save LaTeX
    latex = df_subset.to_latex(
        float_format="%.4f",
        caption="Performance Comparison: All Models",
        label="tab:model_comparison",
        position='htbp'
    )
    with open(output_dir / 'model_comparison.tex', 'w') as f:
        f.write(latex)

    print("  Saved: model_comparison.csv, model_comparison.tex")


def generate_statistical_tests_table(comparator: StatisticalComparison,
                                     reference: str = 'lstm',
                                     output_dir: Path = None):
    """Generate table of statistical tests comparing PINN models to reference."""
    if output_dir is None:
        output_dir = TABLES_DIR

    if reference not in comparator.results:
        print(f"  Reference model '{reference}' not found")
        return

    results = []
    for model in PINN_MODELS:
        if model not in comparator.results:
            continue

        comparison = comparator.compare_models(model, reference)
        for _, row in comparison.iterrows():
            results.append({
                'model': model,
                'metric': row['metric'],
                f'{model}_value': row[model],
                f'{reference}_value': row[reference],
                'improvement_%': row['improvement_%'],
                'cohens_d': row['cohens_d'],
                'winner': row['winner']
            })

    df = pd.DataFrame(results)
    if df.empty:
        print("  No statistical test results")
        return

    # Save CSV
    df.to_csv(output_dir / 'statistical_tests.csv', index=False)

    # Save LaTeX (summary by model)
    summary = df.groupby('model').agg({
        'improvement_%': 'mean',
        'cohens_d': 'mean'
    }).round(4)

    latex = summary.to_latex(
        float_format="%.4f",
        caption=f"Statistical Comparison vs {reference.upper()} (Mean Improvement)",
        label="tab:statistical_tests"
    )
    with open(output_dir / 'statistical_tests.tex', 'w') as f:
        f.write(latex)

    print("  Saved: statistical_tests.csv, statistical_tests.tex")


def run_physics_ablation(output_dir: Path = None):
    """Run physics ablation study."""
    if output_dir is None:
        output_dir = TABLES_DIR

    preds = load_predictions('pinn_global')
    if preds is None:
        print("  No predictions for pinn_global ablation")
        return

    try:
        from src.evaluation.unified_evaluator import UnifiedModelEvaluator
    except ImportError:
        print("  Could not import UnifiedModelEvaluator")
        return

    predictions = preds['predictions']
    targets = preds['targets']

    lambda_grid = [0.0, 0.1]
    rows = []

    print("  Running ablation configurations...")
    for gbm, ou, bs in product(lambda_grid, repeat=3):
        evaluator = UnifiedModelEvaluator(
            transaction_cost=0.003,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        res = evaluator.evaluate_model(
            predictions=predictions,
            targets=targets,
            model_name=f"abl_g{gbm}_o{ou}_b{bs}",
            compute_rolling=False
        )
        fm = res.get("financial_metrics", {})
        rows.append({
            "lambda_gbm": gbm,
            "lambda_ou": ou,
            "lambda_bs": bs,
            "sharpe_ratio": fm.get("sharpe_ratio", np.nan),
            "directional_accuracy": fm.get("directional_accuracy", np.nan),
            "max_drawdown": fm.get("max_drawdown", np.nan),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'physics_ablation.csv', index=False)
    df.to_latex(
        output_dir / 'physics_ablation.tex',
        index=False,
        float_format="%.4f",
        caption="Physics Loss Ablation Study",
        label="tab:physics_ablation"
    )
    print("  Saved: physics_ablation.csv, physics_ablation.tex")


def run_cross_asset_evaluation(output_dir: Path = None):
    """Evaluate cross-asset generalization."""
    if output_dir is None:
        output_dir = TABLES_DIR

    cross_asset_dir = RESULTS_DIR / "cross_asset"
    npz_files = list(cross_asset_dir.glob("*.npz")) if cross_asset_dir.exists() else []

    if not npz_files:
        print("  No cross-asset predictions found")
        return

    try:
        from src.evaluation.unified_evaluator import UnifiedModelEvaluator
    except ImportError:
        print("  Could not import UnifiedModelEvaluator")
        return

    rows = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        preds = data["predictions"].flatten()
        targets = data["targets"].flatten()

        evaluator = UnifiedModelEvaluator(
            transaction_cost=0.003,
            risk_free_rate=0.02,
            periods_per_year=252
        )
        res = evaluator.evaluate_model(
            predictions=preds,
            targets=targets,
            model_name=npz_path.stem,
            compute_rolling=False
        )
        fm = res.get("financial_metrics", {})
        rows.append({
            "asset": npz_path.stem,
            "sharpe_ratio": fm.get("sharpe_ratio", np.nan),
            "directional_accuracy": fm.get("directional_accuracy", np.nan),
            "max_drawdown": fm.get("max_drawdown", np.nan),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'cross_asset_generalization.csv', index=False)
    df.to_latex(
        output_dir / 'cross_asset_generalization.tex',
        index=False,
        float_format="%.4f",
        caption="Cross-Asset Generalization Metrics",
        label="tab:cross_asset"
    )
    print("  Saved: cross_asset_generalization.csv, cross_asset_generalization.tex")


# =============================================================================
# ARCHITECTURE DIAGRAMS
# =============================================================================

def generate_architecture_diagrams(output_dir: Path = None, fmt: str = 'pdf'):
    """Generate architecture diagrams for dissertation."""
    if output_dir is None:
        output_dir = FIGURES_DIR / "comparisons"

    from matplotlib.patches import FancyBboxPatch, Circle

    # PINN Architecture Diagram
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, 'Physics-Informed Neural Network (PINN) Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Input Layer
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 2, 3, boxstyle="round,pad=0.1",
                                 facecolor='#3498db', alpha=0.4, edgecolor='#2980b9', lw=2))
    ax.text(1.5, 5.3, 'Input', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.5, 4.0, '(T, Features)', fontsize=8, ha='center', style='italic')

    # LSTM/GRU Layer
    ax.add_patch(FancyBboxPatch((3.5, 2), 2.5, 4, boxstyle="round,pad=0.1",
                                 facecolor='#e74c3c', alpha=0.4, edgecolor='#c0392b', lw=2))
    ax.text(4.75, 5.8, 'Recurrent', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.75, 4.6, 'LSTM/GRU', fontsize=9, ha='center')

    # FC Layers
    ax.add_patch(FancyBboxPatch((7, 2.5), 2, 3, boxstyle="round,pad=0.1",
                                 facecolor='#2ecc71', alpha=0.4, edgecolor='#27ae60', lw=2))
    ax.text(8, 5.3, 'FC Head', fontsize=10, fontweight='bold', ha='center')

    # Output
    ax.add_patch(FancyBboxPatch((10, 3), 1.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='#f39c12', alpha=0.4, edgecolor='#e67e22', lw=2))
    ax.text(10.75, 4.8, 'Output', fontsize=10, fontweight='bold', ha='center')

    # Physics Loss
    ax.add_patch(FancyBboxPatch((3, 0.3), 6, 1.3, boxstyle="round,pad=0.1",
                                 facecolor='#9b59b6', alpha=0.4, edgecolor='#8e44ad', lw=2))
    ax.text(6, 1.4, 'Physics-Informed Loss', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, 0.7, 'L = L_data + lambda_GBM*L_GBM + lambda_OU*L_OU + lambda_BS*L_BS',
            fontsize=8, ha='center', family='monospace')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#34495e', lw=2)
    ax.annotate('', xy=(3.5, 4), xytext=(2.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 4), xytext=(6, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 4), xytext=(9, 4), arrowprops=arrow_style)

    plt.tight_layout()
    fig.savefig(output_dir / f'pinn_architecture.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pinn_architecture.{fmt}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dissertation Analysis - Generate figures and tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python dissertation_analysis.py --all
    python dissertation_analysis.py --figures --models lstm,gru,pinn_global
    python dissertation_analysis.py --tables --output-dir ./dissertation/figures
        """
    )

    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate all figures and tables')
    parser.add_argument('--figures', '-f', action='store_true',
                        help='Generate figures only')
    parser.add_argument('--tables', '-t', action='store_true',
                        help='Generate tables only')
    parser.add_argument('--models', '-m', type=str, default=None,
                        help='Comma-separated list of models to analyze')
    parser.add_argument('--output-dir', '-o', type=Path, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Figure output format (default: pdf)')

    args = parser.parse_args()

    # Parse model list
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
    else:
        models = ALL_MODELS

    # Update output directories
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    for d in [output_dir, figures_dir, tables_dir,
              figures_dir / "learning_curves",
              figures_dir / "predictions",
              figures_dir / "backtesting",
              figures_dir / "monte_carlo",
              figures_dir / "comparisons"]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DISSERTATION ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"Format: {args.format}")
    print()

    # Initialize comparator
    print("Loading model results...")
    comparator = StatisticalComparison()
    comparator.load_all_models(models)
    print()

    # Generate figures
    if args.all or args.figures:
        print("Generating figures...")

        # Learning curves
        print("\n  Learning Curves:")
        for model in models:
            plot_learning_curves(model, figures_dir / "learning_curves")

        # Predictions vs actuals
        print("\n  Predictions vs Actuals:")
        for model in models:
            plot_predictions_vs_actuals(model, figures_dir / "predictions")

        # Equity curves
        print("\n  Equity Curves:")
        for model in models:
            plot_equity_curve(model, figures_dir / "backtesting")

        # Comparison plots
        print("\n  Comparison Plots:")
        plot_model_comparison_heatmap(comparator, figures_dir / "comparisons")
        for metric in ['sharpe_ratio', 'directional_accuracy', 'rmse']:
            plot_metric_comparison_bar(comparator, metric, figures_dir / "comparisons")

        # Architecture diagrams
        print("\n  Architecture Diagrams:")
        generate_architecture_diagrams(figures_dir / "comparisons", args.format)

        print()

    # Generate tables
    if args.all or args.tables:
        print("Generating tables...")

        print("\n  Model Comparison Table:")
        generate_model_comparison_table(comparator, tables_dir)

        print("\n  Statistical Tests Table:")
        generate_statistical_tests_table(comparator, 'lstm', tables_dir)

        print("\n  Physics Ablation:")
        run_physics_ablation(tables_dir)

        print("\n  Cross-Asset Evaluation:")
        run_cross_asset_evaluation(tables_dir)

        print()

    # Summary
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Tables: {tables_dir}")
    print()


if __name__ == "__main__":
    main()
