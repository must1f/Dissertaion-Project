"""
Plot Generator for Experiment Visualization

Provides comprehensive visualization suite for PINN financial forecasting experiments:
- Learning curves (total loss and components)
- Gradient norm evolution
- PDE residual distributions
- Rolling window performance
- Drawdown and cumulative returns
- Model comparison charts

Designed for dissertation-quality figures with consistent styling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PlotStyle(Enum):
    """Available plot styles"""
    PUBLICATION = "publication"  # Clean, minimal for papers
    PRESENTATION = "presentation"  # Larger fonts for slides
    DASHBOARD = "dashboard"  # Interactive-friendly
    MINIMAL = "minimal"  # Very clean, no grid


@dataclass
class PlotConfig:
    """Configuration for plot generation"""
    style: PlotStyle = PlotStyle.PUBLICATION
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 150
    font_family: str = "serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    line_width: float = 1.5
    marker_size: int = 6
    grid_alpha: float = 0.3
    color_palette: str = "deep"
    save_format: str = "pdf"  # pdf, png, svg
    transparent_bg: bool = False


@dataclass
class ExperimentResults:
    """Container for experiment results to plot"""
    # Training history
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)

    # Loss components (for PINN)
    data_loss: List[float] = field(default_factory=list)
    gbm_loss: List[float] = field(default_factory=list)
    ou_loss: List[float] = field(default_factory=list)
    bs_loss: List[float] = field(default_factory=list)

    # Gradient norms
    gradient_norms: Dict[str, List[float]] = field(default_factory=dict)

    # PDE residuals
    residual_history: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # Predictions and actuals
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    dates: Optional[List[datetime]] = None

    # Rolling window results
    window_metrics: Optional[pd.DataFrame] = None

    # Trading metrics
    cumulative_returns: Optional[np.ndarray] = None
    drawdowns: Optional[np.ndarray] = None
    benchmark_returns: Optional[np.ndarray] = None

    # Metadata
    model_name: str = ""
    experiment_id: str = ""


class PlotGenerator:
    """
    Generates publication-quality plots for experiment results.

    Provides a standard visualization suite including:
    - Learning curves
    - Loss component breakdown
    - Gradient norm evolution
    - PDE residual distributions
    - Rolling performance charts
    - Drawdown analysis
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plot generator.

        Args:
            config: Plot configuration
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for PlotGenerator")

        self.config = config or PlotConfig()
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style based on config"""
        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'legend.fontsize': self.config.legend_size,
            'figure.figsize': self.config.figsize,
            'figure.dpi': self.config.dpi,
            'lines.linewidth': self.config.line_width,
            'lines.markersize': self.config.marker_size,
            'axes.grid': True,
            'grid.alpha': self.config.grid_alpha,
        })

        if HAS_SEABORN:
            sns.set_palette(self.config.color_palette)

        if self.config.style == PlotStyle.PUBLICATION:
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
            })
        elif self.config.style == PlotStyle.MINIMAL:
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': False,
            })

    def plot_learning_curves(
        self,
        results: ExperimentResults,
        show_components: bool = True,
        log_scale: bool = False
    ) -> Figure:
        """
        Plot training and validation loss curves.

        Args:
            results: Experiment results
            show_components: Whether to show loss components
            log_scale: Use log scale for y-axis

        Returns:
            matplotlib Figure
        """
        if show_components and results.data_loss:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax1, ax2 = axes
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.config.figsize)
            ax2 = None

        epochs = results.epochs or list(range(1, len(results.train_loss) + 1))

        # Main loss plot
        ax1.plot(epochs, results.train_loss, label='Train Loss', color='#2ecc71')
        if results.val_loss:
            ax1.plot(epochs, results.val_loss, label='Validation Loss',
                    color='#e74c3c', linestyle='--')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Learning Curves - {results.model_name}')
        ax1.legend()

        if log_scale:
            ax1.set_yscale('log')

        # Loss components plot
        if ax2 is not None and results.data_loss:
            ax2.plot(epochs, results.data_loss, label='Data Loss', color='#3498db')
            if results.gbm_loss:
                ax2.plot(epochs, results.gbm_loss, label='GBM Loss', color='#e74c3c')
            if results.ou_loss:
                ax2.plot(epochs, results.ou_loss, label='OU Loss', color='#9b59b6')
            if results.bs_loss:
                ax2.plot(epochs, results.bs_loss, label='Black-Scholes Loss', color='#f39c12')

            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Component')
            ax2.set_title('Loss Components')
            ax2.legend()

            if log_scale:
                ax2.set_yscale('log')

        plt.tight_layout()
        return fig

    def plot_gradient_norms(
        self,
        results: ExperimentResults,
        smoothing: float = 0.9
    ) -> Figure:
        """
        Plot gradient norm evolution during training.

        Args:
            results: Experiment results
            smoothing: Exponential smoothing factor

        Returns:
            matplotlib Figure
        """
        if not results.gradient_norms:
            logger.warning("No gradient norm data available")
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
            ax.text(0.5, 0.5, 'No gradient data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)

        colors = plt.cm.Set2(np.linspace(0, 1, len(results.gradient_norms)))

        for (name, norms), color in zip(results.gradient_norms.items(), colors):
            epochs = range(1, len(norms) + 1)

            # Apply exponential smoothing
            smoothed = self._exponential_smooth(norms, smoothing)

            ax.plot(epochs, smoothed, label=name, color=color)
            ax.fill_between(epochs, smoothed, alpha=0.1, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm Evolution')
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        return fig

    def plot_residual_histograms(
        self,
        results: ExperimentResults,
        epoch: int = -1
    ) -> Figure:
        """
        Plot histograms of PDE residuals.

        Args:
            results: Experiment results
            epoch: Which epoch to plot (-1 for last)

        Returns:
            matplotlib Figure
        """
        if not results.residual_history:
            logger.warning("No residual data available")
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
            ax.text(0.5, 0.5, 'No residual data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        n_residuals = len(results.residual_history)
        n_cols = min(3, n_residuals)
        n_rows = (n_residuals + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_residuals == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#2ecc71']

        for idx, (name, history) in enumerate(results.residual_history.items()):
            ax = axes[idx]
            residuals = history[epoch] if isinstance(history, list) else history

            ax.hist(residuals.flatten(), bins=50, color=colors[idx % len(colors)],
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')

            mean_val = np.mean(residuals)
            std_val = np.std(residuals)
            ax.axvline(x=mean_val, color='green', linestyle='-', linewidth=1.5,
                      label=f'Mean: {mean_val:.4f}')

            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Residuals\n(std: {std_val:.4f})')
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(n_residuals, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'PDE Residual Distributions - {results.model_name}', fontsize=14)
        plt.tight_layout()
        return fig

    def plot_predictions_vs_actual(
        self,
        results: ExperimentResults,
        n_samples: int = 200
    ) -> Figure:
        """
        Plot predictions against actual values.

        Args:
            results: Experiment results
            n_samples: Number of samples to plot

        Returns:
            matplotlib Figure
        """
        if results.predictions is None or results.actuals is None:
            logger.warning("No prediction data available")
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
            ax.text(0.5, 0.5, 'No prediction data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        predictions = results.predictions[-n_samples:].flatten()
        actuals = results.actuals[-n_samples:].flatten()

        # Time series plot
        ax1 = axes[0]
        x = range(len(predictions))
        ax1.plot(x, actuals, label='Actual', color='#2ecc71', alpha=0.8)
        ax1.plot(x, predictions, label='Predicted', color='#e74c3c', alpha=0.8)
        ax1.fill_between(x, actuals, predictions, alpha=0.2, color='gray')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title('Predictions vs Actuals (Time Series)')
        ax1.legend()

        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(actuals, predictions, alpha=0.5, s=20, color='#3498db')

        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--',
                linewidth=2, label='Perfect Prediction')

        # Compute R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'Scatter Plot (R² = {r_squared:.4f})')
        ax2.legend()

        plt.suptitle(f'Prediction Analysis - {results.model_name}', fontsize=14)
        plt.tight_layout()
        return fig

    def plot_rolling_performance(
        self,
        results: ExperimentResults,
        metric: str = 'sharpe'
    ) -> Figure:
        """
        Plot rolling window performance metrics.

        Args:
            results: Experiment results
            metric: Metric to plot

        Returns:
            matplotlib Figure
        """
        if results.window_metrics is None:
            logger.warning("No window metrics available")
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
            ax.text(0.5, 0.5, 'No window metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)

        df = results.window_metrics

        if metric in df.columns:
            values = df[metric].values
            windows = range(1, len(values) + 1)

            ax.bar(windows, values, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axhline(y=np.mean(values), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(values):.3f}')

            # Add trend line
            z = np.polyfit(windows, values, 1)
            p = np.poly1d(z)
            ax.plot(windows, p(windows), 'g-', linewidth=2, alpha=0.7,
                   label=f'Trend (slope: {z[0]:.4f})')

            ax.set_xlabel('Window')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Rolling Window {metric.replace("_", " ").title()} - {results.model_name}')
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_drawdown(
        self,
        results: ExperimentResults,
        include_benchmark: bool = True
    ) -> Figure:
        """
        Plot cumulative returns and drawdown.

        Args:
            results: Experiment results
            include_benchmark: Include benchmark comparison

        Returns:
            matplotlib Figure
        """
        if results.cumulative_returns is None:
            logger.warning("No return data available")
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)
            ax.text(0.5, 0.5, 'No return data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        returns = results.cumulative_returns
        n = len(returns)
        x = range(n)

        # Cumulative returns
        ax1 = axes[0]
        ax1.plot(x, returns, label='Strategy', color='#2ecc71', linewidth=2)

        if include_benchmark and results.benchmark_returns is not None:
            ax1.plot(x, results.benchmark_returns, label='Benchmark',
                    color='#95a5a6', linestyle='--', linewidth=1.5)

        ax1.fill_between(x, 0, returns, where=returns > 0,
                        color='#2ecc71', alpha=0.3)
        ax1.fill_between(x, 0, returns, where=returns < 0,
                        color='#e74c3c', alpha=0.3)

        ax1.set_ylabel('Cumulative Return')
        ax1.set_title(f'Cumulative Returns - {results.model_name}')
        ax1.legend()
        ax1.axhline(y=0, color='black', linewidth=0.5)

        # Drawdown
        ax2 = axes[1]

        if results.drawdowns is None:
            # Calculate drawdown
            peak = np.maximum.accumulate(returns)
            drawdowns = returns - peak
        else:
            drawdowns = results.drawdowns

        ax2.fill_between(x, drawdowns, 0, color='#e74c3c', alpha=0.7)
        ax2.plot(x, drawdowns, color='#c0392b', linewidth=1)

        max_dd = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        ax2.axhline(y=max_dd, color='red', linestyle='--', linewidth=1,
                   label=f'Max Drawdown: {max_dd:.2%}')

        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Drawdown Analysis')
        ax2.legend()

        plt.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        results_dict: Dict[str, ExperimentResults],
        metric: str = 'val_loss'
    ) -> Figure:
        """
        Plot comparison across multiple models.

        Args:
            results_dict: Dictionary of model name -> results
            metric: Metric to compare

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.config.figsize)

        colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

        for (name, results), color in zip(results_dict.items(), colors):
            if metric == 'val_loss' and results.val_loss:
                values = results.val_loss
            elif metric == 'train_loss' and results.train_loss:
                values = results.train_loss
            elif metric == 'data_loss' and results.data_loss:
                values = results.data_loss
            else:
                continue

            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name, color=color, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}')
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_regime_performance(
        self,
        results: ExperimentResults,
        regime_metrics: pd.DataFrame
    ) -> Figure:
        """
        Plot performance across different market regimes.

        Args:
            results: Experiment results
            regime_metrics: DataFrame with columns [regime, metric_name, value]

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if 'regime' not in regime_metrics.columns:
            ax.text(0.5, 0.5, 'No regime data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        regimes = regime_metrics['regime'].unique()
        metrics = [col for col in regime_metrics.columns if col not in ['regime', 'window_id']]

        x = np.arange(len(regimes))
        width = 0.8 / len(metrics)

        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            values = [regime_metrics[regime_metrics['regime'] == r][metric].mean()
                     for r in regimes]
            ax.bar(x + i * width, values, width, label=metric, color=colors[i])

        ax.set_xlabel('Regime')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Performance by Market Regime - {results.model_name}')
        ax.set_xticks(x + width * (len(metrics[:4]) - 1) / 2)
        ax.set_xticklabels(regimes)
        ax.legend()

        plt.tight_layout()
        return fig

    def generate_all(
        self,
        results: ExperimentResults,
        output_dir: Path,
        include_components: bool = True
    ) -> Dict[str, Path]:
        """
        Generate all standard plots and save to directory.

        Args:
            results: Experiment results
            output_dir: Output directory
            include_components: Include detailed component plots

        Returns:
            Dictionary of plot name -> saved path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = {}

        # Learning curves
        fig = self.plot_learning_curves(results, show_components=include_components)
        path = output_dir / f"learning_curves.{self.config.save_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                   transparent=self.config.transparent_bg)
        plt.close(fig)
        saved_plots['learning_curves'] = path
        logger.info(f"Saved learning curves to {path}")

        # Gradient norms
        if results.gradient_norms:
            fig = self.plot_gradient_norms(results)
            path = output_dir / f"gradient_norms.{self.config.save_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                       transparent=self.config.transparent_bg)
            plt.close(fig)
            saved_plots['gradient_norms'] = path

        # Residual histograms
        if results.residual_history:
            fig = self.plot_residual_histograms(results)
            path = output_dir / f"residual_histograms.{self.config.save_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                       transparent=self.config.transparent_bg)
            plt.close(fig)
            saved_plots['residual_histograms'] = path

        # Predictions vs actuals
        if results.predictions is not None:
            fig = self.plot_predictions_vs_actual(results)
            path = output_dir / f"predictions_vs_actual.{self.config.save_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                       transparent=self.config.transparent_bg)
            plt.close(fig)
            saved_plots['predictions_vs_actual'] = path

        # Rolling performance
        if results.window_metrics is not None:
            fig = self.plot_rolling_performance(results)
            path = output_dir / f"rolling_performance.{self.config.save_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                       transparent=self.config.transparent_bg)
            plt.close(fig)
            saved_plots['rolling_performance'] = path

        # Drawdown
        if results.cumulative_returns is not None:
            fig = self.plot_drawdown(results)
            path = output_dir / f"drawdown.{self.config.save_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                       transparent=self.config.transparent_bg)
            plt.close(fig)
            saved_plots['drawdown'] = path

        logger.info(f"Generated {len(saved_plots)} plots in {output_dir}")
        return saved_plots

    def _exponential_smooth(self, values: List[float], alpha: float) -> np.ndarray:
        """Apply exponential smoothing to values"""
        smoothed = np.zeros(len(values))
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
        return smoothed


def create_standard_plots(
    results: ExperimentResults,
    output_dir: Union[str, Path],
    config: Optional[PlotConfig] = None
) -> Dict[str, Path]:
    """
    Convenience function to generate all standard plots.

    Args:
        results: Experiment results
        output_dir: Output directory
        config: Optional plot configuration

    Returns:
        Dictionary of plot name -> saved path
    """
    generator = PlotGenerator(config)
    return generator.generate_all(results, Path(output_dir))


def save_all_figures(
    figures: Dict[str, Figure],
    output_dir: Union[str, Path],
    format: str = "pdf",
    dpi: int = 150
) -> Dict[str, Path]:
    """
    Save multiple figures to directory.

    Args:
        figures: Dictionary of name -> figure
        output_dir: Output directory
        format: Save format (pdf, png, svg)
        dpi: Resolution

    Returns:
        Dictionary of name -> saved path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for name, fig in figures.items():
        path = output_dir / f"{name}.{format}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved[name] = path

    return saved
