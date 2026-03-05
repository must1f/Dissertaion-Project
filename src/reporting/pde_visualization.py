"""
PDE Visualization for Physics-Informed Neural Networks

Visualization suite for Burgers' equation PINN results:
- 3D surface plots (exact, predicted, error)
- Error heatmaps
- L2 error vs time curves
- Training loss curves (log-scale)
- Spatial slices at fixed times
- Error histograms

Designed for dissertation-quality figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .plot_generator import PlotConfig, PlotStyle

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import Normalize

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..utils.logger import get_logger
from ..evaluation.pde_evaluator import PDEEvaluator, PDEMetrics

logger = get_logger(__name__)


class BurgersVisualization:
    """
    Visualization generator for Burgers' equation PINN results.

    Creates publication-quality figures including:
    - 3D surface plots
    - Error heatmaps
    - Time-resolved error curves
    - Loss curves
    - Spatial slices
    - Error distributions
    """

    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        figsize_3d: Tuple[float, float] = (12, 8),
        figsize_2d: Tuple[float, float] = (10, 6),
        colormap: str = "coolwarm",
        error_colormap: str = "hot",
    ):
        """
        Initialize BurgersVisualization.

        Args:
            config: PlotConfig for styling
            figsize_3d: Figure size for 3D plots
            figsize_2d: Figure size for 2D plots
            colormap: Colormap for solution plots
            error_colormap: Colormap for error plots
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        self.config = config or PlotConfig(style=PlotStyle.PUBLICATION)
        self.figsize_3d = figsize_3d
        self.figsize_2d = figsize_2d
        self.colormap = colormap
        self.error_colormap = error_colormap

        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style."""
        plt.rcParams.update(
            {
                "font.family": self.config.font_family,
                "font.size": self.config.font_size,
                "axes.titlesize": self.config.title_size,
                "axes.labelsize": self.config.label_size,
                "legend.fontsize": self.config.legend_size,
                "figure.dpi": self.config.dpi,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    def plot_3d_surface(
        self,
        X: np.ndarray,
        T: np.ndarray,
        U: np.ndarray,
        title: str = "Solution",
        xlabel: str = "x",
        ylabel: str = "t",
        zlabel: str = "u(x,t)",
        elev: float = 25,
        azim: float = -135,
    ) -> Figure:
        """
        Create 3D surface plot.

        Args:
            X: Spatial meshgrid [n_t, n_x]
            T: Temporal meshgrid [n_t, n_x]
            U: Solution values [n_t, n_x]
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z-axis label
            elev: Elevation angle
            azim: Azimuth angle

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize_3d)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X, T, U, cmap=self.colormap, edgecolor="none", alpha=0.9, antialiased=True
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        fig.colorbar(surf, shrink=0.5, aspect=10, label=zlabel)
        plt.tight_layout()

        return fig

    def plot_comparison_3d(
        self,
        grids: Dict[str, np.ndarray],
        titles: Optional[List[str]] = None,
    ) -> Figure:
        """
        Create side-by-side 3D surface comparison.

        Args:
            grids: Dictionary with X, T, u_exact, u_pred keys
            titles: Optional titles for subplots

        Returns:
            matplotlib Figure
        """
        if titles is None:
            titles = ["Exact Solution", "PINN Prediction", "Absolute Error"]

        fig = plt.figure(figsize=(16, 5))

        X = grids["X"]
        T = grids["T"]
        u_exact = grids["u_exact"]
        u_pred = grids["u_pred"]
        error = grids["error"]

        # Exact solution
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot_surface(X, T, u_exact, cmap=self.colormap, edgecolor="none")
        ax1.set_xlabel("x")
        ax1.set_ylabel("t")
        ax1.set_zlabel("u")
        ax1.set_title(titles[0])
        ax1.view_init(elev=25, azim=-135)

        # PINN prediction
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot_surface(X, T, u_pred, cmap=self.colormap, edgecolor="none")
        ax2.set_xlabel("x")
        ax2.set_ylabel("t")
        ax2.set_zlabel("u")
        ax2.set_title(titles[1])
        ax2.view_init(elev=25, azim=-135)

        # Error
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.plot_surface(X, T, error, cmap=self.error_colormap, edgecolor="none")
        ax3.set_xlabel("x")
        ax3.set_ylabel("t")
        ax3.set_zlabel("|error|")
        ax3.set_title(titles[2])
        ax3.view_init(elev=25, azim=-135)

        plt.tight_layout()
        return fig

    def plot_error_heatmap(
        self,
        grids: Dict[str, np.ndarray],
        log_scale: bool = False,
    ) -> Figure:
        """
        Create error heatmap.

        Args:
            grids: Dictionary with error grid
            log_scale: Use log scale for error

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize_2d)

        error = grids["error"]
        x_grid = grids["x_grid"]
        t_grid = grids["t_grid"]

        if log_scale:
            error = np.log10(error + 1e-10)
            label = "log$_{10}$|error|"
        else:
            label = "|error|"

        im = ax.imshow(
            error,
            extent=[x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]],
            origin="lower",
            aspect="auto",
            cmap=self.error_colormap,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title("Point-wise Prediction Error")
        fig.colorbar(im, ax=ax, label=label)

        plt.tight_layout()
        return fig

    def plot_time_resolved_error(
        self,
        time_slices: List[float],
        l2_errors: List[float],
        model_names: Optional[List[str]] = None,
        multiple_models: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
    ) -> Figure:
        """
        Plot L2 error vs time.

        Args:
            time_slices: Time values
            l2_errors: L2 errors at each time
            model_names: Optional model names for legend
            multiple_models: Dict of model_name -> (times, errors) for comparison

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize_2d)

        if multiple_models is not None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(multiple_models)))
            for (name, (times, errors)), color in zip(multiple_models.items(), colors):
                ax.plot(times, errors, label=name, color=color, linewidth=2)
        else:
            ax.plot(time_slices, l2_errors, linewidth=2, color="#3498db")

        ax.set_xlabel("Time t")
        ax.set_ylabel("Relative L$_2$ Error")
        ax.set_title("Time-Resolved L$_2$ Error")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        if multiple_models is not None or model_names is not None:
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_training_loss(
        self,
        history: Dict[str, List[float]],
        log_scale: bool = True,
    ) -> Figure:
        """
        Plot training loss curves.

        Args:
            history: Training history dictionary
            log_scale: Use log scale for y-axis

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iterations = history.get("iterations", range(len(history["losses"])))

        # Total loss
        ax1 = axes[0]
        ax1.plot(iterations, history["losses"], label="Total Loss", color="#3498db")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        if log_scale:
            ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Component losses
        ax2 = axes[1]
        if "pde_losses" in history and len(history["pde_losses"]) > 0:
            ax2.plot(
                iterations, history["pde_losses"], label="PDE Loss", color="#e74c3c"
            )
        if "ic_losses" in history and len(history["ic_losses"]) > 0:
            ax2.plot(
                iterations, history["ic_losses"], label="IC Loss", color="#2ecc71"
            )
        if "bc_losses" in history and len(history["bc_losses"]) > 0:
            ax2.plot(
                iterations, history["bc_losses"], label="BC Loss", color="#9b59b6"
            )
        if "intermediate_losses" in history and any(history["intermediate_losses"]):
            ax2.plot(
                iterations,
                history["intermediate_losses"],
                label="Intermediate Loss",
                color="#f39c12",
            )

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss Component")
        ax2.set_title("Loss Components")
        if log_scale:
            ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        return fig

    def plot_spatial_slices(
        self,
        grids: Dict[str, np.ndarray],
        time_points: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    ) -> Figure:
        """
        Plot spatial slices at fixed times.

        Args:
            grids: Dictionary with solution grids
            time_points: Times at which to plot slices

        Returns:
            matplotlib Figure
        """
        n_slices = len(time_points)
        n_cols = min(3, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_slices == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        x_grid = grids["x_grid"]
        t_grid = grids["t_grid"]
        u_exact = grids["u_exact"]
        u_pred = grids["u_pred"]

        for idx, t_val in enumerate(time_points):
            ax = axes[idx]

            # Find closest time index
            t_idx = np.argmin(np.abs(t_grid - t_val))
            actual_t = t_grid[t_idx]

            ax.plot(
                x_grid,
                u_exact[t_idx, :],
                label="Exact",
                color="#2ecc71",
                linewidth=2,
            )
            ax.plot(
                x_grid,
                u_pred[t_idx, :],
                label="PINN",
                color="#e74c3c",
                linestyle="--",
                linewidth=2,
            )

            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.set_title(f"t = {actual_t:.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_slices, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Solution at Different Time Slices", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_error_histogram(
        self,
        grids: Dict[str, np.ndarray],
        n_bins: int = 50,
    ) -> Figure:
        """
        Plot error histogram.

        Args:
            grids: Dictionary with error grid
            n_bins: Number of histogram bins

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        error = grids["error"].flatten()

        # Linear scale
        ax1 = axes[0]
        ax1.hist(error, bins=n_bins, color="#3498db", alpha=0.7, edgecolor="black")
        ax1.axvline(
            x=np.mean(error),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(error):.4f}",
        )
        ax1.axvline(
            x=np.median(error),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(error):.4f}",
        )
        ax1.set_xlabel("Absolute Error")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Error Distribution")
        ax1.legend()

        # Log scale
        ax2 = axes[1]
        log_error = np.log10(error + 1e-10)
        ax2.hist(log_error, bins=n_bins, color="#e74c3c", alpha=0.7, edgecolor="black")
        ax2.set_xlabel("log$_{10}$(Error)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Error Distribution (Log Scale)")

        plt.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        model_metrics: Dict[str, PDEMetrics],
    ) -> Figure:
        """
        Create bar chart comparing model performance.

        Args:
            model_metrics: Dictionary of model_name -> PDEMetrics

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        model_names = list(model_metrics.keys())
        n_models = len(model_names)

        # L2 error comparison
        ax1 = axes[0]
        l2_errors = [m.relative_l2_error for m in model_metrics.values()]
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        bars = ax1.bar(model_names, l2_errors, color=colors, edgecolor="black")
        ax1.set_ylabel("Relative L$_2$ Error")
        ax1.set_title("Model Comparison: L$_2$ Error")
        ax1.set_yscale("log")

        # Add value labels
        for bar, val in zip(bars, l2_errors):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Max error comparison
        ax2 = axes[1]
        max_errors = [m.max_error for m in model_metrics.values()]

        bars = ax2.bar(model_names, max_errors, color=colors, edgecolor="black")
        ax2.set_ylabel("Maximum Error")
        ax2.set_title("Model Comparison: Max Error")

        for bar, val in zip(bars, max_errors):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        return fig

    def generate_all(
        self,
        grids: Dict[str, np.ndarray],
        history: Dict[str, List[float]],
        metrics: PDEMetrics,
        output_dir: Union[str, Path],
        model_name: str = "pinn",
        file_format: str = "pdf",
    ) -> Dict[str, Path]:
        """
        Generate all visualization plots and save to directory.

        Args:
            grids: Prediction grids from PDEEvaluator
            history: Training history
            metrics: Evaluation metrics
            output_dir: Output directory
            model_name: Model name for filenames
            file_format: Output format (pdf, png, svg)

        Returns:
            Dictionary of plot_name -> saved path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = {}

        # 3D comparison
        fig = self.plot_comparison_3d(grids)
        path = output_dir / f"{model_name}_3d_comparison.{file_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["3d_comparison"] = path

        # Error heatmap
        fig = self.plot_error_heatmap(grids)
        path = output_dir / f"{model_name}_error_heatmap.{file_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["error_heatmap"] = path

        # Time-resolved error
        fig = self.plot_time_resolved_error(metrics.time_slices, metrics.time_resolved_l2)
        path = output_dir / f"{model_name}_time_error.{file_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["time_error"] = path

        # Training loss
        fig = self.plot_training_loss(history)
        path = output_dir / f"{model_name}_loss_curves.{file_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["loss_curves"] = path

        # Spatial slices
        fig = self.plot_spatial_slices(grids)
        path = output_dir / f"{model_name}_spatial_slices.{file_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["spatial_slices"] = path

        # Error histogram
        fig = self.plot_error_histogram(grids)
        path = output_dir / f"{model_name}_error_histogram.{file_format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["error_histogram"] = path

        logger.info(f"Generated {len(saved_plots)} plots in {output_dir}")
        return saved_plots


def create_comparison_visualization(
    models: Dict[str, nn.Module],
    evaluator: PDEEvaluator,
    histories: Dict[str, Dict],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Create comparison visualizations for multiple models.

    Args:
        models: Dictionary of model_name -> model
        evaluator: PDEEvaluator instance
        histories: Dictionary of model_name -> training_history
        output_dir: Output directory

    Returns:
        Dictionary of plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = BurgersVisualization()
    saved_plots = {}

    # Evaluate all models
    metrics_dict = {}
    grids_dict = {}
    time_errors_dict = {}

    for name, model in models.items():
        metrics = evaluator.evaluate_all(model)
        grids = evaluator.get_prediction_grid(model)
        metrics_dict[name] = metrics
        grids_dict[name] = grids
        time_errors_dict[name] = (metrics.time_slices, metrics.time_resolved_l2)

    # Model comparison bar chart
    fig = viz.plot_model_comparison(metrics_dict)
    path = output_dir / "model_comparison.pdf"
    fig.savefig(path, dpi=viz.config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_plots["model_comparison"] = path

    # Time-resolved error comparison
    fig = viz.plot_time_resolved_error([], [], multiple_models=time_errors_dict)
    path = output_dir / "time_error_comparison.pdf"
    fig.savefig(path, dpi=viz.config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_plots["time_error_comparison"] = path

    # Individual model visualizations
    for name in models:
        model_dir = output_dir / name
        model_plots = viz.generate_all(
            grids=grids_dict[name],
            history=histories.get(name, {"losses": []}),
            metrics=metrics_dict[name],
            output_dir=model_dir,
            model_name=name,
        )
        for plot_name, path in model_plots.items():
            saved_plots[f"{name}_{plot_name}"] = path

    return saved_plots
