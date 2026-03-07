"""
Diagnostic Plots for Dissertation-Quality Evaluation

Produces the 7 required plots for every model/strategy pair:
1. Equity curve (cumulative net returns)
2. Drawdown curve
3. Rolling Sharpe (configurable window)
4. Return histogram with normal overlay
5. Predicted vs realised returns scatter + IC
6. Positions & turnover time-series
7. Quantile (decile) analysis

Integrates with the web-app pipeline: plots are saved to ``results/evaluation/``
and can be served via the FastAPI static-files endpoint.

Usage::

    from src.evaluation.plot_diagnostics import DiagnosticPlotter

    plotter = DiagnosticPlotter()
    paths = plotter.plot_all(
        net_returns, positions, predictions, actuals,
        model_name="LSTM", output_dir=Path("results/evaluation")
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DiagnosticPlotter:
    """Generate all 7 required dissertation plots."""

    def __init__(
        self,
        figsize: tuple = (12, 6),
        style: str = "seaborn-v0_8-whitegrid",
        dpi: int = 150,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.02,
    ):
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    # ===== Master entry point =============================================
    def plot_all(
        self,
        net_returns: np.ndarray,
        positions: np.ndarray,
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str = "Model",
        output_dir: Optional[Path] = None,
        timestamps: Optional[np.ndarray] = None,
        rolling_window: int = 63,
        n_quantiles: int = 10,
        train_loss: Optional[np.ndarray] = None,
        val_loss: Optional[np.ndarray] = None,
        predicted_prices: Optional[np.ndarray] = None,
        actual_prices: Optional[np.ndarray] = None,
    ) -> List[Path]:
        """Generate all 7 required plots and save to *output_dir*.

        Returns list of saved file paths.
        """
        if output_dir is None:
            output_dir = Path("results/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        prefix = model_name.lower().replace(" ", "_")

        # 1. Equity curve
        fig = self.plot_equity_curve(net_returns, timestamps, model_name)
        p = output_dir / f"{prefix}_equity_curve.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 2. Drawdown
        fig = self.plot_drawdown(net_returns, timestamps, model_name)
        p = output_dir / f"{prefix}_drawdown.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 3. Rolling Sharpe
        fig = self.plot_rolling_sharpe(net_returns, timestamps, rolling_window, model_name)
        p = output_dir / f"{prefix}_rolling_sharpe.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 4. Return histogram
        fig = self.plot_return_histogram(net_returns, model_name)
        p = output_dir / f"{prefix}_return_histogram.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 5. Predicted vs realised scatter
        fig = self.plot_pred_vs_realized(predictions, actuals, model_name)
        p = output_dir / f"{prefix}_pred_vs_realized.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 6. Positions & turnover
        fig = self.plot_positions_turnover(positions, timestamps, model_name)
        p = output_dir / f"{prefix}_positions_turnover.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 7. Quantile analysis
        fig = self.plot_quantile_analysis(predictions, actuals, n_quantiles, model_name)
        p = output_dir / f"{prefix}_quantile_analysis.png"
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # 8. Training vs validation loss (optional)
        if train_loss is not None and val_loss is not None:
            fig = self.plot_loss_curves(train_loss, val_loss, model_name)
            p = output_dir / f"{prefix}_loss_curves.png"
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        # 9. Predicted price vs actual price overlay (optional)
        if predicted_prices is not None and actual_prices is not None:
            fig = self.plot_price_overlay(predicted_prices, actual_prices, timestamps, model_name)
            p = output_dir / f"{prefix}_price_overlay.png"
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        logger.info(f"Saved {len(paths)} diagnostic plots to {output_dir}")
        return paths

    # ===== Individual plots ===============================================

    def plot_equity_curve(
        self,
        net_returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 1: Cumulative net returns (equity curve)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        cum = np.cumprod(1 + net_returns) - 1
        x = timestamps if timestamps is not None else np.arange(len(cum))

        ax.plot(x, cum * 100, linewidth=1.5, color="#2196F3")
        ax.fill_between(x, 0, cum * 100, alpha=0.15, color="#2196F3")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"{title} — Equity Curve (Cumulative Net Return)", fontsize=14)
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_drawdown(
        self,
        net_returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 2: Drawdown curve over time."""
        fig, ax = plt.subplots(figsize=self.figsize)
        cum = np.cumprod(1 + net_returns)
        running_max = np.maximum.accumulate(cum)
        drawdown = (cum - running_max) / running_max * 100
        x = timestamps if timestamps is not None else np.arange(len(drawdown))

        ax.fill_between(x, drawdown, 0, alpha=0.5, color="#F44336")
        ax.plot(x, drawdown, linewidth=1.0, color="#D32F2F")
        ax.set_title(f"{title} — Drawdown", fontsize=14)
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_rolling_sharpe(
        self,
        net_returns: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        window: int = 63,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 3: Rolling Sharpe ratio (annualised)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        n = len(net_returns)
        rolling = np.full(n, np.nan)
        rf_daily = self.risk_free_rate / self.periods_per_year

        for i in range(window, n):
            chunk = net_returns[i - window:i]
            mu = np.mean(chunk) - rf_daily
            sigma = np.std(chunk, ddof=1)
            rolling[i] = (mu / sigma * np.sqrt(self.periods_per_year)) if sigma > 1e-10 else 0.0

        x = timestamps if timestamps is not None else np.arange(n)
        ax.plot(x, rolling, linewidth=1.2, color="#4CAF50")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"{title} — Rolling Sharpe ({window}-day window)", fontsize=14)
        ax.set_ylabel("Annualised Sharpe")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_return_histogram(
        self,
        net_returns: np.ndarray,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 4: Histogram of daily net strategy returns + normal overlay."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(net_returns * 100, bins=50, density=True, alpha=0.7,
                color="#9C27B0", edgecolor="white", linewidth=0.5)

        # Normal overlay
        mu, sigma = np.mean(net_returns) * 100, np.std(net_returns) * 100
        if sigma > 1e-10:
            x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
            ax.plot(x_norm, pdf, "k--", linewidth=1.5, label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})")
            ax.legend()

        ax.set_title(f"{title} — Daily Net Return Distribution", fontsize=14)
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_pred_vs_realized(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 5: Predicted vs realised returns scatter + IC."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Subsample if too many points
        n = len(predictions)
        if n > 2000:
            idx = np.random.choice(n, 2000, replace=False)
            pred_plot, act_plot = predictions[idx], actuals[idx]
        else:
            pred_plot, act_plot = predictions, actuals

        ax.scatter(pred_plot * 100, act_plot * 100, alpha=0.3, s=10, color="#FF9800")

        # Correlation
        valid = ~(np.isnan(predictions) | np.isnan(actuals))
        ic = float(np.corrcoef(predictions[valid], actuals[valid])[0, 1]) if valid.sum() > 2 else 0.0

        # Regression line
        if valid.sum() > 2:
            z = np.polyfit(predictions[valid], actuals[valid], 1)
            x_line = np.linspace(predictions[valid].min(), predictions[valid].max(), 100)
            ax.plot(x_line * 100, np.polyval(z, x_line) * 100, "r-", linewidth=2, label=f"IC = {ic:.3f}")
            ax.legend(fontsize=12)

        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(f"{title} — Predicted vs Realised Returns", fontsize=14)
        ax.set_xlabel("Predicted Return (%)")
        ax.set_ylabel("Realised Return (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_positions_turnover(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 6: Time series of positions w_t and turnover |Δw_t|."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2),
                                        sharex=True)
        x = timestamps if timestamps is not None else np.arange(len(positions))
        turnover = np.abs(np.diff(np.concatenate([[0.0], positions])))

        # Positions
        ax1.fill_between(x, positions, 0, alpha=0.4,
                         where=positions >= 0, color="#4CAF50", label="Long")
        ax1.fill_between(x, positions, 0, alpha=0.4,
                         where=positions < 0, color="#F44336", label="Short")
        ax1.set_ylabel("Position (w_t)")
        ax1.set_title(f"{title} — Positions & Turnover", fontsize=14)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Turnover
        ax2.bar(x, turnover, width=1.0, alpha=0.6, color="#FF9800")
        ax2.set_ylabel("|Δw_t| (Turnover)")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_quantile_analysis(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_quantiles: int = 10,
        title: str = "Model",
    ) -> plt.Figure:
        """Plot 7: Quantile (decile) analysis — mean realised return per prediction bucket."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Bucket predictions into quantiles
        valid = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_valid = predictions[valid]
        act_valid = actuals[valid]

        if len(pred_valid) < n_quantiles * 5:
            ax.text(0.5, 0.5, "Insufficient data for quantile analysis",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)
            return fig

        try:
            quantile_labels = pd.qcut(pred_valid, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            # Fallback if too many duplicate values
            quantile_labels = pd.cut(pred_valid, n_quantiles, labels=False)

        unique_q = np.sort(np.unique(quantile_labels[~np.isnan(quantile_labels)]))
        means = []
        labels = []
        for q in unique_q:
            mask = quantile_labels == q
            if mask.sum() > 0:
                means.append(float(np.mean(act_valid[mask])) * 100)
                labels.append(f"Q{int(q)+1}")

        colors = ["#F44336" if m < 0 else "#4CAF50" for m in means]
        ax.bar(labels, means, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"{title} — Quantile Analysis (Mean Realised Return per Prediction Decile)", fontsize=13)
        ax.set_xlabel(f"Prediction Quantile (Q1=lowest … Q{len(labels)}=highest)")
        ax.set_ylabel("Mean Realised Return (%)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    def plot_loss_curves(
        self,
        train_loss: np.ndarray,
        val_loss: np.ndarray,
        title: str = "Model",
    ) -> plt.Figure:
        """Optional: Training vs validation loss curves."""
        fig, ax = plt.subplots(figsize=self.figsize)
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="Train", linewidth=1.5, color="#1B998B")
        ax.plot(epochs, val_loss, label="Val", linewidth=1.5, linestyle="--", color="#E84855")
        ax.set_title(f"{title} — Training vs Validation Loss", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_price_overlay(
        self,
        predicted_prices: np.ndarray,
        actual_prices: np.ndarray,
        timestamps: Optional[np.ndarray],
        title: str = "Model",
    ) -> plt.Figure:
        """Optional: Predicted price vs actual price over time."""
        fig, ax = plt.subplots(figsize=self.figsize)
        x = timestamps if timestamps is not None else np.arange(len(actual_prices))
        ax.plot(x, actual_prices, label="Actual", linewidth=1.5, color="#2E86AB")
        ax.plot(x, predicted_prices, label="Predicted", linewidth=1.2, linestyle="--", color="#F59E0B")
        ax.set_title(f"{title} — Predicted vs Actual Price", fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
