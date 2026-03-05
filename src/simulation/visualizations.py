"""
Publication-Quality Visualizations for Regime-Switching Monte Carlo

This module provides comprehensive visualization tools for comparing
standard vs regime-switching Monte Carlo simulations.

Visualizations:
1. Simulated path comparison
2. Terminal return distributions
3. Tail density comparison (QQ-plots, kernel density)
4. Regime evolution over time
5. Transition matrix heatmap
6. Drawdown analysis
7. Risk metrics comparison

Design Principles:
- Publication-quality figures (300+ DPI)
- Consistent color scheme
- Clear labels and legends
- LaTeX-compatible fonts (optional)

Author: Dissertation Research Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Optional imports - visualization is not required for core functionality
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)

if not HAS_MATPLOTLIB:
    logger.warning("matplotlib not installed. Visualizations will be disabled.")

# =============================================================================
# Style Configuration
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    'standard': '#1f77b4',      # Blue
    'regime': '#d62728',        # Red
    'low_vol': '#2ca02c',       # Green
    'normal': '#ff7f0e',        # Orange
    'high_vol': '#9467bd',      # Purple
    'neutral': '#7f7f7f',       # Gray
    'confidence': '#a6cee3',    # Light blue
}

# Figure defaults
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 150
PUBLICATION_DPI = 300


def setup_publication_style():
    """Configure matplotlib for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'figure.figsize': DEFAULT_FIGSIZE,
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': PUBLICATION_DPI,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })


# Initialize style
try:
    setup_publication_style()
except Exception:
    pass  # Fallback to default style


# =============================================================================
# Simulation Path Visualization
# =============================================================================

def plot_simulation_comparison(
    standard_paths: np.ndarray,
    regime_paths: np.ndarray,
    n_sample_paths: int = 100,
    initial_price: float = 100.0,
    title: str = "Monte Carlo Simulation: Standard vs Regime-Switching",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of simulated paths from both models

    Args:
        standard_paths: Paths from standard MC [n_paths, horizon+1]
        regime_paths: Paths from regime-switching MC [n_paths, horizon+1]
        n_sample_paths: Number of sample paths to plot
        initial_price: Starting price
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    horizon = standard_paths.shape[1] - 1
    days = np.arange(horizon + 1)

    # Sample paths for visualization
    n_paths = min(n_sample_paths, standard_paths.shape[0])
    sample_idx = np.random.choice(standard_paths.shape[0], n_paths, replace=False)

    # Plot 1: Standard MC paths
    ax1 = axes[0, 0]
    for i in sample_idx[:50]:  # Limit to 50 for clarity
        ax1.plot(days, standard_paths[i], alpha=0.1, color=COLORS['standard'])

    # Mean and percentiles
    mean_std = np.mean(standard_paths, axis=0)
    p5_std = np.percentile(standard_paths, 5, axis=0)
    p95_std = np.percentile(standard_paths, 95, axis=0)

    ax1.plot(days, mean_std, color=COLORS['standard'], linewidth=2, label='Mean')
    ax1.fill_between(days, p5_std, p95_std, alpha=0.2, color=COLORS['standard'], label='90% CI')
    ax1.axhline(y=initial_price, color='black', linestyle='--', alpha=0.5)

    ax1.set_title("Standard Monte Carlo")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, horizon)

    # Plot 2: Regime-switching paths
    ax2 = axes[0, 1]
    for i in sample_idx[:50]:
        ax2.plot(days, regime_paths[i], alpha=0.1, color=COLORS['regime'])

    mean_reg = np.mean(regime_paths, axis=0)
    p5_reg = np.percentile(regime_paths, 5, axis=0)
    p95_reg = np.percentile(regime_paths, 95, axis=0)

    ax2.plot(days, mean_reg, color=COLORS['regime'], linewidth=2, label='Mean')
    ax2.fill_between(days, p5_reg, p95_reg, alpha=0.2, color=COLORS['regime'], label='90% CI')
    ax2.axhline(y=initial_price, color='black', linestyle='--', alpha=0.5)

    ax2.set_title("Regime-Switching Monte Carlo")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price")
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, horizon)

    # Plot 3: Confidence interval comparison
    ax3 = axes[1, 0]
    ax3.fill_between(days, p5_std, p95_std, alpha=0.3, color=COLORS['standard'],
                     label='Standard 90% CI')
    ax3.fill_between(days, p5_reg, p95_reg, alpha=0.3, color=COLORS['regime'],
                     label='Regime 90% CI')
    ax3.plot(days, mean_std, color=COLORS['standard'], linewidth=2, label='Standard Mean')
    ax3.plot(days, mean_reg, color=COLORS['regime'], linewidth=2, label='Regime Mean')
    ax3.axhline(y=initial_price, color='black', linestyle='--', alpha=0.5)

    ax3.set_title("Confidence Interval Comparison")
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Price")
    ax3.legend(loc='upper left')
    ax3.set_xlim(0, horizon)

    # Plot 4: Terminal distribution comparison
    ax4 = axes[1, 1]
    terminal_std = standard_paths[:, -1]
    terminal_reg = regime_paths[:, -1]

    ax4.hist(terminal_std, bins=50, alpha=0.5, color=COLORS['standard'],
             label=f'Standard (μ={np.mean(terminal_std):.1f})', density=True)
    ax4.hist(terminal_reg, bins=50, alpha=0.5, color=COLORS['regime'],
             label=f'Regime (μ={np.mean(terminal_reg):.1f})', density=True)

    ax4.axvline(x=initial_price, color='black', linestyle='--', alpha=0.5, label='Initial')
    ax4.axvline(x=np.percentile(terminal_std, 5), color=COLORS['standard'],
                linestyle=':', linewidth=2, label='Std VaR 95%')
    ax4.axvline(x=np.percentile(terminal_reg, 5), color=COLORS['regime'],
                linestyle=':', linewidth=2, label='Reg VaR 95%')

    ax4.set_title("Terminal Price Distribution")
    ax4.set_xlabel("Terminal Price")
    ax4.set_ylabel("Density")
    ax4.legend(loc='upper right')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PUBLICATION_DPI, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


# =============================================================================
# Distribution Visualization
# =============================================================================

def plot_tail_comparison(
    standard_returns: np.ndarray,
    regime_returns: np.ndarray,
    title: str = "Tail Risk Comparison",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare tail distributions between standard and regime-switching MC

    Creates:
    1. Histogram comparison
    2. Log-scale tail density
    3. QQ-plot vs normal
    4. Empirical CDF comparison

    Args:
        standard_returns: Terminal returns from standard MC
        regime_returns: Terminal returns from regime-switching MC
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    std_rets = standard_returns.flatten()
    reg_rets = regime_returns.flatten()

    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    bins = np.linspace(min(std_rets.min(), reg_rets.min()),
                       max(std_rets.max(), reg_rets.max()), 80)

    ax1.hist(std_rets, bins=bins, alpha=0.5, color=COLORS['standard'],
             label='Standard', density=True)
    ax1.hist(reg_rets, bins=bins, alpha=0.5, color=COLORS['regime'],
             label='Regime-Switching', density=True)

    # Fit and plot normal for reference
    x = np.linspace(bins[0], bins[-1], 200)
    mu_std, sigma_std = np.mean(std_rets), np.std(std_rets)
    ax1.plot(x, stats.norm.pdf(x, mu_std, sigma_std), '--',
             color=COLORS['standard'], linewidth=2, label='Normal fit (Std)')

    ax1.set_title("Return Distribution Comparison")
    ax1.set_xlabel("Terminal Return")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Plot 2: Log-scale tail density (focus on tails)
    ax2 = axes[0, 1]

    # Left tail (losses)
    left_threshold = np.percentile(np.concatenate([std_rets, reg_rets]), 10)
    left_std = std_rets[std_rets < left_threshold]
    left_reg = reg_rets[reg_rets < left_threshold]

    bins_left = np.linspace(min(left_std.min(), left_reg.min()),
                            left_threshold, 30)

    counts_std, _ = np.histogram(left_std, bins=bins_left, density=True)
    counts_reg, _ = np.histogram(left_reg, bins=bins_left, density=True)
    bin_centers = (bins_left[:-1] + bins_left[1:]) / 2

    ax2.semilogy(bin_centers, counts_std + 1e-10, 'o-', color=COLORS['standard'],
                 label='Standard (left tail)', markersize=4)
    ax2.semilogy(bin_centers, counts_reg + 1e-10, 's-', color=COLORS['regime'],
                 label='Regime (left tail)', markersize=4)

    ax2.set_title("Left Tail Density (Log Scale)")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Log Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: QQ-plot vs normal
    ax3 = axes[1, 0]

    # Standard MC QQ
    osm_std, osr_std = stats.probplot(std_rets, dist="norm", fit=False)
    ax3.scatter(osm_std, osr_std, alpha=0.3, s=10, color=COLORS['standard'],
                label='Standard')

    # Regime MC QQ
    osm_reg, osr_reg = stats.probplot(reg_rets, dist="norm", fit=False)
    ax3.scatter(osm_reg, osr_reg, alpha=0.3, s=10, color=COLORS['regime'],
                label='Regime-Switching')

    # Reference line
    xlim = ax3.get_xlim()
    ax3.plot(xlim, xlim, 'k--', linewidth=1, label='Normal reference')

    ax3.set_title("QQ-Plot vs Normal Distribution")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.legend()

    # Plot 4: Empirical CDF comparison
    ax4 = axes[1, 1]

    # Sort returns for CDF
    std_sorted = np.sort(std_rets)
    reg_sorted = np.sort(reg_rets)
    n_std = len(std_sorted)
    n_reg = len(reg_sorted)

    # CDF
    ax4.plot(std_sorted, np.arange(1, n_std + 1) / n_std,
             color=COLORS['standard'], label='Standard CDF')
    ax4.plot(reg_sorted, np.arange(1, n_reg + 1) / n_reg,
             color=COLORS['regime'], label='Regime CDF')

    # Mark VaR levels
    var_95_std = np.percentile(std_rets, 5)
    var_95_reg = np.percentile(reg_rets, 5)
    ax4.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=var_95_std, color=COLORS['standard'], linestyle=':',
                label=f'VaR 95% Std: {var_95_std:.3f}')
    ax4.axvline(x=var_95_reg, color=COLORS['regime'], linestyle=':',
                label=f'VaR 95% Reg: {var_95_reg:.3f}')

    ax4.set_title("Empirical CDF Comparison")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Cumulative Probability")
    ax4.legend(loc='lower right')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PUBLICATION_DPI, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


# =============================================================================
# Regime Visualization
# =============================================================================

def plot_regime_evolution(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    regime_probs: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Market Regime Evolution",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize regime evolution over time

    Args:
        returns: Historical return series
        regime_labels: Regime labels (0, 1, 2)
        regime_probs: Optional regime probabilities [n, n_regimes]
        dates: Optional date index
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1])

    n = len(returns)
    x = dates if dates is not None else np.arange(n)

    regime_colors = [COLORS['low_vol'], COLORS['normal'], COLORS['high_vol']]
    regime_names = ['Low Volatility', 'Normal', 'High Volatility']

    # Plot 1: Returns with regime coloring
    ax1 = axes[0]

    # Plot returns
    ax1.bar(x, returns, width=1, alpha=0.6, color='gray')

    # Color background by regime
    for i in range(3):
        mask = regime_labels == i
        if np.any(mask):
            # Find contiguous regions
            changes = np.diff(mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if mask[0]:
                starts = np.concatenate([[0], starts])
            if mask[-1]:
                ends = np.concatenate([ends, [n]])

            for start, end in zip(starts, ends):
                if dates is not None:
                    ax1.axvspan(x[start], x[min(end, n-1)],
                               alpha=0.2, color=regime_colors[i])
                else:
                    ax1.axvspan(start, end, alpha=0.2, color=regime_colors[i])

    ax1.set_title("Returns with Regime Classification")
    ax1.set_ylabel("Return")
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=regime_colors[i], alpha=0.3, label=regime_names[i])
                       for i in range(3)]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Plot 2: Regime labels
    ax2 = axes[1]

    # Create step plot of regime
    ax2.step(x, regime_labels, where='post', color='black', linewidth=1)
    ax2.fill_between(x, 0, regime_labels, step='post', alpha=0.3)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Low Vol', 'Normal', 'High Vol'])
    ax2.set_ylabel("Regime")
    ax2.set_title("Regime State Over Time")

    # Plot 3: Rolling volatility
    ax3 = axes[2]

    # Compute rolling volatility (21-day)
    window = min(21, n // 5)
    if window > 1:
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        ax3.plot(x, rolling_vol, color='purple', linewidth=1.5, label='21-day Rolling Vol')
        ax3.fill_between(x, 0, rolling_vol, alpha=0.2, color='purple')

    ax3.set_ylabel("Annualized Volatility")
    ax3.set_xlabel("Time" if dates is None else "Date")
    ax3.set_title("Rolling Volatility")
    ax3.legend()

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PUBLICATION_DPI, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    regime_names: Optional[List[str]] = None,
    title: str = "Regime Transition Probability Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create heatmap visualization of transition matrix

    Args:
        transition_matrix: n x n transition probability matrix
        regime_names: Names for each regime
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = transition_matrix.shape[0]
    if regime_names is None:
        regime_names = ['Low Vol', 'Normal', 'High Vol'][:n]

    # Create custom colormap (white to blue)
    cmap = plt.cm.Blues

    # Plot heatmap
    im = ax.imshow(transition_matrix, cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Transition Probability")

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(regime_names)
    ax.set_yticklabels(regime_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            value = transition_matrix[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                   color=text_color, fontsize=12, fontweight='bold')

    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PUBLICATION_DPI, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


# =============================================================================
# Risk Metrics Visualization
# =============================================================================

def plot_risk_comparison(
    comparison: Dict[str, Dict[str, float]],
    title: str = "Risk Metrics Comparison",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar chart comparison of risk metrics

    Args:
        comparison: Output from RiskMetricsCalculator.compare_simulations()
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    std = comparison['standard']
    reg = comparison['regime_switching']

    # Plot 1: VaR comparison
    ax1 = axes[0, 0]
    var_metrics = ['var_95_hist', 'var_99_hist']
    x = np.arange(len(var_metrics))
    width = 0.35

    std_vals = [abs(std[m]) for m in var_metrics]
    reg_vals = [abs(reg[m]) for m in var_metrics]

    bars1 = ax1.bar(x - width/2, std_vals, width, label='Standard', color=COLORS['standard'])
    bars2 = ax1.bar(x + width/2, reg_vals, width, label='Regime-Switching', color=COLORS['regime'])

    ax1.set_ylabel('|VaR| (Loss)')
    ax1.set_title('Value at Risk Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['VaR 95%', 'VaR 99%'])
    ax1.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9)

    # Plot 2: ES comparison
    ax2 = axes[0, 1]
    es_metrics = ['es_95', 'es_99']
    x = np.arange(len(es_metrics))

    std_vals = [abs(std[m]) for m in es_metrics]
    reg_vals = [abs(reg[m]) for m in es_metrics]

    bars1 = ax2.bar(x - width/2, std_vals, width, label='Standard', color=COLORS['standard'])
    bars2 = ax2.bar(x + width/2, reg_vals, width, label='Regime-Switching', color=COLORS['regime'])

    ax2.set_ylabel('|Expected Shortfall| (Loss)')
    ax2.set_title('Expected Shortfall Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['ES 95%', 'ES 99%'])
    ax2.legend()

    # Plot 3: Distribution metrics
    ax3 = axes[1, 0]
    dist_metrics = ['volatility', 'skewness', 'kurtosis']
    x = np.arange(len(dist_metrics))

    std_vals = [std[m] for m in dist_metrics]
    reg_vals = [reg[m] for m in dist_metrics]

    bars1 = ax3.bar(x - width/2, std_vals, width, label='Standard', color=COLORS['standard'])
    bars2 = ax3.bar(x + width/2, reg_vals, width, label='Regime-Switching', color=COLORS['regime'])

    ax3.set_ylabel('Value')
    ax3.set_title('Distribution Characteristics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Volatility', 'Skewness', 'Kurtosis'])
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 4: Risk-adjusted metrics
    ax4 = axes[1, 1]
    ra_metrics = ['sharpe_ratio', 'sortino_ratio']
    x = np.arange(len(ra_metrics))

    std_vals = [std[m] for m in ra_metrics]
    reg_vals = [reg[m] for m in ra_metrics]

    bars1 = ax4.bar(x - width/2, std_vals, width, label='Standard', color=COLORS['standard'])
    bars2 = ax4.bar(x + width/2, reg_vals, width, label='Regime-Switching', color=COLORS['regime'])

    ax4.set_ylabel('Ratio')
    ax4.set_title('Risk-Adjusted Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Sharpe Ratio', 'Sortino Ratio'])
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PUBLICATION_DPI, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


# =============================================================================
# Comprehensive Visualization Class
# =============================================================================

class RegimeVisualizer:
    """
    Comprehensive visualization suite for regime-switching Monte Carlo

    Provides unified interface for generating all visualizations.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        style: str = 'publication'
    ):
        """
        Args:
            output_dir: Directory to save figures
            style: 'publication' for high-quality, 'interactive' for display
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.style = style

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if style == 'publication':
            setup_publication_style()

    def plot_all(
        self,
        standard_result,  # SimulationResult
        regime_result,    # SimulationResult
        regime_estimates, # RegimeEstimates
        historical_returns: np.ndarray,
        comparison: Dict[str, Dict[str, float]],
        prefix: str = "mc_comparison"
    ) -> Dict[str, plt.Figure]:
        """
        Generate all visualizations

        Args:
            standard_result: Standard MC simulation result
            regime_result: Regime-switching MC result
            regime_estimates: Estimated regime parameters
            historical_returns: Original return series
            comparison: Risk metrics comparison
            prefix: Prefix for saved filenames

        Returns:
            Dict of figure names to Figure objects
        """
        figures = {}

        # 1. Path comparison
        save_path = str(self.output_dir / f"{prefix}_paths.png") if self.output_dir else None
        figures['paths'] = plot_simulation_comparison(
            standard_result.paths,
            regime_result.paths,
            initial_price=standard_result.initial_price,
            save_path=save_path
        )

        # 2. Tail comparison
        save_path = str(self.output_dir / f"{prefix}_tails.png") if self.output_dir else None
        figures['tails'] = plot_tail_comparison(
            standard_result.terminal_returns,
            regime_result.terminal_returns,
            save_path=save_path
        )

        # 3. Regime evolution
        save_path = str(self.output_dir / f"{prefix}_regimes.png") if self.output_dir else None
        figures['regimes'] = plot_regime_evolution(
            historical_returns,
            regime_estimates.regime_labels,
            regime_estimates.regime_probabilities,
            save_path=save_path
        )

        # 4. Transition matrix
        save_path = str(self.output_dir / f"{prefix}_transition.png") if self.output_dir else None
        figures['transition'] = plot_transition_matrix(
            regime_estimates.transition_matrix,
            save_path=save_path
        )

        # 5. Risk comparison
        save_path = str(self.output_dir / f"{prefix}_risk.png") if self.output_dir else None
        figures['risk'] = plot_risk_comparison(
            comparison,
            save_path=save_path
        )

        logger.info(f"Generated {len(figures)} visualizations")

        return figures

    def create_summary_figure(
        self,
        standard_result,
        regime_result,
        regime_estimates,
        comparison: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a single summary figure combining key visualizations

        Good for dissertation/paper where space is limited.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Top row: Path comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Middle row: Distributions
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        # Bottom row: Metrics
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        # --- Top Row ---

        # Path samples (standard)
        horizon = standard_result.paths.shape[1]
        days = np.arange(horizon)
        for i in range(min(50, standard_result.n_paths)):
            ax1.plot(days, standard_result.paths[i], alpha=0.05, color=COLORS['standard'])
        ax1.plot(days, np.mean(standard_result.paths, axis=0), color=COLORS['standard'], lw=2)
        ax1.set_title("Standard MC Paths")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")

        # Path samples (regime)
        for i in range(min(50, regime_result.n_paths)):
            ax2.plot(days, regime_result.paths[i], alpha=0.05, color=COLORS['regime'])
        ax2.plot(days, np.mean(regime_result.paths, axis=0), color=COLORS['regime'], lw=2)
        ax2.set_title("Regime-Switching MC Paths")
        ax2.set_xlabel("Days")

        # Transition matrix
        im = ax3.imshow(regime_estimates.transition_matrix, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                val = regime_estimates.transition_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels(['L', 'N', 'H'])
        ax3.set_yticklabels(['L', 'N', 'H'])
        ax3.set_title("Transition Matrix")
        ax3.set_xlabel("To")
        ax3.set_ylabel("From")

        # --- Middle Row ---

        # Terminal distribution
        ax4.hist(standard_result.terminal_returns, bins=50, alpha=0.5,
                 color=COLORS['standard'], density=True, label='Standard')
        ax4.hist(regime_result.terminal_returns, bins=50, alpha=0.5,
                 color=COLORS['regime'], density=True, label='Regime')
        ax4.set_title("Terminal Return Distribution")
        ax4.set_xlabel("Return")
        ax4.legend()

        # QQ plot
        osm_std, osr_std = stats.probplot(standard_result.terminal_returns, dist="norm", fit=False)
        osm_reg, osr_reg = stats.probplot(regime_result.terminal_returns, dist="norm", fit=False)
        ax5.scatter(osm_std, osr_std, alpha=0.3, s=5, color=COLORS['standard'], label='Std')
        ax5.scatter(osm_reg, osr_reg, alpha=0.3, s=5, color=COLORS['regime'], label='Reg')
        lims = [min(ax5.get_xlim()[0], ax5.get_ylim()[0]),
                max(ax5.get_xlim()[1], ax5.get_ylim()[1])]
        ax5.plot(lims, lims, 'k--', alpha=0.5)
        ax5.set_title("QQ Plot vs Normal")
        ax5.legend()

        # Regime probabilities
        if regime_estimates.regime_probabilities is not None:
            n = len(regime_estimates.regime_labels)
            x = np.arange(n)
            ax6.stackplot(x, regime_estimates.regime_probabilities.T,
                         colors=[COLORS['low_vol'], COLORS['normal'], COLORS['high_vol']],
                         alpha=0.7, labels=['Low Vol', 'Normal', 'High Vol'])
            ax6.set_title("Regime Probabilities Over Time")
            ax6.set_xlabel("Time")
            ax6.legend(loc='upper right', fontsize=8)

        # --- Bottom Row ---

        # VaR comparison
        std = comparison['standard']
        reg = comparison['regime_switching']

        metrics = ['var_95_hist', 'var_99_hist']
        x = np.arange(len(metrics))
        width = 0.35
        ax7.bar(x - width/2, [abs(std[m]) for m in metrics], width,
                color=COLORS['standard'], label='Standard')
        ax7.bar(x + width/2, [abs(reg[m]) for m in metrics], width,
                color=COLORS['regime'], label='Regime')
        ax7.set_xticks(x)
        ax7.set_xticklabels(['VaR 95%', 'VaR 99%'])
        ax7.set_title("Value at Risk")
        ax7.legend()

        # ES comparison
        metrics = ['es_95', 'es_99']
        x = np.arange(len(metrics))
        ax8.bar(x - width/2, [abs(std[m]) for m in metrics], width,
                color=COLORS['standard'], label='Standard')
        ax8.bar(x + width/2, [abs(reg[m]) for m in metrics], width,
                color=COLORS['regime'], label='Regime')
        ax8.set_xticks(x)
        ax8.set_xticklabels(['ES 95%', 'ES 99%'])
        ax8.set_title("Expected Shortfall")
        ax8.legend()

        # Distribution stats
        metrics = ['skewness', 'kurtosis']
        x = np.arange(len(metrics))
        ax9.bar(x - width/2, [std[m] for m in metrics], width,
                color=COLORS['standard'], label='Standard')
        ax9.bar(x + width/2, [reg[m] for m in metrics], width,
                color=COLORS['regime'], label='Regime')
        ax9.set_xticks(x)
        ax9.set_xticklabels(['Skewness', 'Kurtosis'])
        ax9.set_title("Distribution Shape")
        ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax9.legend()

        plt.suptitle("Regime-Switching Monte Carlo: Summary Analysis", fontsize=16, y=1.01)

        if save_path:
            plt.savefig(save_path, dpi=PUBLICATION_DPI, bbox_inches='tight')
            logger.info(f"Summary figure saved to {save_path}")

        return fig


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    """Demonstrate visualization capabilities"""

    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_simulation_comparison()")
    print("  - plot_tail_comparison()")
    print("  - plot_regime_evolution()")
    print("  - plot_transition_matrix()")
    print("  - plot_risk_comparison()")
    print("  - RegimeVisualizer.plot_all()")
    print("  - RegimeVisualizer.create_summary_figure()")
