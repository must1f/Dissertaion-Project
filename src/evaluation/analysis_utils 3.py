"""
Analysis Utilities for Comprehensive PINN Financial Forecasting Dashboard

This module provides data loading, processing, and computation utilities
for the comprehensive analysis dashboard. It handles:

1. Loading model results, training histories, and predictions
2. Computing derived metrics (rolling windows, regime analysis, etc.)
3. Generating backtest equity curves and drawdowns
4. Statistical analysis (Diebold-Mariano, bootstrap CIs, etc.)

Author: PINN Financial Forecasting Project
Documentation: All functions are fully documented with type hints and docstrings.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import scipy.stats as stats
from scipy.stats import norm, t as t_dist
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AnalysisConfig:
    """
    Configuration for analysis utilities.

    Attributes:
        project_root: Root directory of the project
        results_dir: Directory containing model results
        models_dir: Directory containing trained models and histories
        checkpoints_dir: Directory containing model checkpoints
        rolling_window_size: Default size for rolling window calculations
        risk_free_rate: Annual risk-free rate for Sharpe calculations
        trading_days_per_year: Number of trading days per year
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    results_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    checkpoints_dir: Path = field(default=None)
    rolling_window_size: int = 21  # ~1 month of trading days
    risk_free_rate: float = 0.02  # 2% annual
    trading_days_per_year: int = 252

    def __post_init__(self):
        """Set default paths based on project root."""
        if self.results_dir is None:
            self.results_dir = self.project_root / 'results'
        if self.models_dir is None:
            self.models_dir = self.project_root / 'Models'
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.project_root / 'checkpoints'


def get_config() -> AnalysisConfig:
    """
    Get the default analysis configuration.

    Returns:
        AnalysisConfig: Default configuration instance
    """
    return AnalysisConfig()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_training_history(
    model_name: str,
    config: Optional[AnalysisConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Load training history for a specific model.

    The function searches for history files using multiple naming patterns
    to handle different model naming conventions.

    Args:
        model_name: Name of the model (e.g., 'lstm', 'pinn_gbm', 'PINN GBM')
        config: Optional configuration object

    Returns:
        Dictionary containing training history with keys:
        - train_loss: List of training losses per epoch
        - val_loss: List of validation losses per epoch
        - train_data_loss: List of data losses (PINN only)
        - train_physics_loss: List of physics losses (PINN only)
        - learning_rates: List of learning rates per epoch
        - epochs: List of epoch numbers

        Returns None if no history file is found.

    Example:
        >>> history = load_training_history('pinn_gbm')
        >>> print(f"Trained for {len(history['epochs'])} epochs")
        >>> print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    """
    if config is None:
        config = get_config()

    # Normalize model name for file matching
    normalized_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

    # Try multiple naming patterns
    patterns = [
        f"{model_name}_history.json",
        f"{normalized_name}_history.json",
        f"pinn_{normalized_name.replace('pinn_', '')}_history.json",
    ]

    for pattern in patterns:
        path = config.models_dir / pattern
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)

    return None


def load_model_results(
    model_name: str,
    config: Optional[AnalysisConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Load evaluation results for a specific model.

    Args:
        model_name: Name of the model
        config: Optional configuration object

    Returns:
        Dictionary containing model results with keys:
        - model_name: Display name of the model
        - n_samples: Number of samples evaluated
        - ml_metrics: Dict with MSE, MAE, RMSE, R², MAPE
        - financial_metrics: Dict with Sharpe, Sortino, drawdown, etc.
        - rolling_metrics: Dict with stability analysis

        Returns None if no results file is found.

    Example:
        >>> results = load_model_results('pinn_gbm')
        >>> print(f"Sharpe Ratio: {results['financial_metrics']['sharpe_ratio']:.2f}")
    """
    if config is None:
        config = get_config()

    normalized_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

    # Try multiple naming patterns
    patterns = [
        f"{model_name}_results.json",
        f"{normalized_name}_results.json",
        f"pinn_{normalized_name.replace('pinn_', '')}_results.json",
        f"rigorous_{normalized_name}_results.json",
        f"rigorous_pinn_{normalized_name.replace('pinn_', '')}_results.json",
    ]

    for pattern in patterns:
        path = config.results_dir / pattern
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)

    return None


def load_all_models(
    config: Optional[AnalysisConfig] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load all available model results from the results directory.

    Scans the results directory for all *_results.json files and loads them.
    Excludes summary files that don't contain model-specific results.

    Args:
        config: Optional configuration object

    Returns:
        Dictionary mapping model names to their results dictionaries.
        Empty dict if no results found.

    Example:
        >>> models = load_all_models()
        >>> print(f"Loaded {len(models)} models")
        >>> for name, data in models.items():
        ...     sharpe = data['financial_metrics']['sharpe_ratio']
        ...     print(f"{name}: Sharpe={sharpe:.2f}")
    """
    if config is None:
        config = get_config()

    models = {}

    if not config.results_dir.exists():
        return models

    # Exclude summary/aggregate files
    exclude_patterns = ['summary', 'comparison', 'aggregate', 'physics_equation']

    for file in config.results_dir.glob("*_results.json"):
        # Skip excluded files
        if any(excl in file.stem.lower() for excl in exclude_patterns):
            continue

        try:
            with open(file, 'r') as f:
                data = json.load(f)

            # Get model name from data or filename
            model_name = data.get('model_name', file.stem.replace('_results', ''))

            # Ensure required keys exist
            if 'financial_metrics' in data or 'ml_metrics' in data:
                models[model_name] = data

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {file}: {e}")
            continue

    return models


def load_predictions(
    model_name: str,
    config: Optional[AnalysisConfig] = None
) -> Optional[pd.DataFrame]:
    """
    Load prediction data for a specific model.

    Supports both CSV and NPZ formats. For NPZ, converts to DataFrame.

    Args:
        model_name: Name of the model
        config: Optional configuration object

    Returns:
        DataFrame with columns:
        - actual: Actual/true values
        - predicted: Model predictions
        - residual: Prediction errors (predicted - actual)
        - timestamp: Optional time index

        Returns None if no predictions found.

    Example:
        >>> preds = load_predictions('pinn_gbm')
        >>> if preds is not None:
        ...     corr = preds['actual'].corr(preds['predicted'])
        ...     print(f"Correlation: {corr:.4f}")
    """
    if config is None:
        config = get_config()

    normalized_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

    # Try CSV first
    csv_patterns = [
        f"{model_name}_predictions.csv",
        f"{normalized_name}_predictions.csv",
    ]

    for pattern in csv_patterns:
        path = config.results_dir / pattern
        if path.exists():
            df = pd.read_csv(path)
            # Ensure standard column names
            if 'residual' not in df.columns and 'actual' in df.columns and 'predicted' in df.columns:
                df['residual'] = df['predicted'] - df['actual']
            return df

    # Try NPZ format
    npz_patterns = [
        f"{model_name}_predictions.npz",
        f"{normalized_name}_predictions.npz",
    ]

    for pattern in npz_patterns:
        path = config.results_dir / pattern
        if path.exists():
            data = np.load(path)
            predictions = data.get('predictions', data.get('preds', None))
            targets = data.get('targets', data.get('actual', data.get('y', None)))

            if predictions is not None and targets is not None:
                # Flatten if needed
                predictions = np.array(predictions).flatten()
                targets = np.array(targets).flatten()

                df = pd.DataFrame({
                    'actual': targets,
                    'predicted': predictions,
                    'residual': predictions - targets
                })
                return df

    return None


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_rolling_returns(
    returns: np.ndarray,
    window: int = 21
) -> np.ndarray:
    """
    Compute rolling cumulative returns over a window.

    Args:
        returns: Array of period returns
        window: Rolling window size (default 21 days = ~1 month)

    Returns:
        Array of rolling cumulative returns

    Example:
        >>> returns = np.random.randn(100) * 0.02
        >>> rolling = compute_rolling_returns(returns, window=21)
    """
    series = pd.Series(returns)
    rolling = (1 + series).rolling(window=window).apply(
        lambda x: np.prod(x) - 1, raw=True
    )
    return rolling.values


def compute_rolling_sharpe(
    returns: np.ndarray,
    window: int = 21,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> np.ndarray:
    """
    Compute rolling Sharpe ratio.

    Args:
        returns: Array of period returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        annualization_factor: Trading days per year

    Returns:
        Array of rolling Sharpe ratios (clipped to [-5, 5])

    Example:
        >>> returns = np.random.randn(100) * 0.02
        >>> sharpe = compute_rolling_sharpe(returns)
    """
    series = pd.Series(returns)
    daily_rf = risk_free_rate / annualization_factor

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Annualized Sharpe
    sharpe = ((rolling_mean - daily_rf) / rolling_std) * np.sqrt(annualization_factor)

    # Clip extreme values
    sharpe = np.clip(sharpe.values, -5, 5)

    return sharpe


def compute_rolling_sortino(
    returns: np.ndarray,
    window: int = 21,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> np.ndarray:
    """
    Compute rolling Sortino ratio (uses downside deviation only).

    Args:
        returns: Array of period returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        annualization_factor: Trading days per year

    Returns:
        Array of rolling Sortino ratios (clipped to [-10, 10])
    """
    series = pd.Series(returns)
    daily_rf = risk_free_rate / annualization_factor

    def sortino_window(x):
        excess = x - daily_rf
        downside = x[x < 0]
        if len(downside) < 2:
            return np.nan
        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return np.nan
        return (np.mean(excess) / downside_std) * np.sqrt(annualization_factor)

    sortino = series.rolling(window=window).apply(sortino_window, raw=True)

    return np.clip(sortino.values, -10, 10)


def compute_drawdown_series(
    returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute drawdown series from returns.

    Args:
        returns: Array of period returns

    Returns:
        Tuple of:
        - cumulative: Cumulative wealth (starting at 1)
        - drawdown: Drawdown series (0 to -1)
        - underwater: Days since last peak

    Example:
        >>> returns = np.random.randn(252) * 0.02
        >>> cumulative, drawdown, underwater = compute_drawdown_series(returns)
        >>> max_dd = np.min(drawdown)
        >>> print(f"Max Drawdown: {max_dd*100:.2f}%")
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    # Compute underwater periods
    underwater = np.zeros_like(drawdown)
    days_underwater = 0
    for i in range(len(drawdown)):
        if drawdown[i] < 0:
            days_underwater += 1
        else:
            days_underwater = 0
        underwater[i] = days_underwater

    return cumulative, drawdown, underwater


def compute_var_cvar(
    returns: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Value at Risk and Conditional VaR (Expected Shortfall).

    Args:
        returns: Array of returns
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (VaR, CVaR) as percentages

    Example:
        >>> returns = np.random.randn(1000) * 0.02
        >>> var, cvar = compute_var_cvar(returns, confidence=0.95)
        >>> print(f"VaR(95%): {var:.2%}, CVaR: {cvar:.2%}")
    """
    alpha = 1 - confidence
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var

    return var, cvar


def compute_rolling_volatility(
    returns: np.ndarray,
    window: int = 21,
    annualize: bool = True,
    annualization_factor: int = 252
) -> np.ndarray:
    """
    Compute rolling volatility.

    Args:
        returns: Array of period returns
        window: Rolling window size
        annualize: Whether to annualize the volatility
        annualization_factor: Trading days per year

    Returns:
        Array of rolling volatility values
    """
    series = pd.Series(returns)
    rolling_vol = series.rolling(window=window).std()

    if annualize:
        rolling_vol = rolling_vol * np.sqrt(annualization_factor)

    return rolling_vol.values


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    power: int = 2
) -> Tuple[float, float]:
    """
    Perform the Diebold-Mariano test for comparing forecast accuracy.

    Tests whether two forecasts have significantly different accuracy.

    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        h: Forecast horizon (default 1)
        power: Power for loss function (2 = MSE, 1 = MAE)

    Returns:
        Tuple of (DM statistic, p-value)

    Interpretation:
        - DM > 0: Model 1 is worse than Model 2
        - DM < 0: Model 1 is better than Model 2
        - p < 0.05: Difference is statistically significant

    Example:
        >>> e1 = np.random.randn(100)
        >>> e2 = np.random.randn(100) * 1.2  # Worse model
        >>> dm_stat, p_val = diebold_mariano_test(e1, e2)
        >>> if p_val < 0.05:
        ...     print("Models are significantly different")
    """
    d = np.abs(errors1) ** power - np.abs(errors2) ** power

    n = len(d)
    d_mean = np.mean(d)

    # Compute autocovariance
    gamma = np.zeros(h)
    for k in range(h):
        gamma[k] = np.cov(d[:-k-1] if k > 0 else d, d[k:] if k > 0 else d)[0, 1]

    # Long-run variance
    v = gamma[0] + 2 * np.sum(gamma[1:])

    # DM statistic
    dm_stat = d_mean / np.sqrt(v / n) if v > 0 else 0

    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Input data array
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> data = np.random.randn(100)
        >>> mean, lower, upper = bootstrap_confidence_interval(data, np.mean)
        >>> print(f"Mean: {mean:.3f} ({lower:.3f}, {upper:.3f})")
    """
    if seed is not None:
        np.random.seed(seed)

    point_estimate = statistic_func(data)

    boot_estimates = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_estimates.append(statistic_func(sample))

    boot_estimates = np.array(boot_estimates)
    alpha = 1 - confidence
    lower = np.percentile(boot_estimates, alpha / 2 * 100)
    upper = np.percentile(boot_estimates, (1 - alpha / 2) * 100)

    return point_estimate, lower, upper


# ============================================================================
# REGIME ANALYSIS FUNCTIONS
# ============================================================================

def classify_market_regime(
    returns: np.ndarray,
    volatility_threshold: float = 0.02,
    trend_threshold: float = 0.001
) -> np.ndarray:
    """
    Classify market regime based on returns and volatility.

    Regimes:
    - 0: Bear market (negative trend)
    - 1: Sideways (low trend)
    - 2: Bull market (positive trend)
    - 3: High volatility

    Args:
        returns: Array of period returns
        volatility_threshold: Threshold for high volatility regime
        trend_threshold: Threshold for trend detection

    Returns:
        Array of regime labels

    Example:
        >>> returns = np.random.randn(252) * 0.02
        >>> regimes = classify_market_regime(returns)
        >>> regime_counts = np.bincount(regimes, minlength=4)
    """
    series = pd.Series(returns)

    # Rolling statistics
    rolling_mean = series.rolling(window=21).mean()
    rolling_vol = series.rolling(window=21).std()

    regimes = np.zeros(len(returns), dtype=int)

    for i in range(len(returns)):
        if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_vol.iloc[i]):
            regimes[i] = 1  # Default to sideways
            continue

        vol = rolling_vol.iloc[i]
        trend = rolling_mean.iloc[i]

        if vol > volatility_threshold:
            regimes[i] = 3  # High volatility
        elif trend > trend_threshold:
            regimes[i] = 2  # Bull
        elif trend < -trend_threshold:
            regimes[i] = 0  # Bear
        else:
            regimes[i] = 1  # Sideways

    return regimes


def compute_regime_performance(
    returns: np.ndarray,
    regimes: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute performance metrics for each market regime.

    Args:
        returns: Array of period returns
        regimes: Array of regime labels

    Returns:
        Dictionary mapping regime names to performance metrics

    Example:
        >>> regimes = classify_market_regime(returns)
        >>> perf = compute_regime_performance(returns, regimes)
        >>> print(f"Bull Sharpe: {perf['Bull']['sharpe']:.2f}")
    """
    regime_names = ['Bear', 'Sideways', 'Bull', 'High Volatility']
    results = {}

    for regime_id, regime_name in enumerate(regime_names):
        mask = regimes == regime_id
        regime_returns = returns[mask]

        if len(regime_returns) < 5:
            results[regime_name] = {
                'count': len(regime_returns),
                'mean_return': np.nan,
                'sharpe': np.nan,
                'volatility': np.nan,
                'max_drawdown': np.nan
            }
            continue

        # Compute metrics
        mean_ret = np.mean(regime_returns) * 252  # Annualized
        vol = np.std(regime_returns) * np.sqrt(252)
        sharpe = mean_ret / vol if vol > 0 else 0

        # Drawdown
        _, dd, _ = compute_drawdown_series(regime_returns)
        max_dd = np.min(dd) if len(dd) > 0 else 0

        results[regime_name] = {
            'count': len(regime_returns),
            'mean_return': mean_ret,
            'sharpe': sharpe,
            'volatility': vol,
            'max_drawdown': max_dd
        }

    return results


# ============================================================================
# PREDICTION GENERATION UTILITIES
# ============================================================================

def generate_predictions_from_model(
    model,
    data_loader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from a trained model.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        device: Device to run inference on

    Returns:
        Tuple of (predictions, targets) as numpy arrays

    Example:
        >>> preds, targets = generate_predictions_from_model(model, test_loader)
        >>> mse = np.mean((preds - targets) ** 2)
    """
    import torch

    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]

            x = x.to(device)

            # Handle different model output formats
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]  # Some models return (pred, physics_loss)

            all_preds.append(output.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    return predictions, targets


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    config: Optional[AnalysisConfig] = None
) -> Path:
    """
    Save predictions to file for later analysis.

    Saves both NPZ (for fast loading) and CSV (for external tools).

    Args:
        predictions: Model predictions
        targets: Ground truth values
        model_name: Name of the model
        config: Optional configuration object

    Returns:
        Path to the saved CSV file

    Example:
        >>> path = save_predictions(preds, targets, 'pinn_gbm')
        >>> print(f"Saved to {path}")
    """
    if config is None:
        config = get_config()

    # Create results directory if needed
    config.results_dir.mkdir(parents=True, exist_ok=True)

    normalized_name = model_name.lower().replace(' ', '_')

    # Save NPZ
    npz_path = config.results_dir / f"{normalized_name}_predictions.npz"
    np.savez(npz_path, predictions=predictions, targets=targets)

    # Save CSV
    csv_path = config.results_dir / f"{normalized_name}_predictions.csv"
    df = pd.DataFrame({
        'actual': targets,
        'predicted': predictions,
        'residual': predictions - targets
    })
    df.to_csv(csv_path, index=False)

    print(f"Saved predictions to {csv_path}")
    return csv_path


# ============================================================================
# SYNTHETIC DATA GENERATION (for demo/testing)
# ============================================================================

def generate_synthetic_returns(
    n_samples: int = 1000,
    drift: float = 0.0002,
    volatility: float = 0.02,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic returns for testing.

    Uses GBM-like process with optional fat tails.

    Args:
        n_samples: Number of return samples
        drift: Daily drift (mean return)
        volatility: Daily volatility
        seed: Random seed

    Returns:
        Array of synthetic returns
    """
    if seed is not None:
        np.random.seed(seed)

    # Add some fat tails via t-distribution
    returns = t_dist.rvs(df=5, loc=drift, scale=volatility, size=n_samples)

    return returns


def generate_synthetic_predictions(
    n_samples: int = 1000,
    correlation: float = 0.7,
    noise_scale: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic prediction-actual pairs for testing.

    Args:
        n_samples: Number of samples
        correlation: Target correlation between predictions and actuals
        noise_scale: Scale of prediction noise
        seed: Random seed

    Returns:
        Tuple of (predictions, actuals)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate actuals
    actuals = np.random.randn(n_samples) * 0.02

    # Generate correlated predictions
    noise = np.random.randn(n_samples) * noise_scale
    predictions = correlation * actuals + np.sqrt(1 - correlation**2) * noise * np.std(actuals)

    return predictions, actuals


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_model_summary(
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract key summary statistics from model results.

    Args:
        results: Model results dictionary

    Returns:
        Dictionary with key metrics for quick comparison

    Example:
        >>> results = load_model_results('pinn_gbm')
        >>> summary = compute_model_summary(results)
        >>> print(f"Model: {summary['name']}, Sharpe: {summary['sharpe']:.2f}")
    """
    fm = results.get('financial_metrics', {})
    mm = results.get('ml_metrics', {})
    rm = results.get('rolling_metrics', {}).get('stability', {})

    return {
        'name': results.get('model_name', 'Unknown'),
        'n_samples': results.get('n_samples', 0),

        # ML metrics
        'mse': mm.get('mse', np.nan),
        'rmse': mm.get('rmse', np.nan),
        'mae': mm.get('mae', np.nan),
        'r2': mm.get('r2', np.nan),
        'mape': mm.get('mape', np.nan),

        # Financial metrics
        'sharpe': fm.get('sharpe_ratio', np.nan),
        'sortino': fm.get('sortino_ratio', np.nan),
        'max_drawdown': fm.get('max_drawdown', np.nan),
        'calmar': fm.get('calmar_ratio', np.nan),
        'total_return': fm.get('total_return', np.nan),
        'annualized_return': fm.get('annualized_return', np.nan),
        'volatility': fm.get('volatility', np.nan),
        'directional_accuracy': fm.get('directional_accuracy', np.nan),
        'win_rate': fm.get('win_rate', np.nan),
        'profit_factor': fm.get('profit_factor', np.nan),

        # Stability metrics
        'sharpe_stability': rm.get('sharpe_ratio_consistency', np.nan),
        'return_stability': rm.get('total_return_consistency', np.nan),
    }


def create_comparison_dataframe(
    models_data: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a comparison DataFrame from multiple model results.

    Args:
        models_data: Dictionary mapping model names to results

    Returns:
        DataFrame with one row per model and columns for each metric

    Example:
        >>> models = load_all_models()
        >>> df = create_comparison_dataframe(models)
        >>> best_sharpe = df.loc[df['sharpe'].idxmax()]
    """
    summaries = []

    for name, data in models_data.items():
        summary = compute_model_summary(data)
        summaries.append(summary)

    df = pd.DataFrame(summaries)

    # Sort by Sharpe ratio descending
    df = df.sort_values('sharpe', ascending=False)

    return df


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_results_structure(results: Dict[str, Any]) -> List[str]:
    """
    Validate that results dictionary has expected structure.

    Args:
        results: Model results dictionary

    Returns:
        List of warning messages (empty if valid)
    """
    warnings_list = []

    required_keys = ['model_name', 'ml_metrics', 'financial_metrics']
    for key in required_keys:
        if key not in results:
            warnings_list.append(f"Missing required key: {key}")

    ml_keys = ['mse', 'rmse', 'mae', 'r2']
    if 'ml_metrics' in results:
        for key in ml_keys:
            if key not in results['ml_metrics']:
                warnings_list.append(f"Missing ML metric: {key}")

    fin_keys = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown']
    if 'financial_metrics' in results:
        for key in fin_keys:
            if key not in results['financial_metrics']:
                warnings_list.append(f"Missing financial metric: {key}")

    return warnings_list


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    """Test the analysis utilities."""
    print("=" * 60)
    print("Testing Analysis Utilities")
    print("=" * 60)

    # Test configuration
    config = get_config()
    print(f"\nProject root: {config.project_root}")
    print(f"Results dir: {config.results_dir}")
    print(f"Models dir: {config.models_dir}")

    # Test loading all models
    print("\n" + "-" * 40)
    print("Loading all models...")
    models = load_all_models(config)
    print(f"Found {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")

    # Test loading specific model
    if models:
        test_model = list(models.keys())[0]
        print(f"\n" + "-" * 40)
        print(f"Testing with model: {test_model}")

        # Load history
        history = load_training_history(test_model, config)
        if history:
            print(f"  Training history: {len(history.get('epochs', []))} epochs")
        else:
            print("  No training history found")

        # Load predictions
        preds = load_predictions(test_model, config)
        if preds is not None:
            print(f"  Predictions: {len(preds)} samples")
            print(f"  Columns: {list(preds.columns)}")
        else:
            print("  No predictions found")

        # Compute summary
        summary = compute_model_summary(models[test_model])
        print(f"\n  Summary:")
        print(f"    Sharpe: {summary['sharpe']:.3f}")
        print(f"    Max DD: {summary['max_drawdown']:.3%}")
        print(f"    RMSE: {summary['rmse']:.6f}")

    # Test synthetic data generation
    print("\n" + "-" * 40)
    print("Testing synthetic data generation...")

    returns = generate_synthetic_returns(1000, seed=42)
    print(f"  Generated {len(returns)} synthetic returns")
    print(f"  Mean: {np.mean(returns):.6f}")
    print(f"  Std: {np.std(returns):.6f}")

    # Test metric computations
    print("\n" + "-" * 40)
    print("Testing metric computations...")

    rolling_sharpe = compute_rolling_sharpe(returns)
    print(f"  Rolling Sharpe (mean): {np.nanmean(rolling_sharpe):.3f}")

    cum, dd, _ = compute_drawdown_series(returns)
    print(f"  Max Drawdown: {np.min(dd):.3%}")

    var, cvar = compute_var_cvar(returns)
    print(f"  VaR(95%): {var:.3%}")
    print(f"  CVaR(95%): {cvar:.3%}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
