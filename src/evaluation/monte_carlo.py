"""
Monte Carlo Simulation for Uncertainty Quantification

Implements:
- Price path simulation using model predictions
- Confidence interval generation
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Stress testing under extreme scenarios

This addresses the specification requirement for Monte Carlo-based robustness validation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""
    paths: np.ndarray  # [n_simulations, horizon]
    mean_path: np.ndarray  # [horizon]
    median_path: np.ndarray  # [horizon]
    lower_ci: np.ndarray  # [horizon] - Lower confidence interval
    upper_ci: np.ndarray  # [horizon] - Upper confidence interval
    var_5: np.ndarray  # [horizon] - 5% Value at Risk
    cvar_5: np.ndarray  # [horizon] - 5% Conditional VaR
    confidence_level: float
    n_simulations: int
    horizon: int


class MonteCarloSimulator:
    """
    Monte Carlo simulator for financial forecasting models

    Generates simulated price paths using model predictions with
    uncertainty quantification via bootstrap and stochastic noise.
    """

    def __init__(
        self,
        model: nn.Module,
        n_simulations: int = 1000,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        """
        Initialize Monte Carlo simulator

        Args:
            model: Trained neural network model
            n_simulations: Number of Monte Carlo simulations
            device: Torch device (cpu/cuda)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.n_simulations = n_simulations
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed

        # Set model to evaluation mode
        self.model.eval()

        logger.info(f"MonteCarloSimulator initialized with {n_simulations} simulations")

    def simulate_paths(
        self,
        initial_data: np.ndarray,
        horizon: int = 30,
        volatility: Optional[float] = None,
        return_type: str = 'price'
    ) -> MonteCarloResults:
        """
        Generate Monte Carlo simulated price/return paths

        Args:
            initial_data: Initial input data [seq_len, features] or [batch, seq_len, features]
            horizon: Number of steps to simulate forward
            volatility: Volatility for noise injection (estimated from data if None)
            return_type: 'price' for price paths, 'returns' for return paths

        Returns:
            MonteCarloResults with simulated paths and statistics
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Ensure proper shape
        if len(initial_data.shape) == 2:
            initial_data = initial_data[np.newaxis, :, :]  # Add batch dimension

        # Convert to tensor
        x = torch.FloatTensor(initial_data).to(self.device)

        # Estimate volatility from data if not provided
        if volatility is None:
            # Use returns column or estimate from prices
            if initial_data.shape[-1] > 1:
                # Assume first feature is price, compute returns
                prices = initial_data[0, :, 0]
                returns = np.diff(prices) / (prices[:-1] + 1e-8)
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                volatility = 0.2  # Default 20% annual volatility

        logger.info(f"Simulating {self.n_simulations} paths over {horizon} steps")
        logger.info(f"Using volatility: {volatility:.4f}")

        # Generate simulated paths
        all_paths = []

        with torch.no_grad():
            for sim in range(self.n_simulations):
                # Start with initial data
                current_x = x.clone()
                path = []

                for step in range(horizon):
                    # Get model prediction
                    pred = self.model(current_x)
                    if isinstance(pred, tuple):
                        pred = pred[0]  # Handle models that return multiple outputs

                    pred_value = pred.cpu().numpy().flatten()[-1]

                    # Add stochastic noise based on volatility
                    # Daily volatility = annual volatility / sqrt(252)
                    daily_vol = volatility / np.sqrt(252)
                    noise = np.random.normal(0, daily_vol)
                    noisy_pred = pred_value * (1 + noise)

                    path.append(noisy_pred)

                    # Update input sequence for next prediction (rolling window)
                    # Shift sequence and add new prediction
                    current_x = self._update_sequence(current_x, noisy_pred)

                all_paths.append(path)

        # Convert to numpy array
        paths = np.array(all_paths)  # [n_simulations, horizon]

        # Compute statistics
        results = self._compute_statistics(paths, confidence_level=0.95)

        return results

    def _update_sequence(self, x: torch.Tensor, new_value: float) -> torch.Tensor:
        """Update input sequence with new predicted value"""
        # Roll sequence forward
        x_new = x.clone()

        # Shift all timesteps back by 1
        x_new[:, :-1, :] = x[:, 1:, :]

        # Set last timestep's first feature to new value
        x_new[:, -1, 0] = new_value

        return x_new

    def _compute_statistics(
        self,
        paths: np.ndarray,
        confidence_level: float = 0.95
    ) -> MonteCarloResults:
        """Compute statistics from simulated paths"""
        alpha = 1 - confidence_level

        # Basic statistics
        mean_path = np.mean(paths, axis=0)
        median_path = np.median(paths, axis=0)

        # Confidence intervals
        lower_ci = np.percentile(paths, alpha / 2 * 100, axis=0)
        upper_ci = np.percentile(paths, (1 - alpha / 2) * 100, axis=0)

        # Value at Risk (5%)
        var_5 = np.percentile(paths, 5, axis=0)

        # Conditional VaR (Expected Shortfall at 5%)
        cvar_5 = np.zeros(paths.shape[1])
        for t in range(paths.shape[1]):
            threshold = var_5[t]
            below_var = paths[:, t][paths[:, t] <= threshold]
            cvar_5[t] = np.mean(below_var) if len(below_var) > 0 else threshold

        return MonteCarloResults(
            paths=paths,
            mean_path=mean_path,
            median_path=median_path,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            var_5=var_5,
            cvar_5=cvar_5,
            confidence_level=confidence_level,
            n_simulations=self.n_simulations,
            horizon=paths.shape[1]
        )

    def compute_confidence_intervals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Compute confidence intervals via bootstrap

        Args:
            predictions: Model predictions [n_samples]
            targets: Actual values [n_samples]
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dict with confidence intervals for various metrics
        """
        np.random.seed(self.seed)

        predictions = predictions.flatten()
        targets = targets.flatten()
        n_samples = len(predictions)

        # Bootstrap samples
        metrics_bootstrap = {
            'mse': [],
            'mae': [],
            'directional_accuracy': [],
            'correlation': []
        }

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_sample = predictions[indices]
            target_sample = targets[indices]

            # Compute metrics
            metrics_bootstrap['mse'].append(np.mean((pred_sample - target_sample) ** 2))
            metrics_bootstrap['mae'].append(np.mean(np.abs(pred_sample - target_sample)))

            # Directional accuracy
            pred_dir = np.sign(np.diff(pred_sample))
            target_dir = np.sign(np.diff(target_sample))
            dir_acc = np.mean(pred_dir == target_dir)
            metrics_bootstrap['directional_accuracy'].append(dir_acc)

            # Correlation
            if np.std(pred_sample) > 0 and np.std(target_sample) > 0:
                corr = np.corrcoef(pred_sample, target_sample)[0, 1]
            else:
                corr = 0
            metrics_bootstrap['correlation'].append(corr)

        # Compute confidence intervals
        alpha = 1 - confidence_level
        results = {}

        for metric_name, values in metrics_bootstrap.items():
            values = np.array(values)
            results[f'{metric_name}_mean'] = np.mean(values)
            results[f'{metric_name}_std'] = np.std(values)
            results[f'{metric_name}_lower_ci'] = np.percentile(values, alpha / 2 * 100)
            results[f'{metric_name}_upper_ci'] = np.percentile(values, (1 - alpha / 2) * 100)

        logger.info(f"Bootstrap CIs computed ({n_bootstrap} samples, {confidence_level*100:.0f}% confidence)")

        return results

    def stress_test(
        self,
        initial_data: np.ndarray,
        horizon: int = 30,
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, MonteCarloResults]:
        """
        Perform stress testing under various scenarios

        Args:
            initial_data: Initial input data
            horizon: Simulation horizon
            scenarios: Dict of scenario configs {name: {'volatility_mult': float, 'drift': float}}

        Returns:
            Dict of MonteCarloResults for each scenario
        """
        if scenarios is None:
            scenarios = {
                'base': {'volatility_mult': 1.0, 'drift': 0.0},
                'high_volatility': {'volatility_mult': 2.0, 'drift': 0.0},
                'market_crash': {'volatility_mult': 3.0, 'drift': -0.02},
                'bull_market': {'volatility_mult': 0.8, 'drift': 0.01},
                'black_swan': {'volatility_mult': 5.0, 'drift': -0.05}
            }

        logger.info(f"Running stress test with {len(scenarios)} scenarios")

        results = {}

        # Estimate base volatility
        if len(initial_data.shape) == 2:
            prices = initial_data[:, 0]
        else:
            prices = initial_data[0, :, 0]
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        base_volatility = np.std(returns) * np.sqrt(252)

        for scenario_name, config in scenarios.items():
            logger.info(f"  Scenario: {scenario_name}")

            vol_mult = config.get('volatility_mult', 1.0)
            drift = config.get('drift', 0.0)

            scenario_vol = base_volatility * vol_mult

            # Run simulation with modified volatility
            mc_results = self.simulate_paths(
                initial_data=initial_data,
                horizon=horizon,
                volatility=scenario_vol
            )

            # Apply drift adjustment
            if drift != 0:
                drift_factor = np.exp(np.arange(horizon) * drift / 252)
                mc_results.paths = mc_results.paths * drift_factor
                mc_results.mean_path = mc_results.mean_path * drift_factor
                mc_results.median_path = mc_results.median_path * drift_factor
                mc_results.lower_ci = mc_results.lower_ci * drift_factor
                mc_results.upper_ci = mc_results.upper_ci * drift_factor

            results[scenario_name] = mc_results

        return results


def compute_var_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute Value at Risk and Conditional VaR

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Dict with VaR and CVaR values
    """
    returns = returns.flatten()
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {'var': 0.0, 'cvar': 0.0}

    alpha = 1 - confidence_level

    # VaR: the worst expected loss at confidence level
    var = np.percentile(returns, alpha * 100)

    # CVaR (Expected Shortfall): average loss beyond VaR
    losses_beyond_var = returns[returns <= var]
    cvar = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var

    return {
        f'var_{int(confidence_level*100)}': float(var),
        f'cvar_{int(confidence_level*100)}': float(cvar)
    }


def monte_carlo_price_path(
    model: nn.Module,
    initial_sequence: np.ndarray,
    horizon: int = 30,
    n_paths: int = 1000,
    volatility: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for Monte Carlo price path simulation

    Args:
        model: Trained model
        initial_sequence: Initial input sequence
        horizon: Number of steps to simulate
        n_paths: Number of paths to simulate
        volatility: Annual volatility
        seed: Random seed

    Returns:
        Tuple of (paths, lower_ci_95, upper_ci_95)
    """
    simulator = MonteCarloSimulator(model, n_simulations=n_paths, seed=seed)
    results = simulator.simulate_paths(initial_sequence, horizon=horizon, volatility=volatility)

    return results.paths, results.lower_ci, results.upper_ci
