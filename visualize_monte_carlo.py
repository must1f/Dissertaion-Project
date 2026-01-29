#!/usr/bin/env python3
"""
Monte Carlo Simulation Visualization (Terminal)

Displays Monte Carlo simulation results with ASCII charts and tables.
Run this script to see confidence intervals, VaR, and stress test results.

Usage:
    python visualize_monte_carlo.py --model-path models/pinn_global_best.pt --ticker AAPL
    python visualize_monte_carlo.py --synthetic  # Use synthetic data for demo
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults,
    compute_var_cvar
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_header(title: str, char: str = "="):
    """Print a formatted header"""
    width = 80
    print()
    print(char * width)
    print(f" {title}".center(width))
    print(char * width)


def print_section(title: str):
    """Print a section header"""
    print()
    print(f"--- {title} ---")


def format_number(value: float, decimals: int = 4) -> str:
    """Format number with color coding for terminal"""
    if value > 0:
        return f"\033[92m{value:+.{decimals}f}\033[0m"  # Green
    elif value < 0:
        return f"\033[91m{value:+.{decimals}f}\033[0m"  # Red
    else:
        return f"{value:.{decimals}f}"


def ascii_line_chart(values: np.ndarray, height: int = 10, width: int = 60, title: str = "") -> str:
    """Generate ASCII line chart"""
    if len(values) == 0:
        return "No data"

    # Normalize to height
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val:
        max_val = min_val + 1

    normalized = ((values - min_val) / (max_val - min_val) * (height - 1)).astype(int)

    # Create chart grid
    chart = []
    for row in range(height - 1, -1, -1):
        line = ""
        for col in range(min(len(values), width)):
            idx = int(col * len(values) / width)
            if normalized[idx] == row:
                line += "*"
            elif normalized[idx] > row:
                line += "|"
            else:
                line += " "

        # Add y-axis label
        y_val = min_val + (max_val - min_val) * row / (height - 1)
        chart.append(f"{y_val:8.4f} |{line}")

    # X-axis
    chart.append(" " * 9 + "+" + "-" * width)
    chart.append(" " * 9 + "0" + " " * (width // 2 - 1) + str(len(values) // 2) + " " * (width // 2 - len(str(len(values)))) + str(len(values)))

    if title:
        chart.insert(0, f"  {title}")

    return "\n".join(chart)


def ascii_confidence_band(
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    height: int = 12,
    width: int = 60
) -> str:
    """Generate ASCII chart with confidence bands"""
    all_values = np.concatenate([mean, lower, upper])
    min_val, max_val = np.min(all_values), np.max(all_values)
    if max_val == min_val:
        max_val = min_val + 1

    def normalize(v):
        return int((v - min_val) / (max_val - min_val) * (height - 1))

    chart = []
    for row in range(height - 1, -1, -1):
        line = ""
        for col in range(min(len(mean), width)):
            idx = int(col * len(mean) / width)
            mean_row = normalize(mean[idx])
            lower_row = normalize(lower[idx])
            upper_row = normalize(upper[idx])

            if row == mean_row:
                line += "\033[94m*\033[0m"  # Blue for mean
            elif lower_row <= row <= upper_row:
                line += "\033[90m.\033[0m"  # Gray for CI band
            else:
                line += " "

        y_val = min_val + (max_val - min_val) * row / (height - 1)
        chart.append(f"{y_val:8.4f} |{line}")

    chart.append(" " * 9 + "+" + "-" * width)
    chart.append(" " * 9 + "Day 0" + " " * (width - 15) + f"Day {len(mean)}")

    return "\n".join(chart)


def print_monte_carlo_results(results: MonteCarloResults, title: str = "Monte Carlo Simulation"):
    """Print Monte Carlo results in terminal format"""
    print_header(title)

    print(f"\nSimulation Parameters:")
    print(f"  Number of simulations: {results.n_simulations:,}")
    print(f"  Forecast horizon: {results.horizon} days")
    print(f"  Confidence level: {results.confidence_level * 100:.0f}%")

    print_section("Forecast Statistics")

    # Statistics table
    print("\n  Day     Mean      Median    Lower CI    Upper CI    VaR(5%)    CVaR(5%)")
    print("  " + "-" * 75)

    # Show first, middle, and last days
    days_to_show = [0, results.horizon // 4, results.horizon // 2, 3 * results.horizon // 4, results.horizon - 1]
    days_to_show = sorted(set(d for d in days_to_show if d < results.horizon))

    for day in days_to_show:
        print(f"  {day+1:3d}    {results.mean_path[day]:8.4f}  {results.median_path[day]:8.4f}  "
              f"{results.lower_ci[day]:10.4f}  {results.upper_ci[day]:8.4f}  "
              f"{results.var_5[day]:9.4f}  {results.cvar_5[day]:9.4f}")

    print_section(f"Forecast Chart with {results.confidence_level*100:.0f}% Confidence Interval")
    print("\n  Legend: \033[94m*\033[0m = Mean path, \033[90m.\033[0m = Confidence band")
    print()
    print(ascii_confidence_band(results.mean_path, results.lower_ci, results.upper_ci))

    # Final statistics
    print_section("Summary Statistics at Horizon")

    final_mean = results.mean_path[-1]
    final_lower = results.lower_ci[-1]
    final_upper = results.upper_ci[-1]

    print(f"\n  Expected value:     {final_mean:.4f}")
    print(f"  95% CI:             [{final_lower:.4f}, {final_upper:.4f}]")
    print(f"  CI Width:           {final_upper - final_lower:.4f}")
    print(f"  5% VaR:             {results.var_5[-1]:.4f}")
    print(f"  5% CVaR (ES):       {results.cvar_5[-1]:.4f}")

    # Distribution at final day
    final_returns = (results.paths[:, -1] - results.paths[:, 0]) / (results.paths[:, 0] + 1e-8)

    print_section("Return Distribution at Horizon")
    print(f"\n  Mean return:        {np.mean(final_returns)*100:+.2f}%")
    print(f"  Std deviation:      {np.std(final_returns)*100:.2f}%")
    print(f"  Min return:         {np.min(final_returns)*100:+.2f}%")
    print(f"  Max return:         {np.max(final_returns)*100:+.2f}%")
    print(f"  % positive:         {np.mean(final_returns > 0)*100:.1f}%")


def print_stress_test_results(stress_results: dict):
    """Print stress test results"""
    print_header("Stress Test Results")

    print("\n  Scenario              Final Mean    Final Lower    Final Upper    Change vs Base")
    print("  " + "-" * 80)

    base_mean = stress_results.get('base', stress_results[list(stress_results.keys())[0]]).mean_path[-1]

    for name, results in stress_results.items():
        final_mean = results.mean_path[-1]
        final_lower = results.lower_ci[-1]
        final_upper = results.upper_ci[-1]
        change = (final_mean - base_mean) / base_mean * 100 if base_mean != 0 else 0

        change_str = format_number(change, 2) + "%"
        print(f"  {name:20s}  {final_mean:10.4f}    {final_lower:10.4f}     {final_upper:10.4f}    {change_str}")

    print_section("Worst Case Scenario (5% VaR)")

    for name, results in stress_results.items():
        var_5 = results.var_5[-1]
        print(f"  {name:20s}: VaR = {var_5:.4f}")


def generate_synthetic_model():
    """Generate a simple synthetic model for demonstration"""
    import torch.nn as nn

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = SimpleLSTM()
    return model


def generate_synthetic_data(n_samples: int = 100, n_features: int = 10):
    """Generate synthetic price data"""
    np.random.seed(42)

    # Generate random walk price series
    returns = np.random.normal(0.0005, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create feature matrix
    features = np.zeros((n_samples, n_features))
    features[:, 0] = prices  # Price as first feature

    # Add some technical indicators as features
    for i in range(1, n_features):
        features[:, i] = np.random.randn(n_samples) * 0.1

    return features, prices


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Visualization")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--ticker", type=str, default="SYNTHETIC", help="Ticker symbol")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--n-simulations", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for demo")
    parser.add_argument("--stress-test", action="store_true", help="Run stress tests")

    args = parser.parse_args()

    print_header("MONTE CARLO SIMULATION VISUALIZER", "=")
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Simulations: {args.n_simulations:,}")

    # Load or generate model
    if args.synthetic or args.model_path is None:
        print("\n  Using synthetic model for demonstration...")
        model = generate_synthetic_model()
        features, prices = generate_synthetic_data()
    else:
        print(f"\n  Loading model from: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            # This would need to be adapted based on actual model architecture
            from src.models.pinn import PINNModel
            model = PINNModel(input_dim=10)  # Adjust based on actual config
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

            # Load real data
            from src.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            data = fetcher.fetch_yahoo_finance([args.ticker], period='1y')
            prices = data['close'].values
            features = np.random.randn(len(prices), 10)  # Would need proper features
        except Exception as e:
            print(f"  Error loading model: {e}")
            print("  Falling back to synthetic model...")
            model = generate_synthetic_model()
            features, prices = generate_synthetic_data()

    # Set model to eval mode
    model.eval()

    # Create simulator
    simulator = MonteCarloSimulator(
        model=model,
        n_simulations=args.n_simulations,
        seed=42
    )

    # Use last sequence as initial data
    seq_len = min(60, len(features))
    initial_data = features[-seq_len:]

    # Estimate volatility from prices
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    print(f"  Estimated annual volatility: {volatility*100:.1f}%")

    # Run Monte Carlo simulation
    print("\n  Running Monte Carlo simulation...")
    results = simulator.simulate_paths(
        initial_data=initial_data,
        horizon=args.horizon,
        volatility=volatility
    )

    # Print results
    print_monte_carlo_results(results, f"Monte Carlo Simulation: {args.ticker}")

    # Stress testing
    if args.stress_test:
        print("\n  Running stress tests...")
        stress_results = simulator.stress_test(
            initial_data=initial_data,
            horizon=args.horizon
        )
        print_stress_test_results(stress_results)

    # Compute VaR/CVaR on final returns
    final_returns = (results.paths[:, -1] - results.paths[:, 0]) / (results.paths[:, 0] + 1e-8)
    var_cvar = compute_var_cvar(final_returns, confidence_level=0.95)

    print_section("Risk Metrics")
    print(f"\n  95% Value at Risk:       {var_cvar['var_95']*100:+.2f}%")
    print(f"  95% Conditional VaR:     {var_cvar['cvar_95']*100:+.2f}%")

    print("\n" + "=" * 80)
    print(" Simulation Complete ".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
