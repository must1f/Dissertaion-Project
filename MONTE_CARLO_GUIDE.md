# Monte Carlo Simulation Guide

## Overview

This module provides Monte Carlo simulation capabilities for uncertainty quantification in financial forecasting models. It addresses the specification requirement for MC-based robustness validation.

## Features

- **Price Path Simulation**: Generate thousands of possible future price paths
- **Confidence Intervals**: 95% (configurable) prediction intervals
- **Risk Metrics**: Value at Risk (VaR) and Conditional VaR (Expected Shortfall)
- **Stress Testing**: Evaluate model under extreme market conditions
- **Bootstrap Confidence Intervals**: Statistical uncertainty for metrics

---

## Quick Start

### Terminal Visualization

Run the terminal-based visualization for quick analysis:

```bash
# Using synthetic data (demo mode)
python visualize_monte_carlo.py --synthetic --stress-test

# With specific parameters
python visualize_monte_carlo.py --synthetic --horizon 30 --n-simulations 1000

# With a trained model
python visualize_monte_carlo.py --model-path models/pinn_global_best.pt --ticker AAPL
```

**Terminal Output Example:**
```
================================================================================
                    MONTE CARLO SIMULATION VISUALIZER
================================================================================

  Date: 2026-01-29 14:30:00
  Ticker: SYNTHETIC
  Horizon: 30 days
  Simulations: 1,000

--- Forecast Statistics ---

  Day     Mean      Median    Lower CI    Upper CI    VaR(5%)    CVaR(5%)
  ---------------------------------------------------------------------------
    1    0.9985    0.9983      0.9702      1.0268     0.9523     0.9412
   15    0.9956    0.9951      0.9312      1.0600     0.9102     0.8956
   30    0.9923    0.9915      0.8945      1.0901     0.8734     0.8523

--- Summary Statistics at Horizon ---

  Expected value:     0.9923
  95% CI:             [0.8945, 1.0901]
  5% VaR:             0.8734
  5% CVaR (ES):       0.8523
```

### Web Dashboard

Launch the interactive Streamlit dashboard:

```bash
# Option 1: Using launch script
chmod +x launch_monte_carlo.sh
./launch_monte_carlo.sh

# Option 2: Direct streamlit command
streamlit run src/web/monte_carlo_dashboard.py --server.port 8503
```

Then open http://localhost:8503 in your browser.

---

## Using the Monte Carlo Module in Code

### Basic Usage

```python
from src.evaluation.monte_carlo import MonteCarloSimulator, compute_var_cvar
import numpy as np
import torch

# Load your trained model
model = load_model('models/pinn_global_best.pt')

# Create simulator
simulator = MonteCarloSimulator(
    model=model,
    n_simulations=1000,
    seed=42
)

# Prepare input data (last 60 days of features)
initial_data = features[-60:]  # Shape: [60, n_features]

# Run simulation
results = simulator.simulate_paths(
    initial_data=initial_data,
    horizon=30,
    volatility=0.20  # 20% annual volatility
)

# Access results
print(f"Mean forecast at day 30: {results.mean_path[-1]:.4f}")
print(f"95% CI: [{results.lower_ci[-1]:.4f}, {results.upper_ci[-1]:.4f}]")
print(f"5% VaR: {results.var_5[-1]:.4f}")
```

### Stress Testing

```python
# Run stress test with multiple scenarios
stress_results = simulator.stress_test(
    initial_data=initial_data,
    horizon=30,
    scenarios={
        'base': {'volatility_mult': 1.0, 'drift': 0.0},
        'high_vol': {'volatility_mult': 2.0, 'drift': 0.0},
        'crash': {'volatility_mult': 3.0, 'drift': -0.02},
        'bull': {'volatility_mult': 0.8, 'drift': 0.01}
    }
)

# Compare scenarios
for name, res in stress_results.items():
    print(f"{name}: Final Mean = {res.mean_path[-1]:.4f}, VaR = {res.var_5[-1]:.4f}")
```

### Bootstrap Confidence Intervals

```python
# Compute bootstrap CIs for model metrics
ci_results = simulator.compute_confidence_intervals(
    predictions=model_predictions,
    targets=actual_values,
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"MSE: {ci_results['mse_mean']:.6f} [{ci_results['mse_lower_ci']:.6f}, {ci_results['mse_upper_ci']:.6f}]")
print(f"Directional Accuracy: {ci_results['directional_accuracy_mean']:.2%}")
```

### Computing VaR/CVaR

```python
from src.evaluation.monte_carlo import compute_var_cvar

# Compute risk metrics on returns
strategy_returns = np.array([0.02, -0.01, 0.03, -0.05, ...])

var_cvar = compute_var_cvar(strategy_returns, confidence_level=0.95)
print(f"95% VaR: {var_cvar['var_95']*100:.2f}%")
print(f"95% CVaR: {var_cvar['cvar_95']*100:.2f}%")
```

---

## Understanding the Results

### MonteCarloResults Fields

| Field | Description |
|-------|-------------|
| `paths` | All simulated paths [n_simulations, horizon] |
| `mean_path` | Average across all simulations |
| `median_path` | Median across all simulations |
| `lower_ci` | Lower confidence interval bound |
| `upper_ci` | Upper confidence interval bound |
| `var_5` | 5% Value at Risk (5th percentile) |
| `cvar_5` | 5% Conditional VaR (mean of values below VaR) |

### Risk Metrics Interpretation

**Value at Risk (VaR)**
- "There is a 5% probability of the value being below this threshold"
- Example: VaR of -10% means 95% of the time, losses won't exceed 10%

**Conditional VaR (Expected Shortfall)**
- "If we do breach VaR, what's the average loss?"
- Always worse (lower) than VaR
- Better captures tail risk

**Confidence Interval**
- "We're 95% confident the true value falls within this range"
- Wider CI = more uncertainty

---

## Stress Test Scenarios

| Scenario | Volatility | Drift | Description |
|----------|------------|-------|-------------|
| Base | 1.0x | 0% | Normal market conditions |
| High Volatility | 2.0x | 0% | Elevated market fear/uncertainty |
| Market Crash | 3.0x | -2% daily | Severe downturn with panic |
| Bull Market | 0.8x | +1% daily | Low volatility rally |
| Black Swan | 5.0x | -5% daily | Extreme tail event |

---

## Integration with PINN Models

The Monte Carlo module integrates with all PINN variants:

```python
from src.models.pinn import PINNModel
from src.evaluation.monte_carlo import MonteCarloSimulator

# Load PINN model
model = PINNModel(input_dim=10, hidden_dim=128)
checkpoint = torch.load('models/pinn_global_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Run Monte Carlo
simulator = MonteCarloSimulator(model=model, n_simulations=1000)
results = simulator.simulate_paths(initial_data, horizon=30)
```

---

## Dashboard Features

### Confidence Bands Tab
- Visualizes mean forecast with shaded confidence region
- Shows VaR boundary (worst-case line)
- Hover for exact values at each day

### Path Distribution Tab
- Displays sample simulated paths
- Shows spread of possible outcomes
- Helps identify tail risk scenarios

### Final Distribution Tab
- Histogram of final values
- Key statistics (mean, VaR, probability of positive return)
- Vertical lines marking important thresholds

### Summary Metrics Tab
- Complete table of all statistics
- Download as CSV for reporting
- Risk-adjusted metrics

### Stress Test Section
- Side-by-side scenario comparison
- Box plots of final value distributions
- VaR comparison across scenarios

---

## Command Line Options

```
python visualize_monte_carlo.py [OPTIONS]

Options:
  --model-path PATH    Path to trained model (.pt file)
  --ticker SYMBOL      Ticker symbol for display
  --horizon N          Forecast horizon in days (default: 30)
  --n-simulations N    Number of Monte Carlo simulations (default: 1000)
  --synthetic          Use synthetic data for demo
  --stress-test        Include stress test scenarios
```

---

## File Locations

| File | Description |
|------|-------------|
| `src/evaluation/monte_carlo.py` | Core Monte Carlo module |
| `src/web/monte_carlo_dashboard.py` | Streamlit web dashboard |
| `visualize_monte_carlo.py` | Terminal visualization script |
| `launch_monte_carlo.sh` | Launch script for easy access |

---

## Best Practices

1. **Sufficient Simulations**: Use at least 1,000 simulations for stable estimates
2. **Volatility Estimation**: Use historical volatility matched to your data period
3. **Horizon Selection**: Longer horizons have wider confidence intervals
4. **Stress Testing**: Always run stress tests before deployment
5. **Validation**: Compare MC predictions against actual outcomes

---

## Troubleshooting

**Issue: Very wide confidence intervals**
- Check if volatility estimate is reasonable
- Longer horizons naturally have more uncertainty

**Issue: All paths trending same direction**
- Model may have persistent bias
- Check if predictions are realistic

**Issue: Stress test crashes**
- Extreme scenarios may produce NaN values
- Add bounds checking to model outputs
