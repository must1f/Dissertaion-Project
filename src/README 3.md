# PINN Financial Forecasting - ML Core

Core machine learning library for Physics-Informed Neural Network financial forecasting.

## Installation

```bash
# Install as editable package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Directory Structure

```
src/
├── __init__.py
│
├── data/                    # Data handling
│   ├── __init__.py
│   ├── fetcher.py          # Yahoo Finance data fetching
│   ├── preprocessor.py     # Feature engineering
│   └── dataset.py          # PyTorch datasets
│
├── models/                  # Neural network models
│   ├── __init__.py
│   ├── pinn.py             # Physics-Informed Neural Network
│   ├── stacked_pinn.py     # Multi-equation PINN
│   ├── baseline.py         # LSTM, GRU, BiLSTM
│   ├── transformer.py      # Transformer model
│   ├── uncertainty.py      # MC Dropout uncertainty
│   └── model_registry.py   # Model loading/caching
│
├── training/                # Training pipelines
│   ├── __init__.py
│   ├── trainer.py          # Base trainer class
│   ├── train.py            # Training script
│   ├── train_pinn_variants.py
│   ├── train_stacked_pinn.py
│   ├── curriculum.py       # Curriculum learning
│   └── walk_forward.py     # Walk-forward validation
│
├── evaluation/              # Metrics & evaluation
│   ├── __init__.py
│   ├── metrics.py          # ML metrics (RMSE, MAE, etc.)
│   ├── financial_metrics.py # Financial metrics (Sharpe, etc.)
│   ├── backtester.py       # Backtesting engine
│   ├── backtesting_platform.py
│   ├── monte_carlo.py      # Monte Carlo simulation
│   ├── rolling_metrics.py  # Rolling window metrics
│   └── unified_evaluator.py
│
├── trading/                 # Trading strategies
│   ├── __init__.py
│   ├── agent.py            # Trading agent
│   └── position_sizing.py  # Kelly criterion, etc.
│
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── config.py           # Configuration
│   ├── database.py         # Database utilities
│   ├── logger.py           # Logging setup
│   └── reproducibility.py  # Random seed management
│
└── web/                     # Streamlit dashboards (legacy)
    ├── app.py
    ├── pinn_dashboard.py
    └── ...
```

## Quick Start

### Data Fetching

```python
from src.data.fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch("AAPL", period="1y")
```

### Model Training

```python
from src.models.pinn import PINN
from src.training.trainer import PINNTrainer

model = PINN(
    input_dim=20,
    hidden_dims=[128, 64, 32],
    physics_type="gbm_ou"
)

trainer = PINNTrainer(model, learning_rate=0.001)
trainer.train(train_loader, val_loader, epochs=100)
```

### Prediction

```python
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()
model = registry.get_model("pinn_gbm_ou")

predictions = model.predict(features)
```

### Backtesting

```python
from src.evaluation.backtester import Backtester

backtester = Backtester(
    model=model,
    initial_capital=100000
)
results = backtester.run(test_data)
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

### Monte Carlo Simulation

```python
from src.evaluation.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator(model)
results = simulator.simulate(
    initial_price=185.0,
    n_simulations=1000,
    horizon_days=30
)
print(f"95% VaR: ${results['var_95']:.2f}")
```

## PINN Physics Equations

### Geometric Brownian Motion (GBM)
```
dS = μS·dt + σS·dW
```

### Ornstein-Uhlenbeck (OU)
```
dX = θ(μ - X)dt + σdW
```

### Combined GBM + OU
```
dS = μS·dt + σS·dW + θ(γ - log(S))·dt
```

### Black-Scholes PDE
```
∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0
```

## Model Registry

The `ModelRegistry` provides centralized model management:

```python
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()

# List available models
models = registry.list_models()

# Load model with caching
model = registry.get_model("pinn_gbm_ou")

# Get model info
info = registry.get_model_info("pinn_gbm_ou")
```

## Configuration

```python
from src.utils.config import Config

config = Config()
config.SEQUENCE_LENGTH = 60
config.BATCH_SIZE = 32
config.LEARNING_RATE = 0.001
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Streamlit Dashboards (Legacy)

The `web/` directory contains Streamlit dashboards that have been superseded by the React frontend:

```bash
# Run main dashboard
streamlit run src/web/app.py

# Run PINN analysis
streamlit run src/web/pinn_dashboard.py
```

These are kept for reference and quick prototyping.
