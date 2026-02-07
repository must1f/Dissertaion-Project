# Physics-Informed Neural Network (PINN) for Financial Forecasting

## Complete Project Documentation

**Version:** 1.2
**Last Updated:** February 6, 2026
**Status:** Production-Ready Academic Research System

---

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS FOR ACADEMIC RESEARCH ONLY - NOT FINANCIAL ADVICE**

- This system is a dissertation research project
- NOT investment advice or recommendations
- Simulation only - no real trading functionality
- Past performance does not guarantee future results
- Always consult qualified financial advisors before making investment decisions
- The authors assume no liability for any financial losses

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Data Pipeline](#4-data-pipeline)
5. [Model Architectures](#5-model-architectures)
6. [Training Guide](#6-training-guide)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Backtesting Platform](#8-backtesting-platform)
9. [Web Dashboards](#9-web-dashboards)
10. [Monte Carlo Simulation](#10-monte-carlo-simulation)
11. [Stacked PINN Details](#11-stacked-pinn-details)
12. [Financial Metrics Guide](#12-financial-metrics-guide)
13. [Database Setup](#13-database-setup)
14. [Troubleshooting & Debugging](#14-troubleshooting--debugging)
15. [API Reference](#15-api-reference)
16. [Research Findings](#16-research-findings)
17. [References](#17-references)
18. [Methodology Visualizations Dashboard](#18-methodology-visualizations-dashboard)
19. [Known Bugs and Issues](#19-known-bugs-and-issues)
20. [Files Modified Log](#20-files-modified-log)
21. [Uncertainty Estimation](#21-uncertainty-estimation)
22. [Model Registry & Dynamic Loading](#22-model-registry--dynamic-loading)
23. [Kelly Criterion Position Sizing](#23-kelly-criterion-position-sizing)
24. [Statistical Comparison Framework](#24-statistical-comparison-framework)

---

## 1. Project Overview

### 1.1 Introduction

This project implements a **Physics-Informed Neural Network (PINN)** framework for financial forecasting that embeds quantitative finance equations directly into neural network loss functions. The system combines deep learning with financial physics constraints to improve stock price forecasting.

### 1.2 Core Philosophy

The system bridges data-driven machine learning with domain knowledge from quantitative finance by embedding physical laws (stochastic differential equations) as soft constraints in the training process.

### 1.3 Key Physics Constraints

| Constraint | Equation | Description |
|------------|----------|-------------|
| **Geometric Brownian Motion (GBM)** | `dS = μS dt + σS dW` | Models trending behavior |
| **Black-Scholes PDE** | `∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0` | No-arbitrage pricing |
| **Ornstein-Uhlenbeck (OU)** | `dX = θ(μ-X)dt + σdW` | Mean reversion |
| **Langevin Dynamics** | `dX = -γ∇U(X)dt + √(2γT)dW` | Momentum modeling |

### 1.4 Key Features

**Data Pipeline:**
- ✅ 10 years of S&P 500 data via yfinance and Alpha Vantage APIs
- ✅ PostgreSQL + TimescaleDB for time-series storage
- ✅ Parquet backups for offline access
- ✅ Feature engineering: log returns, volatility, technical indicators (RSI, MACD, Bollinger Bands)
- ✅ Temporal train/val/test splits (no data leakage)

**Model Architectures:**
- ✅ Baseline LSTM/GRU/BiLSTM
- ✅ Transformer with attention
- ✅ PINN with physics constraints
- ✅ StackedPINN with curriculum learning
- ✅ ResidualPINN with physics corrections

**Evaluation & Backtesting:**
- ✅ Comprehensive financial metrics (Sharpe, Sortino, Drawdown)
- ✅ Walk-forward validation
- ✅ Monte Carlo simulation
- ✅ Multi-strategy backtesting platform

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
Data Fetching → Preprocessing → Feature Engineering → Model Training → Evaluation → Backtesting → Dashboard
```

### 2.2 Directory Structure

```
Dissertaion-Project/
├── src/
│   ├── data/                    # Data pipeline
│   │   ├── fetcher.py           # yfinance/Alpha Vantage API
│   │   ├── preprocessor.py      # Feature engineering
│   │   └── dataset.py           # PyTorch datasets
│   ├── models/                  # Model architectures
│   │   ├── baseline.py          # LSTM, GRU, BiLSTM
│   │   ├── transformer.py       # Transformer encoder
│   │   ├── pinn.py              # Physics-Informed NN
│   │   └── stacked_pinn.py      # StackedPINN, ResidualPINN
│   ├── training/                # Training logic
│   │   ├── trainer.py           # Training loop
│   │   ├── train.py             # Main training script
│   │   ├── train_pinn_variants.py
│   │   ├── train_stacked_pinn.py
│   │   ├── curriculum.py        # Curriculum schedulers
│   │   └── walk_forward.py      # Walk-forward validation
│   ├── evaluation/              # Metrics and backtesting
│   │   ├── metrics.py           # ML metrics
│   │   ├── financial_metrics.py # Financial metrics
│   │   ├── backtester.py        # Basic backtester
│   │   ├── backtesting_platform.py # Advanced platform
│   │   ├── monte_carlo.py       # MC simulation
│   │   └── rolling_metrics.py   # Rolling analysis
│   ├── trading/                 # Trading agent
│   │   ├── agent.py             # Signal generation
│   │   └── position_sizing.py   # Kelly Criterion, etc.
│   ├── web/                     # Web interface
│   │   ├── app.py               # Main dashboard
│   │   ├── pinn_dashboard.py    # PINN comparison
│   │   ├── all_models_dashboard.py
│   │   ├── monte_carlo_dashboard.py
│   │   ├── backtesting_dashboard.py
│   │   └── prediction_visualizer.py
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration
│       ├── logger.py            # Logging
│       ├── database.py          # TimescaleDB
│       └── reproducibility.py   # Seed control
├── models/                      # Saved model checkpoints
├── results/                     # Training results
├── data/                        # Data storage
├── docker/                      # Docker configurations
├── tests/                       # Unit tests
├── examples/                    # Demo scripts
└── requirements.txt             # Dependencies
```

---

## 3. Installation & Setup

### 3.1 Prerequisites

- Python 3.10+
- Docker & Docker Compose (for database)
- CUDA-capable GPU (optional, for faster training)

### 3.2 Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Dissertaion-Project

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 3. Run setup script
./setup.sh

# 4. Activate virtual environment
source venv/bin/activate

# 5. Start database (optional but recommended)
docker-compose up -d timescaledb
```

### 3.3 Quick Start

```bash
# Interactive menu with all options
./run.sh

# Quick demo
./run.sh  # Select option 1

# Train all models
./run.sh  # Select option 11

# Launch dashboard
streamlit run src/web/app.py
```

---

## 4. Data Pipeline

### 4.1 Data Sources

**Primary:** Yahoo Finance via yfinance
- S&P 500 constituents (50+ stocks)
- 10-year historical data (2014-2024)
- OHLCV data (Open, High, Low, Close, Volume)

**Backup:** Alpha Vantage API
- Used when yfinance fails
- Requires API key in `.env`

### 4.2 Feature Engineering

```python
features = [
    'price', 'volume',
    'log_return', 'simple_return',
    'rolling_volatility_5', 'rolling_volatility_20', 'rolling_volatility_60',
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
    'rsi_14',
    'macd', 'macd_signal',
    'bollinger_upper', 'bollinger_lower',
    'atr_14', 'obv', 'stochastic'
]
```

### 4.3 Data Splits

| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

**Important:** Temporal splits ensure no data leakage (future data never used for training).

---

## 5. Model Architectures

### 5.1 Baseline Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **LSTM** | Long Short-Term Memory | Standard sequence modeling |
| **GRU** | Gated Recurrent Unit | Faster, fewer parameters |
| **BiLSTM** | Bidirectional LSTM | Better context capture |
| **Attention LSTM** | LSTM with attention | Focus on important timesteps |
| **Transformer** | Multi-head self-attention | Long-range dependencies |

### 5.2 PINN Variants

| Variant | Physics | λ Weights | Best For |
|---------|---------|-----------|----------|
| **Baseline** | None | λ = 0 | Benchmark |
| **Pure GBM** | Geometric Brownian Motion | λ_GBM = 0.1 | Trending markets |
| **Pure OU** | Ornstein-Uhlenbeck | λ_OU = 0.1 | Mean-reverting |
| **Pure BS** | Black-Scholes | λ_BS = 0.1 | Derivatives |
| **GBM+OU** | Hybrid | λ_GBM = 0.05, λ_OU = 0.05 | General |
| **Global** | All constraints | Multiple | Maximum regularization |

### 5.3 Advanced Architectures

#### StackedPINN
```
Input Features
    ↓
PhysicsEncoder (Feature-level encoding)
    ↓
Parallel Processing:
├── LSTM Head
└── GRU Head
    ↓
Attention-based Fusion
    ↓
Dense Prediction Head
├── Regression (Return prediction)
└── Classification (Direction: up/down)
```

#### ResidualPINN
```
Input Features
    ↓
Base Model (LSTM/GRU) → Base Prediction
    ↓
Physics-informed Correction Network → Correction
    ↓
Final Prediction = Base + Correction
```

---

## 6. Training Guide

### 6.1 Training Commands

```bash
# Train single model
python -m src.training.train --model lstm --epochs 100

# Train all PINN variants
python src/training/train_pinn_variants.py --epochs 100

# Train StackedPINN with curriculum learning
python src/training/train_stacked_pinn.py --model-type stacked --epochs 100 --curriculum-strategy cosine

# Train ResidualPINN
python src/training/train_stacked_pinn.py --model-type residual --epochs 100
```

### 6.2 Training Configuration

```python
# Default configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Physics weights
LAMBDA_GBM = 0.1
LAMBDA_OU = 0.1
LAMBDA_BS = 0.03
```

### 6.3 Curriculum Learning

Curriculum learning gradually increases physics constraints during training:

| Strategy | Progression | Best For |
|----------|-------------|----------|
| **Linear** | `scale = progress` | General use |
| **Exponential** | `scale = progress^2` | Fast convergence |
| **Cosine** | `scale = 0.5*(1-cos(π*progress))` | Smooth transition |
| **Step** | Discrete jumps at 25/50/75% | Controlled phases |

### 6.4 Loss Function

```
Total_Loss = Data_Loss + λ * Physics_Loss

where:
    Data_Loss = MSE(predictions, targets)
    Physics_Loss = λ_GBM * L_GBM + λ_OU * L_OU + λ_BS * L_BS
```

---

## 7. Evaluation & Metrics

### 7.1 ML Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | Mean((pred - actual)²) | Prediction error |
| **RMSE** | √MSE | Error in original units |
| **MAE** | Mean(\|pred - actual\|) | Absolute error |
| **MAPE** | Mean(\|pred - actual\|/actual) | Percentage error |
| **R²** | 1 - SS_res/SS_tot | Variance explained |

### 7.2 Financial Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Sharpe Ratio** | (Return - RiskFree) / Volatility | > 1.0 (good), > 2.0 (excellent) |
| **Sortino Ratio** | (Return - RiskFree) / Downside_Vol | > 1.5 (good) |
| **Max Drawdown** | Max(Peak - Trough) / Peak | > -20% (acceptable) |
| **Calmar Ratio** | Annual_Return / \|Max_DD\| | > 0.5 (good) |
| **Win Rate** | Profitable_Trades / Total_Trades | > 50% |
| **Profit Factor** | Gross_Profit / Gross_Loss | > 1.5 (good) |

### 7.3 Signal Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Directional Accuracy** | % correct direction | > 55% |
| **Information Coefficient** | Prediction-actual correlation | > 0.05 |
| **Precision** | TP / (TP + FP) | > 0.6 |
| **Recall** | TP / (TP + FN) | > 0.6 |

### 7.4 Running Evaluation

```bash
# Evaluate all trained models
python evaluate_existing_models.py --models all

# Compute comprehensive financial metrics
python compute_all_financial_metrics.py

# Dissertation-grade evaluation
python evaluate_dissertation_rigorous.py
```

---

## 8. Backtesting Platform

### 8.1 Features

- **Multiple Strategy Support**: Model-based, Buy & Hold, SMA, Momentum, Mean Reversion
- **Transaction Cost Modeling**: Commission (0.1%), Slippage (0.05%)
- **Risk Management**: Stop-loss, Take-profit, Max position size
- **Position Sizing**: Fixed, Kelly Criterion, Volatility-based, Confidence-based
- **Walk-Forward Validation**: Realistic time-series evaluation
- **Monte Carlo Simulation**: Bootstrap confidence intervals

### 8.2 Configuration

```python
BacktestConfig(
    initial_capital=100000.0,
    commission_rate=0.001,      # 0.1%
    slippage_rate=0.0005,       # 0.05%
    max_position_size=0.20,     # 20% of portfolio
    stop_loss=0.02,             # 2%
    take_profit=0.05,           # 5%
    position_sizing=PositionSizingMethod.KELLY,
    kelly_fraction=0.5          # Half-Kelly
)
```

### 8.3 Available Strategies

| Strategy | Description |
|----------|-------------|
| **Model Strategy** | Based on neural network predictions |
| **Buy and Hold** | Passive long-only benchmark |
| **SMA Crossover** | Moving average crossover signals |
| **Momentum** | Trend-following based on recent returns |
| **Mean Reversion** | Bollinger Band-based contrarian |

### 8.4 Running Backtests

```bash
# Launch backtesting dashboard
streamlit run src/web/backtesting_dashboard.py
```

---

## 9. Web Dashboards

### 9.1 Main Dashboard

```bash
streamlit run src/web/app.py
```

**Features:**
- Model comparison
- Real-time predictions
- Portfolio performance
- Interactive visualizations

### 9.2 PINN Comparison Dashboard

```bash
streamlit run src/web/pinn_dashboard.py
```

**Features:**
- Side-by-side PINN variant comparison
- 5 metric tabs: Metrics Comparison, **Prediction Comparison**, Rolling Performance, Training History, Model Details
- Rolling performance analysis
- Training history with curriculum visualization

**Prediction Comparison Tab (New):**
- Model A vs Model B selection
- Time series: predictions vs actual values
- Error distribution histograms
- Scatter plot with perfect prediction line
- Summary statistics (RMSE, MAE, Correlation)

### 9.3 Backtesting Dashboard

```bash
streamlit run src/web/backtesting_dashboard.py
```

**Features:**
- Strategy configuration
- Equity curve visualization
- Drawdown analysis
- Walk-forward validation

### 9.4 Monte Carlo Dashboard

```bash
streamlit run src/web/monte_carlo_dashboard.py --server.port 8503
```

**Features:**
- Confidence intervals
- VaR/CVaR analysis
- Stress testing
- Ensemble uncertainty

---

## 10. Monte Carlo Simulation

### 10.1 Features

- **Price Path Simulation**: Generate thousands of future scenarios
- **Confidence Intervals**: 95% prediction bounds
- **Risk Metrics**: Value at Risk (VaR), Conditional VaR (Expected Shortfall)
- **Stress Testing**: Evaluate under extreme conditions
- **Bootstrap Confidence Intervals**: Statistical uncertainty

### 10.2 Usage

```python
from src.evaluation.monte_carlo import MonteCarloSimulator

# Create simulator
simulator = MonteCarloSimulator(
    model=model,
    n_simulations=1000,
    seed=42
)

# Run simulation
results = simulator.simulate_paths(
    initial_data=features[-60:],
    horizon=30,
    volatility=0.20
)

# Access results
print(f"Mean forecast: {results.mean_path[-1]:.4f}")
print(f"95% CI: [{results.lower_ci[-1]:.4f}, {results.upper_ci[-1]:.4f}]")
print(f"5% VaR: {results.var_5[-1]:.4f}")
```

### 10.3 Stress Test Scenarios

| Scenario | Volatility | Drift | Description |
|----------|------------|-------|-------------|
| Base | 1.0x | 0% | Normal conditions |
| High Vol | 2.0x | 0% | Elevated uncertainty |
| Crash | 3.0x | -2% | Severe downturn |
| Bull | 0.8x | +1% | Low volatility rally |
| Black Swan | 5.0x | -5% | Extreme tail event |

---

## 11. Stacked PINN Details

### 11.1 Architecture Components

**PhysicsEncoder:**
- Multi-layer feature encoder
- LayerNorm, GELU, Dropout
- Physics-aware projection

**ParallelHeads:**
- LSTM and GRU processed in parallel
- Attention mechanism for combining outputs

**PredictionHead:**
- Shared dense layers
- Regression head (return prediction)
- Classification head (direction)

### 11.2 Training

```bash
python src/training/train_stacked_pinn.py \
    --model-type stacked \
    --epochs 100 \
    --warmup-epochs 10 \
    --final-lambda-gbm 0.1 \
    --final-lambda-ou 0.1 \
    --curriculum-strategy cosine
```

### 11.3 Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | stacked | `stacked` or `residual` |
| `--epochs` | 100 | Training epochs |
| `--warmup-epochs` | 10 | Warmup with λ=0 |
| `--final-lambda-gbm` | 0.1 | Final GBM weight |
| `--final-lambda-ou` | 0.1 | Final OU weight |
| `--curriculum-strategy` | cosine | Curriculum type |

### 11.4 Return-Based Features

The system uses **only return-based features** to avoid look-ahead bias:

- `log_return`: Logarithmic returns
- `simple_return`: Percentage returns
- `rolling_volatility_*`: Rolling volatility windows
- `momentum_*`: Price momentum
- `rsi_14`: Relative Strength Index
- `macd`, `macd_signal`: MACD indicators

---

## 12. Financial Metrics Guide

### 12.1 Risk-Adjusted Performance

**Sharpe Ratio:**
```python
sharpe = (mean_return - risk_free_rate) / volatility
# Interpretation: Return per unit of total risk
# Good: > 1.0, Excellent: > 2.0
```

**Sortino Ratio:**
```python
sortino = (mean_return - risk_free_rate) / downside_deviation
# Interpretation: Return per unit of downside risk
# More stringent than Sharpe
```

### 12.2 Capital Preservation

**Maximum Drawdown:**
```python
max_drawdown = max((peak - trough) / peak)
# Interpretation: Worst loss from peak
# Target: > -20%
```

**Drawdown Duration:**
```python
duration = time_from_peak_to_recovery / periods_per_year
# Interpretation: Recovery time
# Shorter is better
```

**Calmar Ratio:**
```python
calmar = annualized_return / abs(max_drawdown)
# Interpretation: Return per unit of drawdown
# Good: > 0.5
```

### 12.3 Trading Viability

**Profit Factor:**
```python
profit_factor = sum(positive_returns) / abs(sum(negative_returns))
# Interpretation: Profit per dollar lost
# Good: > 1.5
```

**Win Rate:**
```python
win_rate = count(positive_returns) / total_trades
# Interpretation: Percentage profitable
# Must balance with profit factor
```

### 12.4 Important Note on Sharpe Ratios

All PINN models may show identical Sharpe ratios (~26) when:
1. All models predict predominantly positive returns
2. This results in identical 100% long positions
3. Identical positions → identical returns → identical Sharpe

**Better comparison metrics:**
- Directional Accuracy
- Information Coefficient
- RMSE/MAE
- Prediction correlation

---

## 13. Database Setup

### 13.1 TimescaleDB Configuration

```bash
# Start database
docker-compose up -d timescaledb

# Initialize schema
python init_db_schema.py

# Verify tables
docker exec -it pinn-timescaledb psql -U pinn_user -d pinn_finance -c "\dt finance.*"
```

### 13.2 Database Schema

**Tables:**
- `finance.stock_prices` - OHLCV data (Hypertable)
- `finance.features` - Engineered features (Hypertable)
- `finance.predictions` - Model predictions
- `finance.model_metrics` - Training metrics
- `finance.training_history` - Epoch-by-epoch history

### 13.3 Fallback

If Docker/TimescaleDB unavailable, the system automatically falls back to local Parquet files.

---

## 14. Troubleshooting & Debugging

### 14.1 Debug Mode

```bash
# Enable verbose logging
DEBUG=1 ./run.sh
DEBUG=1 ./setup.sh
```

### 14.2 Common Issues

**"No comprehensive metrics" message:**
```bash
./run.sh  # Select option 13 to compute metrics
```

**"Checkpoint not found":**
```bash
# Train models first
./run.sh  # Select option 11 or 12
```

**Dashboard not loading:**
```bash
pip install streamlit plotly pandas numpy
streamlit run src/web/app.py
```

**Database connection failed:**
```bash
# Check Docker
docker ps | grep timescale

# Restart container
docker-compose down
docker-compose up -d timescaledb
```

### 14.3 Log Files

- **Setup logs:** `setup_YYYYMMDD_HHMMSS.log`
- **Run logs:** `run_YYYYMMDD_HHMMSS.log`
- **Application logs:** `logs/pinn_finance.log`

### 14.4 Viewing Logs

```bash
# Most recent log
tail -f run_*.log

# Search for errors
grep -i error *.log

# Monitor training
tail -f logs/pinn_finance.log
```

---

## 15. API Reference

### 15.1 Data Module

```python
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import FinancialDataset

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch_stock_data(['AAPL', 'MSFT'], start='2020-01-01')

# Preprocess
preprocessor = DataPreprocessor()
features = preprocessor.engineer_features(data)

# Create dataset
dataset = FinancialDataset(features, sequence_length=60)
```

### 15.2 Models Module

```python
from src.models.baseline import LSTM, GRU, BiLSTM
from src.models.transformer import TransformerModel
from src.models.pinn import PINNModel
from src.models.stacked_pinn import StackedPINN, ResidualPINN

# Create PINN model
model = PINNModel(
    input_dim=15,
    hidden_dim=128,
    num_layers=2,
    lambda_gbm=0.1,
    lambda_ou=0.1
)
```

### 15.3 Evaluation Module

```python
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.financial_metrics import FinancialMetrics, compute_all_metrics
from src.evaluation.backtesting_platform import BacktestingPlatform

# Calculate metrics
metrics = FinancialMetrics.compute_all_metrics(predictions, targets)

# Run backtest
platform = BacktestingPlatform()
result = platform.run_backtest(strategy, prices, predictions)
```

### 15.4 Trading Module

```python
from src.trading.agent import SignalGenerator, TradingAgent
from src.trading.position_sizing import KellyCriterionSizer

# Generate signals
generator = SignalGenerator(model)
signals = generator.generate_signals(sequences, prices, tickers, timestamps)

# Position sizing
sizer = KellyCriterionSizer(fractional_kelly=0.5)
size = sizer.calculate(capital, price, win_rate, avg_win, avg_loss)
```

---

## 16. Research Findings

### 16.1 Key Results

1. **Physics constraints improve generalization** - PINN models show better out-of-sample performance
2. **Curriculum learning stabilizes training** - Gradual λ increase prevents divergence
3. **All models achieve similar directional accuracy** - ~99% in trending markets
4. **Traditional metrics may not differentiate models** - Use directional accuracy and IC

### 16.2 Model Comparison

| Model | Directional Accuracy | Information Coefficient | RMSE |
|-------|---------------------|------------------------|------|
| PINN Global | 99.94% | 0.922 | 1.020 |
| StackedPINN | 99.90% | 0.920 | 1.024 |
| PINN GBM+OU | 99.94% | 0.921 | 1.022 |
| Baseline LSTM | 99.90% | 0.918 | 1.028 |

### 16.3 Recommendations

1. **Use multiple metrics** - Don't rely on a single metric
2. **Prioritize stability** - Consistent performance > peak performance
3. **Include transaction costs** - Many strategies fail with realistic costs
4. **Walk-forward validation** - Essential for realistic evaluation

---

## 17. References

### Academic Papers

1. **Physics-Informed Neural Networks:**
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics.

2. **Financial Models:**
   - Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." Journal of Political Economy.
   - Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the Theory of the Brownian Motion." Physical Review.

3. **Performance Evaluation:**
   - Sharpe, W. F. (1966). "Mutual Fund Performance." Journal of Business.
   - Sortino, F. A., & Van Der Meer, R. (1991). "Downside Risk." Journal of Portfolio Management.
   - Diebold, F. X., & Mariano, R. S. (1995). "Comparing Predictive Accuracy." Journal of Business & Economic Statistics.

4. **Curriculum Learning:**
   - Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "Curriculum Learning." ICML.

### Technical Documentation

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Appendix A: Quick Reference Commands

```bash
# Setup
./setup.sh

# Interactive menu
./run.sh

# Quick demo
./run.sh  # Select option 1

# Train all models
./run.sh  # Select option 11

# Train baseline models only
./run.sh  # Select option 12

# Train PINN variants
./run.sh  # Select option 10

# Compute metrics
python compute_all_financial_metrics.py

# View metrics
python view_metrics.py --compare

# Launch dashboard
streamlit run src/web/app.py

# PINN comparison
streamlit run src/web/pinn_dashboard.py

# Backtesting
streamlit run src/web/backtesting_dashboard.py

# Monte Carlo
streamlit run src/web/monte_carlo_dashboard.py
```

---

## Appendix B: Performance Benchmarks

### Training Time (100 epochs)

| Model | GPU (RTX 3080) | CPU (i7-10700) |
|-------|----------------|----------------|
| LSTM | 10 min | 45 min |
| GRU | 8 min | 35 min |
| BiLSTM | 15 min | 60 min |
| Transformer | 20 min | 90 min |
| PINN | 15 min | 60 min |
| StackedPINN | 25 min | 120 min |

### Memory Requirements

| Model | VRAM | RAM |
|-------|------|-----|
| LSTM | 2 GB | 4 GB |
| Transformer | 4 GB | 6 GB |
| StackedPINN | 6 GB | 8 GB |

---

## Appendix C: Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-06 | 1.2 | **Major Update:** Uncertainty estimation, Kelly Criterion integration, model registry dynamic loading, prediction comparison visualization, statistical comparison framework improvements, exception handling fixes |
| 2026-02-05 | 1.1 | Added All Models Comparison tab, comprehensive bug documentation |
| 2026-02-04 | 1.0 | Complete documentation consolidated |
| 2026-01-28 | 0.9 | Financial metrics system completed |
| 2026-01-26 | 0.8 | StackedPINN implementation |
| 2026-01-20 | 0.7 | Monte Carlo simulation added |
| 2026-01-15 | 0.6 | Backtesting platform enhanced |

---

## 18. Methodology Visualizations Dashboard

### 18.1 Overview

The Methodology Visualizations dashboard provides academic visualizations demonstrating key concepts from the dissertation. Access via:

```bash
streamlit run src/web/app.py
# Navigate to "Methodology Visualizations"
```

### 18.2 Available Visualizations

| Tab | Description | Key Features |
|-----|-------------|--------------|
| **Physics/Data Loss** | Curriculum learning dynamics | Loss convergence, physics weight progression, multi-model comparison |
| **Stationarity (EDA)** | Why log-returns are used | Price vs returns distribution, statistical tests |
| **PINN Architecture** | Network structure diagram | Visual architecture schematic |
| **Walk-Forward Validation** | Temporal integrity protocol | Sliding window visualization |
| **All Models Comparison** | Compare ALL models | Baseline, PINN, Advanced side-by-side |
| **Baseline Comparison** | Traditional NN comparison | LSTM, GRU, Transformer metrics |

### 18.3 All Models Comparison Feature

The "All Models Comparison" tab allows comprehensive comparison of all trained models:

**Model Types:**
- **Baseline**: LSTM, GRU, BiLSTM, Attention-LSTM, Transformer
- **PINN**: Baseline, GBM, OU, Black-Scholes, GBM+OU, Global
- **Advanced**: StackedPINN, ResidualPINN

**Comparison Features:**
- Filter by model type (baseline, pinn, advanced)
- Switch between ML metrics (MSE, RMSE, R²) and Financial metrics (Sharpe, Sortino)
- Grouped bar charts showing performance by model type
- Radar chart for multi-dimensional comparison
- Automatic highlighting of best performers

**Usage:**
1. Select model types to compare
2. Choose metric category (ML, Financial, or All)
3. View summary table with highlighted best values
4. Analyze performance aggregated by model type
5. Select specific models for detailed radar comparison

### 18.4 Training History Comparison

The Physics/Data Loss tab supports multi-model training history comparison:

1. Check "Compare multiple models"
2. Select models from dropdown
3. View overlaid training curves
4. Compare convergence statistics

### 18.5 Visualization Details

#### Physics vs Data Loss
**Purpose:** Demonstrate curriculum learning solving the "stiff PDE" problem.

**Key Visualizations:**
- Loss Convergence (log scale) - Shows data loss, physics loss, total loss
- Curriculum Learning Schedule - Stacked area showing loss composition
- Physics Weight Progression - How lambda increases over training
- Loss Ratio Over Training - Balance between physics and data

**Key Insight:** The warm-start phase (first ~20 epochs) trains data-only, then physics constraints are gradually introduced to prevent gradient pathology.

#### Stationarity Analysis (EDA)
**Purpose:** Justify using log-returns instead of raw prices.

**Key Visualizations:**
- Raw Price Time Series (non-stationary)
- Price Distribution (non-normal, skewed)
- Log Returns Time Series (stationary)
- Log Returns Distribution (approximately normal)

**Statistical Comparison:**
| Property | Raw Prices | Log Returns |
|----------|------------|-------------|
| Mean | Drifting | Constant (~0) |
| Variance | Heteroscedastic | Constant |
| Distribution | Skewed | Symmetric |

#### PINN Architecture
**Purpose:** Visual representation of the network structure.

**Components:**
- Input Variables (S, t)
- Hidden Layers (3 x 128 neurons, tanh)
- Output V(S,t)
- AutoDiff Branch (computes partial derivatives)
- Combined Loss Function

#### Walk-Forward Validation
**Purpose:** Demonstrate temporal integrity and prevent data leakage.

**Configuration Parameters:**
- Number of validation windows (3-8)
- Training window size (50-200 days)
- Test window size (5-30 days)
- Step size (5-30 days)

**Key Properties:**
- Training data ALWAYS precedes test data
- No look-ahead bias
- Realistic trading simulation

---

## 19. Known Bugs and Issues

### 19.1 Critical Issues (Must Fix)

#### BUG #1: Infinity/NaN Values in ResidualPINN Financial Metrics
- **Location**: `results/pinn_residual_results.json`
- **Symptom**: `total_return: Infinity`, `max_drawdown: NaN`
- **Root Cause**: `compute_strategy_returns()` doesn't handle cumulative product overflow
- **Status**: FIXED - Added overflow protection with return clipping

#### BUG #2: Extreme Values in StackedPINN Financial Metrics
- **Location**: `results/pinn_comparison/detailed_results.json`
- **Symptom**: `max_drawdown: -6696%` (impossible), `total_return: 7.34e+284`
- **Root Cause**: Missing bounds validation on max_drawdown
- **Status**: FIXED - Added -100% cap on max_drawdown

#### BUG #10: Max Drawdown > -100% (Impossible Value)
- **Location**: StackedPINN results
- **Symptom**: Drawdown values exceeding -100%
- **Status**: FIXED - Dashboard now enforces -100% cap

#### BUG #13: ImportError - Missing Standalone Functions
- **Location**: `src/evaluation/backtesting_platform.py` import
- **Symptom**: `ImportError: cannot import name 'calculate_sharpe_ratio'`
- **Root Cause**: Functions were class methods, not standalone
- **Status**: FIXED - Added complete standalone function implementations

### 19.2 High Priority Issues

#### BUG #3: Inconsistent Metric Sources Across Tables
- **Location**: Dashboard metric tables
- **Symptom**: Same model shows different MSE/RMSE values in different tables
- **Root Cause**: Multiple result file formats with different structures
- **Status**: FIXED - Added `_normalize_metrics()` method

#### BUG #4: MSE Missing (Computed as None)
- **Location**: `detailed_results.json`
- **Symptom**: MSE shows as `None` or `NaN`
- **Root Cause**: `train_pinn_variants.py` only saves RMSE, not MSE
- **Status**: FIXED - MSE now computed as RMSE²

#### BUG #6: Information Coefficient (IC) Inconsistency
- **Location**: Various results files
- **Symptom**: High IC with poor Sharpe, or vice versa
- **Root Cause**: IC computed on price levels, not returns
- **Status**: FIXED - Added `use_returns=True` parameter

### 19.3 Medium Priority Issues

#### BUG #7: Directional Accuracy Scale Inconsistency
- **Symptom**: Some tables show 51.4%, others show 0.514
- **Root Cause**: Different modules use different scales (0-100 vs 0-1)
- **Status**: FIXED - Standardized to 0-1 scale internally

#### BUG #8: Calmar Ratio Capping Creates Artifacts
- **Symptom**: Multiple models show exactly `Calmar = 10.0`
- **Root Cause**: Intentional capping at ±10 for extreme values
- **Status**: ACCEPTABLE - Added warning logging

#### BUG #11: Precision/Recall = 0 for StackedPINN
- **Symptom**: Zero precision/recall despite non-zero accuracy
- **Root Cause**: Computed on absolute values, not changes
- **Status**: FIXED - Now uses returns with `use_returns=True`

### 19.4 Low Priority Issues

#### BUG #5: R² Negative with Positive Trading Returns
- **Symptom**: Negative R² but positive Sharpe ratio
- **Root Cause**: Not a bug - conceptual difference between regression fit and trading
- **Status**: DOCUMENTED - Added explanation to dashboard

#### BUG #12: Training History Shows 0% Train Directional Accuracy
- **Location**: StackedPINN/ResidualPINN training history
- **Symptom**: `train_directional_acc: [0.0, 0.0, ...]`
- **Status**: PENDING INVESTIGATION

### 19.5 Bug Verification

After fixing bugs, verify with:

```bash
# Re-run evaluations
python compute_all_financial_metrics.py

# Check for impossible values
python -c "
import json
from pathlib import Path

results_dir = Path('results')
for f in results_dir.glob('*.json'):
    try:
        data = json.load(open(f))
        fm = data.get('financial_metrics', {})
        max_dd = fm.get('max_drawdown', 0)
        total_ret = fm.get('total_return', 0)

        if max_dd < -1.0:
            print(f'{f.name}: INVALID max_drawdown = {max_dd}')
        if abs(total_ret) > 1e6:
            print(f'{f.name}: SUSPICIOUS total_return = {total_ret}')
    except:
        pass
"

# Launch dashboard and verify
streamlit run src/web/app.py
```

---

## 20. Files Modified Log

### Recent Changes (February 6, 2026)

| File | Changes |
|------|---------|
| `src/trading/agent.py` | **NEW:** `UncertaintyEstimator` class, MC Dropout, ensemble predictions, prediction intervals, uncertainty-aware signal generation |
| `src/models/model_registry.py` | **NEW:** `load_model()` method, `_instantiate_model()`, dynamic model loading from checkpoints |
| `src/evaluation/backtester.py` | **NEW:** `PositionSizingMethod` enum, Kelly Criterion integration, trade statistics tracking, `compare_position_sizing_methods()` |
| `src/web/monte_carlo_dashboard.py` | Dynamic model loading from registry, model info display |
| `src/web/pinn_dashboard.py` | **NEW:** `load_predictions()`, `render_prediction_comparison()`, prediction vs actual visualization |
| `src/training/trainer.py` | **FIX:** Added `metadata['inputs'] = sequences` for Black-Scholes autograd |
| `compare_pinn_baseline.py` | **FIX:** Real rolling metrics, bootstrap CI, multiple comparison corrections, specific exception handling |
| Multiple files | **FIX:** Replaced `except: pass` with specific exception types and logging |

### Previous Changes (February 5, 2026)

| File | Changes |
|------|---------|
| `src/evaluation/financial_metrics.py` | Added standalone functions, overflow protection, IC returns parameter |
| `src/evaluation/metrics.py` | Standardized directional_accuracy scale, added MSE |
| `src/web/pinn_dashboard.py` | Added `_normalize_metrics()`, metric source normalization |
| `src/web/all_models_dashboard.py` | Added `_normalize_metrics()`, improved metric loading |
| `src/web/methodology_dashboard.py` | Added "All Models Comparison" tab, multi-model comparison |

### Key Function Changes

**`src/evaluation/financial_metrics.py`:**
- `information_coefficient()`: Added `use_returns` parameter (default: True)
- `precision_recall()`: Added `use_returns` parameter (default: True)
- `compute_strategy_returns()`: Added cumulative overflow detection
- `compute_all_metrics()`: Added inf/nan validation
- Added standalone functions: `calculate_sharpe_ratio()`, `calculate_sortino_ratio()`, `calculate_max_drawdown()`, `calculate_calmar_ratio()`, `compute_all_metrics()`

---

## Appendix D: Model Comparison Quick Reference

### All Available Models

| Key | Name | Type | Physics Constraints |
|-----|------|------|---------------------|
| `lstm` | LSTM | baseline | None |
| `gru` | GRU | baseline | None |
| `bilstm` | BiLSTM | baseline | None |
| `attention_lstm` | Attention-LSTM | baseline | None |
| `transformer` | Transformer | baseline | None |
| `baseline` | PINN Baseline | pinn | λ = 0 (data-only) |
| `gbm` | PINN GBM | pinn | λ_GBM = 0.1 |
| `ou` | PINN OU | pinn | λ_OU = 0.1 |
| `black_scholes` | PINN Black-Scholes | pinn | λ_BS = 0.1 |
| `gbm_ou` | PINN GBM+OU | pinn | λ_GBM = 0.05, λ_OU = 0.05 |
| `global` | PINN Global | pinn | All constraints |
| `stacked` | StackedPINN | advanced | Curriculum learning |
| `residual` | ResidualPINN | advanced | Physics correction |

### Performance Expectations

| Model Type | Expected Strengths | Expected Weaknesses |
|------------|-------------------|---------------------|
| **Baseline** | Fast training, no assumptions | May overfit, no physics guidance |
| **PINN** | Physics regularization, better generalization | Slower training, requires tuning λ |
| **Advanced** | Multi-scale features, curriculum learning | Most complex, longest training |

---

---

## 21. Uncertainty Estimation

### 21.1 Overview

The trading agent now includes comprehensive uncertainty estimation capabilities, enabling more robust trading decisions based on model confidence.

**Location:** `src/trading/agent.py`

### 21.2 UncertaintyEstimator Class

```python
from src.trading.agent import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model=model,
    device=device,
    n_mc_samples=50,           # Number of MC Dropout samples
    ensemble_models=None       # Optional ensemble models
)
```

### 21.3 Available Methods

#### MC Dropout Estimation
```python
mean_preds, std_preds, all_samples = estimator.mc_dropout_estimate(sequences)
```
- Performs multiple forward passes with dropout enabled
- Captures epistemic (model) uncertainty
- Returns mean predictions and standard deviation

#### Ensemble Estimation
```python
mean_preds, std_preds = estimator.ensemble_estimate(sequences)
```
- Uses predictions from multiple models
- Requires `ensemble_models` to be provided

#### Prediction Intervals
```python
lower, upper = estimator.prediction_intervals(mean, std, confidence=0.95)
```
- Computes 95% confidence intervals
- Based on Gaussian assumption

#### Uncertainty to Confidence
```python
confidence = estimator.uncertainty_to_confidence(std, scale='normalized')
```
- Converts uncertainty (std) to confidence scores (0-1)
- Higher uncertainty → lower confidence

### 21.4 Integration with Signal Generation

```python
from src.trading.agent import SignalGenerator

generator = SignalGenerator(
    model=model,
    n_mc_samples=50,
    ensemble_models=None
)

# Generate signals with uncertainty-aware decisions
signals, uncertainty_details = generator.generate_signals(
    sequences=sequences,
    current_prices=prices,
    tickers=tickers,
    timestamps=timestamps,
    estimate_uncertainty=True,
    uncertainty_method='mc_dropout',  # or 'ensemble', 'both'
    risk_adjusted=True               # Adjusts thresholds by uncertainty
)
```

### 21.5 Risk-Adjusted Trading

When `risk_adjusted=True`, the signal generator:
1. Increases buy/sell thresholds for high uncertainty predictions
2. Validates prediction intervals (buy only if lower bound > current price)
3. Uses confidence scores from uncertainty estimation

### 21.6 Signal Dataclass

```python
@dataclass
class Signal:
    timestamp: pd.Timestamp
    ticker: str
    action: str                      # 'BUY', 'SELL', 'HOLD'
    confidence: float                # From uncertainty estimation
    predicted_price: float
    current_price: float
    expected_return: float
    prediction_std: Optional[float]  # Uncertainty
    prediction_interval_lower: Optional[float]
    prediction_interval_upper: Optional[float]
```

---

## 22. Model Registry & Dynamic Loading

### 22.1 Overview

The model registry provides centralized model management with dynamic loading capabilities for trained models.

**Location:** `src/models/model_registry.py`

### 22.2 ModelRegistry Class

```python
from src.models.model_registry import get_model_registry

registry = get_model_registry(project_root)

# Get all models
all_models = registry.get_all_models()

# Get trained models only
trained_models = registry.get_trained_models()

# Get models by type
pinn_models = registry.get_models_by_type('pinn')
```

### 22.3 Dynamic Model Loading

```python
# Load a trained model from checkpoint
model = registry.load_model(
    model_key='global',           # Model key (e.g., 'lstm', 'global', 'stacked')
    device=torch.device('cuda'),  # Target device
    input_dim=5                   # Input feature dimension
)
```

**Supported Architectures:**
- LSTM, GRU, BiLSTM
- Transformer
- PINNModel (all variants)
- StackedPINN, ResidualPINN

### 22.4 Model Information

```python
model_info = registry.get_model_info('global')

print(f"Name: {model_info.model_name}")
print(f"Type: {model_info.model_type}")
print(f"Architecture: {model_info.architecture}")
print(f"Trained: {model_info.trained}")
print(f"Checkpoint: {model_info.checkpoint_path}")
print(f"Epochs: {model_info.epochs_trained}")
print(f"Date: {model_info.training_date}")
```

### 22.5 Monte Carlo Dashboard Integration

The Monte Carlo dashboard now dynamically loads trained models:

```python
# Dashboard automatically:
# 1. Fetches trained models from registry
# 2. Shows only available models in dropdown
# 3. Loads selected model from checkpoint
# 4. Displays model info (architecture, epochs, date)
```

---

## 23. Kelly Criterion Position Sizing

### 23.1 Overview

The backtester now supports multiple position sizing methods, including Kelly Criterion for optimal capital allocation.

**Location:** `src/evaluation/backtester.py`, `src/trading/position_sizing.py`

### 23.2 Available Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `FIXED` | Fixed percentage per trade | Conservative, simple |
| `KELLY_FULL` | Full Kelly Criterion | Aggressive, high variance |
| `KELLY_HALF` | Half Kelly (recommended) | Balanced growth/risk |
| `KELLY_QUARTER` | Quarter Kelly | Conservative Kelly |
| `VOLATILITY` | Inverse volatility sizing | Risk parity approach |
| `CONFIDENCE` | Based on model confidence | Uncertainty-aware |

### 23.3 Backtester Configuration

```python
from src.evaluation.backtester import Backtester, PositionSizingMethod

backtester = Backtester(
    initial_capital=100000.0,
    position_sizing_method=PositionSizingMethod.KELLY_HALF,
    risk_per_trade=0.02
)

results = backtester.run_backtest(signals, prices)
```

### 23.4 Kelly Criterion Formula

```
f* = (p × b - q) / b

where:
    f* = optimal fraction to bet
    p  = probability of win
    q  = probability of loss (1 - p)
    b  = odds (average win / average loss)
```

### 23.5 Trade Statistics Tracking

The backtester automatically tracks statistics for Kelly calculations:

```python
# After backtest
print(f"Win Rate: {results.metrics['trade_win_rate']:.2%}")
print(f"Avg Win: {results.metrics['avg_win_pct']:.2f}%")
print(f"Avg Loss: {results.metrics['avg_loss_pct']:.2f}%")
```

### 23.6 Comparing Position Sizing Methods

```python
from src.evaluation.backtester import compare_position_sizing_methods

# Compare methods on same signals
results = compare_position_sizing_methods(
    signals=signals_df,
    prices=prices_df,
    initial_capital=100000.0,
    methods=[
        PositionSizingMethod.FIXED,
        PositionSizingMethod.KELLY_HALF,
        PositionSizingMethod.KELLY_QUARTER,
        PositionSizingMethod.CONFIDENCE
    ]
)

# Output comparison table
for method, result in results.items():
    print(f"{method}: Return={result.metrics['total_return_pct']:.2f}%")
```

---

## 24. Statistical Comparison Framework

### 24.1 Overview

The comparison framework performs rigorous statistical analysis between PINN variants and baseline models.

**Location:** `compare_pinn_baseline.py`

### 24.2 Key Features

- **Real Rolling Metrics**: Uses actual rolling window statistics from evaluation results
- **Bootstrap Confidence Intervals**: 95% CI for metric differences
- **Multiple Comparison Corrections**: Bonferroni, FDR, Holm-Bonferroni
- **Effect Size Calculation**: Cohen's d for practical significance

### 24.3 Rolling Metrics Extraction

```python
from compare_pinn_baseline import PINNBaselineComparison

comparison = PINNBaselineComparison()

# Extract rolling statistics from results
stats = comparison.extract_rolling_metric_stats(
    model_name='pinn_global',
    metric='sharpe_ratio'
)

print(f"Mean: {stats['mean']:.4f}")
print(f"Std: {stats['std']:.4f}")
print(f"Windows: {stats['n_windows']}")
```

### 24.4 Bootstrap Confidence Intervals

```python
ci_result = comparison.bootstrap_confidence_interval(
    model1='pinn_global',
    model2='lstm',
    metric='sharpe_ratio',
    n_bootstrap=10000,
    confidence=0.95
)

print(f"Difference: {ci_result['difference']:.4f}")
print(f"95% CI: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
print(f"Significant: {ci_result['significant']}")
```

### 24.5 Multiple Comparison Correction

```python
p_values = [0.01, 0.03, 0.02, 0.08, 0.04]

# Bonferroni correction
corrected = comparison.multiple_comparison_correction(
    p_values,
    method='bonferroni'  # or 'fdr', 'holm'
)
```

### 24.6 Running Full Comparison

```bash
# Generate dissertation-grade comparison
python compare_pinn_baseline.py

# Output includes:
# - Statistical test results (t-test, Wilcoxon)
# - Effect sizes (Cohen's d)
# - Bootstrap confidence intervals
# - LaTeX tables for dissertation
# - Comparison figures (PDF/PNG)
```

---

## Appendix E: Implementation Updates (February 6, 2026)

### Black-Scholes PDE Fix

**Issue:** Black-Scholes autograd was not executing because `inputs` was missing from metadata.

**Fix in `src/training/trainer.py`:**
```python
# CRITICAL: Add inputs to metadata for Black-Scholes autograd
# This enables computing exact derivatives via torch.autograd.grad
metadata['inputs'] = sequences
```

**Verification:** The `black_scholes_autograd_residual()` method in `pinn.py` now receives the input tensor and can compute exact derivatives using `torch.autograd.grad(create_graph=True)`.

### 99.9% Directional Accuracy Clarification

**Investigation Result:** The 99.9% value was **precision/recall**, NOT directional accuracy.

**Actual Metrics:**
| Metric | Value | Explanation |
|--------|-------|-------------|
| Directional Accuracy | ~51% | Reasonable, slightly above random |
| Precision/Recall | ~99.97% | High correlation in price level predictions |
| Information Coefficient | ~0.92 | High prediction-target correlation |
| Win Rate | ~28% | Strategy not profitable despite high IC |

**Conclusion:** No data leakage found. The confusion was between regression accuracy (high) and trading accuracy (low).

### Exception Handling Improvements

**Fixed Files:**
- `src/models/model_registry.py`
- `src/web/methodology_dashboard.py`
- `src/web/all_models_dashboard.py`
- `evaluate_existing_models.py`
- `recompute_metrics.py`
- `empirical_validation.py`
- `compare_pinn_baseline.py`

**Pattern Replaced:**
```python
# Before (bad)
except:
    pass

# After (good)
except (json.JSONDecodeError, IOError, KeyError) as e:
    logger.debug(f"Could not load: {e}")
```

---

**Built for Academic Research**

*This documentation consolidates all project guides into a single comprehensive reference.*

*For questions or contributions, please open an issue on GitHub.*

*Last Updated: February 6, 2026*
