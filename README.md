# Physics-Informed Neural Network (PINN) for Financial Forecasting

A comprehensive deep learning system that embeds quantitative finance equations (Black-Scholes, Ornstein-Uhlenbeck, Langevin dynamics, Geometric Brownian Motion) directly into neural network loss functions for improved financial forecasting and trading.

## ⚠️ DISCLAIMER

**THIS IS FOR ACADEMIC RESEARCH ONLY - NOT FINANCIAL ADVICE**

- This system is a dissertation research project
- NOT investment advice or recommendations
- Simulation only - no real trading functionality
- Past performance does not guarantee future results
- Always consult qualified financial advisors before making investment decisions
- The authors assume no liability for any financial losses

## 🎯 Project Overview

This project implements a Physics-Informed Neural Network (PINN) framework that:

1. **Embeds Financial Physics** into neural network training via custom loss functions
2. **Compares Multiple Architectures**: LSTM, GRU, Transformer, and PINN models
3. **Implements Full Trading Pipeline**: Data → Training → Backtesting → Visualization
4. **Ensures Reproducibility**: Docker containers, random seeds, comprehensive logging

### Key Physics Constraints

- **Geometric Brownian Motion (GBM)**: `dS = μS dt + σS dW`
- **Black-Scholes PDE**: `∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0`
- **Ornstein-Uhlenbeck Process**: `dX = θ(μ-X)dt + σdW` (mean reversion)
- **Langevin Dynamics**: `dX = -γ∇U(X)dt + √(2γT)dW` (momentum modeling)

## 📋 Features

### Data Pipeline
- ✅ 10 years of S&P 500 data via yfinance and Alpha Vantage APIs
- ✅ PostgreSQL + TimescaleDB for time-series storage
- ✅ Parquet backups for offline access
- ✅ Feature engineering: log returns, volatility, technical indicators (RSI, MACD, Bollinger Bands)
- ✅ Stationarity testing (ADF test)
- ✅ Temporal train/val/test splits (no data leakage)

### Model Architectures
- ✅ **Baseline LSTM/GRU**: Standard sequence models
- ✅ **Bidirectional LSTM**: Improved context capture
- ✅ **Transformer**: Attention-based architecture
- ✅ **PINN**: Physics-constrained neural network (primary contribution)

### Evaluation & Backtesting
- ✅ **Predictive Metrics**: RMSE, MAE, MAPE, R², Directional Accuracy
- ✅ **Financial Metrics**: Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Win Rate
- ✅ **Walk-Forward Validation**: Expanding/sliding windows
- ✅ **Statistical Testing**: Diebold-Mariano test, bootstrap confidence intervals
- ✅ **Risk Management**: Stop-loss, take-profit, position sizing

### Trading Agent
- ✅ Signal generation from predictions
- ✅ Portfolio management with risk controls
- ✅ Transaction costs and slippage modeling
- ✅ Benchmark strategies (buy-and-hold, SMA crossover)

### Web Interface
- ✅ **Streamlit Dashboard**: Interactive visualizations
- ✅ **Real-time Predictions**: Model inference interface
- ✅ **Performance Charts**: Portfolio value, returns, metrics
- ✅ **Prominent Disclaimers**: Academic use only

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for database)
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Dissertaion-Project
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. **Install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate  
pip3 install -r requirements.txt
```

4. **Start database (optional but recommended)**
```bash
docker-compose up -d timescaledb
```

### Running the System

#### Option 1: Quick Demo (CPU, subset of data)
```bash
# Fetch data
python3 -m src.data.fetcher

# Train PINN model
python3 -m src.training.train --model pinn --epochs 20

# Launch web interface
streamlit run src/web/app.py
```

#### Option 2: Full Training (GPU recommended)
```bash
# Train all models for comparison
python -m src.training.train --model lstm --epochs 100
python -m src.training.train --model gru --epochs 100
python -m src.training.train --model transformer --epochs 100
python -m src.training.train --model pinn --epochs 100

# View results
streamlit run src/web/app.py
```

#### Option 3: Docker (Full Stack)
```bash
# Build and run all services
docker-compose up --build

# Access web interface at http://localhost:8501
```

## 📁 Project Structure

```
Dissertaion-Project/
├── src/
│   ├── data/               # Data pipeline
│   │   ├── fetcher.py      # yfinance/Alpha Vantage API
│   │   ├── preprocessor.py # Feature engineering, normalization
│   │   └── dataset.py      # PyTorch datasets
│   ├── models/             # Model architectures
│   │   ├── baseline.py     # LSTM, GRU, BiLSTM
│   │   ├── transformer.py  # Transformer encoder
│   │   └── pinn.py         # Physics-Informed NN
│   ├── training/           # Training logic
│   │   ├── trainer.py      # Training loop, early stopping
│   │   └── train.py        # Main training script
│   ├── evaluation/         # Metrics and backtesting
│   │   ├── metrics.py      # RMSE, Sharpe, etc.
│   │   └── backtester.py   # Trading simulation
│   ├── trading/            # Trading agent
│   │   └── agent.py        # Signal generation, risk mgmt
│   ├── web/                # Web interface
│   │   └── app.py          # Streamlit dashboard
│   └── utils/              # Utilities
│       ├── config.py       # Configuration management
│       ├── logger.py       # Logging setup
│       ├── database.py     # TimescaleDB interface
│       └── reproducibility.py  # Random seed control
├── configs/                # Configuration files
├── data/                   # Data storage (gitignored)
├── checkpoints/            # Model checkpoints (gitignored)
├── docker/                 # Docker configurations
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker orchestration
└── README.md              # This file
```

## 🔬 Methodology

### 1. Data Collection
- S&P 500 constituent stocks from yfinance (10 years)
- Backup Alpha Vantage API for missing data
- Storage in TimescaleDB hypertables for efficient time-series queries

### 2. Feature Engineering
```python
features = [
    'price', 'volume',
    'log_return', 'simple_return',
    'rolling_volatility_{5,20,60}',
    'momentum_{5,10,20,60}',
    'rsi_14', 'macd', 'bollinger_bands',
    'atr_14', 'obv', 'stochastic'
]
```

### 3. Model Training

**PINN Loss Function:**
```
Total_Loss = Data_Loss + λ * Physics_Loss

where:
    Data_Loss = MSE(predictions, targets)
    Physics_Loss = λ_GBM * L_GBM + λ_OU * L_OU + λ_Langevin * L_Langevin
```

### 4. Evaluation

**Predictive Metrics:**
- RMSE, MAE, MAPE
- R² score
- Directional Accuracy

**Financial Metrics:**
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk)
- Maximum Drawdown
- Calmar Ratio
- Win Rate

### 5. Backtesting

Walk-forward validation with:
- Transaction costs (0.1%)
- Slippage (0.05%)
- Stop-loss (2%)
- Take-profit (5%)
- Position sizing (max 20% per position)

## 📊 Results

Results will be saved in `results/` directory after training.

Expected performance hierarchy (to be confirmed by experiments):
```
PINN > Transformer > BiLSTM > LSTM/GRU > Baselines
```

**Success Criteria:**
1. ✅ **Minimum**: PINN outperforms baseline on ≥1 metric (p < 0.05)
2. ⭐ **Target**: Positive risk-adjusted out-of-sample returns
3. 🎯 **Stretch**: Deployed web app with real-time predictions

## 🔧 Configuration

Edit `.env` or `src/utils/config.py`:

```python
# Data
START_DATE = "2014-01-01"
END_DATE = "2024-01-01"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100

# Physics Weights
LAMBDA_GBM = 0.1
LAMBDA_OU = 0.1
LAMBDA_LANGEVIN = 0.1

# Trading
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001
STOP_LOSS = 0.02
TAKE_PROFIT = 0.05
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📝 Reproducibility

All experiments are fully reproducible:

1. **Random Seeds**: Set via `RANDOM_SEED=42` in config
2. **Docker**: Complete environment in containers
3. **Data Versioning**: Timestamps recorded for all data fetches
4. **Model Checkpoints**: Saved with hyperparameters
5. **Logging**: All metrics logged to files and database

## 🤝 Contributing

This is a dissertation project. For questions or collaboration:

1. Open an issue
2. Submit a pull request
3. Contact: [Your Email]

## 📄 License

This project is for academic use only. See LICENSE file for details.

## 📚 References

### Academic Papers
1. Raissi et al. (2019) - Physics-informed neural networks
2. Black & Scholes (1973) - Option pricing model
3. Uhlenbeck & Ornstein (1930) - Mean-reverting processes
4. Diebold & Mariano (1995) - Forecast comparison

### Technical Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## 🙏 Acknowledgments

- PyTorch team for deep learning framework
- yfinance for financial data access
- TimescaleDB for time-series database
- Open-source community

---

**Built with ❤️ for Academic Research**

*Remember: This is NOT financial advice. Always do your own research and consult professionals before investing.*
