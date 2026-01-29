# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **Physics-Informed Neural Network (PINN)** system for financial forecasting - a dissertation research project that embeds quantitative finance equations (Black-Scholes PDE, Ornstein-Uhlenbeck, Langevin dynamics, Geometric Brownian Motion) into neural network loss functions for improved stock price prediction and trading simulation.

**⚠️ CRITICAL: This is ACADEMIC RESEARCH ONLY - not financial advice. All code changes must maintain prominent disclaimers.**

## Common Development Commands

### Setup
```bash
# Initial setup
./setup.sh

# Activate virtual environment (ALWAYS do this first)
source venv/bin/activate
```

### Data Operations
```bash
# Fetch financial data (10 years S&P 500)
python -m src.data.fetcher

# Fetch specific number of tickers
python main.py fetch-data --num-tickers 10

# Force refresh data
python main.py fetch-data --force-refresh
```

### Training Models
```bash
# Train specific model
python -m src.training.train --model {lstm|gru|bilstm|transformer|pinn} --epochs 100

# Quick training (20 epochs)
python -m src.training.train --model pinn --epochs 20

# Using main.py wrapper
python main.py train --model pinn --epochs 50
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py -v

# Run slow tests separately
pytest -m "not slow" tests/
```

### Web Interface
```bash
# Launch Streamlit dashboard
streamlit run src/web/app.py

# Specify port
streamlit run src/web/app.py --server.port 8501

# Using main.py
python main.py web
```

### Database Operations
```bash
# Start TimescaleDB (Docker required)
docker-compose up -d timescaledb

# Check database status
docker-compose ps

# Stop database
docker-compose down

# Full stack (all services)
docker-compose up --build
```

### Quick Workflows
```bash
# Interactive menu system
./run.sh

# Full pipeline (fetch → train → visualize)
python main.py full-pipeline --quick
```

## Architecture Overview

### Core Pipeline Flow
```
Data Fetching → Preprocessing → Feature Engineering → Model Training → Evaluation → Backtesting → Web Dashboard
     ↓              ↓                  ↓                   ↓               ↓            ↓             ↓
  yfinance/    TimescaleDB/      Technical          PINN/LSTM/      Metrics:     Trading      Streamlit UI
  AlphaVantage   Parquet         Indicators         Transformer     RMSE, MAE,   Simulation   + Portfolio
                                 (RSI, MACD,         + Physics       Sharpe,      + Risk       Visualization
                                 Bollinger,          Constraints     Sortino      Management
                                 ATR, etc.)
```

### Module Structure

**`src/data/`** - Data pipeline
- `fetcher.py`: Downloads stock data from yfinance/Alpha Vantage APIs
- `preprocessor.py`: Feature engineering, normalization, stationarity testing (ADF)
- `dataset.py`: PyTorch datasets (`FinancialDataset`, `PhysicsAwareDataset`)

**`src/models/`** - Neural network architectures
- `baseline.py`: LSTM, GRU, BiLSTM models
- `transformer.py`: Transformer encoder for sequence modeling
- `pinn.py`: **Core contribution** - Physics-Informed Neural Network
  - Embeds finance equations into loss function
  - `PhysicsLoss` class implements GBM, Black-Scholes PDE, OU process, Langevin dynamics
  - Total loss = Data loss (MSE) + λ × Physics loss

**`src/training/`** - Training infrastructure
- `trainer.py`: Training loop with early stopping, checkpointing, learning rate scheduling
- `train.py`: Main training script with data preparation and model instantiation

**`src/evaluation/`** - Metrics and backtesting
- `metrics.py`: Predictive metrics (RMSE, MAE, R²) + financial metrics (Sharpe, Sortino, max drawdown)
- `backtester.py`: Walk-forward validation, transaction costs, slippage modeling

**`src/trading/`** - Trading agent
- `agent.py`: Signal generation, portfolio management, risk controls (stop-loss, take-profit)

**`src/utils/`** - Shared utilities
- `config.py`: Pydantic-based configuration management (reads from `.env`)
- `database.py`: TimescaleDB interface using SQLAlchemy
- `logger.py`: Logging setup (console + file)
- `reproducibility.py`: Random seed control, device selection

**`src/web/`** - Web interface
- `app.py`: Streamlit dashboard for visualization and model inference

### Key Architectural Patterns

**Physics-Informed Loss Function**:
The PINN model's core innovation is in `src/models/pinn.py`:
```python
Total_Loss = Data_Loss + λ_GBM * L_GBM + λ_OU * L_OU + λ_Langevin * L_Langevin + λ_BS * L_BS
```
Each physics loss term penalizes violations of financial equations (e.g., GBM drift-diffusion, mean reversion).

**Temporal Data Splitting**:
In `src/data/preprocessor.py`, data is split temporally (not randomly) to prevent look-ahead bias:
- Train: 70% (earliest)
- Validation: 15% (middle)
- Test: 15% (latest)

**Per-Ticker Normalization**:
Each stock ticker has its own StandardScaler to handle different price scales. Scalers are stored in a dict keyed by ticker symbol.

**Walk-Forward Validation**:
Backtesting uses expanding/sliding windows to simulate realistic trading conditions where you only know past data.

### Configuration Management

All hyperparameters are in `.env` or `src/utils/config.py`:
- Data: `START_DATE`, `END_DATE`, `TRAIN_RATIO`, `SEQUENCE_LENGTH` (default: 60 timesteps)
- Training: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `EARLY_STOPPING_PATIENCE`
- Physics weights: `LAMBDA_GBM`, `LAMBDA_OU`, `LAMBDA_LANGEVIN`, `LAMBDA_BS`
- Trading: `INITIAL_CAPITAL`, `TRANSACTION_COST`, `STOP_LOSS`, `TAKE_PROFIT`

Configuration is singleton-pattern accessed via `get_config()`.

### Database Schema

TimescaleDB stores time-series data in hypertables:
- `finance.stock_prices`: OHLCV data with time-series indexing
- Efficient bulk inserts using PostgreSQL COPY command
- Fallback to Parquet files in `data/parquet/` for offline access

### Device Selection

Code automatically handles CUDA/CPU:
```python
device = torch.device(config.training.device)
if not torch.cuda.is_available() and device.type == 'cuda':
    device = torch.device('cpu')
```
CUDA recommended for training transformers/PINNs (much faster).

## Development Guidelines

### Adding New Models
1. Create model class in `src/models/` inheriting from `nn.Module`
2. Implement `forward()` method
3. For PINN-style models, implement `compute_loss()` method
4. Register in `create_model()` function in `src/training/train.py`
5. Add corresponding tests in `tests/test_models.py`

### Adding New Features
1. Add feature engineering in `DataPreprocessor.calculate_technical_indicators()`
2. Update `feature_cols` list in `src/training/train.py`
3. Ensure normalization is applied per-ticker
4. Test stationarity if feature is a return/change metric

### Adding Physics Constraints
1. Add new residual method to `PhysicsLoss` class in `src/models/pinn.py`
2. Add lambda weight to `TrainingConfig` in `src/utils/config.py`
3. Update `forward()` method in `PhysicsLoss` to include new term
4. Add corresponding test in `tests/test_models.py`

### Reproducibility
All experiments must be reproducible:
- Set `RANDOM_SEED` in `.env` (default: 42)
- Seeds are set in `src/utils/reproducibility.py` for: PyTorch, NumPy, Python random
- Model checkpoints saved with hyperparameters in `checkpoints/`
- Logs stored in `logs/` directory

### Code Quality Tools
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Important Implementation Notes

### PINN Training
- PINN models require additional metadata (prices, returns, volatilities) beyond standard datasets
- Use `PhysicsAwareDataset` instead of `FinancialDataset` when training PINNs
- Physics loss can be disabled via `enable_physics=False` for ablation studies

### Model Outputs
- LSTM/GRU models return `(predictions, hidden_state)` tuple
- Transformer/PINN models return just `predictions` tensor
- Handle this difference in training loop (see `trainer.py` line 195-198)

### Data Leakage Prevention
- NEVER normalize before temporal splitting
- NEVER use test set for scaler fitting
- ONLY use train set statistics for all normalization
- Features like RSI/MACD use forward-looking windows - be careful with sequence boundaries

### Checkpoint Naming Convention
```
checkpoints/{model_name}_epoch{epoch}_val{val_loss:.4f}.pt
```
Checkpoints include: model state_dict, optimizer state, epoch, config, feature columns.

### API Rate Limits
- yfinance: No API key required but has rate limits (~2000 requests/hour)
- Alpha Vantage: 5 requests/minute (free tier) - used as backup
- Implement exponential backoff in `fetcher.py` if rate-limited

### Docker Environment Variables
When using Docker, database connection uses container service names:
- `POSTGRES_HOST=timescaledb` (not localhost)
- Ports are mapped: container 5432 → host 5432

## Testing Strategy

- **Unit tests**: Individual components (models, preprocessor, metrics)
- **Integration tests**: End-to-end pipeline tests marked with `@pytest.mark.integration`
- **Slow tests**: Full training runs marked with `@pytest.mark.slow`
- Run fast tests during development: `pytest -m "not slow" tests/`

## Evaluation Metrics Priority

When comparing models, prioritize:
1. **Out-of-sample Sharpe Ratio** (risk-adjusted returns)
2. **Maximum Drawdown** (risk control)
3. **Directional Accuracy** (can we predict up/down correctly?)
4. **RMSE/MAE** (prediction accuracy)
5. **Diebold-Mariano test** (statistical significance of forecast differences)

RMSE alone is insufficient for financial forecasting - focus on risk-adjusted profitability metrics.

## Common Pitfalls

- **Don't** train on non-stationary data without differencing (check ADF test results)
- **Don't** use future data in technical indicators (ensure proper rolling windows)
- **Don't** forget to activate venv before running commands
- **Don't** commit `.env` file (contains API keys)
- **Don't** assume CUDA availability (always check device in code)
- **Don't** mix checkpoint formats between model types (LSTM ≠ PINN format)

## Quick Reference

### File Locations
- Data: `data/raw/`, `data/processed/`, `data/parquet/`
- Models: `checkpoints/model_name_*.pt`
- Logs: `logs/training_YYYYMMDD_HHMMSS.log`
- Results: Saved in checkpoint metadata

### Important Constants
- Sequence length: 60 timesteps (configurable)
- Forecast horizon: 1 step ahead
- Trading days per year: 252
- Risk-free rate: 2% annual (configurable in `pinn.py`)

### Default Tickers
Top 50 S&P 500 constituents by market cap (configurable in `config.py`). Can be subset for faster experimentation.
