# Physics-Informed Neural Network (PINN) for Financial Forecasting

A comprehensive deep learning system that embeds quantitative finance equations (Black-Scholes, Ornstein-Uhlenbeck, Langevin dynamics, Geometric Brownian Motion) directly into neural network loss functions for improved financial forecasting and trading.

## Disclaimer

**THIS IS FOR ACADEMIC RESEARCH ONLY - NOT FINANCIAL ADVICE**

- This system is a dissertation research project
- NOT investment advice or recommendations
- Simulation only - no real trading functionality
- Past performance does not guarantee future results
- Always consult qualified financial advisors before making investment decisions
- The authors assume no liability for any financial losses

## Data Integrity Policy

**NO MOCK DATA ALLOWED IN PRODUCTION**

This project enforces strict data integrity:

- **NO synthetic/mock data** in production or evaluation
- **NO demo mode simulations** for model training or predictions
- **ALL data** must come from real sources (Yahoo Finance, Alpha Vantage, or database)
- **ALL models** must be trained with real market data
- Mock data methods exist only for unit testing and are disabled by default

If you see warnings about "demo mode" or "synthetic data", the system is misconfigured. See `TO-DO.md` for the full policy.

## Project Overview

This project implements a Physics-Informed Neural Network (PINN) framework that:

1. **Embeds Financial Physics** into neural network training via custom loss functions
2. **Compares Multiple Architectures**: LSTM, GRU, Transformer, and PINN models
3. **Implements Full Trading Pipeline**: Data → Training → Backtesting → Visualization
4. **Provides Modern Web Interface**: React frontend with FastAPI backend
5. **Ensures Reproducibility**: Docker containers, random seeds, comprehensive logging

### Key Physics Constraints

| Equation | Formula | Description |
|----------|---------|-------------|
| Geometric Brownian Motion | `dS = μS·dt + σS·dW` | Price drift with volatility |
| Black-Scholes PDE | `∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0` | Option pricing |
| Ornstein-Uhlenbeck | `dX = θ(μ-X)dt + σdW` | Mean reversion |
| Langevin Dynamics | `dX = -γ∇U(X)dt + √(2γT)dW` | Momentum modeling |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)                 │
│     Dashboard │ Predictions │ Backtesting │ Training │ ...      │
└───────────────────────────────┬─────────────────────────────────┘
                                │ REST API + WebSocket
┌───────────────────────────────▼─────────────────────────────────┐
│                       Backend (FastAPI)                          │
│          Routes │ Services │ Schemas │ WebSocket Handlers        │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Python Imports
┌───────────────────────────────▼─────────────────────────────────┐
│                      ML Core (src/)                              │
│     PINN Models │ Training │ Evaluation │ Data Processing        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       Data Layer                                 │
│         PostgreSQL + TimescaleDB │ Model Weights │ Results       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ with TimescaleDB (optional)
- CUDA-capable GPU (optional, for faster training)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Dissertaion-Project

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### Running the Application

**Option 1: Full Stack (Recommended)**

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python run.py
# API at http://localhost:8000
# Docs at http://localhost:8000/docs

# Terminal 2: Frontend
cd frontend
npm run dev
# App at http://localhost:5173
```

**Option 2: Streamlit Dashboard (Legacy)**

```bash
streamlit run src/web/app.py
```

## Project Structure

```
Dissertaion-Project/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   ├── core/           # Database, exceptions
│   │   ├── schemas/        # Pydantic models
│   │   └── services/       # Business logic
│   ├── requirements.txt
│   └── run.py
│
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Dashboard pages (14 total)
│   │   ├── hooks/          # React Query hooks
│   │   ├── services/       # API client
│   │   ├── stores/         # Zustand state
│   │   └── types/          # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
├── src/                    # ML Core Library
│   ├── data/               # Data fetching & processing
│   ├── models/             # PINN, LSTM, Transformer
│   ├── training/           # Training pipelines
│   ├── evaluation/         # Metrics & backtesting
│   ├── trading/            # Trading agent
│   └── web/                # Streamlit dashboards (legacy)
│
├── docs/                   # Documentation
│   ├── README.md           # Documentation index
│   ├── architecture.md     # System design
│   ├── api-reference.md    # API documentation
│   ├── setup.md            # Setup guide
│   ├── deployment.md       # Deployment guide
│   └── models.md           # Model documentation
│
├── Models/                 # Saved model weights
├── results/                # Evaluation results
├── tests/                  # Test suite
└── README.md               # This file
```

## Features

### Web Interface (React)

| Page | Description |
|------|-------------|
| Dashboard | Overview with key metrics and charts |
| PINN Analysis | Physics parameters, loss curves |
| Model Comparison | Side-by-side model evaluation |
| Predictions | Price forecasting with confidence intervals |
| Backtesting | Historical strategy performance |
| Training | Real-time training with WebSocket updates |
| Monte Carlo | Risk simulation and VaR analysis |
| Metrics | Financial and ML metrics calculator |

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/models/` | List available models |
| `POST /api/predictions/predict` | Run prediction |
| `POST /api/training/start` | Start training job |
| `WS /api/training/ws/{job_id}` | Real-time training updates |
| `POST /api/backtest/run` | Run backtest |
| `GET /api/metrics/financial` | Get financial metrics |
| `POST /api/monte-carlo/simulate` | Monte Carlo simulation |

### ML Models

| Model | Type | Key Features |
|-------|------|--------------|
| PINN GBM | Physics | Geometric Brownian Motion constraint |
| PINN OU | Physics | Mean reversion constraint |
| PINN GBM+OU | Physics | Combined trend + mean reversion |
| PINN Black-Scholes | Physics | Option pricing PDE |
| LSTM | Baseline | Standard sequence model |
| BiLSTM | Baseline | Bidirectional context |
| GRU | Baseline | Gated recurrent unit |
| Transformer | Baseline | Attention mechanism |

### Evaluation Metrics

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
- Profit Factor

## Configuration

### Backend (.env)

```env
DATABASE_URL=postgresql://user:password@localhost:5432/pinn
MODELS_DIR=../Models
RESULTS_DIR=../results
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Development

### Backend

```bash
cd backend
uvicorn app.main:app --reload  # Auto-reload
pytest tests/ -v                # Run tests
mypy app/                       # Type checking
```

### Frontend

```bash
cd frontend
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Linting
```

## Testing

```bash
# ML Core tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src

# Specific test file
pytest tests/test_models.py -v
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Setup Guide](docs/setup.md) - Installation instructions
- [Architecture](docs/architecture.md) - System design
- [API Reference](docs/api-reference.md) - Complete API docs
- [Model Documentation](docs/models.md) - PINN model details
- [Deployment](docs/deployment.md) - Production deployment

## Reproducibility

All experiments are fully reproducible:

1. **Random Seeds**: Set via `RANDOM_SEED=42` in config
2. **Docker**: Complete environment in containers
3. **Data Versioning**: Timestamps recorded for all data fetches
4. **Model Checkpoints**: Saved with hyperparameters
5. **Logging**: All metrics logged to files and database

## References

### Academic Papers
1. Raissi et al. (2019) - Physics-informed neural networks
2. Black & Scholes (1973) - Option pricing model
3. Uhlenbeck & Ornstein (1930) - Mean-reverting processes
4. Diebold & Mariano (1995) - Forecast comparison

### Technical Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TimescaleDB Documentation](https://docs.timescale.com/)

## License

This project is for academic use only. See LICENSE file for details.

---

**Built for Academic Research**

*Remember: This is NOT financial advice. Always do your own research and consult professionals before investing.*
