# PINN Financial Forecasting - Consolidated Documentation

> **Generated:** 2026-03-03
> **Purpose:** Single consolidated reference for all project documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start Guide](#2-quick-start-guide)
3. [System Architecture](#3-system-architecture)
4. [Model Documentation](#4-model-documentation)
5. [API Reference](#5-api-reference)
6. [Setup Guide](#6-setup-guide)
7. [Deployment Guide](#7-deployment-guide)
8. [Development Guidelines](#8-development-guidelines)
9. [Visualization Framework](#9-visualization-framework)
10. [Physics Equations](#10-physics-equations)
11. [Performance Optimizations](#11-performance-optimizations)
12. [Change Log Summary](#12-change-log-summary)

---

# 1. Project Overview

## Physics-Informed Neural Network (PINN) for Financial Forecasting

A comprehensive deep learning system that embeds quantitative finance equations (Black-Scholes, Ornstein-Uhlenbeck, Langevin dynamics, Geometric Brownian Motion) directly into neural network loss functions for improved financial forecasting and trading.

### Disclaimer

**THIS IS FOR ACADEMIC RESEARCH ONLY - NOT FINANCIAL ADVICE**

- This system is a dissertation research project
- NOT investment advice or recommendations
- Simulation only - no real trading functionality
- Past performance does not guarantee future results
- Always consult qualified financial advisors before making investment decisions
- The authors assume no liability for any financial losses

### Data Integrity Policy

**NO MOCK DATA ALLOWED IN PRODUCTION**

This project enforces strict data integrity:

- **NO synthetic/mock data** in production or evaluation
- **NO demo mode simulations** for model training or predictions
- **ALL data** must come from real sources (Yahoo Finance, Alpha Vantage, or database)
- **ALL models** must be trained with real market data
- Mock data methods exist only for unit testing and are disabled by default

### Key Physics Constraints

| Equation | Formula | Description |
|----------|---------|-------------|
| Geometric Brownian Motion | `dS = μS·dt + σS·dW` | Price drift with volatility |
| Black-Scholes PDE | `∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0` | Option pricing |
| Ornstein-Uhlenbeck | `dX = θ(μ-X)dt + σdW` | Mean reversion |
| Langevin Dynamics | `dX = -γ∇U(X)dt + √(2γT)dW` | Momentum modeling |

### Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.11+, Pydantic |
| ML Core | PyTorch, NumPy, Pandas |
| Database | PostgreSQL + TimescaleDB |
| Charts | Recharts |
| State | Zustand, React Query |

---

# 2. Quick Start Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ with TimescaleDB (optional)
- CUDA-capable GPU (optional, for faster training)

## Installation

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

## Running the Application

**Full Stack (Recommended)**

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

## Development Tools

### Backend Development

```bash
cd backend
uvicorn app.main:app --reload  # Auto-reload
pytest tests/ -v                # Run tests
mypy app/                       # Type checking
ruff check app/                 # Linting
```

### Frontend Development

```bash
cd frontend
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Linting
npm run typecheck  # Type checking
```

---

# 3. System Architecture

## Overview

The PINN Financial Forecasting system follows a three-tier architecture with a React frontend, FastAPI backend, and PyTorch-based ML core.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Pages   │  │  Charts  │  │  Hooks   │  │  State (Zustand) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                              │                                   │
│                    HTTP/REST & WebSocket                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (FastAPI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Routes  │  │ Schemas  │  │ Services │  │   WebSocket      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                              │                                   │
│                    Python Imports                                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML Core (src/)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Models  │  │ Training │  │Evaluation│  │      Data        │ │
│  │  (PINN)  │  │ Pipeline │  │ Metrics  │  │   Processing     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │   PostgreSQL +   │  │  Model Weights   │  │   Results/     │ │
│  │   TimescaleDB    │  │    (Models/)     │  │   Metrics      │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Dissertaion-Project/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   ├── core/           # Database, exceptions
│   │   ├── schemas/        # Pydantic models
│   │   └── services/       # Business logic
│   └── run.py              # Entry point
│
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   │   ├── charts/     # Recharts wrappers
│   │   │   ├── common/     # Shared components
│   │   │   ├── layout/     # Layout components
│   │   │   └── ui/         # shadcn/ui primitives
│   │   ├── hooks/          # React Query hooks
│   │   ├── pages/          # Route pages
│   │   ├── services/       # API client
│   │   ├── stores/         # Zustand stores
│   │   └── types/          # TypeScript types
│   └── vite.config.ts
│
├── src/                    # ML Core Library
│   ├── data/               # Data fetching/processing
│   ├── models/             # PINN implementations
│   ├── training/           # Training pipelines
│   ├── evaluation/         # Metrics & backtesting
│   ├── trading/            # Trading strategies
│   └── utils/              # Utilities
│
├── docs/                   # Documentation
├── tests/                  # Test suite
├── Models/                 # Saved model weights
└── results/                # Evaluation results
```

## Component Details

### Frontend Components

| Directory | Purpose |
|-----------|---------|
| `components/charts/` | Recharts wrappers for financial visualizations |
| `components/ui/` | shadcn/ui base components (Button, Card, etc.) |
| `components/common/` | Shared components (MetricCard, LoadingSpinner) |
| `components/layout/` | Page layout (Sidebar, Header) |
| `pages/` | 14 dashboard pages |
| `hooks/` | React Query data fetching hooks |
| `stores/` | Zustand global state management |
| `services/` | Axios API client modules |
| `types/` | TypeScript interfaces |

### Backend Components

| Directory | Purpose |
|-----------|---------|
| `api/routes/` | FastAPI route handlers |
| `schemas/` | Pydantic request/response models |
| `services/` | Business logic wrapping ML core |
| `core/` | Database connection, exceptions |

### ML Core Components

| Directory | Purpose |
|-----------|---------|
| `models/` | PINN, LSTM, Transformer implementations |
| `training/` | Training loops, curriculum learning |
| `evaluation/` | Metrics, backtesting, Monte Carlo |
| `data/` | Data fetching, preprocessing |
| `trading/` | Trading agent, position sizing |

## Key Design Decisions

1. **Service Layer Pattern**: Backend services wrap ML core, providing clean API for routes
2. **React Query**: Server state caching and synchronization
3. **Zustand**: Lightweight client-side state (theme, selections)
4. **Model Registry**: Centralized model loading and caching
5. **WebSocket for Training**: Real-time progress without polling

---

# 4. Model Documentation

## PINN (Physics-Informed Neural Network) Models

Physics-Informed Neural Networks combine traditional neural network learning with physics-based constraints derived from stochastic differential equations (SDEs).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PINN Architecture                       │
│                                                              │
│  Input Features (X)         Physics Constraints (Φ)         │
│        │                            │                        │
│        ▼                            ▼                        │
│  ┌──────────────┐           ┌──────────────┐                │
│  │   Encoder    │           │   Physics    │                │
│  │  (FC Layers) │           │   Network    │                │
│  └──────┬───────┘           └──────┬───────┘                │
│         │                          │                        │
│         └────────────┬─────────────┘                        │
│                      ▼                                      │
│               ┌──────────────┐                              │
│               │    Fusion    │                              │
│               │    Layer     │                              │
│               └──────┬───────┘                              │
│                      │                                      │
│                      ▼                                      │
│               ┌──────────────┐                              │
│               │   Output     │ ──► Predictions (ŷ)         │
│               │   Layer      │                              │
│               └──────────────┘                              │
│                                                              │
│  Loss = L_data + λ * L_physics                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Variants

### 1. PINN Baseline (`pinn_baseline`)
Standard PINN without specific physics equations. Uses learnable physics parameters.

### 2. PINN GBM - Geometric Brownian Motion (`pinn_gbm`)
Models stock prices following GBM: `dS = μS·dt + σS·dW`

**Learned Parameters:**
- μ (drift): Expected return rate
- σ (volatility): Price volatility

### 3. PINN Ornstein-Uhlenbeck (`pinn_ou`)
Models mean-reverting behavior: `dX = θ(μ - X)dt + σdW`

**Learned Parameters:**
- θ (theta): Mean reversion speed
- μ (mu): Long-term mean
- σ (sigma): Volatility

### 4. PINN GBM + OU Combined (`pinn_gbm_ou`)
Combines trend-following (GBM) with mean reversion (OU):
`dS = μS·dt + σS·dW + θ(γ - log(S))·dt`

### 5. PINN Black-Scholes (`pinn_black_scholes`)
Based on the Black-Scholes PDE:
`∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0`

### 6. PINN Global (`pinn_global`)
Multi-physics model combining multiple SDEs with adaptive weighting.

## Baseline Models

| Model | RNN Layers | Hidden Dim | Bidirectional | FC Output |
|-------|------------|------------|---------------|-----------|
| LSTM | 2 | 128 | No | 128 → 1 |
| GRU | 2 | 128 | No | 128 → 1 |
| BiLSTM | 2 | 128 | Yes | 256 → 1 |
| Transformer | 2 | 64 (d_model) | N/A | Custom |

## Model Architecture Mapping

| Model Key | Class | Type | Physics Constraints |
|-----------|-------|------|---------------------|
| lstm | LSTMModel | baseline | None |
| gru | GRUModel | baseline | None |
| bilstm | LSTMModel (bidirectional) | baseline | None |
| attention_lstm | LSTMModel | baseline | None |
| transformer | TransformerModel | baseline | None |
| baseline_pinn | PINNModel | pinn | λ_gbm=0, λ_ou=0, λ_bs=0 |
| gbm | PINNModel | pinn | λ_gbm=0.1 (trend) |
| ou | PINNModel | pinn | λ_ou=0.1 (mean-reversion) |
| black_scholes | PINNModel | pinn | λ_bs=0.1 (no-arbitrage) |
| gbm_ou | PINNModel | pinn | λ_gbm=0.05, λ_ou=0.05 |
| global | PINNModel | pinn | λ_gbm=0.05, λ_ou=0.05, λ_bs=0.03 |
| stacked | StackedPINN | advanced | Physics encoder + parallel LSTM/GRU |
| residual | ResidualPINN | advanced | Base + physics correction |

## Training Configuration

### Loss Function

```python
L_total = L_data + λ * L_physics + α * L_reg

where:
- L_data = MSE(y_pred, y_true)
- L_physics = physics constraint violation
- L_reg = L2 regularization
- λ = physics weight (tunable)
- α = regularization weight
```

### Hyperparameters

| Parameter | Default | Range |
|-----------|---------|-------|
| learning_rate | 0.001 | 1e-4 to 1e-2 |
| batch_size | 32 | 16 to 128 |
| epochs | 100 | 50 to 500 |
| physics_weight | 0.1 | 0.01 to 1.0 |
| dropout | 0.1 | 0.0 to 0.3 |

## Evaluation Metrics

### ML Metrics
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of determination
- Directional Accuracy

### Financial Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor

### Physics Metrics
- Physics Loss: Constraint violation magnitude
- Parameter Stability: Variance of learned parameters
- SDE Residuals: Individual equation residuals

---

# 5. API Reference

Base URL: `http://localhost:8000`

API documentation is auto-generated at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## Data Endpoints

### GET /api/data/stocks
List available stock tickers.

### GET /api/data/stocks/{ticker}
Get historical stock data.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| ticker | path | Stock ticker symbol |
| start_date | query | Start date (YYYY-MM-DD) |
| end_date | query | End date (YYYY-MM-DD) |
| interval | query | Data interval (1d, 1h) |

### POST /api/data/fetch
Fetch new data from yfinance.

## Model Endpoints

### GET /api/models/
List all available models.

### GET /api/models/{model_key}
Get model details.

### GET /api/models/compare
Compare multiple models.

## Prediction Endpoints

### POST /api/predictions/predict
Run prediction.

**Request:**
```json
{
  "model_key": "pinn_gbm_ou",
  "ticker": "AAPL",
  "horizon": 5,
  "include_uncertainty": true
}
```

### GET /api/predictions/history
Get prediction history.

## Training Endpoints

### POST /api/training/start
Start training job.

**Request:**
```json
{
  "model_type": "pinn_gbm",
  "ticker": "AAPL",
  "epochs": 100,
  "learning_rate": 0.001,
  "batch_size": 32,
  "physics_weight": 0.1
}
```

### GET /api/training/status/{job_id}
Get training status.

### WebSocket /api/training/ws/{job_id}
Real-time training updates.

## Backtest Endpoints

### POST /api/backtest/run
Run backtest.

### GET /api/backtest/results/{id}
Get backtest results.

## Metrics Endpoints

### GET /api/metrics/financial
Get financial metrics (Sharpe, Sortino, Max Drawdown, etc.)

### GET /api/metrics/ml
Get ML metrics (RMSE, MAE, MAPE, R², Directional Accuracy)

### GET /api/metrics/physics
Get physics constraint metrics.

## Monte Carlo Endpoints

### POST /api/monte-carlo/simulate
Run Monte Carlo simulation.

## Error Responses

All endpoints return errors in this format:
```json
{
  "detail": {
    "message": "Model not found",
    "code": "MODEL_NOT_FOUND",
    "model_key": "invalid_model"
  }
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Schema mismatch |
| 500 | Internal Server Error |

---

# 6. Setup Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ with TimescaleDB extension
- Git

## Database Setup

```bash
# Install PostgreSQL and TimescaleDB
# macOS
brew install postgresql@14 timescaledb

# Ubuntu
sudo apt install postgresql-14 postgresql-14-timescaledb

# Create database
psql -U postgres
CREATE DATABASE pinn_forecasting;
\c pinn_forecasting
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

## Environment Configuration

**backend/.env:**
```env
DATABASE_URL=postgresql://user:password@localhost:5432/pinn_forecasting
MODELS_DIR=../Models
RESULTS_DIR=../results
SRC_DIR=../src
HOST=0.0.0.0
PORT=8000
DEBUG=true
CORS_ORIGINS=http://localhost:5173
```

**frontend/.env:**
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ../  # Install ML core as editable package
python run.py
```

Verify: Open http://localhost:8000/docs

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Verify: Open http://localhost:5173

## Troubleshooting

### Backend Issues

**ImportError: No module named 'src'**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Dissertaion-Project"
```

**Database connection error**
```bash
pg_isready
psql $DATABASE_URL
```

### Frontend Issues

**Module not found errors**
```bash
rm -rf node_modules package-lock.json
npm install
```

**API calls failing**
```bash
curl http://localhost:8000/api/models/
```

---

# 7. Deployment Guide

## Production Architecture

```
                    ┌─────────────┐
                    │   Nginx     │
                    │  (Reverse   │
                    │   Proxy)    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Frontend   │ │   Backend   │ │  Backend    │
    │   (Static)  │ │  (Uvicorn)  │ │  (Worker)   │
    └─────────────┘ └──────┬──────┘ └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  PostgreSQL │
                    │ +TimescaleDB│
                    └─────────────┘
```

## Docker Compose

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/pinn
      - MODELS_DIR=/app/Models
      - RESULTS_DIR=/app/results
    depends_on:
      - db
    volumes:
      - ./Models:/app/Models:ro
      - ./results:/app/results:ro

  db:
    image: timescale/timescaledb:latest-pg14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=pinn
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## Cloud Deployment

### AWS Deployment
1. **Frontend**: Deploy to S3 + CloudFront
2. **Backend**: Deploy to ECS or Lambda
3. **Database**: Use RDS with TimescaleDB

### GCP Deployment
1. **Frontend**: Deploy to Cloud Storage + Cloud CDN
2. **Backend**: Deploy to Cloud Run
3. **Database**: Use Cloud SQL

## Security Checklist

- [ ] Enable HTTPS
- [ ] Set secure CORS origins
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Validate all inputs
- [ ] Keep dependencies updated
- [ ] Enable security headers (CSP, HSTS)

---

# 8. Development Guidelines

## Critical Requirements

**IMPORTANT: Always work on the React frontend (`frontend/`) and FastAPI backend (`backend/`).
NEVER work on the Streamlit app (`src/web/`) unless explicitly requested.**

## Research Project - Model Integrity

**THIS IS A DISSERTATION/RESEARCH PROJECT. All models MUST be connected to their REAL neural network implementations.**

When working on model training or inference:
1. **NEVER use simulated/mock training** - All models must use actual PyTorch implementations
2. **Verify model connections** - Check that `HAS_SRC=True` in training_service.py
3. **Test ALL model types** - Every model (LSTM, GRU, Transformer, PINN variants) must create correctly
4. **Check backend logs** - Should show `[REAL TRAINING]` not simulated mode

### Verification Command

```bash
source backend/venv/bin/activate && python -c "
import sys; sys.path.insert(0, '.')
import torch
from pathlib import Path
from src.models.model_registry import ModelRegistry

registry = ModelRegistry(Path('.'))
test_input = torch.randn(2, 30, 5)

models = ['lstm', 'gru', 'bilstm', 'transformer',
          'baseline_pinn', 'gbm', 'ou', 'black_scholes', 'gbm_ou', 'global',
          'stacked', 'residual']

for m in models:
    model = registry.create_model(m, input_dim=5)
    if model:
        out = model(test_input)
        pred = out[0] if isinstance(out, tuple) else out
        is_pinn = hasattr(model, 'compute_loss')
        print(f'✓ {m}: {model.__class__.__name__} ({\"PINN\" if is_pinn else \"Baseline\"})')
    else:
        print(f'✗ {m}: FAILED')
"
```

### Training Mode Check
GET `/api/training/mode` should return `{"mode": "real", "using_real_models": true}`

## Documentation Requirements

**ALL changes, bug fixes, and new features MUST be documented in `DOCUMENT.md`.**

| Change Type | Required Documentation |
|-------------|------------------------|
| Bug fixes | Issue description, root cause, solution with code snippets |
| New features | Feature description, API endpoints, usage examples |
| Architecture changes | Before/after diagrams, rationale, files modified |
| Configuration changes | Old vs new values, impact on training/results |
| Training improvements | Parameters changed, expected impact, verification steps |

## Key Files

- Frontend entry: `frontend/src/App.tsx`
- Backend entry: `backend/run.py`
- API routes: `backend/app/api/routes/`
- React hooks: `frontend/src/hooks/`
- Services: `frontend/src/services/`
- **Model Registry**: `src/models/model_registry.py` - Central model creation
- **Training Service**: `backend/app/services/training_service.py` - Must use HAS_SRC=True
- **PINN Models**: `src/models/pinn.py` - Physics-informed neural networks

---

# 9. Visualization Framework

## Dissertation Visualization Framework Blueprint

### Core Forecast Accuracy Visualizations

#### 1. Predicted vs Realized Volatility
**Purpose:** Demonstrates the model's ability to track volatility dynamics over time.

**Validation patterns (good model):**
- Predicted closely tracks realized, especially during regime shifts
- No systematic over/under-prediction
- Quick response to volatility changes

#### 2. Multi-Horizon Forecasts (1-day, 5-day, 20-day)
**Purpose:** Shows how forecast quality degrades with prediction horizon.

**What to look for:**
- Gradual, smooth degradation with horizon
- PINN should degrade slower than pure ML (physics constraints help)
- R² should remain meaningful even at 20-day horizon

#### 3. Rolling Forecast Error Plots
**Components:**
- Rolling bias (mean error over window)
- Rolling MAE (mean absolute error)
- Rolling RMSE
- ±2σ confidence bands

### Loss and Calibration Diagnostics

#### QLIKE for Volatility Forecasting
**Formula:**
```
QLIKE = (1/T) Σ [σ²_realized / σ²_predicted - ln(σ²_realized / σ²_predicted) - 1]
```

**Why QLIKE is preferred:**
1. Scale-independent (unlike MSE)
2. Robust to heteroskedasticity
3. Consistent ranking across volatility regimes

### Economic Performance Graphs

#### Volatility-Targeting Strategy
```python
def volatility_targeting_returns(returns, predicted_vol, target_vol=0.15):
    target_daily = target_vol / np.sqrt(252)
    lagged_vol = np.roll(predicted_vol, 1)  # Use t-1 forecast
    weights = np.clip(target_daily / lagged_vol, 0.25, 2.0)
    return weights * returns
```

**Economic Significance Thresholds:**
- Sharpe Ratio > 0.5: Reasonable
- Sharpe Ratio > 1.0: Good
- Sharpe Ratio > 2.0: Excellent (verify for overfitting)

### Statistical Tests

#### Diebold-Mariano Test
**Purpose:** Test whether forecast accuracy differences are statistically significant.

**Test statistic:**
```
DM = d̄ / √(Var(d)/T)
```

**Interpretation:**
- |DM| > 1.96: Significant difference at 5% level
- Positive DM: Model 2 is better
- Negative DM: Model 1 is better

### Key Formulas Reference

| Metric | Formula | Notes |
|--------|---------|-------|
| QLIKE | `Σ[σ²_r/σ²_p - ln(σ²_r/σ²_p) - 1]/T` | Preferred for volatility |
| M-Z R² | `R²` from `σ²_r = α + β·σ²_p + ε` | Forecast efficiency |
| Sharpe | `(μ - r_f) / σ` | Risk-adjusted return |
| Max DD | `min(V_t - max(V_0..V_t)) / max(V_0..V_t)` | Worst drawdown |
| VaR | `-z_α · σ̂_t` | Value at Risk |
| DM | `d̄ / √(Var(d)/T)` | Compare forecasts |

---

# 10. Physics Equations

## Alternative Physics Equations Considered

- **Heston Stochastic Volatility:** Not adopted due to lack of implied vol surface inputs; would require option data and adds two extra latent factors.
- **Variance Gamma / Jump-Diffusion:** Rejected for dissertation scope; jump calibration unstable on daily equity closes.
- **Rough Volatility (RFSV):** Excluded because fractional Brownian motion terms are not supported in current autograd implementation.

## Current Choice Justification

Current choice of GBM, OU, Black-Scholes, and Langevin covers:
- Diffusion with drift
- Mean reversion
- Pricing PDE consistency
- Stochastic friction

Empirical validation favored OU overwhelmingly, justifying its prominence.

## Physics Loss Equations

1. **GBM Loss:** `L_gbm = |dS/dt - μS - σS·ε|²` (Geometric Brownian Motion)
2. **OU Loss:** `L_ou = |dS/dt - θ(μ - S)|²` (Ornstein-Uhlenbeck mean-reversion)
3. **Black-Scholes Loss:** `L_bs = |½σ²S²∂²V/∂S² + rS∂V/∂S - rV|²` (No-arbitrage PDE)
4. **Langevin Loss:** `L_langevin = |m·d²S/dt² + γ·dS/dt - F(S) - √(2γkT)·ξ|²`

**Total Loss:** `L = L_data + λ_gbm·L_gbm + λ_bs·L_bs + λ_ou·L_ou + λ_langevin·L_langevin`

---

# 11. Performance Optimizations

## Checkpoint Loading Performance Fixes

### Issues Fixed

1. **Serial Path Existence Checks**: Model registry was checking up to 5 paths for each of ~20 models, resulting in ~100 synchronous `path.exists()` calls.
   - **Solution**: Single glob scan with caching (60-second TTL)

2. **No Caching in Dashboard Classes**: `load_all_results()` loaded JSON files on every call.
   - **Solution**: Streamlit-cached loading functions with 5-minute TTL

3. **Cache TTL Mismatch**: Predictions cached for 5 minutes while metrics cached for 60 seconds.
   - **Solution**: Aligned all cache TTLs to 5 minutes

### Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| Initial page load | 500-800ms | 100-200ms |
| Subsequent interactions | 300-500ms | <50ms (cached) |
| Filesystem calls per load | ~100 | 2-3 (glob patterns) |
| Cache hit rate | 0% | >90% |

## Training Optimizations

| Optimization | Before | After | Overhead Reduction |
|--------------|--------|-------|-------------------|
| Batch callback interval | 10 batches | 50 batches | 5x fewer callbacks |
| WS polling (regular) | 0.5s | 2.0s | 4x fewer polls |
| WS polling (batch) | 0.3s | 2.0s | 6.7x fewer polls |
| Trainer logging | 100 batches | 200 batches | 2x fewer logs |

## Research-Grade Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 100 | Full training for fair comparison |
| batch_size | 16 | Smaller batches for better gradient estimates |
| learning_rate | 0.0005 | Lower for stability with deep models |
| sequence_length | 180 | ~6 months of market history |
| hidden_dim | 512 | Deep model for complex patterns |
| num_layers | 4 | Multiple layers for hierarchical features |
| dropout | 0.15 | Moderate dropout for deep models |
| weight_decay | 1e-4 | L2 regularization |
| research_mode | true | Disables early stopping |

---

# 12. Change Log Summary

## Major Features & Fixes (from DOCUMENT.md)

| Section | Date | Description |
|---------|------|-------------|
| 1-3 | - | Research Mode, Metrics Recalculation, Frontend-Backend Audit |
| 9 | 2026-02-10 | Fixes Applied (Predictions, Training, Backtesting, etc.) |
| 10 | 2026-02-13 | Research Mode & Fair Model Comparison |
| 11 | 2026-02-15 | Real-Time Trading Agent |
| 12 | 2026-02-17 | PINN Architecture Documentation |
| 13 | 2026-02-20 | Research-Grade Training Bug Fixes |
| 14 | 2026-02-20 | Training Optimization & Polling Fixes |
| 15 | 2026-02-20 | Web App Training Alignment with Terminal |
| 16 | 2026-02-20 | Trainer tqdm Fallback Fix |
| 17 | 2026-02-20 | Batch-Level Progress Updates for Frontend |
| 18 | 2026-02-20 | Dissertation-Grade Reproducibility |
| 19 | 2026-02-23 | Performance Optimizations |
| 20 | 2026-02-23 | Model Storage & Results Persistence |
| 21 | 2026-02-23 | Docker Containerization |
| 22 | 2026-02-23 | Advanced PINN Architectures Web Integration |
| 23 | 2026-02-24 | Regime-Switching Monte Carlo Framework |
| 24 | 2026-02-25 | Evaluation and Reproducibility Infrastructure |
| 25 | 2026-02-25 | PINN Training Stability and Correctness |
| 26 | 2026-02-26 | Data Pipeline and Reporting Infrastructure |
| 27 | 2026-02-26 | Trading Strategy Evaluation and CI Pipeline |
| 28 | 2026-02-26 | Loss Functions Module and Unit Tests |
| 29 | 2026-02-26 | Evaluation Infrastructure Expansion |
| 30 | 2026-02-26 | PINN Correctness and Stability Test Suite |
| 31 | 2026-02-26 | Magic Numbers Elimination - Constants Module |
| 32 | 2026-02-27 | Thread Safety and Memory Management Fixes |
| 33 | 2026-03-01 | Volatility Forecasting Framework |
| 34 | 2026-03-03 | Codebase Refactoring (Dead code removal, archive Streamlit) |

---

# References

## Academic Papers
1. Raissi et al. (2019) - Physics-informed neural networks
2. Black & Scholes (1973) - Option pricing model
3. Uhlenbeck & Ornstein (1930) - Mean-reverting processes
4. Diebold & Mariano (1995) - Forecast comparison
5. Patton, A.J. (2011) - Volatility Forecast Comparison
6. Hansen, Lunde, & Nason (2011) - The Model Confidence Set
7. Bailey & Lopez de Prado (2014) - The Deflated Sharpe Ratio

## Technical Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TimescaleDB Documentation](https://docs.timescale.com/)

---

*This consolidated documentation was generated from the following source files:*
- `README.md`
- `CLAUDE.md`
- `DOCUMENT.md` (summary of 34 sections)
- `docs/README.md`
- `docs/architecture.md`
- `docs/api-reference.md`
- `docs/deployment.md`
- `docs/models.md`
- `docs/setup.md`
- `docs/CHECKPOINT_LOADING_FIXES.md`
- `docs/DISSERTATION_VISUALIZATION_BLUEPRINT.md`
- `docs/alternative_equations.md`

---

**Built for Academic Research**

*Remember: This is NOT financial advice. Always do your own research and consult professionals before investing.*
