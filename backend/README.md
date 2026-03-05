# PINN Financial Forecasting - Backend

FastAPI backend for the PINN Financial Forecasting application.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python run.py
```

Server runs at http://localhost:8000

API documentation at http://localhost:8000/docs

## Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── dependencies.py      # Dependency injection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/          # API route handlers
│   │       ├── data.py          # Stock data endpoints
│   │       ├── models.py        # Model management
│   │       ├── predictions.py   # Prediction endpoints
│   │       ├── training.py      # Training control
│   │       ├── backtesting.py   # Backtest endpoints
│   │       ├── metrics.py       # Metrics endpoints
│   │       ├── monte_carlo.py   # Monte Carlo simulation
│   │       └── websocket.py     # WebSocket handlers
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py      # Database connection
│   │   └── exceptions.py    # Custom exceptions
│   │
│   ├── schemas/             # Pydantic models
│   │   ├── data.py
│   │   ├── models.py
│   │   ├── predictions.py
│   │   ├── training.py
│   │   ├── backtesting.py
│   │   ├── metrics.py
│   │   └── monte_carlo.py
│   │
│   └── services/            # Business logic
│       ├── data_service.py
│       ├── model_service.py
│       ├── prediction_service.py
│       ├── training_service.py
│       ├── backtest_service.py
│       └── metrics_service.py
│
├── requirements.txt
└── run.py                   # Uvicorn runner
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/data/stocks` | List available tickers |
| `GET /api/data/stocks/{ticker}` | Get stock data |
| `GET /api/models/` | List available models |
| `POST /api/predictions/predict` | Run prediction |
| `POST /api/training/start` | Start training job |
| `WS /api/training/ws/{job_id}` | Training updates |
| `POST /api/backtest/run` | Run backtest |
| `GET /api/metrics/financial` | Financial metrics |
| `POST /api/monte-carlo/simulate` | Monte Carlo simulation |

## Configuration

Environment variables (`.env`):

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/pinn

# Paths
MODELS_DIR=../Models
RESULTS_DIR=../results
SRC_DIR=../src

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# CORS
CORS_ORIGINS=http://localhost:5173
```

## Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v

# Type checking
mypy app/

# Linting
ruff check app/
```

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **SQLAlchemy**: Database ORM
- **PyTorch**: ML framework (via src/)

## Service Layer

Services wrap the ML core (`src/`) to provide clean interfaces:

```python
# Example: PredictionService
from app.services.prediction_service import PredictionService

service = PredictionService()
result = service.predict(
    model_key="pinn_gbm_ou",
    ticker="AAPL",
    horizon=5
)
```

## WebSocket Protocol

Training updates sent as JSON:
```json
{
  "epoch": 50,
  "total_epochs": 100,
  "loss": 0.0023,
  "data_loss": 0.0018,
  "physics_loss": 0.0005,
  "val_loss": 0.0028
}
```

## Error Handling

Custom exceptions in `core/exceptions.py`:
- `ModelNotFoundError`
- `DataFetchError`
- `PredictionError`
- `TrainingError`
- `BacktestError`
