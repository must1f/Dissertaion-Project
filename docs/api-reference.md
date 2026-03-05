# API Reference

Base URL: `http://localhost:8000`

API documentation is auto-generated at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## Data Endpoints

### GET /api/data/stocks

List available stock tickers.

**Response:**
```json
{
  "tickers": ["AAPL", "GOOGL", "MSFT", "TSLA"],
  "count": 4
}
```

### GET /api/data/stocks/{ticker}

Get historical stock data.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| ticker | path | Stock ticker symbol |
| start_date | query | Start date (YYYY-MM-DD) |
| end_date | query | End date (YYYY-MM-DD) |
| interval | query | Data interval (1d, 1h) |

**Response:**
```json
{
  "ticker": "AAPL",
  "data": [
    {
      "date": "2024-01-01",
      "open": 185.50,
      "high": 186.20,
      "low": 184.80,
      "close": 185.90,
      "volume": 45000000
    }
  ],
  "count": 252
}
```

### POST /api/data/fetch

Fetch new data from yfinance.

**Request:**
```json
{
  "ticker": "AAPL",
  "period": "1y"
}
```

---

## Model Endpoints

### GET /api/models/

List all available models.

**Response:**
```json
{
  "models": [
    {
      "key": "pinn_gbm_ou",
      "name": "PINN GBM+OU",
      "type": "pinn",
      "physics_equations": ["gbm", "ornstein_uhlenbeck"],
      "has_weights": true
    }
  ]
}
```

### GET /api/models/{model_key}

Get model details.

**Response:**
```json
{
  "key": "pinn_gbm_ou",
  "name": "PINN GBM+OU",
  "type": "pinn",
  "architecture": {
    "hidden_dims": [128, 64, 32],
    "dropout": 0.1
  },
  "physics_parameters": {
    "theta": 0.15,
    "gamma": 0.02,
    "mu": 0.08,
    "sigma": 0.20
  }
}
```

### GET /api/models/compare

Compare multiple models.

**Query Parameters:**
| Name | Type | Description |
|------|------|-------------|
| models | query | Comma-separated model keys |

---

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

**Response:**
```json
{
  "ticker": "AAPL",
  "model": "pinn_gbm_ou",
  "predictions": [
    {
      "date": "2024-01-15",
      "predicted": 188.50,
      "lower": 185.20,
      "upper": 191.80,
      "confidence": 0.95
    }
  ],
  "current_price": 186.00
}
```

### GET /api/predictions/history

Get prediction history.

---

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

**Response:**
```json
{
  "job_id": "abc123",
  "status": "started",
  "ws_url": "/api/training/ws/abc123"
}
```

### GET /api/training/status/{job_id}

Get training status.

### POST /api/training/stop/{job_id}

Stop training job.

### WebSocket /api/training/ws/{job_id}

Real-time training updates.

**Messages:**
```json
{
  "epoch": 50,
  "total_epochs": 100,
  "loss": 0.0023,
  "data_loss": 0.0018,
  "physics_loss": 0.0005,
  "val_loss": 0.0028,
  "learning_rate": 0.0008,
  "elapsed_time": 45.2
}
```

---

## Backtest Endpoints

### POST /api/backtest/run

Run backtest.

**Request:**
```json
{
  "model_key": "pinn_gbm_ou",
  "ticker": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 100000,
  "strategy": "trend_following"
}
```

**Response:**
```json
{
  "backtest_id": "bt123",
  "metrics": {
    "total_return": 0.25,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.12,
    "win_rate": 0.58
  },
  "equity_curve": [...],
  "trades": [...]
}
```

### GET /api/backtest/results/{id}

Get backtest results.

---

## Metrics Endpoints

### GET /api/metrics/financial

Get financial metrics.

**Query Parameters:**
| Name | Type | Description |
|------|------|-------------|
| model_key | query | Model to evaluate |
| ticker | query | Stock ticker |

**Response:**
```json
{
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.34,
  "max_drawdown": -0.12,
  "calmar_ratio": 15.4,
  "win_rate": 0.58,
  "profit_factor": 1.67
}
```

### GET /api/metrics/ml

Get ML metrics.

**Response:**
```json
{
  "rmse": 2.34,
  "mae": 1.87,
  "mape": 0.012,
  "r2": 0.92,
  "directional_accuracy": 0.67
}
```

### GET /api/metrics/physics

Get physics constraint metrics.

**Response:**
```json
{
  "physics_loss": 0.0005,
  "gbm_residual": 0.0003,
  "ou_residual": 0.0002,
  "learned_parameters": {
    "theta": 0.15,
    "gamma": 0.02,
    "mu": 0.08,
    "sigma": 0.20
  }
}
```

---

## Monte Carlo Endpoints

### POST /api/monte-carlo/simulate

Run Monte Carlo simulation.

**Request:**
```json
{
  "model_key": "pinn_gbm_ou",
  "ticker": "AAPL",
  "n_simulations": 1000,
  "horizon_days": 30,
  "initial_price": 186.00
}
```

**Response:**
```json
{
  "simulation_id": "mc123",
  "results": {
    "final_price_mean": 192.50,
    "final_price_median": 191.20,
    "final_price_std": 12.30,
    "probability_of_gain": 0.65,
    "probability_of_loss": 0.35,
    "value_at_risk_95": 168.50,
    "confidence_intervals": {
      "50": [182.30, 202.70],
      "75": [175.40, 209.60],
      "95": [162.80, 222.20]
    }
  }
}
```

---

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
