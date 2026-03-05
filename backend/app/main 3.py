"""FastAPI application entry point."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.core.exceptions import APIException

# Import routers
from backend.app.api.routes import (
    data,
    models,
    predictions,
    training,
    backtesting,
    metrics,
    monte_carlo,
    websocket,
    trading,
    analysis,
    volatility,
    dissertation,
    spectral,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Project root: {settings.project_root}")
    print(f"Debug mode: {settings.debug}")

    yield

    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## PINN Financial Forecasting API

    A RESTful API for physics-informed neural network based financial forecasting.

    ### Features
    - **Stock Data**: Fetch and manage historical stock data
    - **Models**: Manage and compare ML/PINN models
    - **Predictions**: Generate predictions with uncertainty estimates
    - **Training**: Train models with real-time progress updates
    - **Backtesting**: Run backtests with various strategies
    - **Metrics**: Calculate financial and ML metrics
    - **Monte Carlo**: Run Monte Carlo simulations

    ### Model Types
    - Baseline: LSTM, GRU, BiLSTM, Attention LSTM, Transformer
    - PINN: GBM, Ornstein-Uhlenbeck, Black-Scholes variants
    - Advanced: Stacked PINN, Residual PINN
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for custom exceptions
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
        },
    )


# Include routers
app.include_router(
    data.router,
    prefix="/api/data",
    tags=["Data"],
)

app.include_router(
    models.router,
    prefix="/api/models",
    tags=["Models"],
)

app.include_router(
    predictions.router,
    prefix="/api/predictions",
    tags=["Predictions"],
)

app.include_router(
    training.router,
    prefix="/api/training",
    tags=["Training"],
)

app.include_router(
    backtesting.router,
    prefix="/api/backtest",
    tags=["Backtesting"],
)

app.include_router(
    metrics.router,
    prefix="/api/metrics",
    tags=["Metrics"],
)

app.include_router(
    monte_carlo.router,
    prefix="/api/monte-carlo",
    tags=["Monte Carlo"],
)

app.include_router(
    websocket.router,
    prefix="/api/ws",
    tags=["WebSocket"],
)

app.include_router(
    trading.router,
    prefix="/api/trading",
    tags=["Trading"],
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["Analysis"],
)

app.include_router(
    volatility.router,
    prefix="/api/volatility",
    tags=["Volatility Forecasting"],
)

app.include_router(
    dissertation.router,
    prefix="/api/dissertation",
    tags=["Dissertation Analysis"],
)

app.include_router(
    spectral.router,
    prefix="/api/spectral",
    tags=["Spectral Analysis"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PINN Financial Forecasting API",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
    }
