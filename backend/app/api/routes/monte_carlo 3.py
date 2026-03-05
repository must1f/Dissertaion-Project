"""Monte Carlo simulation API endpoints."""

from typing import Optional, List
from datetime import datetime
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
import numpy as np

# Import constants with fallback for when src is not available
try:
    from src.constants import (
        TRADING_DAYS_PER_YEAR,
        DEFAULT_ANNUAL_RETURN,
        DEFAULT_ANNUAL_VOLATILITY,
        DAILY_TIME_STEP,
    )
except ImportError:
    TRADING_DAYS_PER_YEAR = 252
    DEFAULT_ANNUAL_RETURN = 0.08
    DEFAULT_ANNUAL_VOLATILITY = 0.20
    DAILY_TIME_STEP = 1.0 / 252.0

from backend.app.dependencies import get_model_service, get_data_service
from backend.app.services.model_service import ModelService
from backend.app.services.data_service import DataService
from backend.app.schemas.monte_carlo import (
    MonteCarloRequest,
    MonteCarloResponse,
    MonteCarloResults,
    ConfidenceInterval,
    SimulationPath,
    DistributionStats,
    MonteCarloListResponse,
    MonteCarloSummary,
)
from backend.app.core.exceptions import PredictionError

router = APIRouter()

# Store simulation results
_simulation_results: dict = {}


@router.post("/simulate", response_model=MonteCarloResponse)
async def run_simulation(
    request: MonteCarloRequest,
    model_service: ModelService = Depends(get_model_service),
    data_service: DataService = Depends(get_data_service),
):
    """Run Monte Carlo simulation."""
    start_time = time.time()
    result_id = str(uuid.uuid4())[:8]

    try:
        # Get current price if not provided
        initial_price = request.initial_price
        if initial_price is None:
            stock_data = data_service.get_stock_data(request.ticker)
            if stock_data.data:
                initial_price = stock_data.data[-1].close
            else:
                initial_price = 100.0

        # Set random seed for reproducibility
        if request.random_seed is not None:
            np.random.seed(request.random_seed)

        # Simulate paths
        # Using GBM simulation with estimated parameters
        stock_data = data_service.get_stock_data(request.ticker)
        if stock_data.data:
            prices = np.array([d.close for d in stock_data.data])
            returns = np.diff(np.log(prices))
            mu = np.mean(returns) * TRADING_DAYS_PER_YEAR  # Annualized
            sigma = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)  # Annualized
        else:
            mu = DEFAULT_ANNUAL_RETURN
            sigma = DEFAULT_ANNUAL_VOLATILITY

        # Incorporate ML predictions if a model is selected
        if request.model_key and request.model_key not in ["historical", "default"]:
            try:
                from backend.app.services.prediction_service import PredictionService
                from backend.app.schemas.predictions import PredictionRequest
                
                prediction_service = PredictionService()
                pred_req = PredictionRequest(
                    ticker=request.ticker,
                    model_key=request.model_key,
                    estimate_uncertainty=True
                )
                pred_res = prediction_service.predict(pred_req)
                
                if pred_res.prediction and pred_res.prediction.expected_return is not None:
                    daily_mu = pred_res.prediction.expected_return
                    predicted_annual_mu = daily_mu * TRADING_DAYS_PER_YEAR
                    
                    # Blend ML prediction with historical drift to prevent extreme long-horizon paths
                    # 30% weight to short-term ML model, 70% to historical norm
                    mu = 0.3 * predicted_annual_mu + 0.7 * mu
                    
                    # Clamp drift to realistic annualized bounds [-0.5, 0.5]
                    mu = max(min(mu, 0.5), -0.5)
                
                if pred_res.prediction and pred_res.prediction.uncertainty_std is not None:
                    daily_sigma = pred_res.prediction.uncertainty_std
                    predicted_annual_sigma = daily_sigma * np.sqrt(TRADING_DAYS_PER_YEAR)
                    
                    # Blend ML uncertainty with historical volatility
                    sigma = 0.3 * predicted_annual_sigma + 0.7 * sigma
                    
                    # Clamp volatility to realistic bounds [5%, 80%]
                    sigma = max(min(sigma, 0.8), 0.05)
                    
            except Exception as e:
                print(f"Failed to fetch ML prediction for Monte Carlo drift blending, using historical parameters only: {e}")

        # GBM simulation
        dt = DAILY_TIME_STEP
        n_steps = request.horizon_days
        n_sims = request.n_simulations

        # Generate random paths
        Z = np.random.normal(0, 1, (n_sims, n_steps))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.insert(log_prices, 0, 0, axis=1)  # Add initial price

        all_prices = initial_price * np.exp(log_prices)

        # Calculate statistics
        final_prices = all_prices[:, -1]
        final_returns = (final_prices / initial_price) - 1

        # Confidence intervals
        confidence_intervals = []
        for level in request.confidence_levels:
            lower_pct = (1 - level) / 2 * 100
            upper_pct = (1 + level) / 2 * 100
            confidence_intervals.append(ConfidenceInterval(
                level=level,
                lower=float(np.percentile(final_prices, lower_pct)),
                upper=float(np.percentile(final_prices, upper_pct)),
            ))

        # Daily distributions
        daily_distributions = []
        for day in range(0, n_steps + 1, max(1, n_steps // 10)):
            day_prices = all_prices[:, day]
            daily_distributions.append(DistributionStats(
                day=day,
                mean=float(np.mean(day_prices)),
                median=float(np.median(day_prices)),
                std=float(np.std(day_prices)),
                skewness=float(_skewness(day_prices)),
                kurtosis=float(_kurtosis(day_prices)),
                min=float(np.min(day_prices)),
                max=float(np.max(day_prices)),
                percentiles={
                    "5": float(np.percentile(day_prices, 5)),
                    "25": float(np.percentile(day_prices, 25)),
                    "50": float(np.percentile(day_prices, 50)),
                    "75": float(np.percentile(day_prices, 75)),
                    "95": float(np.percentile(day_prices, 95)),
                },
            ))

        # Sample paths for visualization
        sample_indices = np.random.choice(n_sims, min(20, n_sims), replace=False)
        sample_paths = []
        for i, idx in enumerate(sample_indices):
            path_prices = all_prices[idx]
            path_return = (path_prices[-1] / path_prices[0]) - 1

            # Max drawdown
            peak = np.maximum.accumulate(path_prices)
            drawdown = (path_prices - peak) / peak
            max_dd = float(np.min(drawdown))

            sample_paths.append(SimulationPath(
                path_id=i,
                prices=path_prices.tolist(),
                final_price=float(path_prices[-1]),
                total_return=float(path_return),
                max_drawdown=max_dd,
            ))

        # Risk metrics
        prob_loss = float(np.mean(final_returns < 0))
        prob_gain = float(np.mean(final_returns > 0))
        var_95 = float(np.percentile(final_prices, 5))
        es_95 = float(np.mean(final_prices[final_prices <= var_95]))

        # Histogram
        hist_counts, hist_bins = np.histogram(final_prices, bins=50)

        results = MonteCarloResults(
            model_key=request.model_key,
            ticker=request.ticker,
            n_simulations=n_sims,
            horizon_days=n_steps,
            initial_price=initial_price,
            run_date=datetime.now(),
            final_price_mean=float(np.mean(final_prices)),
            final_price_median=float(np.median(final_prices)),
            final_price_std=float(np.std(final_prices)),
            final_return_mean=float(np.mean(final_returns)),
            final_return_std=float(np.std(final_returns)),
            confidence_intervals=confidence_intervals,
            daily_distributions=daily_distributions,
            probability_of_loss=prob_loss,
            probability_of_gain=prob_gain,
            value_at_risk_95=var_95,
            expected_shortfall_95=es_95,
            sample_paths=sample_paths,
            histogram_bins=hist_bins.tolist(),
            histogram_counts=hist_counts.tolist(),
        )

        # Store results
        _simulation_results[result_id] = results

        processing_time = (time.time() - start_time) * 1000

        return MonteCarloResponse(
            success=True,
            result_id=result_id,
            results=results,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}", response_model=MonteCarloResults)
async def get_simulation_results(result_id: str):
    """Get Monte Carlo simulation results."""
    if result_id not in _simulation_results:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

    return _simulation_results[result_id]


@router.get("/results", response_model=MonteCarloListResponse)
async def list_simulations():
    """List all Monte Carlo simulations."""
    results = [
        MonteCarloSummary(
            result_id=rid,
            model_key=r.model_key,
            ticker=r.ticker,
            run_date=r.run_date,
            n_simulations=r.n_simulations,
            horizon_days=r.horizon_days,
            expected_return=r.final_return_mean,
            value_at_risk_95=r.value_at_risk_95,
        )
        for rid, r in _simulation_results.items()
    ]

    return MonteCarloListResponse(
        results=results,
        total=len(results),
    )


def _skewness(x: np.ndarray) -> float:
    """Calculate skewness."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.sum(((x - mean) / std) ** 3) / n)


def _kurtosis(x: np.ndarray) -> float:
    """Calculate excess kurtosis."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.sum(((x - mean) / std) ** 4) / n - 3)
