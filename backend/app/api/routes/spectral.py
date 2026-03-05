"""Spectral analysis and regime detection API endpoints."""

from typing import Optional, List
from datetime import datetime
import time

from fastapi import APIRouter, Depends, HTTPException
import numpy as np

from backend.app.dependencies import get_data_service
from backend.app.services.data_service import DataService
from backend.app.schemas.spectral import (
    SpectralAnalysisRequest,
    SpectralAnalysisResponse,
    SpectralSnapshot,
    PowerSpectrumData,
    RegimeDetectionRequest,
    RegimeDetectionResponse,
    RegimeCharacteristics,
    TransitionMatrix,
    RegimeHistoryPoint,
    FanChartRequest,
    FanChartResponse,
    PercentileBand,
    RegimePeriod,
)

# Import from src if available
try:
    from src.data.spectral_analyzer import SpectralAnalyzer
    from src.evaluation.regime_detector import (
        get_regime_detector,
        SpectralHMMRegimeDetector,
        RegimeLabel,
    )
    from src.simulation.regime_monte_carlo import (
        RegimeSwitchingMC,
        SimulationConfig,
    )
    HAS_SRC = True
except ImportError:
    HAS_SRC = False

router = APIRouter()

REGIME_NAMES = {0: "Low Volatility", 1: "Normal", 2: "High Volatility"}


@router.post("/analyze", response_model=SpectralAnalysisResponse)
async def analyze_spectral(
    request: SpectralAnalysisRequest,
    data_service: DataService = Depends(get_data_service),
):
    """
    Compute spectral features for a ticker.

    Spectral analysis extracts frequency-domain features that capture:
    - Market randomness (spectral entropy)
    - Dominant cycles (dominant frequency/period)
    - Signal vs noise (power ratio)
    """
    start_time = time.time()

    if not HAS_SRC:
        raise HTTPException(
            status_code=501,
            detail="Spectral analysis requires src module. Install dependencies."
        )

    try:
        # Get stock data
        stock_data = data_service.get_stock_data(request.ticker)
        if not stock_data.data or len(stock_data.data) < request.window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {request.ticker}. Need at least {request.window_size} days."
            )

        # Extract prices and compute returns
        prices = np.array([d.close for d in stock_data.data])
        returns = np.diff(np.log(prices))

        # Create spectral analyzer
        analyzer = SpectralAnalyzer(window_size=request.window_size)

        # Compute current spectral features (last window)
        current_features = analyzer.compute_single_window(returns[-request.window_size:])

        # Compute power spectrum for visualization
        freqs, power = analyzer.compute_power_spectrum(
            returns[-request.window_size:],
            normalize=True
        )

        # Calculate frequency band totals
        low_mask = freqs < 0.1
        mid_mask = (freqs >= 0.1) & (freqs < 0.25)
        high_mask = freqs >= 0.25

        frequency_bands = {
            "low": {
                "min_freq": 0.0,
                "max_freq": 0.1,
                "total_power": float(np.sum(power[low_mask])),
            },
            "mid": {
                "min_freq": 0.1,
                "max_freq": 0.25,
                "total_power": float(np.sum(power[mid_mask])),
            },
            "high": {
                "min_freq": 0.25,
                "max_freq": 0.5,
                "total_power": float(np.sum(power[high_mask])),
            }
        }

        # Calculate dominant period
        dominant_freq = current_features['dominant_frequency']
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')

        # Build response
        current_snapshot = SpectralSnapshot(
            date=stock_data.data[-1].timestamp.isoformat() if hasattr(stock_data.data[-1].timestamp, 'isoformat') else str(stock_data.data[-1].timestamp),
            spectral_entropy=current_features['spectral_entropy'],
            dominant_frequency=dominant_freq,
            dominant_period=min(dominant_period, 252),  # Cap at 1 year
            power_low=current_features['power_low'],
            power_mid=current_features['power_mid'],
            power_high=current_features['power_high'],
            power_ratio=min(current_features['power_ratio'], 100),  # Cap ratio
            autocorrelation_lag1=current_features['autocorrelation_lag1'],
            spectral_slope=current_features['spectral_slope'],
        )

        power_spectrum = PowerSpectrumData(
            frequencies=freqs.tolist(),
            power=power.tolist(),
            frequency_bands=frequency_bands,
        )

        processing_time = (time.time() - start_time) * 1000

        return SpectralAnalysisResponse(
            success=True,
            ticker=request.ticker,
            analysis_date=datetime.now(),
            window_size=request.window_size,
            current_features=current_snapshot,
            power_spectrum=power_spectrum,
            historical_features=None,  # Could add rolling history
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/power-spectrum/{ticker}")
async def get_power_spectrum(
    ticker: str,
    window_size: int = 64,
    data_service: DataService = Depends(get_data_service),
):
    """Get power spectrum data for visualization."""
    if not HAS_SRC:
        raise HTTPException(
            status_code=501,
            detail="Spectral analysis requires src module."
        )

    try:
        stock_data = data_service.get_stock_data(ticker)
        if not stock_data.data or len(stock_data.data) < window_size:
            raise HTTPException(status_code=400, detail="Insufficient data")

        prices = np.array([d.close for d in stock_data.data])
        returns = np.diff(np.log(prices))

        analyzer = SpectralAnalyzer(window_size=window_size)
        freqs, power = analyzer.compute_power_spectrum(
            returns[-window_size:],
            normalize=True
        )

        return {
            "ticker": ticker,
            "frequencies": freqs.tolist(),
            "power": power.tolist(),
            "window_size": window_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regimes/detect", response_model=RegimeDetectionResponse)
async def detect_regimes(
    request: RegimeDetectionRequest,
    data_service: DataService = Depends(get_data_service),
):
    """
    Detect market regimes using HMM or spectral-enhanced HMM.

    Returns current regime, transition probabilities, and characteristics
    for each regime (volatility level, expected duration, etc.).
    """
    start_time = time.time()

    if not HAS_SRC:
        raise HTTPException(
            status_code=501,
            detail="Regime detection requires src module."
        )

    try:
        # Get stock data
        stock_data = data_service.get_stock_data(request.ticker)
        if not stock_data.data or len(stock_data.data) < request.lookback_days:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least {request.lookback_days} days."
            )

        # Extract returns
        prices = np.array([d.close for d in stock_data.data[-request.lookback_days:]])
        returns = np.diff(np.log(prices))

        # Get regime detector — filter kwargs for detector compatibility
        try:
            detector = get_regime_detector(
                method=request.method,
                n_regimes=request.n_regimes,
            )
        except TypeError:
            # Fallback detector may not accept n_regimes
            detector = get_regime_detector(method=request.method)

        # Fit detector
        detector.fit(returns)

        # Get predictions
        regimes = detector.predict(returns)

        # Get probabilities (some detectors lack predict_proba)
        if hasattr(detector, 'predict_proba'):
            probs = detector.predict_proba(returns)
        else:
            # One-hot fallback from discrete predictions
            n_unique = max(int(np.max(regimes)) + 1, request.n_regimes)
            probs = np.zeros((len(regimes), n_unique))
            for i, r in enumerate(regimes):
                probs[i, int(r)] = 1.0

        # Current state
        if hasattr(detector, 'get_current_state'):
            current_state = detector.get_current_state(returns)
        else:
            # Build a minimal state from the last regime
            from dataclasses import dataclass
            @dataclass
            class _MinimalState:
                regime: object
                probability: float
            class _RegimeProxy:
                def __init__(self, val):
                    self.value = int(val)
                    self.name = REGIME_NAMES.get(int(val), f"regime_{val}")
            current_state = _MinimalState(
                regime=_RegimeProxy(regimes[-1]),
                probability=float(probs[-1, int(regimes[-1])]),
            )

        # Get transition matrix
        if hasattr(detector, 'get_transition_matrix'):
            trans_matrix = detector.get_transition_matrix()
        else:
            # Estimate from observed regime sequence
            n_states = max(int(np.max(regimes)) + 1, request.n_regimes)
            trans_matrix = np.zeros((n_states, n_states))
            for i in range(len(regimes) - 1):
                trans_matrix[int(regimes[i]), int(regimes[i + 1])] += 1
            row_sums = trans_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans_matrix = trans_matrix / row_sums

        # Compute regime characteristics
        characteristics = []
        for regime_idx in range(request.n_regimes):
            mask = regimes == regime_idx
            regime_returns = returns[mask] if np.sum(mask) > 0 else np.array([0])

            # Expected duration from transition matrix
            if trans_matrix is not None:
                stay_prob = trans_matrix[regime_idx, regime_idx]
                expected_duration = 1.0 / (1.0 - stay_prob + 1e-10)
            else:
                expected_duration = 10.0

            # Stationary probability
            if hasattr(detector, 'transition_estimator') and detector.transition_estimator:
                stat_dist = detector.transition_estimator.get_stationary_distribution()
                stationary_prob = stat_dist[regime_idx]
            else:
                stationary_prob = 1.0 / request.n_regimes

            # Spectral characteristics (if spectral HMM)
            spectral_entropy = None
            dominant_freq = None
            power_ratio = None

            if hasattr(detector, 'get_spectral_regime_characteristics'):
                spectral_chars = detector.get_spectral_regime_characteristics(returns)
                if regime_idx in spectral_chars:
                    spectral_entropy = spectral_chars[regime_idx].get('spectral_entropy')
                    dominant_freq = spectral_chars[regime_idx].get('dominant_frequency')
                    power_ratio = spectral_chars[regime_idx].get('power_ratio')

            characteristics.append(RegimeCharacteristics(
                regime_id=regime_idx,
                regime_name=REGIME_NAMES.get(regime_idx, f"Regime {regime_idx}"),
                mean_return_annual=float(np.mean(regime_returns) * 252),
                volatility_annual=float(np.std(regime_returns) * np.sqrt(252)),
                stationary_probability=float(stationary_prob),
                expected_duration_days=float(expected_duration),
                spectral_entropy=spectral_entropy,
                dominant_frequency=dominant_freq,
                power_ratio=power_ratio,
                sample_count=int(np.sum(mask)),
            ))

        # Build transition matrix response
        trans_response = TransitionMatrix(
            matrix=trans_matrix.tolist() if trans_matrix is not None else [[1/request.n_regimes] * request.n_regimes] * request.n_regimes,
            labels=[REGIME_NAMES.get(i, f"Regime {i}") for i in range(request.n_regimes)],
        )

        # Recent history (last 20 days)
        recent_history = []
        for i in range(-min(20, len(regimes)), 0):
            date_idx = len(stock_data.data) + i
            if date_idx >= 0:
                date = stock_data.data[date_idx].timestamp
                date_str = date.isoformat() if hasattr(date, 'isoformat') else str(date)

                regime = int(regimes[i])
                prob = float(probs[i, regime])

                recent_history.append(RegimeHistoryPoint(
                    date=date_str,
                    regime=regime,
                    regime_name=REGIME_NAMES.get(regime, f"Regime {regime}"),
                    probability=prob,
                    all_probabilities={
                        REGIME_NAMES.get(j, f"Regime {j}"): float(probs[i, j])
                        for j in range(request.n_regimes)
                    },
                ))

        processing_time = (time.time() - start_time) * 1000

        return RegimeDetectionResponse(
            success=True,
            ticker=request.ticker,
            method=request.method,
            n_regimes=request.n_regimes,
            current_regime=current_state.regime.value,
            current_regime_name=current_state.regime.name.replace('_', ' ').title(),
            current_probability=current_state.probability,
            regime_characteristics=characteristics,
            transition_matrix=trans_response,
            recent_history=recent_history,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regimes/history/{ticker}")
async def get_regime_history(
    ticker: str,
    method: str = "spectral_hmm",
    n_regimes: int = 3,
    lookback_days: int = 252,
    data_service: DataService = Depends(get_data_service),
):
    """Get regime classification history for a ticker."""
    if not HAS_SRC:
        raise HTTPException(status_code=501, detail="Requires src module")

    try:
        stock_data = data_service.get_stock_data(ticker)
        if not stock_data.data or len(stock_data.data) < lookback_days:
            raise HTTPException(status_code=400, detail="Insufficient data")

        prices = np.array([d.close for d in stock_data.data[-lookback_days:]])
        returns = np.diff(np.log(prices))

        try:
            detector = get_regime_detector(method=method, n_regimes=n_regimes)
        except TypeError:
            detector = get_regime_detector(method=method)
        detector.fit(returns)

        regimes = detector.predict(returns)
        if hasattr(detector, 'predict_proba'):
            probs = detector.predict_proba(returns)
        else:
            n_unique = max(int(np.max(regimes)) + 1, n_regimes)
            probs = np.zeros((len(regimes), n_unique))
            for idx, r in enumerate(regimes):
                probs[idx, int(r)] = 1.0

        # Build history
        history = []
        for i, (regime, prob_row) in enumerate(zip(regimes, probs)):
            date_idx = len(stock_data.data) - lookback_days + i + 1
            if date_idx >= 0 and date_idx < len(stock_data.data):
                date = stock_data.data[date_idx].timestamp
                history.append({
                    "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    "regime": int(regime),
                    "regime_name": REGIME_NAMES.get(regime, f"Regime {regime}"),
                    "probabilities": {
                        REGIME_NAMES.get(j, f"Regime {j}"): float(prob_row[j])
                        for j in range(n_regimes)
                    }
                })

        return {
            "ticker": ticker,
            "method": method,
            "n_regimes": n_regimes,
            "history": history,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fan-chart", response_model=FanChartResponse)
async def generate_fan_chart(
    request: FanChartRequest,
    data_service: DataService = Depends(get_data_service),
):
    """
    Generate regime-aware Monte Carlo fan chart.

    Fan charts show forecast uncertainty with confidence bands that
    are colored by dominant regime (low/normal/high volatility).
    """
    start_time = time.time()

    if not HAS_SRC:
        raise HTTPException(status_code=501, detail="Requires src module")

    try:
        # Get stock data
        stock_data = data_service.get_stock_data(request.ticker)
        if not stock_data.data:
            raise HTTPException(status_code=400, detail="No data available")

        prices = np.array([d.close for d in stock_data.data])
        returns = np.diff(np.log(prices))

        initial_price = request.initial_price or float(prices[-1])

        if request.use_regime_switching:
            # Regime-switching Monte Carlo
            config = SimulationConfig(
                n_paths=request.n_simulations,
                horizon=request.horizon_days,
                seed=42,
            )

            mc = RegimeSwitchingMC(config=config, n_regimes=3)
            mc.fit(returns, method='gmm')

            fan_data = mc.generate_fan_chart(
                initial_price=initial_price,
                n_paths=request.n_simulations,
                horizon=request.horizon_days,
                percentiles=request.percentiles,
            )

            # Build response
            percentile_bands = [
                PercentileBand(percentile=p, values=fan_data.percentiles[p].tolist())
                for p in request.percentiles
            ]

            regime_periods = [
                RegimePeriod(
                    start=p['start'],
                    end=p['end'],
                    regime=p['regime'],
                    regime_name=REGIME_NAMES.get(p['regime'], f"Regime {p['regime']}")
                )
                for p in fan_data.get_regime_periods()
            ]

            regime_probs_list = [
                {REGIME_NAMES.get(j, f"Regime {j}"): float(fan_data.regime_probs[i, j]) for j in range(3)}
                for i in range(len(fan_data.regime_probs))
            ]

            median_path = fan_data.get_median().tolist()
            terminal_prices = fan_data.percentiles[50][-1] if 50 in fan_data.percentiles else initial_price
            var_95 = fan_data.percentiles[5][-1] if 5 in fan_data.percentiles else initial_price * 0.8

        else:
            # Standard Monte Carlo (simpler)
            mu = np.mean(returns) * 252
            sigma = np.std(returns) * np.sqrt(252)

            dt = 1/252
            Z = np.random.randn(request.n_simulations, request.horizon_days)
            log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            cum_returns = np.cumsum(log_returns, axis=1)
            paths = initial_price * np.exp(np.hstack([np.zeros((request.n_simulations, 1)), cum_returns]))

            percentile_bands = [
                PercentileBand(percentile=p, values=np.percentile(paths, p, axis=0).tolist())
                for p in request.percentiles
            ]

            median_path = np.median(paths, axis=0).tolist()
            regime_periods = []
            regime_probs_list = []
            terminal_prices = np.median(paths[:, -1])
            var_95 = float(np.percentile(paths[:, -1], 5))

        # Compute summary stats
        expected_return = (terminal_prices / initial_price - 1)
        probability_of_loss = np.mean([b.values[-1] < initial_price for b in percentile_bands]) if percentile_bands else 0.5

        processing_time = (time.time() - start_time) * 1000

        return FanChartResponse(
            success=True,
            ticker=request.ticker,
            initial_price=initial_price,
            horizon_days=request.horizon_days,
            n_simulations=request.n_simulations,
            dates=list(range(request.horizon_days + 1)),
            percentile_bands=percentile_bands,
            median_path=median_path,
            regime_periods=regime_periods,
            regime_probabilities=regime_probs_list,
            expected_return=float(expected_return),
            value_at_risk_95=float(var_95),
            probability_of_loss=float(probability_of_loss),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
