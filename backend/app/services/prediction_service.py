"""Service for predictions wrapping src/trading/."""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import uuid
import time

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings
from backend.app.core.exceptions import PredictionError, ModelNotTrainedError
from backend.app.schemas.predictions import (
    PredictionRequest,
    PredictionResult,
    PredictionResponse,
    PredictionInterval,
    SignalAction,
    UncertaintyMethod,
)
from backend.app.services.model_service import ModelService, _get_available_device
from backend.app.services.data_service import DataService

# Import from existing src/
try:
    from src.trading.agent import SignalGenerator, UncertaintyEstimator
    HAS_SRC = True
except ImportError:
    HAS_SRC = False
    SignalGenerator = None
    UncertaintyEstimator = None


class PredictionService:
    """Service for running predictions."""

    def __init__(self):
        """Initialize prediction service."""
        self._model_service = ModelService()
        self._data_service = DataService()
        self._signal_generators: Dict[str, "SignalGenerator"] = {}
        self._prediction_history: List[Dict] = []
        self._device = _get_available_device(settings.default_device)

    def _get_signal_generator(self, model_key: str) -> "SignalGenerator":
        """Get or create signal generator for a model."""
        if model_key in self._signal_generators:
            return self._signal_generators[model_key]

        if not HAS_SRC:
            raise PredictionError("Prediction requires src/ modules")

        model = self._model_service.load_model(model_key)
        if model is None:
            raise PredictionError(
                f"Model '{model_key}' could not be loaded. "
                "Ensure it has been trained and a checkpoint exists."
            )
        generator = SignalGenerator(
            model=model,
            device=self._device,
            n_mc_samples=50,
        )
        self._signal_generators[model_key] = generator
        return generator

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run prediction for a single ticker."""
        start_time = time.time()

        try:
            # Get data
            sequences, targets, df = self._data_service.prepare_sequences(
                ticker=request.ticker,
                sequence_length=request.sequence_length,
            )

            if len(sequences) == 0:
                raise PredictionError("Not enough data for prediction")

            # Get latest sequence
            latest_sequence = sequences[-1:]
            latest_sequence_tensor = torch.FloatTensor(latest_sequence).to(self._device)

            # Get current price (last close)
            stock_data = self._data_service.get_stock_data(request.ticker)
            current_price = stock_data.data[-1].close if stock_data.data else 100.0

            # Run prediction
            if HAS_SRC:
                generator = self._get_signal_generator(request.model_key)

                # Get prediction with uncertainty
                predictions, uncertainties, uncertainty_details = generator.predict(
                    sequences=latest_sequence_tensor,
                    estimate_uncertainty=request.estimate_uncertainty,
                    method=request.uncertainty_method.value,
                )

                # ── De-normalise model output ──
                # The model was trained to predict normalized log_return
                ret_mean = float(getattr(df, '_scaler_mean', {}).get('log_return', 0.0))
                ret_std  = float(getattr(df, '_scaler_std',  {}).get('log_return', 1.0))

                raw_pred = float(predictions[0])
                real_log_return = raw_pred * ret_std + ret_mean
                predicted_price = current_price * np.exp(real_log_return)
                predicted_return = (predicted_price - current_price) / current_price if current_price else 0.0

                # Uncertainty (also in normalised space)
                uncertainty_std = None
                if uncertainties is not None:
                    raw_unc = float(uncertainties[0])
                    uncertainty_std = raw_unc * ret_std

                prediction_interval = None
                confidence_score = None

                if uncertainty_details:
                    # Handle keys from SignalGenerator.predict()
                    lower_key = "prediction_interval_lower" if "prediction_interval_lower" in uncertainty_details else "lower"
                    upper_key = "prediction_interval_upper" if "prediction_interval_upper" in uncertainty_details else "upper"
                    confidence_key = "confidence_scores" if "confidence_scores" in uncertainty_details else "confidence"

                    if lower_key in uncertainty_details and upper_key in uncertainty_details:
                        lower_ret = float(uncertainty_details[lower_key][0]) * ret_std + ret_mean
                        upper_ret = float(uncertainty_details[upper_key][0]) * ret_std + ret_mean
                        lower_price = current_price * np.exp(lower_ret)
                        upper_price = current_price * np.exp(upper_ret)
                        prediction_interval = PredictionInterval(
                            lower=lower_price,
                            upper=upper_price,
                            confidence=0.95,
                        )
                    if confidence_key in uncertainty_details:
                        confidence_score = float(uncertainty_details[confidence_key][0])

                # Generate signal
                signal_action = None
                if request.generate_signal:
                    if predicted_return > request.signal_threshold:
                        signal_action = SignalAction.BUY
                    elif predicted_return < -request.signal_threshold:
                        signal_action = SignalAction.SELL
                    else:
                        signal_action = SignalAction.HOLD

                # Get physics parameters if available
                model = self._model_service.load_model(request.model_key)
                physics_params = None
                if hasattr(model, "get_learned_physics_params"):
                    physics_params = model.get_learned_physics_params()

            else:
                if settings.demo_mode:
                    # Mock prediction when src/ not available (demo only)
                    predicted_return = 0.01  # 1% return
                    predicted_price = current_price * 1.01
                    uncertainty_std = 0.02
                    prediction_interval = PredictionInterval(
                        lower=current_price * 0.97,
                        upper=current_price * 1.05,
                        confidence=0.95,
                    )
                    confidence_score = 0.75
                    signal_action = SignalAction.HOLD
                    physics_params = None
                else:
                    raise PredictionError(
                        "Prediction requires src/ modules and trained models. "
                        "Install dependencies or enable DEMO_MODE for mock responses."
                    )

            # Build result
            result = PredictionResult(
                timestamp=datetime.now(),
                ticker=request.ticker,
                model_key=request.model_key,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                current_price=current_price,
                uncertainty_std=uncertainty_std,
                prediction_interval=prediction_interval,
                confidence_score=confidence_score,
                signal_action=signal_action,
                expected_return=predicted_return,
            )

            # Store in history
            self._prediction_history.append({
                "id": str(uuid.uuid4()),
                "timestamp": result.timestamp,
                "ticker": result.ticker,
                "model_key": result.model_key,
                "predicted_price": result.predicted_price,
                "signal_action": result.signal_action.value if result.signal_action else None,
            })

            processing_time = (time.time() - start_time) * 1000

            return PredictionResponse(
                success=True,
                prediction=result,
                model_info={
                    "model_key": request.model_key,
                    "sequence_length": request.sequence_length,
                    "uncertainty_method": request.uncertainty_method.value,
                },
                physics_parameters=physics_params,
                processing_time_ms=processing_time,
            )

        except ModelNotTrainedError:
            raise
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")

    def batch_predict(
        self,
        tickers: List[str],
        model_key: str,
        sequence_length: int = 60,
        estimate_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """Run predictions for multiple tickers."""
        start_time = time.time()
        results = []
        failed = []

        for ticker in tickers:
            try:
                request = PredictionRequest(
                    ticker=ticker,
                    model_key=model_key,
                    sequence_length=sequence_length,
                    estimate_uncertainty=estimate_uncertainty,
                )
                response = self.predict(request)
                results.append(response.prediction)
            except Exception as e:
                failed.append(ticker)
                print(f"Failed prediction for {ticker}: {e}")

        total_time = (time.time() - start_time) * 1000

        return {
            "success": len(results) > 0,
            "predictions": results,
            "failed_tickers": failed,
            "total_processing_time_ms": total_time,
        }

    def get_prediction_history(
        self,
        ticker: Optional[str] = None,
        model_key: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """Get prediction history."""
        history = self._prediction_history.copy()

        # Filter
        if ticker:
            history = [p for p in history if p["ticker"] == ticker]
        if model_key:
            history = [p for p in history if p["model_key"] == model_key]

        # Sort by timestamp descending
        history.sort(key=lambda x: x["timestamp"], reverse=True)

        # Paginate
        total = len(history)
        start = (page - 1) * page_size
        end = start + page_size
        history = history[start:end]

        return {
            "predictions": history,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_latest_predictions(
        self,
        ticker: str,
    ) -> Dict[str, PredictionResult]:
        """Get latest predictions for a ticker from all models."""
        results = {}

        # Get trained models
        trained_models = self._model_service.get_trained_models()

        for model_info in trained_models:
            try:
                request = PredictionRequest(
                    ticker=ticker,
                    model_key=model_info.model_key,
                    estimate_uncertainty=True,
                    generate_signal=True,
                )
                response = self.predict(request)
                results[model_info.model_key] = response.prediction
            except Exception as e:
                print(f"Failed to get prediction from {model_info.model_key}: {e}")
                continue

        return results

    def clear_cache(self, model_key: Optional[str] = None):
        """Clear signal generator cache."""
        if model_key:
            if model_key in self._signal_generators:
                del self._signal_generators[model_key]
        else:
            self._signal_generators.clear()
