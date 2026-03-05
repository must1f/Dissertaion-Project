"""
Model Uncertainty Quantification using MC Dropout

Implements Bayesian approximation via Monte Carlo Dropout to provide
prediction intervals and confidence estimates.

Reference:
    Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning. ICML.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty quantification

    Enables dropout at inference time and runs multiple forward passes
    to estimate prediction uncertainty.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        dropout_rate: float = 0.1,
        confidence_level: float = 0.95
    ):
        """
        Initialize MC Dropout predictor

        Args:
            model: Trained neural network model
            n_samples: Number of MC samples (forward passes)
            dropout_rate: Dropout probability (if model doesn't have dropout)
            confidence_level: Confidence level for prediction intervals (default: 95%)
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.confidence_level = confidence_level

        # Calculate quantiles for confidence intervals
        alpha = 1 - confidence_level
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - (alpha / 2)

        logger.info(
            f"MCDropoutPredictor initialized: "
            f"n_samples={n_samples}, "
            f"dropout_rate={dropout_rate}, "
            f"confidence={confidence_level*100:.0f}%"
        )

    def enable_dropout(self):
        """
        Enable dropout layers at inference time

        This is critical for MC Dropout - we need dropout active during prediction
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Set to training mode to enable dropout

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates

        Args:
            x: Input tensor [batch, seq_len, features]
            return_samples: If True, return all MC samples

        Returns:
            Dictionary containing:
                - mean: Mean prediction [batch, output_dim]
                - std: Standard deviation (epistemic uncertainty) [batch, output_dim]
                - lower_bound: Lower confidence interval [batch, output_dim]
                - upper_bound: Upper confidence interval [batch, output_dim]
                - coefficient_of_variation: Relative uncertainty (std/mean) [batch, output_dim]
                - samples: All MC samples [n_samples, batch, output_dim] (if return_samples=True)
        """
        # Store original training mode
        was_training = self.model.training

        # Set model to eval mode but enable dropout
        self.model.eval()
        self.enable_dropout()

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                # Forward pass with dropout enabled
                pred = self.model(x)

                # Handle different model output formats
                if isinstance(pred, tuple):
                    pred = pred[0]  # Take first element if tuple

                predictions.append(pred)

        # Stack predictions: [n_samples, batch, output_dim]
        predictions = torch.stack(predictions)

        # Compute statistics
        mean_pred = predictions.mean(dim=0)  # [batch, output_dim]
        std_pred = predictions.std(dim=0)    # [batch, output_dim]

        # Confidence intervals
        lower_bound = torch.quantile(predictions, self.lower_quantile, dim=0)
        upper_bound = torch.quantile(predictions, self.upper_quantile, dim=0)

        # Coefficient of variation (relative uncertainty)
        # Avoid division by zero
        epsilon = 1e-8
        cv = std_pred / (torch.abs(mean_pred) + epsilon)

        # Restore original training mode
        self.model.train(was_training)

        result = {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'coefficient_of_variation': cv
        }

        if return_samples:
            result['samples'] = predictions

        return result

    def predict_with_confidence(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified interface: return mean, std, and confidence score

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean_prediction, std_prediction, confidence_score)
                - mean_prediction: Expected value [batch, output_dim]
                - std_prediction: Uncertainty estimate [batch, output_dim]
                - confidence_score: Confidence level in [0, 1] [batch, output_dim]
        """
        uncertainty_dict = self.predict_with_uncertainty(x)

        mean_pred = uncertainty_dict['mean']
        std_pred = uncertainty_dict['std']

        # Confidence score: inverse of normalized uncertainty
        # Higher uncertainty → lower confidence
        # Normalize std to [0, 1] range using sigmoid-like transformation
        confidence = 1.0 / (1.0 + uncertainty_dict['coefficient_of_variation'])

        return mean_pred, std_pred, confidence

    def compute_prediction_intervals(
        self,
        x: torch.Tensor,
        confidence_levels: Optional[list] = None
    ) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute prediction intervals at multiple confidence levels

        Args:
            x: Input tensor
            confidence_levels: List of confidence levels (default: [0.68, 0.95, 0.99])

        Returns:
            Dictionary mapping confidence level to (lower_bound, upper_bound) tuples
        """
        if confidence_levels is None:
            confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ equivalent

        # Store original training mode
        was_training = self.model.training

        # Enable dropout
        self.model.eval()
        self.enable_dropout()

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(pred)

        predictions = torch.stack(predictions)

        # Restore training mode
        self.model.train(was_training)

        intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            lower_q = alpha / 2
            upper_q = 1 - (alpha / 2)

            lower = torch.quantile(predictions, lower_q, dim=0)
            upper = torch.quantile(predictions, upper_q, dim=0)

            intervals[level] = (lower, upper)

        return intervals

    @torch.no_grad()
    def calibrate_uncertainty(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates on validation data

        Computes metrics like:
        - Prediction interval coverage (should match confidence level)
        - Mean prediction interval width
        - Calibration error

        Args:
            val_loader: Validation data loader
            device: Device to run on

        Returns:
            Dictionary of calibration metrics
        """
        self.model.eval()
        self.model.to(device)

        all_targets = []
        all_means = []
        all_lowers = []
        all_uppers = []

        for batch in val_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
            else:
                x = batch
                y = None

            x = x.to(device)

            # Get predictions with uncertainty
            uncertainty_dict = self.predict_with_uncertainty(x)

            all_means.append(uncertainty_dict['mean'].cpu())
            all_lowers.append(uncertainty_dict['lower_bound'].cpu())
            all_uppers.append(uncertainty_dict['upper_bound'].cpu())

            if y is not None:
                all_targets.append(y.cpu())

        # Concatenate all batches
        all_means = torch.cat(all_means, dim=0)
        all_lowers = torch.cat(all_lowers, dim=0)
        all_uppers = torch.cat(all_uppers, dim=0)

        metrics = {}

        if all_targets:
            all_targets = torch.cat(all_targets, dim=0)

            # Prediction interval coverage
            in_interval = (all_targets >= all_lowers) & (all_targets <= all_uppers)
            coverage = in_interval.float().mean().item()

            # Mean interval width
            interval_width = (all_uppers - all_lowers).mean().item()

            # Calibration error (difference between expected and actual coverage)
            calibration_error = abs(coverage - self.confidence_level)

            # RMSE
            rmse = torch.sqrt(torch.mean((all_means - all_targets) ** 2)).item()

            metrics.update({
                'coverage': coverage,
                'expected_coverage': self.confidence_level,
                'interval_width': interval_width,
                'calibration_error': calibration_error,
                'rmse': rmse
            })

        logger.info(f"Calibration metrics: {metrics}")

        return metrics


class EnsemblePredictor:
    """
    Ensemble-based uncertainty quantification

    Uses multiple independently trained models to estimate uncertainty.
    Generally more accurate than MC Dropout but requires training multiple models.
    """

    def __init__(
        self,
        model_paths: list,
        model_class: type,
        model_kwargs: dict,
        confidence_level: float = 0.95
    ):
        """
        Initialize ensemble predictor

        Args:
            model_paths: List of paths to model checkpoints
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            confidence_level: Confidence level for intervals
        """
        self.model_paths = model_paths
        self.confidence_level = confidence_level

        # Load all models
        self.models = []
        for path in model_paths:
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self.models.append(model)

        logger.info(f"Loaded {len(self.models)} models for ensemble prediction")

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make ensemble predictions with uncertainty

        Args:
            x: Input tensor
            return_samples: If True, return individual model predictions

        Returns:
            Dictionary with mean, std, bounds, etc.
        """
        predictions = []

        for model in self.models:
            model.eval()
            pred = model(x)

            if isinstance(pred, tuple):
                pred = pred[0]

            predictions.append(pred)

        # Stack predictions
        predictions = torch.stack(predictions)

        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        alpha = 1 - self.confidence_level
        lower_bound = torch.quantile(predictions, alpha / 2, dim=0)
        upper_bound = torch.quantile(predictions, 1 - alpha / 2, dim=0)

        epsilon = 1e-8
        cv = std_pred / (torch.abs(mean_pred) + epsilon)

        result = {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'coefficient_of_variation': cv
        }

        if return_samples:
            result['samples'] = predictions

        return result


def add_uncertainty_to_results(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    confidence: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Package predictions with uncertainty into a results dictionary

    Args:
        predictions: Point predictions [n_samples]
        uncertainties: Uncertainty estimates (std) [n_samples]
        confidence: Confidence scores [n_samples]

    Returns:
        Dictionary with predictions and uncertainty metrics
    """
    return {
        'predictions': predictions,
        'uncertainty_std': uncertainties,
        'confidence': confidence,
        'lower_bound_95': predictions - 1.96 * uncertainties,
        'upper_bound_95': predictions + 1.96 * uncertainties,
        'lower_bound_68': predictions - 1.0 * uncertainties,
        'upper_bound_68': predictions + 1.0 * uncertainties
    }
