"""
Demonstration of Uncertainty Quantification in Financial Forecasting

Shows how to use MC Dropout to:
1. Generate predictions with confidence intervals
2. Make uncertainty-aware trading decisions
3. Visualize prediction uncertainty over time
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.uncertainty import MCDropoutPredictor
from src.models.baseline import LSTMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model_with_uncertainty(
    model_path: str,
    input_dim: int = 10,
    hidden_dim: int = 128,
    n_mc_samples: int = 100
) -> MCDropoutPredictor:
    """
    Load a trained model and wrap it with MC Dropout

    Args:
        model_path: Path to model checkpoint
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        n_mc_samples: Number of MC samples for uncertainty

    Returns:
        MCDropoutPredictor instance
    """
    # Load model
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        output_dim=1,
        dropout=0.2  # Important: model must have dropout layers
    )

    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Wrap with MC Dropout
    mc_predictor = MCDropoutPredictor(
        model=model,
        n_samples=n_mc_samples,
        dropout_rate=0.2,
        confidence_level=0.95
    )

    logger.info(f"Model loaded from {model_path}")
    logger.info(f"MC Dropout enabled with {n_mc_samples} samples")

    return mc_predictor


def predict_with_uncertainty_example():
    """Example: Make predictions with uncertainty estimates"""

    print("=" * 60)
    print("MC Dropout Uncertainty Quantification Demo")
    print("=" * 60)

    # Create dummy model for demonstration
    model = LSTMModel(
        input_dim=10,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2
    )

    # Initialize MC Dropout predictor
    mc_predictor = MCDropoutPredictor(
        model=model,
        n_samples=100,
        confidence_level=0.95
    )

    # Generate dummy input data [batch, seq_len, features]
    batch_size = 16
    seq_len = 60  # 60 days of history
    features = 10

    x = torch.randn(batch_size, seq_len, features)

    # Make predictions with uncertainty
    print("\nMaking predictions with uncertainty...")
    result = mc_predictor.predict_with_uncertainty(x)

    print(f"\nMean predictions shape: {result['mean'].shape}")
    print(f"Uncertainty (std) shape: {result['std'].shape}")
    print(f"\nSample prediction:")
    print(f"  Mean: {result['mean'][0].item():.4f}")
    print(f"  Std:  {result['std'][0].item():.4f}")
    print(f"  95% CI: [{result['lower_bound'][0].item():.4f}, "
          f"{result['upper_bound'][0].item():.4f}]")
    print(f"  Coefficient of Variation: {result['coefficient_of_variation'][0].item():.4f}")

    # Simplified interface
    print("\n" + "=" * 60)
    print("Simplified Prediction Interface")
    print("=" * 60)

    mean, std, confidence = mc_predictor.predict_with_confidence(x)

    print(f"\nPredictions for all {batch_size} samples:")
    for i in range(min(5, batch_size)):
        print(f"  Sample {i+1}: μ={mean[i].item():7.4f}, "
              f"σ={std[i].item():6.4f}, "
              f"confidence={confidence[i].item():.3f}")

    # Multiple confidence levels
    print("\n" + "=" * 60)
    print("Prediction Intervals at Multiple Confidence Levels")
    print("=" * 60)

    intervals = mc_predictor.compute_prediction_intervals(
        x,
        confidence_levels=[0.68, 0.95, 0.99]
    )

    sample_idx = 0
    print(f"\nPrediction intervals for sample {sample_idx + 1}:")
    for level, (lower, upper) in intervals.items():
        width = (upper - lower)[sample_idx].item()
        print(f"  {level*100:.0f}% CI: "
              f"[{lower[sample_idx].item():7.4f}, {upper[sample_idx].item():7.4f}]  "
              f"(width: {width:.4f})")


def uncertainty_aware_trading_example():
    """Example: Use uncertainty for trading decisions"""

    print("\n" + "=" * 60)
    print("Uncertainty-Aware Trading Signals")
    print("=" * 60)

    # Create model
    model = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2)
    mc_predictor = MCDropoutPredictor(model, n_samples=50)

    # Dummy data
    x = torch.randn(10, 60, 10)

    # Get predictions with uncertainty
    mean, std, confidence = mc_predictor.predict_with_confidence(x)

    # Current prices (dummy)
    current_prices = torch.rand(10, 1) * 100 + 100  # Prices around 100-200

    # Convert to numpy for easier manipulation
    predictions = mean.numpy()
    uncertainties = std.numpy()
    confidences = confidence.numpy()
    current = current_prices.numpy()

    # Trading logic
    print("\nTrading signals based on predictions and uncertainty:\n")
    print(f"{'#':<3} {'Current':<8} {'Predicted':<10} {'Return%':<8} "
          f"{'Confidence':<11} {'Signal':<10}")
    print("-" * 60)

    for i in range(len(predictions)):
        pred = predictions[i, 0]
        curr = current[i, 0]
        conf = confidences[i, 0]
        uncert = uncertainties[i, 0]

        # Expected return
        expected_return = (pred - curr) / curr * 100

        # Trading signal logic
        # Only trade if:
        # 1. Expected return > threshold
        # 2. Confidence > threshold
        # 3. Consider uncertainty (higher uncertainty = reduce position size)

        if expected_return > 2.0 and conf > 0.6:
            # Position size scaled by confidence
            position_size = int(conf * 100)  # 0-100%
            signal = f"BUY ({position_size}%)"
        elif expected_return < -2.0 and conf > 0.6:
            position_size = int(conf * 100)
            signal = f"SELL ({position_size}%)"
        else:
            signal = "HOLD"

        print(f"{i+1:<3} {curr:7.2f} {pred:9.2f} {expected_return:7.2f}% "
              f"{conf:10.3f} {signal:<10}")


def visualize_uncertainty():
    """Example: Visualize predictions with uncertainty bands"""

    print("\n" + "=" * 60)
    print("Generating Uncertainty Visualization")
    print("=" * 60)

    # Create model
    model = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2)
    mc_predictor = MCDropoutPredictor(model, n_samples=100)

    # Generate time series data
    n_timesteps = 50
    x_series = [torch.randn(1, 60, 10) for _ in range(n_timesteps)]

    # Make predictions
    predictions = []
    lower_bounds = []
    upper_bounds = []

    print("\nGenerating predictions for 50 time steps...")
    for x in x_series:
        result = mc_predictor.predict_with_uncertainty(x)
        predictions.append(result['mean'][0, 0].item())
        lower_bounds.append(result['lower_bound'][0, 0].item())
        upper_bounds.append(result['upper_bound'][0, 0].item())

    # Convert to numpy
    predictions = np.array(predictions)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Generate "true" values (for demonstration)
    true_values = predictions + np.random.randn(n_timesteps) * 0.5

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    time_steps = np.arange(n_timesteps)

    # Plot prediction with uncertainty band
    ax.plot(time_steps, predictions, 'b-', label='Predicted', linewidth=2)
    ax.fill_between(time_steps, lower_bounds, upper_bounds,
                     alpha=0.3, color='blue', label='95% Confidence Interval')

    # Plot true values
    ax.plot(time_steps, true_values, 'ro', label='Actual', alpha=0.6, markersize=4)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Prediction Value')
    ax.set_title('Stock Price Predictions with MC Dropout Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    output_dir = Path(__file__).parent.parent / 'dissertation' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'mc_dropout_uncertainty_demo.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\nVisualization saved to: {output_path}")
    print("\nShowing plot...")
    plt.show()


def calibration_example():
    """Example: Calibrate uncertainty estimates"""

    print("\n" + "=" * 60)
    print("Uncertainty Calibration Example")
    print("=" * 60)

    # Create model
    model = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2)
    mc_predictor = MCDropoutPredictor(model, n_samples=100, confidence_level=0.95)

    # Generate synthetic validation data
    print("\nGenerating synthetic validation data...")

    n_samples = 200
    val_data = []

    for _ in range(n_samples):
        x = torch.randn(1, 60, 10)
        y = torch.randn(1, 1)  # Dummy target
        val_data.append((x, y))

    # Create simple DataLoader-like iterator
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return iter(self.data)

    val_loader = SimpleDataset(val_data)

    # Calibrate
    print("\nCalibrating uncertainty estimates...")
    calibration_metrics = mc_predictor.calibrate_uncertainty(val_loader, device='cpu')

    print("\nCalibration Results:")
    print(f"  Expected Coverage (95% CI): {calibration_metrics['expected_coverage']:.1%}")
    print(f"  Actual Coverage:            {calibration_metrics['coverage']:.1%}")
    print(f"  Calibration Error:          {calibration_metrics['calibration_error']:.4f}")
    print(f"  Mean Interval Width:        {calibration_metrics['interval_width']:.4f}")
    print(f"  RMSE:                       {calibration_metrics['rmse']:.4f}")

    if calibration_metrics['calibration_error'] < 0.05:
        print("\n✅ Model is well-calibrated (error < 5%)")
    else:
        print("\n⚠️  Model may need recalibration")


def main():
    """Run all uncertainty quantification examples"""

    # Basic prediction with uncertainty
    predict_with_uncertainty_example()

    # Trading application
    uncertainty_aware_trading_example()

    # Calibration
    calibration_example()

    # Visualization
    try:
        visualize_uncertainty()
    except Exception as e:
        print(f"\nVisualization skipped: {e}")


if __name__ == "__main__":
    main()
