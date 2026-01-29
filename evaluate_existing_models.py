#!/usr/bin/env python3
"""
Evaluate Existing Trained Models with Comprehensive Financial Metrics

This script loads already-trained models and computes comprehensive financial
metrics without retraining. Use this to add Sharpe ratio, Sortino ratio,
drawdown, profit factor, and other advanced metrics to existing models.

Usage:
    python evaluate_existing_models.py
    python evaluate_existing_models.py --models pinn  # Only PINN variants
    python evaluate_existing_models.py --models baseline  # Only baseline models
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.models.pinn import PINNModel
from src.models.baseline import LSTMModel, GRUModel, BiLSTMModel, AttentionLSTM
from src.models.transformer import TransformerModel
from src.evaluation.unified_evaluator import UnifiedModelEvaluator
from src.utils.logger import get_logger, ensure_logger_initialized

# Import data preparation function
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'training'))
from train import prepare_data

ensure_logger_initialized()
logger = get_logger(__name__)


def load_model_checkpoint(model_path: Path, model_type: str, config):
    """Load a trained model from checkpoint"""
    logger.info(f"Loading model from {model_path}")

    # Always use CPU for evaluation to avoid device compatibility issues
    device = 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config - use input_dim from function parameter or default
    # The input_dim is passed from prepare_data function
    hidden_size = getattr(config.model, 'hidden_dim', 128)
    num_layers = getattr(config.model, 'num_layers', 3)
    dropout = getattr(config.model, 'dropout', 0.2)
    num_heads = getattr(config.model, 'num_heads', 8)

    # Output is always 1 (predicting next price)
    output_size = 1

    # For input_size, we need to get it from checkpoint or use a default
    # Try to infer from checkpoint
    input_size = None
    if 'model_state_dict' in checkpoint:
        # Try to get input size from first layer
        for key in checkpoint['model_state_dict'].keys():
            if 'weight' in key and ('lstm' in key.lower() or 'gru' in key.lower() or 'linear' in key.lower()):
                weight_shape = checkpoint['model_state_dict'][key].shape
                if len(weight_shape) >= 2:
                    input_size = weight_shape[-1]  # Last dimension is usually input
                    break

    if input_size is None:
        # Default to 14 features (common for financial data)
        input_size = 14
        logger.warning(f"Could not infer input_size from checkpoint, using default: {input_size}")

    # Determine model class
    if 'pinn' in model_path.name or model_type == 'pinn':
        model = PINNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif 'lstm' in model_path.name or model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif 'gru' in model_path.name or model_type == 'gru':
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif 'bilstm' in model_path.name or model_type == 'bilstm':
        model = BiLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif 'attention' in model_path.name or model_type == 'attention_lstm':
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif 'transformer' in model_path.name or model_type == 'transformer':
        model = TransformerModel(
            input_size=input_size,
            d_model=hidden_size,
            nhead=num_heads,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    else:
        logger.error(f"Unknown model type for {model_path}")
        return None

    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Use CPU for evaluation
    device = 'cpu'
    model.to(device)
    model.eval()

    return model


def get_predictions(model, test_loader, config):
    """Get predictions from a model"""
    all_predictions = []
    all_targets = []

    # Use CPU for evaluation
    device = 'cpu'

    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)

            # Get predictions
            predictions = model(X_batch)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def evaluate_model(model_path: Path, model_key: str, model_name: str, test_loader, config):
    """Evaluate a single model with comprehensive metrics"""
    logger.info("=" * 80)
    logger.info(f"EVALUATING: {model_name}")
    logger.info("=" * 80)

    # Load model
    model = load_model_checkpoint(model_path, model_key, config)
    if model is None:
        logger.error(f"Failed to load model: {model_key}")
        return None

    # Get predictions
    logger.info("Generating predictions on test set...")
    predictions, targets = get_predictions(model, test_loader, config)
    logger.info(f"Generated {len(predictions)} predictions")

    # Comprehensive evaluation
    evaluator = UnifiedModelEvaluator(
        transaction_cost=0.001,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    results = evaluator.evaluate_model(
        predictions=predictions,
        targets=targets,
        model_name=model_name,
        compute_rolling=True,
        rolling_window_size=63
    )

    # Save results
    output_dir = config.project_root / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{model_key}_results.json'

    # Convert to serializable format
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    results_serializable = convert_types(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"✓ Results saved to {output_path}")
    logger.info("")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate existing trained models')
    parser.add_argument('--models', type=str, default='all',
                       choices=['all', 'baseline', 'pinn', 'advanced'],
                       help='Which models to evaluate')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip models that already have comprehensive results')

    args = parser.parse_args()

    config = get_config()
    models_dir = config.project_root / 'models'
    results_dir = config.project_root / 'results'

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION OF EXISTING MODELS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Filter: {args.models}")
    logger.info("")

    # Load test dataset using the same preparation as training
    logger.info("Loading and preparing test dataset...")
    try:
        _, _, test_loader, input_dim = prepare_data(config)
        logger.info(f"✓ Test set ready with {len(test_loader.dataset)} samples")
        logger.info(f"✓ Input dimension: {input_dim}")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        logger.error("Make sure you have run data fetching first:")
        logger.error("  python -m src.data.fetcher")
        return

    # Define models to evaluate
    model_configs = {
        'lstm': ('LSTM', 'baseline'),
        'gru': ('GRU', 'baseline'),
        'bilstm': ('BiLSTM', 'baseline'),
        'attention_lstm': ('Attention LSTM', 'baseline'),
        'transformer': ('Transformer', 'baseline'),
        'pinn_baseline': ('PINN Baseline', 'pinn'),
        'pinn_gbm': ('PINN GBM', 'pinn'),
        'pinn_ou': ('PINN OU', 'pinn'),
        'pinn_black_scholes': ('PINN Black-Scholes', 'pinn'),
        'pinn_gbm_ou': ('PINN GBM+OU', 'pinn'),
        'pinn_global': ('PINN Global', 'pinn'),
    }

    # Filter by model type
    if args.models != 'all':
        model_configs = {k: v for k, v in model_configs.items() if v[1] == args.models}

    # Find and evaluate models
    evaluated = 0
    skipped = 0
    failed = 0

    for model_key, (model_name, model_type) in model_configs.items():
        # Check if model file exists
        model_path = models_dir / f'{model_key}_best.pt'
        if not model_path.exists():
            logger.info(f"⊘ Skipping {model_name}: model file not found")
            continue

        # Check if results already exist
        results_path = results_dir / f'{model_key}_results.json'
        if args.skip_existing and results_path.exists():
            # Check if it has financial_metrics
            try:
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                    if 'financial_metrics' in existing_results:
                        logger.info(f"⊙ Skipping {model_name}: comprehensive results already exist")
                        skipped += 1
                        continue
            except:
                pass

        # Evaluate model
        try:
            results = evaluate_model(model_path, model_key, model_name, test_loader, config)
            if results:
                evaluated += 1
                logger.info(f"✓ {model_name} evaluation complete")
            else:
                failed += 1
                logger.error(f"✗ {model_name} evaluation failed")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {model_name} evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Evaluated: {evaluated}")
    logger.info(f"⊙ Skipped: {skipped}")
    logger.info(f"✗ Failed: {failed}")
    logger.info("")
    logger.info("Results saved to: results/*_results.json")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Launch dashboard: streamlit run src/web/app.py")
    logger.info("  2. Navigate to 'PINN Comparison' or 'All Models Dashboard'")
    logger.info("  3. View comprehensive financial metrics")
    logger.info("")


if __name__ == '__main__':
    main()
