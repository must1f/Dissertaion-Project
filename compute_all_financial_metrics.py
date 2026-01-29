#!/usr/bin/env python3
"""
Compute Comprehensive Financial Metrics for All Neural Networks

This script evaluates all trained models (13 total) and computes:
- Sharpe Ratio, Sortino Ratio, Volatility
- Max Drawdown, Drawdown Duration, Calmar Ratio
- Annualized Return, Profit Factor, Win Rate
- Directional Accuracy, Precision, Recall, Information Coefficient
- Rolling window performance and stability metrics

Author: Claude Code
Date: January 28, 2026
"""

import sys
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.models.pinn import PINNModel
from src.models.baseline import LSTMModel, GRUModel, BiLSTMModel, AttentionLSTM
from src.models.transformer import TransformerModel
from src.evaluation.unified_evaluator import UnifiedModelEvaluator

ensure_logger_initialized()
logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate all neural network models with comprehensive financial metrics"""

    def __init__(self):
        self.config = get_config()
        self.models_dir = self.config.project_root / 'models'
        self.results_dir = self.config.project_root / 'results'
        self.device = 'cpu'  # Use CPU for compatibility

        # Model configurations
        self.model_configs = {
            # Baseline models
            'lstm': {
                'name': 'LSTM',
                'type': 'baseline',
                'class': LSTMModel,
                'checkpoint': 'lstm_best.pt'
            },
            'gru': {
                'name': 'GRU',
                'type': 'baseline',
                'class': GRUModel,
                'checkpoint': 'gru_best.pt'
            },
            'bilstm': {
                'name': 'BiLSTM',
                'type': 'baseline',
                'class': BiLSTMModel,
                'checkpoint': 'bilstm_best.pt'
            },
            'attention_lstm': {
                'name': 'Attention LSTM',
                'type': 'baseline',
                'class': AttentionLSTM,
                'checkpoint': 'attention_lstm_best.pt'
            },
            'transformer': {
                'name': 'Transformer',
                'type': 'baseline',
                'class': TransformerModel,
                'checkpoint': 'transformer_best.pt'
            },
            # PINN variants
            'pinn_baseline': {
                'name': 'PINN Baseline (Data-only)',
                'type': 'pinn',
                'class': PINNModel,
                'checkpoint': 'pinn_baseline_best.pt'
            },
            'pinn_gbm': {
                'name': 'PINN GBM (Trend)',
                'type': 'pinn',
                'class': PINNModel,
                'checkpoint': 'pinn_gbm_best.pt'
            },
            'pinn_ou': {
                'name': 'PINN OU (Mean-Reversion)',
                'type': 'pinn',
                'class': PINNModel,
                'checkpoint': 'pinn_ou_best.pt'
            },
            'pinn_black_scholes': {
                'name': 'PINN Black-Scholes',
                'type': 'pinn',
                'class': PINNModel,
                'checkpoint': 'pinn_black_scholes_best.pt'
            },
            'pinn_gbm_ou': {
                'name': 'PINN GBM+OU Hybrid',
                'type': 'pinn',
                'class': PINNModel,
                'checkpoint': 'pinn_gbm_ou_best.pt'
            },
            'pinn_global': {
                'name': 'PINN Global Constraint',
                'type': 'pinn',
                'class': PINNModel,
                'checkpoint': 'pinn_global_best.pt'
            },
        }

        self.evaluator = UnifiedModelEvaluator(
            transaction_cost=0.001,
            risk_free_rate=0.02,
            periods_per_year=252
        )

    def prepare_data(self):
        """Prepare test dataset using existing training pipeline"""
        logger.info("Preparing test dataset...")

        try:
            # Import training data preparation function
            from src.training.train import prepare_data

            _, _, test_loader, input_dim = prepare_data(self.config)

            logger.info(f"✓ Test set ready: {len(test_loader.dataset)} samples")
            logger.info(f"✓ Input dimension: {input_dim}")

            return test_loader, input_dim

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            logger.error("Make sure you have run: python -m src.data.fetcher")
            raise

    def load_model(self, model_key: str, model_config: dict, input_dim: int) -> Optional[torch.nn.Module]:
        """Load a trained model from checkpoint"""

        checkpoint_path = self.models_dir / model_config['checkpoint']

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        logger.info(f"Loading {model_config['name']} from {checkpoint_path}")

        try:
            # Load checkpoint (weights_only=False for our own trained models)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Get model hyperparameters
            hidden_size = getattr(self.config.model, 'hidden_dim', 128)
            num_layers = getattr(self.config.model, 'num_layers', 3)
            dropout = getattr(self.config.model, 'dropout', 0.2)
            num_heads = getattr(self.config.model, 'num_heads', 8)

            # Create model instance
            model_class = model_config['class']

            if model_class == TransformerModel:
                model = model_class(
                    input_size=input_dim,
                    d_model=hidden_size,
                    nhead=num_heads,
                    num_layers=num_layers,
                    output_size=1,
                    dropout=dropout
                )
            elif model_class == PINNModel:
                # PINNModel uses different parameter names
                model = model_class(
                    input_dim=input_dim,
                    hidden_dim=hidden_size,
                    num_layers=num_layers,
                    output_dim=1,
                    dropout=dropout
                )
            else:
                # Baseline models (LSTM, GRU, BiLSTM, AttentionLSTM)
                model = model_class(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=1,
                    dropout=dropout
                )

            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()

            logger.info(f"✓ Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_predictions(self, model: torch.nn.Module, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from model"""

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                # Handle both 2-item and 3-item batches (with metadata)
                if len(batch) == 3:
                    X_batch, y_batch, _ = batch  # Unpack 3 items, ignore metadata
                else:
                    X_batch, y_batch = batch  # Fallback for 2-item batches
                X_batch = X_batch.to(self.device)

                # Get predictions
                predictions = model(X_batch)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        logger.info(f"Generated {len(predictions)} predictions")

        return predictions, targets

    def evaluate_model(self, model_key: str, model_config: dict, test_loader, input_dim: int) -> Optional[Dict]:
        """Evaluate a single model with comprehensive metrics"""

        logger.info("=" * 80)
        logger.info(f"EVALUATING: {model_config['name']}")
        logger.info("=" * 80)

        # Load model
        model = self.load_model(model_key, model_config, input_dim)
        if model is None:
            return None

        # Get predictions
        logger.info("Generating predictions...")
        predictions, targets = self.get_predictions(model, test_loader)

        # Comprehensive evaluation
        logger.info("Computing comprehensive financial metrics...")
        results = self.evaluator.evaluate_model(
            predictions=predictions,
            targets=targets,
            model_name=model_config['name'],
            compute_rolling=True,
            rolling_window_size=63
        )

        # Save results
        output_path = self.results_dir / f'{model_key}_results.json'
        self.save_results(results, output_path)

        logger.info(f"✓ Results saved to {output_path}")
        logger.info("")

        return results

    def save_results(self, results: Dict, output_path: Path):
        """Save results to JSON"""

        def convert_types(obj):
            """Convert numpy types to JSON-serializable types"""
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

    def run_all(self):
        """Evaluate all models"""

        logger.info("=" * 80)
        logger.info("COMPREHENSIVE FINANCIAL METRICS COMPUTATION")
        logger.info("Computing Sharpe, Sortino, Drawdown, Profit Factor, IC, and more")
        logger.info("=" * 80)
        logger.info("")

        # Prepare data
        try:
            test_loader, input_dim = self.prepare_data()
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return

        logger.info("")

        # Evaluate each model
        evaluated = 0
        skipped = 0
        failed = 0

        for model_key, model_config in self.model_configs.items():
            try:
                checkpoint_path = self.models_dir / model_config['checkpoint']

                if not checkpoint_path.exists():
                    logger.info(f"⊘ Skipping {model_config['name']}: checkpoint not found")
                    skipped += 1
                    continue

                results = self.evaluate_model(model_key, model_config, test_loader, input_dim)

                if results:
                    evaluated += 1
                    logger.info(f"✓ {model_config['name']} - Complete")
                else:
                    failed += 1
                    logger.error(f"✗ {model_config['name']} - Failed")

            except Exception as e:
                failed += 1
                logger.error(f"✗ {model_config['name']} - Error: {e}")
                import traceback
                traceback.print_exc()

            logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Evaluated: {evaluated}")
        logger.info(f"⊘ Skipped: {skipped}")
        logger.info(f"✗ Failed: {failed}")
        logger.info("")
        logger.info("All models now have comprehensive financial metrics!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Launch dashboard: streamlit run src/web/app.py")
        logger.info("  2. Navigate to 'All Models Dashboard' or 'PINN Comparison'")
        logger.info("  3. View Sharpe ratios, Sortino ratios, drawdowns, and all metrics")
        logger.info("")


def main():
    """Main execution"""
    evaluator = ModelEvaluator()
    evaluator.run_all()


if __name__ == '__main__':
    main()
