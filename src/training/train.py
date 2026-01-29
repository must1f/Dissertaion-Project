"""
Main training script
"""

import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing

# Set multiprocessing start method for macOS compatibility
# This must be done before any other multiprocessing code
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.utils.reproducibility import set_seed, log_system_info, get_device
from src.utils.database import get_db
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import FinancialDataset, PhysicsAwareDataset, create_dataloaders
from src.models.baseline import LSTMModel, GRUModel, BiLSTMModel
from src.models.transformer import TransformerModel
from src.models.pinn import PINNModel
from src.training.trainer import Trainer

logger = get_logger(__name__)


def prepare_data(config):
    """
    Fetch and prepare data for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader, feature_cols)
    """
    logger.info("=" * 80)
    logger.info("DATA PREPARATION")
    logger.info("=" * 80)

    # Fetch data
    fetcher = DataFetcher(config)

    logger.info("Fetching stock data...")
    df = fetcher.fetch_and_store(
        tickers=config.data.tickers[:10],  # Use subset for faster training
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        force_refresh=False
    )

    if df.empty:
        logger.error("No data fetched! Exiting...")
        sys.exit(1)

    # Preprocess data
    preprocessor = DataPreprocessor(config)

    logger.info("Preprocessing data...")
    df_processed = preprocessor.process_and_store(df)

    # Define feature columns
    feature_cols = [
        'close', 'volume',
        'log_return', 'simple_return',
        'rolling_volatility_5', 'rolling_volatility_20',
        'momentum_5', 'momentum_20',
        'rsi_14', 'macd', 'macd_signal',
        'bollinger_upper', 'bollinger_lower', 'atr_14'
    ]

    # Filter available features
    feature_cols = [col for col in feature_cols if col in df_processed.columns]

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Temporal split
    train_df, val_df, test_df = preprocessor.split_temporal(df_processed)

    # Test stationarity
    logger.info("Testing stationarity...")
    stationarity_results = preprocessor.test_all_stationarity(train_df)
    stationary_returns = stationarity_results[
        (stationarity_results['series'] == 'log_return') &
        (stationarity_results['is_stationary'] == True)
    ]
    logger.info(f"Stationary returns: {len(stationary_returns)}/{len(train_df['ticker'].unique())}")

    # Normalize features
    logger.info("Normalizing features...")
    train_df_norm, scalers = preprocessor.normalize_features(
        train_df, feature_cols, method='standard'
    )

    # Apply same normalization to val and test
    for ticker in val_df['ticker'].unique():
        if ticker in scalers:
            val_mask = val_df['ticker'] == ticker
            val_df.loc[val_mask, feature_cols] = scalers[ticker].transform(
                val_df.loc[val_mask, feature_cols]
            )

    for ticker in test_df['ticker'].unique():
        if ticker in scalers:
            test_mask = test_df['ticker'] == ticker
            test_df.loc[test_mask, feature_cols] = scalers[ticker].transform(
                test_df.loc[test_mask, feature_cols]
            )

    # Create sequences
    logger.info("Creating sequences...")

    X_train, y_train, tickers_train = preprocessor.create_sequences(
        train_df_norm, feature_cols, target_col='close',
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon
    )

    X_val, y_val, tickers_val = preprocessor.create_sequences(
        val_df, feature_cols, target_col='close',
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon
    )

    X_test, y_test, tickers_test = preprocessor.create_sequences(
        test_df, feature_cols, target_col='close',
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon
    )

    logger.info(f"Train sequences: {X_train.shape}")
    logger.info(f"Val sequences: {X_val.shape}")
    logger.info(f"Test sequences: {X_test.shape}")

    # Create datasets (use PhysicsAwareDataset for PINN)
    train_dataset = FinancialDataset(X_train, y_train, tickers_train)
    val_dataset = FinancialDataset(X_val, y_val, tickers_val)
    test_dataset = FinancialDataset(X_test, y_test, tickers_test)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.training.batch_size
    )

    return train_loader, val_loader, test_loader, len(feature_cols)


def create_model(model_type: str, input_dim: int, config) -> torch.nn.Module:
    """
    Create model based on type

    Args:
        model_type: Model type ('lstm', 'gru', 'bilstm', 'transformer', 'pinn')
        input_dim: Input feature dimension
        config: Configuration object

    Returns:
        Model instance
    """
    logger.info(f"Creating {model_type} model...")

    if model_type == 'lstm':
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            output_dim=1,
            dropout=config.model.dropout,
            bidirectional=False
        )
    elif model_type == 'gru':
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            output_dim=1,
            dropout=config.model.dropout,
            bidirectional=False
        )
    elif model_type == 'bilstm':
        model = BiLSTMModel(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            output_dim=1,
            dropout=config.model.dropout
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config.model.hidden_dim,
            nhead=config.model.num_heads,
            num_encoder_layers=config.model.num_layers,
            dim_feedforward=config.model.feedforward_dim,
            dropout=config.model.dropout,
            output_dim=1
        )
    elif model_type == 'pinn':
        model = PINNModel(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            output_dim=1,
            dropout=config.model.dropout,
            base_model='lstm',
            lambda_gbm=config.training.lambda_gbm,
            lambda_ou=config.training.lambda_ou,
            lambda_langevin=config.training.lambda_langevin
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {n_params:,} parameters")

    return model


def main(args):
    """Main training function"""

    # Setup logging (only once per process)
    ensure_logger_initialized()

    logger.info("=" * 80)
    logger.info("PINN FINANCIAL FORECASTING - TRAINING")
    logger.info("=" * 80)

    # Load configuration
    config = get_config()

    # Set random seed for reproducibility
    set_seed(config.training.random_seed)

    # Log system info
    log_system_info()

    # Get device
    device = get_device(prefer_cuda=(config.training.device == 'cuda'))

    # Prepare data
    train_loader, val_loader, test_loader, input_dim = prepare_data(config)

    # Create model
    model = create_model(args.model, input_dim, config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    enable_physics = (args.model == 'pinn')

    history = trainer.train(
        epochs=args.epochs or config.training.epochs,
        enable_physics=enable_physics,
        save_best=True,
        model_name=args.model  # Pass model name so checkpoints save to /models/{model}_best.pt
    )

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    test_metrics = trainer.evaluate()

    # Save final results
    results_path = config.project_root / 'results' / f'{args.model}_results.json'
    results_path.parent.mkdir(exist_ok=True)

    import json
    import numpy as np

    def convert_to_python_types(obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(results_path, 'w') as f:
        json.dump({
            'model': args.model,
            'test_metrics': convert_to_python_types(test_metrics),
            'training_history': convert_to_python_types(history)
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train financial forecasting models')

    parser.add_argument(
        '--model',
        type=str,
        default='pinn',
        choices=['lstm', 'gru', 'bilstm', 'transformer', 'pinn'],
        help='Model type to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (uses config if not specified)'
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
