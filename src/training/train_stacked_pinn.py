"""
Training Script for Stacked Physics-Informed Neural Networks

Features:
- StackedPINN and ResidualPINN architectures
- Curriculum training with gradually increasing physics weights
- Walk-forward validation
- Financial metrics evaluation (Sharpe, drawdown, PnL, directional accuracy)
- Return-based features only (no price-level constraints)
- Combined loss: prediction + physics
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.stacked_pinn import StackedPINN, ResidualPINN
from src.training.curriculum import CurriculumScheduler, AdaptiveCurriculumScheduler
from src.training.walk_forward import WalkForwardValidator
from src.evaluation.financial_metrics import FinancialMetrics, compute_strategy_returns
from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.utils.reproducibility import set_seed, get_device
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor

logger = get_logger(__name__)


def prepare_return_based_data(config):
    """
    Prepare return-based features (no price levels)

    Returns:
        Tuple of (X_train, y_train, returns_train, X_val, y_val, returns_val, feature_names)
    """
    logger.info("=" * 80)
    logger.info("RETURN-BASED DATA PREPARATION")
    logger.info("=" * 80)

    # Fetch data
    fetcher = DataFetcher(config)
    df = fetcher.fetch_and_store(
        tickers=config.data.tickers[:10],
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        force_refresh=False
    )

    if df.empty:
        logger.error("No data fetched!")
        sys.exit(1)

    # Preprocess
    preprocessor = DataPreprocessor(config)
    df_processed = preprocessor.process_and_store(df)

    # Define RETURN-BASED features only (no prices!)
    return_features = [
        'log_return',
        'simple_return',
        'rolling_volatility_5',
        'rolling_volatility_20',
        'rolling_volatility_60',
        'momentum_5',
        'momentum_20',
        'rsi_14',
        'macd',
        'macd_signal'
    ]

    # Filter available features
    feature_cols = [col for col in return_features if col in df_processed.columns]

    logger.info(f"Using {len(feature_cols)} return-based features: {feature_cols}")
    logger.info("NOTE: No price-level features - physics on returns only!")

    # Temporal split
    train_df, val_df, test_df = preprocessor.split_temporal(df_processed)

    # Normalize
    train_df_norm, scalers = preprocessor.normalize_features(
        train_df, feature_cols, method='standard'
    )

    for ticker in val_df['ticker'].unique():
        if ticker in scalers:
            val_mask = val_df['ticker'] == ticker
            val_df.loc[val_mask, feature_cols] = scalers[ticker].transform(
                val_df.loc[val_mask, feature_cols]
            )

    # Create sequences
    seq_length = config.data.sequence_length
    forecast_horizon = 1  # Predict next period return

    X_train, y_train, _ = preprocessor.create_sequences(
        train_df_norm, feature_cols, target_col='log_return',
        sequence_length=seq_length, forecast_horizon=forecast_horizon
    )

    X_val, y_val, _ = preprocessor.create_sequences(
        val_df, feature_cols, target_col='log_return',
        sequence_length=seq_length, forecast_horizon=forecast_horizon
    )

    # Extract return sequences for physics losses
    # Returns are the log_return feature in the sequence
    return_idx = feature_cols.index('log_return') if 'log_return' in feature_cols else 0
    returns_train = X_train[:, :, return_idx]  # (n_samples, seq_length)
    returns_val = X_val[:, :, return_idx]

    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}, returns={returns_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}, returns={returns_val.shape}")

    return X_train, y_train, returns_train, X_val, y_val, returns_val, feature_cols


def create_model(
    model_type: str,
    input_dim: int,
    config,
    initial_lambda_gbm: float = 0.0,
    initial_lambda_ou: float = 0.0
) -> nn.Module:
    """
    Create StackedPINN or ResidualPINN model

    Args:
        model_type: 'stacked' or 'residual'
        input_dim: Input feature dimension
        config: Configuration object
        initial_lambda_gbm: Initial GBM weight
        initial_lambda_ou: Initial OU weight

    Returns:
        Model instance
    """
    logger.info(f"Creating {model_type} PINN model...")

    if model_type == 'stacked':
        model = StackedPINN(
            input_dim=input_dim,
            encoder_dim=config.model.hidden_dim,
            lstm_hidden_dim=config.model.hidden_dim,
            num_encoder_layers=2,
            num_rnn_layers=config.model.num_layers,
            prediction_hidden_dim=64,
            dropout=config.model.dropout,
            lambda_gbm=initial_lambda_gbm,
            lambda_ou=initial_lambda_ou
        )
    elif model_type == 'residual':
        model = ResidualPINN(
            input_dim=input_dim,
            base_model_type='lstm',
            base_hidden_dim=config.model.hidden_dim,
            correction_hidden_dim=64,
            num_base_layers=config.model.num_layers,
            num_correction_layers=2,
            dropout=config.model.dropout,
            lambda_gbm=initial_lambda_gbm,
            lambda_ou=initial_lambda_ou
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {n_params:,} parameters")

    return model


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    returns_batch: torch.Tensor,
    device: torch.device,
    curriculum_weights: Dict[str, float],
    enable_physics: bool = True
) -> Dict[str, float]:
    """
    Train for one batch

    Returns:
        Dict with losses
    """
    model.train()

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    returns_batch = returns_batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    if isinstance(model, StackedPINN):
        return_pred, direction_logits, _ = model(X_batch, compute_physics=True)
    else:  # ResidualPINN
        return_pred, direction_logits, _ = model(X_batch, return_components=False)

    # Prediction losses
    # 1. Regression loss (return prediction)
    regression_loss = nn.functional.mse_loss(return_pred, y_batch)

    # 2. Classification loss (direction)
    direction_targets = (y_batch > 0).long().squeeze()
    classification_loss = nn.functional.cross_entropy(direction_logits, direction_targets)

    # Combined prediction loss
    prediction_loss = regression_loss + 0.1 * classification_loss

    # Physics loss
    if enable_physics:
        physics_loss, physics_dict = model.compute_physics_loss(X_batch, returns_batch)

        # Apply curriculum weights
        physics_loss = (
            curriculum_weights['lambda_gbm'] * physics_dict.get('gbm_loss', 0.0) +
            curriculum_weights['lambda_ou'] * physics_dict.get('ou_loss', 0.0)
        )
        physics_loss = torch.tensor(physics_loss, device=device)
    else:
        physics_loss = torch.tensor(0.0, device=device)
        physics_dict = {}

    # Total loss
    total_loss = prediction_loss + physics_loss

    # Backward
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Return metrics
    return {
        'total_loss': total_loss.item(),
        'prediction_loss': prediction_loss.item(),
        'regression_loss': regression_loss.item(),
        'classification_loss': classification_loss.item(),
        'physics_loss': physics_loss.item(),
        **physics_dict
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    returns: torch.Tensor,
    device: torch.device,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model

    Returns:
        predictions, direction_probs, metrics_dict
    """
    model.eval()

    n_samples = X.shape[0]
    all_predictions = []
    all_direction_probs = []

    for i in range(0, n_samples, batch_size):
        batch_X = X[i:i+batch_size].to(device)

        if isinstance(model, StackedPINN):
            pred, dir_logits, _ = model(batch_X, compute_physics=False)
        else:
            pred, dir_logits, _ = model(batch_X, return_components=False)

        all_predictions.append(pred.cpu().numpy())

        # Direction probabilities
        dir_probs = torch.softmax(dir_logits, dim=-1)
        all_direction_probs.append(dir_probs.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    direction_probs = np.concatenate(all_direction_probs)

    # Compute metrics
    y_np = y.numpy().flatten()
    pred_np = predictions.flatten()

    mse = float(np.mean((pred_np - y_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - y_np)))
    # Compare direction of price CHANGES (not price levels)
    # Since predictions and targets are normalized prices, use are_returns=False
    directional_acc = FinancialMetrics.directional_accuracy(pred_np, y_np, are_returns=False)

    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': float(np.sqrt(mse)),
        'directional_accuracy': directional_acc
    }

    return predictions, direction_probs, metrics


def train_with_curriculum(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    curriculum: CurriculumScheduler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    returns_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    returns_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    save_path: Path
) -> Dict:
    """
    Train with curriculum learning

    Returns:
        Training history
    """
    logger.info(f"Starting curriculum training for {epochs} epochs")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if len(y_train.shape) == 1 else torch.FloatTensor(y_train)
    returns_train_tensor = torch.FloatTensor(returns_train)

    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1) if len(y_val.shape) == 1 else torch.FloatTensor(y_val)
    returns_val_tensor = torch.FloatTensor(returns_val)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_directional_acc': [],
        'val_directional_acc': [],
        'lambda_gbm': [],
        'lambda_ou': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Get curriculum weights
        curriculum_weights = curriculum.step(epoch)
        history['lambda_gbm'].append(curriculum_weights['lambda_gbm'])
        history['lambda_ou'].append(curriculum_weights['lambda_ou'])

        logger.info(f"\nEpoch {epoch+1}/{epochs} - λ_gbm={curriculum_weights['lambda_gbm']:.4f}, "
                   f"λ_ou={curriculum_weights['lambda_ou']:.4f}")

        # Training
        n_train = X_train.shape[0]
        indices = torch.randperm(n_train)

        epoch_losses = []

        with tqdm(range(0, n_train, batch_size), desc="Training") as pbar:
            for i in pbar:
                batch_indices = indices[i:min(i+batch_size, n_train)]

                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]
                batch_returns = returns_train_tensor[batch_indices]

                losses = train_epoch(
                    model, optimizer, batch_X, batch_y, batch_returns,
                    device, curriculum_weights, enable_physics=True
                )

                epoch_losses.append(losses)
                pbar.set_postfix({'loss': f"{losses['total_loss']:.4f}"})

        # Average epoch losses
        avg_train_loss = np.mean([l['total_loss'] for l in epoch_losses])
        history['train_loss'].append(avg_train_loss)

        # Validation
        val_pred, _, val_metrics = evaluate(
            model, X_val_tensor, y_val_tensor, returns_val_tensor, device
        )

        history['val_loss'].append(val_metrics['mse'])
        history['train_directional_acc'].append(np.mean([l.get('directional_accuracy', 0.0) for l in epoch_losses]))
        history['val_directional_acc'].append(val_metrics['directional_accuracy'])

        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val MSE: {val_metrics['mse']:.4f} | "
                   f"Val Dir Acc: {val_metrics['directional_accuracy']:.3f}")

        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'history': history
            }, save_path)
            logger.info(f"✓ Best model saved (val_loss={best_val_loss:.4f})")

        # LR scheduler
        scheduler.step(val_metrics['mse'])

    logger.info(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    return history


def evaluate_financial_performance(
    predictions: np.ndarray,
    actual_returns: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate using financial metrics

    Args:
        predictions: Predicted returns
        actual_returns: Actual returns

    Returns:
        Dict with financial metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("FINANCIAL PERFORMANCE EVALUATION")
    logger.info("=" * 80)

    # Compute strategy returns
    strategy_returns = compute_strategy_returns(predictions, actual_returns, transaction_cost=0.001)

    # Compute all metrics
    metrics = FinancialMetrics.compute_all_metrics(
        returns=strategy_returns,
        predictions=predictions,
        targets=actual_returns,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    # Log results
    logger.info("\nFinancial Metrics:")
    logger.info(f"  Total Return:         {metrics['total_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Sortino Ratio:        {metrics['sortino_ratio']:.3f}")
    logger.info(f"  Max Drawdown:         {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Calmar Ratio:         {metrics['calmar_ratio']:.3f}")
    logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']*100:.1f}%")
    logger.info(f"  Win Rate:             {metrics['win_rate']*100:.1f}%")
    logger.info(f"  Volatility (Annual):  {metrics['volatility']*100:.2f}%")

    return metrics


def main(args):
    """Main training function"""
    ensure_logger_initialized()

    logger.info("=" * 80)
    logger.info("STACKED PINN TRAINING - RETURN-BASED FEATURES")
    logger.info("=" * 80)

    # Configuration
    config = get_config()
    set_seed(config.training.random_seed)
    device = get_device(prefer_cuda=(config.training.device == 'cuda'))

    # Prepare data
    X_train, y_train, returns_train, X_val, y_val, returns_val, feature_cols = prepare_return_based_data(config)

    # Create model
    model = create_model(
        model_type=args.model_type,
        input_dim=len(feature_cols),
        config=config,
        initial_lambda_gbm=0.0,  # Start at 0, curriculum will increase
        initial_lambda_ou=0.0
    )
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=1e-5
    )

    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Curriculum scheduler
    curriculum = CurriculumScheduler(
        initial_lambda_gbm=0.0,
        final_lambda_gbm=args.final_lambda_gbm,
        initial_lambda_ou=0.0,
        final_lambda_ou=args.final_lambda_ou,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        strategy=args.curriculum_strategy
    )

    # Save path
    save_dir = config.project_root / 'models' / 'stacked_pinn'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{args.model_type}_pinn_best.pt'

    # Train
    history = train_with_curriculum(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        curriculum=curriculum,
        X_train=X_train,
        y_train=y_train,
        returns_train=returns_train,
        X_val=X_val,
        y_val=y_val,
        returns_val=returns_val,
        device=device,
        epochs=args.epochs,
        batch_size=config.training.batch_size,
        save_path=save_path
    )

    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    val_predictions, _, val_metrics = evaluate(
        model,
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.FloatTensor(returns_val),
        device
    )

    # Financial metrics
    financial_metrics = evaluate_financial_performance(
        predictions=val_predictions.flatten(),
        actual_returns=y_val.flatten()
    )

    # Save results
    results = {
        'model_type': args.model_type,
        'val_metrics': val_metrics,
        'financial_metrics': financial_metrics,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }

    results_path = save_dir / f'{args.model_type}_pinn_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stacked PINN')

    parser.add_argument('--model-type', type=str, default='stacked',
                       choices=['stacked', 'residual'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                       help='Warmup epochs (no physics)')
    parser.add_argument('--final-lambda-gbm', type=float, default=0.1,
                       help='Final GBM weight')
    parser.add_argument('--final-lambda-ou', type=float, default=0.1,
                       help='Final OU weight')
    parser.add_argument('--curriculum-strategy', type=str, default='cosine',
                       choices=['linear', 'exponential', 'cosine', 'step'],
                       help='Curriculum strategy')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
