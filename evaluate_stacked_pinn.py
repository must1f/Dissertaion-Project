"""
Evaluate trained Stacked/Residual PINN models and save results for dashboard

This script loads the trained models and generates result files
in the format expected by the PINN comparison dashboard.
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.stacked_pinn import StackedPINN, ResidualPINN
from src.evaluation.financial_metrics import FinancialMetrics, compute_strategy_returns
from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.utils.reproducibility import set_seed, get_device
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor

ensure_logger_initialized()
logger = get_logger(__name__)


def prepare_test_data(config):
    """Prepare test data for evaluation"""
    fetcher = DataFetcher(config)
    df = fetcher.fetch_and_store(
        tickers=config.data.tickers[:10],
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        force_refresh=False
    )

    if df.empty:
        raise ValueError("No data fetched!")

    preprocessor = DataPreprocessor(config)
    df_processed = preprocessor.process_and_store(df)

    return_features = [
        'log_return', 'simple_return',
        'rolling_volatility_5', 'rolling_volatility_20', 'rolling_volatility_60',
        'momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal'
    ]

    feature_cols = [col for col in return_features if col in df_processed.columns]

    train_df, val_df, test_df = preprocessor.split_temporal(df_processed)

    train_df_norm, scalers = preprocessor.normalize_features(
        train_df, feature_cols, method='standard'
    )

    for ticker in test_df['ticker'].unique():
        if ticker in scalers:
            test_mask = test_df['ticker'] == ticker
            test_df.loc[test_mask, feature_cols] = scalers[ticker].transform(
                test_df.loc[test_mask, feature_cols]
            )

    seq_length = config.data.sequence_length
    X_test, y_test, _ = preprocessor.create_sequences(
        test_df, feature_cols, target_col='log_return',
        sequence_length=seq_length, forecast_horizon=1
    )

    return_idx = feature_cols.index('log_return') if 'log_return' in feature_cols else 0
    returns_test = X_test[:, :, return_idx]

    return X_test, y_test, returns_test, feature_cols


def load_model(model_type: str, input_dim: int, model_path: Path, config, device):
    """Load a trained model"""
    if model_type == 'stacked':
        model = StackedPINN(
            input_dim=input_dim,
            encoder_dim=config.model.hidden_dim,
            lstm_hidden_dim=config.model.hidden_dim,
            num_encoder_layers=2,
            num_rnn_layers=config.model.num_layers,
            prediction_hidden_dim=64,
            dropout=config.model.dropout,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
    else:
        model = ResidualPINN(
            input_dim=input_dim,
            base_model_type='lstm',
            base_hidden_dim=config.model.hidden_dim,
            correction_hidden_dim=64,
            num_base_layers=config.model.num_layers,
            num_correction_layers=2,
            dropout=config.model.dropout,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint.get('history', {})


@torch.no_grad()
def evaluate_model(model, X_test, y_test, device, batch_size=256):
    """Evaluate model and get predictions"""
    X_tensor = torch.FloatTensor(X_test)

    all_predictions = []

    for i in range(0, len(X_test), batch_size):
        batch_X = X_tensor[i:i+batch_size].to(device)

        if isinstance(model, StackedPINN):
            pred, _, _ = model(batch_X, compute_physics=False)
        else:
            pred, _, _ = model(batch_X, return_components=False)

        all_predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(all_predictions).flatten()
    targets = y_test.flatten()

    # ML metrics
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    dir_acc = FinancialMetrics.directional_accuracy(predictions, targets, are_returns=False)

    # Financial metrics
    strategy_returns = compute_strategy_returns(predictions, targets, transaction_cost=0.001)
    financial_metrics = FinancialMetrics.compute_all_metrics(
        returns=strategy_returns,
        predictions=predictions,
        targets=targets,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    return {
        'test_rmse': rmse,
        'test_mae': mae,
        'test_mape': float(np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100),
        'test_r2': r2,
        'test_directional_accuracy': dir_acc * 100,
        **financial_metrics
    }, predictions


def main():
    logger.info("=" * 80)
    logger.info("EVALUATING STACKED/RESIDUAL PINN MODELS")
    logger.info("=" * 80)

    config = get_config()
    set_seed(config.training.random_seed)
    device = get_device(prefer_cuda=(config.training.device == 'cuda'))

    # Prepare test data
    logger.info("Preparing test data...")
    X_test, y_test, returns_test, feature_cols = prepare_test_data(config)
    logger.info(f"Test data: X={X_test.shape}, y={y_test.shape}")

    models_dir = config.project_root / 'models' / 'stacked_pinn'
    results_dir = config.project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    models_to_evaluate = [
        ('stacked', 'stacked_pinn_best.pt', 'StackedPINN'),
        ('residual', 'residual_pinn_best.pt', 'ResidualPINN')
    ]

    for model_type, model_file, model_name in models_to_evaluate:
        model_path = models_dir / model_file

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        logger.info(f"\nEvaluating {model_name}...")

        # Load model
        model, history = load_model(model_type, len(feature_cols), model_path, config, device)

        # Evaluate
        metrics, predictions = evaluate_model(model, X_test, y_test, device)

        # Log results
        logger.info(f"  RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  MAE: {metrics['test_mae']:.4f}")
        logger.info(f"  R²: {metrics['test_r2']:.4f}")
        logger.info(f"  Directional Accuracy: {metrics['test_directional_accuracy']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")

        # Save results in dashboard-compatible format (matching compute_all_financial_metrics.py)
        from datetime import datetime
        result = {
            'model_name': model_name,
            'variant_key': model_type,
            'variant_name': model_name,
            'n_samples': len(y_test),
            'evaluation_timestamp': datetime.now().isoformat(),
            'configuration': {
                'lambda_gbm': 0.1,
                'lambda_ou': 0.1,
                'enable_physics': True,
                'model_type': model_type
            },
            # ML metrics in standard format (used by all_models_dashboard.py)
            'ml_metrics': {
                'mse': metrics['test_rmse'] ** 2,
                'rmse': metrics['test_rmse'],
                'mae': metrics['test_mae'],
                'r2': metrics['test_r2'],
                'mape': metrics['test_mape']
            },
            # Also include test_metrics for backward compatibility with pinn_dashboard.py
            'test_metrics': {
                'test_rmse': metrics['test_rmse'],
                'test_mae': metrics['test_mae'],
                'test_mape': metrics['test_mape'],
                'test_r2': metrics['test_r2'],
                'test_directional_accuracy': metrics['test_directional_accuracy']
            },
            'financial_metrics': {k: v for k, v in metrics.items() if not k.startswith('test_')},
            'history': {k: [float(v) if isinstance(v, (int, float, np.floating)) else v for v in vals]
                       for k, vals in history.items()} if history else {},
            'model_path': str(model_path)
        }

        # Save to results directory with standard naming
        result_path = results_dir / f'pinn_{model_type}_results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved: {result_path}")

        # Save predictions
        pred_path = results_dir / f'pinn_{model_type}_predictions.npz'
        np.savez(pred_path, predictions=predictions, targets=y_test.flatten())
        logger.info(f"  Saved: {pred_path}")

    # Update detailed_results.json to include stacked/residual
    detailed_path = results_dir / 'pinn_comparison' / 'detailed_results.json'
    if detailed_path.exists():
        logger.info("\nUpdating detailed_results.json...")
        with open(detailed_path, 'r') as f:
            detailed_results = json.load(f)

        # Add stacked/residual if not present
        existing_keys = {r.get('variant_key') for r in detailed_results}

        for model_type, _, _ in models_to_evaluate:
            if model_type not in existing_keys:
                result_path = results_dir / f'pinn_{model_type}_results.json'
                if result_path.exists():
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    detailed_results.append(result)
                    logger.info(f"  Added {model_type} to detailed_results.json")

        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
