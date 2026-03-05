"""
Volatility Forecasting Service

Backend service for volatility model training, prediction, and evaluation.
Integrates with the src/ volatility forecasting framework.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import threading
import time

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import settings

# Import from src/
HAS_VOLATILITY_MODULES = False

try:
    from src.models.volatility import (
        VolatilityLSTM,
        VolatilityGRU,
        VolatilityTransformer,
        VolatilityPINN,
        HestonPINN,
        StackedVolatilityPINN,
        create_volatility_model,
    )
    from src.models.volatility_baselines import (
        NaiveRollingVol,
        EWMA,
        GARCHModel,
        GJRGARCHModel,
        create_volatility_baseline,
    )
    from src.training.volatility_trainer import (
        VolatilityTrainer,
        VolatilityDataPreparer,
        VolatilityDataset,
        WalkForwardVolatilityValidator,
    )
    from src.evaluation.volatility_metrics import (
        VolatilityMetrics,
        EconomicVolatilityMetrics,
        VolatilityDiagnostics,
        evaluate_volatility_forecast,
        compare_volatility_models,
    )
    from src.trading.volatility_strategy import (
        VolatilityTargetingStrategy,
        StrategyResult,
    )
    from src.data.fetcher import DataFetcher
    from src.data.preprocessor import DataPreprocessor
    HAS_VOLATILITY_MODULES = True
    print("[VolatilityService] Successfully imported volatility modules")
except ImportError as e:
    print(f"[VolatilityService] WARNING: Failed to import volatility modules: {e}")


# Available volatility models
VOLATILITY_MODELS = {
    # Neural network models
    'vol_lstm': {
        'name': 'Volatility LSTM',
        'type': 'neural',
        'description': 'LSTM for volatility forecasting',
    },
    'vol_gru': {
        'name': 'Volatility GRU',
        'type': 'neural',
        'description': 'GRU for volatility forecasting',
    },
    'vol_transformer': {
        'name': 'Volatility Transformer',
        'type': 'neural',
        'description': 'Transformer for volatility forecasting',
    },
    'vol_pinn': {
        'name': 'Volatility PINN',
        'type': 'pinn',
        'description': 'PINN with GARCH and OU constraints',
    },
    'heston_pinn': {
        'name': 'Heston PINN',
        'type': 'pinn',
        'description': 'PINN based on Heston stochastic volatility',
    },
    'stacked_vol_pinn': {
        'name': 'Stacked Volatility PINN',
        'type': 'pinn',
        'description': 'Advanced stacked architecture',
    },
    # Baseline models
    'rolling': {
        'name': 'Rolling Volatility',
        'type': 'baseline',
        'description': 'Naive rolling window volatility',
    },
    'ewma': {
        'name': 'EWMA',
        'type': 'baseline',
        'description': 'Exponentially weighted moving average',
    },
    'garch': {
        'name': 'GARCH(1,1)',
        'type': 'baseline',
        'description': 'Standard GARCH model',
    },
    'gjr_garch': {
        'name': 'GJR-GARCH',
        'type': 'baseline',
        'description': 'Asymmetric GARCH for leverage effect',
    },
}


class VolatilityService:
    """
    Service for volatility forecasting operations.
    """

    def __init__(self):
        self.models_dir = settings.models_path
        self.results_dir = settings.results_path
        self.device = self._get_device()

        # Active training jobs
        self.training_jobs: Dict[str, Dict] = {}
        self.trained_models: Dict[str, Any] = {}

        # Data cache
        self._data_cache = None
        self._features_cache = None

    def _get_device(self) -> torch.device:
        """Get available compute device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def get_available_models(self) -> Dict[str, Any]:
        """Get all available volatility models."""
        return {
            'models': VOLATILITY_MODELS,
            'total': len(VOLATILITY_MODELS),
            'by_type': {
                'neural': len([m for m in VOLATILITY_MODELS.values() if m['type'] == 'neural']),
                'pinn': len([m for m in VOLATILITY_MODELS.values() if m['type'] == 'pinn']),
                'baseline': len([m for m in VOLATILITY_MODELS.values() if m['type'] == 'baseline']),
            },
            'has_modules': HAS_VOLATILITY_MODULES,
        }

    def prepare_data(
        self,
        ticker: str = 'SPY',
        start_date: str = '2015-01-01',
        end_date: Optional[str] = None,
        horizon: int = 5,
        seq_length: int = 40,
    ) -> Dict[str, Any]:
        """
        Prepare data for volatility forecasting.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            horizon: Forecast horizon
            seq_length: Sequence length

        Returns:
            Dictionary with prepared data info
        """
        if not HAS_VOLATILITY_MODULES:
            return {'error': 'Volatility modules not available'}

        # Fetch data
        fetcher = DataFetcher()
        df = fetcher.fetch_and_store(tickers=[ticker], start_date=start_date, end_date=end_date)

        if df is None or len(df) < 100:
            return {'error': 'Insufficient data'}

        # Compute features
        preparer = VolatilityDataPreparer(
            seq_length=seq_length,
            horizon=horizon,
        )

        features_df = preparer.compute_volatility_features(df)

        # Extract returns
        returns = features_df['log_return'].values

        # Prepare dataset
        features = features_df.values
        dataset = preparer.prepare(features, returns)

        # Cache data
        self._data_cache = {
            'ticker': ticker,
            'dataset': dataset,
            'features_df': features_df,
            'returns': returns,
            'dates': df.index.values,
            'horizon': horizon,
        }

        return {
            'ticker': ticker,
            'n_samples': len(features_df),
            'n_train': len(dataset.X_train),
            'n_val': len(dataset.X_val),
            'n_test': len(dataset.X_test) if dataset.X_test is not None else 0,
            'n_features': features.shape[1],
            'feature_names': list(features_df.columns),
            'horizon': horizon,
        }

    def train_model(
        self,
        model_type: str,
        ticker: str = 'SPY',
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        enable_physics: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train a volatility forecasting model.

        Args:
            model_type: Model type key
            ticker: Stock ticker
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            enable_physics: Enable physics losses for PINN models

        Returns:
            Training result dictionary
        """
        if not HAS_VOLATILITY_MODULES:
            return {'error': 'Volatility modules not available'}

        # Prepare data if not cached
        if self._data_cache is None or self._data_cache.get('ticker') != ticker:
            data_result = self.prepare_data(ticker)
            if 'error' in data_result:
                return data_result

        dataset = self._data_cache['dataset']
        input_dim = dataset.X_train.shape[-1]

        # Create model
        if model_type in ['rolling', 'ewma', 'garch', 'gjr_garch']:
            # Baseline model
            return self._train_baseline(model_type, **kwargs)

        # Neural network model
        model = create_volatility_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Create trainer
        trainer = VolatilityTrainer(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=epochs,
            device=self.device,
            checkpoint_dir=self.models_dir / 'volatility',
        )

        # Train
        start_time = time.time()
        history = trainer.fit(dataset, enable_physics=enable_physics)
        training_time = time.time() - start_time

        # Evaluate
        eval_results = trainer.evaluate(
            dataset.X_val,
            dataset.y_val,
            dataset.returns_val,
        )

        # Save model
        self.trained_models[model_type] = {
            'model': model,
            'trainer': trainer,
            'history': history,
        }

        return {
            'model_type': model_type,
            'epochs_trained': len(history.get('train_loss', [])),
            'training_time': training_time,
            'best_val_loss': trainer.best_val_loss,
            'metrics': eval_results,
            'history': {
                'train_loss': history.get('train_loss', []),
                'val_loss': history.get('val_loss', []),
                'val_qlike': history.get('val_qlike', []),
                'val_r2': history.get('val_r2', []),
            },
            'physics_params': model.get_learned_physics_params() if hasattr(model, 'get_learned_physics_params') else None,
        }

    def _train_baseline(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """Train a baseline volatility model."""
        returns = self._data_cache['returns']

        model = create_volatility_baseline(model_type, **kwargs)
        model.fit(returns)

        # Make predictions
        forecast = model.predict(returns)

        # Compute realized variance for evaluation
        horizon = self._data_cache.get('horizon', 5)
        squared_returns = returns ** 2
        realized_var = pd.Series(squared_returns).rolling(horizon).sum().shift(-horizon).values

        # Evaluate
        valid = ~np.isnan(realized_var)
        metrics = evaluate_volatility_forecast(
            predicted_var=forecast.variance[valid],
            realized_var=realized_var[valid],
            returns=returns[valid],
            model_name=model_type,
        )

        self.trained_models[model_type] = {
            'model': model,
            'forecast': forecast,
        }

        return {
            'model_type': model_type,
            'metrics': metrics,
            'params': forecast.params,
        }

    def predict(
        self,
        model_type: str,
        n_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Make volatility predictions.

        Args:
            model_type: Model type key
            n_steps: Number of steps to predict

        Returns:
            Prediction results
        """
        if model_type not in self.trained_models:
            return {'error': f'Model {model_type} not trained'}

        model_info = self.trained_models[model_type]

        if 'trainer' in model_info:
            # Neural network model
            trainer = model_info['trainer']
            dataset = self._data_cache['dataset']

            predictions = trainer.predict(dataset.X_val)

            return {
                'model_type': model_type,
                'predictions': predictions.flatten().tolist(),
                'shape': predictions.shape,
            }
        else:
            # Baseline model
            forecast = model_info['forecast']

            return {
                'model_type': model_type,
                'variance': forecast.variance.tolist(),
                'volatility': forecast.volatility.tolist(),
            }

    def backtest_strategy(
        self,
        model_type: str,
        target_vol: float = 0.15,
        min_weight: float = 0.25,
        max_weight: float = 2.0,
        transaction_cost: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Backtest volatility targeting strategy.

        Args:
            model_type: Model type for volatility predictions
            target_vol: Target annual volatility
            min_weight: Minimum position weight
            max_weight: Maximum leverage
            transaction_cost: Transaction cost

        Returns:
            Strategy backtest results
        """
        if not HAS_VOLATILITY_MODULES:
            return {'error': 'Volatility modules not available'}

        if model_type not in self.trained_models:
            return {'error': f'Model {model_type} not trained'}

        model_info = self.trained_models[model_type]
        returns = self._data_cache['returns']

        # Get volatility predictions
        if 'trainer' in model_info:
            trainer = model_info['trainer']
            dataset = self._data_cache['dataset']
            pred_var = trainer.predict(dataset.X_val).flatten()
            pred_vol = np.sqrt(np.maximum(pred_var, 1e-10))
            eval_returns = returns[-len(pred_vol):]
        else:
            forecast = model_info['forecast']
            pred_vol = forecast.volatility
            eval_returns = returns

        # Create strategy
        strategy = VolatilityTargetingStrategy(
            target_vol=target_vol,
            min_weight=min_weight,
            max_weight=max_weight,
            transaction_cost=transaction_cost,
        )

        # Backtest
        result = strategy.backtest(
            returns=eval_returns,
            predicted_vol=pred_vol,
            benchmark_returns=eval_returns,
        )

        return {
            'model_type': model_type,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'benchmark_sharpe': result.benchmark_sharpe,
            'information_ratio': result.information_ratio,
            'avg_leverage': result.avg_leverage,
            'turnover': result.turnover,
            'realized_vol': result.realized_vol,
            'vol_tracking_error': result.vol_tracking_error,
            'equity_curve': result.equity_curve.tolist(),
            'weights': result.weights.tolist(),
        }

    def compare_models(
        self,
        model_types: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple volatility models.

        Args:
            model_types: List of model types to compare

        Returns:
            Comparison results
        """
        if not HAS_VOLATILITY_MODULES:
            return {'error': 'Volatility modules not available'}

        results = []
        predictions = {}

        returns = self._data_cache['returns']
        horizon = self._data_cache.get('horizon', 5)

        # Compute realized variance
        squared_returns = returns ** 2
        realized_var = pd.Series(squared_returns).rolling(horizon).sum().shift(-horizon).values

        for model_type in model_types:
            if model_type not in self.trained_models:
                continue

            model_info = self.trained_models[model_type]

            if 'trainer' in model_info:
                trainer = model_info['trainer']
                dataset = self._data_cache['dataset']
                pred_var = trainer.predict(dataset.X_val).flatten()
                # Align with validation set
                start_idx = len(returns) - len(dataset.y_val)
                target_var = realized_var[start_idx:start_idx + len(pred_var)]
            else:
                forecast = model_info['forecast']
                pred_var = forecast.variance
                target_var = realized_var

            valid = ~np.isnan(target_var)
            predictions[model_type] = pred_var[valid]

            metrics = evaluate_volatility_forecast(
                predicted_var=pred_var[valid],
                realized_var=target_var[valid],
                model_name=model_type,
            )
            results.append(metrics)

        # Model confidence set
        if len(predictions) > 1:
            # Compute QLIKE losses
            losses = {}
            for name, pred in predictions.items():
                real = realized_var[:len(pred)]
                valid = ~np.isnan(real)
                pred_safe = np.maximum(pred[valid], 1e-10)
                real_safe = np.maximum(real[valid], 1e-10)
                losses[name] = real_safe / pred_safe - np.log(real_safe / pred_safe) - 1

            mcs_result = VolatilityDiagnostics.model_confidence_set(losses)
        else:
            mcs_result = None

        return {
            'results': results,
            'mcs': mcs_result,
            'best_qlike': min(results, key=lambda x: x.get('qlike', float('inf'))).get('model') if results else None,
            'best_r2': max(results, key=lambda x: x.get('r2', 0)).get('model') if results else None,
        }


# Singleton instance
_volatility_service: Optional[VolatilityService] = None


def get_volatility_service() -> VolatilityService:
    """Get or create volatility service instance."""
    global _volatility_service
    if _volatility_service is None:
        _volatility_service = VolatilityService()
    return _volatility_service
