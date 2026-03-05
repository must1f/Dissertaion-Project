"""
Volatility Forecasting Training Pipeline

Training infrastructure specifically designed for volatility forecasting models,
including proper target construction, physics-informed loss computation,
and comprehensive evaluation.

Key Features:
    1. Proper volatility target construction (h-day ahead realized variance)
    2. Look-ahead bias prevention
    3. Physics-informed loss integration
    4. Walk-forward validation support
    5. Early stopping and learning rate scheduling
    6. Comprehensive metric tracking
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logger import get_logger
from ..constants import TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)


@dataclass
class VolatilityDataset:
    """Container for volatility forecasting dataset."""
    X_train: torch.Tensor  # [n_train, seq_len, features]
    y_train: torch.Tensor  # [n_train, horizon]
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: Optional[torch.Tensor] = None
    y_test: Optional[torch.Tensor] = None

    # Metadata for physics losses
    returns_train: Optional[torch.Tensor] = None
    returns_val: Optional[torch.Tensor] = None
    variance_history_train: Optional[torch.Tensor] = None
    variance_history_val: Optional[torch.Tensor] = None

    # Scalers
    feature_scaler: Optional[object] = None
    target_scaler: Optional[object] = None

    # Info
    feature_names: Optional[List[str]] = None
    dates_train: Optional[np.ndarray] = None
    dates_val: Optional[np.ndarray] = None


class VolatilityDataPreparer:
    """
    Data preparation for volatility forecasting.

    Handles:
        1. Feature engineering for volatility prediction
        2. Target construction (realized variance)
        3. Sequence creation
        4. Train/validation/test split
        5. Look-ahead bias prevention
    """

    def __init__(
        self,
        seq_length: int = 40,
        horizon: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scale_features: bool = True,
        scale_targets: bool = False,
    ):
        """
        Initialize data preparer.

        Args:
            seq_length: Input sequence length
            horizon: Forecast horizon (days)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            scale_features: Whether to standardize features
            scale_targets: Whether to scale targets
        """
        self.seq_length = seq_length
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scale_features = scale_features
        self.scale_targets = scale_targets

        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    def compute_volatility_features(
        self,
        df: pd.DataFrame,
        return_col: str = 'log_return',
        price_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Compute volatility-relevant features.

        Args:
            df: DataFrame with price data
            return_col: Column name for returns
            price_col: Column name for prices

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)

        # Returns
        if return_col not in df.columns:
            features['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        else:
            features['log_return'] = df[return_col]

        features['abs_return'] = features['log_return'].abs()
        features['squared_return'] = features['log_return'] ** 2

        # Rolling volatility (lagged to avoid look-ahead)
        for window in [5, 10, 20, 60]:
            features[f'rv_{window}d'] = features['squared_return'].rolling(window).sum().shift(1)
            features[f'vol_{window}d'] = features['log_return'].rolling(window).std().shift(1)

        # Volatility ratios
        features['vol_ratio_5_20'] = features['vol_5d'] / features['vol_20d']
        features['vol_ratio_20_60'] = features['vol_20d'] / features['vol_60d']

        # Range-based estimators (if OHLC available)
        if all(col in df.columns for col in ['high', 'low']):
            # Parkinson volatility
            features['parkinson'] = np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
            features['parkinson_ma'] = features['parkinson'].rolling(5).mean().shift(1)

        if all(col in df.columns for col in ['high', 'low', 'close', 'open']):
            # Garman-Klass volatility
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            features['garman_klass'] = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
            features['garman_klass_ma'] = features['garman_klass'].rolling(5).mean().shift(1)

        # Return momentum
        features['return_ma_5'] = features['log_return'].rolling(5).mean().shift(1)
        features['return_ma_20'] = features['log_return'].rolling(20).mean().shift(1)

        # Sign of recent returns (for leverage effect)
        features['negative_return_count'] = (
            features['log_return'].rolling(5).apply(lambda x: (x < 0).sum())
        ).shift(1)

        return features.dropna()

    def create_target(
        self,
        returns: np.ndarray,
        method: str = 'realized_variance',
    ) -> np.ndarray:
        """
        Create volatility forecast target.

        CRITICAL: Target at time t is realized variance from t+1 to t+horizon.
        This ensures NO look-ahead bias.

        Args:
            returns: Array of returns
            method: Target type ('realized_variance', 'realized_vol', 'log_variance')

        Returns:
            Target array (with NaN at end where target is undefined)
        """
        returns = np.asarray(returns).flatten()
        n = len(returns)

        # Squared returns
        squared_returns = returns ** 2

        # Realized variance: sum of squared returns over horizon
        # Target at t is RV from t+1 to t+horizon
        target = np.full(n, np.nan)

        for t in range(n - self.horizon):
            target[t] = np.sum(squared_returns[t + 1:t + 1 + self.horizon])

        if method == 'realized_vol':
            target = np.sqrt(target)
        elif method == 'log_variance':
            target = np.log(target + 1e-10)

        return target

    def prepare(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> VolatilityDataset:
        """
        Prepare complete dataset for training.

        Args:
            features: Feature matrix [n_samples, n_features]
            returns: Returns array [n_samples]
            dates: Date array (optional)

        Returns:
            VolatilityDataset ready for training
        """
        features = np.asarray(features)
        returns = np.asarray(returns).flatten()
        n = len(returns)

        # Create target
        target = self.create_target(returns, method='realized_variance')

        # Create sequences
        X, y, ret_seq, var_seq, valid_idx = [], [], [], [], []

        for t in range(self.seq_length, n - self.horizon):
            if not np.isnan(target[t]):
                X.append(features[t - self.seq_length:t])
                y.append(target[t])
                ret_seq.append(returns[t - self.seq_length:t])
                var_seq.append(returns[t - self.seq_length:t] ** 2)
                valid_idx.append(t)

        X = np.array(X)  # [n_valid, seq_length, n_features]
        y = np.array(y)  # [n_valid]
        ret_seq = np.array(ret_seq)  # [n_valid, seq_length]
        var_seq = np.array(var_seq)  # [n_valid, seq_length]
        valid_idx = np.array(valid_idx)

        # Scale features (fit on training data only)
        n_valid = len(X)
        n_train = int(n_valid * self.train_ratio)
        n_val = int(n_valid * self.val_ratio)

        if self.scale_features:
            # Fit scaler on training data
            X_train_flat = X[:n_train].reshape(-1, X.shape[-1])
            self.feature_mean = X_train_flat.mean(axis=0)
            self.feature_std = X_train_flat.std(axis=0) + 1e-8

            # Transform all data
            X = (X - self.feature_mean) / self.feature_std

        if self.scale_targets:
            self.target_mean = y[:n_train].mean()
            self.target_std = y[:n_train].std() + 1e-8
            y = (y - self.target_mean) / self.target_std

        # Split data
        X_train = torch.FloatTensor(X[:n_train])
        y_train = torch.FloatTensor(y[:n_train]).unsqueeze(1)
        ret_train = torch.FloatTensor(ret_seq[:n_train])
        var_train = torch.FloatTensor(var_seq[:n_train])

        X_val = torch.FloatTensor(X[n_train:n_train + n_val])
        y_val = torch.FloatTensor(y[n_train:n_train + n_val]).unsqueeze(1)
        ret_val = torch.FloatTensor(ret_seq[n_train:n_train + n_val])
        var_val = torch.FloatTensor(var_seq[n_train:n_train + n_val])

        X_test = torch.FloatTensor(X[n_train + n_val:])
        y_test = torch.FloatTensor(y[n_train + n_val:]).unsqueeze(1)

        # Dates
        if dates is not None:
            dates_train = dates[valid_idx[:n_train]]
            dates_val = dates[valid_idx[n_train:n_train + n_val]]
        else:
            dates_train = None
            dates_val = None

        return VolatilityDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            returns_train=ret_train,
            returns_val=ret_val,
            variance_history_train=var_train,
            variance_history_val=var_val,
            dates_train=dates_train,
            dates_val=dates_val,
        )


class VolatilityTrainer:
    """
    Training pipeline for volatility forecasting models.

    Features:
        - Physics-informed loss support
        - Early stopping
        - Learning rate scheduling
        - Gradient clipping
        - Comprehensive logging
        - Checkpoint saving
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
        min_delta: float = 1e-6,
        grad_clip: float = 1.0,
        weight_decay: float = 1e-5,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Volatility forecasting model
            learning_rate: Initial learning rate
            batch_size: Batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            grad_clip: Gradient clipping value
            weight_decay: L2 regularization
            device: Training device
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.device = device or torch.device('cpu')
        self.checkpoint_dir = checkpoint_dir

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )

        # Training state
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0

        logger.info(f"VolatilityTrainer initialized: lr={learning_rate}, "
                   f"batch_size={batch_size}, device={self.device}")

    def _create_dataloader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        shuffle: bool = True,
        **metadata_tensors,
    ) -> DataLoader:
        """Create DataLoader with optional metadata."""
        tensors = [X, y]
        for v in metadata_tensors.values():
            if v is not None:
                tensors.append(v)

        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        train_loader: DataLoader,
        enable_physics: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            X_batch = batch[0].to(self.device)
            y_batch = batch[1].to(self.device)

            # Extract metadata if available
            metadata = {}
            if len(batch) > 2:
                metadata['returns'] = batch[2].to(self.device)
            if len(batch) > 3:
                metadata['variance_history'] = batch[3].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(X_batch)

            # Compute loss
            if hasattr(self.model, 'compute_loss') and enable_physics:
                loss, components = self.model.compute_loss(
                    predictions, y_batch, metadata, enable_physics=True
                )
                for k, v in components.items():
                    loss_components[k] += v
            else:
                loss = F.mse_loss(predictions, y_batch)
                loss_components['data_loss'] += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Average metrics
        return {
            'loss': total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    def _validate(
        self,
        val_loader: DataLoader,
        enable_physics: bool = True,
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        loss_components = defaultdict(float)
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                metadata = {}
                if len(batch) > 2:
                    metadata['returns'] = batch[2].to(self.device)
                if len(batch) > 3:
                    metadata['variance_history'] = batch[3].to(self.device)

                predictions = self.model(X_batch)

                if hasattr(self.model, 'compute_loss') and enable_physics:
                    loss, components = self.model.compute_loss(
                        predictions, y_batch, metadata, enable_physics=True
                    )
                    for k, v in components.items():
                        loss_components[k] += v
                else:
                    loss = F.mse_loss(predictions, y_batch)
                    loss_components['data_loss'] += loss.item()

                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                n_batches += 1

        preds = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()

        # Additional metrics
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))

        # QLIKE
        pred_safe = np.maximum(preds, 1e-10)
        target_safe = np.maximum(targets, 1e-10)
        qlike = np.mean(target_safe / pred_safe - np.log(target_safe / pred_safe) - 1)

        # R²
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'loss': total_loss / n_batches,
            'mse': mse,
            'mae': mae,
            'qlike': qlike,
            'r2': r2,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    def fit(
        self,
        dataset: VolatilityDataset,
        enable_physics: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            dataset: VolatilityDataset with training and validation data
            enable_physics: Whether to enable physics losses
            verbose: Whether to print progress

        Returns:
            Training history
        """
        # Create data loaders
        train_loader = self._create_dataloader(
            dataset.X_train, dataset.y_train,
            shuffle=True,
            returns=dataset.returns_train,
            variance_history=dataset.variance_history_train,
        )

        val_loader = self._create_dataloader(
            dataset.X_val, dataset.y_val,
            shuffle=False,
            returns=dataset.returns_val,
            variance_history=dataset.variance_history_val,
        )

        start_time = time.time()

        for epoch in range(self.max_epochs):
            # Training
            train_metrics = self._train_epoch(train_loader, enable_physics)

            # Validation
            val_metrics = self._validate(val_loader, enable_physics)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Record history
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)

            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0

                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1

            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"train_loss={train_metrics['loss']:.6f}, "
                    f"val_loss={val_metrics['loss']:.6f}, "
                    f"val_qlike={val_metrics['qlike']:.6f}, "
                    f"val_r2={val_metrics['r2']:.4f}"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s. "
                   f"Best val_loss: {self.best_val_loss:.6f}")

        # Log physics parameters if available
        if hasattr(self.model, 'log_physics_params'):
            self.model.log_physics_params()

        return dict(self.history)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': dict(self.history),
        }

        # Add physics parameters if available
        if hasattr(self.model, 'get_learned_physics_params'):
            checkpoint['physics_params'] = self.model.get_learned_physics_params()

        path = self.checkpoint_dir / 'vol_model_best.pt'
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Make predictions.

        Args:
            X: Input features [n_samples, seq_len, features]
            return_numpy: Whether to return numpy array

        Returns:
            Predictions
        """
        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model(X)

        if return_numpy:
            return predictions.cpu().numpy()
        return predictions

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Test features
            y: Test targets
            returns: Test returns (optional, for economic metrics)

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X, return_numpy=True).flatten()
        targets = y.numpy().flatten() if isinstance(y, torch.Tensor) else y.flatten()

        # Import metrics
        from ..evaluation.volatility_metrics import VolatilityMetrics, evaluate_volatility_forecast

        if returns is not None:
            if len(returns.shape) > 1 and returns.shape[1] > 1:
                # 2D return sequence from dataset, not suitable for evaluation metrics
                returns_np = None
            else:
                returns_np = returns.numpy()
        else:
            returns_np = None

        return evaluate_volatility_forecast(
            predicted_var=predictions,
            realized_var=targets,
            returns=returns_np,
            model_name=self.model.__class__.__name__,
        )


class WalkForwardVolatilityValidator:
    """
    Walk-forward validation for volatility forecasting models.

    Properly handles time-series cross-validation without look-ahead bias.
    """

    def __init__(
        self,
        train_window: int = 252 * 3,  # 3 years
        val_window: int = 252,         # 1 year
        step_size: int = 22,           # Monthly steps
        min_train_size: int = 252,     # Minimum 1 year
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_window: Training window size
            val_window: Validation window size
            step_size: Step size between folds
            min_train_size: Minimum training samples
        """
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.min_train_size = min_train_size

    def generate_splits(
        self,
        n_samples: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation indices for walk-forward validation.

        Args:
            n_samples: Total number of samples

        Yields:
            Tuple of (train_indices, val_indices)
        """
        start = self.min_train_size

        while start + self.val_window <= n_samples:
            train_start = max(0, start - self.train_window)
            train_end = start
            val_start = start
            val_end = min(start + self.val_window, n_samples)

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx

            start += self.step_size

    def evaluate(
        self,
        model_factory: Callable[[], nn.Module],
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None,
        trainer_kwargs: Optional[Dict] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run walk-forward evaluation.

        Args:
            model_factory: Function that creates a fresh model instance
            X: Features [n_samples, seq_len, features]
            y: Targets [n_samples]
            returns: Returns [n_samples] (optional)
            trainer_kwargs: Keyword arguments for trainer
            verbose: Whether to print progress

        Returns:
            DataFrame with results for each fold
        """
        trainer_kwargs = trainer_kwargs or {}
        results = []

        for fold, (train_idx, val_idx) in enumerate(self.generate_splits(len(X))):
            if verbose:
                logger.info(f"Fold {fold}: train=[{train_idx[0]}, {train_idx[-1]}], "
                           f"val=[{val_idx[0]}, {val_idx[-1]}]")

            # Create fresh model
            model = model_factory()

            # Create dataset for this fold
            X_train = torch.FloatTensor(X[train_idx])
            y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1)
            X_val = torch.FloatTensor(X[val_idx])
            y_val = torch.FloatTensor(y[val_idx]).unsqueeze(1)

            # Prepare returns if available
            ret_train = torch.FloatTensor(returns[train_idx]) if returns is not None else None
            ret_val = torch.FloatTensor(returns[val_idx]) if returns is not None else None

            dataset = VolatilityDataset(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                returns_train=ret_train,
                returns_val=ret_val,
            )

            # Train
            trainer = VolatilityTrainer(model, **trainer_kwargs)
            trainer.fit(dataset, verbose=False)

            # Evaluate
            predictions = trainer.predict(X_val)
            targets = y_val.numpy()

            from ..evaluation.volatility_metrics import VolatilityMetrics

            fold_results = {
                'fold': fold,
                'train_start': int(train_idx[0]),
                'train_end': int(train_idx[-1]),
                'val_start': int(val_idx[0]),
                'val_end': int(val_idx[-1]),
                'mse': VolatilityMetrics.mse(predictions, targets),
                'mae': VolatilityMetrics.mae(predictions, targets),
                'qlike': VolatilityMetrics.qlike(predictions, targets),
                'r2': VolatilityMetrics.mincer_zarnowitz_r2(predictions, targets),
            }

            results.append(fold_results)

        return pd.DataFrame(results)
