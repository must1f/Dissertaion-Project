"""
Batch Trainer - Trains multiple models with real-time progress updates

Provides:
- Sequential and parallel model training
- Real-time progress callbacks
- Training history persistence
- Integration with the web dashboard
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import threading
from queue import Queue
from datetime import datetime
import numpy as np

from ..utils.config import get_config, get_research_config
from ..utils.logger import get_logger
from ..utils.reproducibility import set_seed
from ..data.fetcher import DataFetcher
from ..data.preprocessor import DataPreprocessor
from ..data.dataset import FinancialDataset, create_dataloaders
from ..models.baseline import LSTMModel, GRUModel, BiLSTMModel
from ..models.transformer import TransformerModel
from ..models.pinn import PINNModel
from ..evaluation.metrics import calculate_metrics


logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a single model training run"""
    model_key: str
    model_type: str  # 'lstm', 'gru', 'bilstm', 'transformer', 'pinn', etc.
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    gradient_clip_norm: float = 1.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 15  # 0 to disable
    physics_constraints: Optional[Dict[str, float]] = None


@dataclass
class EpochProgress:
    """Progress data for a single epoch"""
    epoch: int
    train_loss: float
    val_loss: float
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None
    learning_rate: float = 0.0
    elapsed_time: float = 0.0


@dataclass
class ModelTrainingState:
    """Complete training state for a model"""
    model_key: str
    config: TrainingConfig
    status: str = "pending"  # pending, training, completed, failed
    current_epoch: int = 0
    total_epochs: int = 100
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    data_losses: List[float] = field(default_factory=list)
    physics_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None


ProgressCallback = Callable[[str, EpochProgress], None]


class BatchTrainer:
    """
    Trains multiple models sequentially with real-time progress updates.

    Usage:
        trainer = BatchTrainer(project_root)
        trainer.prepare_data()

        configs = [
            TrainingConfig(model_key='lstm', model_type='lstm', epochs=50),
            TrainingConfig(model_key='pinn_gbm', model_type='pinn', epochs=50,
                          physics_constraints={'lambda_gbm': 0.1}),
        ]

        def on_progress(model_key: str, progress: EpochProgress):
            print(f"{model_key}: Epoch {progress.epoch}, Loss: {progress.train_loss:.4f}")

        results = trainer.train_all(configs, progress_callback=on_progress)
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = get_config()
        self.device = self._get_device()

        # Data loaders (shared across models)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.input_dim: int = 0

        # Training state
        self.training_states: Dict[str, ModelTrainingState] = {}
        self.is_training: bool = False
        self.should_stop: bool = False

    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def prepare_data(self, tickers: Optional[List[str]] = None, force_refresh: bool = False) -> bool:
        """
        Prepare data for training all models.

        Args:
            tickers: List of stock tickers (uses config default if None)
            force_refresh: Force data re-fetch

        Returns:
            True if data prepared successfully
        """
        logger.info("Preparing training data...")

        try:
            # Use config tickers if not specified
            tickers = tickers or self.config.data.tickers[:10]

            # Fetch data
            fetcher = DataFetcher(self.config)
            df = fetcher.fetch_and_store(
                tickers=tickers,
                start_date=self.config.data.start_date,
                end_date=self.config.data.end_date,
                force_refresh=force_refresh
            )

            if df.empty:
                logger.error("No data fetched!")
                return False

            # Preprocess
            preprocessor = DataPreprocessor(self.config)
            df_processed = preprocessor.process_and_store(df)

            # Feature columns
            feature_cols = [
                'close', 'volume',
                'log_return', 'simple_return',
                'rolling_volatility_5', 'rolling_volatility_20',
                'momentum_5', 'momentum_20',
                'rsi_14', 'macd', 'macd_signal',
                'bollinger_upper', 'bollinger_lower', 'atr_14'
            ]
            feature_cols = [col for col in feature_cols if col in df_processed.columns]
            self.input_dim = len(feature_cols)

            # Temporal split
            train_df, val_df, test_df = preprocessor.split_temporal(df_processed)

            # Normalize
            train_df_norm, scalers = preprocessor.normalize_features(
                train_df, feature_cols, method='standard'
            )

            # Apply normalization to val/test
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
            X_train, y_train, tickers_train = preprocessor.create_sequences(
                train_df_norm, feature_cols, target_col='close',
                sequence_length=self.config.data.sequence_length,
                forecast_horizon=self.config.data.forecast_horizon
            )

            X_val, y_val, tickers_val = preprocessor.create_sequences(
                val_df, feature_cols, target_col='close',
                sequence_length=self.config.data.sequence_length,
                forecast_horizon=self.config.data.forecast_horizon
            )

            X_test, y_test, tickers_test = preprocessor.create_sequences(
                test_df, feature_cols, target_col='close',
                sequence_length=self.config.data.sequence_length,
                forecast_horizon=self.config.data.forecast_horizon
            )

            logger.info(f"Train sequences: {X_train.shape}")
            logger.info(f"Val sequences: {X_val.shape}")
            logger.info(f"Test sequences: {X_test.shape}")

            # Create datasets
            train_dataset = FinancialDataset(X_train, y_train, tickers_train)
            val_dataset = FinancialDataset(X_val, y_val, tickers_val)
            test_dataset = FinancialDataset(X_test, y_test, tickers_test)

            # Create dataloaders with default batch size (will be recreated per model)
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                train_dataset, val_dataset, test_dataset,
                batch_size=self.config.training.batch_size
            )

            # Store datasets for recreating loaders with different batch sizes
            self._train_dataset = train_dataset
            self._val_dataset = val_dataset
            self._test_dataset = test_dataset

            logger.info("Data preparation complete!")
            return True

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return False

    def _create_dataloaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders with specified batch size"""
        return create_dataloaders(
            self._train_dataset, self._val_dataset, self._test_dataset,
            batch_size=batch_size
        )

    def _create_model(self, config: TrainingConfig) -> nn.Module:
        """Create model based on configuration"""
        model_type = config.model_type.lower()
        physics = config.physics_constraints or {}

        if model_type == 'lstm':
            return LSTMModel(
                input_dim=self.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_dim=1,
                dropout=config.dropout,
                bidirectional=False
            )

        elif model_type == 'gru':
            return GRUModel(
                input_dim=self.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_dim=1,
                dropout=config.dropout,
                bidirectional=False
            )

        elif model_type == 'bilstm':
            return BiLSTMModel(
                input_dim=self.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_dim=1,
                dropout=config.dropout
            )

        elif model_type == 'transformer':
            return TransformerModel(
                input_dim=self.input_dim,
                d_model=config.hidden_dim,
                nhead=8,
                num_encoder_layers=config.num_layers,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                output_dim=1
            )

        elif model_type == 'pinn' or model_type.startswith('pinn_'):
            return PINNModel(
                input_dim=self.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_dim=1,
                dropout=config.dropout,
                base_model='lstm',
                lambda_gbm=physics.get('lambda_gbm', 0.0),
                lambda_bs=physics.get('lambda_bs', 0.0),
                lambda_ou=physics.get('lambda_ou', 0.0),
                lambda_langevin=physics.get('lambda_langevin', 0.0)
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_all(
        self,
        configs: List[TrainingConfig],
        progress_callback: Optional[ProgressCallback] = None,
        save_checkpoints: bool = True
    ) -> Dict[str, ModelTrainingState]:
        """
        Train all models sequentially.

        Args:
            configs: List of training configurations
            progress_callback: Called after each epoch with progress data
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Dictionary of model key -> training state
        """
        if self.train_loader is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        self.is_training = True
        self.should_stop = False
        self.training_states = {}

        # Initialize states
        for config in configs:
            self.training_states[config.model_key] = ModelTrainingState(
                model_key=config.model_key,
                config=config,
                total_epochs=config.epochs
            )

        # Train each model
        for config in configs:
            if self.should_stop:
                logger.info("Training stopped by user")
                break

            state = self.training_states[config.model_key]

            try:
                self._train_model(config, state, progress_callback, save_checkpoints)
            except Exception as e:
                logger.error(f"Training failed for {config.model_key}: {e}")
                state.status = 'failed'
                state.error_message = str(e)

        self.is_training = False
        return self.training_states

    def _train_model(
        self,
        config: TrainingConfig,
        state: ModelTrainingState,
        progress_callback: Optional[ProgressCallback],
        save_checkpoints: bool
    ):
        """Train a single model"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {config.model_key}")
        logger.info(f"{'='*60}")

        state.status = 'training'
        state.start_time = time.time()

        # Create dataloaders with model-specific batch size
        train_loader, val_loader, test_loader = self._create_dataloaders(config.batch_size)

        # Create model
        model = self._create_model(config)
        model = model.to(self.device)

        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )

        # Loss function
        criterion = nn.MSELoss()

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Check if PINN
        is_pinn = hasattr(model, 'compute_loss')

        # Training loop
        for epoch in range(1, config.epochs + 1):
            if self.should_stop:
                break

            epoch_start = time.time()

            # Training
            model.train()
            train_loss = 0.0
            train_data_loss = 0.0
            train_physics_loss = 0.0
            n_batches = 0

            for sequences, targets, metadata in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                if is_pinn:
                    predictions = model(sequences)

                    # Move metadata to device
                    for key in ['prices', 'returns', 'volatilities']:
                        if key in metadata:
                            metadata[key] = metadata[key].to(self.device)
                    metadata['inputs'] = sequences

                    loss, loss_dict = model.compute_loss(predictions, targets, metadata)
                    train_data_loss += loss_dict.get('data_loss', 0.0)
                    train_physics_loss += loss_dict.get('physics_loss', 0.0)
                else:
                    # Standard forward pass
                    model_name = model.__class__.__name__.lower()
                    if 'lstm' in model_name or 'gru' in model_name:
                        predictions, _ = model(sequences)
                    else:
                        predictions = model(sequences)
                    loss = criterion(predictions, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            train_data_loss /= n_batches if is_pinn else 1
            train_physics_loss /= n_batches if is_pinn else 1

            # Validation
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for sequences, targets, metadata in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    if is_pinn:
                        predictions = model(sequences)
                        for key in ['prices', 'returns', 'volatilities']:
                            if key in metadata:
                                metadata[key] = metadata[key].to(self.device)
                        metadata['inputs'] = sequences
                        loss, _ = model.compute_loss(predictions, targets, metadata)
                    else:
                        model_name = model.__class__.__name__.lower()
                        if 'lstm' in model_name or 'gru' in model_name:
                            predictions, _ = model(sequences)
                        else:
                            predictions = model(sequences)
                        loss = criterion(predictions, targets)

                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Update state
            state.current_epoch = epoch
            state.train_losses.append(train_loss)
            state.val_losses.append(val_loss)
            state.learning_rates.append(current_lr)

            if is_pinn:
                state.data_losses.append(train_data_loss)
                state.physics_losses.append(train_physics_loss)

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state.best_val_loss = val_loss
                state.best_epoch = epoch
                patience_counter = 0

                if save_checkpoints:
                    self._save_checkpoint(model, optimizer, scheduler, state, config)
            else:
                patience_counter += 1

            # Progress callback
            elapsed = time.time() - epoch_start

            if progress_callback:
                progress = EpochProgress(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    data_loss=train_data_loss if is_pinn else None,
                    physics_loss=train_physics_loss if is_pinn else None,
                    learning_rate=current_lr,
                    elapsed_time=elapsed
                )
                progress_callback(config.model_key, progress)

            # Logging
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{config.epochs} - "
                    f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}"
                )

            # Early stopping
            if config.early_stopping_patience > 0 and patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        state.status = 'completed'
        state.end_time = time.time()

        # Save final results
        self._save_results(state, config)

        logger.info(f"Training completed for {config.model_key}")
        logger.info(f"Best val loss: {state.best_val_loss:.4f} at epoch {state.best_epoch}")

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        state: ModelTrainingState,
        config: TrainingConfig
    ):
        """Save model checkpoint"""
        checkpoint_dir = self.project_root / 'models'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': state.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': state.best_val_loss,
            'config': {
                'model_key': config.model_key,
                'epochs': config.epochs,
                'learning_rate': config.learning_rate,
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'dropout': config.dropout,
                'physics_constraints': config.physics_constraints
            },
            'history': {
                'train_loss': state.train_losses,
                'val_loss': state.val_losses,
                'train_data_loss': state.data_losses,
                'train_physics_loss': state.physics_losses,
                'learning_rates': state.learning_rates,
                'epochs': list(range(1, len(state.train_losses) + 1))
            }
        }

        checkpoint_path = checkpoint_dir / f'{config.model_key}_best.pt'
        torch.save(checkpoint, checkpoint_path)

        # Also save history as JSON
        history_path = checkpoint_dir / f'{config.model_key}_history.json'
        with open(history_path, 'w') as f:
            json.dump(checkpoint['history'], f, indent=2)

    def _save_results(self, state: ModelTrainingState, config: TrainingConfig):
        """Save training results"""
        results_dir = self.project_root / 'results'
        results_dir.mkdir(exist_ok=True)

        results = {
            'model': config.model_key,
            'model_type': config.model_type,
            'training_date': datetime.now().isoformat(),
            'config': {
                'epochs': config.epochs,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'dropout': config.dropout,
                'physics_constraints': config.physics_constraints
            },
            'results': {
                'total_epochs_trained': state.current_epoch,
                'best_epoch': state.best_epoch,
                'best_val_loss': float(state.best_val_loss),
                'final_train_loss': float(state.train_losses[-1]) if state.train_losses else None,
                'final_val_loss': float(state.val_losses[-1]) if state.val_losses else None,
                'training_time_seconds': state.end_time - state.start_time if state.end_time and state.start_time else None
            },
            'history': {
                'train_loss': [float(x) for x in state.train_losses],
                'val_loss': [float(x) for x in state.val_losses],
                'train_data_loss': [float(x) for x in state.data_losses] if state.data_losses else [],
                'train_physics_loss': [float(x) for x in state.physics_losses] if state.physics_losses else [],
                'learning_rates': [float(x) for x in state.learning_rates],
                'epochs': list(range(1, len(state.train_losses) + 1))
            }
        }

        results_path = results_dir / f'{config.model_key}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    def stop_training(self):
        """Stop the current training session"""
        self.should_stop = True

    def get_training_state(self, model_key: str) -> Optional[ModelTrainingState]:
        """Get the current training state for a model"""
        return self.training_states.get(model_key)

    def get_all_states(self) -> Dict[str, ModelTrainingState]:
        """Get all training states"""
        return self.training_states


def get_default_training_configs() -> List[TrainingConfig]:
    """Get default training configurations for all models"""

    configs = [
        # Baseline models
        TrainingConfig(model_key='lstm', model_type='lstm'),
        TrainingConfig(model_key='gru', model_type='gru'),
        TrainingConfig(model_key='bilstm', model_type='bilstm'),
        TrainingConfig(model_key='transformer', model_type='transformer'),

        # PINN variants
        TrainingConfig(
            model_key='pinn_baseline', model_type='pinn',
            physics_constraints={'lambda_gbm': 0.0, 'lambda_bs': 0.0, 'lambda_ou': 0.0}
        ),
        TrainingConfig(
            model_key='pinn_gbm', model_type='pinn',
            physics_constraints={'lambda_gbm': 0.1, 'lambda_bs': 0.0, 'lambda_ou': 0.0}
        ),
        TrainingConfig(
            model_key='pinn_ou', model_type='pinn',
            physics_constraints={'lambda_gbm': 0.0, 'lambda_bs': 0.0, 'lambda_ou': 0.1}
        ),
        TrainingConfig(
            model_key='pinn_black_scholes', model_type='pinn',
            physics_constraints={'lambda_gbm': 0.0, 'lambda_bs': 0.1, 'lambda_ou': 0.0}
        ),
        TrainingConfig(
            model_key='pinn_gbm_ou', model_type='pinn',
            physics_constraints={'lambda_gbm': 0.05, 'lambda_bs': 0.0, 'lambda_ou': 0.05}
        ),
        TrainingConfig(
            model_key='pinn_global', model_type='pinn',
            physics_constraints={'lambda_gbm': 0.05, 'lambda_bs': 0.03, 'lambda_ou': 0.05, 'lambda_langevin': 0.02}
        ),
    ]

    return configs
