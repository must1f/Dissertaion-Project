"""
Data Refresh and Model Retraining Service for Streamlit

Handles fetching fresh data from yfinance and retraining all models
on the same dataset with consistent train/val/test splits.
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config, get_research_config
from src.utils.logger import get_logger
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import FinancialDataset, create_dataloaders
from src.models.pinn import PINNModel
from src.training.trainer import Trainer
from src.evaluation.unified_evaluator import UnifiedModelEvaluator
from src.utils.reproducibility import set_seed, get_device

logger = get_logger(__name__)

# Research mode default - ensures fair model comparison
RESEARCH_MODE_DEFAULT = True

# PINN configurations for retraining
PINN_CONFIGURATIONS = {
    'baseline': {
        'name': 'Baseline (Data-only)',
        'lambda_gbm': 0.0,
        'lambda_bs': 0.0,
        'lambda_ou': 0.0,
        'lambda_langevin': 0.0,
        'enable_physics': False,
    },
    'gbm': {
        'name': 'Pure GBM (Trend)',
        'lambda_gbm': 0.1,
        'lambda_bs': 0.0,
        'lambda_ou': 0.0,
        'lambda_langevin': 0.0,
        'enable_physics': True,
    },
    'ou': {
        'name': 'Pure OU (Mean-Reversion)',
        'lambda_gbm': 0.0,
        'lambda_bs': 0.0,
        'lambda_ou': 0.1,
        'lambda_langevin': 0.0,
        'enable_physics': True,
    },
    'black_scholes': {
        'name': 'Pure Black-Scholes',
        'lambda_gbm': 0.0,
        'lambda_bs': 0.1,
        'lambda_ou': 0.0,
        'lambda_langevin': 0.0,
        'enable_physics': True,
    },
    'gbm_ou': {
        'name': 'GBM+OU Hybrid',
        'lambda_gbm': 0.05,
        'lambda_bs': 0.0,
        'lambda_ou': 0.05,
        'lambda_langevin': 0.0,
        'enable_physics': True,
    },
    'global': {
        'name': 'Global Constraint',
        'lambda_gbm': 0.05,
        'lambda_bs': 0.03,
        'lambda_ou': 0.05,
        'lambda_langevin': 0.02,
        'enable_physics': True,
    }
}


class DataRefreshService:
    """
    Service for refreshing data and retraining models.

    Ensures all models are trained on the same dataset with consistent splits.
    """

    def __init__(self):
        self.config = get_config()
        self.fetcher = DataFetcher(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.results_dir = self.config.project_root / 'results'
        self.models_dir = self.config.project_root / 'models'

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Feature columns used for training
        self.feature_cols = [
            'close', 'volume',
            'log_return', 'simple_return',
            'rolling_volatility_5', 'rolling_volatility_20',
            'momentum_5', 'momentum_20',
            'rsi_14', 'macd', 'macd_signal',
            'bollinger_upper', 'bollinger_lower', 'atr_14'
        ]

    def fetch_fresh_data(
        self,
        tickers: Optional[List[str]] = None,
        end_date: Optional[str] = None,
        progress_callback=None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch fresh data from yfinance.

        Args:
            tickers: List of ticker symbols (None = use config)
            end_date: End date for data (None = today)
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (DataFrame with raw data, metadata dict)
        """
        tickers = tickers or self.config.data.tickers
        start_date = self.config.data.start_date
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        if progress_callback:
            progress_callback(0.1, f"Fetching data for {len(tickers)} tickers...")

        logger.info(f"Fetching fresh data: {start_date} to {end_date}")

        # Force refresh to get latest data
        df = self.fetcher.fetch_and_store(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            force_refresh=True
        )

        if df.empty:
            raise ValueError("Failed to fetch data from yfinance")

        metadata = {
            'fetch_timestamp': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'n_tickers': len(df['ticker'].unique()),
            'n_records': len(df),
            'tickers': list(df['ticker'].unique())
        }

        if progress_callback:
            progress_callback(0.3, f"Fetched {len(df)} records for {metadata['n_tickers']} tickers")

        return df, metadata

    def preprocess_data(
        self,
        df: pd.DataFrame,
        progress_callback=None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess raw data with feature engineering.

        Args:
            df: Raw DataFrame
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (processed DataFrame, metadata dict)
        """
        if progress_callback:
            progress_callback(0.35, "Calculating returns and volatility...")

        # Process data
        df_processed = self.preprocessor.process_and_store(df)

        if progress_callback:
            progress_callback(0.45, "Feature engineering complete")

        # Filter available features
        available_features = [col for col in self.feature_cols if col in df_processed.columns]

        metadata = {
            'n_features': len(available_features),
            'features': available_features,
            'n_samples': len(df_processed)
        }

        return df_processed, metadata

    def create_data_splits(
        self,
        df_processed: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        progress_callback=None
    ) -> Dict:
        """
        Create train/val/test splits and dataloaders.

        All models will use these exact same splits for fair comparison.

        Args:
            df_processed: Processed DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with splits information and dataloaders
        """
        if progress_callback:
            progress_callback(0.5, "Creating temporal splits...")

        # Get available features
        feature_cols = [col for col in self.feature_cols if col in df_processed.columns]

        # Temporal split
        train_df, val_df, test_df = self.preprocessor.split_temporal(
            df_processed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        if progress_callback:
            progress_callback(0.55, "Normalizing features...")

        # Normalize features (fit on train only)
        train_df_norm, scalers = self.preprocessor.normalize_features(
            train_df, feature_cols, method='standard'
        )

        # Apply same normalization to val and test
        val_df = val_df.copy()
        test_df = test_df.copy()

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

        if progress_callback:
            progress_callback(0.6, "Creating sequences...")

        # Create sequences
        sequence_length = self.config.data.sequence_length
        forecast_horizon = self.config.data.forecast_horizon

        X_train, y_train, tickers_train = self.preprocessor.create_sequences(
            train_df_norm, feature_cols, target_col='close',
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )

        X_val, y_val, tickers_val = self.preprocessor.create_sequences(
            val_df, feature_cols, target_col='close',
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )

        X_test, y_test, tickers_test = self.preprocessor.create_sequences(
            test_df, feature_cols, target_col='close',
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )

        if progress_callback:
            progress_callback(0.65, "Creating dataloaders...")

        # Create datasets
        train_dataset = FinancialDataset(X_train, y_train, tickers_train)
        val_dataset = FinancialDataset(X_val, y_val, tickers_val)
        test_dataset = FinancialDataset(X_test, y_test, tickers_test)

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.config.training.batch_size
        )

        # Store test targets for metrics calculation
        split_info = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'input_dim': len(feature_cols),
            'feature_cols': feature_cols,
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'X_test': X_test,
            'y_test': y_test,
            'scalers': scalers,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }

        return split_info

    def train_model(
        self,
        variant_key: str,
        split_info: Dict,
        epochs: int = 50,
        device: str = 'auto',
        progress_callback=None,
        research_mode: bool = True
    ) -> Dict:
        """
        Train a single PINN variant.

        Args:
            variant_key: Key from PINN_CONFIGURATIONS
            split_info: Dictionary with data loaders and metadata
            epochs: Number of training epochs (ignored in research mode - uses locked value)
            device: Device to train on ('auto', 'cuda', 'mps', 'cpu')
            progress_callback: Optional callback for progress updates
            research_mode: If True, use locked research parameters for fair comparison

        Returns:
            Dictionary with training results
        """
        if variant_key not in PINN_CONFIGURATIONS:
            raise ValueError(f"Unknown variant: {variant_key}")

        variant_config = PINN_CONFIGURATIONS[variant_key]
        research_config = get_research_config() if research_mode else None

        if progress_callback:
            progress_callback(0.0, f"Initializing {variant_config['name']}...")

        # Get device
        if device == 'auto':
            device = get_device(prefer_cuda=(self.config.training.device == 'cuda'))
        else:
            device = torch.device(device)

        # Get model parameters from research config or standard config
        if research_mode and research_config:
            hidden_dim = research_config.hidden_dim
            num_layers = research_config.num_layers
            dropout = research_config.dropout
            training_epochs = research_config.epochs
            logger.info(f"Research mode: Using locked parameters (epochs={training_epochs}, hidden_dim={hidden_dim})")
        else:
            hidden_dim = self.config.model.hidden_dim
            num_layers = self.config.model.num_layers
            dropout = self.config.model.dropout
            training_epochs = epochs

        # Create model with consistent architecture
        model = PINNModel(
            input_dim=split_info['input_dim'],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
            base_model='lstm',
            lambda_gbm=variant_config['lambda_gbm'],
            lambda_bs=variant_config['lambda_bs'],
            lambda_ou=variant_config['lambda_ou'],
            lambda_langevin=variant_config['lambda_langevin']
        )

        # Create trainer with research mode
        trainer = Trainer(
            model=model,
            train_loader=split_info['train_loader'],
            val_loader=split_info['val_loader'],
            test_loader=split_info['test_loader'],
            config=self.config,
            device=device,
            research_mode=research_mode
        )

        if progress_callback:
            mode_str = "Research" if research_mode else "Standard"
            progress_callback(0.1, f"Training {variant_config['name']} ({mode_str} mode, {training_epochs} epochs)...")

        # Train - epochs parameter ignored in research mode (uses locked value)
        history = trainer.train(
            epochs=training_epochs,
            enable_physics=variant_config['enable_physics'],
            save_best=True,
            model_name=f"pinn_{variant_key}"
        )

        if progress_callback:
            progress_callback(0.8, f"Evaluating {variant_config['name']}...")

        # Evaluate
        test_metrics = trainer.evaluate(enable_physics=variant_config['enable_physics'])

        # Get predictions
        predictions, targets = trainer.get_predictions()

        # Save predictions
        predictions_path = self.results_dir / f'pinn_{variant_key}_predictions.npz'
        np.savez(
            predictions_path,
            predictions=predictions,
            targets=targets
        )

        if progress_callback:
            progress_callback(0.9, f"Computing financial metrics...")

        # Compute comprehensive metrics
        evaluator = UnifiedModelEvaluator(
            transaction_cost=0.003,
            risk_free_rate=0.02,
            periods_per_year=252
        )

        full_metrics = evaluator.evaluate_model(
            predictions=predictions,
            targets=targets,
            model_name=variant_config['name'],
            compute_rolling=True
        )

        # Save results with research mode info
        results = {
            'variant_key': variant_key,
            'variant_name': variant_config['name'],
            'configuration': variant_config,
            'test_metrics': test_metrics,
            'ml_metrics': full_metrics.get('ml_metrics', {}),
            'financial_metrics': full_metrics.get('financial_metrics', {}),
            'rolling_metrics': full_metrics.get('rolling_metrics', {}),
            'history': history,
            'training_timestamp': datetime.now().isoformat(),
            'epochs_trained': history.get('total_epochs_trained', training_epochs),
            'research_mode': research_mode,
            'research_config': research_config.dict() if research_config else None,
            'training_params': {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'epochs': training_epochs,
                'early_stopping_disabled': research_mode,
            }
        }

        # Save JSON results
        results_path = self.results_dir / f'pinn_{variant_key}_results.json'

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        with open(results_path, 'w') as f:
            json.dump(convert_types(results), f, indent=2)

        if progress_callback:
            progress_callback(1.0, f"Completed {variant_config['name']}")

        return results

    def retrain_all_models(
        self,
        split_info: Dict,
        variants: Optional[List[str]] = None,
        epochs: int = 50,
        device: str = 'auto',
        progress_callback=None,
        research_mode: bool = True
    ) -> Dict[str, Dict]:
        """
        Retrain all selected PINN variants on the same data.

        All models are trained with identical parameters for fair comparison.

        Args:
            split_info: Dictionary with data loaders and metadata
            variants: List of variant keys to train (None = all)
            epochs: Number of training epochs (ignored in research mode)
            device: Device to train on
            progress_callback: Optional callback for overall progress
            research_mode: If True, use locked research parameters for fair comparison

        Returns:
            Dictionary mapping variant keys to results
        """
        variants = variants or list(PINN_CONFIGURATIONS.keys())
        all_results = {}

        research_config = get_research_config() if research_mode else None

        if research_mode:
            logger.info("=" * 60)
            logger.info("RESEARCH MODE ENABLED - Fair Model Comparison")
            logger.info("=" * 60)
            logger.info(f"Locked epochs: {research_config.epochs}")
            logger.info(f"Locked batch size: {research_config.batch_size}")
            logger.info(f"Locked learning rate: {research_config.learning_rate}")
            logger.info(f"Early stopping: DISABLED")
            logger.info("=" * 60)

        for i, variant_key in enumerate(variants):
            base_progress = i / len(variants)

            if progress_callback:
                mode_str = "Research" if research_mode else "Standard"
                progress_callback(
                    base_progress,
                    f"[{mode_str}] Training model {i+1}/{len(variants)}: {PINN_CONFIGURATIONS[variant_key]['name']}"
                )

            try:
                # Create variant-specific progress callback
                def variant_progress(p, msg):
                    if progress_callback:
                        overall = base_progress + (p / len(variants))
                        progress_callback(overall, msg)

                results = self.train_model(
                    variant_key=variant_key,
                    split_info=split_info,
                    epochs=epochs,
                    device=device,
                    progress_callback=variant_progress,
                    research_mode=research_mode
                )

                all_results[variant_key] = results

            except Exception as e:
                logger.error(f"Failed to train {variant_key}: {e}")
                all_results[variant_key] = {'error': str(e)}

        # Save summary with research mode info
        summary_path = self.results_dir / 'retraining_summary.json'
        summary = {
            'timestamp': datetime.now().isoformat(),
            'variants_trained': list(all_results.keys()),
            'research_mode': research_mode,
            'training_params': {
                'epochs': research_config.epochs if research_mode else epochs,
                'batch_size': research_config.batch_size if research_mode else self.config.training.batch_size,
                'learning_rate': research_config.learning_rate if research_mode else self.config.training.learning_rate,
                'hidden_dim': research_config.hidden_dim if research_mode else self.config.model.hidden_dim,
                'num_layers': research_config.num_layers if research_mode else self.config.model.num_layers,
                'dropout': research_config.dropout if research_mode else self.config.model.dropout,
                'early_stopping_disabled': research_mode,
            },
            'split_info': {
                'train_samples': split_info['train_samples'],
                'val_samples': split_info['val_samples'],
                'test_samples': split_info['test_samples'],
                'train_ratio': split_info['train_ratio'],
                'val_ratio': split_info['val_ratio'],
                'test_ratio': split_info['test_ratio']
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return all_results

    def clear_caches(self):
        """Clear Streamlit caches to force reload of fresh data."""
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            logger.info("Streamlit caches cleared")
        except Exception as e:
            logger.warning(f"Failed to clear caches: {e}")


def render_data_refresh_panel():
    """
    Render the data refresh and retraining panel in Streamlit.
    """
    st.subheader("Data Refresh & Model Retraining")

    st.info("""
    Fetch fresh data from yfinance and retrain models on the updated dataset.
    All models will be trained on the **same exact data splits** for fair comparison.
    """)

    service = DataRefreshService()
    config = get_config()

    # Data refresh section
    with st.expander("Data Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Ticker selection
            n_tickers = st.selectbox(
                "Number of Tickers",
                options=[5, 10, 20, 50],
                index=1,
                help="Number of tickers to use for training"
            )

            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Fetch data up to this date"
            )

        with col2:
            # Split ratios
            train_ratio = st.slider(
                "Train Ratio",
                min_value=0.5,
                max_value=0.8,
                value=0.7,
                step=0.05
            )

            val_ratio = st.slider(
                "Validation Ratio",
                min_value=0.1,
                max_value=0.2,
                value=0.15,
                step=0.05
            )

            test_ratio = 1.0 - train_ratio - val_ratio
            st.metric("Test Ratio", f"{test_ratio:.2%}")

    # Model selection
    with st.expander("Model Selection", expanded=True):
        st.markdown("Select which PINN variants to retrain:")

        col1, col2 = st.columns(2)

        selected_variants = []
        for i, (key, cfg) in enumerate(PINN_CONFIGURATIONS.items()):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.checkbox(cfg['name'], value=True, key=f"variant_{key}"):
                    selected_variants.append(key)

    # Training settings
    with st.expander("Training Settings", expanded=True):
        # Research mode toggle
        research_config = get_research_config()

        research_mode = st.checkbox(
            "Research Mode (Fair Comparison)",
            value=True,
            help="Enable research mode to lock all training parameters for fair model comparison"
        )

        if research_mode:
            st.success("""
            **Research Mode Enabled** - All parameters locked for fair comparison:
            - All models train for exactly the same number of epochs
            - Early stopping is DISABLED
            - Same learning rate, batch size, and architecture for all models
            """)

            # Display locked parameters (read-only)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Epochs (Locked)", research_config.epochs)
                st.metric("Batch Size (Locked)", research_config.batch_size)

            with col2:
                st.metric("Learning Rate (Locked)", f"{research_config.learning_rate:.4f}")
                st.metric("Hidden Dim (Locked)", research_config.hidden_dim)

            with col3:
                st.metric("Num Layers (Locked)", research_config.num_layers)
                st.metric("Dropout (Locked)", f"{research_config.dropout:.2f}")

            # Device selection (still configurable)
            device_options = ['auto', 'cpu']
            if torch.cuda.is_available():
                device_options.insert(1, 'cuda')
            if torch.backends.mps.is_available():
                device_options.insert(1, 'mps')

            device = st.selectbox(
                "Device",
                options=device_options,
                help="Device for training (configurable even in research mode)"
            )

            # Epochs will be ignored in research mode
            epochs = research_config.epochs

        else:
            st.warning("""
            **Standard Mode** - Parameters are configurable but models may not be directly comparable
            due to different training conditions (early stopping may trigger at different epochs).
            """)

            col1, col2 = st.columns(2)

            with col1:
                epochs = st.number_input(
                    "Epochs",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Number of training epochs per model"
                )

            with col2:
                device_options = ['auto', 'cpu']
                if torch.cuda.is_available():
                    device_options.insert(1, 'cuda')
                if torch.backends.mps.is_available():
                    device_options.insert(1, 'mps')

                device = st.selectbox(
                    "Device",
                    options=device_options,
                    help="Device for training"
                )

    # Action buttons
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        fetch_only = st.button(
            "Fetch Data Only",
            help="Only fetch fresh data without retraining",
            use_container_width=True
        )

    with col2:
        full_retrain = st.button(
            "Fetch & Retrain All",
            type="primary",
            help="Fetch fresh data and retrain all selected models",
            use_container_width=True
        )

    with col3:
        clear_cache = st.button(
            "Clear Caches",
            help="Clear Streamlit caches to reload fresh results",
            use_container_width=True
        )

    # Handle actions
    if clear_cache:
        service.clear_caches()
        st.success("Caches cleared!")
        st.rerun()

    if fetch_only:
        tickers = config.data.tickers[:n_tickers]

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(p, msg):
            progress_bar.progress(p)
            status_text.text(msg)

        try:
            with st.spinner("Fetching data..."):
                df, metadata = service.fetch_fresh_data(
                    tickers=tickers,
                    end_date=str(end_date),
                    progress_callback=update_progress
                )

            st.success(f"Fetched {metadata['n_records']} records for {metadata['n_tickers']} tickers")

            # Show sample data
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df.head(100))

            # Clear caches
            service.clear_caches()

        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            logger.exception(e)

    if full_retrain:
        if not selected_variants:
            st.warning("Please select at least one model variant to train")
            return

        tickers = config.data.tickers[:n_tickers]

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(p, msg):
            progress_bar.progress(min(p, 1.0))
            status_text.text(msg)

        try:
            # Step 1: Fetch data
            update_progress(0.05, "Fetching fresh data from yfinance...")
            df, fetch_metadata = service.fetch_fresh_data(
                tickers=tickers,
                end_date=str(end_date),
                progress_callback=lambda p, m: update_progress(0.05 + p * 0.15, m)
            )

            # Step 2: Preprocess
            update_progress(0.2, "Preprocessing data...")
            df_processed, preprocess_metadata = service.preprocess_data(
                df,
                progress_callback=lambda p, m: update_progress(0.2 + p * 0.1, m)
            )

            # Step 3: Create splits
            update_progress(0.3, "Creating data splits...")
            split_info = service.create_data_splits(
                df_processed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                progress_callback=lambda p, m: update_progress(0.3 + p * 0.1, m)
            )

            # Display split info
            st.info(f"""
            **Data Split Summary:**
            - Train: {split_info['train_samples']} samples ({train_ratio:.0%})
            - Validation: {split_info['val_samples']} samples ({val_ratio:.0%})
            - Test: {split_info['test_samples']} samples ({test_ratio:.0%})
            - Features: {split_info['input_dim']}
            """)

            # Step 4: Train models
            update_progress(0.4, "Starting model training...")

            # Set seed for reproducibility
            set_seed(config.training.random_seed)

            all_results = service.retrain_all_models(
                split_info=split_info,
                variants=selected_variants,
                epochs=epochs,
                device=device,
                progress_callback=lambda p, m: update_progress(0.4 + p * 0.55, m),
                research_mode=research_mode
            )

            # Step 5: Clear caches and show results
            update_progress(0.95, "Clearing caches...")
            service.clear_caches()

            update_progress(1.0, "Complete!")

            # Show results summary
            mode_str = "Research" if research_mode else "Standard"
            epochs_info = f"{research_config.epochs} epochs (locked)" if research_mode else f"{epochs} epochs"
            st.success(f"Retraining complete! Trained {len(all_results)} models in {mode_str} mode ({epochs_info}).")

            # Results table
            st.subheader("Results Summary")

            results_data = []
            for key, result in all_results.items():
                if 'error' in result:
                    results_data.append({
                        'Model': PINN_CONFIGURATIONS[key]['name'],
                        'Status': 'Failed',
                        'Error': result['error']
                    })
                else:
                    ml_metrics = result.get('ml_metrics', {})
                    fin_metrics = result.get('financial_metrics', {})
                    results_data.append({
                        'Model': result['variant_name'],
                        'Status': 'Success',
                        'MSE': f"{ml_metrics.get('mse', 0):.6f}",
                        'R²': f"{ml_metrics.get('r2', 0):.4f}",
                        'Sharpe': f"{fin_metrics.get('sharpe_ratio', 0):.3f}",
                        'Dir Acc': f"{fin_metrics.get('directional_accuracy', 0)*100:.2f}%"
                    })

            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)

            st.info("Refresh the 'Live Metrics' page to see updated results.")

        except Exception as e:
            st.error(f"Retraining failed: {e}")
            logger.exception(e)
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    st.set_page_config(page_title="Data Refresh", layout="wide")
    st.title("Data Refresh & Model Retraining")
    render_data_refresh_panel()
