"""
Batch Training Dashboard - Train multiple models with real-time visualization

Provides a unified interface for:
- Selecting and configuring multiple models for training
- Setting hyperparameters with sensible defaults
- Real-time training progress visualization
- Live loss curves and physics loss decomposition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..models.model_registry import ModelRegistry, ModelInfo
from ..training.batch_trainer import BatchTrainer, TrainingConfig, EpochProgress


logger = get_logger(__name__)


@dataclass
class ModelTrainingConfig:
    """Configuration for training a single model"""
    model_key: str
    model_name: str
    model_type: str
    architecture: str
    enabled: bool = True
    physics_constraints: Optional[Dict[str, float]] = None


@dataclass
class TrainingProgress:
    """Real-time training progress for a model"""
    model_key: str
    current_epoch: int = 0
    total_epochs: int = 100
    train_loss: float = 0.0
    val_loss: float = 0.0
    data_loss: float = 0.0
    physics_loss: float = 0.0
    learning_rate: float = 0.001
    status: str = "pending"  # pending, training, completed, failed
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    data_losses: List[float] = field(default_factory=list)
    physics_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None


# Default hyperparameters with descriptions
DEFAULT_HYPERPARAMETERS = {
    'epochs': {
        'value': 100,
        'min': 10,
        'max': 500,
        'step': 10,
        'description': 'Number of training epochs'
    },
    'learning_rate': {
        'value': 0.001,
        'min': 0.00001,
        'max': 0.1,
        'description': 'Initial learning rate (Adam optimizer)'
    },
    'batch_size': {
        'value': 32,
        'options': [16, 32, 64, 128, 256],
        'description': 'Training batch size'
    },
    'hidden_dim': {
        'value': 128,
        'options': [64, 128, 256, 512],
        'description': 'Hidden layer dimension'
    },
    'num_layers': {
        'value': 2,
        'min': 1,
        'max': 6,
        'step': 1,
        'description': 'Number of recurrent/transformer layers'
    },
    'dropout': {
        'value': 0.2,
        'min': 0.0,
        'max': 0.5,
        'step': 0.05,
        'description': 'Dropout rate for regularization'
    },
    'gradient_clip_norm': {
        'value': 1.0,
        'min': 0.1,
        'max': 10.0,
        'step': 0.1,
        'description': 'Gradient clipping norm'
    },
    'scheduler_patience': {
        'value': 5,
        'min': 3,
        'max': 20,
        'step': 1,
        'description': 'LR scheduler patience (epochs)'
    },
    'early_stopping_patience': {
        'value': 15,
        'min': 0,
        'max': 50,
        'step': 5,
        'description': 'Early stopping patience (0 to disable)'
    }
}

# Model type to display name mapping
MODEL_TYPES = {
    'lstm': ('LSTM', 'baseline', 'Long Short-Term Memory network'),
    'gru': ('GRU', 'baseline', 'Gated Recurrent Unit network'),
    'bilstm': ('BiLSTM', 'baseline', 'Bidirectional LSTM'),
    'transformer': ('Transformer', 'baseline', 'Multi-head attention transformer'),
    'pinn_baseline': ('PINN Baseline', 'pinn', 'Pure data-driven (no physics)'),
    'pinn_gbm': ('PINN GBM', 'pinn', 'Geometric Brownian Motion constraint'),
    'pinn_ou': ('PINN OU', 'pinn', 'Ornstein-Uhlenbeck mean-reversion'),
    'pinn_black_scholes': ('PINN Black-Scholes', 'pinn', 'No-arbitrage PDE constraint'),
    'pinn_gbm_ou': ('PINN GBM+OU', 'pinn', 'Combined trend + mean-reversion'),
    'pinn_global': ('PINN Global', 'pinn', 'All physics constraints combined'),
}

PHYSICS_CONSTRAINTS = {
    'pinn_baseline': {'lambda_gbm': 0.0, 'lambda_bs': 0.0, 'lambda_ou': 0.0},
    'pinn_gbm': {'lambda_gbm': 0.1, 'lambda_bs': 0.0, 'lambda_ou': 0.0},
    'pinn_ou': {'lambda_gbm': 0.0, 'lambda_bs': 0.0, 'lambda_ou': 0.1},
    'pinn_black_scholes': {'lambda_gbm': 0.0, 'lambda_bs': 0.1, 'lambda_ou': 0.0},
    'pinn_gbm_ou': {'lambda_gbm': 0.05, 'lambda_bs': 0.0, 'lambda_ou': 0.05},
    'pinn_global': {'lambda_gbm': 0.05, 'lambda_bs': 0.03, 'lambda_ou': 0.05, 'lambda_langevin': 0.02},
}


class BatchTrainingDashboard:
    """Dashboard for batch model training with real-time visualization"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = get_config()
        self.registry = ModelRegistry(project_root)
        self.models_dir = project_root / 'models'

        # Initialize session state for training
        if 'training_progress' not in st.session_state:
            st.session_state.training_progress = {}
        if 'training_active' not in st.session_state:
            st.session_state.training_active = False
        if 'model_configs' not in st.session_state:
            st.session_state.model_configs = self._get_default_model_configs()
        if 'global_hyperparams' not in st.session_state:
            st.session_state.global_hyperparams = self._get_default_hyperparams()
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = False  # NO_MOCK_DATA policy: demo mode off by default

    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters"""
        return {k: v['value'] for k, v in DEFAULT_HYPERPARAMETERS.items()}

    def _get_default_model_configs(self) -> Dict[str, ModelTrainingConfig]:
        """Get default configurations for all models"""
        configs = {}

        for key, (name, model_type, desc) in MODEL_TYPES.items():
            configs[key] = ModelTrainingConfig(
                model_key=key,
                model_name=name,
                model_type=model_type,
                architecture=key.upper(),
                enabled=True,
                physics_constraints=PHYSICS_CONSTRAINTS.get(key)
            )

        return configs

    def _load_training_history(self, model_key: str) -> Optional[Dict]:
        """Load training history from file"""
        history_path = self.models_dir / f'{model_key}_history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def render(self):
        """Main render function"""
        st.title("Batch Model Training")
        st.markdown("Train multiple models with configurable hyperparameters and real-time progress visualization.")

        # Demo mode toggle
        col1, col2 = st.columns([4, 1])
        with col2:
            st.session_state.demo_mode = st.toggle(
                "Demo Mode",
                value=st.session_state.demo_mode,
                help="Demo mode simulates training. Disable for actual model training."
            )

        if st.session_state.demo_mode:
            st.info("**Demo Mode**: Training is simulated for UI demonstration. Disable Demo Mode for actual training.")

        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Model Selection & Configuration",
            "Training Progress",
            "Results Comparison"
        ])

        with tab1:
            self._render_configuration_tab()

        with tab2:
            self._render_training_progress_tab()

        with tab3:
            self._render_results_tab()

    def _render_configuration_tab(self):
        """Render model selection and configuration"""

        # Global hyperparameters section
        st.subheader("Global Hyperparameters")
        st.markdown("These settings apply to all selected models.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.session_state.global_hyperparams['epochs'] = st.number_input(
                "Epochs",
                min_value=10,
                max_value=500,
                value=st.session_state.global_hyperparams['epochs'],
                step=10,
                help=DEFAULT_HYPERPARAMETERS['epochs']['description']
            )

            st.session_state.global_hyperparams['learning_rate'] = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=st.session_state.global_hyperparams['learning_rate'],
                help=DEFAULT_HYPERPARAMETERS['learning_rate']['description']
            )

        with col2:
            st.session_state.global_hyperparams['batch_size'] = st.selectbox(
                "Batch Size",
                options=[16, 32, 64, 128, 256],
                index=[16, 32, 64, 128, 256].index(
                    st.session_state.global_hyperparams['batch_size']
                ),
                help=DEFAULT_HYPERPARAMETERS['batch_size']['description']
            )

            st.session_state.global_hyperparams['hidden_dim'] = st.selectbox(
                "Hidden Dimension",
                options=[64, 128, 256, 512],
                index=[64, 128, 256, 512].index(
                    st.session_state.global_hyperparams['hidden_dim']
                ),
                help=DEFAULT_HYPERPARAMETERS['hidden_dim']['description']
            )

        with col3:
            st.session_state.global_hyperparams['num_layers'] = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=6,
                value=st.session_state.global_hyperparams['num_layers'],
                step=1,
                help=DEFAULT_HYPERPARAMETERS['num_layers']['description']
            )

            st.session_state.global_hyperparams['dropout'] = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.global_hyperparams['dropout'],
                step=0.05,
                help=DEFAULT_HYPERPARAMETERS['dropout']['description']
            )

        # Advanced hyperparameters (collapsible)
        with st.expander("Advanced Training Settings"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.session_state.global_hyperparams['gradient_clip_norm'] = st.number_input(
                    "Gradient Clip Norm",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.global_hyperparams['gradient_clip_norm'],
                    step=0.1,
                    help=DEFAULT_HYPERPARAMETERS['gradient_clip_norm']['description']
                )

            with col2:
                st.session_state.global_hyperparams['scheduler_patience'] = st.number_input(
                    "LR Scheduler Patience",
                    min_value=3,
                    max_value=20,
                    value=st.session_state.global_hyperparams['scheduler_patience'],
                    step=1,
                    help=DEFAULT_HYPERPARAMETERS['scheduler_patience']['description']
                )

            with col3:
                st.session_state.global_hyperparams['early_stopping_patience'] = st.number_input(
                    "Early Stopping Patience",
                    min_value=0,
                    max_value=50,
                    value=st.session_state.global_hyperparams['early_stopping_patience'],
                    step=5,
                    help="Set to 0 to disable early stopping"
                )

        st.divider()

        # Model selection section
        st.subheader("Model Selection")

        # Quick selection buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Select All", use_container_width=True):
                for key in st.session_state.model_configs:
                    st.session_state.model_configs[key].enabled = True
                st.rerun()
        with col2:
            if st.button("Deselect All", use_container_width=True):
                for key in st.session_state.model_configs:
                    st.session_state.model_configs[key].enabled = False
                st.rerun()
        with col3:
            if st.button("Select Baselines", use_container_width=True):
                for key, cfg in st.session_state.model_configs.items():
                    cfg.enabled = cfg.model_type == 'baseline'
                st.rerun()
        with col4:
            if st.button("Select PINNs", use_container_width=True):
                for key, cfg in st.session_state.model_configs.items():
                    cfg.enabled = cfg.model_type == 'pinn'
                st.rerun()

        # Model cards by category
        self._render_model_category("Baseline Models", "baseline")
        self._render_model_category("PINN Variants", "pinn")

        st.divider()

        # Training summary and start button
        self._render_training_summary()

    def _render_model_category(self, title: str, model_type: str):
        """Render models for a specific category"""
        st.markdown(f"#### {title}")

        # Get models of this type
        models = [cfg for cfg in st.session_state.model_configs.values()
                  if cfg.model_type == model_type]

        if not models:
            st.info(f"No {model_type} models available")
            return

        # Create columns for model cards (3 per row)
        cols = st.columns(3)

        for i, cfg in enumerate(models):
            with cols[i % 3]:
                self._render_model_card(cfg)

    def _render_model_card(self, cfg: ModelTrainingConfig):
        """Render a single model card with checkbox and info"""
        # Check if model is trained
        checkpoint_path = self.models_dir / f'{cfg.model_key}_best.pt'
        is_trained = checkpoint_path.exists()

        status_color = "green" if is_trained else "gray"
        status_text = "Trained" if is_trained else "Not Trained"

        # Get description
        desc = MODEL_TYPES.get(cfg.model_key, (cfg.model_name, '', ''))[2]

        with st.container():
            # Checkbox for selection
            cfg.enabled = st.checkbox(
                f"**{cfg.model_name}**",
                value=cfg.enabled,
                key=f"model_select_{cfg.model_key}"
            )

            # Model info
            st.markdown(f"""
            <div style="font-size: 0.85em; color: #666; margin-top: -10px;">
                <span style="color: {status_color};">●</span> {status_text}<br/>
                {desc}
            </div>
            """, unsafe_allow_html=True)

            # Show physics constraints for PINN models
            if cfg.physics_constraints:
                active_constraints = [
                    f"λ_{k.replace('lambda_', '')}={v}"
                    for k, v in cfg.physics_constraints.items()
                    if v > 0
                ]
                if active_constraints:
                    st.markdown(
                        f"<div style='font-size: 0.8em; color: #888; margin-top: 5px;'>"
                        f"Physics: {', '.join(active_constraints)}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='font-size: 0.8em; color: #888;'>Physics: None (baseline)</div>",
                        unsafe_allow_html=True
                    )

    def _render_training_summary(self):
        """Render training summary and start button"""
        selected_models = [cfg for cfg in st.session_state.model_configs.values() if cfg.enabled]

        st.subheader("Training Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Selected Models", len(selected_models))
        with col2:
            total_epochs = len(selected_models) * st.session_state.global_hyperparams['epochs']
            st.metric("Total Epochs", total_epochs)
        with col3:
            baseline_count = len([m for m in selected_models if m.model_type == 'baseline'])
            pinn_count = len([m for m in selected_models if m.model_type == 'pinn'])
            st.metric("Baselines / PINNs", f"{baseline_count} / {pinn_count}")
        with col4:
            lr = st.session_state.global_hyperparams['learning_rate']
            st.metric("Learning Rate", f"{lr:.4f}")

        # Selected models list
        if selected_models:
            st.markdown("**Models to train:**")
            model_names = [f"`{cfg.model_name}`" for cfg in selected_models]
            st.markdown(", ".join(model_names))
        else:
            st.warning("No models selected. Please select at least one model to train.")

        st.divider()

        # Hyperparameters summary table
        with st.expander("View Full Configuration"):
            config_df = pd.DataFrame([
                {"Parameter": "Epochs", "Value": st.session_state.global_hyperparams['epochs']},
                {"Parameter": "Learning Rate", "Value": st.session_state.global_hyperparams['learning_rate']},
                {"Parameter": "Batch Size", "Value": st.session_state.global_hyperparams['batch_size']},
                {"Parameter": "Hidden Dimension", "Value": st.session_state.global_hyperparams['hidden_dim']},
                {"Parameter": "Number of Layers", "Value": st.session_state.global_hyperparams['num_layers']},
                {"Parameter": "Dropout", "Value": st.session_state.global_hyperparams['dropout']},
                {"Parameter": "Gradient Clip Norm", "Value": st.session_state.global_hyperparams['gradient_clip_norm']},
                {"Parameter": "Scheduler Patience", "Value": st.session_state.global_hyperparams['scheduler_patience']},
                {"Parameter": "Early Stopping Patience", "Value": st.session_state.global_hyperparams['early_stopping_patience']},
            ])
            st.dataframe(config_df, hide_index=True, use_container_width=True)

        # Training buttons
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.session_state.training_active:
                st.warning("Training in progress...")
                if st.button("Stop Training", type="secondary", use_container_width=True):
                    st.session_state.training_active = False
                    st.rerun()
            else:
                button_text = "Train All Selected Models (Demo)" if st.session_state.demo_mode else "Train All Selected Models"
                if st.button(
                    button_text,
                    type="primary",
                    disabled=len(selected_models) == 0,
                    use_container_width=True
                ):
                    self._start_batch_training(selected_models)

        with col2:
            if st.button("Reset Progress", use_container_width=True):
                st.session_state.training_progress = {}
                st.rerun()

    def _start_batch_training(self, models: List[ModelTrainingConfig]):
        """Initialize and start batch training"""
        st.session_state.training_active = True

        # Initialize progress for each model
        epochs = st.session_state.global_hyperparams['epochs']
        lr = st.session_state.global_hyperparams['learning_rate']

        for cfg in models:
            st.session_state.training_progress[cfg.model_key] = TrainingProgress(
                model_key=cfg.model_key,
                total_epochs=epochs,
                learning_rate=lr,
                status="pending"
            )

        # Store training queue
        st.session_state.training_queue = [cfg.model_key for cfg in models]
        st.session_state.current_training_idx = 0

        st.rerun()

    def _render_training_progress_tab(self):
        """Render real-time training progress"""

        if not st.session_state.training_progress:
            st.info("No training in progress. Go to 'Model Selection & Configuration' tab to start training.")

            # Show existing training histories
            st.subheader("Previous Training Results")
            self._show_existing_histories()
            return

        # Check if we should continue training
        if st.session_state.training_active and hasattr(st.session_state, 'training_queue'):
            self._continue_training()

        # Overall progress
        st.subheader("Training Overview")

        progress_data = st.session_state.training_progress
        total_models = len(progress_data)
        completed = len([p for p in progress_data.values() if p.status == 'completed'])
        training = len([p for p in progress_data.values() if p.status == 'training'])
        pending = len([p for p in progress_data.values() if p.status == 'pending'])
        failed = len([p for p in progress_data.values() if p.status == 'failed'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completed", f"{completed}/{total_models}")
        with col2:
            st.metric("Training", training)
        with col3:
            st.metric("Pending", pending)
        with col4:
            st.metric("Failed", failed)

        # Overall progress bar
        overall_progress = completed / total_models if total_models > 0 else 0
        st.progress(overall_progress, text=f"Overall Progress: {overall_progress*100:.1f}%")

        st.divider()

        # Individual model progress cards
        st.subheader("Model Training Status")

        # Training models first, then others
        sorted_progress = sorted(
            progress_data.values(),
            key=lambda x: (
                0 if x.status == 'training' else
                1 if x.status == 'pending' else
                2 if x.status == 'completed' else 3
            )
        )

        for progress in sorted_progress:
            self._render_model_progress_card(progress)

        # Live loss curves
        st.divider()
        st.subheader("Live Training Curves")

        # Get models with training data
        models_with_data = [p for p in progress_data.values() if len(p.train_losses) > 0]

        if models_with_data:
            self._render_live_loss_curves(models_with_data)
        else:
            st.info("Training curves will appear here once training begins.")

        # Auto-refresh while training
        if st.session_state.training_active:
            time.sleep(0.3)
            st.rerun()

    def _show_existing_histories(self):
        """Show existing training histories from files"""
        histories = {}

        for key in MODEL_TYPES.keys():
            history = self._load_training_history(key)
            if history:
                histories[key] = history

        if not histories:
            st.info("No previous training histories found.")
            return

        st.markdown(f"Found **{len(histories)}** trained models with history.")

        # Create comparison chart
        fig = go.Figure()

        for model_key, history in histories.items():
            model_name = MODEL_TYPES.get(model_key, (model_key, '', ''))[0]

            if 'val_loss' in history:
                epochs = list(range(1, len(history['val_loss']) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=history['val_loss'],
                    mode='lines',
                    name=model_name
                ))

        fig.update_layout(
            title='Validation Loss History (Previous Training)',
            xaxis_title='Epoch',
            yaxis_title='Validation Loss',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _continue_training(self):
        """Continue training the next model in queue"""
        if not hasattr(st.session_state, 'training_queue'):
            return

        queue = st.session_state.training_queue
        current_idx = st.session_state.current_training_idx

        if current_idx >= len(queue):
            st.session_state.training_active = False
            return

        current_model_key = queue[current_idx]
        progress = st.session_state.training_progress.get(current_model_key)

        if progress is None:
            st.session_state.current_training_idx += 1
            return

        # If model is pending, start training
        if progress.status == 'pending':
            progress.status = 'training'
            progress.start_time = time.time()
            self._train_single_epoch(current_model_key)

        # If model is training, continue
        elif progress.status == 'training':
            if progress.current_epoch < progress.total_epochs:
                self._train_single_epoch(current_model_key)
            else:
                # Model completed
                progress.status = 'completed'
                progress.end_time = time.time()
                st.session_state.current_training_idx += 1

    def _train_single_epoch(self, model_key: str):
        """Train a single epoch using real BatchTrainer (NO_MOCK_DATA policy)"""
        progress = st.session_state.training_progress[model_key]
        progress.current_epoch += 1

        if st.session_state.demo_mode:
            # Demo mode still allowed for UI testing only - clearly marked
            st.warning("Demo mode: Using simulated training. Disable for real training.")
            self._simulate_epoch(progress, model_key)
        else:
            # Use real BatchTrainer for actual training
            self._train_real_epoch(progress, model_key)

    def _train_real_epoch(self, progress: TrainingProgress, model_key: str):
        """Train a real epoch using the BatchTrainer module"""
        try:
            # Get or create batch trainer
            if 'batch_trainer' not in st.session_state:
                st.session_state.batch_trainer = BatchTrainer(self.project_root)
                # Prepare data once
                if not st.session_state.batch_trainer.prepare_data():
                    raise RuntimeError("Failed to prepare training data")

            trainer = st.session_state.batch_trainer
            model_cfg = st.session_state.model_configs.get(model_key)

            # Load existing checkpoint if resuming
            history = self._load_training_history(model_key)
            if history and 'train_loss' in history:
                # Update progress from history if we have fewer epochs
                if len(history['train_loss']) >= progress.current_epoch:
                    epoch_idx = progress.current_epoch - 1
                    progress.train_loss = history['train_loss'][epoch_idx]
                    progress.val_loss = history['val_loss'][epoch_idx]
                    progress.train_losses = history['train_loss'][:progress.current_epoch]
                    progress.val_losses = history['val_loss'][:progress.current_epoch]
                    if history.get('train_data_loss'):
                        progress.data_loss = history['train_data_loss'][epoch_idx]
                        progress.data_losses = history['train_data_loss'][:progress.current_epoch]
                    if history.get('train_physics_loss'):
                        progress.physics_loss = history['train_physics_loss'][epoch_idx]
                        progress.physics_losses = history['train_physics_loss'][:progress.current_epoch]
                    return

            # Create training config
            hp = st.session_state.global_hyperparams
            config = TrainingConfig(
                model_key=model_key,
                model_type=model_cfg.model_type if model_cfg else 'lstm',
                epochs=1,  # Train one epoch at a time for UI responsiveness
                learning_rate=hp['learning_rate'],
                batch_size=hp['batch_size'],
                hidden_dim=hp['hidden_dim'],
                num_layers=hp['num_layers'],
                dropout=hp['dropout'],
                gradient_clip_norm=hp.get('gradient_clip_norm', 1.0),
                physics_constraints=model_cfg.physics_constraints if model_cfg else None
            )

            # Define progress callback
            def on_progress(key: str, epoch_progress: EpochProgress):
                progress.train_loss = epoch_progress.train_loss
                progress.val_loss = epoch_progress.val_loss
                progress.train_losses.append(epoch_progress.train_loss)
                progress.val_losses.append(epoch_progress.val_loss)
                progress.learning_rate = epoch_progress.learning_rate
                progress.learning_rates.append(epoch_progress.learning_rate)
                if epoch_progress.data_loss is not None:
                    progress.data_loss = epoch_progress.data_loss
                    progress.data_losses.append(epoch_progress.data_loss)
                if epoch_progress.physics_loss is not None:
                    progress.physics_loss = epoch_progress.physics_loss
                    progress.physics_losses.append(epoch_progress.physics_loss)

            # Train the model
            trainer.train_all([config], progress_callback=on_progress, save_checkpoints=True)

        except Exception as e:
            logger.error(f"Real training failed for {model_key}: {e}")
            progress.status = 'failed'
            progress.error_message = str(e)
            st.error(f"Training failed: {e}")

    def _simulate_epoch(self, progress: TrainingProgress, model_key: str):
        """Simulate training an epoch with realistic loss curves"""
        epoch = progress.current_epoch
        total = progress.total_epochs

        # More realistic loss simulation with model-specific characteristics
        model_cfg = st.session_state.model_configs.get(model_key)
        is_pinn = model_cfg and model_cfg.model_type == 'pinn'

        # Base loss curves (exponential decay + noise)
        base_train = 0.5 * np.exp(-epoch * 0.025) + 0.05
        base_val = 0.55 * np.exp(-epoch * 0.022) + 0.06

        # Add model-specific variations
        if 'transformer' in model_key:
            base_train *= 0.9
            base_val *= 0.95
        elif 'pinn_global' in model_key:
            base_train *= 1.1
            base_val *= 1.05

        # Add noise
        noise_scale = max(0.02, 0.05 * (1 - epoch / total))
        progress.train_loss = max(0.01, base_train + np.random.uniform(-noise_scale, noise_scale))
        progress.val_loss = max(0.02, base_val + np.random.uniform(-noise_scale * 1.5, noise_scale * 1.5))

        # For PINN models, simulate physics loss decomposition
        if is_pinn:
            physics_weight = 0.15 + 0.05 * (epoch / total)  # Increasing physics weight
            progress.data_loss = progress.train_loss * (1 - physics_weight)
            progress.physics_loss = progress.train_loss * physics_weight
            progress.data_losses.append(progress.data_loss)
            progress.physics_losses.append(progress.physics_loss)

        # Learning rate decay (step decay every 20 epochs)
        if epoch > 1 and epoch % 20 == 0:
            progress.learning_rate *= 0.5

        # Store history
        progress.train_losses.append(progress.train_loss)
        progress.val_losses.append(progress.val_loss)
        progress.learning_rates.append(progress.learning_rate)

    def _render_model_progress_card(self, progress: TrainingProgress):
        """Render progress card for a single model"""
        model_cfg = st.session_state.model_configs.get(progress.model_key)
        model_name = model_cfg.model_name if model_cfg else progress.model_key

        # Status indicator
        status_icons = {
            'pending': '⏳',
            'training': '🔄',
            'completed': '✅',
            'failed': '❌'
        }
        status_icon = status_icons.get(progress.status, '⚪')

        with st.expander(
            f"{status_icon} {model_name} - {progress.status.upper()}",
            expanded=(progress.status == 'training')
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Epoch",
                    f"{progress.current_epoch}/{progress.total_epochs}"
                )
            with col2:
                st.metric(
                    "Train Loss",
                    f"{progress.train_loss:.4f}" if progress.train_loss else "-"
                )
            with col3:
                st.metric(
                    "Val Loss",
                    f"{progress.val_loss:.4f}" if progress.val_loss else "-"
                )
            with col4:
                st.metric(
                    "Learning Rate",
                    f"{progress.learning_rate:.6f}"
                )

            # Progress bar
            epoch_progress = progress.current_epoch / progress.total_epochs if progress.total_epochs > 0 else 0
            st.progress(epoch_progress)

            # Physics loss for PINN models
            if progress.data_losses and progress.physics_losses:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Loss", f"{progress.data_loss:.4f}")
                with col2:
                    st.metric("Physics Loss", f"{progress.physics_loss:.4f}")
                with col3:
                    ratio = progress.physics_loss / (progress.data_loss + 1e-8)
                    st.metric("Physics/Data Ratio", f"{ratio:.2%}")

            # Training time
            if progress.start_time:
                if progress.status == 'training':
                    elapsed = time.time() - progress.start_time
                elif progress.end_time:
                    elapsed = progress.end_time - progress.start_time
                else:
                    elapsed = 0

                st.caption(f"Elapsed time: {elapsed:.1f}s")

    def _render_live_loss_curves(self, models: List[TrainingProgress]):
        """Render live loss curves for all training models"""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Training Loss',
                'Validation Loss',
                'Physics Loss Decomposition (PINN)',
                'Learning Rate Schedule'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i, progress in enumerate(models):
            color = colors[i % len(colors)]
            model_cfg = st.session_state.model_configs.get(progress.model_key)
            model_name = model_cfg.model_name if model_cfg else progress.model_key

            epochs = list(range(1, len(progress.train_losses) + 1))

            # Training loss
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=progress.train_losses,
                    mode='lines', name=f'{model_name}',
                    line=dict(color=color),
                    legendgroup=model_name,
                    showlegend=True
                ),
                row=1, col=1
            )

            # Validation loss
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=progress.val_losses,
                    mode='lines', name=f'{model_name}',
                    line=dict(color=color, dash='dash'),
                    legendgroup=model_name,
                    showlegend=False
                ),
                row=1, col=2
            )

            # Physics loss decomposition (for PINN models)
            if progress.data_losses and progress.physics_losses:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=progress.data_losses,
                        mode='lines', name=f'{model_name} (Data)',
                        line=dict(color=color),
                        legendgroup=model_name,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=progress.physics_losses,
                        mode='lines', name=f'{model_name} (Physics)',
                        line=dict(color=color, dash='dot'),
                        legendgroup=model_name,
                        showlegend=False
                    ),
                    row=2, col=1
                )

            # Learning rate
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=progress.learning_rates,
                    mode='lines', name=f'{model_name}',
                    line=dict(color=color),
                    legendgroup=model_name,
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=600,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Axis labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)

        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate", type='log', row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

    def _render_results_tab(self):
        """Render training results comparison"""

        completed_models = [
            p for p in st.session_state.training_progress.values()
            if p.status == 'completed'
        ]

        if not completed_models:
            st.info("No completed training sessions yet. Results will appear here after models finish training.")

            # Show existing results from files
            st._show_existing_results()
            return

        st.subheader("Training Results Summary")

        # Create comparison table
        results_data = []
        for progress in completed_models:
            model_cfg = st.session_state.model_configs.get(progress.model_key)

            best_val_loss = min(progress.val_losses) if progress.val_losses else None
            best_epoch = progress.val_losses.index(best_val_loss) + 1 if best_val_loss else None

            results_data.append({
                'Model': model_cfg.model_name if model_cfg else progress.model_key,
                'Type': model_cfg.model_type if model_cfg else 'unknown',
                'Final Train Loss': progress.train_losses[-1] if progress.train_losses else None,
                'Final Val Loss': progress.val_losses[-1] if progress.val_losses else None,
                'Best Val Loss': best_val_loss,
                'Best Epoch': best_epoch,
                'Training Time (s)': progress.end_time - progress.start_time if progress.end_time and progress.start_time else None
            })

        results_df = pd.DataFrame(results_data)

        # Style the dataframe
        st.dataframe(
            results_df.style.format({
                'Final Train Loss': '{:.4f}',
                'Final Val Loss': '{:.4f}',
                'Best Val Loss': '{:.4f}',
                'Training Time (s)': '{:.1f}'
            }).highlight_min(subset=['Best Val Loss'], color='lightgreen'),
            use_container_width=True
        )

        st.divider()

        # Final loss comparison chart
        st.subheader("Model Comparison")

        fig = go.Figure()

        model_names = [r['Model'] for r in results_data]
        train_losses = [r['Final Train Loss'] for r in results_data]
        val_losses = [r['Final Val Loss'] for r in results_data]

        fig.add_trace(go.Bar(
            name='Train Loss',
            x=model_names,
            y=train_losses,
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            name='Val Loss',
            x=model_names,
            y=val_losses,
            marker_color='#ff7f0e'
        ))

        fig.update_layout(
            barmode='group',
            title='Final Loss Comparison',
            yaxis_title='Loss',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Best validation loss comparison
        st.subheader("Best Validation Loss Ranking")

        sorted_results = sorted(results_data, key=lambda x: x['Best Val Loss'] or float('inf'))

        for i, result in enumerate(sorted_results):
            col1, col2, col3 = st.columns([0.5, 3, 1])
            with col1:
                if i == 0:
                    st.markdown("🥇")
                elif i == 1:
                    st.markdown("🥈")
                elif i == 2:
                    st.markdown("🥉")
                else:
                    st.markdown(f"#{i+1}")
            with col2:
                st.markdown(f"**{result['Model']}**")
            with col3:
                st.markdown(f"`{result['Best Val Loss']:.4f}`")


    def _show_existing_results(self):
        """Show results from existing training files (NO_MOCK_DATA policy: real data only)"""
        st.subheader("Existing Training Results")

        # Load all available training histories
        models_dir = self.project_root / 'Models'
        results_dir = self.project_root / 'results'

        existing_results = []

        for model_key in MODEL_TYPES.keys():
            history = None

            # Try to load history from Models directory
            history_path = models_dir / f'{model_key}_history.json'
            if history_path.exists():
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(f"Could not load {history_path}: {e}")

            # Try to load from results directory
            results_path = results_dir / f'{model_key}_results.json'
            results_data = None
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        results_data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(f"Could not load {results_path}: {e}")

            if history or results_data:
                name, model_type, desc = MODEL_TYPES[model_key]

                # Extract metrics
                if history:
                    train_losses = history.get('train_loss', [])
                    val_losses = history.get('val_loss', [])
                    final_train = train_losses[-1] if train_losses else None
                    final_val = val_losses[-1] if val_losses else None
                    best_val = min(val_losses) if val_losses else None
                    epochs_trained = len(train_losses)
                elif results_data and 'results' in results_data:
                    res = results_data['results']
                    final_train = res.get('final_train_loss')
                    final_val = res.get('final_val_loss')
                    best_val = res.get('best_val_loss')
                    epochs_trained = res.get('total_epochs_trained', 0)
                else:
                    continue

                existing_results.append({
                    'Model': name,
                    'Type': model_type.upper(),
                    'Epochs': epochs_trained,
                    'Final Train Loss': final_train,
                    'Final Val Loss': final_val,
                    'Best Val Loss': best_val
                })

        if not existing_results:
            st.info("No existing training results found. Train models first.")
            return

        # Display results table
        results_df = pd.DataFrame(existing_results)

        st.dataframe(
            results_df.style.format({
                'Final Train Loss': '{:.4f}',
                'Final Val Loss': '{:.4f}',
                'Best Val Loss': '{:.4f}'
            }).highlight_min(subset=['Best Val Loss'], color='lightgreen'),
            use_container_width=True
        )

        # Best model ranking
        st.subheader("Best Models by Validation Loss")
        sorted_results = sorted(
            [r for r in existing_results if r['Best Val Loss'] is not None],
            key=lambda x: x['Best Val Loss']
        )

        for i, result in enumerate(sorted_results[:5]):
            col1, col2, col3 = st.columns([0.5, 3, 1])
            with col1:
                medals = ["1st", "2nd", "3rd", "4th", "5th"]
                st.markdown(f"**{medals[i]}**")
            with col2:
                st.markdown(f"{result['Model']} ({result['Type']})")
            with col3:
                st.markdown(f"`{result['Best Val Loss']:.4f}`")


def render_batch_training_dashboard():
    """Entry point for the batch training dashboard"""
    config = get_config()
    dashboard = BatchTrainingDashboard(config.project_root)
    dashboard.render()
