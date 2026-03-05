"""
Comprehensive Training Visualization Dashboard

Provides detailed visualizations of the training process including:
- Loss curves (train vs validation)
- Learning rate schedules
- Physics loss vs data loss decomposition (for PINN models)
- Convergence analysis
- Overfitting detection
- Model comparison across training
- Best epoch highlighting
- Training statistics summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized

ensure_logger_initialized()
logger = get_logger(__name__)


# Model categories for organization
MODEL_CATEGORIES = {
    'baseline': {
        'lstm': 'LSTM',
        'gru': 'GRU',
        'bilstm': 'BiLSTM',
        'transformer': 'Transformer'
    },
    'pinn_variants': {
        'pinn_baseline': 'PINN Baseline',
        'pinn_gbm': 'PINN GBM',
        'pinn_ou': 'PINN OU',
        'pinn_black_scholes': 'PINN Black-Scholes',
        'pinn_gbm_ou': 'PINN GBM+OU',
        'pinn_global': 'PINN Global'
    },
    'advanced': {
        'stacked_pinn': 'Stacked PINN',
        'residual_pinn': 'Residual PINN'
    }
}

# Flatten for easy lookup
ALL_MODELS = {}
for category in MODEL_CATEGORIES.values():
    ALL_MODELS.update(category)


@dataclass
class TrainingStats:
    """Statistics computed from training history"""
    model_name: str
    total_epochs: int
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float
    initial_lr: float
    final_lr: float
    lr_reductions: int
    convergence_rate: float  # Average loss reduction per epoch
    overfitting_score: float  # val_loss / train_loss ratio at end
    has_physics_loss: bool


class TrainingDashboard:
    """Dashboard for comprehensive training visualization"""

    def __init__(self):
        self.config = get_config()
        self.models_dir = self.config.project_root / 'Models'
        self.checkpoints_dir = self.config.project_root / 'checkpoints'

    def load_training_history(self, model_name: str) -> Optional[Dict]:
        """Load training history for a specific model"""
        # Try multiple locations
        patterns = [
            self.models_dir / f'{model_name}_history.json',
            self.models_dir / 'stacked_pinn' / f'{model_name}_history.json',
            self.checkpoints_dir / f'{model_name}_history.json',
            self.checkpoints_dir / 'history.json'
        ]

        for path in patterns:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        return None

    def load_all_histories(self) -> Dict[str, Dict]:
        """Load training histories for all available models"""
        histories = {}

        # Check all known model names
        all_model_names = [
            'lstm', 'gru', 'bilstm', 'transformer',
            'pinn_baseline', 'pinn_gbm', 'pinn_ou', 'pinn_black_scholes',
            'pinn_gbm_ou', 'pinn_global', 'stacked_pinn', 'residual_pinn'
        ]

        for model_name in all_model_names:
            history = self.load_training_history(model_name)
            if history and 'train_loss' in history and len(history['train_loss']) > 0:
                histories[model_name] = history
                logger.info(f"Loaded training history for {model_name}")

        return histories

    def compute_training_stats(self, model_name: str, history: Dict) -> TrainingStats:
        """Compute statistics from training history"""
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        learning_rates = history.get('learning_rates', [])
        physics_loss = history.get('train_physics_loss', [])

        if not train_loss:
            return None

        # Find best epoch (lowest validation loss)
        if val_loss:
            best_epoch = int(np.argmin(val_loss)) + 1
            best_val_loss = float(min(val_loss))
        else:
            best_epoch = len(train_loss)
            best_val_loss = float('nan')

        # Compute convergence rate (average loss reduction)
        if len(train_loss) > 1:
            convergence_rate = (train_loss[0] - train_loss[-1]) / len(train_loss)
        else:
            convergence_rate = 0.0

        # Compute overfitting score
        if val_loss and train_loss:
            overfitting_score = val_loss[-1] / max(train_loss[-1], 1e-10)
        else:
            overfitting_score = 1.0

        # Count learning rate reductions
        lr_reductions = 0
        if learning_rates and len(learning_rates) > 1:
            for i in range(1, len(learning_rates)):
                if learning_rates[i] < learning_rates[i-1]:
                    lr_reductions += 1

        return TrainingStats(
            model_name=model_name,
            total_epochs=len(train_loss),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_train_loss=float(train_loss[-1]) if train_loss else float('nan'),
            final_val_loss=float(val_loss[-1]) if val_loss else float('nan'),
            initial_lr=float(learning_rates[0]) if learning_rates else 0.001,
            final_lr=float(learning_rates[-1]) if learning_rates else 0.001,
            lr_reductions=lr_reductions,
            convergence_rate=convergence_rate,
            overfitting_score=overfitting_score,
            has_physics_loss=bool(physics_loss and len(physics_loss) > 0)
        )

    def render_overview(self, histories: Dict[str, Dict]):
        """Render training overview with summary statistics"""
        st.subheader("Training Overview")

        # Compute stats for all models
        all_stats = []
        for model_name, history in histories.items():
            stats = self.compute_training_stats(model_name, history)
            if stats:
                all_stats.append(stats)

        if not all_stats:
            st.warning("No training statistics available.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Models Trained", len(all_stats))
        with col2:
            avg_epochs = np.mean([s.total_epochs for s in all_stats])
            st.metric("Avg Epochs", f"{avg_epochs:.1f}")
        with col3:
            best_model = min(all_stats, key=lambda x: x.best_val_loss if not np.isnan(x.best_val_loss) else float('inf'))
            st.metric("Best Model", ALL_MODELS.get(best_model.model_name, best_model.model_name))
        with col4:
            st.metric("Best Val Loss", f"{best_model.best_val_loss:.6f}")

        # Training statistics table
        st.markdown("### Training Statistics Summary")

        stats_data = []
        for stats in all_stats:
            stats_data.append({
                'Model': ALL_MODELS.get(stats.model_name, stats.model_name),
                'Epochs': stats.total_epochs,
                'Best Epoch': stats.best_epoch,
                'Best Val Loss': stats.best_val_loss,
                'Final Train Loss': stats.final_train_loss,
                'Final Val Loss': stats.final_val_loss,
                'LR Reductions': stats.lr_reductions,
                'Overfitting Score': stats.overfitting_score,
                'Physics Loss': 'Yes' if stats.has_physics_loss else 'No'
            })

        df = pd.DataFrame(stats_data)

        # Style the dataframe
        styled_df = df.style.highlight_min(
            subset=['Best Val Loss', 'Final Val Loss'],
            color='lightgreen'
        ).highlight_max(
            subset=['Epochs'],
            color='lightyellow'
        ).format({
            'Best Val Loss': '{:.6f}',
            'Final Train Loss': '{:.6f}',
            'Final Val Loss': '{:.6f}',
            'Overfitting Score': '{:.2f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Training completion chart
        st.markdown("### Training Progress by Model")

        fig = go.Figure()

        # Add bars for epochs trained
        models = [ALL_MODELS.get(s.model_name, s.model_name) for s in all_stats]
        epochs = [s.total_epochs for s in all_stats]
        best_epochs = [s.best_epoch for s in all_stats]

        fig.add_trace(go.Bar(
            name='Total Epochs',
            x=models,
            y=epochs,
            marker_color='steelblue',
            opacity=0.7
        ))

        fig.add_trace(go.Scatter(
            name='Best Epoch',
            x=models,
            y=best_epochs,
            mode='markers',
            marker=dict(color='red', size=12, symbol='star')
        ))

        fig.update_layout(
            title='Training Epochs by Model (Star = Best Epoch)',
            xaxis_title='Model',
            yaxis_title='Epochs',
            height=400,
            template='plotly_white',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_loss_curves(self, histories: Dict[str, Dict]):
        """Render loss curves for all models"""
        st.subheader("Loss Curves")

        # Model selection
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_models = st.multiselect(
                "Select Models to Compare",
                options=list(histories.keys()),
                default=list(histories.keys())[:4],
                format_func=lambda x: ALL_MODELS.get(x, x)
            )

        with col2:
            show_val = st.checkbox("Show Validation Loss", value=True)
            log_scale = st.checkbox("Log Scale", value=False)

        if not selected_models:
            st.warning("Please select at least one model.")
            return

        # Create subplots for train and validation loss
        fig = make_subplots(
            rows=1, cols=2 if show_val else 1,
            subplot_titles=('Training Loss', 'Validation Loss') if show_val else ('Training Loss',),
            horizontal_spacing=0.1
        )

        # Color palette
        colors = px.colors.qualitative.Set2

        for i, model_name in enumerate(selected_models):
            history = histories[model_name]
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            epochs = list(range(1, len(train_loss) + 1))

            color = colors[i % len(colors)]
            model_display = ALL_MODELS.get(model_name, model_name)

            # Training loss
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_loss,
                    mode='lines',
                    name=f'{model_display}',
                    line=dict(color=color, width=2),
                    legendgroup=model_name,
                    showlegend=True
                ),
                row=1, col=1
            )

            # Validation loss
            if show_val and val_loss:
                fig.add_trace(
                    go.Scatter(
                        x=epochs[:len(val_loss)],
                        y=val_loss,
                        mode='lines',
                        name=f'{model_display} (Val)',
                        line=dict(color=color, width=2, dash='dash'),
                        legendgroup=model_name,
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # Mark best epoch
                best_epoch = int(np.argmin(val_loss)) + 1
                best_val = min(val_loss)
                fig.add_trace(
                    go.Scatter(
                        x=[best_epoch],
                        y=[best_val],
                        mode='markers',
                        marker=dict(color=color, size=10, symbol='star'),
                        name=f'{model_display} Best',
                        legendgroup=model_name,
                        showlegend=False
                    ),
                    row=1, col=2
                )

        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
        )

        if log_scale:
            fig.update_yaxes(type='log')

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)

        if show_val:
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    def render_single_model_analysis(self, histories: Dict[str, Dict]):
        """Render detailed analysis for a single model"""
        st.subheader("Single Model Deep Dive")

        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            options=list(histories.keys()),
            format_func=lambda x: ALL_MODELS.get(x, x)
        )

        history = histories[selected_model]
        stats = self.compute_training_stats(selected_model, history)

        if not stats:
            st.warning("No statistics available for this model.")
            return

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Epochs", stats.total_epochs)
            st.metric("Best Epoch", stats.best_epoch)

        with col2:
            st.metric("Best Val Loss", f"{stats.best_val_loss:.6f}")
            st.metric("Final Val Loss", f"{stats.final_val_loss:.6f}")

        with col3:
            st.metric("Initial LR", f"{stats.initial_lr:.6f}")
            st.metric("Final LR", f"{stats.final_lr:.6f}")

        with col4:
            st.metric("LR Reductions", stats.lr_reductions)
            overfit_color = "inverse" if stats.overfitting_score > 5 else "normal"
            st.metric("Overfitting Score", f"{stats.overfitting_score:.2f}")

        # Detailed plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss Curves with Best Epoch',
                'Learning Rate Schedule',
                'Train-Val Gap (Overfitting Indicator)',
                'Convergence Analysis'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        learning_rates = history.get('learning_rates', [])
        epochs = list(range(1, len(train_loss) + 1))

        # 1. Loss curves with best epoch marker
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

        if val_loss:
            fig.add_trace(
                go.Scatter(x=epochs[:len(val_loss)], y=val_loss, name='Val Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )

            # Best epoch marker
            fig.add_vline(
                x=stats.best_epoch, line_dash="dash", line_color="green",
                annotation_text=f"Best: {stats.best_epoch}",
                row=1, col=1
            )

        # 2. Learning rate schedule
        if learning_rates:
            fig.add_trace(
                go.Scatter(x=epochs[:len(learning_rates)], y=learning_rates,
                          name='Learning Rate', line=dict(color='green', width=2),
                          fill='tozeroy'),
                row=1, col=2
            )

        # 3. Train-Val gap (overfitting indicator)
        if val_loss and len(val_loss) == len(train_loss):
            gap = [v - t for v, t in zip(val_loss, train_loss)]
            fig.add_trace(
                go.Scatter(x=epochs, y=gap, name='Val-Train Gap',
                          line=dict(color='orange', width=2),
                          fill='tozeroy'),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # 4. Convergence analysis (smoothed loss trend)
        if len(train_loss) > 5:
            window = min(5, len(train_loss) // 3)
            smoothed = pd.Series(train_loss).rolling(window=window, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(x=epochs, y=smoothed, name='Smoothed Train Loss',
                          line=dict(color='purple', width=2)),
                row=2, col=2
            )

            # Loss reduction per epoch
            reductions = [train_loss[i-1] - train_loss[i] for i in range(1, len(train_loss))]
            fig.add_trace(
                go.Bar(x=epochs[1:], y=reductions, name='Loss Reduction',
                      marker_color=['green' if r > 0 else 'red' for r in reductions],
                      opacity=0.6),
                row=2, col=2
            )

        fig.update_layout(height=700, template='plotly_white', showlegend=True)

        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
        fig.update_yaxes(title_text="Gap (Val - Train)", row=2, col=1)
        fig.update_yaxes(title_text="Loss / Reduction", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Physics loss analysis for PINN models
        physics_loss = history.get('train_physics_loss', [])
        data_loss = history.get('train_data_loss', [])

        if physics_loss and data_loss and len(physics_loss) > 0 and len(data_loss) > 0:
            st.markdown("### Physics Loss Decomposition (PINN)")

            fig_physics = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Data Loss vs Physics Loss', 'Loss Ratio Over Training')
            )

            # Data vs Physics loss
            fig_physics.add_trace(
                go.Scatter(x=epochs[:len(data_loss)], y=data_loss, name='Data Loss',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig_physics.add_trace(
                go.Scatter(x=epochs[:len(physics_loss)], y=physics_loss, name='Physics Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )

            # Loss ratio
            if len(physics_loss) == len(data_loss):
                ratio = [p / max(d, 1e-10) for p, d in zip(physics_loss, data_loss)]
                fig_physics.add_trace(
                    go.Scatter(x=epochs[:len(ratio)], y=ratio, name='Physics/Data Ratio',
                              line=dict(color='purple', width=2), fill='tozeroy'),
                    row=1, col=2
                )

            fig_physics.update_layout(height=400, template='plotly_white')
            fig_physics.update_xaxes(title_text="Epoch")
            fig_physics.update_yaxes(title_text="Loss", row=1, col=1)
            fig_physics.update_yaxes(title_text="Ratio", row=1, col=2)

            st.plotly_chart(fig_physics, use_container_width=True)

            st.info("""
            **Physics Loss Decomposition:**
            - **Data Loss**: MSE between predictions and actual targets
            - **Physics Loss**: Violation of physics constraints (GBM, OU, Black-Scholes)
            - **Ratio**: Lower ratio = model adheres better to physics constraints
            """)

    def render_model_comparison(self, histories: Dict[str, Dict]):
        """Render comparative analysis across all models"""
        st.subheader("Model Training Comparison")

        # Compute stats for all
        all_stats = []
        for model_name, history in histories.items():
            stats = self.compute_training_stats(model_name, history)
            if stats:
                all_stats.append(stats)

        if not all_stats:
            st.warning("No training data available for comparison.")
            return

        # Convergence comparison
        st.markdown("### Convergence Speed Comparison")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Best Validation Loss', 'Convergence Rate')
        )

        models = [ALL_MODELS.get(s.model_name, s.model_name) for s in all_stats]
        best_losses = [s.best_val_loss for s in all_stats]
        convergence_rates = [s.convergence_rate * 1000 for s in all_stats]  # Scale for visibility

        # Best validation loss
        colors = ['lightgreen' if l == min(best_losses) else 'steelblue' for l in best_losses]
        fig.add_trace(
            go.Bar(x=models, y=best_losses, marker_color=colors, showlegend=False),
            row=1, col=1
        )

        # Convergence rate
        fig.add_trace(
            go.Bar(x=models, y=convergence_rates, marker_color='coral', showlegend=False),
            row=1, col=2
        )

        fig.update_layout(height=400, template='plotly_white')
        fig.update_yaxes(title_text="Best Val Loss", row=1, col=1)
        fig.update_yaxes(title_text="Convergence Rate (x1000)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Overfitting analysis
        st.markdown("### Overfitting Analysis")

        overfit_scores = [s.overfitting_score for s in all_stats]

        fig_overfit = go.Figure()
        fig_overfit.add_trace(go.Bar(
            x=models,
            y=overfit_scores,
            marker_color=['red' if o > 5 else 'orange' if o > 2 else 'green' for o in overfit_scores],
            text=[f'{o:.2f}' for o in overfit_scores],
            textposition='outside'
        ))

        fig_overfit.add_hline(y=1.0, line_dash="dash", line_color="green",
                             annotation_text="No Overfitting (Ratio=1)")
        fig_overfit.add_hline(y=2.0, line_dash="dash", line_color="orange",
                             annotation_text="Mild Overfitting")
        fig_overfit.add_hline(y=5.0, line_dash="dash", line_color="red",
                             annotation_text="Severe Overfitting")

        fig_overfit.update_layout(
            title='Overfitting Score (Val Loss / Train Loss)',
            xaxis_title='Model',
            yaxis_title='Overfitting Score',
            height=400,
            template='plotly_white'
        )

        st.plotly_chart(fig_overfit, use_container_width=True)

        st.info("""
        **Overfitting Score Interpretation:**
        - **< 2**: Good generalization
        - **2 - 5**: Mild overfitting, consider regularization
        - **> 5**: Severe overfitting, model memorizing training data
        """)

        # All models loss curves overlay
        st.markdown("### All Models - Training Loss Comparison")

        fig_all = go.Figure()
        colors = px.colors.qualitative.Set3

        for i, (model_name, history) in enumerate(histories.items()):
            train_loss = history.get('train_loss', [])
            epochs = list(range(1, len(train_loss) + 1))

            fig_all.add_trace(go.Scatter(
                x=epochs,
                y=train_loss,
                mode='lines',
                name=ALL_MODELS.get(model_name, model_name),
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig_all.update_layout(
            title='Training Loss Convergence - All Models',
            xaxis_title='Epoch',
            yaxis_title='Training Loss',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig_all, use_container_width=True)

    def render_learning_rate_analysis(self, histories: Dict[str, Dict]):
        """Render learning rate schedule analysis"""
        st.subheader("Learning Rate Schedules")

        fig = go.Figure()
        colors = px.colors.qualitative.Pastel

        for i, (model_name, history) in enumerate(histories.items()):
            lrs = history.get('learning_rates', [])
            if lrs:
                epochs = list(range(1, len(lrs) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=lrs,
                    mode='lines+markers',
                    name=ALL_MODELS.get(model_name, model_name),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title='Learning Rate Schedule by Model',
            xaxis_title='Epoch',
            yaxis_title='Learning Rate',
            height=450,
            template='plotly_white',
            yaxis_type='log'  # Log scale for learning rates
        )

        st.plotly_chart(fig, use_container_width=True)

        # LR reduction summary
        st.markdown("### Learning Rate Reduction Summary")

        lr_data = []
        for model_name, history in histories.items():
            lrs = history.get('learning_rates', [])
            if lrs:
                reductions = sum(1 for i in range(1, len(lrs)) if lrs[i] < lrs[i-1])
                lr_data.append({
                    'Model': ALL_MODELS.get(model_name, model_name),
                    'Initial LR': lrs[0],
                    'Final LR': lrs[-1],
                    'Reduction Factor': lrs[0] / max(lrs[-1], 1e-10),
                    'LR Reductions': reductions
                })

        if lr_data:
            df = pd.DataFrame(lr_data)
            st.dataframe(df.style.format({
                'Initial LR': '{:.6f}',
                'Final LR': '{:.6f}',
                'Reduction Factor': '{:.1f}x'
            }), use_container_width=True)


def render_training_visualizations():
    """Main function to render training visualizations in Streamlit"""
    st.header("Training Visualizations")

    st.markdown("""
    Comprehensive analysis of model training including loss curves, learning rate schedules,
    convergence analysis, and overfitting detection.
    """)

    # Initialize dashboard
    dashboard = TrainingDashboard()

    # Load all training histories
    with st.spinner("Loading training histories..."):
        histories = dashboard.load_all_histories()

    if not histories:
        st.warning("No training histories found.")
        st.markdown("""
        ### To Generate Training Data:

        **Train baseline models:**
        ```bash
        python -m src.training.train --model lstm --epochs 50
        python -m src.training.train --model gru --epochs 50
        ```

        **Train PINN variants:**
        ```bash
        python src/training/train_pinn_variants.py --epochs 100
        ```

        **Train advanced PINN models:**
        ```bash
        python src/training/train_stacked_pinn.py --model-type stacked --epochs 100
        ```
        """)
        return

    st.success(f"Loaded training history for {len(histories)} models")

    # Display model categories
    st.markdown("### Available Models:")
    cols = st.columns(3)

    baseline_models = [m for m in histories.keys() if m in MODEL_CATEGORIES['baseline']]
    pinn_models = [m for m in histories.keys() if m in MODEL_CATEGORIES['pinn_variants']]
    advanced_models = [m for m in histories.keys() if m in MODEL_CATEGORIES['advanced']]

    with cols[0]:
        st.markdown("**Baseline Models**")
        for m in baseline_models:
            st.markdown(f"- {ALL_MODELS.get(m, m)}")

    with cols[1]:
        st.markdown("**PINN Variants**")
        for m in pinn_models:
            st.markdown(f"- {ALL_MODELS.get(m, m)}")

    with cols[2]:
        st.markdown("**Advanced Models**")
        for m in advanced_models:
            st.markdown(f"- {ALL_MODELS.get(m, m)}")

    st.markdown("---")

    # Sub-navigation
    section = st.radio(
        "Select Visualization",
        [
            "Overview",
            "Loss Curves",
            "Single Model Analysis",
            "Model Comparison",
            "Learning Rate Analysis"
        ],
        horizontal=True
    )

    st.markdown("---")

    if section == "Overview":
        dashboard.render_overview(histories)

    elif section == "Loss Curves":
        dashboard.render_loss_curves(histories)

    elif section == "Single Model Analysis":
        dashboard.render_single_model_analysis(histories)

    elif section == "Model Comparison":
        dashboard.render_model_comparison(histories)

    elif section == "Learning Rate Analysis":
        dashboard.render_learning_rate_analysis(histories)


def main():
    """Standalone dashboard for training visualizations"""
    st.set_page_config(
        page_title="Training Visualizations",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Training Visualizations Dashboard")

    render_training_visualizations()


if __name__ == '__main__':
    main()
