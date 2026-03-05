"""
Monte Carlo Simulation Dashboard

Interactive visualization of Monte Carlo simulations for PINN models:
- Confidence interval projections
- VaR and CVaR risk metrics
- Stress test scenarios
- Path distribution analysis

Run: streamlit run src/web/monte_carlo_dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.evaluation.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults,
    compute_var_cvar
)
from src.models.model_registry import get_model_registry
import time

ensure_logger_initialized()
logger = get_logger(__name__)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


class RealTimeMonteCarloSimulator:
    """Real-time Monte Carlo simulator that yields paths incrementally"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.model.eval()

    def simulate_paths_streaming(
        self,
        initial_data: np.ndarray,
        horizon: int = 30,
        n_simulations: int = 1000,
        volatility: float = 0.2,
        batch_size: int = 10
    ):
        """
        Generator that yields paths in batches for real-time visualization

        Yields:
            Dict with current paths, statistics, and progress
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Ensure proper shape
        if len(initial_data.shape) == 2:
            initial_data = initial_data[np.newaxis, :, :]

        x = torch.FloatTensor(initial_data).to(self.device)
        all_paths = []

        with torch.no_grad():
            for batch_start in range(0, n_simulations, batch_size):
                batch_end = min(batch_start + batch_size, n_simulations)
                batch_paths = []

                for sim in range(batch_start, batch_end):
                    current_x = x.clone()
                    path = []

                    for step in range(horizon):
                        pred = self.model(current_x)
                        if isinstance(pred, tuple):
                            pred = pred[0]

                        pred_value = pred.cpu().numpy().flatten()[-1]
                        daily_vol = volatility / np.sqrt(252)
                        noise = np.random.normal(0, daily_vol)
                        noisy_pred = pred_value * (1 + noise)
                        path.append(noisy_pred)
                        current_x = self._update_sequence(current_x, noisy_pred)

                    batch_paths.append(path)

                all_paths.extend(batch_paths)
                paths_array = np.array(all_paths)

                # Compute running statistics
                stats = self._compute_running_stats(paths_array)

                yield {
                    'paths': paths_array,
                    'n_completed': len(all_paths),
                    'n_total': n_simulations,
                    'progress': len(all_paths) / n_simulations,
                    'stats': stats,
                    'horizon': horizon
                }

    def _update_sequence(self, x: torch.Tensor, new_value: float) -> torch.Tensor:
        x_new = x.clone()
        x_new[:, :-1, :] = x[:, 1:, :]
        if isinstance(new_value, (np.ndarray, np.float32, np.float64)):
            new_value = float(new_value)
        x_new[:, -1, 0] = new_value
        return x_new

    def _compute_running_stats(self, paths: np.ndarray) -> Dict:
        """Compute statistics from current paths"""
        mean_path = np.mean(paths, axis=0)
        median_path = np.median(paths, axis=0)
        lower_ci = np.percentile(paths, 2.5, axis=0)
        upper_ci = np.percentile(paths, 97.5, axis=0)
        var_5 = np.percentile(paths, 5, axis=0)

        # Compute returns
        final_returns = (paths[:, -1] - paths[:, 0]) / (paths[:, 0] + 1e-8)

        return {
            'mean_path': mean_path,
            'median_path': median_path,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'var_5': var_5,
            'mean_return': np.mean(final_returns) * 100,
            'std_return': np.std(final_returns) * 100,
            'prob_positive': np.mean(final_returns > 0) * 100,
            'var_return': np.percentile(final_returns, 5) * 100
        }


# Page configuration
st.set_page_config(
    page_title="Monte Carlo Simulation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class MonteCarloViz:
    """Visualization components for Monte Carlo simulations"""

    @staticmethod
    def plot_confidence_bands(
        results: MonteCarloResults,
        title: str = "Monte Carlo Forecast with Confidence Intervals"
    ) -> go.Figure:
        """Create confidence band plot"""
        days = np.arange(1, results.horizon + 1)

        fig = go.Figure()

        # Add confidence band (shaded area)
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([results.upper_ci, results.lower_ci[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{results.confidence_level*100:.0f}% Confidence Interval',
            showlegend=True
        ))

        # Add mean path
        fig.add_trace(go.Scatter(
            x=days,
            y=results.mean_path,
            mode='lines',
            name='Mean Forecast',
            line=dict(color='blue', width=3)
        ))

        # Add median path
        fig.add_trace(go.Scatter(
            x=days,
            y=results.median_path,
            mode='lines',
            name='Median Forecast',
            line=dict(color='darkblue', width=2, dash='dash')
        ))

        # Add VaR line
        fig.add_trace(go.Scatter(
            x=days,
            y=results.var_5,
            mode='lines',
            name='5% VaR (Worst Case)',
            line=dict(color='red', width=2, dash='dot')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Days Ahead',
            yaxis_title='Predicted Value',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def plot_path_distribution(
        results: MonteCarloResults,
        n_sample_paths: int = 50
    ) -> go.Figure:
        """Plot sample paths with distribution"""
        days = np.arange(1, results.horizon + 1)

        fig = go.Figure()

        # Sample random paths to display
        n_paths = min(n_sample_paths, results.n_simulations)
        sample_indices = np.random.choice(results.n_simulations, n_paths, replace=False)

        # Add sample paths
        for i, idx in enumerate(sample_indices):
            fig.add_trace(go.Scatter(
                x=days,
                y=results.paths[idx],
                mode='lines',
                line=dict(color='lightgray', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add mean path on top
        fig.add_trace(go.Scatter(
            x=days,
            y=results.mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color='blue', width=3)
        ))

        # Add CI bounds
        fig.add_trace(go.Scatter(
            x=days,
            y=results.upper_ci,
            mode='lines',
            name='95% Upper Bound',
            line=dict(color='green', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=results.lower_ci,
            mode='lines',
            name='95% Lower Bound',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f'Monte Carlo Simulation Paths ({n_paths} samples of {results.n_simulations})',
            xaxis_title='Days Ahead',
            yaxis_title='Predicted Value',
            height=500
        )

        return fig

    @staticmethod
    def plot_final_distribution(results: MonteCarloResults) -> go.Figure:
        """Plot histogram of final values"""
        final_values = results.paths[:, -1]

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            name='Distribution',
            marker_color='steelblue',
            opacity=0.7
        ))

        # Add vertical lines for key statistics
        fig.add_vline(x=results.mean_path[-1], line_dash="solid", line_color="blue",
                      annotation_text="Mean", annotation_position="top right")
        fig.add_vline(x=results.median_path[-1], line_dash="dash", line_color="darkblue",
                      annotation_text="Median", annotation_position="top left")
        fig.add_vline(x=results.var_5[-1], line_dash="dot", line_color="red",
                      annotation_text="5% VaR", annotation_position="bottom right")

        fig.update_layout(
            title=f'Distribution of Final Values (Day {results.horizon})',
            xaxis_title='Predicted Value',
            yaxis_title='Frequency',
            height=400,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_stress_test_comparison(stress_results: Dict[str, MonteCarloResults]) -> go.Figure:
        """Compare stress test scenarios"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Path by Scenario',
                'Final Value Distribution',
                'VaR Comparison',
                'CI Width Over Time'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        colors = {
            'base': 'blue',
            'high_volatility': 'orange',
            'market_crash': 'red',
            'bull_market': 'green',
            'black_swan': 'purple'
        }

        # Plot 1: Mean paths
        for name, results in stress_results.items():
            days = np.arange(1, results.horizon + 1)
            fig.add_trace(
                go.Scatter(x=days, y=results.mean_path, name=name,
                          line=dict(color=colors.get(name, 'gray'))),
                row=1, col=1
            )

        # Plot 2: Final value box plot
        final_data = []
        for name, results in stress_results.items():
            for val in results.paths[:, -1]:
                final_data.append({'Scenario': name, 'Final Value': val})

        final_df = pd.DataFrame(final_data)
        for name in stress_results.keys():
            scenario_data = final_df[final_df['Scenario'] == name]['Final Value']
            fig.add_trace(
                go.Box(y=scenario_data, name=name, marker_color=colors.get(name, 'gray')),
                row=1, col=2
            )

        # Plot 3: VaR comparison
        var_data = {name: results.var_5[-1] for name, results in stress_results.items()}
        fig.add_trace(
            go.Bar(x=list(var_data.keys()), y=list(var_data.values()),
                  marker_color=[colors.get(n, 'gray') for n in var_data.keys()],
                  showlegend=False),
            row=2, col=1
        )

        # Plot 4: CI width over time
        for name, results in stress_results.items():
            days = np.arange(1, results.horizon + 1)
            ci_width = results.upper_ci - results.lower_ci
            fig.add_trace(
                go.Scatter(x=days, y=ci_width, name=name,
                          line=dict(color=colors.get(name, 'gray')),
                          showlegend=False),
                row=2, col=2
            )

        fig.update_layout(height=800, title_text="Stress Test Scenario Comparison")

        return fig

    @staticmethod
    def create_metrics_table(results: MonteCarloResults) -> pd.DataFrame:
        """Create summary metrics table"""
        # Compute returns
        final_returns = (results.paths[:, -1] - results.paths[:, 0]) / (results.paths[:, 0] + 1e-8)

        metrics = {
            'Metric': [
                'Mean Final Value',
                'Median Final Value',
                '95% CI Lower',
                '95% CI Upper',
                'CI Width',
                '5% VaR',
                '5% CVaR (Expected Shortfall)',
                'Mean Return',
                'Return Std Dev',
                'Probability of Positive Return',
                'Min Return',
                'Max Return',
                'Number of Simulations',
                'Forecast Horizon'
            ],
            'Value': [
                f'{results.mean_path[-1]:.4f}',
                f'{results.median_path[-1]:.4f}',
                f'{results.lower_ci[-1]:.4f}',
                f'{results.upper_ci[-1]:.4f}',
                f'{results.upper_ci[-1] - results.lower_ci[-1]:.4f}',
                f'{results.var_5[-1]:.4f}',
                f'{results.cvar_5[-1]:.4f}',
                f'{np.mean(final_returns)*100:.2f}%',
                f'{np.std(final_returns)*100:.2f}%',
                f'{np.mean(final_returns > 0)*100:.1f}%',
                f'{np.min(final_returns)*100:.2f}%',
                f'{np.max(final_returns)*100:.2f}%',
                f'{results.n_simulations:,}',
                f'{results.horizon} days'
            ]
        }

        return pd.DataFrame(metrics)

    @staticmethod
    def create_realtime_plot(
        paths: np.ndarray,
        stats: Dict,
        n_completed: int,
        n_total: int,
        n_display_paths: int = 100
    ) -> go.Figure:
        """Create real-time updating Monte Carlo plot"""
        horizon = paths.shape[1]
        days = np.arange(1, horizon + 1)

        fig = go.Figure()

        # Add sample paths (show up to n_display_paths)
        n_show = min(n_display_paths, len(paths))
        if len(paths) > n_display_paths:
            indices = np.linspace(0, len(paths) - 1, n_display_paths, dtype=int)
        else:
            indices = np.arange(len(paths))

        # Add individual paths with gradient colors based on when they were generated
        for i, idx in enumerate(indices):
            alpha = 0.1 + 0.3 * (i / len(indices))
            fig.add_trace(go.Scatter(
                x=days,
                y=paths[idx],
                mode='lines',
                line=dict(color=f'rgba(100, 149, 237, {alpha})', width=0.8),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([stats['upper_ci'], stats['lower_ci'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))

        # Add mean path
        fig.add_trace(go.Scatter(
            x=days,
            y=stats['mean_path'],
            mode='lines',
            name='Mean Forecast',
            line=dict(color='#2196F3', width=3)
        ))

        # Add median path
        fig.add_trace(go.Scatter(
            x=days,
            y=stats['median_path'],
            mode='lines',
            name='Median Forecast',
            line=dict(color='#1565C0', width=2, dash='dash')
        ))

        # Add VaR line
        fig.add_trace(go.Scatter(
            x=days,
            y=stats['var_5'],
            mode='lines',
            name='5% VaR (Worst Case)',
            line=dict(color='#F44336', width=2, dash='dot')
        ))

        fig.update_layout(
            title=dict(
                text=f'Real-Time Monte Carlo Simulation ({n_completed:,} / {n_total:,} paths)',
                font=dict(size=18)
            ),
            xaxis_title='Days Ahead',
            yaxis_title='Predicted Value',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_realtime_distribution(paths: np.ndarray, stats: Dict) -> go.Figure:
        """Create real-time updating distribution plot"""
        final_values = paths[:, -1]

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=min(50, len(paths) // 5 + 1),
            name='Distribution',
            marker_color='rgba(33, 150, 243, 0.7)',
            opacity=0.8
        ))

        # Add vertical lines for key statistics
        mean_val = stats['mean_path'][-1]
        median_val = stats['median_path'][-1]
        var_val = stats['var_5'][-1]

        fig.add_vline(x=mean_val, line_dash="solid", line_color="#2196F3",
                      annotation_text="Mean", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#1565C0",
                      annotation_text="Median", annotation_position="top left")
        fig.add_vline(x=var_val, line_dash="dot", line_color="#F44336",
                      annotation_text="5% VaR", annotation_position="bottom right")

        fig.update_layout(
            title='Distribution of Final Values (Live)',
            xaxis_title='Predicted Value',
            yaxis_title='Frequency',
            height=350,
            showlegend=False,
            template='plotly_white'
        )

        return fig


def generate_synthetic_model():
    """Generate a simple model for demonstration"""
    import torch.nn as nn

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return SimpleLSTM()


def generate_synthetic_data(n_samples: int = 100, n_features: int = 10):
    """Generate synthetic price data for demo"""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    features = np.zeros((n_samples, n_features))
    features[:, 0] = prices

    for i in range(1, n_features):
        features[:, i] = np.random.randn(n_samples) * 0.1

    return features, prices


def render_realtime_simulation(viz: MonteCarloViz):
    """Render the real-time Monte Carlo simulation tab"""
    if not DEMO_MODE:
        st.info("Real-time simulation uses synthetic models and is disabled because DEMO_MODE is false.")
        return
    st.header("Real-Time Monte Carlo Simulation")
    st.markdown("""
    Watch Monte Carlo paths being generated in real-time. This visualization shows how
    the distribution and statistics evolve as more simulation paths are added.
    """)

    # Configuration
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rt_n_sims = st.number_input("Number of Simulations", min_value=50, max_value=2000,
                                     value=500, step=50, key="rt_n_sims")
    with col2:
        rt_horizon = st.number_input("Forecast Horizon (days)", min_value=5, max_value=60,
                                      value=30, step=5, key="rt_horizon")
    with col3:
        rt_batch_size = st.number_input("Batch Size", min_value=5, max_value=50,
                                         value=10, step=5, key="rt_batch",
                                         help="Paths generated per update")
    with col4:
        rt_volatility = st.number_input("Annual Volatility (%)", min_value=5, max_value=100,
                                         value=20, step=5, key="rt_vol") / 100

    # Animation speed
    speed = st.select_slider(
        "Animation Speed",
        options=["Slow", "Medium", "Fast", "Maximum"],
        value="Fast",
        key="rt_speed"
    )
    delay_map = {"Slow": 0.3, "Medium": 0.15, "Fast": 0.05, "Maximum": 0.0}
    delay = delay_map[speed]

    # Start button
    start_rt_sim = st.button("Start Real-Time Simulation", type="primary", key="start_rt")

    if start_rt_sim:
        # Generate model and data
        model = generate_synthetic_model()
        features, prices = generate_synthetic_data()

        # Create simulator
        rt_simulator = RealTimeMonteCarloSimulator(model=model, seed=int(time.time()) % 10000)

        # Create placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Metrics row
        metric_cols = st.columns(4)
        metric_placeholders = [col.empty() for col in metric_cols]

        # Plot placeholders
        plot_col1, plot_col2 = st.columns([2, 1])
        with plot_col1:
            main_plot = st.empty()
        with plot_col2:
            dist_plot = st.empty()
            stats_table = st.empty()

        # Prepare initial data
        seq_len = min(60, len(features))
        initial_data = features[-seq_len:]

        # Run streaming simulation
        for update in rt_simulator.simulate_paths_streaming(
            initial_data=initial_data,
            horizon=rt_horizon,
            n_simulations=rt_n_sims,
            volatility=rt_volatility,
            batch_size=rt_batch_size
        ):
            # Update progress
            progress_bar.progress(update['progress'])
            status_text.markdown(
                f"**Simulating:** {update['n_completed']:,} / {update['n_total']:,} paths "
                f"({update['progress']*100:.1f}%)"
            )

            # Update metrics
            stats = update['stats']
            metric_placeholders[0].metric(
                "Mean Return",
                f"{stats['mean_return']:.2f}%",
                delta=None
            )
            metric_placeholders[1].metric(
                "Std Dev",
                f"{stats['std_return']:.2f}%"
            )
            metric_placeholders[2].metric(
                "5% VaR",
                f"{stats['var_return']:.2f}%"
            )
            metric_placeholders[3].metric(
                "P(Return > 0)",
                f"{stats['prob_positive']:.1f}%"
            )

            # Update main plot
            main_fig = viz.create_realtime_plot(
                paths=update['paths'],
                stats=stats,
                n_completed=update['n_completed'],
                n_total=update['n_total']
            )
            main_plot.plotly_chart(main_fig, use_container_width=True, key=f"rt_main_{update['n_completed']}")

            # Update distribution plot
            if update['n_completed'] >= 20:
                dist_fig = viz.create_realtime_distribution(update['paths'], stats)
                dist_plot.plotly_chart(dist_fig, use_container_width=True, key=f"rt_dist_{update['n_completed']}")

            # Update stats table
            stats_df = pd.DataFrame({
                'Statistic': ['Mean Final', 'Median Final', '95% CI Lower', '95% CI Upper', 'CI Width'],
                'Value': [
                    f"{stats['mean_path'][-1]:.4f}",
                    f"{stats['median_path'][-1]:.4f}",
                    f"{stats['lower_ci'][-1]:.4f}",
                    f"{stats['upper_ci'][-1]:.4f}",
                    f"{stats['upper_ci'][-1] - stats['lower_ci'][-1]:.4f}"
                ]
            })
            stats_table.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Delay for animation effect
            if delay > 0:
                time.sleep(delay)

        # Simulation complete
        status_text.markdown("**Simulation Complete!**")
        progress_bar.progress(1.0)

        # Store final results
        final_paths = update['paths']
        st.session_state['rt_final_paths'] = final_paths
        st.session_state['rt_final_stats'] = stats

        st.success(f"Completed {rt_n_sims:,} Monte Carlo simulations over {rt_horizon} days")

        # Show convergence analysis
        st.subheader("Convergence Analysis")
        st.markdown("""
        The charts above show how the Monte Carlo statistics converge as more paths are simulated.
        With enough paths, the mean and confidence intervals stabilize, giving reliable estimates
        of prediction uncertainty.
        """)


def main():
    st.title("Monte Carlo Simulation Dashboard")
    st.markdown("""
    **Uncertainty Quantification for Financial Forecasting Models**

    This dashboard visualizes Monte Carlo simulations to understand prediction uncertainty,
    compute risk metrics (VaR, CVaR), and stress test models under various scenarios.
    """)

    if DEMO_MODE:
        st.warning("DEMO_MODE is ON: synthetic models/data will be used when real models or data are missing.")

    # Sidebar configuration
    st.sidebar.header("⚙️ Simulation Settings")

    # Get available trained models from registry
    config = get_config()
    registry = get_model_registry(config.project_root)
    trained_models = registry.get_trained_models()

    # Build model options - add synthetic demo only if allowed
    model_options = []
    model_key_map = {}

    if DEMO_MODE:
        model_options.append("Synthetic Demo")
        model_key_map["Synthetic Demo"] = None

    for key, info in trained_models.items():
        display_name = f"{info.model_name} ({info.architecture})"
        model_options.append(display_name)
        model_key_map[display_name] = key

    if not model_options:
        st.error("No trained models found. Train a model or enable DEMO_MODE to use the synthetic demo.")
        return

    selected_model = st.sidebar.selectbox("Select Model", model_options)
    selected_model_key = model_key_map.get(selected_model)

    # Simulation parameters
    n_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, 100)
    horizon = st.sidebar.slider("Forecast Horizon (days)", 5, 90, 30, 5)
    confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95, 1) / 100

    # Volatility
    volatility_mode = st.sidebar.radio("Volatility", ["Auto-estimate", "Manual"])
    if volatility_mode == "Manual":
        volatility = st.sidebar.slider("Annual Volatility (%)", 5, 100, 20, 5) / 100
    else:
        volatility = None

    # Run button
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

    # Stress test option
    run_stress_test = st.sidebar.checkbox("Include Stress Test", value=False)

    # Main content
    if run_simulation or 'mc_results' not in st.session_state:

        with st.spinner("Running Monte Carlo simulation..."):
            # Load/generate model
            if selected_model_key is None:
                if not DEMO_MODE:
                    st.error("Synthetic demo is disabled. Select a trained model or enable DEMO_MODE.")
                    return
                model = generate_synthetic_model()
                features, prices = generate_synthetic_data()
                st.info("Using synthetic model and data for demonstration.")
            else:
                # Load actual trained model from registry
                try:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = registry.load_model(
                        model_key=selected_model_key,
                        device=device,
                        input_dim=5  # Default feature dimension
                    )

                    if model is not None:
                        if DEMO_MODE:
                            features, prices = generate_synthetic_data()
                            st.info("DEMO_MODE: using synthetic data with trained model.")
                        else:
                            st.error("Real price features are required for Monte Carlo simulation. "
                                     "Generate analysis data or provide a data loader.")
                            return

                        st.success(f"Loaded trained model: {selected_model}")

                        model_info = registry.get_model_info(selected_model_key)
                        if model_info:
                            st.sidebar.info(
                                f"Architecture: {model_info.architecture}\n"
                                f"Trained: {model_info.training_date}\n"
                                f"Epochs: {model_info.epochs_trained or 'N/A'}"
                            )
                    else:
                        st.warning(f"Could not load {selected_model}.")
                        if DEMO_MODE:
                            st.info("Falling back to synthetic demo (DEMO_MODE enabled).")
                            model = generate_synthetic_model()
                            features, prices = generate_synthetic_data()
                        else:
                            return

                except Exception as e:
                    if DEMO_MODE:
                        st.warning(f"Could not load model: {e}. Using synthetic demo (DEMO_MODE).")
                        model = generate_synthetic_model()
                        features, prices = generate_synthetic_data()
                    else:
                        st.error(f"Could not load model: {e}")
                        return

            # Estimate volatility if auto
            if volatility is None:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)
                st.sidebar.info(f"Estimated volatility: {volatility*100:.1f}%")

            # Create simulator
            simulator = MonteCarloSimulator(
                model=model,
                n_simulations=n_simulations,
                seed=42
            )

            # Run simulation
            seq_len = min(60, len(features))
            initial_data = features[-seq_len:]

            results = simulator.simulate_paths(
                initial_data=initial_data,
                horizon=horizon,
                volatility=volatility
            )

            # Store in session state
            st.session_state['mc_results'] = results

            # Run stress test if requested
            if run_stress_test:
                stress_results = simulator.stress_test(
                    initial_data=initial_data,
                    horizon=horizon
                )
                st.session_state['stress_results'] = stress_results

    # Create visualization object
    viz = MonteCarloViz()

    # Main tabs - including real-time simulation
    main_tab1, main_tab2 = st.tabs([
        "Standard Simulation",
        "Real-Time Simulation"
    ])

    with main_tab2:
        render_realtime_simulation(viz)

    with main_tab1:
        # Display results
        if 'mc_results' in st.session_state:
            results = st.session_state['mc_results']

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Confidence Bands",
                "Path Distribution",
                "Final Distribution",
                "Summary Metrics"
            ])

            with tab1:
                st.plotly_chart(
                    viz.plot_confidence_bands(results),
                    use_container_width=True
                )

                st.markdown("""
                **Interpretation:**
                - **Blue Line**: Mean forecast path
                - **Shaded Area**: 95% confidence interval
                - **Red Dotted**: 5% Value at Risk (worst-case boundary)

                The wider the confidence band, the higher the uncertainty in predictions.
                """)

            with tab2:
                st.plotly_chart(
                    viz.plot_path_distribution(results, n_sample_paths=100),
                    use_container_width=True
                )

                st.markdown("""
                **Sample Paths:**
                Gray lines show individual simulated paths. The spread indicates
                the range of possible outcomes. Paths that deviate significantly
                from the mean represent tail risk scenarios.
                """)

            with tab3:
                st.plotly_chart(
                    viz.plot_final_distribution(results),
                    use_container_width=True
                )

                # Return statistics
                final_returns = (results.paths[:, -1] - results.paths[:, 0]) / (results.paths[:, 0] + 1e-8)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Return", f"{np.mean(final_returns)*100:.2f}%")
                with col2:
                    st.metric("Std Dev", f"{np.std(final_returns)*100:.2f}%")
                with col3:
                    st.metric("5% VaR", f"{np.percentile(final_returns, 5)*100:.2f}%")
                with col4:
                    st.metric("Prob(Return>0)", f"{np.mean(final_returns>0)*100:.1f}%")

            with tab4:
                metrics_df = viz.create_metrics_table(results)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                # Download button
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Download Metrics as CSV",
                    data=csv,
                    file_name="monte_carlo_metrics.csv",
                    mime="text/csv"
                )

            # Stress test results
            if 'stress_results' in st.session_state and run_stress_test:
                st.markdown("---")
                st.header("Stress Test Results")

                stress_results = st.session_state['stress_results']
                st.plotly_chart(
                    viz.plot_stress_test_comparison(stress_results),
                    use_container_width=True
                )

                # Stress test summary table
                st.subheader("Scenario Comparison")

                stress_summary = []
                for name, res in stress_results.items():
                    final_returns = (res.paths[:, -1] - res.paths[:, 0]) / (res.paths[:, 0] + 1e-8)
                    stress_summary.append({
                        'Scenario': name,
                        'Mean Final': f'{res.mean_path[-1]:.4f}',
                        'Mean Return': f'{np.mean(final_returns)*100:.2f}%',
                        '5% VaR': f'{res.var_5[-1]:.4f}',
                        '5% CVaR': f'{res.cvar_5[-1]:.4f}',
                        'CI Width': f'{res.upper_ci[-1] - res.lower_ci[-1]:.4f}'
                    })

                stress_df = pd.DataFrame(stress_summary)
                st.dataframe(stress_df, use_container_width=True, hide_index=True)

                st.markdown("""
                **Scenario Descriptions:**
                - **Base**: Normal market conditions
                - **High Volatility**: 2x normal volatility
                - **Market Crash**: 3x volatility + negative drift
                - **Bull Market**: 0.8x volatility + positive drift
                - **Black Swan**: 5x volatility + severe negative drift
                """)

    # Information section
    with st.expander("ℹ️ About Monte Carlo Simulation"):
        st.markdown("""
        ## What is Monte Carlo Simulation?

        Monte Carlo simulation is a computational technique that uses random sampling
        to understand the behavior of complex systems and quantify uncertainty.

        ### Key Metrics:

        - **Value at Risk (VaR)**: The maximum expected loss at a given confidence level.
          5% VaR means there's only a 5% chance of losing more than this amount.

        - **Conditional VaR (CVaR)**: Also called Expected Shortfall. The average loss
          when losses exceed VaR. More sensitive to tail risk than VaR.

        - **Confidence Interval**: Range where we expect the true value to fall with
          specified probability (e.g., 95% CI).

        ### How It Works:

        1. The model makes a prediction
        2. Random noise (based on estimated volatility) is added
        3. The process repeats for each time step
        4. Thousands of paths are simulated
        5. Statistics are computed from all paths

        ### Use Cases:

        - **Risk Assessment**: Understand worst-case scenarios
        - **Uncertainty Quantification**: Know how confident to be in predictions
        - **Stress Testing**: Evaluate performance under extreme conditions
        """)


if __name__ == "__main__":
    main()
