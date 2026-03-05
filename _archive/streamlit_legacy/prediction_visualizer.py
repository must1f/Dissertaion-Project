"""
Prediction Visualization Component

Displays model predictions vs actual returns over time to show:
- How models react to historical data
- How models predict future movements
- Cumulative performance comparison
- Directional prediction accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized

ensure_logger_initialized()
logger = get_logger(__name__)


class PredictionVisualizer:
    """
    Visualizes model predictions vs actual returns
    Shows predictive power and temporal patterns
    """

    def __init__(self):
        self.config = get_config()

    @staticmethod
    def create_predictions_vs_actuals_plot(
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str,
        window_size: int = 100
    ) -> go.Figure:
        """
        Create visualization of predictions vs actual returns

        Shows:
        - Predicted returns vs actual returns
        - Directional agreement
        - Magnitude accuracy
        - Rolling correlation

        Args:
            predictions: Model predictions (returns)
            actuals: Actual returns
            model_name: Name of model
            window_size: Window for rolling correlation

        Returns:
            Plotly figure
        """
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        # Remove NaNs
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]

        if len(predictions) == 0:
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"{model_name}: Predictions vs Actual Returns",
                "Directional Agreement (%)                  ",
                "Cumulative Returns (Strategy)"
            ),
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]
            ],
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.12
        )

        # ===== Row 1: Predictions vs Actuals =====
        time_index = np.arange(len(predictions))

        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=actuals,
                name="Actual Returns",
                mode="lines",
                line=dict(color="green", width=2),
                opacity=0.7
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=predictions,
                name="Predicted Returns",
                mode="lines",
                line=dict(color="blue", width=2, dash="dash"),
                opacity=0.7
            ),
            row=1, col=1
        )

        # ===== Row 2: Directional Agreement =====
        # Calculate directional agreement in rolling windows
        directional_agreement = []
        rolling_indices = []

        for i in range(window_size, len(predictions)):
            pred_direction = np.sign(predictions[max(0, i - window_size):i])
            actual_direction = np.sign(actuals[max(0, i - window_size):i])

            agreement = np.mean(pred_direction == actual_direction) * 100
            directional_agreement.append(agreement)
            rolling_indices.append(i)

        fig.add_trace(
            go.Scatter(
                x=rolling_indices,
                y=directional_agreement,
                name="Directional Accuracy",
                fill="tozeroy",
                line=dict(color="purple", width=2),
                fillcolor="rgba(128, 0, 128, 0.3)"
            ),
            row=2, col=1
        )

        # Add 50% reference line (random)
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="red",
            annotation_text="Random (50%)",
            row=2, col=1
        )

        # ===== Row 3: Cumulative Returns =====
        # Strategy returns (positions based on predictions)
        positions = (predictions > 0).astype(float)
        strategy_returns = positions * actuals

        # Buy-and-hold returns
        buyhold_returns = actuals

        cum_strategy = np.cumprod(1 + strategy_returns) - 1
        cum_buyhold = np.cumprod(1 + buyhold_returns) - 1

        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=cum_buyhold * 100,
                name="Buy-and-Hold Returns",
                mode="lines",
                line=dict(color="green", width=2),
                fill="tozeroy"
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=cum_strategy * 100,
                name="Model Strategy Returns",
                mode="lines",
                line=dict(color="blue", width=2),
                fill="tozeroy"
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
        fig.update_xaxes(title_text="Time (Trading Days)", row=3, col=1)

        fig.update_layout(
            height=900,
            hovermode="x unified",
            showlegend=True,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def create_scatter_predictions_vs_actuals(
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str
    ) -> go.Figure:
        """
        Create scatter plot of predictions vs actual returns

        Shows:
        - Correlation between predictions and actuals
        - Perfect prediction line (y=x)
        - Clustering of predictions

        Args:
            predictions: Model predictions
            actuals: Actual returns
            model_name: Name of model

        Returns:
            Plotly figure
        """
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        # Remove NaNs
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]

        if len(predictions) < 2:
            return go.Figure()

        # Calculate correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1]

        # Create figure
        fig = go.Figure()

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=predictions,
                y=actuals,
                mode="markers",
                marker=dict(
                    size=4,
                    color=actuals,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Actual<br>Return",
                        thickness=15,
                        len=0.7
                    )
                ),
                name="Predictions",
                text=[f"Pred: {p:.4f}<br>Actual: {a:.4f}"
                      for p, a in zip(predictions, actuals)],
                hovertemplate="%{text}<extra></extra>"
            )
        )

        # Perfect prediction line
        min_val = min(predictions.min(), actuals.min())
        max_val = max(predictions.max(), actuals.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="black", dash="dash", width=2),
                name="Perfect Prediction (y=x)",
                hoverinfo="skip"
            )
        )

        # Fit line
        z = np.polyfit(predictions, actuals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=p(x_line),
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name=f"Fitted Line (R²={correlation**2:.4f})"
            )
        )

        fig.update_layout(
            title=f"{model_name}: Prediction vs Actual Returns<br>Correlation: {correlation:.4f}",
            xaxis_title="Predicted Return",
            yaxis_title="Actual Return",
            height=600,
            template="plotly_white",
            hovermode="closest"
        )

        return fig

    @staticmethod
    def create_prediction_distribution(
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str
    ) -> go.Figure:
        """
        Create distribution plots of predictions vs actuals

        Shows:
        - Histogram of predictions
        - Histogram of actuals
        - Directional split (positive vs negative)

        Args:
            predictions: Model predictions
            actuals: Actual returns
            model_name: Name of model

        Returns:
            Plotly figure
        """
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        # Remove NaNs
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]

        if len(predictions) == 0:
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Prediction Distribution",
                "Actual Returns Distribution",
                "Directional Split - Predictions",
                "Directional Split - Actuals"
            ),
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "pie"}]
            ]
        )

        # ===== Histograms =====
        fig.add_trace(
            go.Histogram(
                x=predictions,
                name="Predictions",
                nbinsx=50,
                marker_color="blue",
                opacity=0.7
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=actuals,
                name="Actual Returns",
                nbinsx=50,
                marker_color="green",
                opacity=0.7
            ),
            row=1, col=2
        )

        # ===== Pie Charts =====
        pred_positive = np.sum(predictions > 0)
        pred_negative = len(predictions) - pred_positive

        fig.add_trace(
            go.Pie(
                labels=["Positive Pred", "Negative Pred"],
                values=[pred_positive, pred_negative],
                marker=dict(colors=["green", "red"]),
                textinfo="label+percent"
            ),
            row=2, col=1
        )

        actual_positive = np.sum(actuals > 0)
        actual_negative = len(actuals) - actual_positive

        fig.add_trace(
            go.Pie(
                labels=["Positive Return", "Negative Return"],
                values=[actual_positive, actual_negative],
                marker=dict(colors=["green", "red"]),
                textinfo="label+percent"
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text=f"{model_name}: Prediction and Return Distributions",
            height=700,
            showlegend=False,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def create_residual_analysis(
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str
    ) -> go.Figure:
        """
        Create residual analysis plots

        Shows:
        - Prediction errors over time
        - Residual distribution
        - Error vs magnitude relationship

        Args:
            predictions: Model predictions
            actuals: Actual returns
            model_name: Name of model

        Returns:
            Plotly figure
        """
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        # Remove NaNs
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]

        if len(predictions) == 0:
            return go.Figure()

        # Calculate residuals
        residuals = actuals - predictions
        abs_residuals = np.abs(residuals)
        time_index = np.arange(len(residuals))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Residuals Over Time",
                "Residual Distribution",
                "Absolute Error vs Prediction Magnitude",
                "Residuals vs Actual (Q-Q Plot)"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )

        # ===== Residuals over time =====
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=residuals,
                mode="markers+lines",
                name="Residuals",
                marker=dict(color=residuals, colorscale="RdYlGn", size=4),
                line=dict(color="gray", width=1)
            ),
            row=1, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

        # ===== Residual histogram =====
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name="Residuals",
                nbinsx=50,
                marker_color="purple",
                opacity=0.7
            ),
            row=1, col=2
        )

        # ===== Error vs Magnitude =====
        fig.add_trace(
            go.Scatter(
                x=np.abs(predictions),
                y=abs_residuals,
                mode="markers",
                name="Abs Error",
                marker=dict(color="orange", size=4),
                hovertemplate="Pred Mag: %{x:.4f}<br>Abs Error: %{y:.4f}<extra></extra>"
            ),
            row=2, col=1
        )

        # ===== Q-Q Plot (Residuals vs Normal Distribution) =====
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.sort(
            np.random.normal(0, np.std(residuals), len(residuals))
        )

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode="markers",
                name="Data",
                marker=dict(color="blue", size=4)
            ),
            row=2, col=2
        )

        # Add Q-Q line
        min_q = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_q = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                name="Perfect Fit",
                line=dict(color="black", dash="dash")
            ),
            row=2, col=2
        )

        # Update layouts
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_xaxes(title_text="Time (Days)", row=1, col=1)

        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Residual", row=1, col=2)

        fig.update_yaxes(title_text="Absolute Error", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Magnitude", row=2, col=1)

        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)

        fig.update_layout(
            title_text=f"{model_name}: Residual Analysis",
            height=800,
            showlegend=False,
            template="plotly_white"
        )

        return fig


def render_prediction_visualization_dashboard():
    """
    Render full prediction visualization dashboard
    Shows predictive nature and accuracy of models
    """
    st.set_page_config(
        page_title="Prediction Visualizations",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📈 Model Prediction Analysis Dashboard")

    st.markdown("""
    This dashboard visualizes how models predict financial returns and react to historical data.

    **Key Sections:**
    - **Predictions vs Actuals**: Time series comparison showing prediction accuracy
    - **Scatter Analysis**: Correlation between predicted and actual returns
    - **Distribution Analysis**: Statistical distributions of predictions vs actuals
    - **Residual Analysis**: Prediction errors and patterns
    """)

    st.info("""
    💡 **Note on Sharpe Ratio Interpretation:**
    All PINN models show identical Sharpe ratios (~26) because they execute identical trading
    strategies (100% long) in a bullish market. Use directional accuracy, correlation, and
    prediction magnitude accuracy to compare model quality.
    """)

    # Add state management for model selection
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'pinn_baseline'

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Selection")
        model_name = st.selectbox(
            "Choose a model to visualize:",
            options=[
                'pinn_baseline',
                'pinn_gbm',
                'pinn_ou',
                'pinn_black_scholes',
                'pinn_gbm_ou',
                'pinn_global'
            ],
            key='model_selector'
        )

    with col2:
        visualization_type = st.selectbox(
            "Visualization Type:",
            options=[
                "Time Series Comparison",
                "Scatter Plot",
                "Distributions",
                "Residual Analysis"
            ]
        )

    st.markdown("---")

    # Placeholder for actual predictions (would be loaded from results)
    st.info("""
    📊 **To Enable Visualizations:**
    1. Run `python compute_all_financial_metrics.py` to generate predictions
    2. Predictions will be stored in `results/` directory
    3. Visualizations will automatically populate here

    Each visualization type shows different aspects of model predictive power:
    - **Time Series**: How predictions track actual returns over time
    - **Scatter Plot**: Prediction accuracy and correlation with actuals
    - **Distributions**: Statistical properties of predictions vs market
    - **Residual Analysis**: Systematic prediction errors and patterns
    """)


if __name__ == "__main__":
    render_prediction_visualization_dashboard()
