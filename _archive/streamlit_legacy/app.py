"""
Streamlit web interface for PINN Financial Forecasting

⚠️ DISCLAIMER: This is for academic research only - NOT financial advice!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import torch
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.data.fetcher import DataFetcher
from src.evaluation.backtester import BacktestResults

# Ensure logger is initialized before using it
ensure_logger_initialized()
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="PINN Financial Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .disclaimer {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def show_disclaimer():
    """Display prominent disclaimer"""
    st.markdown("""
    <div class="disclaimer">
        <h3>⚠️ IMPORTANT DISCLAIMER</h3>
        <p><strong>This application is for ACADEMIC RESEARCH ONLY.</strong></p>
        <ul>
            <li>NOT financial advice</li>
            <li>NOT investment recommendations</li>
            <li>Simulation only - no real trading</li>
            <li>Past performance does not guarantee future results</li>
            <li>Always consult a qualified financial advisor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def load_config():
    """Load configuration"""
    return get_config()


def load_results(model_name: str):
    """Load training results for a model"""
    config = get_config()

    # Try multiple result file patterns
    possible_paths = [
        # Direct match (e.g., lstm_results.json, pinn_baseline_results.json)
        config.project_root / 'results' / f'{model_name}_results.json',
        # With pinn_ prefix (e.g., pinn_stacked_results.json for "stacked")
        config.project_root / 'results' / f'pinn_{model_name}_results.json',
        # Rigorous results (e.g., rigorous_pinn_baseline_results.json)
        config.project_root / 'results' / f'rigorous_{model_name}_results.json',
        # In subdirectory
        config.project_root / 'results' / 'pinn_comparison' / f'{model_name}_results.json',
    ]

    for results_path in possible_paths:
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
    return None


def plot_price_history(df: pd.DataFrame, ticker: str):
    """Plot price history with candlestick chart"""
    ticker_df = df[df['ticker'] == ticker].copy()
    ticker_df = ticker_df.sort_values('time')

    fig = go.Figure(data=[go.Candlestick(
        x=ticker_df['time'],
        open=ticker_df['open'],
        high=ticker_df['high'],
        low=ticker_df['low'],
        close=ticker_df['close'],
        name=ticker
    )])

    fig.update_layout(
        title=f'{ticker} Price History',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=400,
        template='plotly_white'
    )

    return fig


def plot_predictions(dates, actual, predicted, ticker: str):
    """Plot actual vs predicted prices"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f'{ticker} - Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def plot_training_history(history):
    """Plot training and validation loss"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss Curves', 'Learning Rate')
    )

    # Loss curves
    fig.add_trace(
        go.Scatter(
            x=history['epochs'],
            y=history['train_loss'],
            mode='lines',
            name='Train Loss',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=history['epochs'],
            y=history['val_loss'],
            mode='lines',
            name='Val Loss',
            line=dict(color='red')
        ),
        row=1, col=1
    )

    # Learning rate
    fig.add_trace(
        go.Scatter(
            x=history['epochs'],
            y=history['learning_rates'],
            mode='lines',
            name='Learning Rate',
            line=dict(color='green')
        ),
        row=1, col=2
    )

    fig.update_layout(height=400, template='plotly_white')
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Learning Rate", row=1, col=2)

    return fig


def plot_portfolio_performance(results: BacktestResults):
    """Plot portfolio value over time"""
    df = results.to_dataframe()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Portfolio Value', 'Returns'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1
    )

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )

    # Returns
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['returns'] * 100,
            name='Returns (%)',
            marker_color=np.where(df['returns'] > 0, 'green', 'red')
        ),
        row=2, col=1
    )

    fig.update_layout(height=600, template='plotly_white', showlegend=False)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1)

    return fig


def main():
    """Main Streamlit app"""

    # Title
    st.title("📈 Physics-Informed Neural Network (PINN)")
    st.subheader("Financial Forecasting & Trading System")

    # Disclaimer
    show_disclaimer()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Batch Training", "Comprehensive Analysis", "Methodology Visualizations", "Training Visualizations", "All Models Dashboard", "PINN Comparison", "Live Metrics", "Model Comparison", "Prediction Visualizations", "Monte Carlo Simulation", "Data Explorer", "Backtesting", "Live Demo"]
    )

    config = load_config()

    # HOME PAGE
    if page == "Home":
        st.header("Welcome to PINN Financial Forecasting")

        st.markdown("""
        This system combines **Physics-Informed Neural Networks (PINNs)** with financial time series forecasting.

        ### Key Features:
        - **8 PINN Variants**: Different physics constraint combinations
        - **Physics Constraints**: GBM (trend), OU (mean-reversion), Black-Scholes (no-arbitrage)
        - **Advanced Architectures**: Stacked PINN, Residual PINN with curriculum learning
        - **Comprehensive Metrics**: Sharpe, Sortino, Calmar, drawdown, profit factor
        - **Robustness Analysis**: Rolling out-of-sample performance, regime sensitivity
        - **Training Visualizations**: Loss curves, learning rate schedules, convergence analysis, overfitting detection

        ### PINN Model Variants:

        **Basic PINN Variants** (Different Physics Constraints):
        1. **Baseline** - Pure data-driven (no physics)
        2. **Pure GBM** - Trend-following dynamics
        3. **Pure OU** - Mean-reversion dynamics
        4. **Pure Black-Scholes** - No-arbitrage constraint
        5. **GBM+OU Hybrid** - Combined trend and mean-reversion
        6. **Global Constraint** - All physics equations combined

        **Advanced PINN Architectures**:
        7. **StackedPINN** - Physics encoder + parallel LSTM/GRU + curriculum learning
        8. **ResidualPINN** - Base model + physics-informed correction

        ### Financial Evaluation Metrics:
        - **Risk-Adjusted**: Sharpe ratio, Sortino ratio
        - **Capital Preservation**: Max drawdown, drawdown duration, Calmar ratio
        - **Trading Viability**: Annualized return, profit factor, transaction-cost-adjusted PnL
        - **Signal Quality**: Directional accuracy, precision/recall, information coefficient
        - **Robustness**: Rolling window stability, regime sensitivity

        ### How It Works:
        1. **Data Collection**: Fetch financial data (stocks, indices)
        2. **Feature Engineering**: Returns, volatility, technical indicators
        3. **PINN Training**: Combine data loss + physics constraints
        4. **Curriculum Learning**: Gradually increase physics weights
        5. **Walk-Forward Validation**: Realistic out-of-sample testing
        6. **Comprehensive Evaluation**: Financial metrics + stability analysis

        ### Data Refresh & Retraining:
        Need up-to-date predictions? Go to **Live Metrics** page and use the
        **"Refresh Data & Retrain"** tab to:
        - Fetch fresh data from yfinance up to today
        - Retrain all models on the same dataset
        - Ensure consistent train/val/test splits across all models

        ### Navigate using the sidebar to explore different features!
        """)

        # Add PINN variant summary
        st.markdown("### Quick Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **When to Use Each Variant:**
            - **Baseline**: No assumptions, maximum flexibility
            - **GBM**: Trending markets (bull/bear)
            - **OU**: Range-bound, mean-reverting markets
            - **Black-Scholes**: Derivative pricing, no-arbitrage
            """)

        with col2:
            st.markdown("""
            **Advanced Models:**
            - **GBM+OU**: Balanced, general forecasting
            - **Global**: Maximum physics regularization
            - **StackedPINN**: Multi-scale features + parallel processing
            - **ResidualPINN**: Physics-constrained corrections
            """)

        # Show quick stats
        st.subheader("Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tickers", len(config.data.tickers))
            st.metric("Sequence Length", config.data.sequence_length)

        with col2:
            st.metric("Training Epochs", config.training.epochs)
            st.metric("Batch Size", config.training.batch_size)

        with col3:
            st.metric("Initial Capital", f"${config.trading.initial_capital:,.0f}")
            st.metric("Max Position Size", f"{config.trading.max_position_size*100:.0f}%")

    # BATCH TRAINING PAGE
    elif page == "Batch Training":
        st.header("Train All Models")
        st.markdown("### Configure and train multiple models with real-time progress visualization")

        try:
            from src.web.batch_training_dashboard import render_batch_training_dashboard
            render_batch_training_dashboard()

        except Exception as e:
            st.error(f"Error loading Batch Training Dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

            st.info("""
            **Troubleshooting:**
            - Ensure all dependencies are installed
            - Check that the data directory exists
            - Verify database connection (if required)
            """)

    # COMPREHENSIVE ANALYSIS DASHBOARD
    elif page == "Comprehensive Analysis":
        st.header("📊 Comprehensive PINN Analysis Dashboard")

        with st.expander("ℹ️ About This Dashboard", expanded=False):
            st.markdown("""
            This dashboard provides comprehensive visualizations across **6 key categories**:

            | Category | What It Tells You |
            |----------|------------------|
            | **1. Predictive Performance** | Loss curves, predictions, residuals - Is the model learning? |
            | **2. Economic Effectiveness** | Returns, Sharpe, drawdowns - Does it make money? |
            | **3. Risk Diagnostics** | VaR, CVaR, stress tests - Does it survive? |
            | **4. Comparative Evaluation** | Model comparisons, statistical tests - Which is best? |
            | **5. Explainability** | Physics constraints, feature importance - Does it make sense? |
            | **6. Regime Robustness** | Market regimes, correlations - Is it robust? |

            **Use the sidebar dropdown to select different analysis categories.**
            """)

        try:
            from src.web.comprehensive_analysis_dashboard import render_comprehensive_analysis
            # Run the comprehensive dashboard
            render_comprehensive_analysis()

        except Exception as e:
            st.error(f"Error loading Comprehensive Analysis Dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

            st.info("""
            **Troubleshooting:**
            - Ensure model results exist in `results/` directory
            - Run `python compute_all_financial_metrics.py` to generate metrics
            - Check that training histories exist in `Models/` directory
            """)

    # METHODOLOGY VISUALIZATIONS
    elif page == "Methodology Visualizations":
        st.header("Research Methodology Visualizations")
        st.markdown("### Key visualizations demonstrating PINN methodology and evaluation approach")

        try:
            from src.web.methodology_dashboard import render_methodology_section
            render_methodology_section()

        except Exception as e:
            st.error(f"Error loading Methodology Dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

    # TRAINING VISUALIZATIONS
    elif page == "Training Visualizations":
        st.header("Training Visualizations")
        st.markdown("### Comprehensive analysis of model training progress")

        try:
            from src.web.training_dashboard import render_training_visualizations
            render_training_visualizations()

        except Exception as e:
            st.error(f"Error loading Training Dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ALL MODELS DASHBOARD
    elif page == "All Models Dashboard":
        st.header("🤖 All Neural Network Models")
        st.markdown("### Complete Model Registry with Training Status")

        try:
            from src.web.all_models_dashboard import AllModelsDashboard

            dashboard = AllModelsDashboard()

            # Sub-navigation
            section = st.radio(
                "Select Section",
                ["Overview", "Model List", "Metrics Comparison"],
                horizontal=True
            )

            if section == "Overview":
                dashboard.render_model_overview()

            elif section == "Model List":
                dashboard.render_model_list()

            elif section == "Metrics Comparison":
                dashboard.render_metrics_comparison()

        except Exception as e:
            st.error(f"Error loading All Models Dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

    # PINN COMPARISON
    elif page == "PINN Comparison":
        st.header("PINN Model Comparison")
        st.markdown("### Comprehensive Financial Performance Analysis")

        # Load PINN dashboard
        try:
            from src.web.pinn_dashboard import PINNDashboard, PINN_VARIANTS

            dashboard = PINNDashboard()
            all_results = dashboard.load_all_results()

            if not all_results:
                st.warning("No PINN results found. Please train models first.")
                st.markdown("""
                ### Quick Start:

                **Train all 6 PINN physics variants:**
                ```bash
                python src/training/train_pinn_variants.py --epochs 100
                ```

                **Train Stacked PINN (encoder + parallel LSTM/GRU):**
                ```bash
                python src/training/train_stacked_pinn.py --model-type stacked --epochs 100
                ```

                **Train Residual PINN (base + physics correction):**
                ```bash
                python src/training/train_stacked_pinn.py --model-type residual --epochs 100
                ```
                """)
            else:
                st.success(f"✓ Loaded {len(all_results)} PINN models")

                # Display model list
                st.markdown("### Available Models:")
                cols = st.columns(4)
                for i, (key, name) in enumerate(PINN_VARIANTS.items()):
                    col_idx = i % 4
                    with cols[col_idx]:
                        if key in all_results:
                            st.success(f"✓ {name}")
                        else:
                            st.info(f"○ {name}")

                # Tabs for different views
                tab1, tab2, tab3 = st.tabs([
                    "📊 Metrics Comparison",
                    "📈 Rolling Performance",
                    "📚 Training History"
                ])

                with tab1:
                    df = dashboard.render_metrics_comparison(all_results)

                with tab2:
                    dashboard.render_rolling_performance(all_results)

                with tab3:
                    dashboard.render_training_history(all_results)

        except Exception as e:
            st.error(f"Error loading PINN dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

    # LIVE METRICS COMPUTATION
    elif page == "Live Metrics":
        st.header("🧮 Live Metrics Computation")

        st.markdown("""
        Compute comprehensive financial metrics on-demand from model predictions.
        This allows you to analyze model performance without running separate scripts.

        **Features:**
        - Compute metrics from saved predictions
        - Upload custom prediction data
        - **Refresh data from yfinance and retrain models**
        """)

        # Quick info about last data refresh
        config = get_config()
        summary_path = config.project_root / 'results' / 'retraining_summary.json'
        if summary_path.exists():
            try:
                import json
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                last_refresh = summary.get('timestamp', 'Unknown')
                st.success(f"Last data refresh: {last_refresh}")
            except Exception:
                pass

        try:
            from src.web.metrics_calculator import StreamlitMetricsCalculator

            calculator = StreamlitMetricsCalculator()
            calculator.render_computation_panel()

        except Exception as e:
            st.error(f"Error loading metrics calculator: {e}")
            import traceback
            st.code(traceback.format_exc())

    # DATA EXPLORER
    elif page == "Data Explorer":
        st.header("Data Explorer")

        st.info("Loading sample data from database...")

        try:
            from src.utils.database import get_db
            db = get_db()

            # Get available tickers
            query = "SELECT DISTINCT ticker FROM finance.stock_prices ORDER BY ticker LIMIT 20"
            result = db.execute_query(query)
            tickers = [r['ticker'] for r in result]

            if not tickers:
                st.warning("No data in database. Please run data fetching first.")
                return

            # Ticker selection
            selected_ticker = st.selectbox("Select Ticker", tickers)

            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
            with col2:
                end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

            # Fetch data
            if st.button("Load Data"):
                with st.spinner("Loading data..."):
                    df = db.get_stock_prices(
                        tickers=[selected_ticker],
                        start_date=str(start_date),
                        end_date=str(end_date)
                    )

                    if not df.empty:
                        st.success(f"Loaded {len(df)} records")

                        # Show statistics
                        st.subheader("Statistics")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Start Price", f"${df.iloc[0]['close']:.2f}")
                        with col2:
                            st.metric("End Price", f"${df.iloc[-1]['close']:.2f}")
                        with col3:
                            change = ((df.iloc[-1]['close'] / df.iloc[0]['close']) - 1) * 100
                            st.metric("Total Change", f"{change:.2f}%")
                        with col4:
                            st.metric("Avg Volume", f"{df['volume'].mean():,.0f}")

                        # Plot
                        st.plotly_chart(plot_price_history(df, selected_ticker), use_container_width=True)

                        # Show data table
                        with st.expander("View Raw Data"):
                            st.dataframe(df)

                    else:
                        st.warning("No data found for selected parameters")

        except Exception as e:
            st.error(f"Error loading data: {e}")

    # MODEL COMPARISON
    elif page == "Model Comparison":
        st.header("Model Comparison")

        st.info("Compare all architectures: Baseline, PINN variants, and advanced models")

        # Load results for all models - including all PINN types
        all_model_types = [
            'lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer',
            'baseline', 'gbm', 'ou', 'black_scholes', 'gbm_ou', 'global',
            'stacked', 'residual'
        ]

        results = {}

        for model in all_model_types:
            result = load_results(model)
            if result:
                results[model] = result

        if not results:
            st.warning("No model results found. Please train models first.")
            st.markdown("""
            ### Training Instructions:
            ```bash
            # Train PINN variants
            python src/training/train_pinn_variants.py --epochs 100

            # Train Stacked/Residual PINN
            python src/training/train_stacked_pinn.py --model-type stacked
            ```
            """)
            return

        st.success(f"✓ Loaded {len(results)} models")

        # Show comprehensive comparison table
        st.subheader("📊 Comprehensive Performance Comparison")

        comparison_data = []
        for model, result in results.items():
            # Get metrics from different possible locations
            if 'financial_metrics' in result:
                metrics = result['financial_metrics']
            elif 'test_metrics' in result:
                metrics = result['test_metrics']
            else:
                metrics = {}

            # Get ML metrics separately (MSE, RMSE, MAE, R², MAPE)
            ml_metrics = result.get('ml_metrics', {})

            row = {
                'Model': model.upper(),

                # Traditional ML Metrics - check ml_metrics first
                'MSE': ml_metrics.get('mse', metrics.get('mse', metrics.get('test_mse', np.nan))),
                'RMSE': ml_metrics.get('rmse', metrics.get('rmse', metrics.get('test_rmse', np.nan))),
                'MAE': ml_metrics.get('mae', metrics.get('mae', metrics.get('test_mae', np.nan))),
                'R²': ml_metrics.get('r2', metrics.get('r2', metrics.get('test_r2', np.nan))),
                'MAPE': ml_metrics.get('mape', metrics.get('mape', np.nan)),

                # Financial Metrics
                'Sharpe': metrics.get('sharpe_ratio', np.nan),
                'Sortino': metrics.get('sortino_ratio', np.nan),
                'Max DD %': metrics.get('max_drawdown', 0) * 100,
                'Calmar': metrics.get('calmar_ratio', np.nan),
                'Annual Ret %': metrics.get('annualized_return', metrics.get('total_return', 0)) * 100,
                'Dir Acc %': metrics.get('directional_accuracy', metrics.get('test_directional_accuracy', 0)) * 100,
                'Win Rate %': metrics.get('win_rate', 0) * 100,
                'Profit Factor': metrics.get('profit_factor', np.nan),
                'IC': metrics.get('information_coefficient', np.nan)
            }

            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)

        # Create tabs for different metric views
        tab1, tab2 = st.tabs(["Traditional ML Metrics", "Financial Metrics"])

        with tab1:
            ml_cols = ['Model', 'MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Dir Acc %']
            ml_df = df_comparison[ml_cols].copy()

            styled_df = ml_df.style.highlight_min(
                subset=['MSE', 'RMSE', 'MAE', 'MAPE'],
                color='lightgreen'
            ).highlight_max(
                subset=['R²', 'Dir Acc %'],
                color='lightgreen'
            ).format({
                'MSE': '{:.6f}',
                'RMSE': '{:.6f}',
                'MAE': '{:.6f}',
                'R²': '{:.4f}',
                'MAPE': '{:.2f}%',
                'Dir Acc %': '{:.2f}%'
            })

            st.dataframe(styled_df, use_container_width=True)

        with tab2:
            fin_cols = ['Model', 'Sharpe', 'Sortino', 'Max DD %', 'Calmar',
                       'Annual Ret %', 'Win Rate %', 'Profit Factor', 'IC']
            fin_df = df_comparison[fin_cols].copy()

            styled_df = fin_df.style.highlight_max(
                subset=['Sharpe', 'Sortino', 'Calmar', 'Annual Ret %',
                       'Win Rate %', 'Profit Factor', 'IC'],
                color='lightgreen'
            ).highlight_max(  # Less negative drawdown is better
                subset=['Max DD %'],
                color='lightgreen'
            ).format({
                'Sharpe': '{:.3f}',
                'Sortino': '{:.3f}',
                'Max DD %': '{:.2f}%',
                'Calmar': '{:.3f}',
                'Annual Ret %': '{:.2f}%',
                'Win Rate %': '{:.2f}%',
                'Profit Factor': '{:.2f}',
                'IC': '{:.3f}'
            })

            st.dataframe(styled_df, use_container_width=True)

        # Visualizations
        st.subheader("📊 Performance Visualizations")

        viz_cols = st.columns(2)

        with viz_cols[0]:
            # Sharpe ratio comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_comparison['Model'],
                y=df_comparison['Sharpe'],
                marker_color='steelblue'
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                         annotation_text="Sharpe = 1.0 (Good)")
            fig.update_layout(
                title='Sharpe Ratio Comparison',
                xaxis_title='Model',
                yaxis_title='Sharpe Ratio',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with viz_cols[1]:
            # Directional accuracy
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_comparison['Model'],
                y=df_comparison['Dir Acc %'],
                marker_color='forestgreen'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="red",
                         annotation_text="50% Random Baseline")
            fig.update_layout(
                title='Directional Accuracy',
                xaxis_title='Model',
                yaxis_title='Accuracy (%)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Plot training history for selected model
        st.subheader("Training History")
        selected_model = st.selectbox("Select model to view training history", list(results.keys()))

        if 'training_history' in results[selected_model]:
            history = results[selected_model]['training_history']
            st.plotly_chart(plot_training_history(history), use_container_width=True)
        elif 'history' in results[selected_model]:
            history = results[selected_model]['history']
            st.plotly_chart(plot_training_history(history), use_container_width=True)
        else:
            st.info("Training history not available for this model")

    # PREDICTION VISUALIZATIONS
    elif page == "Prediction Visualizations":
        st.header("📊 Model Prediction Analysis")

        st.markdown("""
        This dashboard visualizes how models predict financial returns and react to historical data.

        **Key Visualizations:**
        - **Predictions vs Actuals**: Time series showing prediction accuracy and strategy performance
        - **Scatter Analysis**: Correlation between predicted and actual returns
        - **Distribution Analysis**: Statistical properties of predictions vs market returns
        - **Residual Analysis**: Prediction errors and systematic patterns
        """)

        st.info("""
        💡 **Note on Identical Sharpe Ratios:**
        All PINN models show identical Sharpe ratios (~26) because they execute identical trading
        strategies (100% long) in a bullish market. This dashboard shows metrics that actually
        differentiate model quality: directional accuracy, correlation, and prediction magnitude.

        📖 See SHARPE_RATIO_INVESTIGATION.md for detailed explanation.
        """)

        try:
            # Import prediction visualizer
            from src.web.prediction_visualizer import PredictionVisualizer

            # Model selection
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Select Model and Visualization Type")
                model_name = st.selectbox(
                    "Choose a model:",
                    options=[
                        'lstm',
                        'gru',
                        'bilstm',
                        'transformer',
                        'pinn_baseline',
                        'pinn_gbm',
                        'pinn_ou',
                        'pinn_black_scholes',
                        'pinn_gbm_ou',
                        'pinn_global'
                    ],
                    key='pred_model_selector'
                )

            with col2:
                visualization_type = st.selectbox(
                    "Visualization:",
                    options=[
                        "Time Series",
                        "Scatter Plot",
                        "Distributions",
                        "Residual Analysis"
                    ],
                    key='pred_viz_selector'
                )

            st.markdown("---")

            # Try to load predictions
            predictions_path = config.project_root / 'results' / f'{model_name}_predictions.npz'

            if predictions_path.exists():
                # Load predictions and targets
                data = np.load(predictions_path)
                predictions = data['predictions']
                targets = data['targets']

                st.success(f"✓ Loaded {len(predictions)} predictions for {model_name}")

                # Create visualization based on selection
                if visualization_type == "Time Series":
                    fig = PredictionVisualizer.create_predictions_vs_actuals_plot(
                        predictions, targets, model_name.replace('_', ' ').title()
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif visualization_type == "Scatter Plot":
                    fig = PredictionVisualizer.create_scatter_predictions_vs_actuals(
                        predictions, targets, model_name.replace('_', ' ').title()
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif visualization_type == "Distributions":
                    fig = PredictionVisualizer.create_prediction_distribution(
                        predictions, targets, model_name.replace('_', ' ').title()
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif visualization_type == "Residual Analysis":
                    fig = PredictionVisualizer.create_residual_analysis(
                        predictions, targets, model_name.replace('_', ' ').title()
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Display info about how to enable visualizations
                st.info("""
                📊 **To Enable Visualizations:**
                1. Run: `python compute_all_financial_metrics.py`
                2. This generates predictions and stores them in `results/` directory
                3. Visualizations will automatically populate

                **Each visualization shows:**
                - **Time Series**: How predictions track actual returns over time
                - **Scatter Plot**: Prediction accuracy and correlation strength
                - **Distributions**: Statistical properties of predictions vs market
                - **Residual Analysis**: Systematic prediction errors and patterns
                """)

                st.warning("""
                ⏳ Waiting for prediction data...
                Run compute_all_financial_metrics.py to generate visualizations
                """)

        except ImportError:
            st.error("Prediction visualizer module not found")
        except Exception as e:
            st.error(f"Error loading prediction visualizations: {e}")
            import traceback
            st.code(traceback.format_exc())

    # MONTE CARLO SIMULATION
    elif page == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        st.markdown("""
        Simulate thousands of possible future price paths using **Geometric Brownian Motion (GBM)**
        to understand the range of potential outcomes and associated risks.
        """)

        show_disclaimer()

        # Model selection for Monte Carlo
        st.subheader("Model Selection")

        # Available models for Monte Carlo simulation
        available_models = [
            'manual',  # Manual parameter input
            'lstm', 'gru', 'bilstm', 'attention_lstm', 'transformer',
            'pinn_baseline', 'pinn_gbm', 'pinn_ou', 'pinn_black_scholes',
            'pinn_gbm_ou', 'pinn_global', 'stacked', 'residual'
        ]

        model_descriptions = {
            'manual': 'Manual Parameters - Enter your own drift and volatility',
            'lstm': 'LSTM - Long Short-Term Memory Network',
            'gru': 'GRU - Gated Recurrent Unit',
            'bilstm': 'BiLSTM - Bidirectional LSTM',
            'attention_lstm': 'Attention LSTM - LSTM with Attention Mechanism',
            'transformer': 'Transformer - Attention-based Architecture',
            'pinn_baseline': 'PINN Baseline - Pure data-driven (no physics)',
            'pinn_gbm': 'PINN GBM - Geometric Brownian Motion constraint',
            'pinn_ou': 'PINN OU - Ornstein-Uhlenbeck mean-reversion',
            'pinn_black_scholes': 'PINN Black-Scholes - No-arbitrage constraint',
            'pinn_gbm_ou': 'PINN GBM+OU - Combined trend and mean-reversion',
            'pinn_global': 'PINN Global - All physics constraints combined',
            'stacked': 'Stacked PINN - Physics encoder + parallel LSTM/GRU',
            'residual': 'Residual PINN - Base model + physics correction'
        }

        selected_model = st.selectbox(
            "Select Model for Monte Carlo Simulation",
            options=available_models,
            format_func=lambda x: model_descriptions.get(x, x),
            help="Choose a trained model to extract drift/volatility parameters, or use manual input"
        )

        # Check if model results exist and load parameters
        model_params_loaded = False
        loaded_drift = 0.10  # Default 10%
        loaded_volatility = 0.20  # Default 20%

        def is_valid_number(val, min_val=None, max_val=None):
            """Check if value is a valid finite number within optional bounds"""
            if val is None:
                return False
            try:
                if np.isnan(val) or np.isinf(val):
                    return False
                if min_val is not None and val < min_val:
                    return False
                if max_val is not None and val > max_val:
                    return False
                return True
            except (TypeError, ValueError):
                return False

        if selected_model != 'manual':
            # Try to load model results
            model_result = load_results(selected_model)

            if model_result:
                st.success(f"✓ Loaded {selected_model} model results")

                # Try to extract drift and volatility from model results
                if 'financial_metrics' in model_result:
                    metrics = model_result['financial_metrics']

                    # Try multiple fields for drift (annualized return)
                    drift_candidates = [
                        metrics.get('annualized_return'),
                        metrics.get('total_return'),
                        metrics.get('cumulative_return_final'),
                    ]

                    for drift_val in drift_candidates:
                        if is_valid_number(drift_val, min_val=-1.0, max_val=10.0):
                            # Value is already in decimal form (e.g., 0.10 = 10%)
                            loaded_drift = float(drift_val)
                            model_params_loaded = True
                            break

                    # Try multiple fields for volatility
                    vol_candidates = [
                        metrics.get('volatility'),
                        metrics.get('annualized_volatility'),
                    ]

                    for vol_val in vol_candidates:
                        if is_valid_number(vol_val, min_val=0.001):
                            vol_float = float(vol_val)
                            # If volatility > 1, it might be in percentage form or daily
                            # Convert to reasonable annual volatility (0.05 to 1.0 range)
                            if vol_float > 1.0:
                                # Assume it's daily volatility, annualize it
                                # Annual vol = daily vol * sqrt(252)
                                if vol_float < 10:
                                    loaded_volatility = min(vol_float / np.sqrt(252), 1.0)
                                else:
                                    # Too high, use default
                                    loaded_volatility = 0.20
                            else:
                                loaded_volatility = vol_float
                            break

                # Final validation - ensure we have valid parameters
                if not is_valid_number(loaded_drift, min_val=-1.0, max_val=10.0):
                    loaded_drift = 0.10
                    st.warning("Model drift parameter invalid, using default 10%")

                if not is_valid_number(loaded_volatility, min_val=0.01, max_val=1.0):
                    loaded_volatility = 0.20
                    st.warning("Model volatility parameter invalid, using default 20%")

                # Display loaded parameters
                if model_params_loaded:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Drift (μ)", f"{loaded_drift*100:.2f}%")
                    with col2:
                        st.metric("Model Volatility (σ)", f"{loaded_volatility*100:.2f}%")
                else:
                    st.info("Could not extract valid parameters from model. Using defaults.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Default Drift (μ)", f"{loaded_drift*100:.2f}%")
                    with col2:
                        st.metric("Default Volatility (σ)", f"{loaded_volatility*100:.2f}%")
            else:
                st.warning(f"No results found for {selected_model}. Using default parameters.")
                st.info("Train this model first to use its learned parameters.")

        st.markdown("---")

        # Simulation parameters
        st.sidebar.markdown("### Simulation Parameters")

        col1, col2 = st.columns(2)

        with col1:
            initial_price = st.number_input(
                "Initial Price ($)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=1.0,
                help="Starting price for the simulation"
            )

            # Use model parameters if loaded, otherwise allow manual input
            if selected_model != 'manual' and model_params_loaded:
                st.info(f"Using {selected_model} model parameters")

                # Ensure we have safe values for display and sliders
                safe_drift = loaded_drift if is_valid_number(loaded_drift, -1.0, 10.0) else 0.10
                safe_volatility = loaded_volatility if is_valid_number(loaded_volatility, 0.01, 1.0) else 0.20

                expected_return = safe_drift
                volatility = safe_volatility

                # Show as display only
                st.text(f"Expected Annual Return: {safe_drift*100:.2f}%")
                st.text(f"Annual Volatility: {safe_volatility*100:.2f}%")

                # Allow override option
                override_params = st.checkbox("Override model parameters", value=False)
                if override_params:
                    # Safe slider default values
                    drift_slider_val = float(max(-50.0, min(100.0, safe_drift * 100)))
                    vol_slider_val = float(max(1.0, min(100.0, safe_volatility * 100)))

                    expected_return = st.slider(
                        "Expected Annual Return (%)",
                        min_value=-50.0,
                        max_value=100.0,
                        value=drift_slider_val,
                        step=1.0,
                        help="Drift parameter (mu) - expected return"
                    ) / 100

                    volatility = st.slider(
                        "Annual Volatility (%)",
                        min_value=1.0,
                        max_value=100.0,
                        value=vol_slider_val,
                        step=1.0,
                        help="Sigma parameter - standard deviation of returns"
                    ) / 100
            else:
                expected_return = st.slider(
                    "Expected Annual Return (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="Drift parameter (mu) - expected return"
                ) / 100

                volatility = st.slider(
                    "Annual Volatility (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    help="Sigma parameter - standard deviation of returns"
                ) / 100

        with col2:
            time_horizon = st.selectbox(
                "Time Horizon",
                options=[30, 60, 90, 180, 252, 504],
                index=3,
                format_func=lambda x: f"{x} days ({x/252:.1f} years)" if x >= 252 else f"{x} days ({x/30:.1f} months)"
            )

            num_simulations = st.selectbox(
                "Number of Simulations",
                options=[100, 500, 1000, 5000, 10000],
                index=2,
                help="More simulations = more accurate distribution"
            )

            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=90,
                max_value=99,
                value=95,
                help="Confidence level for VaR and prediction intervals"
            )

        # Run simulation
        model_label = model_descriptions.get(selected_model, selected_model) if selected_model != 'manual' else 'Manual Parameters'
        if st.button(f"Run Monte Carlo Simulation ({selected_model.upper()})", type="primary"):
            # Final parameter validation before simulation
            sim_drift = expected_return
            sim_volatility = volatility

            # Validate and sanitize parameters
            if not is_valid_number(sim_drift, min_val=-1.0, max_val=10.0):
                sim_drift = 0.10
                st.warning(f"Invalid drift value detected, using default: {sim_drift*100:.1f}%")

            if not is_valid_number(sim_volatility, min_val=0.01, max_val=1.0):
                sim_volatility = 0.20
                st.warning(f"Invalid volatility value detected, using default: {sim_volatility*100:.1f}%")

            if not is_valid_number(initial_price, min_val=0.01):
                initial_price = 100.0
                st.warning(f"Invalid initial price, using default: ${initial_price:.2f}")

            # Display parameters being used
            st.markdown("**Simulation Parameters:**")
            param_cols = st.columns(4)
            with param_cols[0]:
                st.caption(f"Drift (μ): {sim_drift*100:.2f}%")
            with param_cols[1]:
                st.caption(f"Volatility (σ): {sim_volatility*100:.2f}%")
            with param_cols[2]:
                st.caption(f"Initial Price: ${initial_price:.2f}")
            with param_cols[3]:
                st.caption(f"Time Horizon: {time_horizon} days")

            with st.spinner(f"Running {num_simulations} simulations using {model_label}..."):
                # Set random seed for reproducibility
                np.random.seed(42)

                # Time parameters
                dt = 1/252  # Daily time step
                N = time_horizon  # Number of steps

                # Generate random paths using GBM
                # dS = S * (mu*dt + sigma*dW)
                # S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))

                # Generate random shocks
                Z = np.random.standard_normal((num_simulations, N))

                # Calculate cumulative returns using validated parameters
                drift = (sim_drift - 0.5 * sim_volatility**2) * dt
                diffusion = sim_volatility * np.sqrt(dt) * Z

                # Cumulative sum for the path
                log_returns = drift + diffusion
                cumulative_returns = np.cumsum(log_returns, axis=1)

                # Price paths
                price_paths = initial_price * np.exp(cumulative_returns)

                # Add initial price
                price_paths = np.column_stack([np.full(num_simulations, initial_price), price_paths])

                # Verify simulation produced valid results
                if np.any(np.isnan(price_paths)) or np.any(np.isinf(price_paths)):
                    st.error("Simulation produced invalid values (NaN or Inf). Check parameters.")
                    st.stop()

                # Calculate statistics
                final_prices = price_paths[:, -1]

                # Filter out any invalid values
                valid_final_prices = final_prices[np.isfinite(final_prices)]
                if len(valid_final_prices) == 0:
                    st.error("All simulation paths produced invalid results.")
                    st.stop()

                mean_final = np.mean(valid_final_prices)
                median_final = np.median(valid_final_prices)
                std_final = np.std(valid_final_prices)
                min_final = np.min(valid_final_prices)
                max_final = np.max(valid_final_prices)

                # VaR and Expected Shortfall
                alpha = (100 - confidence_level) / 100
                var_price = np.percentile(valid_final_prices, alpha * 100)
                var_return = (var_price - initial_price) / initial_price * 100
                es_prices = valid_final_prices[valid_final_prices <= var_price]
                expected_shortfall = np.mean(es_prices) if len(es_prices) > 0 else var_price
                es_return = (expected_shortfall - initial_price) / initial_price * 100

                # Probability of profit/loss
                prob_profit = np.mean(valid_final_prices > initial_price) * 100
                prob_loss = 100 - prob_profit

                # Percentiles
                p5 = np.percentile(valid_final_prices, 5)
                p25 = np.percentile(valid_final_prices, 25)
                p75 = np.percentile(valid_final_prices, 75)
                p95 = np.percentile(valid_final_prices, 95)

            # Display results
            st.success(f"Simulation complete! {num_simulations} paths generated using **{model_label}**.")

            # Summary metrics
            st.subheader("Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                change = (mean_final - initial_price) / initial_price * 100
                st.metric("Mean Final Price", f"${mean_final:.2f}", f"{change:+.2f}%")
                st.metric("Median Final Price", f"${median_final:.2f}")

            with col2:
                st.metric("Std Deviation", f"${std_final:.2f}")
                st.metric("Range", f"${min_final:.2f} - ${max_final:.2f}")

            with col3:
                st.metric(f"VaR ({confidence_level}%)", f"${var_price:.2f}", f"{var_return:.2f}%", delta_color="inverse")
                st.metric(f"Expected Shortfall", f"${expected_shortfall:.2f}", f"{es_return:.2f}%", delta_color="inverse")

            with col4:
                st.metric("Prob. of Profit", f"{prob_profit:.1f}%")
                st.metric("Prob. of Loss", f"{prob_loss:.1f}%")

            # Visualizations
            st.subheader("Simulation Visualizations")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Price Paths",
                "Final Price Distribution",
                "Confidence Intervals",
                "Risk Analysis"
            ])

            with tab1:
                # Plot sample paths
                fig = go.Figure()

                # Plot a sample of paths (max 100 for visibility)
                sample_size = min(100, num_simulations)
                sample_indices = np.random.choice(num_simulations, sample_size, replace=False)

                days = np.arange(time_horizon + 1)

                for i in sample_indices:
                    fig.add_trace(go.Scatter(
                        x=days,
                        y=price_paths[i],
                        mode='lines',
                        line=dict(width=0.5, color='rgba(100, 149, 237, 0.3)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Add mean path
                mean_path = np.mean(price_paths, axis=0)
                fig.add_trace(go.Scatter(
                    x=days,
                    y=mean_path,
                    mode='lines',
                    name='Mean Path',
                    line=dict(width=3, color='red')
                ))

                # Add percentile bands
                p5_path = np.percentile(price_paths, 5, axis=0)
                p95_path = np.percentile(price_paths, 95, axis=0)

                fig.add_trace(go.Scatter(
                    x=days,
                    y=p95_path,
                    mode='lines',
                    name='95th Percentile',
                    line=dict(width=2, color='green', dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=days,
                    y=p5_path,
                    mode='lines',
                    name='5th Percentile',
                    line=dict(width=2, color='orange', dash='dash')
                ))

                fig.update_layout(
                    title=f'Monte Carlo Simulation ({selected_model.upper()}): {num_simulations} Price Paths',
                    xaxis_title='Days',
                    yaxis_title='Price ($)',
                    height=500,
                    template='plotly_white',
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Final price distribution
                fig = make_subplots(rows=1, cols=2, subplot_titles=(
                    'Histogram of Final Prices',
                    'Cumulative Distribution'
                ))

                # Histogram - use valid_final_prices
                fig.add_trace(
                    go.Histogram(
                        x=valid_final_prices,
                        nbinsx=50,
                        name='Final Prices',
                        marker_color='steelblue',
                        opacity=0.7
                    ),
                    row=1, col=1
                )

                # Add vertical lines for key statistics
                fig.add_vline(x=initial_price, line_dash="solid", line_color="black",
                             annotation_text="Initial", row=1, col=1)
                fig.add_vline(x=mean_final, line_dash="dash", line_color="red",
                             annotation_text="Mean", row=1, col=1)
                fig.add_vline(x=var_price, line_dash="dash", line_color="orange",
                             annotation_text=f"VaR {confidence_level}%", row=1, col=1)

                # CDF - use valid_final_prices
                sorted_prices = np.sort(valid_final_prices)
                cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)

                fig.add_trace(
                    go.Scatter(
                        x=sorted_prices,
                        y=cdf * 100,
                        mode='lines',
                        name='CDF',
                        line=dict(color='forestgreen', width=2)
                    ),
                    row=1, col=2
                )

                # Add horizontal lines for percentiles
                fig.add_hline(y=5, line_dash="dash", line_color="orange", row=1, col=2)
                fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=2)
                fig.add_hline(y=95, line_dash="dash", line_color="green", row=1, col=2)

                fig.update_layout(height=450, template='plotly_white', showlegend=False)
                fig.update_xaxes(title_text="Price ($)", row=1, col=1)
                fig.update_xaxes(title_text="Price ($)", row=1, col=2)
                fig.update_yaxes(title_text="Frequency", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative %", row=1, col=2)

                st.plotly_chart(fig, use_container_width=True)

                # Statistics table
                st.markdown("### Price Distribution Statistics")
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', '5th Percentile', '25th Percentile',
                                 '75th Percentile', '95th Percentile', 'Min', 'Max'],
                    'Value ($)': [mean_final, median_final, std_final, p5, p25, p75, p95, min_final, max_final],
                    'Return (%)': [
                        (mean_final - initial_price) / initial_price * 100,
                        (median_final - initial_price) / initial_price * 100,
                        std_final / initial_price * 100,
                        (p5 - initial_price) / initial_price * 100,
                        (p25 - initial_price) / initial_price * 100,
                        (p75 - initial_price) / initial_price * 100,
                        (p95 - initial_price) / initial_price * 100,
                        (min_final - initial_price) / initial_price * 100,
                        (max_final - initial_price) / initial_price * 100
                    ]
                })
                stats_df['Value ($)'] = stats_df['Value ($)'].apply(lambda x: f"${x:.2f}")
                stats_df['Return (%)'] = stats_df['Return (%)'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            with tab3:
                # Confidence intervals over time
                fig = go.Figure()

                days = np.arange(time_horizon + 1)

                # Calculate percentiles at each time step
                p5_path = np.percentile(price_paths, 5, axis=0)
                p25_path = np.percentile(price_paths, 25, axis=0)
                p50_path = np.percentile(price_paths, 50, axis=0)
                p75_path = np.percentile(price_paths, 75, axis=0)
                p95_path = np.percentile(price_paths, 95, axis=0)

                # 90% confidence interval (5-95)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([days, days[::-1]]),
                    y=np.concatenate([p95_path, p5_path[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 100, 80, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='90% CI'
                ))

                # 50% confidence interval (25-75)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([days, days[::-1]]),
                    y=np.concatenate([p75_path, p25_path[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 100, 80, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='50% CI'
                ))

                # Median path
                fig.add_trace(go.Scatter(
                    x=days,
                    y=p50_path,
                    mode='lines',
                    name='Median',
                    line=dict(width=3, color='darkgreen')
                ))

                # Initial price line
                fig.add_hline(y=initial_price, line_dash="dash", line_color="gray",
                             annotation_text="Initial Price")

                fig.update_layout(
                    title='Price Confidence Intervals Over Time',
                    xaxis_title='Days',
                    yaxis_title='Price ($)',
                    height=500,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                # Risk analysis
                st.markdown("### Risk Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    **Value at Risk (VaR) at {confidence_level}% confidence:**

                    There is a {100-confidence_level}% chance that the final price will be
                    **below ${var_price:.2f}** (loss of **{abs(var_return):.2f}%** or more).

                    **Expected Shortfall (CVaR):**

                    If prices fall below the VaR threshold, the expected price is
                    **${expected_shortfall:.2f}** (average loss of **{abs(es_return):.2f}%**).
                    """)

                with col2:
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob_profit,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probability of Profit"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen" if prob_profit > 50 else "darkred"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightcoral"},
                                {'range': [50, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                # Returns distribution
                st.markdown("### Returns Distribution")

                returns = (valid_final_prices - initial_price) / initial_price * 100

                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns',
                    marker_color=np.where(returns >= 0, 'green', 'red'),
                    opacity=0.7
                ))

                fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
                fig.add_vline(x=np.mean(returns), line_dash="dash", line_color="blue",
                             annotation_text=f"Mean: {np.mean(returns):.2f}%")

                fig.update_layout(
                    title='Distribution of Simulated Returns',
                    xaxis_title='Return (%)',
                    yaxis_title='Frequency',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Scenario analysis
                st.markdown("### Scenario Analysis")

                scenarios = pd.DataFrame({
                    'Scenario': ['Best Case (95th %)', 'Bullish (75th %)', 'Base Case (Median)',
                                'Bearish (25th %)', 'Worst Case (5th %)'],
                    'Final Price': [f"${p95:.2f}", f"${p75:.2f}", f"${median_final:.2f}",
                                   f"${p25:.2f}", f"${p5:.2f}"],
                    'Return': [
                        f"{(p95-initial_price)/initial_price*100:+.2f}%",
                        f"{(p75-initial_price)/initial_price*100:+.2f}%",
                        f"{(median_final-initial_price)/initial_price*100:+.2f}%",
                        f"{(p25-initial_price)/initial_price*100:+.2f}%",
                        f"{(p5-initial_price)/initial_price*100:+.2f}%"
                    ],
                    'Probability': ['5%', '25%', '50%', '25%', '5%']
                })

                st.dataframe(scenarios, use_container_width=True, hide_index=True)

        else:
            st.info("Configure parameters and click 'Run Monte Carlo Simulation' to start.")

            st.markdown("""
            ### About Monte Carlo Simulation

            **Geometric Brownian Motion (GBM)** is used to model stock prices:

            $$dS = S(\\mu \\, dt + \\sigma \\, dW)$$

            Where:
            - $S$ = Stock price
            - $\\mu$ = Expected return (drift)
            - $\\sigma$ = Volatility
            - $dW$ = Wiener process (random shock)

            **Key Outputs:**
            - **Price Paths**: Visualize potential future trajectories
            - **VaR (Value at Risk)**: Maximum expected loss at confidence level
            - **Expected Shortfall**: Average loss in worst-case scenarios
            - **Confidence Intervals**: Range of likely outcomes over time
            """)

    # BACKTESTING
    elif page == "Backtesting":
        st.header("Backtesting Results")

        st.info("View backtesting performance of trading strategies")

        st.markdown("""
        *Backtesting simulates trading on historical data to evaluate strategy performance.*
        """)

        # Placeholder for backtesting results
        st.warning("Run backtesting module to generate results")

        st.code("""
# Run backtesting
from src.evaluation.backtester import Backtester
from src.trading.agent import TradingAgent

# Create agent and run backtest
agent = TradingAgent(model)
results = agent.run_backtest(signals_df, prices_df)
        """)

    # LIVE DEMO
    elif page == "Live Demo":
        st.header("Live Prediction Demo")

        st.warning("⚠️ This is a SIMULATION only - no real trading occurs!")

        st.markdown("""
        Generate predictions and trading signals for selected stocks.
        """)

        # Placeholder
        st.info("Live demo requires trained model. Please train a model first:")
        st.code("python -m src.training.train --model pinn --epochs 50")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **PINN Financial Forecasting**

    Dissertation Project
    Physics-Informed ML for Finance

    ⚠️ Academic Research Only
    """)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        st.error("Please check the logs for more details")
        import traceback
        st.code(traceback.format_exc())
