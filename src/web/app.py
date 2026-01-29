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
    results_path = config.project_root / 'results' / f'{model_name}_results.json'

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
        ["Home", "All Models Dashboard", "PINN Comparison", "Model Comparison", "Prediction Visualizations", "Data Explorer", "Backtesting", "Live Demo"]
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
            'lstm', 'gru', 'transformer',
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

            row = {
                'Model': model.upper(),

                # Traditional ML Metrics
                'RMSE': metrics.get('rmse', metrics.get('test_rmse', np.nan)),
                'MAE': metrics.get('mae', metrics.get('test_mae', np.nan)),
                'R²': metrics.get('r2', metrics.get('test_r2', np.nan)),

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
            ml_cols = ['Model', 'RMSE', 'MAE', 'R²', 'Dir Acc %']
            ml_df = df_comparison[ml_cols].copy()

            styled_df = ml_df.style.highlight_min(
                subset=['RMSE', 'MAE'],
                color='lightgreen'
            ).highlight_max(
                subset=['R²', 'Dir Acc %'],
                color='lightgreen'
            ).format({
                'RMSE': '{:.6f}',
                'MAE': '{:.6f}',
                'R²': '{:.4f}',
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

            # Placeholder for actual data (would be loaded once predictions are available)
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
