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
from src.utils.logger import get_logger
from src.data.fetcher import DataFetcher
from src.evaluation.backtester import BacktestResults

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
        ["Home", "Data Explorer", "Model Comparison", "Backtesting", "Live Demo"]
    )

    config = load_config()

    # HOME PAGE
    if page == "Home":
        st.header("Welcome to PINN Financial Forecasting")

        st.markdown("""
        This system combines **Physics-Informed Neural Networks** with financial time series forecasting.

        ### Key Features:
        - **Physics Constraints**: Embeds GBM, Black-Scholes, OU, and Langevin dynamics
        - **Multiple Architectures**: LSTM, GRU, Transformer, and PINN models
        - **Risk Management**: Stop-loss, take-profit, position sizing
        - **Comprehensive Evaluation**: Sharpe ratio, max drawdown, win rate, etc.

        ### How It Works:
        1. **Data Collection**: Fetch S&P 500 data via yfinance
        2. **Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands)
        3. **Model Training**: Train baseline and PINN models
        4. **Backtesting**: Test strategies on historical data
        5. **Evaluation**: Compare performance metrics

        ### Navigate using the sidebar to explore different features!
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

        st.info("Compare performance of different models")

        # Load results for all models
        models = ['lstm', 'gru', 'transformer', 'pinn']
        results = {}

        for model in models:
            result = load_results(model)
            if result:
                results[model] = result

        if not results:
            st.warning("No model results found. Please train models first.")
            st.code("python -m src.training.train --model pinn")
            return

        # Show comparison table
        st.subheader("Test Set Performance")

        comparison_data = []
        for model, result in results.items():
            metrics = result.get('test_metrics', {})
            comparison_data.append({
                'Model': model.upper(),
                'RMSE': metrics.get('test_rmse', 0),
                'MAE': metrics.get('test_mae', 0),
                'MAPE': metrics.get('test_mape', 0),
                'R²': metrics.get('test_r2', 0),
                'Dir. Acc.': metrics.get('test_directional_accuracy', 0)
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen')
                                        .highlight_max(subset=['R²', 'Dir. Acc.'], color='lightgreen'))

        # Plot training history for selected model
        selected_model = st.selectbox("Select model to view training history", list(results.keys()))

        if 'training_history' in results[selected_model]:
            history = results[selected_model]['training_history']
            st.plotly_chart(plot_training_history(history), use_container_width=True)

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
    main()
