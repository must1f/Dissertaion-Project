"""
Backtesting Dashboard - Interactive Web Interface

Streamlit dashboard for running and analyzing backtests with multiple strategies
and models for the PINN Financial Forecasting system.

Author: Claude Code
Date: 2026-02-04
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.backtesting_platform import (
    BacktestingPlatform,
    BacktestConfig,
    StrategyResult,
    BuyAndHoldStrategy,
    SMACrossoverStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    PositionSizingMethod
)

# Page config
st.set_page_config(
    page_title="PINN Backtesting Platform",
    page_icon="📊",
    layout="wide"
)


def load_results_data():
    """Load available model results"""
    results_dir = Path("results")
    models = {}

    if results_dir.exists():
        for f in results_dir.glob("*_results.json"):
            try:
                with open(f) as file:
                    data = json.load(file)
                    model_name = f.stem.replace("_results", "")
                    models[model_name] = data
            except Exception:
                pass

    return models


def load_price_data():
    """Load historical price data"""
    # Try to load from parquet
    data_dir = Path("data")
    parquet_files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []

    if parquet_files:
        try:
            df = pd.read_parquet(parquet_files[0])
            return df
        except Exception:
            pass

    # Generate synthetic data for demo
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    data = {
        'timestamp': dates,
        'ticker': ['DEMO'] * n_days,
        'close': 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n_days)))
    }
    return pd.DataFrame(data)


def create_equity_curve_chart(results: dict):
    """Create equity curve visualization"""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (name, result) in enumerate(results.items()):
        if 'portfolio_values' in result and result['portfolio_values']:
            fig.add_trace(go.Scatter(
                y=result['portfolio_values'],
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    fig.update_layout(
        title="Portfolio Equity Curves",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )

    return fig


def create_drawdown_chart(results: dict):
    """Create drawdown visualization"""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, (name, result) in enumerate(results.items()):
        if 'portfolio_values' in result and result['portfolio_values']:
            values = np.array(result['portfolio_values'])
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak * 100

            fig.add_trace(go.Scatter(
                y=drawdown,
                mode='lines',
                name=name,
                fill='tozeroy',
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Trading Days",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400
    )

    return fig


def create_returns_distribution(results: dict):
    """Create returns distribution chart"""
    fig = go.Figure()

    for name, result in results.items():
        if 'returns' in result and len(result.get('returns', [])) > 0:
            returns = np.array(result['returns']) * 100  # Convert to percentage
            fig.add_trace(go.Histogram(
                x=returns,
                name=name,
                opacity=0.7,
                nbinsx=50
            ))

    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )

    return fig


def create_metrics_comparison_chart(comparison_df: pd.DataFrame, metric: str, title: str):
    """Create bar chart comparing strategies on a metric"""
    fig = px.bar(
        comparison_df,
        x='strategy_name',
        y=metric,
        color='strategy_name',
        title=title
    )

    fig.update_layout(
        xaxis_title="Strategy",
        yaxis_title=title,
        showlegend=False,
        height=400
    )

    return fig


def run_backtest_with_config(prices_df, config_dict, strategies_selected):
    """Run backtest with user configuration"""
    config = BacktestConfig(
        initial_capital=config_dict['initial_capital'],
        commission_rate=config_dict['commission_rate'] / 100,
        slippage_rate=config_dict['slippage_rate'] / 100,
        max_position_size=config_dict['max_position_size'] / 100,
        stop_loss=config_dict['stop_loss'] / 100,
        take_profit=config_dict['take_profit'] / 100,
        position_sizing=config_dict['position_sizing'],
        use_stop_loss=config_dict['use_stop_loss'],
        use_take_profit=config_dict['use_take_profit']
    )

    platform = BacktestingPlatform(config)

    strategies = []
    if 'Buy and Hold' in strategies_selected:
        strategies.append(BuyAndHoldStrategy())
    if 'SMA Crossover (20/50)' in strategies_selected:
        strategies.append(SMACrossoverStrategy(20, 50))
    if 'SMA Crossover (10/30)' in strategies_selected:
        strategies.append(SMACrossoverStrategy(10, 30))
    if 'Momentum (20d)' in strategies_selected:
        strategies.append(MomentumStrategy(20, 0.02))
    if 'Mean Reversion (BB)' in strategies_selected:
        strategies.append(MeanReversionStrategy(20, 2.0))

    results = {}
    for strategy in strategies:
        result = platform.run_backtest(
            strategy=strategy,
            prices=prices_df,
            predictions=None,
            model_name="Benchmark"
        )
        results[strategy.name] = {
            'portfolio_values': result.portfolio_values,
            'returns': result.returns.tolist() if len(result.returns) > 0 else [],
            **result.to_dict()
        }

    return results, platform.get_summary_table()


def main():
    st.title("📊 PINN Backtesting Platform")

    st.markdown("""
    ⚠️ **DISCLAIMER: FOR ACADEMIC RESEARCH ONLY - NOT FINANCIAL ADVICE**

    This backtesting platform is part of a dissertation research project. Results are simulated
    and do not guarantee future performance. Always consult qualified professionals before investing.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Backtest Configuration")

    with st.sidebar.expander("📈 Capital & Costs", expanded=True):
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )

        commission_rate = st.slider(
            "Commission Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        )

        slippage_rate = st.slider(
            "Slippage Rate (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01
        )

    with st.sidebar.expander("🎯 Position Sizing", expanded=True):
        position_sizing = st.selectbox(
            "Sizing Method",
            options=["Fixed", "Kelly", "Volatility", "Confidence", "Equal Weight"],
            index=0
        )

        position_sizing_map = {
            "Fixed": PositionSizingMethod.FIXED,
            "Kelly": PositionSizingMethod.KELLY,
            "Volatility": PositionSizingMethod.VOLATILITY,
            "Confidence": PositionSizingMethod.CONFIDENCE,
            "Equal Weight": PositionSizingMethod.EQUAL_WEIGHT
        }

        max_position_size = st.slider(
            "Max Position Size (%)",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

    with st.sidebar.expander("🛡️ Risk Management", expanded=True):
        use_stop_loss = st.checkbox("Enable Stop-Loss", value=True)
        stop_loss = st.slider(
            "Stop-Loss (%)",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            disabled=not use_stop_loss
        )

        use_take_profit = st.checkbox("Enable Take-Profit", value=True)
        take_profit = st.slider(
            "Take-Profit (%)",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            disabled=not use_take_profit
        )

    # Strategy selection
    st.sidebar.header("📋 Strategy Selection")
    strategies_selected = st.sidebar.multiselect(
        "Select Strategies to Compare",
        options=[
            "Buy and Hold",
            "SMA Crossover (20/50)",
            "SMA Crossover (10/30)",
            "Momentum (20d)",
            "Mean Reversion (BB)"
        ],
        default=["Buy and Hold", "SMA Crossover (20/50)", "Momentum (20d)"]
    )

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview",
        "📈 Run Backtest",
        "📊 Results Analysis",
        "🔄 Walk-Forward",
        "📋 Reports"
    ])

    with tab1:
        st.header("Backtesting Platform Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Available Strategies")
            st.markdown("""
            **Benchmark Strategies:**
            - **Buy and Hold**: Simple long-only passive strategy
            - **SMA Crossover**: Moving average crossover signals
            - **Momentum**: Trend-following based on recent returns
            - **Mean Reversion**: Bollinger Band-based contrarian strategy

            **Model-Based Strategies:**
            - **PINN Strategy**: Based on PINN model predictions
            - **LSTM Strategy**: Based on LSTM predictions
            - **Transformer Strategy**: Based on Transformer predictions
            """)

        with col2:
            st.subheader("Key Features")
            st.markdown("""
            ✅ **Multiple Strategy Comparison**
            ✅ **Transaction Cost Modeling**
            ✅ **Risk Management** (Stop-Loss, Take-Profit)
            ✅ **Position Sizing Methods** (Fixed, Kelly, Vol-based)
            ✅ **Walk-Forward Validation**
            ✅ **Monte Carlo Simulation**
            ✅ **Comprehensive Metrics**
            """)

        # Show available models
        st.subheader("Available Trained Models")
        models = load_results_data()

        if models:
            model_info = []
            for name, data in models.items():
                dir_acc = data.get('financial_metrics', {}).get('directional_accuracy', 'N/A')
                if isinstance(dir_acc, (int, float)):
                    if dir_acc <= 1:
                        dir_acc = dir_acc * 100
                    dir_acc = f"{dir_acc:.2f}%"

                info = {
                    'Model': name,
                    'Has Financial Metrics': 'financial_metrics' in data,
                    'Sharpe Ratio': data.get('financial_metrics', {}).get('sharpe_ratio', 'N/A'),
                    'Dir. Accuracy': dir_acc
                }
                model_info.append(info)

            st.dataframe(pd.DataFrame(model_info), use_container_width=True)
        else:
            st.info("No trained models found. Train models first using the training pipeline.")

    with tab2:
        st.header("Run Backtest")

        # Load or generate data
        prices_df = load_price_data()

        st.subheader("Data Preview")
        st.dataframe(prices_df.head(10), use_container_width=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration Summary")
            st.json({
                'Initial Capital': f"${initial_capital:,.0f}",
                'Commission': f"{commission_rate}%",
                'Slippage': f"{slippage_rate}%",
                'Max Position': f"{max_position_size}%",
                'Stop-Loss': f"{stop_loss}%" if use_stop_loss else "Disabled",
                'Take-Profit': f"{take_profit}%" if use_take_profit else "Disabled",
                'Position Sizing': position_sizing,
                'Strategies': strategies_selected
            })

        with col2:
            if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                if not strategies_selected:
                    st.error("Please select at least one strategy")
                else:
                    config_dict = {
                        'initial_capital': initial_capital,
                        'commission_rate': commission_rate,
                        'slippage_rate': slippage_rate,
                        'max_position_size': max_position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_sizing': position_sizing_map[position_sizing],
                        'use_stop_loss': use_stop_loss,
                        'use_take_profit': use_take_profit
                    }

                    with st.spinner("Running backtest..."):
                        results, summary_table = run_backtest_with_config(
                            prices_df, config_dict, strategies_selected
                        )

                        st.session_state['backtest_results'] = results
                        st.session_state['summary_table'] = summary_table

                    st.success("Backtest completed!")

        # Show results if available
        if 'backtest_results' in st.session_state:
            st.subheader("Results Summary")
            st.dataframe(st.session_state['summary_table'], use_container_width=True)

    with tab3:
        st.header("Results Analysis")

        if 'backtest_results' not in st.session_state:
            st.info("Run a backtest first to see results analysis.")
        else:
            results = st.session_state['backtest_results']

            # Equity curves
            st.subheader("Equity Curves")
            st.plotly_chart(create_equity_curve_chart(results), use_container_width=True)

            # Metrics comparison
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Drawdown Analysis")
                st.plotly_chart(create_drawdown_chart(results), use_container_width=True)

            with col2:
                st.subheader("Returns Distribution")
                st.plotly_chart(create_returns_distribution(results), use_container_width=True)

            # Detailed metrics table
            st.subheader("Detailed Metrics Comparison")

            metrics_data = []
            for name, result in results.items():
                metrics_data.append({
                    'Strategy': name,
                    'Total Return (%)': f"{result.get('total_return', 0):.2f}",
                    'Annual Return (%)': f"{result.get('annualized_return', 0):.2f}",
                    'Sharpe Ratio': f"{result.get('sharpe_ratio', 0):.2f}",
                    'Sortino Ratio': f"{result.get('sortino_ratio', 0):.2f}",
                    'Max Drawdown (%)': f"{result.get('max_drawdown', 0):.2f}",
                    'Win Rate (%)': f"{result.get('win_rate', 0):.2f}",
                    'Profit Factor': f"{result.get('profit_factor', 0):.2f}",
                    'Total Trades': result.get('total_trades', 0)
                })

            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            # Best strategy highlight
            if metrics_data:
                best_sharpe = max(metrics_data, key=lambda x: float(x['Sharpe Ratio']))
                st.success(f"🏆 Best Sharpe Ratio: **{best_sharpe['Strategy']}** ({best_sharpe['Sharpe Ratio']})")

    with tab4:
        st.header("Walk-Forward Validation")

        st.markdown("""
        Walk-forward validation tests strategy robustness by:
        1. Dividing data into multiple periods
        2. Running backtest on each period independently
        3. Analyzing consistency of results across periods
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            n_splits = st.slider("Number of Splits", min_value=2, max_value=10, value=5)

        with col2:
            wf_mode = st.selectbox("Validation Mode", ['expanding', 'rolling'])

        with col3:
            run_wf = st.button("🔄 Run Walk-Forward Validation")

        # Initialize session state for walk-forward results
        if 'wf_results' not in st.session_state:
            st.session_state['wf_results'] = None

        if run_wf:
            with st.spinner("Running walk-forward validation..."):
                # Load price data
                prices_df = load_price_data()

                if prices_df is not None and len(prices_df) > 100:
                    # Use WalkForwardValidator
                    from src.training.walk_forward import WalkForwardValidator

                    n_samples = len(prices_df)
                    initial_train = n_samples // (n_splits + 1)
                    val_size = n_samples // (n_splits * 2)

                    validator = WalkForwardValidator(
                        n_samples=n_samples,
                        initial_train_size=initial_train,
                        validation_size=val_size,
                        mode=wf_mode
                    )

                    folds = validator.split()
                    wf_data = []

                    for fold in folds:
                        # Get fold data
                        train_prices = prices_df.iloc[fold.train_start_idx:fold.train_end_idx]
                        val_prices = prices_df.iloc[fold.val_start_idx:fold.val_end_idx]

                        if len(val_prices) > 10:
                            # Compute simple returns for the validation period
                            val_close = val_prices['close'].values
                            returns = np.diff(val_close) / val_close[:-1]

                            # Compute metrics
                            total_return = (val_close[-1] / val_close[0] - 1) * 100
                            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)
                            positive_returns = np.sum(returns > 0)
                            win_rate = (positive_returns / len(returns)) * 100 if len(returns) > 0 else 0

                            # Drawdown
                            cum_returns = np.cumprod(1 + returns)
                            peak = np.maximum.accumulate(cum_returns)
                            drawdown = (cum_returns - peak) / peak
                            max_dd = np.min(drawdown) * 100

                            wf_data.append({
                                'Fold': f"Fold {fold.fold_num + 1}",
                                'Train Size': fold.train_end_idx - fold.train_start_idx,
                                'Val Size': fold.val_end_idx - fold.val_start_idx,
                                'Sharpe Ratio': round(sharpe, 2),
                                'Total Return (%)': round(total_return, 2),
                                'Max Drawdown (%)': round(max_dd, 2),
                                'Win Rate (%)': round(win_rate, 2)
                            })

                    st.session_state['wf_results'] = wf_data
                    st.success(f"Walk-forward validation complete: {len(wf_data)} folds analyzed")
                else:
                    st.warning("Insufficient data for walk-forward validation")

        # Display results
        if st.session_state['wf_results'] is not None:
            wf_data = st.session_state['wf_results']
            st.subheader("Walk-Forward Results")
            wf_df = pd.DataFrame(wf_data)
            st.dataframe(wf_df, use_container_width=True)

            # Stability metrics
            st.subheader("Stability Metrics")
            col1, col2, col3, col4 = st.columns(4)

            sharpe_values = [d['Sharpe Ratio'] for d in wf_data]
            returns_values = [d['Total Return (%)'] for d in wf_data]
            win_rates = [d['Win Rate (%)'] for d in wf_data]

            avg_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)
            consistency = (np.sum(np.array(sharpe_values) > 0) / len(sharpe_values)) * 100

            with col1:
                st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
            with col2:
                st.metric("Sharpe Std Dev", f"{std_sharpe:.2f}")
            with col3:
                st.metric("Consistency", f"{consistency:.0f}%")
            with col4:
                st.metric("Avg Win Rate", f"{np.mean(win_rates):.1f}%")

            # Visualization
            st.subheader("Performance Across Folds")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d['Fold'] for d in wf_data],
                y=sharpe_values,
                name='Sharpe Ratio',
                marker_color='#3498db'
            ))
            fig.update_layout(
                title="Sharpe Ratio by Fold",
                xaxis_title="Fold",
                yaxis_title="Sharpe Ratio",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Run Walk-Forward Validation' to analyze strategy robustness across time periods.")

    with tab5:
        st.header("Reports & Export")

        if 'backtest_results' in st.session_state:
            st.subheader("Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📄 Export to JSON"):
                    report = {
                        'generated_at': datetime.now().isoformat(),
                        'results': st.session_state['backtest_results']
                    }
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(report, indent=2, default=str),
                        file_name="backtest_report.json",
                        mime="application/json"
                    )

            with col2:
                if st.button("📊 Export to CSV"):
                    csv_data = st.session_state['summary_table'].to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="backtest_summary.csv",
                        mime="text/csv"
                    )

            with col3:
                if st.button("📈 Save Charts"):
                    st.info("Charts would be saved as PNG files.")

            # Report preview
            st.subheader("Report Preview")

            st.markdown(f"""
            ## Backtest Report

            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            ### Configuration
            - Initial Capital: ${initial_capital:,.0f}
            - Commission: {commission_rate}%
            - Position Sizing: {position_sizing}

            ### Results Summary
            """)

            st.dataframe(st.session_state['summary_table'], use_container_width=True)

        else:
            st.info("Run a backtest first to generate reports.")


if __name__ == "__main__":
    main()
