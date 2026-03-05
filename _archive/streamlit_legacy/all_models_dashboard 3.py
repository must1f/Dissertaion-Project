"""
All Models Dashboard - Complete Neural Network Model Comparison

Shows ALL available models (LSTM, GRU, Transformer, all PINN variants)
with comprehensive financial metrics and training status

Performance optimizations:
- Cached model registry to avoid repeated filesystem scans
- Cached results loading with @st.cache_data
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
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.models.model_registry import get_model_registry, ModelInfo
from src.utils.logger import get_logger, ensure_logger_initialized

ensure_logger_initialized()
logger = get_logger(__name__)


# Cached model registry getter - prevents repeated filesystem scans
@st.cache_resource(ttl=300)
def _get_cached_registry(project_root: str):
    """Get cached model registry instance (5 min TTL)."""
    return get_model_registry(Path(project_root))


# Cached results file loader
@st.cache_data(ttl=300)
def _load_results_file(file_path: str) -> Optional[Dict]:
    """Load a results JSON file with caching (5 min TTL)."""
    path = Path(file_path)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Could not load {file_path}: {e}")
    return None


class AllModelsDashboard:
    """Dashboard for all neural network models"""

    def __init__(self):
        self.config = get_config()
        # Use cached registry to avoid repeated filesystem scans
        self.registry = _get_cached_registry(str(self.config.project_root))

    def render_model_overview(self):
        """Render overview of all models"""
        st.subheader("📋 Model Overview")

        summary = self.registry.get_summary()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Models", summary['total_models'])

        with col2:
            st.metric("Trained Models", summary['trained'],
                     delta=f"{summary['trained']/summary['total_models']*100:.0f}%")

        with col3:
            st.metric("Untrained Models", summary['untrained'])

        with col4:
            completion = summary['trained'] / summary['total_models'] * 100
            st.metric("Training Completion", f"{completion:.0f}%")

        # Progress by type
        st.markdown("### Training Progress by Type")

        type_data = []
        for model_type, stats in summary['by_type'].items():
            type_data.append({
                'Type': model_type.title(),
                'Total': stats['total'],
                'Trained': stats['trained'],
                'Untrained': stats['untrained'],
                'Completion_%': stats['trained'] / stats['total'] * 100 if stats['total'] > 0 else 0
            })

        df_types = pd.DataFrame(type_data)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Trained',
            x=df_types['Type'],
            y=df_types['Trained'],
            marker_color='lightgreen'
        ))

        fig.add_trace(go.Bar(
            name='Untrained',
            x=df_types['Type'],
            y=df_types['Untrained'],
            marker_color='lightcoral'
        ))

        fig.update_layout(
            title='Training Status by Model Type',
            xaxis_title='Model Type',
            yaxis_title='Count',
            barmode='stack',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_model_list(self):
        """Render list of all models with status"""
        st.subheader("🗂️ All Available Models")

        all_models = self.registry.get_all_models()

        # Create DataFrame
        model_data = []
        for key, model in all_models.items():
            model_data.append({
                'Model': model.model_name,
                'Type': model.model_type.title(),
                'Architecture': model.architecture,
                'Status': '✅ Trained' if model.trained else '⚪ Untrained',
                'Trained': model.trained,
                'Training_Date': model.training_date or '-',
                'Epochs': model.epochs_trained or '-',
                'Model_Key': key,
                'Description': model.description
            })

        df_models = pd.DataFrame(model_data)

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Models", "Trained Only", "Untrained Only"])

        with tab1:
            self._render_model_table(df_models, show_all=True)

        with tab2:
            trained_df = df_models[df_models['Trained'] == True]
            if len(trained_df) > 0:
                st.success(f"✅ {len(trained_df)} models trained")
                self._render_model_table(trained_df, show_all=False)
            else:
                st.warning("No trained models yet. Train models to see results here.")

        with tab3:
            untrained_df = df_models[df_models['Trained'] == False]
            if len(untrained_df) > 0:
                st.info(f"⚪ {len(untrained_df)} models need training")
                self._render_model_table(untrained_df, show_all=False)
                self._show_training_instructions(untrained_df)
            else:
                st.success("All models trained! 🎉")

    def _render_model_table(self, df: pd.DataFrame, show_all: bool = True):
        """Render model table"""
        display_cols = ['Model', 'Type', 'Architecture', 'Status', 'Training_Date', 'Epochs']

        # Color coding based on Status column
        def highlight_trained(row):
            if '✅' in str(row['Status']):
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)

        styled_df = df[display_cols].style.apply(highlight_trained, axis=1)

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Expandable details
        with st.expander("📖 Model Descriptions"):
            for _, row in df.iterrows():
                st.markdown(f"**{row['Model']}** ({row['Architecture']}): {row['Description']}")

    def _show_training_instructions(self, untrained_df: pd.DataFrame):
        """Show training instructions for untrained models"""
        st.markdown("### 🎓 Training Instructions")

        # Group by type
        baseline_models = untrained_df[untrained_df['Type'] == 'Baseline']['Model_Key'].tolist()
        pinn_models = untrained_df[untrained_df['Type'] == 'Pinn']['Model_Key'].tolist()
        advanced_models = untrained_df[untrained_df['Type'] == 'Advanced']['Model_Key'].tolist()

        if baseline_models:
            st.markdown("**Baseline Models (LSTM, GRU, Transformer):**")
            st.code(f"""
# Train all baseline models
python src/training/train.py --model lstm --epochs 100
python src/training/train.py --model gru --epochs 100
python src/training/train.py --model transformer --epochs 100

# Or train all at once (option 4 in run.sh)
./run.sh
# Select option 4: Train All Models
            """)

        if pinn_models:
            st.markdown("**PINN Variants (6 physics configurations):**")
            st.code(f"""
# Train all 6 PINN variants
python src/training/train_pinn_variants.py --epochs 100

# Or use run.sh option 10
./run.sh
# Select option 10: Systematic PINN Physics Comparison
            """)

        if advanced_models:
            st.markdown("**Advanced PINN Architectures:**")
            st.code(f"""
# Train StackedPINN
python src/training/train_stacked_pinn.py --model-type stacked --epochs 100

# Train ResidualPINN
python src/training/train_stacked_pinn.py --model-type residual --epochs 100
            """)

    def render_metrics_comparison(self):
        """Render comprehensive metrics comparison"""
        st.subheader("📊 Comprehensive Metrics Comparison")

        st.info("Comparing all trained models with comprehensive financial metrics")

        trained_models = self.registry.get_trained_models()

        if not trained_models:
            st.warning("No trained models found. Train models first to see comparisons.")
            return

        # Load results for all trained models
        model_results = {}
        for key, model_info in trained_models.items():
            result = self._load_model_results(key, model_info)
            if result:
                model_results[key] = result

        if not model_results:
            st.warning("No results found. Ensure models are trained and results are saved.")
            return

        st.success(f"✓ Loaded results for {len(model_results)} models")

        # Create comparison DataFrame
        comparison_data = []
        for model_key, result in model_results.items():
            model_info = trained_models[model_key]

            # Get metrics from different possible locations
            if 'financial_metrics' in result:
                metrics = result['financial_metrics']
            elif 'test_metrics' in result:
                metrics = result['test_metrics']
            else:
                metrics = {}

            ml_metrics = result.get('ml_metrics', {})

            dir_acc = metrics.get('directional_accuracy', 0)
            if dir_acc <= 1:
                dir_acc *= 100

            row = {
                'Model': model_info.model_name,
                'Type': model_info.model_type.title(),
                'Architecture': model_info.architecture,

                # ML Metrics
                'MSE': ml_metrics.get('mse', metrics.get('mse', np.nan)),
                'RMSE': ml_metrics.get('rmse', metrics.get('rmse', np.nan)),
                'MAE': ml_metrics.get('mae', metrics.get('mae', np.nan)),
                'R²': ml_metrics.get('r2', metrics.get('r2', np.nan)),
                'MAPE': ml_metrics.get('mape', metrics.get('mape', np.nan)),

                # Financial Metrics
                'Sharpe': metrics.get('sharpe_ratio', np.nan),
                'Sortino': metrics.get('sortino_ratio', np.nan),
                'Max_DD_%': metrics.get('max_drawdown', 0) * 100,
                'Calmar': metrics.get('calmar_ratio', np.nan),
                'Annual_Ret_%': metrics.get('annualized_return', metrics.get('total_return', 0)) * 100,
                'Dir_Acc_%': dir_acc,
                'Profit_Factor': metrics.get('profit_factor', np.nan),
                'Win_Rate_%': metrics.get('win_rate', 0) * 100,
                'IC': metrics.get('information_coefficient', np.nan),
                'Volatility_%': metrics.get('volatility', 0) * 100
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Risk-Adjusted",
            "Capital Preservation",
            "Trading Viability",
            "Signal Quality"
        ])

        with tab1:
            self._render_overview_comparison(df)

        with tab2:
            self._render_risk_adjusted(df)

        with tab3:
            self._render_capital_preservation(df)

        with tab4:
            self._render_trading_viability(df)

        with tab5:
            self._render_signal_quality(df)

    def _load_model_results(self, model_key: str, model_info: ModelInfo) -> Optional[Dict]:
        """
        Load results for a model.

        OPTIMIZED: Uses cached file loading to prevent repeated disk reads.
        """
        # Try multiple locations (including pinn_ prefix variants)
        possible_paths = [
            self.config.project_root / 'results' / f'{model_key}_results.json',
            self.config.project_root / 'results' / f'pinn_{model_key}_results.json',
            self.config.project_root / 'results' / 'pinn_comparison' / f'{model_key}_results.json',
            self.config.project_root / 'results' / 'pinn_comparison' / f'pinn_{model_key}_results.json',
            self.config.project_root / 'models' / 'stacked_pinn' / f'{model_key}_pinn_results.json',
        ]

        if model_info.results_path:
            possible_paths.insert(0, model_info.results_path)

        for path in possible_paths:
            # Use cached loading function
            result = _load_results_file(str(path))
            if result:
                # Normalize metrics before returning
                return self._normalize_metrics(result)

        return None

    def _normalize_metrics(self, result: Dict) -> Dict:
        """Normalize metrics from different result formats and fix invalid values"""
        # Ensure ml_metrics exists
        if 'ml_metrics' not in result:
            result['ml_metrics'] = {}

        ml = result['ml_metrics']
        test = result.get('test_metrics', {})

        # Copy test_metrics to ml_metrics if missing
        if 'mse' not in ml:
            rmse = test.get('test_rmse') or ml.get('rmse', 0)
            ml['mse'] = rmse ** 2 if rmse else 0
        if 'rmse' not in ml:
            ml['rmse'] = test.get('test_rmse', 0)
        if 'mae' not in ml:
            ml['mae'] = test.get('test_mae', 0)
        if 'r2' not in ml:
            ml['r2'] = test.get('test_r2', 0)
        if 'mape' not in ml:
            ml['mape'] = test.get('test_mape', 0)

        # FIX: Validate financial metrics - replace inf/nan
        if 'financial_metrics' in result:
            fm = result['financial_metrics']
            for key, value in list(fm.items()):
                if isinstance(value, (int, float)):
                    if np.isinf(value) or value > 1e10 or value < -1e10:
                        fm[key] = 10.0 if value > 0 else -10.0
                    elif np.isnan(value):
                        fm[key] = 0.0

            # FIX: Ensure max_drawdown is capped at -100%
            if fm.get('max_drawdown', 0) < -1.0:
                fm['max_drawdown'] = -1.0

        return result

    def _render_overview_comparison(self, df: pd.DataFrame):
        """Render overview comparison"""
        st.markdown("### Complete Metrics Overview")

        # Select key metrics
        key_cols = ['Model', 'Type', 'MSE', 'RMSE', 'MAE', 'R²', 'Sharpe', 'Max_DD_%', 'Dir_Acc_%', 'Profit_Factor']

        styled_df = df[key_cols].style.highlight_min(
            subset=['MSE', 'RMSE', 'MAE', 'Max_DD_%'],
            color='lightgreen'
        ).highlight_max(
            subset=['R²', 'Sharpe', 'Dir_Acc_%', 'Profit_Factor'],
            color='lightgreen'
        ).format({
            'MSE': '{:.6f}',
            'RMSE': '{:.6f}',
            'MAE': '{:.6f}',
            'R²': '{:.4f}',
            'Sharpe': '{:.3f}',
            'Max_DD_%': '{:.2f}%',
            'Dir_Acc_%': '{:.2f}%',
            'Profit_Factor': '{:.2f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Best model summary
        best_sharpe = df.loc[df['Sharpe'].idxmax()]
        best_dd = df.loc[df['Max_DD_%'].idxmax()]
        best_acc = df.loc[df['Dir_Acc_%'].idxmax()]

        st.markdown("### 🏆 Best Performers")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Sharpe Ratio",
                     best_sharpe['Model'],
                     f"{best_sharpe['Sharpe']:.3f}")

        with col2:
            st.metric("Best Drawdown",
                     best_dd['Model'],
                     f"{best_dd['Max_DD_%']:.2f}%")

        with col3:
            st.metric("Best Dir. Accuracy",
                     best_acc['Model'],
                     f"{best_acc['Dir_Acc_%']:.2f}%")

    def _render_risk_adjusted(self, df: pd.DataFrame):
        """Render risk-adjusted metrics"""
        st.markdown("### Risk-Adjusted Performance")

        risk_cols = ['Model', 'Type', 'Sharpe', 'Sortino', 'Volatility_%']

        styled_df = df[risk_cols].style.highlight_max(
            subset=['Sharpe', 'Sortino'],
            color='lightgreen'
        ).highlight_min(
            subset=['Volatility_%'],
            color='lightgreen'
        ).format({
            'Sharpe': '{:.3f}',
            'Sortino': '{:.3f}',
            'Volatility_%': '{:.2f}%'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Chart
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Sharpe Ratio', 'Sortino Ratio'))

        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Sharpe'], marker_color='steelblue',
                  showlegend=False),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Sortino'], marker_color='forestgreen',
                  showlegend=False),
            row=1, col=2
        )

        fig.add_hline(y=1.0, line_dash="dash", line_color="green", row=1, col=1,
                     annotation_text="Sharpe = 1.0")
        fig.add_hline(y=1.5, line_dash="dash", line_color="green", row=1, col=2,
                     annotation_text="Sortino = 1.5")

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def _render_capital_preservation(self, df: pd.DataFrame):
        """Render capital preservation metrics"""
        st.markdown("### Capital Preservation")

        capital_cols = ['Model', 'Type', 'Max_DD_%', 'Calmar']

        styled_df = df[capital_cols].style.highlight_max(
            subset=['Max_DD_%', 'Calmar'],  # Less negative DD is better
            color='lightgreen'
        ).format({
            'Max_DD_%': '{:.2f}%',
            'Calmar': '{:.3f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df['Max_DD_%'],
            marker_color='coral',
            name='Max Drawdown'
        ))

        fig.add_hline(y=-20, line_dash="dash", line_color="orange",
                     annotation_text="-20% threshold")

        fig.update_layout(
            title='Maximum Drawdown Comparison',
            xaxis_title='Model',
            yaxis_title='Drawdown (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_trading_viability(self, df: pd.DataFrame):
        """Render trading viability metrics"""
        st.markdown("### Trading Viability")

        trading_cols = ['Model', 'Type', 'Annual_Ret_%', 'Profit_Factor', 'Win_Rate_%']

        styled_df = df[trading_cols].style.highlight_max(
            subset=['Annual_Ret_%', 'Profit_Factor', 'Win_Rate_%'],
            color='lightgreen'
        ).format({
            'Annual_Ret_%': '{:.2f}%',
            'Profit_Factor': '{:.2f}',
            'Win_Rate_%': '{:.2f}%'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Model'],
            y=df['Annual_Ret_%'],
            mode='markers+lines',
            marker=dict(size=12, color=df['Profit_Factor'], colorscale='Viridis',
                       showscale=True, colorbar=dict(title="Profit Factor")),
            line=dict(color='lightblue'),
            name='Annual Return'
        ))

        fig.update_layout(
            title='Annualized Return vs Profit Factor',
            xaxis_title='Model',
            yaxis_title='Return (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_signal_quality(self, df: pd.DataFrame):
        """Render signal quality metrics"""
        st.markdown("### Signal Quality")

        signal_cols = ['Model', 'Type', 'Dir_Acc_%', 'IC']

        styled_df = df[signal_cols].style.highlight_max(
            subset=['Dir_Acc_%', 'IC'],
            color='lightgreen'
        ).format({
            'Dir_Acc_%': '{:.2f}%',
            'IC': '{:.3f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Model'],
            y=df['Dir_Acc_%'],
            mode='markers+lines',
            marker=dict(size=12),
            line=dict(width=3),
            name='Directional Accuracy'
        ))

        fig.add_hline(y=50, line_dash="dash", line_color="red",
                     annotation_text="50% Random Baseline")
        fig.add_hline(y=55, line_dash="dash", line_color="orange",
                     annotation_text="55% Good")

        fig.update_layout(
            title='Directional Accuracy (Signal Quality)',
            xaxis_title='Model',
            yaxis_title='Accuracy (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard app"""
    st.set_page_config(
        page_title="All Models Dashboard",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 All Neural Network Models Dashboard")
    st.markdown("### Complete Model Comparison with Training Status")

    st.warning("""
    ⚠️ **ACADEMIC RESEARCH ONLY** - Not financial advice.
    Past performance does not guarantee future results.
    """)

    # Initialize dashboard
    dashboard = AllModelsDashboard()

    # Navigation
    page = st.sidebar.radio(
        "Dashboard Sections",
        [
            "Model Overview",
            "Model List & Status",
            "Metrics Comparison",
            "Training Guide"
        ]
    )

    # Render selected page
    if page == "Model Overview":
        dashboard.render_model_overview()

    elif page == "Model List & Status":
        dashboard.render_model_list()

    elif page == "Metrics Comparison":
        dashboard.render_metrics_comparison()

    elif page == "Training Guide":
        st.subheader("🎓 Complete Training Guide")
        st.markdown("""
        ### Training All Models

        **Quick Start (All Models):**
        ```bash
        ./run.sh
        # Select option 9: Complete Pipeline
        ```

        **Individual Training:**

        1. **Baseline Models:**
           ```bash
           python src/training/train.py --model lstm --epochs 100
           python src/training/train.py --model gru --epochs 100
           python src/training/train.py --model transformer --epochs 100
           ```

        2. **PINN Variants (6 configurations):**
           ```bash
           python src/training/train_pinn_variants.py --epochs 100
           ```

        3. **Advanced PINN Architectures:**
           ```bash
           python src/training/train_stacked_pinn.py --model-type stacked --epochs 100
           python src/training/train_stacked_pinn.py --model-type residual --epochs 100
           ```

        ### After Training

        1. **Launch Dashboard:**
           ```bash
           streamlit run src/web/all_models_dashboard.py
           # or
           streamlit run src/web/app.py
           ```

        2. **View Results:**
           - Model Overview: See training status
           - Metrics Comparison: Compare all models
           - Export results as needed

        ### Expected Training Times

        | Model Type | Epochs | Approximate Time |
        |-----------|--------|------------------|
        | LSTM/GRU | 100 | 15-30 minutes |
        | Transformer | 100 | 30-45 minutes |
        | PINN (single) | 100 | 20-40 minutes |
        | PINN (all 6) | 100 | 2-4 hours |
        | StackedPINN | 100 | 45-90 minutes |
        | ResidualPINN | 100 | 45-90 minutes |

        *Times vary based on dataset size and hardware*
        """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Model Types
    **Baseline**: LSTM, GRU, Transformer
    **PINN**: 6 physics variants
    **Advanced**: StackedPINN, ResidualPINN

    **Total**: 14 models available
    """)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        import traceback
        st.code(traceback.format_exc())
