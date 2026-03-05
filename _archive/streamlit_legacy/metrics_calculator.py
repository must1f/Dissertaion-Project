"""
In-App Metrics Calculator for Streamlit

Enables live computation of financial metrics within the dashboard
without requiring pre-computed JSON result files.

Includes data refresh functionality to fetch updated data from yfinance
and retrain models on the same dataset with consistent splits.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.unified_evaluator import UnifiedModelEvaluator
from src.evaluation.financial_metrics import FinancialMetrics, compute_strategy_returns
from src.evaluation.metrics import MetricsCalculator
from src.utils.config import get_config, get_research_config
from src.utils.logger import get_logger
import json

logger = get_logger(__name__)


@st.cache_resource
def get_evaluator(
    transaction_cost: float = 0.003,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> UnifiedModelEvaluator:
    """
    Get cached evaluator instance.

    Uses @st.cache_resource since evaluator is stateless but reusable.
    """
    return UnifiedModelEvaluator(
        transaction_cost=transaction_cost,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year
    )


@st.cache_data(ttl=300)
def load_predictions_cached(model_key: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load predictions with caching.

    TTL of 300 seconds (5 minutes) to allow for updated results.
    """
    config = get_config()
    results_dir = config.project_root / 'results'

    patterns = [
        results_dir / f'pinn_{model_key}_predictions.npz',
        results_dir / f'{model_key}_predictions.npz',
        results_dir / 'pinn_comparison' / f'{model_key}_predictions.npz',
        results_dir / 'pinn_comparison' / f'pinn_{model_key}_predictions.npz',
    ]

    for path in patterns:
        if path.exists():
            try:
                data = np.load(path)
                return {
                    'predictions': data['predictions'].flatten(),
                    'targets': data['targets'].flatten(),
                    'source': str(path)
                }
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    return None


@st.cache_data(ttl=300)
def compute_metrics_cached(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    transaction_cost: float = 0.003,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    compute_rolling: bool = True
) -> Dict:
    """
    Compute comprehensive metrics with caching.

    TTL of 300 seconds (5 min) aligned with predictions cache for consistency.

    Args:
        predictions: Model predictions
        targets: Actual values
        model_name: Display name for the model
        transaction_cost: Trading transaction cost
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        compute_rolling: Whether to compute rolling window metrics

    Returns:
        Dict with ml_metrics, financial_metrics, and optionally rolling_metrics
    """
    evaluator = UnifiedModelEvaluator(
        transaction_cost=transaction_cost,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year
    )

    return evaluator.evaluate_model(
        predictions=predictions,
        targets=targets,
        model_name=model_name,
        compute_rolling=compute_rolling
    )


def compute_quick_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute quick ML metrics without full financial analysis.

    Useful for fast feedback during data exploration.
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Remove NaN
    valid = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[valid]
    targets = targets[valid]

    if len(predictions) == 0:
        return {}

    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Directional accuracy
    pred_returns = np.diff(predictions)
    actual_returns = np.diff(targets)
    if len(pred_returns) > 0:
        dir_acc = float(np.mean(np.sign(pred_returns) == np.sign(actual_returns)))
    else:
        dir_acc = 0.0

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_accuracy': dir_acc,
        'n_samples': len(predictions)
    }


class StreamlitMetricsCalculator:
    """
    Streamlit-integrated metrics calculator.

    Provides UI components for computing and displaying metrics.
    Includes data refresh and model retraining functionality.
    """

    def __init__(self):
        self.config = get_config()
        self.research_config = get_research_config()
        self.results_dir = self.config.project_root / 'results'
        self.models_dir = self.config.project_root / 'models'

    def get_all_trained_models(self) -> List[Dict]:
        """Get list of all trained models with their prediction files."""
        models = []

        # Check for prediction files
        patterns = [
            self.results_dir.glob('*_predictions.npz'),
        ]

        # Also check pinn_comparison subdirectory
        pinn_dir = self.results_dir / 'pinn_comparison'
        if pinn_dir.exists():
            patterns.append(pinn_dir.glob('*_predictions.npz'))

        for pattern in patterns:
            for path in pattern:
                # Extract model key from filename
                name = path.stem.replace('_predictions', '')
                name = name.replace('pinn_', '')

                # Check for corresponding results file
                results_path = path.parent / f'{path.stem.replace("_predictions", "")}_results.json'

                models.append({
                    'key': name,
                    'predictions_path': path,
                    'results_path': results_path if results_path.exists() else None,
                    'has_results': results_path.exists()
                })

        return models

    def recalculate_all_metrics(
        self,
        transaction_cost: float = None,
        risk_free_rate: float = None,
        periods_per_year: int = None,
        compute_rolling: bool = True,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        Recalculate metrics for all trained models.

        Uses consistent parameters from research config for fair comparison.

        Args:
            transaction_cost: Transaction cost (defaults to research config)
            risk_free_rate: Risk-free rate (defaults to research config)
            periods_per_year: Periods per year (defaults to research config)
            compute_rolling: Whether to compute rolling metrics
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping model keys to their metrics
        """
        # Use research config defaults for consistency
        transaction_cost = transaction_cost or self.research_config.transaction_cost
        risk_free_rate = risk_free_rate or self.research_config.risk_free_rate
        periods_per_year = periods_per_year or self.research_config.periods_per_year

        models = self.get_all_trained_models()

        if not models:
            logger.warning("No trained models found for metrics recalculation")
            return {}

        all_results = {}
        evaluator = UnifiedModelEvaluator(
            transaction_cost=transaction_cost,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        )

        for i, model_info in enumerate(models):
            model_key = model_info['key']

            if progress_callback:
                progress_callback(
                    i / len(models),
                    f"Recalculating metrics for {model_key}..."
                )

            try:
                # Load predictions
                data = np.load(model_info['predictions_path'])
                predictions = data['predictions'].flatten()
                targets = data['targets'].flatten()

                # Compute metrics
                results = evaluator.evaluate_model(
                    predictions=predictions,
                    targets=targets,
                    model_name=model_key,
                    compute_rolling=compute_rolling
                )

                # Add metadata
                results['recalculation_timestamp'] = pd.Timestamp.now().isoformat()
                results['parameters'] = {
                    'transaction_cost': transaction_cost,
                    'risk_free_rate': risk_free_rate,
                    'periods_per_year': periods_per_year
                }

                # Save updated results
                output_path = self.results_dir / f'{model_key}_results.json'

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

                with open(output_path, 'w') as f:
                    json.dump(convert_types(results), f, indent=2)

                all_results[model_key] = results
                logger.info(f"Metrics recalculated for {model_key}")

            except Exception as e:
                logger.error(f"Failed to recalculate metrics for {model_key}: {e}")
                all_results[model_key] = {'error': str(e)}

        if progress_callback:
            progress_callback(1.0, "Recalculation complete!")

        return all_results

    def render_recalculate_all_panel(self):
        """Render the recalculate all metrics UI panel."""
        st.subheader("Recalculate All Metrics")

        st.info("""
        Recalculate metrics for all trained models using consistent parameters.
        This ensures fair comparison across all models with the same financial assumptions.
        """)

        # Get available models
        models = self.get_all_trained_models()

        if not models:
            st.warning("No trained models found. Please train models first using the 'Refresh Data & Retrain' tab.")
            return

        st.write(f"Found **{len(models)}** trained models:")

        # Display models in a table
        model_data = []
        for m in models:
            model_data.append({
                'Model': m['key'],
                'Has Predictions': 'Yes',
                'Has Results': 'Yes' if m['has_results'] else 'No'
            })

        st.dataframe(pd.DataFrame(model_data), use_container_width=True)

        # Parameters section
        st.markdown("### Evaluation Parameters")
        st.caption("These parameters are used consistently for all models to ensure fair comparison.")

        col1, col2, col3 = st.columns(3)

        with col1:
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=5.0,
                value=self.research_config.transaction_cost * 100,
                step=0.1,
                help="Cost per trade (bid-ask + slippage)",
                key="recalc_tx_cost"
            ) / 100

        with col2:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=self.research_config.risk_free_rate * 100,
                step=0.5,
                help="Annual risk-free rate",
                key="recalc_rf_rate"
            ) / 100

        with col3:
            periods_per_year = st.selectbox(
                "Trading Days/Year",
                [252, 365, 260],
                index=0,
                help="Trading periods per year",
                key="recalc_periods"
            )

        compute_rolling = st.checkbox(
            "Compute Rolling Window Analysis",
            value=True,
            help="Include rolling window stability metrics (slower but more comprehensive)",
            key="recalc_rolling"
        )

        # Recalculate button
        if st.button("Recalculate All Metrics", type="primary", use_container_width=True, key="recalc_all_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(p, msg):
                progress_bar.progress(min(p, 1.0))
                status_text.text(msg)

            try:
                results = self.recalculate_all_metrics(
                    transaction_cost=transaction_cost,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                    compute_rolling=compute_rolling,
                    progress_callback=update_progress
                )

                # Clear Streamlit caches
                st.cache_data.clear()

                # Display summary
                st.success(f"Recalculated metrics for {len(results)} models!")

                # Show comparison table
                st.markdown("### Results Summary")

                comparison_data = []
                for model_key, result in results.items():
                    if 'error' in result:
                        comparison_data.append({
                            'Model': model_key,
                            'Status': 'Error',
                            'Error': result['error']
                        })
                    else:
                        ml = result.get('ml_metrics', {})
                        fin = result.get('financial_metrics', {})
                        comparison_data.append({
                            'Model': model_key,
                            'Status': 'Success',
                            'MSE': f"{ml.get('mse', 0):.6f}",
                            'RMSE': f"{ml.get('rmse', 0):.6f}",
                            'R²': f"{ml.get('r2', 0):.4f}",
                            'Sharpe': f"{fin.get('sharpe_ratio', 0):.3f}",
                            'Max DD': f"{fin.get('max_drawdown', 0)*100:.2f}%",
                            'Dir Acc': f"{fin.get('directional_accuracy', 0)*100:.2f}%"
                        })

                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

                # Save comparison summary
                summary_path = self.results_dir / 'metrics_recalculation_summary.json'
                summary = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'transaction_cost': transaction_cost,
                        'risk_free_rate': risk_free_rate,
                        'periods_per_year': periods_per_year
                    },
                    'models_processed': len(results),
                    'successful': sum(1 for r in results.values() if 'error' not in r),
                    'failed': sum(1 for r in results.values() if 'error' in r)
                }

                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                st.info("Results saved. Refresh the page to see updated metrics in other dashboards.")

            except Exception as e:
                st.error(f"Recalculation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    def render_data_refresh_section(self):
        """Render the data refresh and retraining UI section."""
        st.subheader("Data Refresh & Model Retraining")

        st.info("""
        Fetch fresh data from yfinance and retrain models on the updated dataset.
        All models will be trained on the **same exact data splits** for fair comparison.
        """)

        try:
            from src.web.data_refresh_service import (
                DataRefreshService,
                PINN_CONFIGURATIONS,
                render_data_refresh_panel
            )

            render_data_refresh_panel()

        except ImportError as e:
            st.error(f"Could not load data refresh service: {e}")
            st.info("Please ensure all dependencies are installed.")

        except Exception as e:
            st.error(f"Error in data refresh section: {e}")
            import traceback
            st.code(traceback.format_exc())

    def get_available_models(self) -> List[str]:
        """Get list of models with available predictions."""
        available = []

        # Check for prediction files
        patterns = [
            self.results_dir.glob('*_predictions.npz'),
            (self.results_dir / 'pinn_comparison').glob('*_predictions.npz')
            if (self.results_dir / 'pinn_comparison').exists() else []
        ]

        for pattern in patterns:
            for path in pattern:
                # Extract model key from filename
                name = path.stem.replace('_predictions', '')
                name = name.replace('pinn_', '')
                if name not in available:
                    available.append(name)

        return sorted(available)

    def render_computation_panel(self):
        """Render the metrics computation UI panel."""
        st.subheader("Live Metrics Computation")

        st.info("""
        Compute financial metrics on-demand from prediction files or uploaded data.
        Results are cached for performance.
        """)

        # Main tabs for computation vs refresh vs recalculate all
        main_tab1, main_tab2, main_tab3 = st.tabs([
            "Compute Metrics",
            "Refresh Data & Retrain",
            "Recalculate All Metrics"
        ])

        with main_tab2:
            self.render_data_refresh_section()
            return None

        with main_tab3:
            self.render_recalculate_all_panel()
            return None

        with main_tab1:
            # Data source selection
            data_source = st.radio(
                "Data Source",
                ["From Saved Predictions", "Upload Custom Data"],
                horizontal=True
            )

            if data_source == "Upload Custom Data":
                return self._render_upload_panel()

            available_models = self.get_available_models()

            if not available_models:
                st.warning("""
                No prediction files found. Run model evaluation first:
                ```bash
                python compute_all_financial_metrics.py
                ```
                Or use "Upload Custom Data" to compute metrics from your own predictions.

                **Or use the "Refresh Data & Retrain" tab** to fetch fresh data and train models.
                """)
                return None

            # Model selection
            col1, col2 = st.columns([2, 1])

            with col1:
                selected_model = st.selectbox(
                    "Select Model",
                    available_models,
                    key="metrics_calc_model"
                )

            with col2:
                compute_rolling = st.checkbox(
                    "Include Rolling Metrics",
                    value=True,
                    help="Compute rolling window stability analysis"
                )

            # Configuration expander
            with st.expander("⚙️ Computation Parameters"):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    transaction_cost = st.number_input(
                        "Transaction Cost (%)",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.3,
                        step=0.1,
                        help="Cost per trade (bid-ask + slippage)"
                    ) / 100

                with col_b:
                    risk_free_rate = st.number_input(
                        "Risk-Free Rate (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.5,
                        help="Annual risk-free rate"
                    ) / 100

                with col_c:
                    periods_per_year = st.selectbox(
                        "Trading Days/Year",
                        [252, 365, 260],
                        index=0,
                        help="Trading periods per year"
                    )

            # Compute button
            if st.button("🚀 Compute Metrics", type="primary", use_container_width=True):
                return self._compute_and_display(
                    selected_model,
                    transaction_cost,
                    risk_free_rate,
                    periods_per_year,
                    compute_rolling
                )

            return None

    def _render_upload_panel(self):
        """Render panel for uploading custom prediction data."""
        st.markdown("### Upload Custom Predictions")

        st.markdown("""
        Upload a CSV or NPZ file containing predictions and targets.

        **Required columns/arrays:**
        - `predictions` or `predicted`: Model predictions
        - `targets` or `actual`: Actual values
        """)

        uploaded_file = st.file_uploader(
            "Upload predictions file",
            type=['csv', 'npz'],
            help="CSV or NPZ file with predictions and targets"
        )

        if uploaded_file is None:
            return None

        # Load data based on file type
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)

                # Find prediction column
                pred_col = None
                for col in ['predictions', 'predicted', 'pred', 'y_pred']:
                    if col in df.columns:
                        pred_col = col
                        break

                # Find target column
                target_col = None
                for col in ['targets', 'target', 'actual', 'y_true', 'y']:
                    if col in df.columns:
                        target_col = col
                        break

                if pred_col is None or target_col is None:
                    st.error(f"Could not find prediction/target columns. Available: {list(df.columns)}")
                    return None

                predictions = df[pred_col].values
                targets = df[target_col].values

            else:  # NPZ file
                data = np.load(uploaded_file)
                keys = list(data.keys())

                # Find prediction array
                pred_key = None
                for key in ['predictions', 'predicted', 'pred']:
                    if key in keys:
                        pred_key = key
                        break

                # Find target array
                target_key = None
                for key in ['targets', 'target', 'actual']:
                    if key in keys:
                        target_key = key
                        break

                if pred_key is None or target_key is None:
                    st.error(f"Could not find prediction/target arrays. Available: {keys}")
                    return None

                predictions = data[pred_key].flatten()
                targets = data[target_key].flatten()

            st.success(f"Loaded {len(predictions):,} samples")

        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

        # Configuration
        with st.expander("⚙️ Computation Parameters", expanded=True):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                transaction_cost = st.number_input(
                    "Transaction Cost (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.3,
                    step=0.1,
                    key="upload_tx_cost"
                ) / 100

            with col_b:
                risk_free_rate = st.number_input(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.0,
                    step=0.5,
                    key="upload_rf_rate"
                ) / 100

            with col_c:
                periods_per_year = st.selectbox(
                    "Trading Days/Year",
                    [252, 365, 260],
                    index=0,
                    key="upload_periods"
                )

            compute_rolling = st.checkbox(
                "Include Rolling Metrics",
                value=True,
                key="upload_rolling"
            )

        model_name = st.text_input("Model Name", value="Custom Model")

        if st.button("🚀 Compute Metrics", type="primary", use_container_width=True, key="upload_compute"):
            with st.spinner("Computing comprehensive metrics..."):
                results = compute_metrics_cached(
                    predictions=predictions,
                    targets=targets,
                    model_name=model_name,
                    transaction_cost=transaction_cost,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                    compute_rolling=compute_rolling
                )

            if results:
                self._display_results(results)
                return results

        return None

    def _compute_and_display(
        self,
        model_key: str,
        transaction_cost: float,
        risk_free_rate: float,
        periods_per_year: int,
        compute_rolling: bool
    ) -> Optional[Dict]:
        """Compute metrics and display results."""

        # Load predictions
        with st.spinner(f"Loading predictions for {model_key}..."):
            data = load_predictions_cached(model_key)

        if data is None:
            st.error(f"Could not load predictions for model: {model_key}")
            return None

        predictions = data['predictions']
        targets = data['targets']

        st.success(f"Loaded {len(predictions):,} samples from {data['source']}")

        # Compute metrics
        with st.spinner("Computing comprehensive metrics..."):
            results = compute_metrics_cached(
                predictions=predictions,
                targets=targets,
                model_name=model_key,
                transaction_cost=transaction_cost,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                compute_rolling=compute_rolling
            )

        if not results:
            st.error("Metrics computation failed.")
            return None

        # Display results
        self._display_results(results)

        return results

    def _display_results(self, results: Dict):
        """Display computed metrics in organized tabs."""
        st.markdown("---")
        st.subheader(f"📊 Results: {results.get('model_name', 'Model')}")

        ml_metrics = results.get('ml_metrics', {})
        fin_metrics = results.get('financial_metrics', {})
        rolling = results.get('rolling_metrics')

        # Quick summary cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            sharpe = fin_metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                delta="Good" if sharpe > 1 else "Low" if sharpe < 0.5 else None,
                delta_color="normal" if sharpe > 1 else "off"
            )
            dsr = fin_metrics.get('deflated_sharpe_ratio', 0)
            st.metric("Deflated Sharpe", f"{dsr:.3f}")

        with col2:
            max_dd = fin_metrics.get('max_drawdown', 0) * 100
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2f}%",
                delta="Acceptable" if max_dd > -20 else "High Risk",
                delta_color="normal" if max_dd > -20 else "inverse"
            )

        with col3:
            dir_acc = fin_metrics.get('directional_accuracy', 0) * 100
            st.metric(
                "Directional Accuracy",
                f"{dir_acc:.2f}%",
                delta="Above Random" if dir_acc > 50 else "Below Random",
                delta_color="normal" if dir_acc > 50 else "inverse"
            )

        with col4:
            r2 = ml_metrics.get('r2', 0)
            st.metric(
                "R² Score",
                f"{r2:.4f}",
                delta="Good Fit" if r2 > 0.8 else None
            )

        # Detailed tabs
        tab1, tab2, tab3 = st.tabs([
            "📈 ML Metrics",
            "💰 Financial Metrics",
            "📊 Rolling Analysis"
        ])

        with tab1:
            self._display_ml_metrics(ml_metrics)

        with tab2:
            self._display_financial_metrics(fin_metrics)

        with tab3:
            if rolling:
                self._display_rolling_metrics(rolling)
            else:
                st.info("Rolling metrics not computed. Enable 'Include Rolling Metrics' option.")

    def _display_ml_metrics(self, metrics: Dict):
        """Display ML metrics."""
        st.markdown("### Machine Learning Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MSE", f"{metrics.get('mse', 0):.6f}")
            st.metric("MAE", f"{metrics.get('mae', 0):.6f}")
            st.metric("RMSE", f"{metrics.get('rmse', 0):.6f}")

        with col2:
            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

        # Interpretation
        r2 = metrics.get('r2', 0)
        if r2 > 0.95:
            st.success("Excellent predictive accuracy (R² > 0.95)")
        elif r2 > 0.8:
            st.info("Good predictive accuracy (R² > 0.8)")
        elif r2 > 0.5:
            st.warning("Moderate predictive accuracy (R² 0.5-0.8)")
        else:
            st.error("Poor predictive accuracy (R² < 0.5)")

    def _display_financial_metrics(self, metrics: Dict):
        """Display financial metrics in organized categories."""

        # Risk-Adjusted Performance
        st.markdown("#### Risk-Adjusted Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        with col2:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
        with col3:
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")
        with col4:
            st.metric("Deflated Sharpe", f"{metrics.get('deflated_sharpe_ratio', 0):.3f}")

        # Capital Preservation
        st.markdown("#### Capital Preservation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
        with col2:
            st.metric("Drawdown Duration", f"{metrics.get('drawdown_duration', 0):.1f} days")
        with col3:
            st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")

        # Trading Viability
        st.markdown("#### Trading Viability")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0)*100:.2f}%")
        with col2:
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        with col3:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.2f}%")

        # Signal Quality
        st.markdown("#### Signal Quality")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dir. Accuracy", f"{metrics.get('directional_accuracy', 0)*100:.2f}%")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("Info Coefficient", f"{metrics.get('information_coefficient', 0):.3f}")

    def _display_rolling_metrics(self, rolling: Dict):
        """Display rolling window analysis."""
        st.markdown("### Rolling Window Stability Analysis")

        n_windows = rolling.get('n_windows', 0)
        stability = rolling.get('stability', {})

        st.info(f"Analysis based on {n_windows} overlapping windows")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Sharpe CV",
                f"{stability.get('sharpe_ratio_cv', 0):.3f}",
                help="Coefficient of Variation (lower = more stable)"
            )
            st.metric(
                "Sharpe Consistency",
                f"{stability.get('sharpe_ratio_consistency', 0)*100:.1f}%",
                help="% of windows with positive Sharpe"
            )

        with col2:
            st.metric(
                "Dir Acc Consistency",
                f"{stability.get('directional_accuracy_consistency', 0)*100:.1f}%",
                help="% of windows above 50% accuracy"
            )
            st.metric(
                "Mean Sharpe",
                f"{stability.get('sharpe_ratio_mean', 0):.3f}"
            )

        # Interpretation
        cv = stability.get('sharpe_ratio_cv', 1)
        if cv < 0.5:
            st.success("Low variance across windows - model is robust")
        elif cv < 1.0:
            st.info("Moderate variance - performance varies by market regime")
        else:
            st.warning("High variance - model may be overfitting or regime-dependent")


def render_metrics_computation_page():
    """
    Standalone page for metrics computation.

    Can be integrated into the main app or run separately.
    """
    st.set_page_config(
        page_title="Metrics Calculator",
        page_icon="🧮",
        layout="wide"
    )

    st.title("🧮 Financial Metrics Calculator")
    st.markdown("""
    Compute comprehensive financial metrics from model predictions.
    """)

    calculator = StreamlitMetricsCalculator()
    calculator.render_computation_panel()


if __name__ == "__main__":
    render_metrics_computation_page()
