"""
Systematic PINN Physics Configuration Comparison
=================================================

This script trains 6 distinct PINN variants to evaluate the impact of different
physics constraints on financial forecasting performance.

Configurations:
1. Baseline (Data-only): No physics constraints
2. Pure GBM (Trend): Geometric Brownian Motion only
3. Pure OU (Mean-Reversion): Ornstein-Uhlenbeck only
4. Pure Black-Scholes: No-arbitrage PDE only
5. GBM+OU Hybrid: Combined trend and mean-reversion
6. Global Constraint: All equations combined

Outputs:
- Individual model checkpoints (e.g., pinn_black_scholes.pt)
- Per-variant training logs with data_loss and physics_loss
- Violation scores (physics_loss / data_loss ratio)
- Comparison report CSV ranking all variants
- README_theory.md with financial justifications
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.utils.reproducibility import set_seed, log_system_info, get_device
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import FinancialDataset, create_dataloaders
from src.models.pinn import PINNModel
from src.training.trainer import Trainer

logger = get_logger(__name__)

# Physics configuration definitions
PINN_CONFIGURATIONS = {
    'baseline': {
        'name': 'Baseline (Data-only)',
        'description': 'Pure data-driven learning without physics constraints',
        'lambda_gbm': 0.0,
        'lambda_bs': 0.0,
        'lambda_ou': 0.0,
        'lambda_langevin': 0.0,
        'enable_physics': False,
        'theory': '''
**Baseline (Data-only)**

Mathematical Formulation:
    L = L_data = MSE(ŷ, y)

Financial Justification:
    - Represents pure machine learning approach to forecasting
    - No assumptions about market dynamics or price evolution
    - Serves as control group to measure physics constraint benefits
    - Maximizes flexibility but may overfit to historical patterns
    - Does not enforce any market equilibrium conditions

Use Case:
    - Benchmark for evaluating physics-informed improvements
    - Markets with non-traditional dynamics
    - When theoretical assumptions may not hold
'''
    },
    'gbm': {
        'name': 'Pure GBM (Trend)',
        'description': 'Geometric Brownian Motion - trend-following dynamics',
        'lambda_gbm': 0.1,
        'lambda_bs': 0.0,
        'lambda_ou': 0.0,
        'lambda_langevin': 0.0,
        'enable_physics': True,
        'theory': '''
**Pure GBM (Geometric Brownian Motion)**

Mathematical Formulation:
    dS = μS dt + σS dW
    L = L_data + λ_gbm · ||dS/dt - μS||²

Financial Justification:
    - Models exponential growth/decay with constant volatility
    - μ represents drift (expected return)
    - σ represents volatility (risk)
    - Assumes continuous compounding and log-normal price distribution
    - Foundation of Black-Scholes option pricing theory
    - No mean reversion - trends can persist indefinitely

Market Regimes:
    - Bull/bear markets with strong directional momentum
    - Growth stocks with sustained trends
    - Commodities in supply/demand imbalance
    - Markets without strong reverting forces

Limitations:
    - Cannot model mean reversion
    - Assumes constant volatility (unrealistic)
    - May amplify trend predictions excessively
'''
    },
    'ou': {
        'name': 'Pure OU (Mean-Reversion)',
        'description': 'Ornstein-Uhlenbeck - mean-reverting dynamics',
        'lambda_gbm': 0.0,
        'lambda_bs': 0.0,
        'lambda_ou': 0.1,
        'lambda_langevin': 0.0,
        'enable_physics': True,
        'theory': '''
**Pure OU (Ornstein-Uhlenbeck Process)**

Mathematical Formulation:
    dX = θ(μ - X)dt + σdW
    L = L_data + λ_ou · ||dX/dt - θ(μ - X)||²

Financial Justification:
    - Models mean-reverting processes (X = returns or spreads)
    - θ represents mean-reversion speed
    - μ represents long-term equilibrium level
    - Prices/returns pulled back to fundamental value
    - Used extensively in pairs trading and statistical arbitrage

Market Regimes:
    - Range-bound markets oscillating around fair value
    - Interest rates reverting to equilibrium
    - Volatility (VIX) reverting to long-term mean
    - Pair spreads in convergence trades
    - Markets with strong fundamental anchors

Limitations:
    - Cannot model sustained trends
    - Assumes constant equilibrium (μ may shift)
    - May underpredict during regime changes
'''
    },
    'black_scholes': {
        'name': 'Pure Black-Scholes',
        'description': 'No-arbitrage PDE constraint',
        'lambda_gbm': 0.0,
        'lambda_bs': 0.1,
        'lambda_ou': 0.0,
        'lambda_langevin': 0.0,
        'enable_physics': True,
        'theory': '''
**Pure Black-Scholes PDE**

Mathematical Formulation:
    ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
    L = L_data + λ_bs · ||PDE residual||²

Financial Justification:
    - Fundamental no-arbitrage condition in continuous time
    - V represents asset value evolving without arbitrage opportunities
    - Enforces risk-neutral pricing framework
    - Second-order term captures convexity (gamma) effects
    - First-order term captures directional exposure (delta)
    - Discount term enforces present value relationship

Market Assumptions:
    - Frictionless markets (no transaction costs)
    - Continuous trading possible
    - Risk-free rate r is constant
    - No arbitrage opportunities exist
    - Prices follow diffusion process

Use Cases:
    - Option pricing and hedging
    - Portfolio valuation under no-arbitrage
    - Markets with active arbitrageurs
    - Derivative modeling

Limitations:
    - Requires automatic differentiation for ∂²V/∂S²
    - Computationally expensive
    - Assumes market efficiency
    - May be too restrictive for illiquid assets
'''
    },
    'gbm_ou': {
        'name': 'GBM+OU Hybrid',
        'description': 'Combined trend and mean-reversion',
        'lambda_gbm': 0.05,
        'lambda_bs': 0.0,
        'lambda_ou': 0.05,
        'lambda_langevin': 0.0,
        'enable_physics': True,
        'theory': '''
**GBM+OU Hybrid**

Mathematical Formulation:
    L = L_data + λ_gbm·||GBM residual||² + λ_ou·||OU residual||²

Financial Justification:
    - Captures both trending and mean-reverting behavior
    - GBM component models medium-term trends
    - OU component models short-term corrections
    - Balances momentum with fundamental value
    - More realistic than pure trend or pure reversion

Market Regimes:
    - Normal market conditions with mixed dynamics
    - Stocks with both growth trends and value anchors
    - Markets alternating between trend and range
    - Commodities with long-term supply curves but short-term shocks

Interpretation:
    - λ_gbm/λ_ou ratio controls trend vs reversion strength
    - Equal weights (0.05/0.05) provide balanced dynamics
    - Model learns when to follow trends vs revert

Advantages:
    - Flexible across market regimes
    - Reduces overfitting to single dynamic
    - Better generalization to unseen conditions
'''
    },
    'global': {
        'name': 'Global Constraint',
        'description': 'All physics equations combined',
        'lambda_gbm': 0.05,
        'lambda_bs': 0.03,
        'lambda_ou': 0.05,
        'lambda_langevin': 0.02,
        'enable_physics': True,
        'theory': '''
**Global Constraint (All Equations)**

Mathematical Formulation:
    L = L_data + λ_gbm·||GBM||² + λ_bs·||BS-PDE||² + λ_ou·||OU||² + λ_L·||Langevin||²

Financial Justification:
    - Maximum physics regularization
    - GBM: Captures exponential trends
    - Black-Scholes: Enforces no-arbitrage
    - OU: Models mean reversion
    - Langevin: Models momentum and friction
    - Attempts to satisfy all constraints simultaneously

Hypothesis:
    - Real markets exhibit all dynamics at different timescales
    - Comprehensive constraints improve generalization
    - Physics acts as multi-scale regularization

Risks:
    - Constraints may conflict (trend vs reversion)
    - Over-constrained optimization may not converge
    - High computational cost (especially Black-Scholes)
    - May underfit if physics assumptions violated

Success Criteria:
    - Should achieve lowest violation score
    - May have slightly higher data loss
    - Should generalize best to unseen data
    - Convergence indicates market consistency with theory

Failure Modes:
    - If fails to converge: Constraints are contradictory
    - If violation score is high: Theory doesn't match data
    - If test error is high: Over-regularization
'''
    }
}


def prepare_data(config):
    """
    Fetch and prepare data for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader, feature_cols, input_dim)
    """
    logger.info("=" * 80)
    logger.info("DATA PREPARATION")
    logger.info("=" * 80)

    # Fetch data
    fetcher = DataFetcher(config)

    logger.info("Fetching stock data...")
    df = fetcher.fetch_and_store(
        tickers=config.data.tickers[:10],  # Use subset for faster training
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        force_refresh=False
    )

    if df.empty:
        logger.error("No data fetched! Exiting...")
        sys.exit(1)

    # Preprocess data
    preprocessor = DataPreprocessor(config)

    logger.info("Preprocessing data...")
    df_processed = preprocessor.process_and_store(df)

    # Define feature columns
    feature_cols = [
        'close', 'volume',
        'log_return', 'simple_return',
        'rolling_volatility_5', 'rolling_volatility_20',
        'momentum_5', 'momentum_20',
        'rsi_14', 'macd', 'macd_signal',
        'bollinger_upper', 'bollinger_lower', 'atr_14'
    ]

    # Filter available features
    feature_cols = [col for col in feature_cols if col in df_processed.columns]

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Temporal split
    train_df, val_df, test_df = preprocessor.split_temporal(df_processed)

    # Normalize features
    logger.info("Normalizing features...")
    train_df_norm, scalers = preprocessor.normalize_features(
        train_df, feature_cols, method='standard'
    )

    # Apply same normalization to val and test
    for ticker in val_df['ticker'].unique():
        if ticker in scalers:
            val_mask = val_df['ticker'] == ticker
            val_df.loc[val_mask, feature_cols] = scalers[ticker].transform(
                val_df.loc[val_mask, feature_cols]
            )

    for ticker in test_df['ticker'].unique():
        if ticker in scalers:
            test_mask = test_df['ticker'] == ticker
            test_df.loc[test_mask, feature_cols] = scalers[ticker].transform(
                test_df.loc[test_mask, feature_cols]
            )

    # Create sequences
    logger.info("Creating sequences...")

    X_train, y_train, tickers_train = preprocessor.create_sequences(
        train_df_norm, feature_cols, target_col='close',
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon
    )

    X_val, y_val, tickers_val = preprocessor.create_sequences(
        val_df, feature_cols, target_col='close',
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon
    )

    X_test, y_test, tickers_test = preprocessor.create_sequences(
        test_df, feature_cols, target_col='close',
        sequence_length=config.data.sequence_length,
        forecast_horizon=config.data.forecast_horizon
    )

    logger.info(f"Train sequences: {X_train.shape}")
    logger.info(f"Val sequences: {X_val.shape}")
    logger.info(f"Test sequences: {X_test.shape}")

    # Create datasets
    train_dataset = FinancialDataset(X_train, y_train, tickers_train)
    val_dataset = FinancialDataset(X_val, y_val, tickers_val)
    test_dataset = FinancialDataset(X_test, y_test, tickers_test)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.training.batch_size
    )

    return train_loader, val_loader, test_loader, feature_cols, len(feature_cols)


def train_single_variant(
    variant_key: str,
    variant_config: Dict,
    input_dim: int,
    train_loader,
    val_loader,
    test_loader,
    config,
    device,
    epochs: int
) -> Dict:
    """
    Train a single PINN variant

    Returns:
        Dictionary with training results and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING VARIANT: {variant_config['name']}")
    logger.info("=" * 80)
    logger.info(f"Description: {variant_config['description']}")
    logger.info(f"Configuration: λ_gbm={variant_config['lambda_gbm']}, "
                f"λ_bs={variant_config['lambda_bs']}, λ_ou={variant_config['lambda_ou']}, "
                f"λ_langevin={variant_config['lambda_langevin']}")

    # Create model
    model = PINNModel(
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        output_dim=1,
        dropout=config.model.dropout,
        base_model='lstm',
        lambda_gbm=variant_config['lambda_gbm'],
        lambda_bs=variant_config['lambda_bs'],
        lambda_ou=variant_config['lambda_ou'],
        lambda_langevin=variant_config['lambda_langevin']
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Train
    history = trainer.train(
        epochs=epochs,
        enable_physics=variant_config['enable_physics'],
        save_best=True,
        model_name=f"pinn_{variant_key}"
    )

    # Evaluate
    test_metrics = trainer.evaluate(enable_physics=variant_config['enable_physics'])

    # Calculate violation score
    violation_score = calculate_violation_score(history)

    # Compile results
    results = {
        'variant_key': variant_key,
        'variant_name': variant_config['name'],
        'configuration': {
            'lambda_gbm': variant_config['lambda_gbm'],
            'lambda_bs': variant_config['lambda_bs'],
            'lambda_ou': variant_config['lambda_ou'],
            'lambda_langevin': variant_config['lambda_langevin'],
            'enable_physics': variant_config['enable_physics']
        },
        'test_metrics': test_metrics,
        'history': history,
        'violation_score': violation_score,
        'model_path': config.project_root / 'models' / f'pinn_{variant_key}_best.pt'
    }

    logger.info(f"\n✓ {variant_config['name']} training complete")
    logger.info(f"  Test MSE: {test_metrics.get('mse', 0):.6f}")
    logger.info(f"  Test MAE: {test_metrics.get('mae', 0):.6f}")
    logger.info(f"  Violation Score: {violation_score:.6f}")

    return results


def calculate_violation_score(history: Dict) -> float:
    """
    Calculate theoretical violation score

    Violation Score = L_physics / (L_data + ε)

    Lower score = better adherence to physics constraints
    """
    epsilon = 1e-8

    # Get final epoch losses
    if not history or 'train_loss' not in history:
        return 0.0

    # Try to get physics loss from history
    if 'train_physics_loss' in history and history['train_physics_loss']:
        final_physics_loss = history['train_physics_loss'][-1]
    else:
        final_physics_loss = 0.0

    # Get data loss
    if 'train_data_loss' in history and history['train_data_loss']:
        final_data_loss = history['train_data_loss'][-1]
    else:
        # If no separate data loss, use total loss
        final_data_loss = history['train_loss'][-1] if history['train_loss'] else epsilon

    violation_score = final_physics_loss / (final_data_loss + epsilon)

    return violation_score


def generate_theory_readme(results: List[Dict], output_path: Path):
    """
    Generate README_theory.md with financial justifications
    """
    logger.info("Generating README_theory.md...")

    content = """# PINN Physics Configuration Theory
## Financial Justifications for Mathematical Constraints

This document provides the theoretical foundation for each physics-informed neural network
configuration tested in this study. Each constraint embeds specific assumptions about
market dynamics into the learning process.

---

"""

    for result in results:
        variant_key = result['variant_key']
        config = PINN_CONFIGURATIONS[variant_key]

        content += f"## {config['name']}\n\n"
        content += config['theory'] + "\n\n"

        # Add empirical results
        test_metrics = result['test_metrics']
        violation_score = result['violation_score']

        content += "### Empirical Results\n\n"
        content += f"- **Test MSE**: {test_metrics.get('mse', 0):.6f}\n"
        content += f"- **Test MAE**: {test_metrics.get('mae', 0):.6f}\n"
        content += f"- **Test R²**: {test_metrics.get('r2', 0):.6f}\n"
        content += f"- **Violation Score**: {violation_score:.6f}\n"
        content += f"- **Model Path**: `{result['model_path']}`\n\n"
        content += "---\n\n"

    # Add interpretation guide
    content += """
## Interpreting Results

### Violation Score
The violation score measures how well the model adheres to physics constraints:

$$\\text{Violation Score} = \\frac{\\mathcal{L}_{physics}}{\\mathcal{L}_{data} + \\epsilon}$$

- **Lower score**: Model satisfies physics constraints with minimal penalty
- **Higher score**: Model violates physics assumptions significantly
- **Zero score**: Baseline model (no physics constraints)

### Model Ranking Criteria

1. **Primary**: Test MSE (empirical accuracy)
2. **Secondary**: Violation Score (theoretical consistency)
3. **Tertiary**: Generalization gap (train vs test)

### Convergence Analysis

If **Global Constraint** fails to converge compared to **Pure Black-Scholes**:

1. **Contradictory Constraints**:
   - GBM assumes perpetual trends (dS ~ μS)
   - OU assumes mean reversion (dX ~ -θX)
   - These forces oppose each other, creating optimization conflicts

2. **Over-Regularization**:
   - Too many constraints restrict the solution space excessively
   - Model cannot fit data while satisfying all physics equations
   - Similar to L2 regularization that's too strong

3. **Timescale Mismatch**:
   - Different physics operate at different timescales
   - GBM: Medium-term (weeks/months)
   - OU: Short-term (days)
   - Black-Scholes: Instantaneous (continuous time)
   - Simultaneous enforcement may be incompatible

4. **Data-Theory Mismatch**:
   - Real market data may not satisfy all theoretical assumptions
   - Frictions, discrete trading, transaction costs violate Black-Scholes
   - Non-stationary parameters (μ, σ, θ change over time)

### Recommendations

- **Trending Markets**: Use Pure GBM or GBM+OU Hybrid
- **Range-Bound Markets**: Use Pure OU
- **Derivative Pricing**: Use Pure Black-Scholes
- **General Forecasting**: Start with GBM+OU Hybrid (balanced)
- **Maximum Regularization**: Use Global Constraint (if converges)
- **No Assumptions**: Use Baseline (data-only)

### Further Research

- Investigate adaptive physics weights (time-varying λ parameters)
- Test regime-switching between configurations
- Explore hierarchical constraints (coarse + fine timescales)
- Add market microstructure constraints (bid-ask spread, order flow)
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    logger.info(f"Theory README saved to {output_path}")


def generate_comparison_report(results: List[Dict], output_path: Path):
    """
    Generate comparison_report.csv ranking all variants
    """
    logger.info("Generating comparison_report.csv...")

    rows = []
    for result in results:
        test_metrics = result['test_metrics']

        row = {
            'Variant': result['variant_name'],
            'Variant_Key': result['variant_key'],
            'Test_MSE': test_metrics.get('mse', np.nan),
            'Test_MAE': test_metrics.get('mae', np.nan),
            'Test_RMSE': np.sqrt(test_metrics.get('mse', 0)),
            'Test_R2': test_metrics.get('r2', np.nan),
            'Violation_Score': result['violation_score'],
            'Lambda_GBM': result['configuration']['lambda_gbm'],
            'Lambda_BS': result['configuration']['lambda_bs'],
            'Lambda_OU': result['configuration']['lambda_ou'],
            'Lambda_Langevin': result['configuration']['lambda_langevin'],
            'Physics_Enabled': result['configuration']['enable_physics'],
            'Model_Path': str(result['model_path'])
        }

        rows.append(row)

    # Create DataFrame and rank
    df = pd.DataFrame(rows)

    # Rank by test MSE (lower is better)
    df['MSE_Rank'] = df['Test_MSE'].rank(method='min')

    # Rank by violation score (lower is better, but only for physics-enabled models)
    df['Violation_Rank'] = df['Violation_Score'].rank(method='min')

    # Combined rank (weighted average)
    df['Combined_Rank'] = 0.7 * df['MSE_Rank'] + 0.3 * df['Violation_Rank']

    # Sort by combined rank
    df = df.sort_values('Combined_Rank')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Comparison report saved to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\n{df[['Variant', 'Test_MSE', 'Violation_Score', 'Combined_Rank']].to_string(index=False)}\n")

    return df


def save_detailed_results(results: List[Dict], output_path: Path):
    """
    Save detailed results as JSON
    """
    logger.info("Saving detailed results...")

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    results_serializable = convert_types(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"Detailed results saved to {output_path}")


def main(epochs: int = 100, variants: List[str] = None):
    """
    Main function to run systematic PINN comparison

    Args:
        epochs: Number of training epochs per variant
        variants: List of variant keys to train (None = all)
    """
    # Setup logging
    ensure_logger_initialized()

    logger.info("=" * 80)
    logger.info("SYSTEMATIC PINN PHYSICS CONFIGURATION COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Epochs per variant: {epochs}")

    # Load configuration
    config = get_config()

    # Set random seed
    set_seed(config.training.random_seed)

    # Log system info
    log_system_info()

    # Get device
    device = get_device(prefer_cuda=(config.training.device == 'cuda'))

    # Prepare data (shared across all variants)
    train_loader, val_loader, test_loader, feature_cols, input_dim = prepare_data(config)

    # Determine which variants to train
    if variants is None:
        variants_to_train = list(PINN_CONFIGURATIONS.keys())
    else:
        variants_to_train = [v for v in variants if v in PINN_CONFIGURATIONS]

    logger.info(f"\nTraining {len(variants_to_train)} variants: {variants_to_train}")

    # Train each variant
    all_results = []

    for i, variant_key in enumerate(variants_to_train, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"VARIANT {i}/{len(variants_to_train)}: {variant_key}")
        logger.info(f"{'='*80}")

        variant_config = PINN_CONFIGURATIONS[variant_key]

        try:
            result = train_single_variant(
                variant_key=variant_key,
                variant_config=variant_config,
                input_dim=input_dim,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=device,
                epochs=epochs
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"Failed to train {variant_key}: {e}")
            logger.exception(e)
            continue

    # Generate outputs
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 80)

    results_dir = config.project_root / 'results' / 'pinn_comparison'

    # Theory README
    theory_path = results_dir / 'README_theory.md'
    generate_theory_readme(all_results, theory_path)

    # Comparison CSV
    csv_path = results_dir / 'comparison_report.csv'
    comparison_df = generate_comparison_report(all_results, csv_path)

    # Detailed JSON
    json_path = results_dir / 'detailed_results.json'
    save_detailed_results(all_results, json_path)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEMATIC COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {results_dir}")
    logger.info(f"  - Theory: {theory_path}")
    logger.info(f"  - Comparison: {csv_path}")
    logger.info(f"  - Detailed: {json_path}")
    logger.info(f"\nBest model (by MSE): {comparison_df.iloc[0]['Variant']}")
    logger.info(f"Best MSE: {comparison_df.iloc[0]['Test_MSE']:.6f}")
    logger.info(f"Best violation score: {comparison_df.iloc[0]['Violation_Score']:.6f}")

    return all_results, comparison_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Systematic PINN Physics Configuration Comparison'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs per variant'
    )
    parser.add_argument(
        '--variants',
        type=str,
        nargs='+',
        default=None,
        choices=list(PINN_CONFIGURATIONS.keys()),
        help='Specific variants to train (default: all)'
    )

    args = parser.parse_args()

    try:
        main(epochs=args.epochs, variants=args.variants)
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        sys.exit(1)
