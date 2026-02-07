#!/usr/bin/env python3
"""
Re-compute Financial Metrics for All Models

This script re-evaluates all trained models using the unified evaluation pipeline
to ensure consistent metrics across all models.

Fixes addressed:
- BUG #9: Inconsistent metric sources across evaluation paths
- Ensures all models use the same compute_strategy_returns() function
- Validates all metrics and replaces invalid values (inf/nan)

Usage:
    python recompute_metrics.py
    python recompute_metrics.py --models pinn_gbm pinn_ou
    python recompute_metrics.py --validate-only
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config
from src.utils.logger import get_logger, ensure_logger_initialized
from src.evaluation.financial_metrics import (
    FinancialMetrics,
    compute_strategy_returns,
    validate_metrics
)
from src.evaluation.metrics import calculate_metrics, calculate_financial_metrics
from src.models.model_registry import get_model_registry

ensure_logger_initialized()
logger = get_logger(__name__)


def load_predictions(model_key: str, results_dir: Path) -> tuple:
    """Load predictions and targets from results files"""
    # Try different result file patterns
    patterns = [
        results_dir / f'{model_key}_results.json',
        results_dir / f'pinn_{model_key}_results.json',
        results_dir / 'pinn_comparison' / f'{model_key}_results.json',
    ]

    for pattern in patterns:
        if pattern.exists():
            try:
                with open(pattern) as f:
                    data = json.load(f)

                # Try to get predictions and targets
                predictions = data.get('predictions')
                targets = data.get('targets', data.get('actuals'))

                if predictions is not None and targets is not None:
                    return (
                        np.array(predictions),
                        np.array(targets),
                        pattern
                    )
            except Exception as e:
                logger.warning(f"Failed to load {pattern}: {e}")

    return None, None, None


def recompute_model_metrics(model_key: str, predictions: np.ndarray,
                           targets: np.ndarray, result_path: Path) -> dict:
    """Recompute all metrics for a single model"""
    logger.info(f"Recomputing metrics for {model_key}")

    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Calculate ML metrics
    ml_metrics = calculate_metrics(targets, predictions)

    # Compute strategy returns using unified function
    strategy_returns = compute_strategy_returns(predictions, targets)

    # Calculate financial metrics
    financial_metrics = FinancialMetrics.compute_all_metrics(
        returns=strategy_returns,
        predictions=predictions,
        targets=targets,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    # Validate metrics
    validation = validate_metrics(financial_metrics)

    if not validation['is_valid']:
        logger.warning(f"Validation errors for {model_key}: {validation['errors']}")

    if validation['warnings']:
        logger.info(f"Validation warnings for {model_key}: {validation['warnings']}")

    # Combine all metrics
    result = {
        'model_key': model_key,
        'ml_metrics': ml_metrics,
        'financial_metrics': financial_metrics,
        'strategy_returns_stats': {
            'mean': float(np.mean(strategy_returns)),
            'std': float(np.std(strategy_returns)),
            'min': float(np.min(strategy_returns)),
            'max': float(np.max(strategy_returns)),
            'count': len(strategy_returns)
        },
        'validation': validation,
        'recomputed': True
    }

    return result


def update_result_file(result_path: Path, new_metrics: dict):
    """Update existing result file with recomputed metrics"""
    try:
        with open(result_path) as f:
            existing_data = json.load(f)

        # Update with new metrics
        existing_data['ml_metrics'] = new_metrics['ml_metrics']
        existing_data['financial_metrics'] = new_metrics['financial_metrics']
        existing_data['strategy_returns_stats'] = new_metrics['strategy_returns_stats']
        existing_data['validation'] = new_metrics['validation']
        existing_data['recomputed'] = True

        # Write back
        with open(result_path, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)

        logger.info(f"Updated {result_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to update {result_path}: {e}")
        return False


def validate_all_results(results_dir: Path):
    """Validate all result files without modifying them"""
    logger.info("Validating all result files...")

    issues = []

    for result_file in results_dir.glob('*_results.json'):
        try:
            with open(result_file) as f:
                data = json.load(f)

            fm = data.get('financial_metrics', {})

            # Check for critical issues
            max_dd = fm.get('max_drawdown', 0)
            if max_dd < -1.0:
                issues.append(f"{result_file.name}: max_drawdown = {max_dd:.2%} (exceeds -100%)")

            total_return = fm.get('total_return', 0)
            if abs(total_return) > 1e6:
                issues.append(f"{result_file.name}: total_return = {total_return:.2e} (overflow)")

            # Check for inf/nan
            for key, value in fm.items():
                if isinstance(value, (int, float)):
                    if np.isinf(value):
                        issues.append(f"{result_file.name}: {key} is infinite")
                    elif np.isnan(value):
                        issues.append(f"{result_file.name}: {key} is NaN")

        except Exception as e:
            issues.append(f"{result_file.name}: Failed to load - {e}")

    if issues:
        logger.warning("Validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All result files passed validation")

    return issues


def main():
    parser = argparse.ArgumentParser(description='Recompute financial metrics for all models')
    parser.add_argument('--models', nargs='+', help='Specific models to recompute')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate without recomputing')
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation even if already done')
    args = parser.parse_args()

    config = get_config()
    results_dir = config.project_root / 'results'

    # Validate only mode
    if args.validate_only:
        issues = validate_all_results(results_dir)
        sys.exit(1 if issues else 0)

    # Get models to process
    if args.models:
        model_keys = args.models
    else:
        # Get all models from registry
        registry = get_model_registry(config.project_root)
        all_models = registry.get_all_models()
        model_keys = [key for key, model in all_models.items() if model.trained]

    logger.info(f"Processing {len(model_keys)} models: {model_keys}")

    results_summary = {
        'processed': 0,
        'updated': 0,
        'failed': 0,
        'skipped': 0,
        'details': {}
    }

    for model_key in model_keys:
        predictions, targets, result_path = load_predictions(model_key, results_dir)

        if predictions is None or targets is None:
            logger.warning(f"No predictions found for {model_key}")
            results_summary['skipped'] += 1
            results_summary['details'][model_key] = 'No predictions found'
            continue

        # Skip if already recomputed (unless --force)
        if result_path:
            try:
                with open(result_path) as f:
                    existing = json.load(f)
                if existing.get('recomputed') and not args.force:
                    logger.info(f"Skipping {model_key} (already recomputed)")
                    results_summary['skipped'] += 1
                    results_summary['details'][model_key] = 'Already recomputed'
                    continue
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.debug(f"Could not check recomputed status for {model_key}: {e}")

        results_summary['processed'] += 1

        # Recompute metrics
        new_metrics = recompute_model_metrics(model_key, predictions, targets, result_path)

        # Update result file
        if result_path:
            if update_result_file(result_path, new_metrics):
                results_summary['updated'] += 1
                results_summary['details'][model_key] = 'Updated successfully'
            else:
                results_summary['failed'] += 1
                results_summary['details'][model_key] = 'Update failed'
        else:
            results_summary['failed'] += 1
            results_summary['details'][model_key] = 'No result file to update'

    # Summary
    print("\n" + "=" * 60)
    print("RECOMPUTATION SUMMARY")
    print("=" * 60)
    print(f"Total models processed: {results_summary['processed']}")
    print(f"Successfully updated: {results_summary['updated']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"Skipped: {results_summary['skipped']}")
    print("=" * 60)

    # Final validation
    print("\nRunning final validation...")
    issues = validate_all_results(results_dir)

    if issues:
        print(f"\n⚠️  {len(issues)} issues found. See logs for details.")
    else:
        print("\n✅ All metrics validated successfully!")


if __name__ == '__main__':
    main()
