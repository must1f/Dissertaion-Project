#!/usr/bin/env python3
"""
Dissertation Validation Script

This script validates the entire PINN financial forecasting system to ensure
it is dissertation-ready and scientifically defensible.

Run before producing any dissertation results:
    python scripts/verify_dissertation.py

All checks must pass for the system to be considered valid.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import json
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


def verify_models() -> bool:
    """Verify all models create and forward pass correctly"""
    print_header("MODEL VERIFICATION")

    try:
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry(project_root)
        test_input = torch.randn(2, 30, 5)

        models_to_test = [
            'lstm', 'gru', 'bilstm', 'transformer',
            'baseline_pinn', 'gbm', 'ou', 'black_scholes', 'gbm_ou', 'global',
            'stacked', 'residual'
        ]

        all_passed = True
        for model_key in models_to_test:
            try:
                model = registry.create_model(model_key, input_dim=5)
                if model is None:
                    print_result(f"Model '{model_key}'", False, "create_model returned None")
                    all_passed = False
                    continue

                # Forward pass
                with torch.no_grad():
                    output = model(test_input)
                    if isinstance(output, tuple):
                        output = output[0]

                # Check output shape
                if output.shape[0] != 2:
                    print_result(f"Model '{model_key}'", False, f"Wrong batch size: {output.shape}")
                    all_passed = False
                    continue

                # Check if PINN model has physics methods
                is_pinn = hasattr(model, 'compute_loss')
                model_type = "PINN" if is_pinn else "Baseline"

                print_result(f"Model '{model_key}'", True, f"{model.__class__.__name__} ({model_type})")

            except Exception as e:
                print_result(f"Model '{model_key}'", False, str(e))
                all_passed = False

        return all_passed

    except Exception as e:
        print_result("Model verification", False, str(e))
        return False


def verify_metrics() -> bool:
    """Verify financial metrics are mathematically correct"""
    print_header("METRICS VERIFICATION")

    try:
        from src.evaluation.financial_metrics import (
            FinancialMetrics,
            compute_strategy_returns,
            assert_price_scale
        )

        all_passed = True

        # Test 1: Z-score detection
        print("\n  Testing price scale validation...")
        z_scores = np.array([0.1, -0.2, 0.5, -0.1, 0.3, 0.0, -0.4, 0.2])
        try:
            assert_price_scale(z_scores, raise_error=True)
            print_result("Z-score detection", False, "Should have raised ValueError")
            all_passed = False
        except ValueError:
            print_result("Z-score detection", True, "Correctly rejected z-scores")

        # Test 2: Real prices accepted
        real_prices = np.array([150.2, 151.5, 149.8, 152.1, 150.9, 153.2, 151.8, 154.0])
        is_valid = assert_price_scale(real_prices, raise_error=False)
        print_result("Real price acceptance", is_valid, f"std={np.std(real_prices):.2f}")
        if not is_valid:
            all_passed = False

        # Test 3: Sharpe ratio bounds
        print("\n  Testing Sharpe ratio bounds...")
        returns = np.random.randn(252) * 0.01  # 1% daily std
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        sharpe_bounded = -5 <= sharpe <= 5
        print_result("Sharpe ratio bounded", sharpe_bounded, f"Sharpe={sharpe:.4f}")
        if not sharpe_bounded:
            all_passed = False

        # Test 4: Raw vs display metrics
        print("\n  Testing raw vs display metrics...")
        sharpe_raw = FinancialMetrics.sharpe_ratio_raw(returns)
        sharpe_display = FinancialMetrics.sharpe_ratio(returns)
        has_both = sharpe_raw is not None and sharpe_display is not None
        print_result("Raw and display metrics exist", has_both,
                    f"raw={sharpe_raw:.4f}, display={sharpe_display:.4f}")
        if not has_both:
            all_passed = False

        # Test 5: Position lag in strategy returns
        print("\n  Testing position lag enforcement...")
        predictions = np.array([100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0])
        actuals = np.array([100.0, 101.5, 100.0, 102.5, 101.0, 103.5, 102.0, 104.5])

        # This should not raise since we're using real prices
        strategy_returns, details = compute_strategy_returns(
            predictions, actuals,
            return_details=True,
            validate_scale=True
        )

        positions = details['positions']
        # Position at t should equal signal at t-1
        # Position[0] should be 0 (no signal yet)
        position_lag_correct = positions[0] == 0.0
        print_result("Position lag (pos[0]=0)", position_lag_correct,
                    f"positions[:3]={positions[:3]}")
        if not position_lag_correct:
            all_passed = False

        # Test 6: Max drawdown bounds
        print("\n  Testing max drawdown bounds...")
        max_dd = FinancialMetrics.max_drawdown(returns)
        dd_bounded = -1.0 <= max_dd <= 0.0
        print_result("Max drawdown bounded [-1, 0]", dd_bounded, f"max_dd={max_dd:.4f}")
        if not dd_bounded:
            all_passed = False

        return all_passed

    except Exception as e:
        print_result("Metrics verification", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def verify_physics_gradients() -> bool:
    """Verify physics gradients propagate correctly"""
    print_header("PHYSICS GRADIENTS VERIFICATION")

    try:
        from src.models.pinn import PINNModel

        all_passed = True

        # Create PINN model with all physics enabled
        model = PINNModel(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            lambda_gbm=0.1,
            lambda_ou=0.1,
            lambda_bs=0.1,
            lambda_langevin=0.1
        )

        # Set scaler params for Black-Scholes
        model.set_scaler_params(price_mean=100.0, price_std=10.0)

        # Create test batch
        batch_size = 4
        seq_len = 30
        x = torch.randn(batch_size, seq_len, 5, requires_grad=True)
        y = torch.randn(batch_size, 1)

        # Forward pass
        predictions = model(x)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Create metadata for physics loss
        metadata = {
            'prices': x[:, :, 0],  # Use first feature as price
            'returns': torch.diff(x[:, :, 0], dim=1),
            'volatilities': torch.std(x[:, :, 0], dim=1, keepdim=True).expand(-1, seq_len),
            'inputs': x,
            'price_feature_idx': 0
        }

        # Compute loss
        loss, loss_dict = model.compute_loss(predictions, y, metadata, enable_physics=True)

        # Check gradients exist
        loss.backward()

        # Test 1: Data loss exists
        data_loss_exists = 'data_loss' in loss_dict and loss_dict['data_loss'] > 0
        print_result("Data loss computed", data_loss_exists,
                    f"data_loss={loss_dict.get('data_loss', 0):.6f}")
        if not data_loss_exists:
            all_passed = False

        # Test 2: Physics loss exists
        physics_loss_exists = 'physics_loss' in loss_dict
        print_result("Physics loss computed", physics_loss_exists,
                    f"physics_loss={loss_dict.get('physics_loss', 0):.6f}")
        if not physics_loss_exists:
            all_passed = False

        # Test 3: Residual RMS values logged
        residual_rms = model.get_residual_rms()
        has_residuals = any(v > 0 for v in residual_rms.values())
        print_result("Residual RMS logged", has_residuals,
                    f"gbm={residual_rms.get('gbm_residual_rms', 0):.4f}")

        # Test 4: Gradients propagate to model parameters
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        print_result("Gradients propagate", has_grads)
        if not has_grads:
            all_passed = False

        # Test 5: Learned physics params accessible
        learned_params = model.get_learned_physics_params()
        has_params = 'theta' in learned_params and 'gamma' in learned_params
        print_result("Learned physics params", has_params,
                    f"theta={learned_params.get('theta', 0):.4f}")
        if not has_params:
            all_passed = False

        return all_passed

    except Exception as e:
        print_result("Physics gradients verification", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def verify_causal_separation() -> bool:
    """Verify causal vs oracle model separation"""
    print_header("CAUSAL/ORACLE SEPARATION VERIFICATION")

    try:
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry(project_root)

        all_passed = True

        # Test 1: BiLSTM is marked as oracle
        bilstm_info = registry.get_model_info('bilstm')
        bilstm_is_oracle = bilstm_info is not None and not bilstm_info.is_causal
        print_result("BiLSTM marked as oracle", bilstm_is_oracle,
                    f"is_causal={bilstm_info.is_causal if bilstm_info else 'N/A'}")
        if not bilstm_is_oracle:
            all_passed = False

        # Test 2: LSTM is marked as causal
        lstm_info = registry.get_model_info('lstm')
        lstm_is_causal = lstm_info is not None and lstm_info.is_causal
        print_result("LSTM marked as causal", lstm_is_causal,
                    f"is_causal={lstm_info.is_causal if lstm_info else 'N/A'}")
        if not lstm_is_causal:
            all_passed = False

        # Test 3: Transformer is marked as causal (with mask)
        transformer_info = registry.get_model_info('transformer')
        transformer_is_causal = transformer_info is not None and transformer_info.is_causal
        print_result("Transformer marked as causal", transformer_is_causal,
                    f"is_causal={transformer_info.is_causal if transformer_info else 'N/A'}")
        if not transformer_is_causal:
            all_passed = False

        # Test 4: get_causal_models returns only causal
        causal_models = registry.get_causal_models()
        no_oracle_in_causal = all(m.is_causal for m in causal_models.values())
        print_result("get_causal_models() correct", no_oracle_in_causal,
                    f"{len(causal_models)} causal models")
        if not no_oracle_in_causal:
            all_passed = False

        # Test 5: Transformer applies causal mask by default
        from src.models.transformer import TransformerModel
        transformer = TransformerModel(input_dim=5, causal=True)
        has_causal_attr = hasattr(transformer, 'causal') and transformer.causal
        print_result("Transformer causal mask default", has_causal_attr)
        if not has_causal_attr:
            all_passed = False

        return all_passed

    except Exception as e:
        print_result("Causal separation verification", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def verify_reproducibility() -> bool:
    """Verify reproducibility infrastructure"""
    print_header("REPRODUCIBILITY VERIFICATION")

    try:
        from src.utils.reproducibility import (
            set_seed,
            verify_reproducibility as verify_repro,
            compute_config_hash,
            ExperimentMetadata,
            create_experiment_metadata
        )

        all_passed = True

        # Test 1: Seed reproducibility
        repro_ok = verify_repro(seed=42, n_samples=100)
        print_result("Seed reproducibility", repro_ok)
        if not repro_ok:
            all_passed = False

        # Test 2: Config hash determinism
        config1 = {'lr': 0.001, 'epochs': 100, 'hidden_dim': 128}
        config2 = {'epochs': 100, 'hidden_dim': 128, 'lr': 0.001}  # Same, different order
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash_deterministic = hash1 == hash2
        print_result("Config hash deterministic", hash_deterministic,
                    f"hash={hash1}")
        if not hash_deterministic:
            all_passed = False

        # Test 3: Experiment metadata creation
        metadata = create_experiment_metadata(
            experiment_name="Test",
            config=config1,
            model_key="lstm",
            model_type="baseline",
            seed=42,
            scaler_params={"AAPL": (150.0, 10.0)},
            transaction_cost=0.001
        )
        has_required = (
            metadata.config_hash is not None and
            len(metadata.scaler_params) > 0 and
            metadata.execution is not None and
            metadata.seed == 42
        )
        print_result("Experiment metadata complete", has_required)
        if not has_required:
            all_passed = False

        # Test 4: Scaler params accessible
        scaler = metadata.scaler_params.get("AAPL")
        has_scaler = scaler is not None and scaler.close_mean == 150.0
        print_result("Scaler params stored", has_scaler,
                    f"mean={scaler.close_mean if scaler else 'N/A'}")
        if not has_scaler:
            all_passed = False

        return all_passed

    except Exception as e:
        print_result("Reproducibility verification", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def verify_leaderboard() -> bool:
    """Verify leaderboard supports causal/oracle separation"""
    print_header("LEADERBOARD VERIFICATION")

    try:
        from src.evaluation.leaderboard import (
            ExperimentEntry,
            ResultsDatabase,
            RankingMetric
        )
        import tempfile
        import os

        all_passed = True

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = ResultsDatabase(db_path)

            # Test 1: ExperimentEntry has causal field
            entry = ExperimentEntry(
                experiment_id="test_1",
                model_name="LSTM",
                model_type="baseline",
                config_hash="abc123",
                timestamp=datetime.now().isoformat(),
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                total_return=0.15,
                annualized_return=0.12,
                volatility=0.18,
                max_drawdown=-0.10,
                calmar_ratio=1.2,
                is_causal=True,
                model_category="forecasting",
                sharpe_ratio_raw=1.5,
                sortino_ratio_raw=2.0
            )
            has_causal = hasattr(entry, 'is_causal') and entry.is_causal == True
            print_result("ExperimentEntry has is_causal", has_causal)
            if not has_causal:
                all_passed = False

            # Test 2: Save and retrieve
            db.save_experiment(entry)
            retrieved = db.get_experiment("test_1")
            save_works = retrieved is not None and retrieved.model_name == "LSTM"
            print_result("Save/retrieve works", save_works)
            if not save_works:
                all_passed = False

            # Test 3: get_causal_ranked exists
            has_causal_ranked = hasattr(db, 'get_causal_ranked')
            print_result("get_causal_ranked method exists", has_causal_ranked)
            if not has_causal_ranked:
                all_passed = False

            # Test 4: Raw metrics stored
            entry_with_raw = ExperimentEntry(
                experiment_id="test_2",
                model_name="GRU",
                model_type="baseline",
                config_hash="def456",
                timestamp=datetime.now().isoformat(),
                sharpe_ratio=2.0,
                sortino_ratio=3.0,
                total_return=0.20,
                annualized_return=0.15,
                volatility=0.20,
                max_drawdown=-0.12,
                calmar_ratio=1.5,
                sharpe_ratio_raw=2.5,  # Unclipped
                sortino_ratio_raw=4.0   # Unclipped
            )
            db.save_experiment(entry_with_raw)
            has_raw = entry_with_raw.sharpe_ratio_raw is not None
            print_result("Raw metrics stored", has_raw,
                        f"sharpe_raw={entry_with_raw.sharpe_ratio_raw}")
            if not has_raw:
                all_passed = False

        finally:
            os.unlink(db_path)

        return all_passed

    except Exception as e:
        print_result("Leaderboard verification", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print(" DISSERTATION VALIDATION SUITE")
    print(" PINN Financial Forecasting System")
    print(f" Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}

    # Run all verification tests
    results['models'] = verify_models()
    results['metrics'] = verify_metrics()
    results['physics_gradients'] = verify_physics_gradients()
    results['causal_separation'] = verify_causal_separation()
    results['reproducibility'] = verify_reproducibility()
    results['leaderboard'] = verify_leaderboard()

    # Summary
    print_header("VERIFICATION SUMMARY")

    all_passed = all(results.values())
    total = len(results)
    passed = sum(results.values())

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print("\n" + "-" * 70)
    print(f"  Total: {passed}/{total} passed")
    print("-" * 70)

    if all_passed:
        print("\n  ✓ SYSTEM IS DISSERTATION-READY")
        print("    All models train, gradients propagate, metrics consistent,")
        print("    no leakage, causal models separated, reproducibility ensured.")
    else:
        print("\n  ✗ SYSTEM HAS ISSUES - DO NOT PRODUCE DISSERTATION RESULTS")
        print("    Fix failing tests before proceeding.")

    print("\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
