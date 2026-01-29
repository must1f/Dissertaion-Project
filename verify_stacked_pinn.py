"""
Verification Script for Stacked PINN Implementation

Checks that all components are properly installed and can be imported.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_imports():
    """Verify all modules can be imported"""
    print("=" * 80)
    print("STACKED PINN IMPLEMENTATION VERIFICATION")
    print("=" * 80)
    print()

    checks = []

    # Check 1: Model imports
    print("✓ Checking model imports...")
    try:
        from src.models.stacked_pinn import (
            PhysicsEncoder, ParallelHeads, PredictionHead,
            StackedPINN, ResidualPINN
        )
        print("  ✓ StackedPINN model classes imported successfully")
        checks.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import model classes: {e}")
        checks.append(False)

    # Check 2: Training imports
    print("✓ Checking training imports...")
    try:
        from src.training.curriculum import (
            CurriculumScheduler, AdaptiveCurriculumScheduler
        )
        print("  ✓ Curriculum schedulers imported successfully")
        checks.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import curriculum: {e}")
        checks.append(False)

    # Check 3: Walk-forward validation
    print("✓ Checking walk-forward validation...")
    try:
        from src.training.walk_forward import (
            WalkForwardValidator, WalkForwardFold,
            create_walk_forward_splits, TimeSeriesCrossValidator
        )
        print("  ✓ Walk-forward validation imported successfully")
        checks.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import walk-forward: {e}")
        checks.append(False)

    # Check 4: Financial metrics
    print("✓ Checking financial metrics...")
    try:
        from src.evaluation.financial_metrics import (
            FinancialMetrics, compute_strategy_returns
        )
        print("  ✓ Financial metrics imported successfully")
        checks.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import financial metrics: {e}")
        checks.append(False)

    print()
    print("=" * 80)

    # Summary
    if all(checks):
        print("✓ ALL CHECKS PASSED")
        print()
        print("Stacked PINN implementation is properly installed!")
        print()
        print("Next steps:")
        print("  1. Run the example: python examples/stacked_pinn_example.py")
        print("  2. Train on real data: python src/training/train_stacked_pinn.py --help")
        print("  3. Read documentation: STACKED_PINN_README.md")
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Please check error messages above and ensure all files are present.")

    print("=" * 80)
    return all(checks)


def test_model_creation():
    """Test that models can be instantiated"""
    print()
    print("=" * 80)
    print("MODEL INSTANTIATION TEST")
    print("=" * 80)
    print()

    try:
        import torch
        from src.models.stacked_pinn import StackedPINN, ResidualPINN

        # Test StackedPINN
        print("✓ Creating StackedPINN...")
        model_stacked = StackedPINN(
            input_dim=10,
            encoder_dim=64,
            lstm_hidden_dim=64,
            num_encoder_layers=2,
            num_rnn_layers=2,
            prediction_hidden_dim=32,
            dropout=0.2,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
        n_params_stacked = sum(p.numel() for p in model_stacked.parameters())
        print(f"  ✓ StackedPINN created: {n_params_stacked:,} parameters")

        # Test ResidualPINN
        print("✓ Creating ResidualPINN...")
        model_residual = ResidualPINN(
            input_dim=10,
            base_model_type='lstm',
            base_hidden_dim=64,
            correction_hidden_dim=32,
            num_base_layers=2,
            num_correction_layers=2,
            dropout=0.2,
            lambda_gbm=0.1,
            lambda_ou=0.1
        )
        n_params_residual = sum(p.numel() for p in model_residual.parameters())
        print(f"  ✓ ResidualPINN created: {n_params_residual:,} parameters")

        # Test forward pass
        print("✓ Testing forward pass...")
        batch_size = 4
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)

        # StackedPINN forward
        return_pred_s, dir_logits_s, attn_s = model_stacked(x, compute_physics=True)
        print(f"  ✓ StackedPINN forward: return={return_pred_s.shape}, direction={dir_logits_s.shape}")

        # ResidualPINN forward
        return_pred_r, dir_logits_r, components = model_residual(x, return_components=True)
        print(f"  ✓ ResidualPINN forward: return={return_pred_r.shape}, direction={dir_logits_r.shape}")

        # Test physics loss
        print("✓ Testing physics loss computation...")
        returns = torch.randn(batch_size, seq_len)
        phys_loss_s, phys_dict_s = model_stacked.compute_physics_loss(x, returns)
        print(f"  ✓ StackedPINN physics loss: {phys_loss_s.item():.6f}")
        print(f"    - GBM loss: {phys_dict_s['gbm_loss']:.6f}")
        print(f"    - OU loss: {phys_dict_s['ou_loss']:.6f}")

        phys_loss_r, phys_dict_r = model_residual.compute_physics_loss(x, returns)
        print(f"  ✓ ResidualPINN physics loss: {phys_loss_r.item():.6f}")

        print()
        print("=" * 80)
        print("✓ MODEL INSTANTIATION TEST PASSED")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"✗ Model instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curriculum():
    """Test curriculum scheduler"""
    print()
    print("=" * 80)
    print("CURRICULUM SCHEDULER TEST")
    print("=" * 80)
    print()

    try:
        from src.training.curriculum import CurriculumScheduler

        print("✓ Creating curriculum scheduler (cosine strategy)...")
        curriculum = CurriculumScheduler(
            initial_lambda_gbm=0.0,
            final_lambda_gbm=0.1,
            initial_lambda_ou=0.0,
            final_lambda_ou=0.1,
            warmup_epochs=5,
            total_epochs=20,
            strategy='cosine'
        )

        print("✓ Testing curriculum schedule:")
        test_epochs = [0, 2, 5, 8, 12, 15, 19]
        for epoch in test_epochs:
            weights = curriculum.step(epoch)
            print(f"  Epoch {epoch:2d}: λ_gbm={weights['lambda_gbm']:.4f}, "
                  f"λ_ou={weights['lambda_ou']:.4f}, "
                  f"progress={weights['progress']:.2f}")

        print()
        print("=" * 80)
        print("✓ CURRICULUM SCHEDULER TEST PASSED")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"✗ Curriculum test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    results = []

    # Test 1: Imports
    results.append(verify_imports())

    # Test 2: Model instantiation
    results.append(test_model_creation())

    # Test 3: Curriculum
    results.append(test_curriculum())

    # Final summary
    print()
    print("=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    print()

    if all(results):
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print()
        print("Your stacked PINN implementation is fully functional!")
        print()
        print("You can now:")
        print("  • Run the example: python examples/stacked_pinn_example.py")
        print("  • Train on your data: python src/training/train_stacked_pinn.py")
        print("  • Read the docs: STACKED_PINN_README.md")
        print()
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print()
        print("Please review error messages above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
