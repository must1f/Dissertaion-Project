"""
Tests for Numerical Stability Utilities

Validates:
- Safe mathematical operations prevent NaN/Inf
- Gradient utilities work correctly
- Normalization handles edge cases
- Stability checks detect issues
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.numerical_stability import (
    # Safe operations
    safe_log,
    safe_exp,
    safe_div,
    safe_sqrt,
    safe_pow,
    safe_softmax,
    # Gradient utilities
    GradientStats,
    compute_gradient_stats,
    clip_gradients,
    scale_gradients,
    zero_nan_gradients,
    # Normalization
    RobustNormalizer,
    # Stability checks
    check_tensor_health,
    check_loss_health,
    # Stable activations
    stable_sigmoid,
    stable_tanh,
    leaky_clamp,
    # Robust losses
    smooth_l1_loss,
    log_cosh_loss,
)


class TestSafeOperations:
    """Tests for safe mathematical operations."""

    def test_safe_log_prevents_nan(self):
        """safe_log should not produce NaN for zero or negative inputs."""
        x = torch.tensor([0.0, -1.0, 1e-20, 1.0, 10.0])
        result = safe_log(x)

        assert not torch.isnan(result).any(), "safe_log produced NaN"
        assert not torch.isinf(result).any(), "safe_log produced Inf"

    def test_safe_log_preserves_normal_values(self):
        """safe_log should match log for normal positive values."""
        x = torch.tensor([1.0, 2.0, 10.0, 100.0])
        result = safe_log(x)
        expected = torch.log(x)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_safe_exp_prevents_overflow(self):
        """safe_exp should not overflow for large inputs."""
        x = torch.tensor([100.0, 500.0, 1000.0])
        result = safe_exp(x)

        assert not torch.isinf(result).any(), "safe_exp produced Inf"
        assert torch.isfinite(result).all(), "safe_exp produced non-finite values"

    def test_safe_exp_preserves_normal_values(self):
        """safe_exp should match exp for normal values."""
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        result = safe_exp(x)
        expected = torch.exp(x)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_safe_div_prevents_division_by_zero(self):
        """safe_div should not produce Inf for zero denominator."""
        numerator = torch.tensor([1.0, 2.0, 3.0])
        denominator = torch.tensor([0.0, 1.0, 0.0])

        result = safe_div(numerator, denominator)

        assert not torch.isinf(result).any(), "safe_div produced Inf"
        assert torch.isfinite(result).all(), "safe_div produced non-finite values"

    def test_safe_div_preserves_normal_division(self):
        """safe_div should be close to normal division for non-zero denominators."""
        numerator = torch.tensor([1.0, 2.0, 3.0])
        denominator = torch.tensor([2.0, 4.0, 6.0])

        result = safe_div(numerator, denominator)
        expected = numerator / denominator

        assert torch.allclose(result, expected, atol=1e-6)

    def test_safe_sqrt_prevents_nan(self):
        """safe_sqrt should not produce NaN for negative inputs."""
        x = torch.tensor([-1.0, 0.0, 1.0, 4.0])
        result = safe_sqrt(x)

        assert not torch.isnan(result).any(), "safe_sqrt produced NaN"
        assert (result >= 0).all(), "safe_sqrt should be non-negative"

    def test_safe_pow_handles_negative_base(self):
        """safe_pow should handle negative bases with fractional exponents."""
        base = torch.tensor([-2.0, 0.0, 2.0])
        exponent = 0.5  # Square root

        result = safe_pow(base, exponent)

        assert not torch.isnan(result).any(), "safe_pow produced NaN"
        assert torch.isfinite(result).all(), "safe_pow produced non-finite values"

    def test_safe_softmax_numerically_stable(self):
        """safe_softmax should handle large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        result = safe_softmax(x)

        assert not torch.isnan(result).any(), "safe_softmax produced NaN"
        assert torch.isclose(result.sum(), torch.tensor(1.0), atol=1e-5)
        assert (result >= 0).all() and (result <= 1).all()


class TestGradientUtilities:
    """Tests for gradient utilities."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 1)

    def test_compute_gradient_stats(self, simple_model):
        """compute_gradient_stats should return valid statistics."""
        # Create gradients
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.mean()
        loss.backward()

        stats = compute_gradient_stats(simple_model)

        assert isinstance(stats, GradientStats)
        assert stats.total_norm >= 0
        assert stats.n_params > 0
        assert stats.n_nan == 0
        assert stats.n_inf == 0

    def test_compute_gradient_stats_no_grads(self, simple_model):
        """compute_gradient_stats should handle model with no gradients."""
        stats = compute_gradient_stats(simple_model)

        assert stats.total_norm == 0
        assert stats.n_params == 0

    def test_clip_gradients(self, simple_model):
        """clip_gradients should limit gradient norm."""
        # Create large gradients
        for p in simple_model.parameters():
            p.grad = torch.randn_like(p) * 100

        max_norm = 1.0
        original_norm = clip_gradients(simple_model, max_norm=max_norm)

        # Compute new norm
        total_norm = torch.sqrt(sum(
            p.grad.norm(2) ** 2 for p in simple_model.parameters() if p.grad is not None
        ))

        assert total_norm <= max_norm + 1e-6, "Gradients not properly clipped"

    def test_scale_gradients(self, simple_model):
        """scale_gradients should multiply all gradients."""
        # Set known gradients
        for p in simple_model.parameters():
            p.grad = torch.ones_like(p)

        scale = 2.0
        scale_gradients(simple_model, scale)

        for p in simple_model.parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, torch.ones_like(p) * scale)

    def test_zero_nan_gradients(self, simple_model):
        """zero_nan_gradients should replace NaN with zeros."""
        # Inject NaN gradients
        for p in simple_model.parameters():
            p.grad = torch.randn_like(p)
            p.grad[0] = float('nan')

        n_nan = zero_nan_gradients(simple_model)

        assert n_nan > 0, "Should have found NaN gradients"

        for p in simple_model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN gradients not zeroed"


class TestRobustNormalizer:
    """Tests for RobustNormalizer."""

    def test_standard_normalization(self):
        """Standard normalization should produce ~N(0,1)."""
        X = torch.randn(100, 5) * 10 + 5

        normalizer = RobustNormalizer(method='standard')
        X_norm = normalizer.fit_transform(X)

        # Should have mean ~0 and std ~1
        assert torch.abs(X_norm.mean()) < 0.5
        assert torch.abs(X_norm.std() - 1.0) < 0.5

    def test_robust_normalization(self):
        """Robust normalization should handle outliers."""
        X = torch.randn(100, 5)
        X[0] = 1000  # Outlier

        normalizer = RobustNormalizer(method='robust')
        X_norm = normalizer.fit_transform(X)

        # Should not have extreme values
        assert X_norm.abs().max() < 100

    def test_inverse_transform(self):
        """Inverse transform should recover original data."""
        X = torch.randn(50, 3) * 5 + 2

        normalizer = RobustNormalizer(method='standard')
        X_norm = normalizer.fit_transform(X)
        X_recovered = normalizer.inverse_transform(X_norm)

        assert torch.allclose(X, X_recovered, atol=1e-5)

    def test_handle_nan_zero(self):
        """Should replace NaN with zero."""
        X = torch.randn(10, 3)
        X[0, 0] = float('nan')

        normalizer = RobustNormalizer(handle_nan='zero')
        X_norm = normalizer.fit_transform(X)

        assert not torch.isnan(X_norm).any()


class TestStabilityChecks:
    """Tests for stability check functions."""

    def test_check_tensor_health_healthy(self):
        """check_tensor_health should pass for normal tensor."""
        x = torch.randn(10, 10)
        health = check_tensor_health(x)

        assert health['healthy'] is True
        assert health['has_nan'] is False
        assert health['has_inf'] is False

    def test_check_tensor_health_nan(self):
        """check_tensor_health should detect NaN."""
        x = torch.randn(10, 10)
        x[0, 0] = float('nan')
        health = check_tensor_health(x)

        assert health['healthy'] is False
        assert health['has_nan'] is True

    def test_check_tensor_health_inf(self):
        """check_tensor_health should detect Inf."""
        x = torch.randn(10, 10)
        x[0, 0] = float('inf')
        health = check_tensor_health(x)

        assert health['healthy'] is False
        assert health['has_inf'] is True

    def test_check_loss_health_healthy(self):
        """check_loss_health should pass for normal loss."""
        loss = torch.tensor(0.5)
        assert check_loss_health(loss) is True

    def test_check_loss_health_nan(self):
        """check_loss_health should fail for NaN loss."""
        loss = torch.tensor(float('nan'))
        assert check_loss_health(loss) is False

    def test_check_loss_health_too_large(self):
        """check_loss_health should fail for very large loss."""
        loss = torch.tensor(1e10)
        assert check_loss_health(loss, max_loss=1e6) is False


class TestStableActivations:
    """Tests for stable activation functions."""

    def test_stable_sigmoid_matches_sigmoid(self):
        """stable_sigmoid should match sigmoid for normal inputs."""
        x = torch.randn(100)
        result = stable_sigmoid(x)
        expected = torch.sigmoid(x)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_stable_sigmoid_handles_extreme(self):
        """stable_sigmoid should not overflow for extreme inputs."""
        x = torch.tensor([-1000.0, 1000.0])
        result = stable_sigmoid(x)

        assert torch.isfinite(result).all()
        assert result[0] < 0.01  # Should be near 0
        assert result[1] > 0.99  # Should be near 1

    def test_stable_tanh_matches_tanh(self):
        """stable_tanh should match tanh for normal inputs."""
        x = torch.randn(100)
        result = stable_tanh(x)
        expected = torch.tanh(x)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_stable_tanh_handles_extreme(self):
        """stable_tanh should not overflow for extreme inputs."""
        x = torch.tensor([-1000.0, 1000.0])
        result = stable_tanh(x)

        assert torch.isfinite(result).all()
        assert torch.isclose(result[0], torch.tensor(-1.0))
        assert torch.isclose(result[1], torch.tensor(1.0))

    def test_leaky_clamp_clamps_values(self):
        """leaky_clamp should limit values."""
        x = torch.tensor([-10.0, 0.0, 10.0])
        result = leaky_clamp(x, min_val=-1.0, max_val=1.0, leak=0.01)

        # Within bounds should be unchanged
        assert result[1] == 0.0
        # Outside bounds should be limited but with leak
        assert -1.1 < result[0] < -1.0
        assert 1.0 < result[2] < 1.1


class TestRobustLosses:
    """Tests for robust loss functions."""

    def test_smooth_l1_loss_basic(self):
        """smooth_l1_loss should work like Huber loss."""
        pred = torch.tensor([0.0, 1.0, 2.0])
        target = torch.tensor([0.0, 0.0, 0.0])

        loss = smooth_l1_loss(pred, target)

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_smooth_l1_loss_robust_to_outliers(self):
        """smooth_l1_loss should be more robust than MSE to outliers."""
        pred = torch.tensor([0.0, 0.0, 0.0, 100.0])  # One outlier
        target = torch.tensor([0.0, 0.0, 0.0, 0.0])

        smooth_loss = smooth_l1_loss(pred, target)
        mse_loss = ((pred - target) ** 2).mean()

        # Smooth L1 should be much smaller due to linear behavior for large errors
        assert smooth_loss.item() < mse_loss.item()

    def test_log_cosh_loss_basic(self):
        """log_cosh_loss should work for normal inputs."""
        pred = torch.randn(10)
        target = torch.randn(10)

        loss = log_cosh_loss(pred, target)

        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_log_cosh_loss_robust(self):
        """log_cosh_loss should be robust to outliers."""
        pred = torch.tensor([0.0, 0.0, 0.0, 50.0])  # Outlier
        target = torch.tensor([0.0, 0.0, 0.0, 0.0])

        loss = log_cosh_loss(pred, target)
        mse_loss = ((pred - target) ** 2).mean()

        # Log-cosh should be smaller for large errors
        assert loss.item() < mse_loss.item()


def test_all_imports():
    """Verify all numerical stability exports are importable."""
    from src.utils import (
        safe_log,
        safe_exp,
        safe_div,
        safe_sqrt,
        safe_pow,
        safe_softmax,
        GradientStats,
        compute_gradient_stats,
        clip_gradients,
        scale_gradients,
        zero_nan_gradients,
        RobustNormalizer,
        check_tensor_health,
        check_loss_health,
        GradScalerWrapper,
        stable_sigmoid,
        stable_tanh,
        leaky_clamp,
        smooth_l1_loss,
        log_cosh_loss,
    )

    # All should be callable or classes
    assert callable(safe_log)
    assert callable(compute_gradient_stats)
    assert callable(check_tensor_health)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
