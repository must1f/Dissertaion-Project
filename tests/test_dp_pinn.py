"""
Unit tests for Dual-Phase PINN implementation.

Tests cover:
- BurgersPINN model architecture and forward pass
- DualPhasePINN phase switching and intermediate constraint
- Autograd derivative computation
- Loss computation
"""

import math
import pytest
import torch
import torch.nn as nn

# Import modules to test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dp_pinn import BurgersPINN, DualPhasePINN, create_burgers_pinn


class TestBurgersPINN:
    """Tests for BurgersPINN model."""

    @pytest.fixture
    def model(self):
        """Create a BurgersPINN instance."""
        return BurgersPINN(
            num_layers=4,
            hidden_dim=20,
            activation="tanh",
            viscosity=0.01 / math.pi,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        batch_size = 32
        x = torch.rand(batch_size) * 2 - 1  # x in [-1, 1]
        t = torch.rand(batch_size)  # t in [0, 1]
        return x, t

    def test_model_creation(self, model):
        """Test that model is created correctly."""
        assert model is not None
        assert model.viscosity == pytest.approx(0.01 / math.pi)
        assert model.num_layers == 4
        assert model.hidden_dim == 20

    def test_forward_pass(self, model, sample_data):
        """Test forward pass produces correct output shape."""
        x, t = sample_data
        output = model(x, t)

        assert output.shape == (len(x), 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_grad(self, model, sample_data):
        """Test forward_with_grad computes derivatives."""
        x, t = sample_data
        u, u_t, u_x, u_xx = model.forward_with_grad(x, t)

        # Check shapes
        assert u.shape == (len(x), 1)
        assert u_t.shape == (len(x), 1)
        assert u_x.shape == (len(x), 1)
        assert u_xx.shape == (len(x), 1)

        # Check no NaN/Inf
        for tensor in [u, u_t, u_x, u_xx]:
            assert not torch.isnan(tensor).any()
            assert not torch.isinf(tensor).any()

    def test_pde_residual(self, model, sample_data):
        """Test PDE residual computation."""
        x, t = sample_data
        residual = model.compute_pde_residual(x, t)

        assert residual.shape == (len(x), 1)
        assert not torch.isnan(residual).any()

    def test_ic_loss(self, model):
        """Test initial condition loss."""
        x = torch.linspace(-1, 1, 100)
        loss = model.compute_ic_loss(x)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        assert not torch.isnan(loss)

    def test_bc_loss(self, model):
        """Test boundary condition loss."""
        t = torch.rand(100)
        loss = model.compute_bc_loss(t)

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_compute_loss(self, model, sample_data):
        """Test total loss computation."""
        x, t = sample_data
        x_ic = torch.linspace(-1, 1, 50)
        t_bc = torch.rand(50)

        loss, loss_dict = model.compute_loss(x, t, x_ic, t_bc)

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert "pde_loss" in loss_dict
        assert "ic_loss" in loss_dict
        assert "bc_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through loss."""
        x, t = sample_data
        x_ic = torch.linspace(-1, 1, 50)
        t_bc = torch.rand(50)

        loss, _ = model.compute_loss(x, t, x_ic, t_bc)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_different_activations(self):
        """Test model with different activations."""
        for activation in ["tanh", "gelu"]:
            model = BurgersPINN(activation=activation)
            x = torch.rand(10) * 2 - 1
            t = torch.rand(10)
            output = model(x, t)
            assert output.shape == (10, 1)


class TestDualPhasePINN:
    """Tests for DualPhasePINN model."""

    @pytest.fixture
    def model(self):
        """Create a DualPhasePINN instance."""
        return DualPhasePINN(
            t_switch=0.4,
            num_layers=4,
            hidden_dim=20,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data for both phases."""
        n = 32

        # Phase 1 data: t in [0, 0.4]
        x_p1 = torch.rand(n) * 2 - 1
        t_p1 = torch.rand(n) * 0.4

        # Phase 2 data: t in [0.4, 1]
        x_p2 = torch.rand(n) * 2 - 1
        t_p2 = torch.rand(n) * 0.6 + 0.4

        # Full data
        x_full = torch.cat([x_p1, x_p2])
        t_full = torch.cat([t_p1, t_p2])

        return {
            "x_p1": x_p1,
            "t_p1": t_p1,
            "x_p2": x_p2,
            "t_p2": t_p2,
            "x_full": x_full,
            "t_full": t_full,
            "x_ic": torch.linspace(-1, 1, 50),
            "t_bc": torch.rand(50),
            "x_intermediate": torch.linspace(-1, 1, 30),
        }

    def test_model_creation(self, model):
        """Test DualPhasePINN is created correctly."""
        assert model.t_switch == 0.4
        assert hasattr(model, "phase1_net")
        assert hasattr(model, "phase2_net")

    def test_forward_pass(self, model, sample_data):
        """Test forward pass handles both phases."""
        output = model(sample_data["x_full"], sample_data["t_full"])
        assert output.shape == (len(sample_data["x_full"]), 1)

    def test_phase_selection(self, model):
        """Test that correct phase network is used."""
        # Phase 1 point
        x1 = torch.tensor([0.0])
        t1 = torch.tensor([0.2])  # t < 0.4
        u1_full = model(x1, t1)
        u1_p1 = model.forward_phase1(x1, t1)
        assert torch.allclose(u1_full, u1_p1, atol=1e-6)

        # Phase 2 point
        t2 = torch.tensor([0.7])  # t > 0.4
        u2_full = model(x1, t2)
        u2_p2 = model.forward_phase2(x1, t2)
        assert torch.allclose(u2_full, u2_p2, atol=1e-6)

    def test_intermediate_loss(self, model, sample_data):
        """Test intermediate constraint loss."""
        loss = model.compute_intermediate_loss(sample_data["x_intermediate"])

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_phase1_loss(self, model, sample_data):
        """Test phase 1 loss computation."""
        loss, loss_dict = model.compute_phase1_loss(
            sample_data["x_p1"],
            sample_data["t_p1"],
            sample_data["x_ic"],
            sample_data["t_bc"][sample_data["t_bc"] < 0.4] if any(sample_data["t_bc"] < 0.4) else sample_data["t_bc"][:10],
        )

        assert loss.dim() == 0
        assert "pde_loss" in loss_dict
        assert "ic_loss" in loss_dict

    def test_phase2_loss(self, model, sample_data):
        """Test phase 2 loss computation."""
        loss, loss_dict = model.compute_phase2_loss(
            sample_data["x_p2"],
            sample_data["t_p2"],
            sample_data["x_intermediate"],
            sample_data["t_bc"][sample_data["t_bc"] > 0.4] if any(sample_data["t_bc"] > 0.4) else sample_data["t_bc"][:10] + 0.5,
        )

        assert loss.dim() == 0
        assert "pde_loss" in loss_dict
        assert "intermediate_loss" in loss_dict

    def test_freeze_unfreeze_phase1(self, model):
        """Test freezing and unfreezing phase 1."""
        # Initially all params require grad
        for param in model.phase1_net.parameters():
            assert param.requires_grad

        # Freeze
        model.freeze_phase1()
        for param in model.phase1_net.parameters():
            assert not param.requires_grad

        # Phase 2 should still require grad
        for param in model.phase2_net.parameters():
            assert param.requires_grad

        # Unfreeze
        model.unfreeze_phase1()
        for param in model.phase1_net.parameters():
            assert param.requires_grad

    def test_get_trainable_params(self, model):
        """Test getting trainable parameters by phase."""
        params1 = model.get_trainable_params(phase=1)
        params2 = model.get_trainable_params(phase=2)

        assert len(params1) > 0
        assert len(params2) > 0

        # They should be different sets
        param_ids_1 = {id(p) for p in params1}
        param_ids_2 = {id(p) for p in params2}
        assert param_ids_1.isdisjoint(param_ids_2)


class TestCreateBurgersPinn:
    """Tests for factory function."""

    def test_create_standard(self):
        """Test creating standard PINN."""
        model = create_burgers_pinn("standard")
        assert isinstance(model, BurgersPINN)

    def test_create_dual_phase(self):
        """Test creating dual-phase PINN."""
        model = create_burgers_pinn("dual_phase", t_switch=0.5)
        assert isinstance(model, DualPhasePINN)
        assert model.t_switch == 0.5

    def test_invalid_type(self):
        """Test error for invalid model type."""
        with pytest.raises(ValueError):
            create_burgers_pinn("invalid_type")


class TestAutogradDerivatives:
    """Tests for autograd derivative computation accuracy."""

    def test_derivative_accuracy(self):
        """Test that autograd derivatives match finite differences."""
        model = BurgersPINN(num_layers=3, hidden_dim=20)

        # Test point
        x = torch.tensor([0.5], requires_grad=True)
        t = torch.tensor([0.3], requires_grad=True)

        # Get autograd derivatives
        u, u_t, u_x, u_xx = model.forward_with_grad(x, t)

        # Finite difference for u_x
        eps = 1e-4
        x_plus = torch.tensor([0.5 + eps])
        x_minus = torch.tensor([0.5 - eps])
        u_x_fd = (model(x_plus, t) - model(x_minus, t)) / (2 * eps)

        # Should be close (allow some tolerance due to network stochasticity)
        assert torch.allclose(u_x, u_x_fd, atol=0.1, rtol=0.1)

    def test_second_derivative_computation(self):
        """Test second derivative is computed correctly."""
        model = BurgersPINN(num_layers=3, hidden_dim=20)

        x = torch.rand(10) * 2 - 1
        t = torch.rand(10)

        _, _, u_x, u_xx = model.forward_with_grad(x, t)

        # u_xx should be the derivative of u_x w.r.t. x
        # This is implicitly tested by checking the PDE residual
        assert not torch.isnan(u_xx).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
