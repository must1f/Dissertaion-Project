import torch
from pathlib import Path

from src.models.model_registry import ModelRegistry
from src.models.volatility import VolatilityPINN
from src.models.pinn import PINNModel
from src.models.financial_dp_pinn import AdaptiveFinancialDualPhasePINN


def _finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def test_volatility_pinn_heston_loss_backward():
    model = VolatilityPINN(
        input_dim=3,
        hidden_dim=16,
        num_layers=1,
        lambda_ou=0.1,
        lambda_garch=0.1,
        lambda_feller=0.05,
        lambda_leverage=0.05,
        lambda_heston=0.1,
        enable_heston_constraint=True,
    )

    batch, seq_len = 4, 6
    x = torch.randn(batch, seq_len, 3)
    targets = torch.abs(torch.randn(batch, 1))

    variance_history = torch.abs(torch.randn(batch, seq_len))
    returns = torch.randn(batch, seq_len)

    preds = model(x)
    loss, loss_dict = model.compute_loss(
        preds,
        targets,
        {
            'variance_history': variance_history,
            'returns': returns,
            'long_run_variance': variance_history.mean(dim=1, keepdim=True),
            'dt': 1.0 / 252,
        },
        enable_physics=True,
    )

    assert _finite_tensor(loss)
    assert 'heston_loss' in loss_dict
    assert loss.requires_grad
    loss.backward()

    # Parameter constraints
    params = model.physics_loss.get_learned_params()
    assert params['kappa'] > 0
    assert params['theta_long'] > 0
    assert params['xi'] > 0
    assert -1 < params['rho'] < 1


def test_pinn_bs_residual_scaling_backward():
    model = PINNModel(
        input_dim=4,
        hidden_dim=16,
        num_layers=1,
        output_dim=1,
        lambda_gbm=0.1,
        lambda_bs=0.1,
        lambda_ou=0.1,
        lambda_langevin=0.0,
        base_model='lstm',
    )

    batch, seq_len, feat = 2, 5, 4
    x = torch.randn(batch, seq_len, feat)
    targets = torch.randn(batch, 1)
    prices = torch.abs(torch.randn(batch, seq_len)) + 50.0
    returns = torch.randn(batch, seq_len)
    vols = torch.abs(torch.randn(batch, seq_len)) + 0.1

    preds = model(x)
    loss, loss_dict = model.compute_loss(
        preds,
        targets,
        {
            'prices': prices,
            'returns': returns,
            'volatilities': vols,
            'inputs': x,
            'price_feature_idx': 0,
        },
        enable_physics=True,
    )

    assert _finite_tensor(loss)
    assert loss.requires_grad
    loss.backward()
    assert 'bs_loss' in loss_dict or model.physics_loss.lambda_bs == 0


def test_registry_vol_pinn_creation():
    registry = ModelRegistry(project_root=Path('.'))
    model = registry.create_model('vol_pinn', input_dim=5, hidden_dim=16, num_layers=1)
    assert model is not None
    assert isinstance(model, VolatilityPINN)


def test_registry_adaptive_dual_phase_creation():
    registry = ModelRegistry(project_root=Path('.'))
    model = registry.create_model(
        'adaptive_dual_phase_pinn',
        input_dim=5,
        hidden_dim=16,
        num_layers=1,
    )
    assert model is not None
    assert isinstance(model, AdaptiveFinancialDualPhasePINN)
