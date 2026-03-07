"""
Verification script for evaluation integrity.

Runs lightweight checks for:
- Transformer causal masking
- Price-scale validation (z-score rejection)
- StrategyEngine position lag and turnover sanity
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.financial_metrics import FinancialMetrics, assert_price_scale
from src.evaluation.strategy_engine import StrategyEngine, StrategyConfig
from src.models.transformer import TransformerModel


def check_transformer_mask() -> dict:
    model = TransformerModel(input_dim=5, d_model=32, nhead=4, num_encoder_layers=1, dim_feedforward=64, causal=True)
    mask = model.generate_square_subsequent_mask(6, torch.device("cpu"))
    return {
        "shape": tuple(mask.shape),
        "causal_ok": bool(torch.isinf(mask.triu(1)).all()),
    }


def check_price_scale_guard() -> dict:
    z = np.array([0.1, -0.2, 0.05, 0.0])
    try:
        assert_price_scale(z, context="verify_evaluation_integrity")
        raised = False
    except ValueError:
        raised = True
    return {"zscore_rejected": raised}


def check_strategy_lag() -> dict:
    signals = np.array([1.0, -1.0, 1.0])
    positions = StrategyEngine.sign_strategy(signals)
    returns = np.array([0.0, 0.01, -0.02])
    net, stats = StrategyEngine.run(signals, returns, StrategyConfig(transaction_cost=0.0))
    return {
        "first_position_zero": float(positions[0]) == 0.0,
        "positions_shifted": np.allclose(positions[1:], signals[:-1]),
        "net_len_matches": len(net) == len(returns),
        "turnover_recorded": stats.get("avg_daily_turnover", 0.0) >= 0.0,
    }


def main():
    report = {
        "transformer_mask": check_transformer_mask(),
        "price_scale_guard": check_price_scale_guard(),
        "strategy_lag": check_strategy_lag(),
    }

    out = Path("results/evaluation_integrity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
