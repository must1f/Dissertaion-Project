import torch

from src.models.transformer import TransformerModel


def test_transformer_generates_causal_mask_by_default():
    model = TransformerModel(input_dim=5, d_model=32, nhead=4, num_encoder_layers=1, dim_feedforward=64, causal=True)
    x = torch.randn(2, 4, 5)
    # Trigger forward to implicitly build mask
    with torch.no_grad():
        out = model(x)
    assert out.shape[:2] == (2, 1)

    # Explicitly check generated mask shape and upper triangular constraint (excluding diagonal)
    mask = model.generate_square_subsequent_mask(4, torch.device("cpu"))
    assert mask.shape == (4, 4)
    idx = torch.triu(torch.ones_like(mask), diagonal=1).bool()
    assert torch.isinf(mask[idx]).all()
