"""
Simple Example: Training Stacked PINN for Financial Forecasting

This example demonstrates:
1. Creating synthetic return-based data
2. Setting up StackedPINN model
3. Training with curriculum learning
4. Evaluating financial performance
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stacked_pinn import StackedPINN, ResidualPINN
from src.training.curriculum import CurriculumScheduler
from src.evaluation.financial_metrics import FinancialMetrics, compute_strategy_returns


def generate_synthetic_data(n_samples=1000, seq_length=60, n_features=10):
    """
    Generate synthetic return-based time series data

    Returns:
        X_train, y_train, returns_train, X_val, y_val, returns_val
    """
    print("Generating synthetic return data...")

    # Generate random walk returns with some structure
    np.random.seed(42)

    # Base returns with trend and mean reversion
    mu = 0.0002  # Daily drift
    sigma = 0.01  # Daily volatility
    theta = 0.1  # Mean reversion speed

    raw_returns = []
    r_t = 0.0

    for t in range(n_samples + seq_length + 100):
        # OU process for returns
        dr = theta * (mu - r_t) + sigma * np.random.randn()
        r_t += dr
        raw_returns.append(r_t)

    raw_returns = np.array(raw_returns)

    # Create features from returns
    X_data = []
    y_data = []
    returns_data = []

    for i in range(len(raw_returns) - seq_length - 1):
        # Features: returns + technical indicators
        seq = raw_returns[i:i+seq_length]

        # Simple features
        features = np.column_stack([
            seq,  # Raw returns
            np.roll(seq, 1),  # Lag 1
            np.roll(seq, 5),  # Lag 5
            np.cumsum(seq) / (np.arange(seq_length) + 1),  # Cumulative avg
            np.std(seq) * np.ones(seq_length),  # Volatility
            seq * 2 + np.random.randn(seq_length) * 0.001,  # Noisy feature
            seq - np.mean(seq),  # Mean-centered
            np.sign(seq),  # Sign
            np.abs(seq),  # Absolute
            seq ** 2  # Squared (volatility proxy)
        ])

        X_data.append(features)
        y_data.append(raw_returns[i+seq_length])
        returns_data.append(seq)

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    returns_data = np.array(returns_data)

    # Train/val split (80/20)
    split_idx = int(0.8 * len(X_data))

    X_train = X_data[:split_idx]
    y_train = y_data[:split_idx]
    returns_train = returns_data[:split_idx]

    X_val = X_data[split_idx:]
    y_val = y_data[split_idx:]
    returns_val = returns_data[split_idx:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features: {X_train.shape[-1]}, Sequence length: {seq_length}")

    return X_train, y_train, returns_train, X_val, y_val, returns_val


def train_epoch_simple(model, optimizer, X_batch, y_batch, returns_batch,
                       device, lambda_gbm, lambda_ou):
    """Simple training step"""
    model.train()

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    returns_batch = returns_batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    return_pred, direction_logits, _ = model(X_batch, compute_physics=True)

    # Prediction losses
    regression_loss = nn.functional.mse_loss(return_pred, y_batch)
    direction_targets = (y_batch > 0).long().squeeze()
    classification_loss = nn.functional.cross_entropy(direction_logits, direction_targets)
    prediction_loss = regression_loss + 0.1 * classification_loss

    # Physics loss
    model.lambda_gbm = lambda_gbm
    model.lambda_ou = lambda_ou
    physics_loss, _ = model.compute_physics_loss(X_batch, returns_batch)

    # Total loss
    total_loss = prediction_loss + physics_loss

    # Backward
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), prediction_loss.item(), physics_loss.item()


@torch.no_grad()
def evaluate_simple(model, X, y, device):
    """Simple evaluation"""
    model.eval()
    X = X.to(device)
    y = y.to(device)

    predictions, direction_logits, _ = model(X, compute_physics=False)

    mse = nn.functional.mse_loss(predictions, y).item()

    pred_np = predictions.cpu().numpy().flatten()
    y_np = y.cpu().numpy().flatten()

    # Compare direction of price CHANGES (predictions/targets are normalized prices)
    directional_acc = FinancialMetrics.directional_accuracy(pred_np, y_np, are_returns=False)

    return pred_np, mse, directional_acc


def main():
    """Main example"""
    print("=" * 80)
    print("STACKED PINN EXAMPLE")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Generate data
    X_train, y_train, returns_train, X_val, y_val, returns_val = generate_synthetic_data()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    returns_train_t = torch.FloatTensor(returns_train)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    returns_val_t = torch.FloatTensor(returns_val)

    # Create model
    print("\nCreating StackedPINN model...")
    model = StackedPINN(
        input_dim=X_train.shape[-1],
        encoder_dim=64,
        lstm_hidden_dim=64,
        num_encoder_layers=2,
        num_rnn_layers=2,
        prediction_hidden_dim=32,
        dropout=0.2,
        lambda_gbm=0.0,  # Will be updated by curriculum
        lambda_ou=0.0
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Curriculum scheduler
    print("Setting up curriculum learning...")
    curriculum = CurriculumScheduler(
        initial_lambda_gbm=0.0,
        final_lambda_gbm=0.1,
        initial_lambda_ou=0.0,
        final_lambda_ou=0.1,
        warmup_epochs=5,
        total_epochs=30,
        strategy='cosine'
    )

    # Training loop
    print("\nStarting training...")
    print("-" * 80)

    epochs = 30
    batch_size = 32
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Get curriculum weights
        weights = curriculum.step(epoch)
        lambda_gbm = weights['lambda_gbm']
        lambda_ou = weights['lambda_ou']

        # Training
        n_train = X_train.shape[0]
        indices = torch.randperm(n_train)

        epoch_losses = []
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:min(i+batch_size, n_train)]

            batch_X = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]
            batch_returns = returns_train_t[batch_idx]

            total_loss, pred_loss, phys_loss = train_epoch_simple(
                model, optimizer, batch_X, batch_y, batch_returns,
                device, lambda_gbm, lambda_ou
            )
            epoch_losses.append(total_loss)

        avg_train_loss = np.mean(epoch_losses)

        # Validation
        val_pred, val_mse, val_dir_acc = evaluate_simple(model, X_val_t, y_val_t, device)

        # Log progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_mse:.4f} | "
                  f"Dir Acc: {val_dir_acc:.3f} | "
                  f"λ_gbm: {lambda_gbm:.4f} | "
                  f"λ_ou: {lambda_ou:.4f}")

        # Track best
        if val_mse < best_val_loss:
            best_val_loss = val_mse

    print("-" * 80)
    print(f"Training complete! Best val MSE: {best_val_loss:.4f}\n")

    # Final evaluation with financial metrics
    print("=" * 80)
    print("FINANCIAL PERFORMANCE EVALUATION")
    print("=" * 80)

    final_pred, final_mse, final_dir_acc = evaluate_simple(model, X_val_t, y_val_t, device)

    # Compute strategy returns
    strategy_returns = compute_strategy_returns(
        predictions=final_pred,
        actual_returns=y_val,
        transaction_cost=0.001
    )

    # Compute financial metrics
    metrics = FinancialMetrics.compute_all_metrics(
        returns=strategy_returns,
        predictions=final_pred,
        targets=y_val,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    # Display results
    print(f"\nPrediction Metrics:")
    print(f"  MSE:                  {final_mse:.6f}")
    print(f"  Directional Accuracy: {final_dir_acc*100:.2f}%")

    print(f"\nFinancial Metrics:")
    print(f"  Total Return:         {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']*100:.2f}%")
    print(f"  Calmar Ratio:         {metrics['calmar_ratio']:.3f}")
    print(f"  Win Rate:             {metrics['win_rate']*100:.2f}%")
    print(f"  Volatility (Annual):  {metrics['volatility']*100:.2f}%")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)

    # Show comparison: with vs without physics
    print("\nNote: To see impact of physics constraints, try:")
    print("  1. Run with final_lambda = 0.0 (no physics)")
    print("  2. Run with final_lambda = 0.1 (with physics)")
    print("  3. Compare Sharpe ratio and directional accuracy")


if __name__ == '__main__':
    main()
