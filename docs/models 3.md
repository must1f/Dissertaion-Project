# Model Documentation

## PINN (Physics-Informed Neural Network) Models

Physics-Informed Neural Networks combine traditional neural network learning with physics-based constraints derived from stochastic differential equations (SDEs).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PINN Architecture                       │
│                                                              │
│  Input Features (X)         Physics Constraints (Φ)         │
│        │                            │                        │
│        ▼                            ▼                        │
│  ┌──────────────┐           ┌──────────────┐                │
│  │   Encoder    │           │   Physics    │                │
│  │  (FC Layers) │           │   Network    │                │
│  └──────┬───────┘           └──────┬───────┘                │
│         │                          │                        │
│         └────────────┬─────────────┘                        │
│                      ▼                                      │
│               ┌──────────────┐                              │
│               │    Fusion    │                              │
│               │    Layer     │                              │
│               └──────┬───────┘                              │
│                      │                                      │
│                      ▼                                      │
│               ┌──────────────┐                              │
│               │   Output     │ ──► Predictions (ŷ)         │
│               │   Layer      │                              │
│               └──────────────┘                              │
│                                                              │
│  Loss = L_data + λ * L_physics                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Variants

### 1. PINN Baseline
**Key:** `pinn_baseline`

Standard PINN without specific physics equations. Uses learnable physics parameters.

```python
L = L_data  # No physics loss
```

### 2. PINN GBM (Geometric Brownian Motion)
**Key:** `pinn_gbm`

Models stock prices following GBM:
```
dS = μS·dt + σS·dW
```

**Physics Loss:**
```python
L_physics = ||dS/dt - μS||² / σ²
```

**Learned Parameters:**
- μ (drift): Expected return rate
- σ (volatility): Price volatility

### 3. PINN Ornstein-Uhlenbeck
**Key:** `pinn_ou`

Models mean-reverting behavior:
```
dX = θ(μ - X)dt + σdW
```

**Physics Loss:**
```python
L_physics = ||dX/dt - θ(μ - X)||²
```

**Learned Parameters:**
- θ (theta): Mean reversion speed
- μ (mu): Long-term mean
- σ (sigma): Volatility

### 4. PINN GBM + OU (Combined)
**Key:** `pinn_gbm_ou`

Combines trend-following (GBM) with mean reversion (OU):
```
dS = μS·dt + σS·dW + θ(γ - log(S))·dt
```

**Learned Parameters:**
- μ: Drift rate
- σ: Volatility
- θ: Mean reversion speed
- γ: Log-price equilibrium

### 5. PINN Black-Scholes
**Key:** `pinn_black_scholes`

Based on the Black-Scholes PDE:
```
∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0
```

**Learned Parameters:**
- r: Risk-free rate
- σ: Implied volatility

### 6. PINN Global
**Key:** `pinn_global`

Multi-physics model combining multiple SDEs with adaptive weighting.

## Baseline Models

### LSTM
**Key:** `lstm`

Standard LSTM network for sequence prediction.
- Hidden dimensions: [128, 64]
- Dropout: 0.2

### BiLSTM
**Key:** `bilstm`

Bidirectional LSTM for capturing forward and backward dependencies.

### GRU
**Key:** `gru`

Gated Recurrent Unit - simpler alternative to LSTM.

### Transformer
**Key:** `transformer`

Attention-based architecture for sequence modeling.
- Heads: 8
- Layers: 4
- d_model: 256

## Model Configuration

```python
# Example model configuration
config = {
    "model_type": "pinn_gbm_ou",
    "hidden_dims": [128, 64, 32],
    "dropout": 0.1,
    "physics_weight": 0.1,  # λ in loss function
    "learnable_physics": True,
    "initial_params": {
        "theta": 0.1,
        "gamma": 0.0,
        "mu": 0.08,
        "sigma": 0.2
    }
}
```

## Training

### Loss Function

```python
L_total = L_data + λ * L_physics + α * L_reg

where:
- L_data = MSE(y_pred, y_true)
- L_physics = physics constraint violation
- L_reg = L2 regularization
- λ = physics weight (tunable)
- α = regularization weight
```

### Curriculum Learning

Physics weight increases during training:
```python
λ(epoch) = λ_final * (1 - exp(-epoch / τ))
```

### Hyperparameters

| Parameter | Default | Range |
|-----------|---------|-------|
| learning_rate | 0.001 | 1e-4 to 1e-2 |
| batch_size | 32 | 16 to 128 |
| epochs | 100 | 50 to 500 |
| physics_weight | 0.1 | 0.01 to 1.0 |
| dropout | 0.1 | 0.0 to 0.3 |

## Uncertainty Quantification

### MC Dropout

Enable dropout during inference for uncertainty estimation:
```python
predictions = []
for _ in range(n_samples):
    with torch.enable_grad():
        pred = model(x, training=True)  # Keep dropout active
    predictions.append(pred)

mean = np.mean(predictions, axis=0)
std = np.std(predictions, axis=0)
confidence_interval = (mean - 1.96*std, mean + 1.96*std)
```

## Model Files

Models are saved in the `Models/` directory:
```
Models/
├── pinn_gbm_ou.pth           # Model weights
├── pinn_gbm_ou_history.json  # Training history
├── lstm.pth
├── lstm_history.json
└── ...
```

## Evaluation Metrics

### ML Metrics
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of determination
- Directional Accuracy

### Financial Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor

### Physics Metrics
- Physics Loss: Constraint violation magnitude
- Parameter Stability: Variance of learned parameters
- SDE Residuals: Individual equation residuals
