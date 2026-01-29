# Stacked Physics-Informed Neural Networks (PINN) for Financial Forecasting

Complete implementation of stacked PINN system with curriculum learning, walk-forward validation, and comprehensive financial metrics evaluation.

## Architecture Overview

### 1. StackedPINN Architecture
```
Input Features (Returns-based only)
    ↓
PhysicsEncoder (Feature-level encoding)
    ↓
Parallel Processing:
├── LSTM Head
└── GRU Head
    ↓
Attention-based Fusion
    ↓
Dense Prediction Head
├── Regression (Return prediction)
└── Classification (Direction: up/down)
```

### 2. ResidualPINN Architecture
```
Input Features
    ↓
Base Model (LSTM/GRU) → Base Prediction
    ↓
Physics-informed Correction Network → Correction
    ↓
Final Prediction = Base + Correction
```

## Key Features

### Physics Constraints on Returns (Not Prices!)
- **GBM (Geometric Brownian Motion)**: `dR/dt ≈ μ + σ·ε`
- **OU (Ornstein-Uhlenbeck)**: `dR = θ(μ - R)dt + σdW`
- Applied as soft constraints during training

### Curriculum Training
- Gradually increases physics loss weights from 0 to final values
- Strategies: `linear`, `exponential`, `cosine`, `step`
- Warmup period with pure data loss (λ = 0)
- After warmup: physics weights gradually increase

### Combined Loss Function
```
L_total = L_prediction + λ_gbm * L_gbm + λ_ou * L_ou

where:
  L_prediction = L_regression + 0.1 * L_classification
  L_regression = MSE(predicted_return, actual_return)
  L_classification = CrossEntropy(predicted_direction, actual_direction)
```

### Financial Evaluation Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside deviation-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: Annual return / |Max Drawdown|
- **Directional Accuracy**: Sign agreement between predictions and actuals
- **Win Rate**: Percentage of positive returns
- **Annualized Volatility**: Risk measurement

## Usage

### Training Stacked PINN

```bash
python src/training/train_stacked_pinn.py \
    --model-type stacked \
    --epochs 100 \
    --warmup-epochs 10 \
    --final-lambda-gbm 0.1 \
    --final-lambda-ou 0.1 \
    --curriculum-strategy cosine
```

### Training Residual PINN

```bash
python src/training/train_stacked_pinn.py \
    --model-type residual \
    --epochs 100 \
    --warmup-epochs 10 \
    --final-lambda-gbm 0.1 \
    --final-lambda-ou 0.1 \
    --curriculum-strategy cosine
```

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-type` | str | `stacked` | Model architecture: `stacked` or `residual` |
| `--epochs` | int | `100` | Total training epochs |
| `--warmup-epochs` | int | `10` | Warmup epochs with λ=0 |
| `--final-lambda-gbm` | float | `0.1` | Final GBM physics weight |
| `--final-lambda-ou` | float | `0.1` | Final OU physics weight |
| `--curriculum-strategy` | str | `cosine` | Curriculum strategy: `linear`, `exponential`, `cosine`, `step` |

## Return-Based Features (No Price Levels!)

The system uses ONLY return-based features to avoid look-ahead bias:

- `log_return`: Logarithmic returns
- `simple_return`: Simple percentage returns
- `rolling_volatility_5/20/60`: Rolling volatility windows
- `momentum_5/20`: Price momentum indicators
- `rsi_14`: Relative Strength Index
- `macd` and `macd_signal`: MACD indicators

**Physics constraints operate exclusively on returns, not prices.**

## Curriculum Training Strategies

### 1. Linear
```python
scale = progress  # Linear increase
```

### 2. Exponential
```python
scale = progress^2  # Slower at start, faster later
```

### 3. Cosine
```python
scale = 0.5 * (1 - cos(π * progress))  # Smooth S-curve
```

### 4. Step
```python
# Discrete jumps at 25%, 50%, 75%
0.0 → 0.33 → 0.66 → 1.0
```

## Walk-Forward Validation

Time series cross-validation preventing look-ahead bias:

```python
from src.training.walk_forward import WalkForwardValidator

validator = WalkForwardValidator(
    n_samples=len(data),
    initial_train_size=252*5,  # 5 years
    validation_size=252//4,     # 3 months
    step_size=21,               # 1 month
    mode='expanding'            # or 'rolling'
)

folds = validator.split()
```

**Modes:**
- `expanding`: Training window grows over time
- `rolling`: Fixed training window size

## Code Structure

```
src/
├── models/
│   └── stacked_pinn.py         # StackedPINN, ResidualPINN architectures
├── training/
│   ├── train_stacked_pinn.py   # Main training script
│   ├── curriculum.py           # Curriculum schedulers
│   └── walk_forward.py         # Walk-forward validation
└── evaluation/
    └── financial_metrics.py    # Financial performance metrics
```

## Implementation Details

### StackedPINN Components

**PhysicsEncoder**
- Multi-layer feature encoder with LayerNorm, GELU, Dropout
- Physics-aware projection layer
- Input: `(batch, seq_len, input_dim)`
- Output: `(batch, seq_len, hidden_dim)`

**ParallelHeads**
- LSTM and GRU processed in parallel
- Attention mechanism for combining outputs
- Returns concatenated features: `(batch, hidden_dim*2)`

**PredictionHead**
- Shared layers followed by task-specific heads
- Regression head: Single output for return prediction
- Classification head: Binary logits for direction (up/down)

### ResidualPINN Components

**Base Model**
- Standard LSTM or GRU
- Makes initial prediction from historical data

**Correction Network**
- Physics-informed residual correction
- Takes base hidden state + base prediction
- Learns to adjust predictions to satisfy physics constraints
- Uses Tanh activation for bounded corrections

## Training Process

1. **Data Preparation**: Load return-based features only
2. **Model Creation**: Initialize StackedPINN or ResidualPINN
3. **Curriculum Initialization**: Set up physics weight schedule
4. **Training Loop**:
   - Epoch 0-9 (warmup): λ_gbm = λ_ou = 0 (pure data loss)
   - Epoch 10+: Gradually increase physics weights using chosen strategy
   - Each batch:
     - Forward pass → predictions
     - Compute prediction loss (MSE + cross-entropy)
     - Compute physics loss (GBM + OU residuals)
     - Total loss = prediction + weighted physics
     - Backward pass with gradient clipping
5. **Validation**: Evaluate on future unseen data
6. **Save Best**: Checkpoint model with lowest validation loss
7. **Financial Evaluation**: Compute Sharpe, drawdown, PnL, accuracy

## Output Files

After training, the following files are saved:

```
models/stacked_pinn/
├── stacked_pinn_best.pt        # Best model checkpoint
├── residual_pinn_best.pt       # (if using residual)
├── stacked_pinn_results.json   # Metrics and history
└── residual_pinn_results.json  # (if using residual)
```

**Results JSON structure:**
```json
{
  "model_type": "stacked",
  "val_metrics": {
    "mse": 0.0012,
    "mae": 0.0234,
    "rmse": 0.0346,
    "directional_accuracy": 0.567
  },
  "financial_metrics": {
    "total_return": 0.234,
    "sharpe_ratio": 1.45,
    "sortino_ratio": 1.89,
    "max_drawdown": -0.156,
    "calmar_ratio": 1.50,
    "directional_accuracy": 0.567,
    "win_rate": 0.543,
    "volatility": 0.189
  },
  "history": {
    "train_loss": [...],
    "val_loss": [...],
    "lambda_gbm": [...],
    "lambda_ou": [...]
  }
}
```

## Example: Complete Training Run

```python
import torch
from src.models.stacked_pinn import StackedPINN
from src.training.curriculum import CurriculumScheduler

# Load your return-based features
# X_train, y_train, returns_train = ...

# Create model
model = StackedPINN(
    input_dim=10,
    encoder_dim=128,
    lstm_hidden_dim=128,
    num_encoder_layers=2,
    num_rnn_layers=2,
    prediction_hidden_dim=64,
    dropout=0.2,
    lambda_gbm=0.0,  # Start at 0
    lambda_ou=0.0    # Start at 0
)

# Curriculum scheduler
curriculum = CurriculumScheduler(
    initial_lambda_gbm=0.0,
    final_lambda_gbm=0.1,
    initial_lambda_ou=0.0,
    final_lambda_ou=0.1,
    warmup_epochs=10,
    total_epochs=100,
    strategy='cosine'
)

# Training loop
for epoch in range(100):
    weights = curriculum.step(epoch)

    # Update model weights
    model.lambda_gbm = weights['lambda_gbm']
    model.lambda_ou = weights['lambda_ou']

    # Training...
    # (See train_stacked_pinn.py for complete implementation)
```

## Advantages of This Approach

1. **Physics-Informed Learning**: Soft constraints guide model toward realistic behavior
2. **Multi-Task Learning**: Simultaneously learns returns and direction
3. **Curriculum Training**: Stable convergence from data → physics
4. **Return-Based**: Avoids price-level bias, focuses on changes
5. **Walk-Forward Validation**: Realistic time series evaluation
6. **Financial Realism**: Transaction costs, comprehensive metrics
7. **Flexible Architecture**: Supports both stacked and residual variants

## Performance Expectations

Typical results on financial time series:

- **Directional Accuracy**: 55-60% (above 50% random baseline)
- **Sharpe Ratio**: 1.0-2.0 (depending on market conditions)
- **Max Drawdown**: -10% to -20%
- **Win Rate**: 50-55%

Physics constraints help:
- Reduce overfitting on training data
- Improve generalization to validation period
- Enforce realistic dynamics (trend + mean reversion)
- Stabilize training (less divergence)

## Troubleshooting

**If physics loss dominates:**
- Reduce `final_lambda_gbm` and `final_lambda_ou` (e.g., 0.01-0.05)
- Increase warmup epochs
- Use slower curriculum (linear instead of exponential)

**If model ignores physics:**
- Increase `final_lambda_gbm` and `final_lambda_ou` (e.g., 0.5-1.0)
- Use faster curriculum (exponential or step)
- Check that return sequences are properly extracted

**If training unstable:**
- Increase warmup epochs (20-30)
- Use cosine curriculum (smoothest)
- Check gradient clipping (default: 1.0)
- Reduce learning rate

## References

- **Physics-Informed Neural Networks**: Raissi et al. (2019)
- **Geometric Brownian Motion**: Black-Scholes model
- **Ornstein-Uhlenbeck Process**: Mean-reverting stochastic process
- **Curriculum Learning**: Bengio et al. (2009)
- **Walk-Forward Validation**: Standard in quantitative finance

## License

See main project LICENSE file.
