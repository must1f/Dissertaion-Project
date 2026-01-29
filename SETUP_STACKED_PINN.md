# Setup Guide for Stacked PINN Implementation

## ✅ Implementation Status: COMPLETE

All stacked PINN components have been successfully implemented and are ready to use!

---

## 📦 Installation

### Step 1: Install Dependencies

The project uses PyTorch, NumPy, and other scientific computing libraries. Install them using:

```bash
# Navigate to project directory
cd /Users/mustif/Documents/GitHub/Dissertaion-Project

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

Run the verification script to ensure everything is properly installed:

```bash
python3 verify_stacked_pinn.py
```

Expected output:
```
================================================================================
STACKED PINN IMPLEMENTATION VERIFICATION
================================================================================

✓ Checking model imports...
  ✓ StackedPINN model classes imported successfully
✓ Checking training imports...
  ✓ Curriculum schedulers imported successfully
✓ Checking walk-forward validation...
  ✓ Walk-forward validation imported successfully
✓ Checking financial metrics...
  ✓ Financial metrics imported successfully

================================================================================
✓ ALL CHECKS PASSED
================================================================================
```

---

## 🚀 Quick Start

### Option 1: Run Simple Example (Synthetic Data)

```bash
python3 examples/stacked_pinn_example.py
```

This will:
- Generate synthetic return-based time series data
- Train a StackedPINN model with curriculum learning
- Display financial metrics (Sharpe, drawdown, accuracy)
- Complete in ~2-3 minutes on CPU

### Option 2: Train on Real Financial Data

```bash
python3 src/training/train_stacked_pinn.py \
    --model-type stacked \
    --epochs 100 \
    --warmup-epochs 10 \
    --final-lambda-gbm 0.1 \
    --final-lambda-ou 0.1 \
    --curriculum-strategy cosine
```

This will:
- Fetch real stock data (configured in your config)
- Prepare return-based features
- Train with curriculum learning
- Save best model and results
- Evaluate financial performance

---

## 📊 What Was Implemented

### 1. Model Architectures (src/models/stacked_pinn.py)

**StackedPINN**:
- PhysicsEncoder for feature-level encoding
- Parallel LSTM + GRU heads with attention fusion
- Dense prediction head (return regression + direction classification)
- GBM and OU physics losses on returns

**ResidualPINN**:
- Base LSTM/GRU model
- Physics-informed correction network
- Final prediction = base + physics correction

### 2. Training Infrastructure

**Curriculum Learning** (src/training/curriculum.py):
- Gradually increases physics weights from 0 → final value
- Strategies: linear, exponential, cosine, step
- Adaptive variant based on validation performance

**Walk-Forward Validation** (src/training/walk_forward.py):
- Expanding and rolling window validation
- Prevents look-ahead bias in time series
- Date-based and index-based splitting

**Training Script** (src/training/train_stacked_pinn.py):
- Complete end-to-end training pipeline
- Return-based feature preparation
- Combined loss: prediction + physics
- GPU support
- Model checkpointing

### 3. Evaluation Metrics (src/evaluation/financial_metrics.py)

- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk focus)
- Maximum Drawdown
- Calmar Ratio
- Directional Accuracy
- Win Rate
- Annualized Volatility
- Strategy returns with transaction costs

---

## 🎯 Usage Examples

### Example 1: Train Stacked PINN

```bash
python3 src/training/train_stacked_pinn.py \
    --model-type stacked \
    --epochs 100 \
    --warmup-epochs 10 \
    --final-lambda-gbm 0.1 \
    --final-lambda-ou 0.1 \
    --curriculum-strategy cosine
```

### Example 2: Train Residual PINN

```bash
python3 src/training/train_stacked_pinn.py \
    --model-type residual \
    --epochs 100 \
    --warmup-epochs 10 \
    --final-lambda-gbm 0.1 \
    --final-lambda-ou 0.1 \
    --curriculum-strategy cosine
```

### Example 3: Compare Curriculum Strategies

```bash
# Cosine (smooth, recommended)
python3 src/training/train_stacked_pinn.py --curriculum-strategy cosine

# Linear
python3 src/training/train_stacked_pinn.py --curriculum-strategy linear

# Exponential (faster increase)
python3 src/training/train_stacked_pinn.py --curriculum-strategy exponential

# Step (discrete jumps)
python3 src/training/train_stacked_pinn.py --curriculum-strategy step
```

### Example 4: Python API Usage

```python
import torch
from src.models.stacked_pinn import StackedPINN
from src.training.curriculum import CurriculumScheduler

# Create model
model = StackedPINN(
    input_dim=10,
    encoder_dim=128,
    lstm_hidden_dim=128,
    lambda_gbm=0.0,
    lambda_ou=0.0
)

# Create curriculum
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
    # Update physics weights
    weights = curriculum.step(epoch)
    model.lambda_gbm = weights['lambda_gbm']
    model.lambda_ou = weights['lambda_ou']

    # Train...
    # (see train_stacked_pinn.py for complete example)
```

---

## 📁 Output Files

After training, you'll find:

```
models/stacked_pinn/
├── stacked_pinn_best.pt         # Best model checkpoint
├── stacked_pinn_results.json    # Metrics and training history
├── residual_pinn_best.pt        # (if trained residual model)
└── residual_pinn_results.json   # (if trained residual model)
```

**Results JSON example**:
```json
{
  "model_type": "stacked",
  "val_metrics": {
    "mse": 0.0012,
    "rmse": 0.0346,
    "mae": 0.0234,
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

---

## 🔧 Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `stacked` | Model architecture: `stacked` or `residual` |
| `--epochs` | `100` | Total training epochs |
| `--warmup-epochs` | `10` | Warmup epochs with λ=0 |
| `--final-lambda-gbm` | `0.1` | Final GBM physics weight |
| `--final-lambda-ou` | `0.1` | Final OU physics weight |
| `--curriculum-strategy` | `cosine` | Curriculum strategy |

### Model Hyperparameters

Edit in code or create config file:

```python
StackedPINN(
    input_dim=10,              # Number of input features
    encoder_dim=128,           # Encoder hidden dimension
    lstm_hidden_dim=128,       # LSTM/GRU hidden dimension
    num_encoder_layers=2,      # Encoder depth
    num_rnn_layers=2,          # LSTM/GRU depth
    prediction_hidden_dim=64,  # Prediction head dimension
    dropout=0.2,               # Dropout rate
    lambda_gbm=0.0,            # Initial GBM weight
    lambda_ou=0.0              # Initial OU weight
)
```

---

## 🐛 Troubleshooting

### Dependencies Not Installed

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Physics Loss Too Large

**Error**: Training unstable, loss diverging

**Solution**: Reduce physics weights
```bash
python3 src/training/train_stacked_pinn.py \
    --final-lambda-gbm 0.01 \
    --final-lambda-ou 0.01
```

### Model Not Learning

**Error**: Validation loss not improving

**Solution**:
- Increase warmup epochs: `--warmup-epochs 20`
- Try different curriculum: `--curriculum-strategy linear`
- Check data quality and feature engineering

### GPU Not Used

**Error**: Training on CPU when GPU available

**Solution**: Check config file, ensure `device='cuda'` in config

---

## 📚 Documentation

- **STACKED_PINN_README.md**: Comprehensive usage guide
- **STACKED_PINN_IMPLEMENTATION_SUMMARY.md**: Implementation details
- **SETUP_STACKED_PINN.md** (this file): Setup and installation

---

## ✅ Verification Checklist

After installation, verify:

- [ ] Dependencies installed: `pip list | grep torch`
- [ ] Verification script passes: `python3 verify_stacked_pinn.py`
- [ ] Example runs: `python3 examples/stacked_pinn_example.py`
- [ ] Training script accessible: `python3 src/training/train_stacked_pinn.py --help`
- [ ] Documentation readable: `cat STACKED_PINN_README.md`

---

## 🎉 You're Ready!

Once dependencies are installed and verification passes, you can:

1. **Run the example** to see it in action
2. **Train on your data** with the full training script
3. **Experiment** with different architectures and hyperparameters
4. **Evaluate** using comprehensive financial metrics

For detailed information, see **STACKED_PINN_README.md**.

---

## 📞 Support

- **Documentation**: See STACKED_PINN_README.md
- **Examples**: Check examples/stacked_pinn_example.py
- **Code**: All source code is in src/ directory
- **Verification**: Run verify_stacked_pinn.py

Enjoy using your physics-informed financial forecasting system!
