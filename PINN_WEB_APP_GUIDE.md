# PINN Web Application - Comprehensive Guide

## Overview

The PINN Web Application provides interactive visualization and comparison of **8 different Physics-Informed Neural Network variants** for financial forecasting, with comprehensive financial metrics focused on real-world trading viability.

---

## 🚀 Quick Start

### Launch the Web App

```bash
cd /Users/mustif/Documents/GitHub/Dissertaion-Project

# Main app
streamlit run src/web/app.py

# Or dedicated PINN dashboard
streamlit run src/web/pinn_dashboard.py
```

Access at: `http://localhost:8501`

---

## 📊 Available PINN Models

### Basic PINN Variants (Physics Constraint Variations)

| Model | Physics Constraints | Best For | Lambda Weights |
|-------|-------------------|----------|----------------|
| **Baseline** | None (pure data-driven) | Benchmark, maximum flexibility | λ = 0 |
| **Pure GBM** | Geometric Brownian Motion | Trending markets (bull/bear) | λ_GBM = 0.1 |
| **Pure OU** | Ornstein-Uhlenbeck | Mean-reverting, range-bound | λ_OU = 0.1 |
| **Pure Black-Scholes** | No-arbitrage PDE | Derivative pricing, efficient markets | λ_BS = 0.1 |
| **GBM+OU Hybrid** | Trend + Mean-Reversion | General forecasting, balanced | λ_GBM = 0.05, λ_OU = 0.05 |
| **Global Constraint** | All equations combined | Maximum regularization | λ_GBM = 0.05, λ_BS = 0.03, λ_OU = 0.05 |

### Advanced PINN Architectures

| Model | Architecture | Features | Use Case |
|-------|-------------|----------|----------|
| **StackedPINN** | Physics Encoder → Parallel LSTM/GRU → Dense Head | • Curriculum learning<br>• Attention fusion<br>• Multi-task (regression + classification) | Multi-scale features, complex dynamics |
| **ResidualPINN** | Base LSTM/GRU + Physics Correction | • Residual learning<br>• Physics-constrained corrections | When base model is strong but needs physics guidance |

---

## 📈 Comprehensive Financial Metrics

### 1. Risk-Adjusted Performance

**Most Important for Neural Network Assessment**

| Metric | Formula | Interpretation | Target Value |
|--------|---------|----------------|--------------|
| **Sharpe Ratio** | (Return - RiskFree) / Volatility | Risk-adjusted return | > 1.0 (good), > 2.0 (excellent) |
| **Sortino Ratio** | (Return - RiskFree) / Downside Volatility | Downside risk-adjusted return | > 1.5 (good) |

**Why These Matter:**
- Raw accuracy doesn't account for risk
- High returns with high volatility = risky strategy
- Sharpe/Sortino normalize by risk taken

### 2. Capital Preservation

**Critical for Real Trading**

| Metric | Formula | Interpretation | Target Value |
|--------|---------|----------------|--------------|
| **Maximum Drawdown** | Max(Peak - Trough) / Peak | Worst loss from peak | < -20% (acceptable) |
| **Drawdown Duration** | Avg time in drawdown | Recovery time | < 30 days (fast recovery) |
| **Calmar Ratio** | Annual Return / \|Max Drawdown\| | Return per unit of drawdown | > 0.5 (good) |

**Why These Matter:**
- Large drawdowns can force strategy abandonment
- Long drawdowns test investor patience
- Capital preservation is as important as returns

### 3. Trading Viability

**Real-World Profitability**

| Metric | Formula | Interpretation | Target Value |
|--------|---------|----------------|--------------|
| **Transaction-Cost-Adjusted PnL** | Gross PnL - Transaction Costs | Net profit after costs | > 0 (profitable) |
| **Annualized Return** | (Total Return)^(1/Years) - 1 | Yearly return | > 10% (good) |
| **Profit Factor** | Gross Profit / Gross Loss | Profit per dollar lost | > 1.5 (good) |

**Why These Matter:**
- Many profitable backtests fail with transaction costs
- Annualized return enables fair comparison
- Profit factor shows win/loss balance

### 4. Signal Quality

**Prediction Accuracy**

| Metric | Formula | Interpretation | Target Value |
|--------|---------|----------------|--------------|
| **Directional Accuracy** | % of correct sign predictions | Direction correctness | > 55% (above random) |
| **Precision** | TP / (TP + FP) | Accuracy of positive predictions | > 0.6 (good) |
| **Recall** | TP / (TP + FN) | Capture rate of positive cases | > 0.6 (good) |
| **Information Coefficient (IC)** | Correlation(Pred, Actual) | Prediction-reality correlation | > 0.05 (good), > 0.10 (excellent) |

**Why These Matter:**
- Directional accuracy > 50% = better than coin flip
- Precision/recall balance false positives vs false negatives
- IC measures linear relationship between predictions and reality

### 5. Robustness & Stability

**Out-of-Sample Performance**

| Metric | Description | Interpretation | Target Value |
|--------|-------------|----------------|--------------|
| **Rolling Window Sharpe** | Sharpe across windows | Consistency over time | Low CV (< 0.5) |
| **Walk-Forward Stability** | Performance across folds | Regime independence | High consistency (> 70%) |
| **Coefficient of Variation (CV)** | Std / Mean | Relative stability | < 0.5 (stable) |
| **Regime Sensitivity** | Variance across market conditions | Adaptability | Low variance preferred |

**Why These Matter:**
- Models must perform consistently, not just in backtests
- High variance = overfitting or regime dependence
- Walk-forward validation simulates realistic deployment

---

## 🖥️ Web App Navigation

### 1. Home Page

- Overview of all 8 PINN variants
- Quick comparison guide
- When to use each model
- System architecture

### 2. PINN Comparison

**Most Important Page for Model Evaluation**

#### Metrics Comparison Tab
- Side-by-side comparison of all models
- 4 sub-tabs:
  - **Risk-Adjusted Performance**: Sharpe, Sortino, Volatility
  - **Capital Preservation**: Drawdown, duration, Calmar
  - **Trading Viability**: Returns, profit factor, win rate
  - **Signal Quality**: Directional accuracy, precision, recall, IC

Features:
- Highlighted best/worst performers
- Interactive bar charts
- Sortable tables
- Export to CSV

#### Rolling Performance Tab
- Rolling out-of-sample analysis
- Stability metrics across time windows
- Regime sensitivity detection
- Violation scores (physics adherence)

Features:
- Window-by-window performance
- Consistency metrics
- CV analysis
- Overfitting detection

#### Training History Tab
- Loss curves (train/validation)
- Directional accuracy over epochs
- **Curriculum learning visualization** (λ weights)
- Learning rate schedule

Features:
- Multi-panel charts
- Physics weight evolution
- Convergence analysis

### 3. Data Explorer

- Interactive price charts
- Candlestick visualization
- Historical data inspection
- Technical indicator preview

### 4. Model Comparison

- Traditional ML metrics (RMSE, MAE, R²)
- Financial metrics comparison
- Cross-model benchmarking
- Training history viewer

### 5. Backtesting (Coming Soon)

- Strategy simulation
- Portfolio performance
- Risk management

### 6. Live Demo (Coming Soon)

- Real-time predictions
- Live signal generation
- Paper trading simulation

---

## 🎯 How to Use the Web App

### Step 1: Train Models

Train all PINN variants:

```bash
# Train all 6 basic PINN variants (Baseline, GBM, OU, BS, Hybrid, Global)
python src/training/train_pinn_variants.py --epochs 100

# Train Stacked PINN with curriculum learning
python src/training/train_stacked_pinn.py --model-type stacked --epochs 100 --curriculum-strategy cosine

# Train Residual PINN
python src/training/train_stacked_pinn.py --model-type residual --epochs 100
```

### Step 2: Launch Web App

```bash
streamlit run src/web/app.py
```

### Step 3: Navigate to "PINN Comparison"

This is the main page for comprehensive model evaluation.

### Step 4: Analyze Results

**Recommended Analysis Flow:**

1. **Metrics Comparison Tab**
   - Check Sharpe ratio (> 1.0 is good)
   - Verify max drawdown (< -20% acceptable)
   - Look for directional accuracy > 55%
   - Identify best overall performer

2. **Rolling Performance Tab**
   - Check Sharpe CV (lower = more stable)
   - Verify consistency (% windows profitable)
   - Look for low regime sensitivity
   - Identify robust models

3. **Training History Tab**
   - Verify convergence (no divergence)
   - Check for overfitting (train/val gap)
   - Observe curriculum learning (λ increases)
   - Ensure smooth training

### Step 5: Select Best Model

**Decision Criteria (in order of importance):**

1. **Sharpe Ratio** > 1.0 (risk-adjusted returns)
2. **Max Drawdown** < -20% (capital preservation)
3. **Rolling Stability** (low CV, high consistency)
4. **Directional Accuracy** > 55% (signal quality)
5. **Profit Factor** > 1.5 (trading viability)

**Example Decision:**

```
Model A: Sharpe 1.8, Max DD -15%, CV 0.4, DirAcc 58%  ✓ Best
Model B: Sharpe 2.2, Max DD -35%, CV 0.7, DirAcc 62%  ✗ Too risky, unstable
Model C: Sharpe 0.8, Max DD -10%, CV 0.2, DirAcc 52%  ✗ Low returns
```

**Model A wins**: Good risk-adjusted returns, acceptable drawdown, stable, good accuracy.

---

## 📊 Interpreting Results

### What Makes a Good PINN Model?

**Excellent Model:**
- Sharpe > 2.0
- Max Drawdown > -15%
- Directional Accuracy > 60%
- Sharpe CV < 0.3 (stable across windows)
- Consistency > 80% (most windows profitable)

**Good Model:**
- Sharpe > 1.0
- Max Drawdown > -20%
- Directional Accuracy > 55%
- Sharpe CV < 0.5
- Consistency > 70%

**Poor Model:**
- Sharpe < 0.5 (underperforming risk-free rate)
- Max Drawdown < -30% (excessive risk)
- Directional Accuracy < 52% (barely better than random)
- High CV (unstable)
- Low consistency (regime-dependent)

### Physics Constraint Trade-offs

| Constraint | Advantage | Disadvantage |
|-----------|-----------|--------------|
| **No Physics (Baseline)** | Maximum flexibility | May overfit, no theoretical grounding |
| **GBM Only** | Good for trends | Cannot model mean reversion |
| **OU Only** | Good for mean reversion | Cannot model trends |
| **Black-Scholes** | No-arbitrage enforced | Computationally expensive, restrictive |
| **Hybrid** | Balanced | May have conflicting constraints |
| **Global** | Maximum regularization | May underfit if assumptions violated |

### Curriculum Learning Benefits

**Why Gradually Increase Physics Weights?**

```
Epoch 0-10:    λ = 0.0  → Pure data fitting, model learns patterns
Epoch 10-50:   λ = 0→0.1 → Gradual physics introduction, smooth transition
Epoch 50-100:  λ = 0.1  → Full physics constraints, regularized predictions
```

**Benefits:**
- Stable convergence (avoids early divergence)
- Better final performance (data + physics)
- Reduced training time (easier optimization)

---

## 🔍 Advanced Features

### Rolling Window Analysis

**Purpose:** Detect overfitting and regime sensitivity

**How It Works:**
1. Split test data into overlapping windows (e.g., 63 days)
2. Compute metrics for each window
3. Calculate mean, std, CV across windows
4. Identify inconsistent performance

**What to Look For:**
- Low CV (< 0.5) = stable
- High consistency (> 70%) = robust
- Low variance = regime-independent

### Walk-Forward Validation

**Purpose:** Simulate realistic trading conditions

**How It Works:**
1. Train on historical data
2. Validate on future unseen data
3. Move window forward
4. Repeat for multiple folds

**Prevents:**
- Look-ahead bias
- Overfitting to specific time periods
- Unrealistic backtest results

### Violation Scores

**Purpose:** Measure physics constraint adherence

**Formula:**
```
Violation Score = Physics Loss / (Data Loss + ε)
```

**Interpretation:**
- **Low score** (< 0.1): Model satisfies physics well
- **High score** (> 1.0): Model violates assumptions
- **Zero**: Baseline (no constraints)

---

## 🎓 Best Practices

### 1. Always Check Multiple Metrics

Don't rely on a single metric. A model with:
- High Sharpe BUT high max drawdown = risky
- High accuracy BUT low Sharpe = not profitable
- High returns BUT high CV = unreliable

### 2. Prioritize Stability Over Peak Performance

A model with:
- Sharpe 1.5 consistently across windows
is better than:
- Sharpe 3.0 in some windows, 0.0 in others

### 3. Validate on Out-of-Sample Data

In-sample performance is meaningless. Always check:
- Validation set
- Test set
- Rolling windows
- Walk-forward folds

### 4. Consider Transaction Costs

Many strategies fail with realistic costs (0.1-0.5% per trade).
Always use transaction-cost-adjusted PnL.

### 5. Understand Your Market Regime

- **Trending market**: GBM-based models
- **Range-bound**: OU-based models
- **Uncertain**: Hybrid or Global
- **No assumptions**: Baseline

---

## 🚨 Common Pitfalls

### ❌ Overfitting

**Signs:**
- High training accuracy, poor validation
- High CV across rolling windows
- Good in-sample, poor out-of-sample

**Solution:**
- Use more physics regularization
- Increase curriculum warmup
- Simplify model architecture

### ❌ Look-Ahead Bias

**Signs:**
- Unrealistically high metrics
- Using future information in features
- Not respecting temporal order

**Solution:**
- Use walk-forward validation
- Check feature construction
- Ensure proper data splitting

### ❌ Regime Overfitting

**Signs:**
- High performance in training period
- Poor performance in different market conditions
- High regime sensitivity

**Solution:**
- Train on diverse market conditions
- Use multiple time periods
- Check rolling window stability

---

## 📞 Support

### Documentation
- **STACKED_PINN_README.md**: Detailed architecture guide
- **SETUP_STACKED_PINN.md**: Installation instructions
- **STACKED_PINN_IMPLEMENTATION_SUMMARY.md**: Implementation details
- **PINN_WEB_APP_GUIDE.md** (this file): Web app guide

### Troubleshooting

**No results showing:**
```bash
# Train models first
python src/training/train_pinn_variants.py --epochs 100
python src/training/train_stacked_pinn.py --model-type stacked
```

**Dashboard not loading:**
```bash
# Check dependencies
pip install streamlit plotly pandas numpy

# Restart app
streamlit run src/web/app.py
```

**Metrics not displaying:**
- Ensure results JSON files exist in `results/` directory
- Check file paths in dashboard code
- Verify metric keys match expected format

---

## 🎉 Summary

The PINN Web App provides comprehensive comparison of 8 PINN variants with:

✓ **Complete Financial Metrics**: Sharpe, Sortino, drawdown, profit factor, IC
✓ **Robustness Analysis**: Rolling windows, walk-forward stability
✓ **Interactive Visualizations**: Charts, tables, comparisons
✓ **Curriculum Learning Tracking**: Physics weight evolution
✓ **Real-World Focus**: Transaction costs, regime sensitivity

**Most Important Metrics for NN Assessment:**
1. Sharpe Ratio (risk-adjusted returns)
2. Maximum Drawdown (capital preservation)
3. Rolling Window Stability (robustness)
4. Directional Accuracy (signal quality)

Use the app to identify the best PINN variant for your specific use case and market conditions!

---

**For academic research only - not financial advice.**
