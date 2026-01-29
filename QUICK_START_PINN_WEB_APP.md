# PINN Web App - Quick Start Guide

## 🚀 Launch in 3 Steps

### Step 1: Train Models (if not already done)

```bash
# Train all 6 basic PINN variants (takes ~1-2 hours)
python src/training/train_pinn_variants.py --epochs 100

# Train Stacked PINN (takes ~30-60 min)
python src/training/train_stacked_pinn.py --model-type stacked --epochs 100

# Train Residual PINN (takes ~30-60 min)
python src/training/train_stacked_pinn.py --model-type residual --epochs 100
```

### Step 2: Launch Dashboard

```bash
# Easy way (one command)
./launch_pinn_dashboard.sh

# Or direct launch
streamlit run src/web/app.py
```

### Step 3: Navigate to "PINN Comparison"

Open browser at `http://localhost:8501` → Click "PINN Comparison" in sidebar

---

## 📊 What You'll See

### All 8 PINN Models

1. **Baseline** (no physics)
2. **Pure GBM** (trend)
3. **Pure OU** (mean-reversion)
4. **Pure Black-Scholes** (no-arbitrage)
5. **GBM+OU Hybrid** (balanced)
6. **Global Constraint** (all physics)
7. **StackedPINN** (advanced architecture)
8. **ResidualPINN** (physics correction)

### Comprehensive Metrics

**Risk-Adjusted Performance:**
- Sharpe Ratio
- Sortino Ratio
- Volatility

**Capital Preservation:**
- Maximum Drawdown
- Drawdown Duration
- Calmar Ratio

**Trading Viability:**
- Annualized Return
- Profit Factor
- Win Rate

**Signal Quality:**
- Directional Accuracy
- Precision / Recall
- Information Coefficient

**Robustness:**
- Rolling Window Stability
- Coefficient of Variation
- Consistency Score

---

## 🎯 How to Pick the Best Model

### Decision Criteria (Priority Order)

1. ✅ **Sharpe Ratio > 1.0**
   - Risk-adjusted returns
   - Higher = better

2. ✅ **Max Drawdown > -20%**
   - Capital preservation
   - Less negative = better

3. ✅ **Sharpe CV < 0.5**
   - Stability across time
   - Lower = more stable

4. ✅ **Directional Accuracy > 55%**
   - Signal quality
   - Above 50% random baseline

5. ✅ **Profit Factor > 1.5**
   - Trading viability
   - Higher = more profitable

### Example Decision

```
Model A: Sharpe 1.8, Max DD -15%, CV 0.3, DirAcc 58%  ← BEST
Model B: Sharpe 2.2, Max DD -35%, CV 0.8, DirAcc 62%  ← Too risky/unstable
Model C: Sharpe 0.8, Max DD -10%, CV 0.2, DirAcc 52%  ← Low returns
```

**Winner: Model A** - Good risk-adjusted returns, acceptable drawdown, stable, good accuracy

---

## 📈 Key Pages

### 1. Metrics Comparison (Main Analysis)

**4 Tabs:**
- Risk-Adjusted Performance
- Capital Preservation
- Trading Viability
- Signal Quality

**What to Check:**
- Green highlighted = best performer
- Compare across all models
- Look for consistent winners

### 2. Rolling Performance (Robustness)

**What to Check:**
- Low CV = stable performance
- High consistency = reliable across time
- Low regime sensitivity = adapts to conditions

### 3. Training History (Curriculum Learning)

**What to Check:**
- Smooth loss curves (no divergence)
- λ weights gradually increasing (curriculum)
- No overfitting (train/val gap small)

---

## 💡 Pro Tips

### ✓ DO

- Check multiple metrics, not just one
- Prioritize stability over peak performance
- Consider transaction costs (built-in)
- Look at rolling window analysis
- Compare apples-to-apples (same timeframe)

### ✗ DON'T

- Rely on a single metric
- Ignore maximum drawdown
- Trust in-sample performance only
- Overlook high coefficient of variation
- Pick model based on return alone

---

## 🔍 Quick Metric Reference

| Metric | Good Value | What It Means |
|--------|-----------|---------------|
| **Sharpe** | > 1.0 | Return per unit of risk |
| **Sortino** | > 1.5 | Return per unit of downside risk |
| **Max Drawdown** | > -20% | Worst peak-to-trough loss |
| **Calmar** | > 0.5 | Return per unit of drawdown |
| **Annual Return** | > 10% | Yearly profit |
| **Dir. Accuracy** | > 55% | Correct direction predictions |
| **Profit Factor** | > 1.5 | Total profit / total loss |
| **IC** | > 0.05 | Prediction-reality correlation |
| **Sharpe CV** | < 0.5 | Stability (lower = more stable) |
| **Consistency** | > 70% | % profitable windows |

---

## 🚨 Common Issues

### "No results found"

**Solution:** Train models first
```bash
python src/training/train_pinn_variants.py --epochs 100
```

### "Dashboard not loading"

**Solution:** Install dependencies
```bash
pip install streamlit plotly pandas numpy
```

### "Port already in use"

**Solution:** Use different port
```bash
streamlit run src/web/app.py --server.port 8502
```

---

## 📚 Full Documentation

- **Complete Guide:** `PINN_WEB_APP_GUIDE.md`
- **Implementation Details:** `COMPREHENSIVE_PINN_SYSTEM_SUMMARY.md`
- **PINN Architectures:** `STACKED_PINN_README.md`

---

## 🎓 When to Use Each PINN Variant

| Market Condition | Recommended Model |
|------------------|-------------------|
| **Trending (Bull/Bear)** | Pure GBM or StackedPINN |
| **Range-Bound** | Pure OU or GBM+OU Hybrid |
| **Uncertain/Mixed** | GBM+OU Hybrid or Global |
| **Derivative Pricing** | Pure Black-Scholes |
| **No Assumptions** | Baseline |
| **Complex Dynamics** | StackedPINN or ResidualPINN |

---

## ✅ Quick Checklist

Before making a decision:

- [ ] Checked Sharpe ratio (> 1.0)
- [ ] Verified max drawdown (> -20%)
- [ ] Reviewed rolling stability (CV < 0.5)
- [ ] Confirmed directional accuracy (> 55%)
- [ ] Examined training convergence
- [ ] Compared across all available models
- [ ] Considered market regime

---

## 🎉 You're Ready!

1. Train models
2. Launch dashboard
3. Navigate to "PINN Comparison"
4. Check metrics in priority order
5. Select best model based on criteria

**Remember:** Prioritize risk-adjusted metrics and stability over raw performance!

---

**For academic research only - not financial advice.**
