# Evaluation Guide - No Retraining Required!

## ✅ Your Models Are Already Trained

You have 6 PINN variant models already trained:
- `pinn_baseline_best.pt` - Pure data-driven
- `pinn_gbm_best.pt` - Geometric Brownian Motion
- `pinn_ou_best.pt` - Ornstein-Uhlenbeck
- `pinn_black_scholes_best.pt` - Black-Scholes PDE
- `pinn_gbm_ou_best.pt` - GBM+OU Hybrid
- `pinn_global_best.pt` - All physics combined

## 🔍 What's Missing

Your trained models currently have:
- ✅ Basic ML metrics (RMSE, MAE, R², Directional Accuracy)
- ✗ Comprehensive financial metrics (Sharpe, Sortino, drawdown, profit factor, etc.)

## 🚀 Solution: Re-Evaluate (No Retraining!)

The `evaluate_existing_models.py` script:
1. **Loads** your existing trained models
2. **Runs** them on the test set
3. **Computes** comprehensive financial metrics
4. **Saves** updated results

**No training happens** - just inference + metrics computation!

## 📊 Running the Evaluation

### Currently Running

The evaluation for PINN models is already running in the background:

```bash
# Check progress
tail -f eval_pinn.log

# Or check if it's still running
ps aux | grep evaluate_existing_models
```

### Expected Timeline

- **Per model:** ~2-5 minutes
- **All 6 PINN models:** ~12-30 minutes

The script is currently:
1. Loading test data ✓
2. For each PINN model:
   - Load model checkpoint
   - Generate predictions
   - Compute 15+ financial metrics
   - Save comprehensive results

### What You'll Get

Updated results files with comprehensive metrics:
```
results/
├── pinn_baseline_results.json  ← Comprehensive metrics
├── pinn_gbm_results.json       ← Comprehensive metrics
├── pinn_ou_results.json        ← Comprehensive metrics
├── pinn_black_scholes_results.json
├── pinn_gbm_ou_results.json
└── pinn_global_results.json
```

Each file will contain:
- **ML Metrics:** MSE, MAE, RMSE, R², MAPE
- **Risk-Adjusted:** Sharpe ratio, Sortino ratio, volatility
- **Capital Preservation:** Max drawdown, drawdown duration, Calmar ratio
- **Trading Viability:** Annualized return, profit factor, win rate
- **Signal Quality:** Directional accuracy, precision, recall, IC
- **Robustness:** Rolling window performance, stability metrics

## 📈 After Evaluation Completes

### 1. Check Completion

```bash
# Check if evaluation is done
tail -20 eval_pinn.log

# Look for:
# "EVALUATION SUMMARY"
# "✓ Evaluated: 6"
```

### 2. Launch Dashboard

```bash
streamlit run src/web/app.py
```

### 3. View Results

Navigate to **"PINN Comparison"** in the dashboard:
- ✅ All 6 models will be detected
- ✅ All tabs will have data:
  - ML Metrics (RMSE, MAE, R², Dir Acc)
  - Risk-Adjusted (Sharpe, Sortino)
  - Capital Preservation (Drawdown, Calmar)
  - Trading Viability (Returns, Profit Factor)
  - Signal Quality (Precision, Recall, IC)
- ✅ All graphs will display properly

## 🎯 Evaluating Other Models

Once PINN evaluation completes, you can also evaluate baseline models:

```bash
# Evaluate baseline models (LSTM, GRU, BiLSTM, Transformer)
python evaluate_existing_models.py --models baseline

# Or evaluate all models at once
python evaluate_existing_models.py --models all

# Skip models that already have comprehensive results
python evaluate_existing_models.py --models all --skip-existing
```

## 🔧 Troubleshooting

### Evaluation Stuck?

```bash
# Check if it's running
ps aux | grep evaluate_existing_models

# Check logs for errors
tail -50 eval_pinn.log
```

### Need to Restart?

```bash
# Kill current evaluation
pkill -f evaluate_existing_models

# Restart
python evaluate_existing_models.py --models pinn
```

### Evaluation Failed?

Check the log file for errors:
```bash
cat eval_pinn.log | grep -i error
```

Common issues:
- **Missing data:** Run `python -m src.data.fetcher` first
- **Out of memory:** Reduce batch size in config
- **Model load error:** Check model file exists and isn't corrupted

## ✅ Summary

**You DON'T need to:**
- ❌ Retrain any models
- ❌ Re-fetch data (unless data is missing)
- ❌ Modify any code

**You just need to:**
- ✅ Wait for evaluation to complete (~12-30 min)
- ✅ Launch dashboard
- ✅ View comprehensive financial metrics

**Current Status:**
- 🔄 Evaluation running in background
- 📊 Results will be saved to `results/*_results.json`
- 🌐 Dashboard will show full metrics when complete

---

**Check progress:** `tail -f eval_pinn.log`
**When done:** `streamlit run src/web/app.py`
