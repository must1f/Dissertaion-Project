# Full Model Pipeline - Complete Training Guide

## 🚀 Overview

The **Full Model Pipeline** (Option 11 in `run.sh`) is the most comprehensive training option that executes end-to-end training for all 13 neural network models in the system, followed by unified evaluation and an interactive dashboard launch.

## 📊 What Gets Trained

### Phase 1: Infrastructure Setup
- **TimescaleDB** - Start database container (if Docker available)
- **Data Fetcher** - Fetch and store all financial data

### Phase 2: Baseline Models (5 models)
1. **LSTM** - Long Short-Term Memory network
2. **GRU** - Gated Recurrent Unit
3. **BiLSTM** - Bidirectional LSTM
4. **Attention LSTM** - LSTM with attention mechanism
5. **Transformer** - Multi-head self-attention

### Phase 3: PINN Variants (6 models)
1. **Baseline** - Pure data-driven (no physics)
2. **Pure GBM** - Geometric Brownian Motion (trend)
3. **Pure OU** - Ornstein-Uhlenbeck (mean-reversion)
4. **Pure Black-Scholes** - No-arbitrage PDE constraint
5. **GBM+OU Hybrid** - Combined trend & mean-reversion
6. **Global Constraint** - All physics equations combined

### Phase 4: Advanced PINN Architectures (2 models)
1. **StackedPINN** - Physics encoder + parallel LSTM/GRU + curriculum learning
2. **ResidualPINN** - Base model + physics-informed correction

### Phase 5: Unified Evaluation & Dashboard
- Compute 15+ comprehensive financial metrics for all models
- Launch interactive Streamlit dashboard with training status

---

## 🎯 How to Use

### Quick Start

```bash
# Launch the script
./run.sh

# Select option 11
Enter your choice [1-12]: 11

# Follow the prompts:
# - Enter epochs for baseline/PINN models (default: 100)
# - Enter epochs for advanced PINN models (default: 100)
```

### Expected Execution Time

| Phase | Models | Time Estimate (100 epochs) |
|-------|--------|----------------------------|
| Infrastructure | - | 1-2 minutes |
| Baseline Models | 5 | 20-40 minutes |
| PINN Variants | 6 | 30-60 minutes |
| Advanced PINN | 2 | 30-60 minutes |
| **Total** | **13** | **~1.5-3 hours** |

*Times vary based on:*
- Hardware (CPU/GPU)
- Dataset size
- Number of epochs
- System load

---

## 📈 What You Get

### 1. Trained Model Checkpoints

All models saved in `models/` directory:

```
models/
├── lstm_best.pt
├── gru_best.pt
├── bilstm_best.pt
├── attention_lstm_best.pt
├── transformer_best.pt
├── pinn_baseline_best.pt
├── pinn_gbm_best.pt
├── pinn_ou_best.pt
├── pinn_black_scholes_best.pt
├── pinn_gbm_ou_best.pt
├── pinn_global_best.pt
└── stacked_pinn/
    ├── stacked_pinn_best.pt
    └── residual_pinn_best.pt
```

### 2. Comprehensive Results

All evaluation results in `results/` directory:

```
results/
├── lstm_results.json
├── gru_results.json
├── ...
├── pinn_comparison/
│   ├── comparison_report.csv
│   ├── detailed_results.json
│   └── README_theory.md
└── stacked_pinn/
    ├── stacked_pinn_results.json
    └── residual_pinn_results.json
```

### 3. Interactive Dashboard

Automatic launch of Streamlit dashboard at `http://localhost:8501` with:

- **All Models Dashboard** - Complete model registry with training status
- **PINN Comparison** - Detailed physics-informed analysis
- **Metrics Comparison** - Side-by-side performance evaluation
- **Live Demo** - Real-time prediction interface

---

## 🎓 Understanding the Phases

### Phase 2: Baseline Models

**Purpose:** Establish pure machine learning baselines without domain knowledge

**Why different architectures?**
- **LSTM** - Handles long-term dependencies
- **GRU** - Faster alternative to LSTM
- **BiLSTM** - Captures forward & backward context
- **Attention LSTM** - Focuses on important time steps
- **Transformer** - State-of-the-art sequence modeling

**Expected output:** Models trained on pure historical patterns

### Phase 3: PINN Variants

**Purpose:** Incorporate financial physics into learning

**Why different physics constraints?**
- **Baseline** - Proves value of physics (should perform worst)
- **Pure GBM** - Best for trending markets (bull/bear)
- **Pure OU** - Best for mean-reverting assets
- **Black-Scholes** - Options pricing, no-arbitrage
- **GBM+OU** - Balanced approach for mixed regimes
- **Global** - Maximum physics guidance

**Expected output:** Models with domain knowledge embedded

### Phase 4: Advanced PINN

**Purpose:** State-of-the-art physics-informed architectures

**Why advanced architectures?**
- **StackedPINN** - Multi-stage learning with curriculum
- **ResidualPINN** - Corrects baseline predictions with physics

**Expected output:** Best performing models combining all techniques

---

## 📊 Comprehensive Metrics

All models evaluated with 15+ financial metrics:

### Risk-Adjusted Performance
- Sharpe Ratio (return per unit risk)
- Sortino Ratio (return per downside risk)
- Volatility

### Capital Preservation
- Maximum Drawdown
- Drawdown Duration
- Calmar Ratio

### Trading Viability
- Annualized Return
- Profit Factor (total profit / loss)
- Win Rate
- Transaction-cost-adjusted returns

### Signal Quality
- Directional Accuracy
- Precision / Recall / F1
- Information Coefficient (IC)

### Robustness
- Rolling window performance
- Sharpe CV (coefficient of variation)
- Consistency across time periods

---

## 🔍 Monitoring Progress

### Console Output

The pipeline provides real-time progress updates:

```bash
========================================
PHASE 2/5: Training Baseline Models (5 models)
========================================

========================================
Baseline Model 1/5: LSTM
========================================
[2026-01-27 22:30:00] Training lstm model...
Epoch 1/100: Train Loss: 0.0234, Val Loss: 0.0256
Epoch 2/100: Train Loss: 0.0198, Val Loss: 0.0221
...
[2026-01-27 22:45:00] ✓ lstm trained successfully (900s)

========================================
Baseline Model 2/5: GRU
========================================
...
```

### Log Files

Detailed logs saved with timestamps:

```bash
# Log file created automatically
run_20260127_223000.log

# Contains:
# - Debug messages
# - Training progress
# - Error messages
# - Timing information
```

---

## 🚨 Handling Failures

### Individual Model Failures

The pipeline is **robust to individual failures**:

```bash
[2026-01-27 22:45:00] Warning: bilstm training failed, continuing...
```

- Pipeline continues with remaining models
- Summary shows success/failure counts
- Failed models marked as untrained in dashboard

### Critical Failures

If critical infrastructure fails:

```bash
[2026-01-27 22:30:00] [ERROR] Data fetch failed
```

- Pipeline aborts immediately
- Check logs for error details
- Fix issue and restart from Option 11

### Common Issues

| Issue | Solution |
|-------|----------|
| `Docker not found` | Install Docker or skip database (data stored locally) |
| `Out of memory` | Reduce batch size or epochs |
| `CUDA out of memory` | Use CPU or reduce model size |
| `Port 8501 in use` | Stop existing Streamlit: `pkill -f streamlit` |
| `Virtual env not found` | Run `./setup.sh` first |

---

## 🎯 After Pipeline Completion

### 1. Review Dashboard

```bash
# Dashboard launches automatically
# Navigate to: http://localhost:8501

Sections to check:
✓ All Models Dashboard → Overview (training status)
✓ All Models Dashboard → Model List (13 models with ✅/⚪)
✓ All Models Dashboard → Metrics Comparison (5 categories)
✓ PINN Comparison (physics analysis)
```

### 2. Identify Best Model

**Decision criteria (priority order):**

1. ✅ **Sharpe Ratio > 1.0** (risk-adjusted returns)
2. ✅ **Max Drawdown > -20%** (capital preservation)
3. ✅ **Sharpe CV < 0.5** (stability over time)
4. ✅ **Directional Accuracy > 55%** (signal quality)
5. ✅ **Profit Factor > 1.5** (trading viability)

### 3. Export Results

```bash
# Results already saved as JSON
results/*_results.json

# Comparison tables available for download in dashboard
# Click "Download as CSV" buttons
```

---

## 🔧 Advanced Options

### Custom Epochs

```bash
# When prompted:
Enter number of epochs for baseline/PINN models (default: 100): 200
Enter number of epochs for advanced PINN models (default: 100): 150
```

### Partial Training

If you only need specific model types:

```bash
# Option 4: Train baseline models only
# Option 10: Train PINN variants only
# Option 11: Train everything
```

### Resume Training

Models automatically save checkpoints. To resume:

1. Run pipeline again
2. Training scripts load existing checkpoints
3. Continue from last saved epoch

---

## 📚 Related Documentation

- **Quick Start:** `QUICK_START_PINN_WEB_APP.md`
- **Dashboard Guide:** `ALL_MODELS_DASHBOARD_SUMMARY.md`
- **PINN Theory:** `results/pinn_comparison/README_theory.md` (created after Phase 3)
- **Advanced PINN:** `STACKED_PINN_README.md`
- **System Overview:** `COMPREHENSIVE_PINN_SYSTEM_SUMMARY.md`

---

## 💡 Pro Tips

### ✓ DO

- **Start small:** Test with 20-50 epochs first
- **Monitor logs:** Check `run_*.log` for issues
- **GPU training:** Ensure CUDA available for faster training
- **Clean data:** Run data fetch before training
- **Review results:** Don't skip dashboard analysis

### ✗ DON'T

- **Interrupt:** Let pipeline complete (or use Ctrl+C gracefully)
- **Concurrent runs:** Only one pipeline at a time
- **Ignore failures:** Check logs if models fail
- **Skip validation:** Always review metrics before using models
- **Trust single metric:** Consider multiple performance indicators

---

## 🎉 Pipeline Complete

After successful completion, you'll have:

✅ 13 trained neural network models
✅ Comprehensive financial metrics for all models
✅ Interactive dashboard for analysis
✅ Detailed logs and results
✅ Model checkpoints ready for inference
✅ Comparison reports for decision-making

**Next step:** Select the best model based on your risk tolerance and market regime!

---

## 📞 Troubleshooting

### Pipeline Stuck?

```bash
# Check if training is actually running
ps aux | grep python

# Check GPU utilization (if using CUDA)
nvidia-smi

# Check logs
tail -f run_*.log
```

### Need to Stop?

```bash
# Graceful stop (completes current model)
Ctrl+C

# Force stop (immediate)
Ctrl+C twice

# Kill all training processes
pkill -f "src.training"
```

### Start Fresh?

```bash
# Remove old models
rm -rf models/*

# Remove old results
rm -rf results/*

# Run pipeline again
./run.sh → Option 11
```

---

**For academic research only - not financial advice**

**Execution Date:** January 27, 2026
**Status:** Production Ready ✅
