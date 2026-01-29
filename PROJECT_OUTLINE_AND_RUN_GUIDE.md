# Project Outline and Run Guide (Consolidated)

## ⚠️ Academic Research Disclaimer
This project is **academic research only** and **not financial advice**. It is a dissertation system for studying physics-informed learning in finance. Do **not** use outputs for real trading.

---

## 1) Project Overview
This repository implements a **Physics-Informed Neural Network (PINN)** system for financial forecasting. It embeds quantitative finance equations (GBM, Ornstein–Uhlenbeck, Langevin, Black–Scholes) into neural network loss functions and compares these against baseline ML models.

**Core goals:**
- Blend data-driven learning with finance equations (PINN).
- Provide an end-to-end pipeline: data → preprocessing → training → evaluation → backtesting → visualization.
- Deliver rigorous, reproducible evaluation with realistic assumptions.

---

## 2) Architecture (High-Level)
**Pipeline:**
Data Fetching → Preprocessing → Feature Engineering → Model Training → Evaluation → Backtesting → Dashboard

**Key modules:**
- **Data:** `src/data/` (fetcher, preprocessor, datasets)
- **Models:** `src/models/` (baseline, transformer, pinn, stacked PINN)
- **Training:** `src/training/` (trainer, training scripts)
- **Evaluation:** `src/evaluation/` (metrics, backtesting, rolling metrics, Monte Carlo)
- **Trading:** `src/trading/` (agent)
- **Web:** `src/web/` (Streamlit dashboards)
- **Utils:** `src/utils/` (config, database, logging, reproducibility)

---

## 3) Data & Features
- **Sources:** yfinance primary, Alpha Vantage backup
- **Storage:** TimescaleDB (optional) + local parquet fallback
- **Features:** log returns, simple returns, rolling volatility, momentum, RSI, MACD, Bollinger Bands, ATR, OBV, etc.
- **Splits:** temporal train/val/test (no leakage)

---

## 4) Models Included
### Baselines
- LSTM, GRU, BiLSTM, Attention LSTM, Transformer

### PINN Variants (Physics Constraints)
- Baseline (no physics)
- Pure GBM
- Pure OU
- Pure Black–Scholes
- GBM+OU Hybrid
- Global Constraint (all physics)

### Advanced PINN
- StackedPINN
- ResidualPINN

---

## 5) Evaluation & Metrics
**Traditional ML:** RMSE, MAE, MAPE, R², directional accuracy

**Financial Metrics:** Sharpe, Sortino, drawdown, Calmar, profit factor, win rate, annualized returns, IC

**Robustness:** rolling window stability, CV, regime sensitivity

**Rigorous evaluation (dissertation-grade):**
- Protected test set
- Realistic transaction costs (0.3%)
- Proper price→return conversion

---

## 6) Quick Start (Recommended)
### Setup
1. **Install dependencies:**
   - `./setup.sh`
2. **Activate environment:**
   - `source venv/bin/activate`

### One-Command Menu
- `./run.sh`

This interactive menu runs everything from data fetch → training → evaluation → dashboard.

---

## 7) Running the System (Common Workflows)

### A) Quick Demo (Full Pipeline)
- Use `./run.sh` → Option **1**
  - Starts DB (if available)
  - Fetches data
  - Trains a model
  - Launches dashboard

### B) Full Pipeline (All Models + Evaluation + Dashboard)
- `./run.sh` → Option **11**
  - Trains **13 models**
  - Computes metrics
  - Launches Streamlit dashboard

### C) Train Baseline Models Only
- `./run.sh` → Option **12**

### D) Train PINN Variants Only
- `./run.sh` → Option **10**

### E) Train a Specific Model
- `python -m src.training.train --model lstm --epochs 100`
- `python -m src.training.train --model pinn --epochs 100`

---

## 8) Evaluation (Without Retraining)
Run evaluation to compute comprehensive financial metrics for existing checkpoints:
- `python evaluate_existing_models.py --models all`

For dissertation-grade evaluation:
- `python evaluate_dissertation_rigorous.py`

---

## 9) Dashboards & Visualization
### Main Dashboard
- `streamlit run src/web/app.py`

### PINN Comparison Dashboard
- `streamlit run src/web/pinn_dashboard.py`

### Prediction Visualizations
- Integrated in main app under “Prediction Visualizations”

### Monte Carlo Dashboard
- `streamlit run src/web/monte_carlo_dashboard.py --server.port 8503`

---

## 10) Monte Carlo Simulation (Uncertainty)
- CLI: `python visualize_monte_carlo.py --synthetic --horizon 30 --n-simulations 1000`
- With model: `python visualize_monte_carlo.py --model-path models/pinn_global_best.pt --ticker AAPL`

Provides: confidence intervals, VaR/CVaR, stress tests, bootstrap CIs.

---

## 11) Stacked PINN (Advanced)
### Quick Verification
- `python3 verify_stacked_pinn.py`

### Train
- `python3 src/training/train_stacked_pinn.py --model-type stacked --epochs 100`
- `python3 src/training/train_stacked_pinn.py --model-type residual --epochs 100`

### Example
- `python3 examples/stacked_pinn_example.py`

---

## 12) Database (Optional but Recommended)
- Start DB: `docker-compose up -d timescaledb`
- Init schema: `python3 init_db_schema.py`

If Docker is unavailable, the pipeline falls back to parquet files.

---

## 13) Outputs
- **Models:** `models/` and `models/stacked_pinn/`
- **Results:** `results/*_results.json`
- **Logs:** `logs/` and `run_*.log`

---

## 14) Troubleshooting & Debug
- Enable verbose: `DEBUG=1 ./run.sh`
- Logs: `run_*.log`, `setup_*.log`

---

## 15) Source Documents Consolidated (Project & Run)
This guide compiles and normalizes content from:
- README.md
- QUICKSTART.md
- FULL_PIPELINE_GUIDE.md
- DATABASE_SETUP.md
- PINN_WEB_APP_GUIDE.md
- QUICK_START_PINN_WEB_APP.md
- EVALUATION_GUIDE.md
- FINANCIAL_METRICS_GUIDE.md
- PREDICTION_VISUALIZATION_GUIDE.md
- MONTE_CARLO_GUIDE.md
- PINN_COMPARISON_GUIDE.md
- SETUP_STACKED_PINN.md
- STACKED_PINN_README.md
- STACKED_PINN_IMPLEMENTATION_SUMMARY.md
- COMPREHENSIVE_PINN_SYSTEM_SUMMARY.md
- ACTION_GUIDE_RIGOROUS_EVALUATION.md
- DEBUGGING_GUIDE.md
- ALL_MODELS_DASHBOARD_SUMMARY.md
- AGENTS.md

---

## 16) Quick “Run Everything” Checklist
- [ ] `./setup.sh`
- [ ] `./run.sh` → Option **11**
- [ ] Wait for training
- [ ] Dashboard opens at `http://localhost:8501`

---

**Reminder:** This project is strictly academic research, not financial advice.
