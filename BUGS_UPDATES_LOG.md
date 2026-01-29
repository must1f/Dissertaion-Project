# Bugs, Fixes, Audits, and Updates (Consolidated)

## ⚠️ Academic Research Disclaimer
This project is **academic research only** and **not financial advice**. Do **not** use outputs for real trading.

---

## 1) Critical Audit & Dissertation Rigor Fixes
**Status:** Complete

### Key Fixes
- **Price→Return conversion** corrected in strategy evaluation
- **Realistic transaction costs** updated to 0.3%
- **Protected test set** for rigorous evaluation
- **Walk-forward framework** ready for extended validation

**Files:**
- CRITICAL_AUDIT_COMPLETE.md
- DISSERTATION_RIGOR_AUDIT_FIXES.md
- ACTION_GUIDE_RIGOROUS_EVALUATION.md

---

## 2) Sharpe Ratio Investigation (Not a Bug)
**Finding:** Identical Sharpe ratios are expected in a strongly bullish market because all models predict mostly positive returns and take identical positions.

**Outcome:**
- New prediction visualization system added
- Dashboards updated with warnings and guidance
- Documentation expanded with analysis + usage guidance

**Files:**
- SHARPE_RATIO_INVESTIGATION.md
- SHARPE_RATIO_SUMMARY.md
- IMPLEMENTATION_SUMMARY_SHARPE_INVESTIGATION.md
- INVESTIGATION_COMPLETE.md
- PREDICTION_VISUALIZATION_GUIDE.md
- QUICKSTART_SHARPE_INVESTIGATION.md

---

## 3) Dashboard Fixes
**Issues fixed:**
- KeyError on All Models table styling
- PINN Comparison not finding results
- Missing Sharpe plots due to absent financial metrics
- PINN registry not detecting trained variants

**File:** DASHBOARD_FIXES_SUMMARY.md

---

## 4) Baseline Checkpoint Naming Fix
**Issue:** Baseline checkpoints saved to generic path; evaluation couldn’t find them.

**Fix:** Trainer now saves to `models/{model}_best.pt`. Baselines require retraining to generate correct checkpoints.

**File:** BASELINE_CHECKPOINT_FIX.md

---

## 5) LSTM/GRU Tuple Return Fix
**Issue:** LSTM/GRU return tuples; trainer expected tensors, causing crashes.

**Fix:** Trainer now detects LSTM/GRU class names and unpacks outputs correctly.

**File:** TUPLE_RETURN_FIX.md

---

## 6) Database Fixes
**Issues fixed:**
- Raw SQL connection context manager misuse
- Missing `rolling_volatility_60` column
- Duplicate key errors on `(time, ticker)`
- Schema auto-init on DB startup

**Files:**
- DATABASE_FIXES.md
- UPSERT_FIX.md
- DATABASE_SETUP.md

---

## 7) Evaluation & Metrics Updates
**Highlights:**
- Unified evaluator with 15+ metrics
- Rolling window stability analysis
- Comprehensive financial metrics
- Updated dashboards to show ML + financial metrics

**Files:**
- FINANCIAL_METRICS_GUIDE.md
- COMPREHENSIVE_PINN_SYSTEM_SUMMARY.md
- ALL_MODELS_DASHBOARD_SUMMARY.md

---

## 8) Spec Compliance Audit (Open Gaps)
**Audit status:** Substantial but incomplete (B / 75%)

**Key gaps noted:**
- Black–Scholes PDE implementation incomplete
- Physics parameters hardcoded (not learned)
- Formal dissertation document not present
- Monte Carlo noted missing at time of audit (see MONTE_CARLO_GUIDE.md for current implementation)

**File:** UPDATE-PROJECT-SPEC.md

---

## 9) Debugging & Reliability Updates
**Enhancements:**
- Timestamped logs for setup/run
- Debug mode for scripts
- Stage-based progress reporting

**File:** DEBUGGING_GUIDE.md

---

## 10) Source Documents Consolidated (Bugs/Updates)
This log compiles and normalizes content from:
- CRITICAL_AUDIT_COMPLETE.md
- DISSERTATION_RIGOR_AUDIT_FIXES.md
- ACTION_GUIDE_RIGOROUS_EVALUATION.md
- SHARPE_RATIO_INVESTIGATION.md
- SHARPE_RATIO_SUMMARY.md
- IMPLEMENTATION_SUMMARY_SHARPE_INVESTIGATION.md
- INVESTIGATION_COMPLETE.md
- QUICKSTART_SHARPE_INVESTIGATION.md
- DASHBOARD_FIXES_SUMMARY.md
- BASELINE_CHECKPOINT_FIX.md
- TUPLE_RETURN_FIX.md
- DATABASE_FIXES.md
- UPSERT_FIX.md
- DATABASE_SETUP.md
- FINANCIAL_METRICS_GUIDE.md
- COMPREHENSIVE_PINN_SYSTEM_SUMMARY.md
- ALL_MODELS_DASHBOARD_SUMMARY.md
- UPDATE-PROJECT-SPEC.md
- DEBUGGING_GUIDE.md

---

## 11) Recommended Next Actions
- Run rigorous evaluation: `python evaluate_dissertation_rigorous.py`
- Retrain baselines to regenerate checkpoints
- Address spec gaps from audit (Black–Scholes PDE, parameter learning, dissertation document)

---

**Reminder:** This project is strictly academic research, not financial advice.
