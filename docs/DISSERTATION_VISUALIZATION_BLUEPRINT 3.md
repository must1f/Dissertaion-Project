# Dissertation Visualization Framework Blueprint

## Physics-Informed Neural Networks for Financial Volatility Forecasting

This document provides a comprehensive guide for creating publication-quality visualizations for a dissertation on PINNs applied to financial volatility forecasting.

---

## Table of Contents

1. [Core Forecast Accuracy Visualizations](#1-core-forecast-accuracy-visualizations)
2. [Loss and Calibration Diagnostics](#2-loss-and-calibration-diagnostics)
3. [Economic Performance Graphs](#3-economic-performance-graphs)
4. [Model Stability & Sensitivity Analysis](#4-model-stability--sensitivity-analysis)
5. [Physics Compliance Visualizations](#5-physics-compliance-visualizations)
6. [Advanced Diagnostic Plots](#6-advanced-diagnostic-plots)
7. [Presentation & Structure](#7-presentation--structure)
8. [Implementation Requirements](#8-implementation-requirements)
9. [Comparison Framework](#9-comparison-framework)

---

## 1. Core Forecast Accuracy Visualizations

### 1.1 Predicted vs Realized Volatility (Time Series Overlay)

**Purpose:** Demonstrates the model's ability to track volatility dynamics over time.

**What this proves:**
- Overall forecast accuracy and systematic bias
- Ability to capture volatility regime changes (low/high vol periods)
- Response lag during volatility spikes

**Validation patterns (good model):**
- Predicted closely tracks realized, especially during regime shifts
- No systematic over/under-prediction
- Quick response to volatility changes

**Failure modes to detect:**
- **Smoothing bias:** Predictions too stable, missing volatility clustering
- **Lag bias:** Predictions shift after realized (potential look-ahead leak)
- **Level bias:** Consistent over/under-estimation

**Mathematical formulation:**
```
Realized volatility: σ_t^realized = √(Σ r_i² / n)  (realized variance)
Predicted volatility: σ_t^pred = f(X_{t-1}, θ)    (model forecast)
```

### 1.2 Multi-Horizon Forecasts (1-day, 5-day, 20-day)

**Purpose:** Shows how forecast quality degrades with prediction horizon.

**What this proves:**
- Term structure of forecast accuracy
- Whether physics constraints improve long-horizon stability
- Model's ability to capture mean reversion vs momentum

**Validation patterns:**
- Gradual, smooth degradation with horizon
- PINN should degrade slower than pure ML (physics constraints help)
- R² should remain meaningful even at 20-day horizon

**Key metrics per horizon:**
- Mincer-Zarnowitz R²
- RMSE
- Directional accuracy

### 1.3 Rolling Forecast Error Plots

**Purpose:** Tracks forecast stability across different market conditions.

**Components:**
- Rolling bias (mean error over window)
- Rolling MAE (mean absolute error)
- Rolling RMSE
- ±2σ confidence bands

**What this proves:**
- Consistency of forecast quality over time
- Detection of structural breaks or model deterioration
- Seasonal patterns in accuracy

### 1.4 Residual Diagnostics

**Components:**
1. **Distribution histogram** with Normal overlay
2. **Q-Q plot** against Normal distribution
3. **Autocorrelation function (ACF)** of residuals
4. **Residuals vs Fitted** (heteroskedasticity check)

**Interpretation:**
- **Normal distribution:** Well-specified model
- **Heavy tails:** Fat-tailed errors, underestimating extremes
- **Significant ACF:** Serial correlation, model leaves predictable structure
- **Funnel shape in residuals vs fitted:** Heteroskedasticity

---

## 2. Loss and Calibration Diagnostics

### 2.1 Why QLIKE for Volatility Forecasting

QLIKE (Quasi-Likelihood) loss is the preferred metric for variance forecasts:

**Mathematical definition:**
```
QLIKE = (1/T) Σ [σ²_realized / σ²_predicted - ln(σ²_realized / σ²_predicted) - 1]
```

**Why QLIKE is preferred:**
1. **Scale-independent:** Unlike MSE, doesn't favor models that systematically underpredict
2. **Robust to heteroskedasticity:** Gives equal weight across volatility regimes
3. **Consistent ranking:** Works even when realized variance is a noisy proxy (Patton, 2011)

**Interpretation:**
- QLIKE = 0: Perfect forecast
- QLIKE > 0: Penalty increases for both under and over-estimation
- Particularly sensitive to under-estimation (important for risk management)

### 2.2 PIT (Probability Integral Transform) Histogram

**Purpose:** Tests whether model's probabilistic forecasts are well-calibrated.

**Under correct model specification:** PIT values ~ Uniform(0,1)

**Interpretation of deviations:**
- **U-shaped:** Model is over-confident (under-dispersed)
- **Inverted U-shaped:** Model is under-confident (over-dispersed)
- **Skewed right:** Systematically under-predicting volatility
- **Skewed left:** Systematically over-predicting volatility

**Statistical test:** Kolmogorov-Smirnov test against Uniform(0,1)

### 2.3 VaR Breach Rate Analysis

**Purpose:** Tests whether Value-at-Risk estimates are accurate.

**VaR definition:**
```
VaR_α(t) = -z_α · σ̂_t
```
where z_α is the α-quantile of standard normal.

**Good calibration:** Actual breach rate ≈ (1 - confidence level)
- 95% VaR: ~5% breaches expected
- 99% VaR: ~1% breaches expected

**Kupiec test for unconditional coverage:**
```
LR_UC = -2[n·ln(α) + (T-n)·ln(1-α) - n·ln(n/T) - (T-n)·ln(1-n/T)]
```
Under H₀: LR_UC ~ χ²(1)

### 2.4 Quantile Calibration Plot

**Purpose:** Reliability diagram for regression (not just VaR levels).

For each nominal coverage level (10%, 20%, ..., 99%):
- Compute symmetric prediction interval
- Plot empirical coverage vs nominal coverage

**Perfect calibration:** Points on 45-degree diagonal.

---

## 3. Economic Performance Graphs

### 3.1 Volatility-Targeting Strategy

**Strategy definition:**
```
w_t = σ_target / σ̂_t    (position weight)
```
with leverage limits: w_t ∈ [0.25, 2.0]

**Implementation (avoiding look-ahead bias):**
1. Use t-1 predicted volatility for t positions
2. Never use realized vol for position sizing
3. Apply transaction costs

**Code:**
```python
def volatility_targeting_returns(returns, predicted_vol, target_vol=0.15):
    target_daily = target_vol / np.sqrt(252)
    lagged_vol = np.roll(predicted_vol, 1)  # Use t-1 forecast
    weights = np.clip(target_daily / lagged_vol, 0.25, 2.0)
    return weights * returns
```

### 3.2 Economic Significance Evaluation

**Thresholds for meaningful performance:**
- Sharpe Ratio > 0.5: Reasonable
- Sharpe Ratio > 1.0: Good
- Sharpe Ratio > 2.0: Excellent (verify for overfitting)

- Max Drawdown < 20%: Conservative
- Max Drawdown < 40%: Moderate
- Max Drawdown > 50%: Aggressive

**Key metrics:**
- Annualized Return
- Annualized Volatility
- Sharpe Ratio: (μ - r_f) / σ
- Sortino Ratio: (μ - r_f) / σ_downside
- Calmar Ratio: Annual Return / |Max Drawdown|
- Information Ratio: (μ_active) / σ_active

### 3.3 Drawdown Analysis

**Maximum Drawdown:**
```
DD_t = (V_t - max(V_0, ..., V_t)) / max(V_0, ..., V_t)
MaxDD = min(DD_0, ..., DD_T)
```

**Additional metrics:**
- Average drawdown duration
- Recovery time from max drawdown
- Drawdown distribution

---

## 4. Model Stability & Sensitivity Analysis

### 4.1 Error vs Horizon Analysis

**What robustness looks like:**
- Gradual, monotonic increase in error with horizon
- No sharp discontinuities
- PINN should degrade slower than pure ML

**Signs of overfitting:**
- Very low error at short horizons, explosive at longer
- Non-monotonic pattern
- Large variance in error estimates

### 4.2 Physics Weight Sensitivity

**Testing λ in:** L = L_data + λ·L_physics

**Grid search over λ ∈ {0.001, 0.01, 0.05, 0.1, 0.2, 0.5}**

**What to look for:**
- Optimal λ: Balance between data fit and physics constraint
- λ too small: Physics has no effect (converges to baseline NN)
- λ too large: Over-constrained, worse data fit

### 4.3 Signs of Under-Constrained PINN

- Learned parameters far from reasonable physical values
- No improvement over baseline neural network
- Physics loss not decreasing during training
- Residual gradients vanishing early

---

## 5. Physics Compliance Visualizations (PINN-Specific)

### 5.1 Computing SDE Residuals

**Geometric Brownian Motion (GBM):**
```
dS = μS·dt + σS·dW

Residual: |ΔS/S - μΔt| / (σ√Δt)
```

**Ornstein-Uhlenbeck (OU) for volatility:**
```
dσ = θ(μ - σ)dt + η·dW

Residual: |Δσ - θ(μ - σ)Δt| / (η√Δt)
```

**Black-Scholes PDE:**
```
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

Residual: |LHS|  (computed via automatic differentiation)
```

### 5.2 Showing Physics Regularization is Meaningful

**Evidence of meaningful physics:**
- Low residuals in regions where physics applies
- Higher residuals during extreme events (acceptable, expected)
- Stable average residual over time
- Learned parameters converge to physically reasonable values

### 5.3 Avoiding Trivial Solutions

**Checks:**
- Predictions are non-constant (variance in forecasts)
- Residuals aren't zero everywhere (physics is being computed)
- Learned parameters aren't at initialization values
- Physics loss contributes meaningfully to total loss

### 5.4 Learned Parameter Evolution

**Track during training:**
- θ (OU mean reversion speed): Should converge to positive value
- μ (mean level): Should match historical average volatility
- σ (diffusion): Should be positive, reasonable magnitude

**Plot:** Parameter values vs training epoch with convergence analysis.

---

## 6. Advanced Diagnostic Plots

### 6.1 Error Distribution Tails

**Purpose:** Analyze how model handles extreme events.

**Components:**
- Empirical vs theoretical quantiles (1%, 5%, 95%, 99%)
- Excess kurtosis (fat tails indicator)
- Asymmetry analysis (over vs under-prediction in stress)

### 6.2 Volatility Regime Heatmap

**Regime classification (based on realized vol percentiles):**
- Low: < 33rd percentile
- Medium: 33rd-67th percentile
- High: > 67th percentile

**Metrics per regime:**
- QLIKE
- Correlation
- Bias
- Sample size

### 6.3 Extreme Event Analysis

**Crisis periods to analyze:**
- 2008 GFC
- 2010 Flash Crash
- 2020 COVID
- 2022 Rate Hikes

**Metrics during crises vs normal periods:**
- Forecast error magnitude
- Direction accuracy
- Response time to vol spikes

---

## 7. Presentation & Structure

### 7.1 Suggested Order of Figures (Main Body)

1. **Predicted vs Realized Volatility** - Demonstrates core capability
2. **Model Comparison Table** - Positions research contribution
3. **Economic Performance (Equity Curve)** - Shows practical significance
4. **Physics Parameters Evolution** - Novel PINN contribution
5. **QLIKE Comparison Bar Chart** - Quantitative model ranking

### 7.2 Appendix Material

1. Residual diagnostics (all models)
2. Rolling error analysis
3. PIT histograms
4. Sensitivity analysis (physics weights)
5. Robustness tests
6. Additional horizons

### 7.3 Narrative Structure for Results Section

1. **Setup:** Data description, train/val/test split, evaluation metrics
2. **Baseline Performance:** How do standard models (GARCH, LSTM) perform?
3. **PINN Introduction:** Does adding physics constraints help?
4. **Physics Comparison:** Which physics equation works best (GBM vs OU vs BS)?
5. **Economic Significance:** Is the improvement economically meaningful?
6. **Robustness:** Is performance stable across regimes and time?
7. **Conclusion:** Summary of key findings

### 7.4 Common Academic Weaknesses to Avoid

1. **No out-of-sample testing:** Always use proper train/val/test split
2. **Look-ahead bias:** Never use future information for predictions
3. **Data snooping:** Report all models tested, not just the best
4. **Ignoring transaction costs:** Include realistic costs in economic evaluation
5. **Overfitting metrics:** Use appropriate metrics (QLIKE, not just MSE)
6. **No statistical significance:** Include confidence intervals and hypothesis tests

---

## 8. Implementation Requirements

### 8.1 Python Libraries

```python
# Core
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Optional: plotly for interactive

# Statistical
from scipy import stats
from scipy.stats import norm, chi2

# Project-specific
from src.evaluation.volatility_metrics import VolatilityMetrics
from src.evaluation.financial_metrics import FinancialMetrics
```

### 8.2 Backtesting Structure

```python
def walk_forward_backtest(data, model, train_window=252, test_window=21):
    """
    Walk-forward validation with expanding or rolling window.

    Parameters:
        data: Full dataset
        model: Model to evaluate
        train_window: Initial training period (days)
        test_window: Out-of-sample test period (days)

    Returns:
        DataFrame with out-of-sample predictions and actuals
    """
    results = []

    for t in range(train_window, len(data) - test_window, test_window):
        # Train on data[:t]
        train_data = data[:t]
        model.fit(train_data)

        # Predict on data[t:t+test_window]
        test_data = data[t:t+test_window]
        predictions = model.predict(test_data)

        results.append({
            'period_start': t,
            'predictions': predictions,
            'actuals': test_data['realized_vol'],
        })

    return pd.concat([pd.DataFrame(r) for r in results])
```

### 8.3 Train/Validation/Test Split Strategy

**Recommended approach:**
```
Total data: T observations

Training: 0 to 0.6T (60%)
Validation: 0.6T to 0.8T (20%) - for hyperparameter tuning
Test: 0.8T to T (20%) - for final evaluation

Alternative (time series):
- Expanding window: Train on all historical data
- Rolling window: Fixed training window size
```

**Important:** Never tune hyperparameters on test set!

---

## 9. Comparison Framework

### 9.1 Models to Compare

| Model | Type | Description |
|-------|------|-------------|
| GARCH(1,1) | Classical | Standard volatility benchmark |
| EWMA | Classical | Exponentially weighted moving average |
| LSTM | Deep Learning | Sequence model baseline |
| Transformer | Deep Learning | Attention-based baseline |
| PINN-Baseline | PINN | Data-only (λ=0) |
| PINN-GBM | PINN | Geometric Brownian Motion |
| PINN-OU | PINN | Ornstein-Uhlenbeck |
| PINN-Global | PINN | All physics combined |

### 9.2 Diebold-Mariano Test

**Purpose:** Test whether forecast accuracy differences are statistically significant.

**H₀:** E[d_t] = 0 where d_t = L(e₁_t) - L(e₂_t)

**Test statistic:**
```
DM = d̄ / √(Var(d)/T)
```

**Interpretation:**
- |DM| > 1.96: Significant difference at 5% level
- Positive DM: Model 2 is better
- Negative DM: Model 1 is better

### 9.3 Model Confidence Set (MCS)

**Purpose:** Identify set of models that cannot be statistically distinguished from the best.

**Hansen, Lunde, & Nason (2011):**
- Sequentially eliminates worst-performing models
- Stops when remaining models are statistically equivalent
- Reports "confidence set" of best models

### 9.4 What Constitutes Meaningful Improvement

**Statistical significance:**
- Diebold-Mariano p-value < 0.05

**Economic significance:**
- QLIKE reduction > 5%
- Sharpe ratio improvement > 0.2
- Max drawdown improvement > 5 percentage points

**Robustness:**
- Improvement consistent across regimes
- Improvement consistent across time periods
- Results stable across random seeds

---

## References

1. Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics*.

2. Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*.

3. Hansen, P.R., Lunde, A., & Nason, J.M. (2011). "The Model Confidence Set." *Econometrica*.

4. Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-Informed Neural Networks." *Journal of Computational Physics*.

5. Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.

---

## Quick Reference: Key Formulas

| Metric | Formula | Notes |
|--------|---------|-------|
| QLIKE | `Σ[σ²_r/σ²_p - ln(σ²_r/σ²_p) - 1]/T` | Preferred for volatility |
| M-Z R² | `R²` from `σ²_r = α + β·σ²_p + ε` | Forecast efficiency |
| Sharpe | `(μ - r_f) / σ` | Risk-adjusted return |
| Max DD | `min(V_t - max(V_0..V_t)) / max(V_0..V_t)` | Worst drawdown |
| VaR | `-z_α · σ̂_t` | Value at Risk |
| DM | `d̄ / √(Var(d)/T)` | Compare forecasts |

---

*Generated by PINN Financial Forecasting Dissertation Framework*
