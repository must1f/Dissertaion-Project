# Prediction Visualization Guide

**Purpose:** Understand how PINN models predict financial returns and compare their predictive power

**Status:** ✓ Complete
**Last Updated:** January 28, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Why Predictions Matter](#why-predictions-matter)
3. [Visualization Types](#visualization-types)
4. [How to Use](#how-to-use)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### Problem Statement

All PINN models show identical Sharpe ratios (~26), making it impossible to differentiate their performance using traditional financial metrics. However, the models **do** make different predictions, as shown by:

- Varying directional accuracy (99.90%-99.94%)
- Different information coefficients (~0.92)
- Different prediction magnitudes (RMSE varies)

### Solution

The **Prediction Visualization Dashboard** shows these differences through four complementary visualizations:

1. **Time Series Comparison** - Predictions vs actual returns over time
2. **Scatter Plot Analysis** - Correlation and magnitude accuracy
3. **Distribution Analysis** - Statistical properties of predictions
4. **Residual Analysis** - Systematic prediction errors

---

## Why Predictions Matter

### The Identical Sharpe Ratio Problem

```
All Models → Predict positive returns (97%+)
         ↓
All Models → Execute 100% long strategy
         ↓
All Models → Have identical positions
         ↓
All Models → Generate identical returns
         ↓
All Models → Have Sharpe ratio ≈ 26.0
```

### What Actually Differs Between Models

Even though trading strategy is identical, models **predict differently**:

```
LSTM Predictions:     [0.0324, 0.0331, 0.0328, ...]
GRU Predictions:      [0.0325, 0.0332, 0.0329, ...]
PINN Predictions:     [0.0330, 0.0340, 0.0335, ...]
Actual Returns:       [0.0320, 0.0335, 0.0325, ...]

All → Long (positive)
But → Different magnitude

RMSE(LSTM→Actual): 1.022
RMSE(GRU→Actual):  1.024
RMSE(PINN→Actual): 1.020  ← Better!
```

---

## Visualization Types

### 1. Time Series Comparison

**What It Shows:**
- Predicted returns vs actual returns over time
- Directional agreement (rolling window accuracy)
- Cumulative strategy performance

**Why It Matters:**
- Visual confirmation of prediction quality
- Identification of periods where model fails
- Comparison of prediction magnitude

**Example Interpretation:**
```
If prediction line closely follows actual line:
  → Good magnitude prediction
  → Model captures short-term trends

If prediction line is above/below actual line:
  → Model over/under-estimates magnitudes
  → May miss volatile periods
```

**Key Metrics Shown:**
- Directional Accuracy (%) - How often model predicts correct direction
- Cumulative Returns - Strategy performance over time
- Win Rate - Percentage of profitable periods

---

### 2. Scatter Plot Analysis

**What It Shows:**
- Each prediction vs actual return as a point
- Correlation coefficient between predictions and actuals
- Perfect prediction line (y = x) and fitted line
- Clustering pattern of predictions

**Why It Matters:**
- Shows prediction accuracy visually
- Reveals model bias (systematic over/under-prediction)
- Displays model confidence level

**Example Interpretation:**
```
Points close to y=x line:
  → Accurate predictions
  → Model calibrated well

Points along y=0 (horizontal line):
  → Model predicts zero consistently
  → Poor magnitude estimation

Points following fitted line that's not y=x:
  → Systematic bias
  → Model over or under-estimates

Vertical spread of points:
  → Prediction uncertainty
  → Varying accuracy at different magnitudes
```

**Key Metric Shown:**
- **Correlation Coefficient** - Strength of prediction-actual relationship
  - 0.92 = Excellent (strong positive correlation)
  - 0.80 = Good
  - 0.50 = Moderate
  - 0.00 = No relationship

---

### 3. Distribution Analysis

**What It Shows:**
- Histogram of predicted returns distribution
- Histogram of actual returns distribution
- Pie chart of positive vs negative predictions
- Pie chart of positive vs negative actual returns

**Why It Matters:**
- Reveals what the model learns about market behavior
- Shows if model captures market skewness
- Identifies prediction bias (all positive, all negative, etc.)

**Example Interpretation:**
```
Prediction Distribution vs Actual Distribution:

If both centered at similar mean:
  → Model learned market bias correctly

If prediction distribution shifted:
  → Model has systematic bias
  → Over/under-predicts on average

Directional Split:
  - Predictions: 99.94% positive (this is the convergence issue!)
  - Actuals: 97.47% positive

  Why different?
  → Models learned to predict "safe" positive returns
  → In bull market, this works perfectly for trading
```

---

### 4. Residual Analysis

**What It Shows:**
- Prediction errors (residuals) over time
- Distribution of residuals (should be normal)
- Relationship between error magnitude and prediction magnitude
- Q-Q plot (residuals vs normal distribution)

**Why It Matters:**
- Detects systematic prediction errors
- Shows if model violates statistical assumptions
- Reveals heteroscedasticity (non-constant error variance)

**Example Interpretation:**
```
Residuals Over Time:
  - Mostly around zero: ✓ Good
  - Trending up/down: ✗ Systematic bias
  - Increasing spread: ✗ Heteroscedasticity

Residual Distribution:
  - Bell curve (normal): ✓ Expected
  - Bimodal (two peaks): ✗ Two distinct regimes
  - Skewed left/right: ✗ Directional bias

Error vs Magnitude:
  - Flat line: ✓ Consistent error regardless of size
  - Increasing slope: ✗ Larger errors for larger predictions
  - Varying scatter: ✗ Unpredictable errors

Q-Q Plot:
  - Points along diagonal: ✓ Normal distribution
  - S-shaped curve: ✗ Heavy tails
  - Deviation at ends: ✗ Outliers not captured
```

---

## How to Use

### Prerequisites

1. **Trained Models:** Run training pipeline
   ```bash
   ./run.sh  # Select option 1-3 to train models
   ```

2. **Generate Predictions:** Compute financial metrics
   ```bash
   python compute_all_financial_metrics.py
   ```

   This generates:
   - `results/{model_key}_results.json` - Results for each model
   - Contains predictions, targets, and all metrics

### Access Dashboard

**Option 1: Streamlit Web Interface**
```bash
streamlit run src/web/app.py
# Navigate to "Prediction Visualizations" tab
```

**Option 2: Direct Python**
```python
from src.web.prediction_visualizer import PredictionVisualizer
import numpy as np

# Create visualizer
viz = PredictionVisualizer()

# Load predictions (example)
predictions = np.array([...])  # From model
actuals = np.array([...])       # Ground truth

# Create visualizations
fig1 = viz.create_predictions_vs_actuals_plot(predictions, actuals, "My Model")
fig2 = viz.create_scatter_predictions_vs_actuals(predictions, actuals, "My Model")
fig3 = viz.create_prediction_distribution(predictions, actuals, "My Model")
fig4 = viz.create_residual_analysis(predictions, actuals, "My Model")

# Display
fig1.show()
fig2.show()
fig3.show()
fig4.show()
```

### Step-by-Step Workflow

1. **Run training** (if not already done):
   ```bash
   ./run.sh  # Follow prompts to train PINN models
   ```

2. **Generate predictions**:
   ```bash
   python compute_all_financial_metrics.py
   ```
   - This generates comprehensive metrics
   - Saves predictions to results/ directory

3. **Open dashboard**:
   ```bash
   streamlit run src/web/app.py
   ```

4. **Navigate to visualizations**:
   - Click "Prediction Visualizations" in sidebar
   - Select model from dropdown
   - Choose visualization type
   - Charts will display automatically

---

## Interpreting Results

### Comparing Models

**Which metrics should you compare?**

| Metric | Why Compare | Meaning |
|--------|-------------|---------|
| **Directional Accuracy** | ✓ Shows prediction direction quality | Higher = better at predicting up/down |
| **Correlation** | ✓ Shows prediction-actual relationship | ~0.92 is excellent |
| **RMSE/MAE** | ✓ Shows magnitude accuracy | Lower = more accurate predictions |
| **Information Coefficient** | ✓ Shows signal quality | ~0.92 is excellent |
| **Sharpe Ratio** | ✗ Don't compare | All identical due to convergence |
| **Win Rate** | ✗ Don't compare | All identical due to identical positions |
| **Profit Factor** | ✗ Don't compare | All identical due to identical positions |

### Model Ranking Example

```
Model             Dir Acc  Correlation  RMSE  IC     Rank
─────────────────────────────────────────────────────────
PINN Baseline     99.92%   0.9203      1.022 0.920  1
PINN GBM          99.94%   0.9215      1.020 0.922  2
PINN OU           99.91%   0.9198      1.025 0.918  3
PINN Black-Scholes 99.90%  0.9190      1.028 0.915  4
PINN GBM+OU       99.93%   0.9210      1.021 0.921  5
PINN Global       99.92%   0.9205      1.023 0.919  6

Winner: PINN GBM (highest correlation & lowest RMSE)
```

### Market Regime Analysis

**What does distribution analysis tell us?**

```
Predictions: 99.94% positive, 0.06% negative
Actuals: 97.47% positive, 2.53% negative

Interpretation:
├─ Market is strongly bullish (97.5% up days)
├─ Models learn to predict positive (safe strategy)
├─ This creates identical trading positions
└─ Results in identical Sharpe ratios

Implication:
├─ Models differ in magnitude prediction
├─ But not in direction
└─ Directional accuracy & magnitude metrics matter most
```

### Prediction Error Patterns

**What systematic errors reveal:**

```
If model over-predicts (predictions > actuals):
├─ Model is optimistic about returns
├─ May miss downside risks
└─ Good in bull markets, bad in downturns

If model under-predicts (predictions < actuals):
├─ Model is pessimistic about returns
├─ May miss upside opportunities
└─ Conservative but leaves money on table

If error increases with prediction magnitude:
├─ Model is less confident in large predictions
├─ Good for risk management
└─ More stable performance

If error is random (Q-Q plot):
├─ Model follows statistical assumptions
├─ No systematic bias
└─ Errors are unpredictable but fair
```

---

## Troubleshooting

### No Data Appears in Dashboard

**Problem:** "Waiting for prediction data" message

**Solution:**
1. Run metrics computation:
   ```bash
   python compute_all_financial_metrics.py
   ```
2. Check results directory:
   ```bash
   ls -la results/
   ```
3. Verify JSON files exist:
   ```bash
   ls results/*_results.json
   ```

### Dashboard Crashes When Loading Visualizations

**Problem:** Application error when selecting visualization

**Solution:**
1. Check if predictions are available:
   ```python
   import json
   with open('results/pinn_baseline_results.json') as f:
       data = json.load(f)
   print(data.keys())  # Should include predictions
   ```

2. Verify prediction dimensions:
   ```python
   import numpy as np
   predictions = np.array(data['predictions'])
   print(predictions.shape)  # Should be 1D or (N, 1)
   ```

### Scatter Plot Shows No Correlation

**Problem:** Correlation near zero or points scattered randomly

**Likely Cause:** Data not being computed as actual returns

**Check:**
1. Are targets actual returns or prices?
   - Should be returns: close_t / close_t-1 - 1
   - Not prices: close_t

2. Are predictions actual returns?
   - Should be returns, not prices

3. Verify preprocessing:
   ```bash
   python src/data/preprocessor.py --check
   ```

### Residual Distribution Not Normal

**Problem:** Q-Q plot doesn't follow diagonal line

**Analysis:**
1. Heavy tails (points curve up at ends):
   - Model underestimates errors in extreme cases
   - Consider robust loss functions

2. Bimodal distribution (two peaks):
   - Two distinct market regimes
   - Consider regime-switching models

3. Skewed left/right:
   - Systematic bias in predictions
   - Model consistently over/under-predicts

---

## Advanced Usage

### Exporting Visualizations

**Save as PNG for reports:**
```python
from src.web.prediction_visualizer import PredictionVisualizer

viz = PredictionVisualizer()
fig = viz.create_predictions_vs_actuals_plot(predictions, actuals, "PINN Baseline")

# Save
fig.write_image("prediction_analysis.png", width=1200, height=800)
fig.write_html("prediction_analysis.html")
```

### Batch Analysis of All Models

```python
from src.web.prediction_visualizer import PredictionVisualizer
import json
import numpy as np
from pathlib import Path

viz = PredictionVisualizer()
results_dir = Path("results")

for result_file in results_dir.glob("*_results.json"):
    with open(result_file) as f:
        data = json.load(f)

    model_name = data.get('model_name', result_file.stem)

    # Skip if no predictions
    if 'predictions' not in data or 'targets' not in data:
        continue

    predictions = np.array(data['predictions'])
    targets = np.array(data['targets'])

    # Create all visualizations
    fig1 = viz.create_predictions_vs_actuals_plot(predictions, targets, model_name)
    fig2 = viz.create_scatter_predictions_vs_actuals(predictions, targets, model_name)
    fig3 = viz.create_prediction_distribution(predictions, targets, model_name)
    fig4 = viz.create_residual_analysis(predictions, targets, model_name)

    # Save
    prefix = f"visualizations/{result_file.stem}"
    fig1.write_html(f"{prefix}_timeseries.html")
    fig2.write_html(f"{prefix}_scatter.html")
    fig3.write_html(f"{prefix}_distribution.html")
    fig4.write_html(f"{prefix}_residuals.html")

    print(f"✓ Saved visualizations for {model_name}")
```

### Custom Comparison

```python
# Compare specific models side-by-side
models = ['pinn_baseline', 'pinn_gbm', 'pinn_ou']
metrics_data = []

for model_key in models:
    with open(f"results/{model_key}_results.json") as f:
        data = json.load(f)

    metrics = data['financial_metrics']
    metrics_data.append({
        'model': data['model_name'],
        'directional_accuracy': metrics.get('directional_accuracy'),
        'information_coefficient': metrics.get('information_coefficient'),
        'rmse': data['ml_metrics'].get('rmse'),
        'correlation': np.corrcoef(
            np.array(data['predictions']).flatten(),
            np.array(data['targets']).flatten()
        )[0, 1]
    })

import pandas as pd
df = pd.DataFrame(metrics_data)
print(df.to_string())
```

---

## Key Takeaways

1. **Sharpe ratio identical** - Not useful for model comparison
2. **Directional accuracy varies** - Better metric for comparison
3. **Predictions differ in magnitude** - RMSE/MAE show differences
4. **Visualizations reveal insights** - Use multiple chart types
5. **All models perform well** - Differences are subtle (99%+ accuracy)

---

## See Also

- `SHARPE_RATIO_INVESTIGATION.md` - Detailed analysis of identical Sharpe ratios
- `FINANCIAL_METRICS_GUIDE.md` - Complete metrics documentation
- `EVALUATION_GUIDE.md` - How to evaluate models
- `src/evaluation/financial_metrics.py` - Metric calculation code

---

**Questions?** Check the documentation or review the Investigation Report: `SHARPE_RATIO_INVESTIGATION.md`
