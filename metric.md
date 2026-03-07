# Metric Calculation Reference (Explanation + Code Snippets)

This file explains how every metric is calculated in the project and shows the exact calculation snippets.

## Scope

Primary sources:
- `src/evaluation/metrics.py`
- `src/evaluation/financial_metrics.py`
- `src/training/trainer.py`
- `src/training/train_stacked_pinn.py`
- `backend/app/services/metrics_service.py`

---

## A) Optimization Metrics (used in training/backprop)

These are optimization objectives, not just reporting metrics.

### A1) `loss` (generic trainer)

Explanation:
- For non-PINN models, `loss` is MSE between predictions and targets.
- For PINN models, `loss` comes from `model.compute_loss(...)` and includes data + physics parts.
- This `loss` is what gets differentiated.

Code snippet:
```python
# src/training/trainer.py
if is_pinn:
    loss, loss_dict = self.model.compute_loss(
        predictions, targets, metadata, enable_physics=enable_physics
    )
else:
    loss = self.criterion(predictions, targets)

loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
self.optimizer.step()
```

### A2) `train_loss`, `val_loss` (generic trainer)

Explanation:
- `train_loss` is epoch mean of batch training losses.
- `val_loss` is epoch mean of validation losses.
- `val_loss` drives scheduler, checkpointing, and early stopping.

Code snippet:
```python
# src/training/trainer.py
avg_loss = total_loss / n_batches
metrics['train_loss'] = avg_loss
...
metrics['val_loss'] = avg_loss
...
self.scheduler.step(val_loss)
if val_loss < best_val_loss:
    best_val_loss = val_loss
if self.early_stopping is not None and self.early_stopping(val_loss):
    ...
```

### A3) Stacked PINN batch losses

Metrics:
- `regression_loss`
- `classification_loss`
- `prediction_loss`
- `physics_loss`
- `total_loss`

Explanation:
- `prediction_loss = MSE + 0.1 * CrossEntropy`
- `physics_loss` is weighted GBM/OU component sum.
- `total_loss = prediction_loss + physics_loss` and is backpropagated.

Code snippet:
```python
# src/training/train_stacked_pinn.py
regression_loss = nn.functional.mse_loss(return_pred, y_batch)
direction_targets = (y_batch > 0).long().squeeze()
classification_loss = nn.functional.cross_entropy(direction_logits, direction_targets)
prediction_loss = regression_loss + 0.1 * classification_loss

physics_loss = (
    curriculum_weights['lambda_gbm'] * physics_dict.get('gbm_loss', 0.0) +
    curriculum_weights['lambda_ou'] * physics_dict.get('ou_loss', 0.0)
)
physics_loss = torch.tensor(physics_loss, device=device)

total_loss = prediction_loss + physics_loss
total_loss.backward()
optimizer.step()
```

---

## B) ML Forecast Metrics (`src/evaluation/metrics.py`)

## B1) RMSE (`rmse`)

Explanation:
- Root mean squared error.
- Sensitive to large errors.

Formula:
- `sqrt(mean((y_true - y_pred)^2))`

Code snippet:
```python
@staticmethod
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

## B2) MAE (`mae`)

Explanation:
- Mean absolute error.
- Linear penalty for misses.

Formula:
- `mean(abs(y_true - y_pred))`

Code snippet:
```python
@staticmethod
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
```

## B3) MAPE (`mape`)

Explanation:
- Mean absolute percentage error.
- Uses epsilon to avoid divide-by-zero.

Formula:
- `mean(abs((y_true - y_pred)/(y_true + eps))) * 100`

Code snippet:
```python
@staticmethod
def mape(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
```

## B4) R-squared (`r2`)

Explanation:
- Fraction of variance explained.

Formula:
- `1 - SS_res/SS_tot`

Code snippet:
```python
@staticmethod
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
```

## B5) Directional Accuracy (`directional_accuracy`) in `MetricsCalculator`

Explanation:
- If `are_returns=True`, compares signs directly.
- Else compares signs of first differences (`np.diff`) to measure direction of change.
- Ignores tiny true moves using `threshold`.
- Returns **0..1** in this function.

Code snippet:
```python
@staticmethod
def directional_accuracy(y_true, y_pred, are_returns=False, threshold=1e-8):
    if are_returns:
        true_direction = y_true
        pred_direction = y_pred
    else:
        true_direction = np.diff(y_true)
        pred_direction = np.diff(y_pred)

    significant_mask = np.abs(true_direction) > threshold
    if np.sum(significant_mask) == 0:
        return 0.5

    true_significant = np.sign(true_direction[significant_mask])
    pred_significant = np.sign(pred_direction[significant_mask])
    correct = (true_significant == pred_significant).sum()
    total = len(true_significant)
    return (correct / total) if total > 0 else 0.0
```

## B6) `calculate_metrics(...)` output metrics

Explanation:
- Computes standard ML metrics and returns prefixed keys.
- If scaler stats are provided, it de-standardizes before metric calc.
- Converts directional accuracy from 0..1 to percent in returned dict.
- Includes explicit `mse = rmse^2`.

Code snippet:
```python
def calculate_metrics(y_true, y_pred, prefix="", price_mean=None, price_std=None):
    if price_mean is not None and price_std is not None:
        y_true_eval = y_true_arr * price_std + price_mean
        y_pred_eval = y_pred_arr * price_std + price_mean
    else:
        y_true_eval = y_true_arr
        y_pred_eval = y_pred_arr

    dir_acc = calc.directional_accuracy(y_true_eval, y_pred_eval)

    metrics = {
        f"{prefix}rmse": calc.rmse(y_true_eval, y_pred_eval),
        f"{prefix}mae": calc.mae(y_true_eval, y_pred_eval),
        f"{prefix}mape": calc.mape(y_true_eval, y_pred_eval),
        f"{prefix}r2": calc.r2(y_true_eval, y_pred_eval),
        f"{prefix}directional_accuracy": dir_acc * 100,
        f"{prefix}mse": calc.rmse(y_true_eval, y_pred_eval) ** 2,
    }
```

---

## C) Simple Financial Metrics (`src/evaluation/metrics.py::MetricsCalculator`)

## C1) Sharpe Ratio (`sharpe_ratio`)

Explanation:
- Uses annualized mean and annualized std, subtracting annual risk-free.
- Clipped to `[-5, 5]`.

Code snippet:
```python
@staticmethod
def sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE, periods_per_year=TRADING_DAYS_PER_YEAR):
    returns = np.clip(returns, -0.99, 1.0)
    std_return = np.std(returns)
    if std_return < 1e-10:
        return 0.0
    mean_return = np.mean(returns) * periods_per_year
    std_return = std_return * np.sqrt(periods_per_year)
    sharpe = (mean_return - risk_free_rate) / std_return
    sharpe = np.clip(sharpe, -5.0, 5.0)
    return float(sharpe)
```

## C2) Sortino Ratio (`sortino_ratio`)

Explanation:
- Uses downside std (`returns < 0`) only.
- Clipped to `[-10, 10]`.

Code snippet:
```python
@staticmethod
def sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE, periods_per_year=TRADING_DAYS_PER_YEAR):
    returns = np.clip(returns, -0.99, 1.0)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    mean_return = np.mean(returns) * periods_per_year
    downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)
    if downside_std < 1e-10:
        return 0.0
    sortino = (mean_return - risk_free_rate) / downside_std
    sortino = np.clip(sortino, -10.0, 10.0)
    return float(sortino)
```

## C3) Maximum Drawdown (`max_drawdown`) in `metrics.py`

Explanation:
- Input is cumulative equity curve.
- Returns **positive percent** (0..100).

Code snippet:
```python
@staticmethod
def max_drawdown(cumulative_returns):
    cumulative_returns = np.maximum(cumulative_returns, 1e-10)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    drawdown = np.maximum(drawdown, -1.0)
    max_dd = min(abs(np.min(drawdown)) * 100, 100.0)
    return float(max_dd)
```

## C4) Calmar Ratio (`calmar_ratio`) in `metrics.py`

Explanation:
- `annualized_return / max_drawdown_decimal`.
- Drawdown comes from cumulative curve and is converted from percent to decimal.

Code snippet:
```python
@staticmethod
def calmar_ratio(returns, periods_per_year=252):
    returns = np.clip(returns, -0.99, 1.0)
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns[-1] - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1.0
    max_dd = MetricsCalculator.max_drawdown(cumulative_returns) / 100
    if max_dd < 0.001:
        return 0.0
    calmar = annualized_return / max_dd
    calmar = np.clip(calmar, -10.0, 10.0)
    return float(calmar)
```

## C5) Win Rate (`win_rate`) in `metrics.py`

Explanation:
- Fraction of positive periods.
- Returns **percent**.

Code snippet:
```python
@staticmethod
def win_rate(returns):
    wins = (returns > 0).sum()
    total = len(returns)
    return (wins / total) * 100
```

## C6) `calculate_financial_metrics(...)` outputs in `metrics.py`

Metrics returned:
- `{prefix}sharpe_ratio`
- `{prefix}sortino_ratio`
- `{prefix}max_drawdown` (positive percent)
- `{prefix}calmar_ratio`
- `{prefix}win_rate` (percent)
- `{prefix}total_return` (percent)
- `{prefix}mean_return` (percent)
- `{prefix}volatility` (annualized percent)

Code snippet:
```python
metrics = {
    f"{prefix}sharpe_ratio": calc.sharpe_ratio(...),
    f"{prefix}sortino_ratio": calc.sortino_ratio(...),
    f"{prefix}max_drawdown": calc.max_drawdown(cumulative_returns),
    f"{prefix}calmar_ratio": calc.calmar_ratio(...),
    f"{prefix}win_rate": calc.win_rate(returns),
    f"{prefix}total_return": total_ret,
    f"{prefix}mean_return": np.mean(returns) * 100,
    f"{prefix}volatility": np.std(returns) * np.sqrt(periods_per_year) * 100,
}
```

---

## D) Advanced Financial Metrics (`src/evaluation/financial_metrics.py`)

## D1) Sharpe (`sharpe_ratio`) and raw Sharpe (`sharpe_ratio_raw`)

Explanation:
- Uses per-period RF (`risk_free_rate / periods_per_year`) and `ddof=1` std.
- `sharpe_ratio` clipped to `[-5, 5]`.
- `sharpe_ratio_raw` returns unclipped (bounded for infinities).

Code snippet:
```python
mean_return = np.mean(returns)
std_return = np.std(returns, ddof=1)
rf_per_period = risk_free_rate / periods_per_year
sharpe = (mean_return - rf_per_period) / std_return * np.sqrt(periods_per_year)
```

## D2) Sortino (`sortino_ratio`) and raw Sortino (`sortino_ratio_raw`)

Explanation:
- Downside uses returns below `target_return` (default 0).
- Annualized with `sqrt(periods_per_year)`.
- Clipped in display variant only.

Code snippet:
```python
downside_returns = returns[returns < target_return]
downside_std = np.std(downside_returns, ddof=1)
sortino = (mean_return - rf_per_period) / downside_std * np.sqrt(periods_per_year)
```

## D3) Max Drawdown (`max_drawdown`) in advanced module

Explanation:
- Computes drawdown from cumulative equity.
- Returns **negative decimal** in [-1, 0].

Code snippet:
```python
returns_clipped = np.clip(returns, -0.99, 1.0)
cum_returns = np.cumprod(1 + returns_clipped)
running_max = np.maximum.accumulate(cum_returns)
drawdown = (cum_returns - running_max) / running_max
max_dd = np.min(np.maximum(drawdown, -1.0))
return float(max_dd)
```

## D4) Calmar Ratio (`calmar_ratio`)

Explanation:
- `annual_return / abs(max_drawdown)`.

Code snippet:
```python
total_return = np.prod(1 + returns) - 1
annual_return = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1.0
max_dd = FinancialMetrics.max_drawdown(returns)
calmar = annual_return / abs(max_dd)
```

## D5) Cumulative Return Series (`cumulative_returns`)

Explanation:
- Computes cumulative return path and floors at -100%.

Code snippet:
```python
cum_returns = np.cumprod(1 + returns) - 1
cum_returns = np.maximum(cum_returns, -1.0)
```

## D6) Total Return (`total_return`)

Explanation:
- End-to-end compounded return.
- Clipped to [-1, 10].

Code snippet:
```python
total_ret = np.prod(1 + returns) - 1
total_ret = np.clip(total_ret, -1.0, 10.0)
```

## D7) Annualized Return (`annualized_return`)

Explanation:
- Converts total compounded return to annual basis.
- Clipped to [-1, 5].

Code snippet:
```python
total_return = np.prod(1 + returns) - 1
n_years = len(returns) / periods_per_year
annual_return = (1 + total_return) ** (1 / n_years) - 1
annual_return = np.clip(annual_return, -1.0, 5.0)
```

## D8) Directional Accuracy (`FinancialMetrics.directional_accuracy`)

Explanation:
- Works on returns or on first-difference price changes.
- Returns **0..1**.

Code snippet:
```python
if are_returns:
    pred_direction = predictions
    actual_direction = targets
else:
    pred_direction = np.diff(predictions)
    actual_direction = np.diff(targets)

significant_mask = (np.abs(actual_direction) > threshold)
if np.sum(significant_mask) == 0:
    return 0.5

correct = np.sign(pred_direction[significant_mask]) == np.sign(actual_direction[significant_mask])
accuracy = np.mean(correct)
```

## D9) Information Ratio (`information_ratio`)

Explanation:
- Annualized mean active return over active return std.

Code snippet:
```python
active_returns = returns - benchmark_returns
mean_active = np.mean(active_returns)
std_active = np.std(active_returns, ddof=1)
ir = mean_active / std_active * np.sqrt(periods_per_year)
```

## D10) Drawdown Duration (`drawdown_duration`)

Explanation:
- Measures average consecutive periods in drawdown below -1%.

Code snippet:
```python
drawdown = (cum_returns - running_max) / running_max
in_drawdown = drawdown < -0.01
# iterate and average contiguous run lengths
```

## D11) Profit Factor (`profit_factor`)

Explanation:
- `gross_profit / gross_loss`, clipped to [0, 10].

Code snippet:
```python
gross_profit = np.sum(returns[returns > 0])
gross_loss = abs(np.sum(returns[returns < 0]))
pf = gross_profit / gross_loss
pf = np.clip(pf, 0.0, 10.0)
```

## D12) Information Coefficient (`information_coefficient`)

Explanation:
- Default: correlation of predicted and actual **changes** (diff series).

Code snippet:
```python
pred_returns = np.diff(predictions)
target_returns = np.diff(targets)
ic = np.corrcoef(pred_returns, target_returns)[0, 1]
```

## D13) Precision / Recall / F1 (`precision_recall`)

Explanation:
- Classification of positive vs non-positive movements.
- By default uses changes (`np.diff`) rather than levels.

Code snippet:
```python
pred_positive = np.diff(predictions) > 0
actual_positive = np.diff(targets) > 0

tp = np.sum(pred_positive & actual_positive)
fp = np.sum(pred_positive & ~actual_positive)
fn = np.sum(~pred_positive & actual_positive)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

## D14) Skewness (`skewness`)

Explanation:
- Third standardized moment of returns.

Code snippet:
```python
return float(scipy_stats.skew(returns, bias=False))
```

## D15) Kurtosis (`kurtosis`)

Explanation:
- Fourth standardized moment (excess by default).

Code snippet:
```python
return float(scipy_stats.kurtosis(returns, fisher=excess, bias=False))
```

## D16) Bootstrapped Sharpe CI (`bootstrapped_sharpe_ci`)

Explanation:
- Block bootstrap to preserve autocorrelation.
- Returns `(point_estimate, lower, upper)`.

Code snippet:
```python
block_size = max(1, int(np.ceil(len(returns) ** (1/3))))
# sample blocks with replacement
boot_sharpes.append(compute_sharpe(sample))
lower = np.percentile(boot_sharpes, alpha / 2 * 100)
upper = np.percentile(boot_sharpes, (1 - alpha / 2) * 100)
```

## D17) Deflated Sharpe Ratio (`deflated_sharpe_ratio`)

Explanation:
- Adjusts observed Sharpe for multiple testing and non-normality.
- Returns probability-like score in [0,1].

Code snippet:
```python
expected_max = (1 - euler_mascheroni) * norm.ppf(1 - 1/n_trials) + \
               euler_mascheroni * norm.ppf(1 - 1/(n_trials * np.e))
adjustment = 1 - skewness * sharpe_ratio / 3 + (kurtosis - 3) / 4 * (sharpe_ratio ** 2)
adjusted_sharpe = sharpe_ratio / np.sqrt(adjustment) if adjustment > 0 else sharpe_ratio
dsr = norm.cdf((adjusted_sharpe - expected_max) / sr_std)
```

## D18) Subsample Stability (`subsample_stability`)

Explanation:
- Splits returns into non-overlapping chunks and computes metric in each chunk.
- Stability score is `1 - clip(std/abs(mean), 0, 1)`.

Code snippet:
```python
metrics = [metric_func(subsample_i), ...]
mean_metric = np.mean(metrics)
std_metric = np.std(metrics, ddof=1)
stability = 1 - np.clip(std_metric / abs(mean_metric), 0, 1)
```

## D19) `FinancialMetrics.compute_all_metrics(...)` output keys

Explanation:
- This aggregator computes and returns:

Return metrics:
- `total_return_raw`, `annualized_return_raw`
- `total_return`, `annualized_return`, `cumulative_return_final`

Risk-adjusted:
- `sharpe_ratio_raw`, `sortino_ratio_raw`
- `sharpe_ratio`, `sortino_ratio`
- `sharpe_ratio_display`, `sortino_ratio_display`

Drawdown/risk:
- `max_drawdown`, `drawdown_duration`
- `calmar_ratio_raw`, `calmar_ratio`
- `volatility`

Trading viability:
- `profit_factor_raw`, `profit_factor`
- `win_rate`

Signal quality (if predictions/targets provided):
- `directional_accuracy`, `information_coefficient`
- `precision`, `recall`, `f1_score`

Benchmark/advanced:
- `information_ratio` (if benchmark passed)
- `skewness`, `kurtosis`
- `sharpe_ci_lower`, `sharpe_ci_upper`
- `sharpe_stability`, `sharpe_subsample_std`, `positive_subsample_pct`

NaN/Inf handling:
- `inf -> +/-10`
- `nan -> 0`

Code snippet:
```python
metrics['total_return_raw'] = total_return_raw
metrics['annualized_return_raw'] = annualized_return_raw
...
metrics['sharpe_ratio_raw'] = sharpe_raw
metrics['sortino_ratio_raw'] = sortino_raw
metrics['sharpe_ratio'] = sharpe_display
...
metrics['directional_accuracy'] = FinancialMetrics.directional_accuracy(...)
metrics['information_coefficient'] = FinancialMetrics.information_coefficient(...)
metrics.update(FinancialMetrics.precision_recall(...))
...
for key, value in metrics.items():
    if np.isinf(value): metrics[key] = 10.0 if value > 0 else -10.0
    elif np.isnan(value): metrics[key] = 0.0
```

---

## E) Strategy Return Metric Input (`compute_strategy_returns`)

Before many financial metrics, strategy returns are built from predictions/targets.

Explanation:
- Converts prices to returns (or uses returns directly).
- Builds signal (`sign`, `scaled`, or `prob`) with threshold.
- Shifts position by one period to avoid look-ahead bias.
- Applies turnover transaction costs.
- Clips strategy returns.

Code snippet:
```python
positions = np.zeros_like(raw_signal)
positions[1:] = raw_signal[:-1]  # look-ahead safe
position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
strategy_returns = positions * actual_returns - position_changes * transaction_cost
strategy_returns = np.clip(strategy_returns, min_return, max_return)
```

---

## F) Stacked PINN Script Metrics (`src/training/train_stacked_pinn.py`)

## F1) Training-direction metric (`directional_accuracy`)

Explanation:
- In training batch return dict, directional accuracy is sign agreement of predicted return vs target return.
- Returns 0..1.

Code snippet:
```python
with torch.no_grad():
    pred_direction = (return_pred > 0).float()
    actual_direction = (y_batch > 0).float()
    directional_accuracy = (pred_direction == actual_direction).float().mean().item()
```

## F2) Validation metrics in this script (`evaluate`)

Metrics:
- `mse`, `mae`, `rmse`, `directional_accuracy`

Code snippet:
```python
mse = float(np.mean((pred_np - y_np) ** 2))
mae = float(np.mean(np.abs(pred_np - y_np)))
directional_acc = FinancialMetrics.directional_accuracy(pred_np, y_np, are_returns=False)
metrics = {'mse': mse, 'mae': mae, 'rmse': float(np.sqrt(mse)), 'directional_accuracy': directional_acc}
```

## F3) Epoch controls in this script

Explanation:
- Uses `val_metrics['mse']` as validation loss proxy for:
  - best model selection
  - LR scheduler step

Code snippet:
```python
history['val_loss'].append(val_metrics['mse'])
if val_metrics['mse'] < best_val_loss:
    best_val_loss = val_metrics['mse']
...
scheduler.step(val_metrics['mse'])
```

---

## G) API Service Metrics (`backend/app/services/metrics_service.py`)

## G1) API ML metrics

Explanation:
- Uses `MetricsCalculator` from `src/evaluation/metrics.py`.
- Converts directional accuracy to percent (`*100`).

Code snippet:
```python
return MLMetrics(
    rmse=MetricsCalculator.rmse(y_true, y_pred),
    mae=MetricsCalculator.mae(y_true, y_pred),
    mape=MetricsCalculator.mape(y_true, y_pred),
    r2=MetricsCalculator.r2(y_true, y_pred),
    directional_accuracy=MetricsCalculator.directional_accuracy(y_true, y_pred) * 100,
)
```

## G2) API financial metrics

Explanation:
- Computes core financial figures for API response:
  - `total_return`, `annual_return`, `daily_return_mean`, `daily_return_std`
  - `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `information_ratio`
  - `max_drawdown`, `win_rate`, `profit_factor`
  - `avg_win`, `avg_loss`, trade counts

Code snippet:
```python
sharpe = metrics_class.sharpe_ratio(returns, risk_free_rate, periods_per_year)
sortino = metrics_class.sortino_ratio(returns, risk_free_rate, periods_per_year)
max_dd = metrics_class.max_drawdown(returns)
...
winning = np.sum(returns > 0)
losing = np.sum(returns < 0)
profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else None
```

---

## H) Statistical Comparison Utilities

These are additional calculated statistics used for evaluation/comparison.

## H1) Diebold-Mariano test (`diebold_mariano_test`)

Explanation:
- Compares forecast error series from two models.
- Returns `(dm_stat, p_value)`.

Code snippet:
```python
d = errors1 ** 2 - errors2 ** 2
d_mean = np.mean(d)
...
dm_stat = d_mean / np.sqrt(d_var)
p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
```

## H2) Bootstrap CI helper (`bootstrap_confidence_interval`)

Explanation:
- Generic bootstrap confidence interval for any scalar metric function.

Code snippet:
```python
point_estimate = metric_func(data)
bootstrap_estimates = []
for _ in range(n_bootstrap):
    sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_estimates.append(metric_func(sample))
lower_ci = np.percentile(bootstrap_estimates, alpha / 2 * 100)
upper_ci = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
```

---

## I) Unit/Scale Notes (critical)

- `directional_accuracy` appears as both:
  - `0..1` in core functions
  - `0..100` in some outputs (after explicit `*100`)
- `max_drawdown` appears as both:
  - positive percent in `src/evaluation/metrics.py`
  - negative decimal in `src/evaluation/financial_metrics.py`
- `win_rate` appears as both:
  - percent in `metrics.py`
  - decimal in advanced `compute_all_metrics`

Always normalize units before comparison/ranking.

---

## J) File references

- `src/training/trainer.py`
- `src/training/train_stacked_pinn.py`
- `src/evaluation/metrics.py`
- `src/evaluation/financial_metrics.py`
- `backend/app/services/metrics_service.py`

