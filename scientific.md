# Scientific Guidance for PINN Financial Forecasting

This note captures the research-grade protocol for training, evaluating, and comparing models in the PINN financial forecasting project.

## Training Protocol
- **Data splits:** 70/15/15 chronological train/val/test with fixed seed; no shuffling to avoid leakage.
- **Lookback window:** 60-step sequences predicting t+1 price/return; ensure feature/target alignment before batching.
- **Normalization:** Fit scalers on training only; apply to val/test. Persist scalers with the model artifact.
- **Physics losses:** Enable/disable GBM, OU, Black–Scholes, Langevin via λ weights; log the active set per run for reproducibility.
- **Optimization:** Adam, lr=1e-3, batch=32, epochs=100, gradient clip 1.0; early stopping disabled in research mode for fair epoch parity.
- **Devices:** Prefer CUDA/MPS when available; fall back to CPU. Record device in run metadata.

## Evaluation Metrics (definitions & intent)
- **Directional Accuracy (DA):** Fraction of correct sign of Δprice (0–1). Percentage = DA×100. Use DA, not precision/recall, for trading signal quality.
- **Sharpe Ratio:** Annualized \((\bar r - r_f)/\sigma \sqrt{252}\), clipped to [-5, 5] to suppress outliers.
- **Sortino Ratio:** Annualized downside-only version, clipped to [-10, 10]; downside set at target return 0.
- **Calmar Ratio:** Annual return / |max drawdown|, clipped to [-10, 10].
- **Max Drawdown:** Minimum of cumulative return path, floored at -100%; equity is never allowed below zero.
- **Profit Factor:** Gross profit / gross loss, capped at 10 to avoid instability when losses are tiny.
- **Information Coefficient:** Pearson correlation of Δpred vs Δactual (returns), not levels.
- **Bootstrap Sharpe CI:** Block bootstrap CI; report point, 2.5/97.5% bounds.
- **Stability:** Subsample Sharpe mean/std and %positive to assess regime robustness.

## Comparison Protocol
1. **Equal footing:** Train all candidates with identical hyperparameters, data splits, and transaction cost (default 0.3%).
2. **Primary table:** Report DA, Sharpe, Sortino, Calmar, MaxDD, Annualized Return, Win Rate, Profit Factor.
3. **Uncertainty:** Include Sharpe CI and stability stats; flag any metric hitting a clip bound.
4. **Significance:** Use paired bootstrap or Diebold–Mariano on per-period returns to test model A vs B; report p-values alongside deltas.
5. **Sensitivity:** Run cost sweep (0.1–0.5%) and pre/post‑2020 regime slice; a research-grade claim should be stable across these.
6. **Reproducibility:** Publish seed, config hash, scaler parameters, device, and commit SHA with every table/figure.

## Interpretation Notes
- DA near 0.51 with IC ≈0.9 can coexist; high IC on levels does not guarantee profitable trading—check win rate and PF.
- Sharpe >5 or PF >10 is suspicious; investigate for scaling/unit errors or leakage.
- Negative Calmar with small drawdown often means low return rather than high risk; examine annualized return jointly.

## Deliverables for Dissertation
- Results tables in `dissertation/tables/*` should include CI/stability columns.
- Figures should annotate transaction cost and regime; avoid single-point Sharpe claims.
- Text should explain why physics priors are chosen (OU fit > GBM) and how they are enforced in training.
