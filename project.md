# PINN Financial Forecasting — Project Status (Research Perspective)

## Current Capabilities (What Works Today)
- **Physics-informed models**: PINNs with GBM, OU, Black–Scholes, and Langevin losses plus baselines (LSTM, GRU, Transformer, Stacked-PINN).
- **Research-mode training**: fixed epochs/batch/lr, reproducible splits, early stopping off for fair comparisons.
- **Evaluation pipeline**: transaction-cost-aware strategy returns; full financial metrics (Sharpe/Sortino/Calmar, drawdown, win rate, profit factor); advanced stats (bootstrapped Sharpe CI, stability, deflated Sharpe); safety guards (clipping/validation).
- **Rolling stability**: per-window metrics serialized for downstream statistical testing.
- **Comparison tooling**: `compare_pinn_baseline.py` recomputes real metrics from prediction NPZs; paired bootstrap tests; multiple-comparison correction; overfitting checks; sector analysis; cost sweeps; regime splits; calibration diagnostics; CSV/LaTeX exports.
- **Dashboards**: Streamlit/React views for ML + financial metrics with unit-consistent directional accuracy and deflated Sharpe.
- **Extensions**: cross-asset evaluation script (`cross_asset_eval.py`); physics-loss ablation tables (`physics_ablation.py`); macro feature merge pipeline (`data/merge_macro_features.py`).
- **Empirical validation**: OU vs GBM fit analysis with figures/tables; alternative equations documented.
- **Docs**: Dissertation LaTeX skeleton (`dissertation/main.tex` + chapter stubs + refs); `DOCUMENT.md` and `project_status.yaml` capture key mechanics; `TO-DO.md` tracks remaining work.
- **Testing**: domain tests (data pipeline, financial metrics, backtester, walk-forward, trading agent, stacked PINN) and smoke import test; CI workflow present.

## Gaps / Needs (What’s Missing)
- **Dissertation content**: all chapters are placeholders; need full text, figures, and references.
- **Data provisioning**: macro CSVs and cross-asset prediction NPZs are required; richer per-ticker sector result files would strengthen sector analysis.
- **Dependencies**: optional libs (loguru, sklearn, matplotlib, streamlit, fastapi, torch, seaborn, etc.) must be installed/locked; pydantic v2 migration remains.
- **Testing depth**: integration/contract tests for APIs and dashboards; end-to-end training/eval tests; enforce dependency installs in CI.
- **Macro integration**: route `prices_with_macro.csv` into feature pipelines and experiments.
- **Retraining ablations**: physics-loss sweeps are inference-only; need retrained runs with varied λ settings.
- **Significance tooling**: add Diebold–Mariano (or similar) time-series tests and automate multi-model cost/regime plots.
- **Cross-asset robustness**: extend walk-forward/protected evaluations to FX/commodities once data arrives.
- **Documentation polish**: README/PROJECT_STATUS should include run commands, data requirements, and a reproducibility checklist.

## Project Goals (Research Outcomes Targeted)
- Demonstrate whether physics priors improve forecasts under realistic trading frictions.
- Show robustness across transaction costs and market regimes with calibrated signals and deflated Sharpe reporting.
- Deliver fully reproducible artifacts: code, configs, data paths, tables/figures, and statistical tests.
- Benchmark PINN variants vs strong baselines across equities and out-of-domain assets, with sector and cross-asset insights.
- Provide maintainable tooling (dashboards, scripts) that others can rerun to validate claims.
