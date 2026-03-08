## Agent1 Remaining Data-Layer Audit (2026-03-07)

### Checklist
- [x] Target branching (return, realized_vol, joint) with metadata and fairness contract
- [x] Volatility dataset artifacts and task-specific parquet naming
- [x] Calendar alignment with asset-specific fill policy
- [x] Cache TTL/freshness + force refresh + actions snapshot hook
- [x] QA summary (coverage by ticker, post-alignment missingness, jump/corp-action thresholds)
- [x] Symbol normalization across universes/ablations
- [x] Smoke tests for targets, cache TTL, force refresh, calendar policy, symbol normalization

### Artifacts and Files
- `src/data/pipeline.py` — target branching, artifact naming, fairness contract fields
- `src/data/targets.py` — realized vol and joint target utilities
- `src/data/calendar.py` — per-symbol fill limits and reports
- `src/data/cache.py` — TTL-aware load_with_meta, cache_key metadata, actions slot
- `src/data/fetcher.py` — TTL-respecting cache loads
- `src/data/universe.py`, `src/utils/config.py`, `src/config/experiment_config.py` — symbol normalization and defaults
- `configs/ablations.yaml` — canonical tickers
- `tests/test_dataset_targets_and_cache.py` — smoke coverage for targets, cache, calendar, normalization

### Verification
- `pytest tests/test_dataset_targets_and_cache.py`

### Notes
- Calendar fills remain bounded and ticker-specific to avoid leakage across non-trading days.
- Scaling remains train-only unless explicitly set to the improper ablation.
