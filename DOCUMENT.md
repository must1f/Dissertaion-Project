## 1. Multi-asset pipeline hardening (2026-03-07)

### Overview
Hardened the multi-asset, leakage-safe data pipeline with stronger QA, calendar alignment reporting, cache/metadata structure, and train-only scaler persistence to support the dissertation’s reproducible benchmark workflow.

### Changes Made
1. **File**: `src/data/quality.py`
   - Added calendar-aware missing-date checks, corporate-action flags, post-alignment missingness, and richer QA reporting for multi-asset raw/aligned data.
2. **File**: `src/data/calendar.py`
   - Added alignment report (added rows, remaining NaNs) with bounded fills and optional reporting return; simplified fill to avoid unsafe forward-fills.
3. **File**: `src/data/cache.py`
   - Ensured saved-at metadata is persisted alongside cached Parquet and QA artifacts.
4. **File**: `src/data/fetcher.py`
   - Pointed cache root to `data/raw_cache`, defaulted dataset_tag to `raw_cache`, and stored universe/date/source metadata when caching raw downloads.
5. **File**: `src/data/pipeline.py`
   - Wired QA to master calendar, captured alignment reports and split boundaries, saved scaler metadata (train span, fitted_at), and used the new cache tag.
6. **File**: `src/data/scaling.py`
   - Serialized scaler parameters as JSON-safe lists with optional metadata payload.
7. **File**: `src/utils/config.py`
   - Created required local folders (`raw_cache`, `processed`, `splits`, `artifacts`, and cache under `data_dir`) for filesystem caching.

### Verification
- Smoke test (synthetic) for QA + calendar alignment + scaler roundtrip:
  ```bash
  python - <<'PY'
  from pathlib import Path
  import pandas as pd, numpy as np
  from src.data.quality import run_qa
  from src.data.calendar import build_master_calendar, align_to_calendar
  from src.data.scaling import fit_scaler, save_scaler, load_scaler
  raw = pd.DataFrame({
      'time': pd.date_range('2024-01-02', periods=5, freq='B').tolist() * 2,
      'ticker': ['SPY'] * 5 + ['QQQ'] * 5,
      'open': np.linspace(100, 104, 5).tolist() + np.linspace(200, 204, 5).tolist(),
      'high': np.linspace(101, 105, 5).tolist() + np.linspace(201, 205, 5).tolist(),
      'low':  np.linspace(99, 103, 5).tolist() + np.linspace(199, 203, 5).tolist(),
      'close': np.linspace(100.5, 104.5, 5).tolist() + np.linspace(200.5, 204.5, 5).tolist(),
      'adjusted_close': np.linspace(100.5, 104.5, 5).tolist() + np.linspace(200.5, 204.5, 5).tolist(),
      'volume': [1_000_000]*10,
  })
  cal = build_master_calendar('2024-01-01','2024-01-10')
  aligned, rep = align_to_calendar(raw, cal, return_report=True)
  qa = run_qa(aligned, expected_calendar=cal)
  aligned['feature_a'] = np.arange(len(aligned))*0.01
  scaler = fit_scaler(aligned, ['feature_a'])
  save_scaler(scaler, Path('/tmp/test_scaler.json'), ['feature_a'], metadata={'train_start':'2024-01-02','train_end':'2024-01-08'})
  load_scaler(Path('/tmp/test_scaler.json'))
  print('rows', qa['coverage']['rows'], 'added_rows', rep['per_ticker']['SPY']['added_rows'])
  PY
  ```

### Files Modified
| File | Changes |
|------|---------|
| src/data/quality.py | Expanded QA checks (calendar-aware, corp-action flags, post-alignment missingness) |
| src/data/calendar.py | Alignment reporting, bounded fill, report option |
| src/data/cache.py | Cache metadata now records saved_at |
| src/data/fetcher.py | Raw cache path/metadata defaults for multi-asset universe |
| src/data/pipeline.py | QA tied to master calendar, alignment report, split boundaries, scaler metadata |
| src/data/scaling.py | JSON-safe scaler serialization with metadata |
| src/utils/config.py | Ensured local cache/artifact directories exist |

## 2. Pipeline Audit Blueprint (2026-03-07)

### Overview
Captured a comprehensive audit of the required migration from a single-series S&P 500 workflow to a multi-asset, adjusted-price, leakage-safe, reproducible market-state forecasting pipeline. The audit consolidates ingestion, QA, calendar alignment, feature/target construction, leakage-safe preprocessing, fairness rules, model/task alignment, caching, experiment tiers, and risk register.

### Changes Made
1. **File**: `docs/PIPELINE_AUDIT.md`
   - Added the full audit document detailing required architectural, data, training, evaluation, and model-alignment changes, including fairness rules and experiment structuring.

### Verification
- Documentation-only addition; no runtime verification required.

### Files Modified
| File | Changes |
|------|---------|
| docs/PIPELINE_AUDIT.md | New audit document for the pipeline redesign |

## 3. Fair Benchmark Tracks & Notebook Refresh (2026-03-07)

### Overview
Standardized the Agent 1 benchmark contract across training/evaluation, added target-aware metrics and regime diagnostics, split leaderboards by track (core, volatility, advanced PINNs, OU/Black-Scholes), refreshed the Colab orchestrator, and introduced structured ablation definitions plus summary outputs.

### Changes Made
1. **File**: `src/data/pipeline.py`
   - Added explicit fairness contract payload (target/dates/lookback/required features/missing + scaling policy/price column) persisted alongside dataset artifacts.
   - Parameterized price column, missing-data policy, scaling policy, and optional QA gating; scaler fitting now honors train-only vs full-dataset ablations while recording policy metadata.
2. **File**: `src/utils/config.py`
   - Exposed `price_column`, `missing_policy`, `scaling_policy`, and `disable_qa` in `DataConfig` to drive the benchmark contract and ablation toggles.
3. **File**: `scripts/train_models.py`
   - Hardened fairness enforcement to core benchmark models, expanded track definitions (core, volatility, advanced PINNs, OU, Black-Scholes), added regime-aware/diagnostic metrics with distribution checks, and prevented scaled/unscaled metric mixups for price targets.
   - Emitted per-track leaderboards and ablation summary CSVs, attached leaderboard paths to batch results, and reused regime context for diagnostics.
4. **File**: `configs/ablations.yaml`
   - Added structured ablations for single vs multi-asset, raw vs adjusted close, VIX/rates removal, leakage-safe vs improper preprocessing, and QA filtering vs none, each with descriptions for summary export.
5. **File**: `Jupyter/Colab_All_Models.ipynb`
   - Rebuilt the notebook around the shared benchmark bundle (prepare_data/build_benchmark_dataset), core benchmark loop, optional extensions, run metadata, fairness_contract.json persistence, per-track leaderboards, and ablation summary emission; removed legacy single-series workflow.

### Verification
- Syntax check for updated benchmark pipeline and training script:
  ```bash
  python -m py_compile src/data/pipeline.py scripts/train_models.py
  ```

### Files Modified
| File | Changes |
|------|---------|
| src/data/pipeline.py | Fairness contract emission; price/missing/scaling policy controls; optional QA skip |
| src/utils/config.py | New data config flags for price column, QA toggle, missing/scaling policies |
| scripts/train_models.py | Track-aware leaderboards, regime diagnostics, scaled/unscaled guardrails, ablation summary export |
| configs/ablations.yaml | Added benchmark ablation definitions with descriptions |
| Jupyter/Colab_All_Models.ipynb | Notebook rebuilt for benchmark bundle, track separation, fairness artifacts, and ablation summary |

## 4. Agent 2 Track/Ablation Hardening (2026-03-07)

### Overview
Hardened model registry, track guards, volatility separation, and ablation execution. Added smoke tests, audit artefacts, and notebook updates to keep core vs volatility vs extensions distinct and reproducible.

### Changes Made
1. **File**: `src/models/model_registry.py`
   - Instantiates `AttentionLSTM` for `attention_lstm` key (regression fix).
2. **File**: `scripts/train_models.py`
   - Track detection/guards for volatility vs core targets; smoke-test mode; qlike/regime diagnostics for vol; results now carry track, contract fingerprint, assumption context.
3. **File**: `scripts/run_ablations.py`
   - Smoke execution of baseline+treatment with comparison CSV output; helper for tests.
4. **File**: `Jupyter/Colab_All_Models.ipynb`
   - Builds/persists separate volatility bundle/contract and uses it for volatility models; preserves core contract.
5. **File**: `tests/test_agent2_tracks.py`
   - Smoke coverage for registry attention, core/vol training, mismatch guard, notebook separation, and ablation execution.
6. **Files**: `docs/AUDIT_AGENT2_REMAINING.md`, `results/audit/agent2_smoke_test_summary.json`, `results/audit/agent2_track_validation.csv`
   - Audit checklist, smoke summary, and per-track validation outputs.

### Verification
- `python -m pytest tests/test_agent2_tracks.py -q`

### Files Modified
| File | Changes |
|------|---------|
| src/models/model_registry.py | Attention LSTM instantiation fixed |
| scripts/train_models.py | Track guardrails, smoke mode, vol metrics, contract metadata |
| scripts/run_ablations.py | Smoke execution and comparison output |
| Jupyter/Colab_All_Models.ipynb | Added volatility bundle/contract and track separation |
| tests/test_agent2_tracks.py | Smoke tests for track separation and ablations |
| docs/AUDIT_AGENT2_REMAINING.md | Audit checklist and notes |
| results/audit/agent2_smoke_test_summary.json | Smoke test summary |
| results/audit/agent2_track_validation.csv | Per-track smoke validation rows |

## 3. Benchmark Dataset Contract & Cache (2026-03-07)

### Overview
Refactored the benchmark data pipeline into a leakage-safe, universe-aware dataset builder with deterministic caching, bounded calendar alignment, reproducibility artifacts (metadata + fairness contract), and regression tests that cover cache hit/miss, adjusted-close consistency, leakage-safe scaling/splitting, sequence shapes, and QA flagging.

### Changes Made
1. **File**: `src/data/universe.py`
   - Locked a stable base universe (SPY, QQQ, IWM, XLK, XLF, XLE, ^VIX, ^TNX with optional GC=F/CL=F) and deterministic cache keys with preserved order.
2. **File**: `src/utils/config.py`, `src/config/experiment_config.py`
   - Surfaced base/optional universes, optional-asset toggle, and updated default feature list to the new macro/commodity signals.
3. **File**: `src/data/calendar.py`
   - Added per-column bounded forward-fill reporting to avoid blanket fills while aligning to the master US trading calendar.
4. **File**: `src/data/fetcher.py`, `src/data/cache.py`
   - Included deterministic cache keys and universe/date metadata in Parquet+JSON cache saves; cache timestamps are now timezone-aware.
5. **File**: `src/data/pipeline.py`
   - Rebuilt the benchmark builder: context features now use adjusted prices for ^VIX/^TNX/commodities, processed cache snapshots (raw + processed) are saved with QA, fairness_contract.json added with splits/feature list/scaler policy/fingerprint, and scaler/metadata timestamps made UTC-aware.
6. **File**: `tests/test_benchmark_dataset_contract.py`
   - New test suite covering cache roundtrip hit/miss, adjusted-close return consistency, leakage-safe split/scaler audit, sequence shape checks, and QA flagging behavior on bad data.

### Verification
- `pytest tests/test_benchmark_dataset_contract.py`

### Files Modified
| File | Changes |
|------|---------|
| src/data/universe.py | Stable universe defaults and cache-key helper |
| src/utils/config.py | Base/optional universe config + feature defaults |
| src/config/experiment_config.py | Mirrored universe/feature defaults for experiments |
| src/data/calendar.py | Asset-specific bounded forward-fill reporting |
| src/data/fetcher.py | Cached metadata now carries deterministic cache keys |
| src/data/cache.py | Timezone-aware cache timestamps |
| src/data/pipeline.py | Processed cache save, fairness contract, adjusted macro features, UTC timestamps |
| tests/test_benchmark_dataset_contract.py | Added regression tests for caching, QA, splits, and sequences |

## 4. Volatility & Cache TTL Completion (2026-03-07)

### Overview
Completed target branching (return, realized vol, joint), volatility-ready artifacts, TTL-aware caching, per-symbol calendar fill policy, normalized ticker configs, and smoke tests to guarantee leakage-safe reproducibility across tasks.

### Changes Made
1. **Files**: `src/data/pipeline.py`, `src/data/targets.py`
   - Added branching for `next_day_log_return`, `realized_vol`, and `joint_return_vol`, updated fairness contract/metadata, and task-suffixed parquet artifacts.
2. **Files**: `src/data/calendar.py`, `src/utils/config.py`, `src/config/experiment_config.py`
   - Introduced per-symbol forward-fill limits with defaults (`^VIX`, `^TNX` no fill), configurability, and symbol normalization.
3. **Files**: `src/data/cache.py`, `src/data/fetcher.py`
   - Implemented TTL-aware cache loading with cache keys and optional actions snapshot slot; respects force_refresh.
4. **Files**: `configs/ablations.yaml`, `src/data/universe.py`
   - Canonicalized universe symbols to caret forms and optional commodities.
5. **File**: `tests/test_dataset_targets_and_cache.py`
   - Smoke tests for return/vol targets, cache TTL + force_refresh, calendar fill policy, and symbol normalization.
6. **File**: `docs/AUDIT_AGENT1_REMAINING.md`
   - Audit checklist covering the remaining data-layer gaps and verification.

### Verification
- `pytest tests/test_dataset_targets_and_cache.py`

### Files Modified
| File | Changes |
|------|---------|
| src/data/pipeline.py | Target branching, fairness contract fields, task-specific artifacts |
| src/data/targets.py | Realized vol + joint target utilities |
| src/data/calendar.py | Per-symbol fill limits and reporting |
| src/data/cache.py | TTL-aware load_with_meta, actions slot, cache keys |
| src/data/fetcher.py | TTL-respecting cache usage |
| src/utils/config.py | Target vol window, cache TTL, fill policy, symbol normalization |
| src/config/experiment_config.py | Mirror of new data settings and normalization |
| src/data/universe.py | Normalized ticker handling |
| configs/ablations.yaml | Canonical caret tickers |
| tests/test_dataset_targets_and_cache.py | Smoke tests for targets/cache/calendar/symbols |
| docs/AUDIT_AGENT1_REMAINING.md | Audit checklist and verification notes |
