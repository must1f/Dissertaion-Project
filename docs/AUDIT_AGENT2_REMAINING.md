## Agent 2 Remaining Gaps Audit (2026-03-07)

### Checklist
- [x] Attention LSTM registry instantiation returns `AttentionLSTM` (not plain LSTM)
- [x] Volatility track separated from core benchmark; guard blocks return targets for vol models unless explicitly overridden
- [x] Volatility bundle in notebook with its own contract/artifacts and leaderboard path
- [x] Ablation runner can execute smoke baseline+treatment and emit comparison output
- [x] Track-level leaderboards produced with track/target/fingerprint metadata
- [x] Smoke tests added for registry, core/vol training, mismatch guard, notebook separation, ablation execution

### Files Changed
- `src/models/model_registry.py` – AttentionLSTM instantiation fixed
- `scripts/train_models.py` – Track-aware guards, smoke mode, vol/price target handling, regime diagnostics
- `scripts/run_ablations.py` – Smoke execution path and comparison output
- `Jupyter/Colab_All_Models.ipynb` – Core vs volatility bundle prep and separate contracts
- `tests/test_agent2_tracks.py` – Smoke tests for registry/track separation/ablations

### Smoke Tests Run
- `python -m pytest tests/test_agent2_tracks.py -q`

### Outstanding Issues
- Type-checker warnings in `src/data/pipeline.py` and `src/models/model_registry.py` pre-existed; not addressed here.
