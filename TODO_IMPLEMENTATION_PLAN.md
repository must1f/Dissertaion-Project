# Implementation Plan - PINN Financial Forecasting

**Project Status**: Phase 5 Complete - Quality & Maintenance Done
**Last Updated**: 2026-02-26
**Priority Order**: 1 (Highest) → 6 (Lowest)

---

## Quick Reference: Implementation Status

| Category | Status | Priority | Est. Complexity |
|----------|--------|----------|-----------------|
| 1. Evaluation & Benchmarking | 🟢 Complete | HIGH | High |
| 2. PINN Correctness & Stability | 🟢 Complete | HIGH | Medium |
| 3. Data Pipeline & Leakage Prevention | 🟢 Complete | HIGH | Medium |
| 4. Experiment Management | 🟢 Complete | MEDIUM | Medium |
| 5. Observability & Reporting | 🟢 Complete | MEDIUM | Low |
| 6. Code Quality & Testing | 🟢 Complete | LOW | Low |

### Completed Modules Summary

| Module | Files | Tests |
|--------|-------|-------|
| `src/losses/` | 4 files (data, physics, composite losses) | 43 tests |
| `src/config/` | 2 files (experiment config) | - |
| `src/data/` | 5 new files (cleaner, versioner, registry, auditor) | 36 tests |
| `src/evaluation/` | 12 new files (harness, stats, stress, leaderboard) | - |
| `src/reporting/` | 3 files (plots, reports) | - |
| `src/trading/` | 4 new files (strategies, benchmarks) | - |
| `tests/unit/` | 2 files (69 tests total) | ✓ All pass |

---

## 1. Evaluation and Benchmarking Features (HIGHEST IMPACT)

### 1.1 Standardised Evaluation Harness
**Goal**: Single interface to train, validate, test any model with consistent methodology.

**Implementation Approach**:
```
src/evaluation/
├── evaluation_harness.py      # Main orchestrator
├── split_manager.py           # Train/val/test splits with versioning
├── metrics_registry.py        # Centralized metrics (MAE, RMSE, Sharpe, etc.)
└── result_logger.py           # Structured logging to DB/files
```

**Tasks**:
- [ ] Create `EvaluationHarness` class with unified `evaluate(model, config)` API
- [ ] Implement `SplitManager` for reproducible data splits (same seeds)
- [ ] Build `MetricsRegistry` with all metrics (MSE, MAE, RMSE, MAPE, Sharpe, Sortino, Max DD)
- [ ] Add structured result logging (JSON + SQLite)
- [ ] Create CLI: `python -m evaluate --model lstm --config configs/eval.yaml`

**Files to Create/Modify**:
- `src/evaluation/evaluation_harness.py` (NEW)
- `src/evaluation/split_manager.py` (NEW)
- `src/evaluation/metrics_registry.py` (NEW)

---

### 1.2 Rolling and Expanding Window Backtesting
**Goal**: Walk-forward evaluation to address non-stationarity.

**Implementation Approach**:
```python
# Window strategies
class WindowStrategy(Enum):
    EXPANDING = "expanding"   # Train on all prior data
    ROLLING = "rolling"       # Fixed lookback window

# Config example
windows:
  strategy: rolling
  train_size: 252  # 1 year
  test_size: 21    # 1 month
  step_size: 21    # Monthly retraining
```

**Tasks**:
- [ ] Implement `WalkForwardValidator` with rolling/expanding modes
- [ ] Store each window's performance separately (window_id, start, end, metrics)
- [ ] Add window-level aggregation (mean, std, min, max across windows)
- [ ] Integrate with existing `walk_forward_validation.py`

**Files to Create/Modify**:
- `src/evaluation/walk_forward_validation.py` (EXTEND)
- `src/evaluation/window_results.py` (NEW)

---

### 1.3 Regime and Stress Testing
**Goal**: Evaluate by volatility regime and market stress events.

**Implementation Approach**:
```python
# Regime detection
class RegimeDetector:
    def detect_volatility_regime(self, returns) -> List[str]:
        # LOW: vol < 15%, MEDIUM: 15-25%, HIGH: > 25%

    def detect_stress_windows(self, dates) -> List[StressWindow]:
        # COVID: 2020-02 to 2020-04
        # GFC: 2008-09 to 2009-03
        # Earnings weeks, Fed announcements
```

**Tasks**:
- [ ] Extend `RegimeDetector` with stress window definitions
- [ ] Create `StressTestEvaluator` for crisis period analysis
- [ ] Add regime-stratified metrics reporting
- [ ] Build shock calendar (earnings, Fed, known crises)

**Files to Create/Modify**:
- `src/evaluation/regime_detector.py` (EXTEND)
- `src/evaluation/stress_test_evaluator.py` (NEW)
- `configs/stress_windows.yaml` (NEW)

---

### 1.4 Statistical Significance Testing
**Goal**: Quantify uncertainty and compare models rigorously.

**Implementation Approach**:
```python
class StatisticalTests:
    def bootstrap_ci(self, metric_values, n_bootstrap=1000, ci=0.95):
        """Bootstrap confidence intervals for any metric."""

    def paired_t_test(self, model_a_metrics, model_b_metrics):
        """Test if model A significantly better than B."""

    def diebold_mariano_test(self, errors_a, errors_b):
        """Forecast comparison test."""
```

**Tasks**:
- [ ] Implement bootstrap confidence intervals for Sharpe, MAE, etc.
- [ ] Add Diebold-Mariano test for forecast comparison
- [ ] Create multi-seed runner (5-10 seeds per model)
- [ ] Build significance summary tables

**Files to Create/Modify**:
- `src/evaluation/statistical_tests.py` (NEW)
- `src/evaluation/multi_seed_runner.py` (NEW)

---

### 1.5 Trading Strategy Evaluation Layer
**Goal**: Convert forecasts to tradeable signals and compare against benchmarks.

**Implementation Approach**:
```python
class StrategyEvaluator:
    strategies = [
        ThresholdStrategy(buy_threshold=0.01, sell_threshold=-0.01),
        RankingStrategy(top_n=3),
        VolatilityScaledStrategy(target_vol=0.15),
    ]

    benchmarks = [
        BuyAndHold(),
        SPY(),
        NaiveLastValue(),
    ]
```

**Tasks**:
- [ ] Create `StrategyConverter` to transform forecasts → positions
- [ ] Implement threshold, ranking, and vol-scaled strategies
- [ ] Add benchmark strategies (buy-hold, SPY, naive)
- [ ] Calculate trading metrics (returns, Sharpe, turnover, costs)

**Files to Create/Modify**:
- `src/trading/strategy_evaluator.py` (NEW)
- `src/trading/benchmark_strategies.py` (NEW)

---

### 1.6 Benchmark Tracking and Leaderboards
**Goal**: Auto-generate comparison tables and persist results.

**Implementation Approach**:
```
Database Schema:
├── experiments (id, config_hash, timestamp)
├── model_runs (id, experiment_id, model_type, seed)
├── window_results (id, run_id, window_id, regime, metrics_json)
└── leaderboard (id, metric_name, model_type, value, rank)
```

**Tasks**:
- [ ] Design SQLite schema for result persistence
- [ ] Create `LeaderboardGenerator` for auto-tables
- [ ] Build comparison views (by model, window, regime)
- [ ] Add export to LaTeX/Markdown tables

**Files to Create/Modify**:
- `src/evaluation/results_db.py` (NEW)
- `src/evaluation/leaderboard.py` (NEW)
- `scripts/generate_leaderboard.py` (NEW)

---

## 2. PINN-Specific Correctness and Stability

### 2.1 Stiffness and Loss-Balancing Diagnostics
**Goal**: Separate PDE stiffness from optimization imbalance; add diagnostics.

**Implementation Approach**:
```python
class LossDiagnostics:
    def compute_gradient_norms(self, loss_terms: Dict[str, Tensor]) -> Dict[str, float]:
        """Gradient norm per loss term."""

    def compute_residual_magnitudes(self, model, batch) -> Dict[str, float]:
        """Magnitude of each PDE residual."""

    def detect_imbalance(self, grad_norms) -> bool:
        """Flag if any term dominates (ratio > 100x)."""
```

**Tasks**:
- [ ] Add gradient norm tracking per loss term (data, GBM, OU, BS)
- [ ] Log residual magnitudes at each epoch
- [ ] Create imbalance detection and warnings
- [ ] Add TensorBoard/WandB logging for loss components

**Files to Create/Modify**:
- `src/training/loss_diagnostics.py` (NEW)
- `src/training/trainer.py` (EXTEND)

---

### 2.2 Adaptive Loss Weighting
**Goal**: Implement automatic loss balancing (GradNorm, uncertainty, or residual-based).

**Implementation Approach**:
```python
class AdaptiveLossWeighter:
    """GradNorm-style adaptive weighting."""

    def __init__(self, num_tasks: int, alpha: float = 1.5):
        self.weights = nn.Parameter(torch.ones(num_tasks))

    def update_weights(self, losses: List[Tensor], shared_params):
        # Compute gradient norms and rebalance
```

**Tasks**:
- [ ] Implement GradNorm adaptive weighting
- [ ] Add uncertainty-based weighting (Kendall & Gal)
- [ ] Log weight evolution over training
- [ ] Add config switch: `adaptive_weighting: gradnorm|uncertainty|none`

**Files to Create/Modify**:
- `src/training/adaptive_loss.py` (NEW)
- `src/models/pinn.py` (EXTEND)

---

### 2.3 Curriculum Learning Scheduler
**Goal**: Formalize schedule for increasing PDE constraint strength.

**Implementation Approach**:
```yaml
# Config
curriculum:
  enabled: true
  warmup_epochs: 10          # Data loss only
  ramp_epochs: 20            # Linear ramp of physics
  final_physics_weight: 1.0  # Target weight
  schedule: linear           # linear, exponential, step
```

**Tasks**:
- [ ] Create `CurriculumScheduler` class
- [ ] Implement linear, exponential, step schedules
- [ ] Log active phase and current weights
- [ ] Add curriculum config to experiment configs

**Files to Create/Modify**:
- `src/training/curriculum_scheduler.py` (NEW)
- `src/training/trainer.py` (EXTEND)
- `configs/curriculum.yaml` (NEW)

---

### 2.4 Residual and Boundary Condition Test Suite
**Goal**: Unit tests verifying PDE residual code on known analytic solutions.

**Implementation Approach**:
```python
# Test cases
class TestGBMResidual:
    def test_exact_gbm_solution(self):
        """GBM residual should be ~0 for S(t) = S0 * exp((mu-0.5*sigma^2)*t + sigma*W_t)"""

    def test_ou_mean_reversion(self):
        """OU residual should detect deviation from mean."""
```

**Tasks**:
- [ ] Create test cases for GBM residual with analytic solution
- [ ] Create test cases for OU mean-reversion property
- [ ] Create test cases for Black-Scholes PDE
- [ ] Add boundary condition verification tests

**Files to Create/Modify**:
- `tests/test_pinn_residuals.py` (NEW)
- `tests/test_boundary_conditions.py` (NEW)

---

### 2.5 Numerical Stability Protections
**Goal**: Ensure robust training with gradient clipping, safe operations.

**Implementation Approach**:
```python
class NumericalStability:
    @staticmethod
    def safe_log(x: Tensor, eps: float = 1e-8) -> Tensor:
        return torch.log(x.clamp(min=eps))

    @staticmethod
    def safe_div(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        return a / (b + eps)
```

**Tasks**:
- [ ] Add `safe_log`, `safe_exp`, `safe_div` utilities
- [ ] Implement gradient clipping (configurable max norm)
- [ ] Add input/target normalization consistency checks
- [ ] Optional: mixed precision training with autocast

**Files to Create/Modify**:
- `src/utils/numerical_stability.py` (NEW)
- `src/training/trainer.py` (EXTEND)

---

## 3. Data Pipeline and Leakage Prevention

### 3.1 Leakage Audit Tooling
**Goal**: Automated checks for lookahead leakage.

**Implementation Approach**:
```python
class LeakageAuditor:
    def check_feature_dates(self, features_df, labels_df) -> List[LeakageWarning]:
        """Verify no feature uses future data relative to label date."""

    def check_scaler_fit(self, scaler, train_dates, test_dates) -> bool:
        """Verify scaler was fit only on train data."""
```

**Tasks**:
- [ ] Create `LeakageAuditor` class
- [ ] Add feature timestamp validation
- [ ] Add scaler/normalizer fit-date tracking
- [ ] Create CI check for leakage in PRs

**Files to Create/Modify**:
- `src/data/leakage_auditor.py` (NEW)
- `tests/test_leakage.py` (NEW)

---

### 3.2 Dataset Versioning
**Goal**: Hash-based versioning for reproducibility.

**Implementation Approach**:
```python
class DatasetVersioner:
    def compute_version_hash(self,
                             raw_data: pd.DataFrame,
                             feature_config: Dict,
                             split_config: Dict) -> str:
        """SHA256 of data + config = version ID."""

    def save_version(self, version_id: str, metadata: Dict):
        """Persist to versions table."""
```

**Tasks**:
- [ ] Implement hash computation for datasets
- [ ] Store version metadata in SQLite
- [ ] Add version validation on experiment load
- [ ] Create `datasets/` directory with version manifest

**Files to Create/Modify**:
- `src/data/dataset_versioner.py` (NEW)
- `src/data/version_manifest.py` (NEW)

---

### 3.3 Feature Provenance and Documentation
**Goal**: Every feature has formula, lag, source, availability time.

**Implementation Approach**:
```yaml
# features.yaml
features:
  returns_1d:
    formula: "(close[t] - close[t-1]) / close[t-1]"
    lag: 1
    source: "price"
    available_at: "close"

  volatility_20d:
    formula: "std(returns[-20:])"
    lag: 20
    source: "derived"
    available_at: "close"
```

**Tasks**:
- [ ] Create `features.yaml` with all feature definitions
- [ ] Build `FeatureRegistry` that validates features
- [ ] Add provenance logging to feature computation
- [ ] Generate feature documentation automatically

**Files to Create/Modify**:
- `configs/features.yaml` (NEW)
- `src/data/feature_registry.py` (NEW)

---

### 3.4 Missing Data and Outlier Policy
**Goal**: Explicit handling rules with tests.

**Implementation Approach**:
```python
class DataCleaner:
    def handle_missing(self, df: pd.DataFrame, config: MissingConfig) -> pd.DataFrame:
        # forward_fill_limit: 5
        # interpolation: linear
        # drop_if_missing_pct: 0.1

    def handle_outliers(self, df: pd.DataFrame, config: OutlierConfig) -> pd.DataFrame:
        # method: winsorize
        # lower_pct: 0.01
        # upper_pct: 0.99
```

**Tasks**:
- [ ] Define missing data policy (forward fill limits, interpolation)
- [ ] Implement winsorization for outliers
- [ ] Add data quality report generation
- [ ] Create tests for edge cases

**Files to Create/Modify**:
- `src/data/data_cleaner.py` (NEW)
- `configs/data_cleaning.yaml` (NEW)
- `tests/test_data_cleaning.py` (NEW)

---

## 4. Experiment Management and Reproducibility

### 4.1 Configuration-Driven Runs
**Goal**: Single config file per experiment.

**Implementation Approach**:
```yaml
# experiments/lstm_baseline_v1.yaml
experiment:
  name: "lstm_baseline_v1"
  model:
    type: lstm
    hidden_size: 128
    num_layers: 2
  training:
    epochs: 100
    learning_rate: 0.001
    batch_size: 32
  data:
    tickers: ["AAPL", "MSFT"]
    sequence_length: 30
```

**Tasks**:
- [ ] Design unified experiment config schema
- [ ] Create config loader with validation (Pydantic)
- [ ] Auto-save config alongside outputs
- [ ] Add config diff tool for comparing experiments

**Files to Create/Modify**:
- `src/config/experiment_config.py` (NEW)
- `configs/experiments/` (NEW directory)

---

### 4.2 Seed Control and Determinism
**Goal**: Full reproducibility with documented environment.

**Implementation Approach**:
```python
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_environment():
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "python_version": sys.version,
        "commit_hash": get_git_hash(),
    }
```

**Tasks**:
- [ ] Extend `reproducibility.py` with full seed control
- [ ] Add environment logging to every run
- [ ] Create determinism verification test
- [ ] Document non-deterministic operations

**Files to Create/Modify**:
- `src/utils/reproducibility.py` (EXTEND)
- `src/utils/environment_logger.py` (NEW)

---

### 4.3 Model Registry and Checkpoints
**Goal**: Track best models with full metadata.

**Implementation Approach**:
```
Models/
├── checkpoints/
│   ├── lstm_v1_seed42_window5_best.pt
│   └── pinn_gbm_v2_seed42_window5_best.pt
├── registry.json
│   {
│     "best_overall": "pinn_gbm_v2_seed42",
│     "best_by_regime": {
│       "high_vol": "pinn_gbm_v2_seed42",
│       "low_vol": "lstm_v1_seed42"
│     }
│   }
```

**Tasks**:
- [ ] Create `ModelCheckpointer` with metadata saving
- [ ] Build registry JSON with best model tracking
- [ ] Add model promotion CLI (`python -m promote_model`)
- [ ] Include commit hash in checkpoint metadata

**Files to Create/Modify**:
- `src/training/model_checkpointer.py` (NEW)
- `src/training/model_registry.py` (NEW)

---

### 4.4 Ablation Framework
**Goal**: Systematic ablation studies with auto-summary.

**Implementation Approach**:
```yaml
# ablations.yaml
ablations:
  curriculum:
    baseline: {curriculum_enabled: false}
    treatment: {curriculum_enabled: true}
  adaptive_weights:
    baseline: {adaptive_weighting: none}
    treatment: {adaptive_weighting: gradnorm}
```

**Tasks**:
- [ ] Create `AblationRunner` class
- [ ] Define ablation configs (curriculum, weights, PDE terms)
- [ ] Auto-generate comparison tables
- [ ] Add statistical significance to ablation results

**Files to Create/Modify**:
- `src/evaluation/ablation_runner.py` (NEW)
- `configs/ablations.yaml` (NEW)
- `scripts/run_ablations.py` (NEW)

---

## 5. Observability, Reporting, and Evidence Generation

### 5.1 Automatic Plotting Pack
**Goal**: Standard visualization suite for every experiment.

**Implementation Approach**:
```python
class PlotGenerator:
    def generate_all(self, results: ExperimentResults, output_dir: Path):
        self.plot_learning_curves(results)       # Loss per epoch
        self.plot_loss_components(results)       # Data vs physics losses
        self.plot_gradient_norms(results)        # Per-term gradients
        self.plot_residual_histograms(results)   # PDE residual distributions
        self.plot_rolling_performance(results)   # Window-by-window metrics
        self.plot_drawdown_curves(results)       # Cumulative returns + DD
```

**Tasks**:
- [ ] Create `PlotGenerator` class
- [ ] Implement learning curve plots (per loss term)
- [ ] Add gradient norm evolution plots
- [ ] Create residual histogram plots
- [ ] Build rolling performance charts

**Files to Create/Modify**:
- `src/reporting/plot_generator.py` (NEW)
- `src/reporting/plot_styles.py` (NEW)

---

### 5.2 Error Analysis Reports
**Goal**: Understand where forecasts fail.

**Implementation Approach**:
```python
class ErrorAnalyzer:
    def analyze_by_volatility(self, errors, volatility):
        """Error distribution in low/med/high vol regimes."""

    def analyze_by_event(self, errors, event_calendar):
        """Error around earnings, Fed, gaps."""

    def residual_error_correlation(self, pde_residuals, forecast_errors):
        """Does physics compliance correlate with accuracy?"""
```

**Tasks**:
- [ ] Create `ErrorAnalyzer` class
- [ ] Implement regime-stratified error analysis
- [ ] Add event-based error analysis
- [ ] Build residual vs error correlation study

**Files to Create/Modify**:
- `src/evaluation/error_analyzer.py` (NEW)
- `src/reporting/error_report.py` (NEW)

---

### 5.3 Exportable Report Artifacts
**Goal**: One-command report generation for dissertation.

**Implementation Approach**:
```bash
python -m generate_report --experiment pinn_v2 --format latex
# Outputs:
#   reports/pinn_v2/
#   ├── figures/
#   ├── tables/
#   └── results.tex
```

**Tasks**:
- [ ] Create report template (LaTeX + Markdown)
- [ ] Build `ReportGenerator` class
- [ ] Auto-generate figure captions
- [ ] Export publication-ready tables

**Files to Create/Modify**:
- `src/reporting/report_generator.py` (NEW)
- `templates/latex_report.tex` (NEW)
- `scripts/generate_report.py` (NEW)

---

## 6. Code Quality, Structure, and Testing

### 6.1 Modular Architecture Review
**Goal**: Clean separation of concerns.

**Current Structure**:
```
src/
├── data/          # ✅ Data fetching and preprocessing
├── models/        # ✅ Model definitions
├── training/      # ✅ Training logic
├── evaluation/    # 🟡 Needs expansion
├── trading/       # ✅ Trading strategies
├── simulation/    # ✅ Monte Carlo
├── web/           # ✅ Dashboards (legacy)
└── utils/         # ✅ Utilities
```

**Tasks**:
- [ ] Add `src/losses/` for loss functions
- [ ] Add `src/reporting/` for visualization
- [ ] Add `src/config/` for config management
- [ ] Review and minimize cross-dependencies

**Files to Create/Modify**:
- `src/losses/__init__.py` (NEW)
- `src/reporting/__init__.py` (NEW)
- `src/config/__init__.py` (NEW)

---

### 6.2 Typed Interfaces and Contracts
**Goal**: Type hints, dataclasses, shape assertions.

**Implementation Approach**:
```python
@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int

@dataclass
class ModelOutput:
    predictions: Tensor  # Shape: (batch, seq, 1)
    physics_residuals: Optional[Dict[str, Tensor]] = None
```

**Tasks**:
- [ ] Add type hints to all public functions
- [ ] Create dataclasses for configs
- [ ] Add tensor shape assertions at module boundaries
- [ ] Enable mypy in CI

**Files to Create/Modify**:
- `src/types/` (NEW directory)
- Add type hints across codebase

---

### 6.3 Unit and Integration Tests
**Goal**: Comprehensive test coverage.

**Test Categories**:
```
tests/
├── unit/
│   ├── test_residuals.py      # PDE residual functions
│   ├── test_transforms.py     # Data transforms
│   ├── test_metrics.py        # Metric calculations
│   └── test_loss_functions.py # Loss computations
├── integration/
│   ├── test_training_loop.py  # Mini end-to-end training
│   └── test_evaluation.py     # Full evaluation pipeline
└── fixtures/
    └── tiny_dataset.py        # Minimal test data
```

**Tasks**:
- [ ] Create unit tests for residual functions
- [ ] Create unit tests for data transforms
- [ ] Create unit tests for all metrics
- [ ] Create integration test with tiny dataset
- [ ] Achieve >80% coverage on core modules

**Files to Create/Modify**:
- `tests/unit/` (NEW directory)
- `tests/integration/` (NEW directory)
- `tests/fixtures/` (NEW directory)

---

### 6.4 Static Checks and CI
**Goal**: Automated quality checks.

**Implementation Approach**:
```yaml
# .github/workflows/ci.yml
jobs:
  lint:
    - ruff check src/ tests/
    - black --check src/ tests/
  type-check:
    - mypy src/
  test:
    - pytest tests/ --cov=src
```

**Tasks**:
- [ ] Add ruff configuration
- [ ] Add black formatter
- [ ] Enable mypy strict mode
- [ ] Create CI workflow for PRs

**Files to Create/Modify**:
- `.github/workflows/ci.yml` (NEW)
- `pyproject.toml` (EXTEND)
- `ruff.toml` (NEW)

---

## Implementation Schedule (Suggested Order)

### Phase 1: Foundation (Critical Path)
1. **3.1 Leakage Audit** - Ensure data integrity first
2. **4.1 Configuration-Driven Runs** - Foundation for reproducibility
3. **4.2 Seed Control** - Required for valid comparisons
4. **1.1 Evaluation Harness** - Unified evaluation interface

### Phase 2: Evaluation Infrastructure
5. **1.2 Walk-Forward Validation** - Non-stationarity handling
6. **1.4 Statistical Significance** - Rigorous comparisons
7. **1.5 Trading Strategy Evaluation** - Real-world relevance
8. **1.6 Leaderboard** - Result tracking

### Phase 3: PINN Improvements
9. **2.1 Loss Diagnostics** - Understand training dynamics
10. **2.2 Adaptive Loss Weighting** - Improve convergence
11. **2.3 Curriculum Scheduler** - Stabilize physics integration
12. **2.4 Residual Tests** - Validate correctness

### Phase 4: Reporting & Polish
13. **5.1 Plotting Pack** - Visualization suite
14. **5.2 Error Analysis** - Failure mode understanding
15. **5.3 Report Generation** - Dissertation artifacts
16. **1.3 Regime Testing** - Stress analysis

### Phase 5: Quality & Maintenance
17. **6.3 Unit Tests** - Coverage improvement
18. **6.4 CI Pipeline** - Automated checks
19. **3.2-3.4 Data Pipeline** - Full provenance
20. **6.1-6.2 Architecture** - Clean up

---

## Quick Start Commands

```bash
# Run evaluation harness (once implemented)
python -m src.evaluation.harness --config configs/eval.yaml

# Run ablation study
python -m scripts.run_ablations --ablation curriculum

# Generate report
python -m scripts.generate_report --experiment pinn_v2

# Run all tests
pytest tests/ -v --cov=src

# Check code quality
ruff check src/ && black --check src/ && mypy src/
```

---

## Notes

- All new modules should follow existing patterns in `src/`
- Document all changes in `DOCUMENT.md`
- Create issues/tasks for each major item
- Prioritize items that impact dissertation results chapter
