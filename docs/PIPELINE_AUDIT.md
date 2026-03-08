# Complete Audit File — Required Changes and Fixes for the Financial Forecasting Pipeline

## Project context

This audit captures **all major changes, fixes, and refactors** required to move the project from a narrow **single-series S&P 500 workflow** into a **multi-asset, adjusted-price, leakage-safe, reproducible financial forecasting pipeline** suitable for dissertation-grade experiments across baseline deep learning models and PINN-style models.

It consolidates the earlier design discussion into one practical audit document covering:

- pipeline-wide architectural changes
- data and preprocessing fixes
- training and evaluation fixes
- model-specific changes
- fairness requirements for comparison
- local caching without a database
- file-by-file refactor implications
- experiment design implications
- critical risks if the old pipeline is left unchanged

---

# 1. Executive audit summary

## 1.1 Core finding

The original pipeline is too dependent on a **single-ticker forecasting assumption**. It likely assumes one asset, one target series, one close column, one sequence structure, and one naive preprocessing path. That is not robust enough for a serious dissertation comparing multiple model families.

## 1.2 Main upgrade required

The codebase must be refactored into a **market-state modelling pipeline** built around:

- a **multi-asset universe**
- **adjusted prices** rather than raw close wherever return targets/features are built
- **formal data quality checks**
- **calendar-aware alignment**
- **leakage-safe preprocessing**
- **identical benchmark conditions** for fair model comparison
- **local file-based caching and versioned artefacts**
- **explicit separation between core benchmark tasks and specialised volatility/PINN extensions**

## 1.3 Practical consequence

This is not just a matter of adding more features. It changes:

- ingestion assumptions
- target construction
- feature engineering logic
- splitting/scaling order
- sequence building
- model input contracts
- evaluation semantics
- reproducibility practices
- how fairness is enforced across models

---

# 2. Old pipeline versus corrected pipeline

## 2.1 Old conceptual pipeline

The old flow is effectively:

```text
download ^GSPC
→ clean data
→ compute indicators
→ scale data
→ create sequences
→ train model
→ predict next value
→ evaluate
```

## 2.2 Corrected pipeline

The corrected flow should be:

```text
define market universe
→ download multi-asset raw data
→ cache locally
→ run data quality checks
→ align to a master trading calendar
→ build adjusted-return targets and features
→ engineer per-asset, cross-asset, and regime features
→ split by time
→ fit preprocessing on train only
→ create identical sequence windows for all core models
→ train under matched benchmark conditions
→ evaluate forecasting accuracy, trading utility, and regime robustness
→ save metrics, plots, configs, and dataset metadata
```

## 2.3 Audit judgement

The original pipeline is acceptable for prototyping, but not for scientifically defensible comparison across:

- LSTM/GRU/BiLSTM
- attention/transformer models
- baseline PINN variants
- volatility-focused models
- advanced structured PINNs

---

# 3. Data ingestion audit

## 3.1 Issue found

The pipeline appears conceptually tied to a single ticker such as `^GSPC`.

## 3.2 Why this is a problem

A single equity index series is too narrow to capture:

- sector rotation
- rate sensitivity
- volatility regimes
- commodity-linked macro stress
- cross-asset risk transitions

This causes the model to learn from one path rather than a broader market state.

## 3.3 Required fix

Refactor the ingestion layer so that it accepts a **universe of tickers**, not a single ticker.

### Minimum acceptable universe
- SPY
- QQQ
- IWM
- ^VIX
- ^TNX
- a small set of sector ETFs such as XLK, XLF, XLE
- optionally GC=F and CL=F

### Better extended universe
- SPY, QQQ, DIA, IWM
- sector ETFs: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB
- volatility: ^VIX
- rates: ^TNX, ^IRX, TLT, IEF
- macro commodities: GC=F, CL=F
- optional FX sentiment proxies: EURUSD=X, DX-Y.NYB

## 3.4 Required code change

Replace single-ticker assumptions like:

```python
download_data(ticker="^GSPC", ...)
```

with multi-ticker ingestion such as:

```python
    tickers=["SPY", "QQQ", "IWM", "^VIX", "^TNX", "GC=F"],
    start=...,
    end=...,
    interval="1d"
)
```

## 3.5 Audit action

**Mandatory refactor**.

---

# 4. Raw data structure audit

## 4.1 Issue found

A single-asset DataFrame design encourages hidden assumptions in later steps.

## 4.2 Required fix

Use two representations:

### Raw storage
Use **long format**:

```text
Date | Ticker | Open | High | Low | Close | Adj Close | Volume | Dividends | Stock Splits
```

### Modelling / feature engineering
Use **wide format**:

```text
Date | SPY_adj_close | QQQ_adj_close | VIX_level | ...
```

## 4.3 Why this matters

Long format is better for:
- auditing
- per-asset QA
- debugging data issues

Wide format is better for:
- feature building
- sequence construction
- model training

## 4.4 Audit action

**Strongly recommended structural change**.

---

# 5. Adjusted price audit

## 5.1 Issue found

There is a risk the old pipeline mixes raw Close and Adj Close inconsistently.

## 5.2 Why this is a serious problem

Raw close can contain non-economic jumps caused by:
- stock splits
- dividend adjustments
- corporate actions

This can distort:
- returns
- targets
- rolling volatility
- backtests
- error metrics

## 5.3 Required fix

Use **Adj Close consistently** whenever constructing:
- return targets
- log-return targets
- momentum features
- trend features
- realised volatility from price series

## 5.4 Minimum implementation rule

If the task is return forecasting:

```python
log_return = np.log(adj_close / adj_close.shift(1))
target = log_return.shift(-1)
```

## 5.5 Audit action

**Mandatory fix**.

---

# 6. Data quality audit

## 6.1 Issue found

The old style pipeline often assumes downloaded data is already clean.

## 6.2 Why this is unsafe

Free market data sources can contain:
- missing sessions
- inconsistent field availability
- duplicate timestamps
- sparse or partially missing asset histories
- action-related jumps
- unexpected NaN patterns

## 6.3 Required QA checks

Every download must produce a QA report including:

- first available date per asset
- last available date per asset
- number of rows
- duplicate timestamp count
- missing session count
- percentage coverage against master calendar
- zero or negative prices
- null counts per column
- extreme return jumps
- action events (dividends, splits)
- missingness after calendar alignment

## 6.4 Extreme-return audit rule

Flag suspicious returns beyond a configured threshold, for example:
- absolute daily return > 20% for broad ETFs/indices
- threshold adjusted by asset class if needed

## 6.5 Audit action

**Mandatory new validation stage**.

---

# 7. Calendar alignment audit

## 7.1 Issue found

The original pipeline likely assumes all assets share the same calendar or drops NaNs too aggressively.

## 7.2 Why this is dangerous

Different instruments may have:
- different trading schedules
- different holiday coverage
- different listing start dates
- sparse macro-like availability

Careless filling or dropping can create:
- false continuity
- hidden leakage
- broken sequence windows
- misaligned targets/features

## 7.3 Required fix

Define a **master trading calendar**, usually based on US equity sessions if SPY is the target.

Then:
- reindex all assets to this calendar
- apply feature-specific fill policies
- record post-alignment missingness

## 7.4 Fill-policy audit rules

### Price-like levels
Forward-fill only when economically sensible and document it.

### Returns
Do not forward-fill returns.

### Volume
Do not blindly forward-fill.

### Event fields
Use event-aware logic, not generic filling.

### Sparse macro proxies
Carry forward cautiously and log staleness if used.

## 7.5 Audit action

**Mandatory alignment module required**.

---

# 8. Target definition audit

## 8.1 Issue found

The target may be inconsistently defined across models or tied too closely to raw price levels.

## 8.2 Required benchmark target

Use a common main target for core fair comparison:
**SPY next-day adjusted log return**

## 8.3 Why this target is preferred

- tradeable proxy
- liquid and stable
- more meaningful for financial metrics than raw index levels
- easier to compare across models
- avoids scale problems from raw prices

## 8.4 Specialised targets allowed in separate experiments

### Volatility extension
- realised volatility
- rolling squared return measures
- downside volatility
- joint return-volatility target

### Option-related extension
- option price
- implied volatility
- volatility surface proxy
- only if option-specific data is actually added

## 8.5 Audit action

**Mandatory standardisation for the benchmark task**.

---

# 9. Feature engineering audit

## 9.1 Issue found

The old feature builder likely assumes one series and local technical indicators only.

## 9.2 Required feature classes

The new pipeline must support three feature families.

## 9.3 Per-asset features
Examples:
- 1-day log return
- 5-day and 20-day return
- rolling volatility
- moving average ratios
- rolling drawdown
- volume z-score
- realised range proxies

## 9.4 Cross-asset features
Examples:
- SPY vs QQQ relative strength
- sector spread measures
- SPY and VIX interaction
- equity vs rates change
- oil vs equities context
- bond ETF vs equity ETF ratio

## 9.5 Regime features
Examples:
- high-volatility regime flags
- trend regime
- drawdown regime
- macro stress indicators
- breadth approximations if available

## 9.6 Required implementation change

Feature engineering must become modular rather than being one monolithic preprocessing function.

## 9.7 Audit action

**Mandatory feature-pipeline refactor**.

---

# 10. Preprocessing and leakage audit

## 10.1 Issue found

A common failure mode is fitting scalers on the full dataset before splitting.

## 10.2 Why this invalidates experiments

If the scaler sees future data, then:
- validation/test information leaks into training
- performance is inflated
- model comparisons become unfair

## 10.3 Required corrected order

```text
build cleaned features
→ split chronologically
→ fit scaler on train only
→ transform val/test using train-fitted scaler
```

## 10.4 Required code fix

Replace:

```python
scaled = scaler.fit_transform(full_df[feature_cols])
```

with:

```python
scaler.fit(train_df[feature_cols])
train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])
```

## 10.5 Additional requirement

Serialise the fitted scaler and save feature metadata.

## 10.6 Audit action

**Mandatory fix**.

---

# 11. Sequence generation audit

## 11.1 Issue found

Sequence builders in single-series pipelines usually assume:
- one target column
- one feature space
- one rigid layout

## 11.2 Required fix

The sequence builder must become configurable by:
- target asset
- target column
- lookback window
- horizon
- feature list
- split boundaries

## 11.3 Benchmark requirement

All models in the core benchmark must receive:
- the same lookback window
- the same target construction
- the same train/val/test split
- the same base feature universe

## 11.4 Audit action

**Mandatory dataset-builder redesign**.

---

# 12. Fair comparison audit

## 12.1 Issue found

Comparisons are only fair if models differ in architecture/loss, not in hidden data advantages.

## 12.2 Mandatory fairness rules

For the **core benchmark**, all models must use:

- the same target
- the same date ranges
- the same sequence windows
- the same base feature set
- the same missing-data policy
- the same train-only scaling policy
- comparable optimisation effort where practical
- the same evaluation framework

## 12.3 Models included in the core benchmark

- LSTM
- GRU
- BiLSTM
- Attention-LSTM
- Transformer
- baseline PINN
- GBM PINN
- OU PINN
- Black-Scholes PINN
- global PINN

## 12.4 Separate specialised experiment groups

### Volatility extension
- vol_lstm
- vol_gru
- vol_transformer
- vol_pinn
- heston_pinn
- stacked_vol_pinn

### Advanced structured PINN extension
- residual PINN
- stacked PINN
- spectral PINN
- financial DP-PINN
- adaptive dual-phase PINN

## 12.5 Audit action

**Mandatory experimental separation**.

---

# 13. Model-by-model audit — what must change

## 13.1 LSTM

### Current likely assumption
Single-series input, modest feature count.

### Required minimum changes
- support dynamic input dimension
- accept wider multi-asset feature tensor
- use stronger regularisation if feature space expands
- ensure sequence builder passes consistent shape

### Recommended changes
- dropout tuning
- weight decay
- possibly smaller hidden size if feature explosion causes overfitting
- optional feature projection layer before recurrent block

### Audit judgement
**Moderate architectural adjustment required**.

---

## 13.2 GRU

### Required minimum changes
- same as LSTM: dynamic input dimension
- stronger regularisation
- ensure fair benchmark input consistency

### Recommended changes
- feature projection layer
- hidden-dimension retuning
- early stopping and better validation monitoring

### Audit judgement
**Moderate change required**.

---

## 13.3 BiLSTM

### Issue
Bidirectionality can be problematic for causal forecasting if not handled correctly.

### Required minimum changes
- verify no future leakage through sequence construction or interpretation
- if used strictly in a forecasting benchmark, justify causal validity carefully

### Recommended changes
- consider replacing or clearly labelling as non-causal representation benchmark
- dynamic input dimension
- stronger regularisation

### Audit judgement
**Needs careful methodological justification**.

---

## 13.4 Attention-LSTM

### Required minimum changes
- support wider feature space
- attention mechanism should handle richer cross-asset context
- dynamic input dimension

### Recommended changes
- improved attention stabilisation
- longer lookback tuning
- pre-attention projection layer
- stronger regularisation

### Audit judgement
**Well suited to richer input, but needs stabilisation**.

---

## 13.5 Transformer

### Required minimum changes
- input projection from wide feature space into model dimension
- time/positional encoding
- masking and sequence handling done correctly
- stronger regularisation because financial datasets are not huge

### Recommended changes
- smaller, efficient transformer rather than oversized architecture
- careful control of context length
- pre-norm and dropout tuning
- avoid over-parameterisation

### Audit judgement
**Good fit for richer multi-asset inputs, but easy to overfit**.

---

## 13.6 Baseline PINN

### Current likely assumption
Physics residual based on time and one series only.

### Required minimum changes
- physics loss should be reviewed to allow richer state information
- can no longer assume target dynamics depend only on time and one scalar state
- include volatility/rate context where justified

### Recommended changes
- context-aware physics residual
- exogenous market-state inputs
- better control of lambda weighting between data loss and physics loss

### Audit judgement
**Loss redesign likely required**.

---

## 13.7 GBM PINN

### Required minimum changes
- ensure target/physics assumptions match the benchmark task
- use volatility proxy in the residual if needed
- do not pretend pure GBM perfectly matches market reality; frame as structured inductive bias

### Recommended changes
- state-dependent volatility proxy
- exogenous context terms
- compare against non-physics baseline under identical data

### Audit judgement
**Conceptually valid as diffusion-informed regulariser, but residual should be modernised**.

---

## 13.8 OU PINN

### Issue
OU dynamics are more naturally mean-reverting.

### Required minimum changes
- if kept in the main benchmark, explicitly frame it as a hypothesis test
- consider redirecting it toward mean-reverting targets such as volatility, spreads, residuals, or standardised deviations

### Recommended changes
- alternative task assignment
- regime- or spread-based inputs
- clearer interpretation in results

### Audit judgement
**Potential mismatch for raw equity return forecasting unless carefully framed**.

---

## 13.9 Black-Scholes PINN

### Issue
Black-Scholes is most natural for option pricing or diffusion-based price evolution under assumptions.

### Required minimum changes
- if the project does not use option targets, present this model as a diffusion-based regulariser, not a full option-pricing model
- include volatility and risk-free rate proxies if using a BS-style residual

### Recommended changes
- options extension only if actual option chain targets and related features are introduced
- time-to-horizon handling if relevant

### Audit judgement
**Needs reframing unless options data is added**.

---

## 13.10 Global PINN

### Required minimum changes
- more general state representation
- broader input encoding for market-state features
- loss must remain consistent across a wider exogenous feature space

### Recommended changes
- latent state encoder
- context-conditioned residual
- shared trunk with specialised head

### Audit judgement
**Likely needs architectural broadening**.

---

## 13.11 vol_lstm

### Required minimum changes
- switch to volatility-oriented target
- stop treating it as the same task as return forecasting
- use volatility-specific features

### Recommended changes
- dual-output head for return and volatility
- variance-aware loss

### Audit judgement
**Must be moved into specialised volatility experiment**.

---

## 13.12 vol_gru

Same audit as vol_lstm.

### Audit judgement
**Specialised volatility model only**.

---

## 13.13 vol_transformer

### Required minimum changes
- volatility target
- volatility-oriented feature set
- careful sequence-length control

### Recommended changes
- dual-head architecture
- uncertainty-aware output

### Audit judgement
**Specialised model, not core benchmark**.

---

## 13.14 vol_pinn

### Required minimum changes
- volatility target or joint task required
- physics residual should target volatility dynamics rather than ordinary return dynamics

### Recommended changes
- realised-volatility residual
- Heston-like or variance-process constraints

### Audit judgement
**Belongs in specialised volatility extension**.

---

## 13.15 heston_pinn

### Required minimum changes
- explicit volatility/variance state
- volatility-aware target and loss design
- not comparable directly to return-only models without caveat

### Recommended changes
- dual-state system for return and variance
- heteroscedastic loss
- volatility proxy calibration

### Audit judgement
**Strong candidate for volatility extension, not main fair benchmark**.

---

## 13.16 stacked_vol_pinn

### Required minimum changes
- same volatility-specialised framing
- verify stacked design actually helps under fair evaluation

### Recommended changes
- stage 1 feature/state extraction, stage 2 variance-dynamics refinement

### Audit judgement
**Advanced volatility extension model**.

---

## 13.17 Residual PINN

### Required minimum changes
- clear residual connections to stabilise deeper structure on richer features
- same benchmark dataset first before specialised expansion

### Recommended changes
- regime-aware residual blocks
- shared trunk plus residual expert components

### Audit judgement
**Good advanced extension candidate**.

---

## 13.18 Stacked PINN

### Required minimum changes
- justify why stacking is needed
- keep benchmark data identical for first comparison

### Recommended changes
- stage-wise training
- hierarchical feature extraction
- explicit separation between market-state encoding and physics-constrained refinement

### Audit judgement
**Advanced extension, useful after benchmark stabilises**.

---

## 13.19 Spectral PINN

### Required minimum changes
- clarify whether spectral treatment matches financial target dynamics
- support richer state inputs

### Recommended changes
- frequency-aware encoding
- use for cyclical/regime signatures rather than as a default benchmark

### Audit judgement
**Research extension, not first-line baseline**.

---

## 13.20 financial DP-PINN

### Required minimum changes
- define clearly what the dual phases represent
- ensure inputs include regime/context features if claiming stress-vs-calm separation

### Recommended changes
- one phase for coarse market-state fitting, second phase for constraint refinement
- dynamic lambda scheduling
- regime-conditioned training

### Audit judgement
**High-value advanced extension if designed carefully**.

---

## 13.21 adaptive dual-phase PINN

### Required minimum changes
- same as financial DP-PINN, plus explicit adaptation logic
- adaptation must be driven by measurable regime or residual behaviour, not vague heuristics

### Recommended changes
- phase weighting based on volatility regime
- adaptive loss balancing
- regime-aware subnetworks or gating

### Audit judgement
**Powerful but high-complexity extension; not suitable as first comparison model without a stable benchmark first**.

---



# 13A. Model-by-model testing audit — what each model should be trained and tested on

This section extends the model audit by making the **training task and testing task explicit for each model family**. A major fairness rule is that a model should be evaluated on the task that matches its underlying assumptions. A model inspired by option-pricing PDEs should not be presented as though it were naturally validated by ordinary next-day equity-return forecasting unless the dissertation clearly frames that as a deliberately exploratory transfer experiment.

## 13A.1 Core principle

Each model must be tested on a task aligned to:

- its mathematical assumptions
- its output type
- the meaning of its loss function
- the economic quantity it is designed to model

That means:
- return-forecasting models should be tested on return forecasting
- volatility models should be tested on volatility prediction
- option-pricing PDE models should be tested on option prices or implied-volatility-related targets
- mean-reversion models should be tested on mean-reverting targets unless explicitly treated as hypothesis tests on other tasks

## 13A.2 LSTM

### Best primary training/testing task
- SPY next-day adjusted log return forecasting
- optionally multi-horizon return forecasting

### Appropriate test data
- same common benchmark dataset used by other core benchmark models
- shared multi-asset input features
- chronological train/validation/test split

### Appropriate evaluation
- forecast metrics: RMSE, MAE, directional accuracy
- financial metrics from a simple, fixed trading rule
- regime-sliced evaluation

### Optional secondary task
- realised volatility prediction, but only in a separate volatility experiment

## 13A.3 GRU

### Best primary training/testing task
- same as LSTM: next-day adjusted return forecasting on the common benchmark dataset

### Appropriate evaluation
- same as LSTM for fair comparison

### Notes
GRU is a compact benchmark and should remain in the main return-forecasting comparison before any specialised tasks are introduced.

## 13A.4 BiLSTM

### Best primary training/testing task
- if kept, test on the same common return-forecasting benchmark as LSTM/GRU

### Important methodological note
Because bidirectional models can raise causality concerns in forecasting, testing must ensure:
- no future leakage in window construction
- the dissertation explicitly explains why the comparison is acceptable

### Appropriate evaluation
- same return-forecasting benchmark metrics
- plus a note on causal interpretation limitations

## 13A.5 Attention-LSTM

### Best primary training/testing task
- next-day adjusted return forecasting with richer multi-asset context
- possibly medium-horizon forecasting if explored consistently across models

### Appropriate test data
- same common benchmark dataset
- same base feature universe
- same split boundaries

### Appropriate evaluation
- same benchmark forecasting and financial metrics
- attention diagnostics are optional and supplementary

## 13A.6 Transformer

### Best primary training/testing task
- next-day adjusted return forecasting on the same common benchmark
- longer-context return forecasting may be an optional extension if applied fairly

### Appropriate test data
- identical shared benchmark dataset as other core models

### Appropriate evaluation
- same return and trading metrics
- calibration/robustness diagnostics if probability-style outputs are added

## 13A.7 Baseline PINN

### Best primary training/testing task
- return forecasting on the shared benchmark task
- target: SPY next-day adjusted log return

### Appropriate training/testing interpretation
This model should be treated as a **physics-regularised return forecaster**, not as a pure closed-form finance model.

### Appropriate evaluation
- same benchmark metrics as LSTM/GRU/Transformer
- additional reporting on physics residual magnitude if useful

## 13A.8 GBM PINN

### Best primary training/testing task
- equity return or price-dynamics forecasting under a GBM-inspired regularisation framework
- use the same benchmark target first for fairness

### Appropriate evaluation
- same core return-forecasting benchmark metrics
- extra analysis showing whether the GBM constraint improves stability/generalisation

### Important note
GBM PINN does **not** need to be tested on option prices by default. It is acceptable on return/price-dynamics tasks because GBM is a price-process prior rather than specifically an option-pricing PDE.

## 13A.9 OU PINN

### Best mathematically aligned task
- mean-reverting targets such as:
  - realised volatility
  - spread series
  - residuals from trend
  - deviation-from-moving-average signals
  - rate spread proxies
  - normalised market-stress indicators

### If used in the core benchmark
It may still be trained/tested on SPY next-day adjusted returns, but only as an exploratory hypothesis test and not as the most naturally matched task.

### Best evaluation
- if on return benchmark: same core metrics, with explicit caveat
- if on mean-reverting task: task-appropriate metrics plus mean-reversion diagnostics

## 13A.10 Black-Scholes PINN

### Best mathematically aligned task
- option pricing
- implied volatility approximation
- option surface or option Greek-related supervised targets
- possibly short-horizon option price dynamics conditioned on underlying state, volatility, strike, maturity, and risk-free rate

### Training data that should be used
If claiming a true Black-Scholes PDE-aligned model, training and testing should use option-specific data such as:
- option prices
- strike
- maturity/time-to-expiry
- underlying spot price
- risk-free rate proxy
- volatility proxy or implied volatility if target design justifies it
- option type (call/put)

### Correct evaluation if used as a true Black-Scholes PDE model
- option-pricing error metrics: RMSE, MAE, relative pricing error
- implied-volatility error if relevant
- performance across moneyness buckets
- performance across maturity buckets
- performance across volatility regimes
- comparison against Black-Scholes closed-form baseline and non-PINN neural baselines

### Important audit requirement
If the dissertation keeps Black-Scholes PINN in the main return-forecasting benchmark, it must be clearly reframed as a **diffusion-informed regulariser** rather than a true option-pricing PDE model. If it is positioned as a Black-Scholes PDE model in the strict sense, then it should be trained and tested on **options pricing data**, not ordinary SPY next-day returns.

## 13A.11 Global PINN

### Best primary training/testing task
- shared benchmark return-forecasting task using the multi-asset market-state dataset

### Appropriate evaluation
- same benchmark metrics as other core models
- possibly additional analysis on whether global/shared latent structure improves robustness across regimes

## 13A.12 vol_lstm

### Best primary training/testing task
- realised volatility prediction
- possibly multi-horizon volatility forecasting

### Appropriate target examples
- 5-day realised volatility
- 10-day realised volatility
- rolling standard deviation of returns
- downside semivolatility

### Appropriate evaluation
- volatility RMSE/MAE
- QLIKE-style volatility evaluation if implemented
- risk-sensitive trading or hedging utility analysis

### Audit requirement
This model should not be judged mainly by next-day return RMSE unless used in a dual-task setting.

## 13A.13 vol_gru

### Best primary training/testing task
- same as vol_lstm: realised volatility forecasting

### Appropriate evaluation
- same volatility metrics and risk-aware diagnostics

## 13A.14 vol_transformer

### Best primary training/testing task
- realised volatility or joint volatility-state forecasting

### Appropriate evaluation
- volatility metrics
- regime robustness
- uncertainty calibration if probabilistic outputs are used

## 13A.15 vol_pinn

### Best primary training/testing task
- volatility dynamics forecasting under a physics-inspired variance-process prior

### Appropriate data
- realised-volatility targets
- VIX-related context
- return-derived variance measures
- possibly options-implied volatility if available

### Appropriate evaluation
- volatility-specific forecasting metrics
- comparison against non-PINN volatility models

## 13A.16 heston_pinn

### Best mathematically aligned task
- joint modelling of price/return and stochastic variance
- volatility forecasting
- option pricing with stochastic volatility extension if sufficient data exists

### Recommended testing setups
#### Setup A — dissertation-friendly volatility task
- target: realised volatility or joint return-volatility outputs
- evaluate with volatility metrics and financial diagnostics

#### Setup B — more advanced derivatives task
- train/test on option prices with stochastic-volatility-aware state inputs
- compare against Black-Scholes and simpler neural option models

### Audit requirement
If Heston-style dynamics are claimed strongly, testing should involve explicit variance-aware targets and ideally option-sensitive evaluation, not only ordinary return forecasting.

## 13A.17 stacked_vol_pinn

### Best primary training/testing task
- advanced volatility forecasting
- multi-stage variance/state estimation

### Appropriate evaluation
- same as other volatility models
- plus ablation showing whether stacking helps materially

## 13A.18 Residual PINN

### Best primary training/testing task
- same shared benchmark return-forecasting task first
- later, possibly richer regime-aware tasks

### Appropriate evaluation
- same benchmark metrics in first comparison
- then ablations showing whether residual depth improves performance or stability

## 13A.19 Stacked PINN

### Best primary training/testing task
- same shared benchmark return-forecasting task first
- then more advanced regime-sensitive forecasting

### Appropriate evaluation
- same benchmark metrics initially
- then stage-wise ablations and robustness analysis

## 13A.20 Spectral PINN

### Best primary training/testing task
- tasks where cyclical structure, frequency content, or regime oscillation matter
- can still be tested on the benchmark return task, but should also be assessed on whether spectral features actually add value

### Appropriate evaluation
- same benchmark metrics if included there
- plus ablation on spectral encoding usefulness

## 13A.21 financial DP-PINN

### Best primary training/testing task
- regime-sensitive return forecasting
- stress-vs-calm market modelling
- two-phase optimisation for difficult, non-stationary financial tasks

### Appropriate evaluation
- same benchmark return metrics if used in core-like tasks
- additional regime-wise performance reporting
- phase-ablation results

## 13A.22 adaptive dual-phase PINN

### Best primary training/testing task
- regime-aware forecasting where loss balance or model behaviour changes across conditions
- especially useful for high-volatility vs low-volatility states

### Appropriate evaluation
- same benchmark metrics if predicting returns
- explicit high-volatility/low-volatility breakdown
- adaptation-ablation analysis

## 13A.23 Summary audit rule

### Core benchmark return task
Train/test on:
- LSTM
- GRU
- BiLSTM
- Attention-LSTM
- Transformer
- baseline PINN
- GBM PINN
- global PINN
- optionally OU PINN and Black-Scholes PINN, but only with clear caveats if their natural mathematical domain is not the same task

### Mean-reversion-oriented task
Best suited for:
- OU PINN

### Volatility task
Best suited for:
- vol_lstm
- vol_gru
- vol_transformer
- vol_pinn
- heston_pinn
- stacked_vol_pinn

### Option-pricing / derivatives task
Best suited for:
- Black-Scholes PINN
- potentially Heston PINN in a more advanced stochastic-volatility derivatives extension

The dissertation should therefore avoid claiming that all model families are being tested on their most natural problem if they are all forced into one single next-day equity-return benchmark. A fair dissertation can still use one common benchmark for comparison, but it must clearly distinguish:
- **benchmark comparability**
from
- **mathematical task alignment**


# 14. Training pipeline audit

## 14.1 Issue found

Training scripts in narrow pipelines often bake in:
- one target column
- one scaler
- one feature list
- weak metadata tracking

## 14.2 Required fix

Training must become config-driven.

Required config dimensions:
- ticker universe
- target asset
- target type
- feature blocks used
- lookback
- split boundaries
- scaler type
- model hyperparameters
- seed
- dataset version

## 14.3 Audit action

**Strongly recommended refactor**.

---

# 15. Evaluation audit

## 15.1 Issue found

A major risk is mixing:
- scaled predictions with unscaled actuals
- returns with prices
- volatility predictions with return-oriented metrics

## 15.2 Required evaluation separation

### Forecasting metrics
- RMSE
- MAE
- MSE
- R²
- directional accuracy

### Financial metrics
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- total return
- annualised return
- max drawdown
- volatility
- win rate
- turnover

### Diagnostic/regime metrics
- bull vs bear performance
- high-VIX vs low-VIX performance
- drawdown-period performance
- prediction distribution diagnostics
- calibration where relevant

## 15.3 Required metadata in evaluation code

Every evaluation call should explicitly know:
- target type
- whether values are scaled
- asset being evaluated
- strategy logic used

## 15.4 Audit action

**Mandatory evaluation refactor**.

---

# 16. Reproducibility and experiment tracking audit

## 16.1 Issue found

Without versioning, results cannot be reliably reproduced.

## 16.2 Required fix

Save:
- universe of tickers
- date range
- raw-data cache key
- feature config
- split dates
- scaler artefact
- model config
- seed
- metrics
- plots
- QA report

## 16.3 Audit action

**Mandatory for dissertation-grade methodology**.

---

# 17. Local caching audit — no database required

## 17.1 Requirement

A database is not necessary for this stage. A local file-based cache is sufficient.

## 17.2 Required cache design

Use:
- **Parquet** for data
- **JSON** for metadata and QA reports
- optionally pickle/joblib for scaler/model artefacts

## 17.3 Suggested structure

```text
  raw_cache/
    daily/
      SPY_QQQ_IWM_VIX_TNX/
        2010-01-01_2025-12-31/
          prices.parquet
          actions.parquet
          metadata.json
          qa_report.json
  processed/
    dataset_v1/
      features.parquet
      targets.parquet
      feature_config.json
      split_info.json
      scaler.pkl
artifacts/
  models/
  evaluations/
  plots/
```

## 17.4 Metadata to save

```json
{
  "tickers": ["SPY", "QQQ", "IWM", "^VIX", "^TNX"],
  "start": "2010-01-01",
  "end": "2025-12-31",
  "interval": "1d",
  "downloaded_at": "2026-03-07T12:00:00",
  "source": "yfinance",
  "price_field": "Adj Close",
  "cache_version": "1.0"
}
```

## 17.5 Why Parquet instead of CSV

Parquet is:
- faster to load
- compressed
- columnar
- more consistent for repeated experiments
- more efficient for larger feature sets

## 17.6 Audit action

**Recommended immediate implementation**.

---

# 18. File-by-file refactor audit

## 18.1 `data/downloader.py`
### Old problem
Single-ticker logic and no robust caching.
### Required fixes
- support ticker lists
- retries/backoff
- local cache read/write
- store actions and metadata

---

## 18.2 `data/quality.py`
### New module required
- coverage checks
- duplicate detection
- negative/zero price checks
- jump detection
- missingness report
- action-event report

---

## 18.3 `data/calendar.py`
### New module required
- master calendar definition
- alignment logic
- fill-policy handling
- post-alignment diagnostics

---

## 18.4 `features/returns.py`
### New/expanded
- adjusted-return computation
- horizon targets
- rolling return features

---

## 18.5 `features/trend.py`
### New/expanded
- moving averages
- momentum
- trend ratios
- drawdown features

---

## 18.6 `features/volatility.py`
### New/expanded
- rolling vol
- downside vol
- realised range proxies
- volatility target creation

---

## 18.7 `features/cross_asset.py`
### New module required
- relative strength
- spread features
- correlation/interplay proxies
- risk-on/risk-off context

---

## 18.8 `features/regimes.py`
### New module required
- volatility-regime labels
- drawdown-state flags
- trend regime features

---

## 18.9 `data/splits.py`
### Required fix
- chronological splitting only
- saved split boundaries
- optional walk-forward support

---

## 18.10 `preprocessing/scalers.py`
### Required fix
- train-only fit
- transform val/test
- serialise artefact
- feature-column consistency checks

---

## 18.11 `datasets/sequence_dataset.py`
### Required fix
- arbitrary feature matrix
- target asset selection
- horizon handling
- same contract for all core models

---

## 18.12 `models/*`
### Required fix
- dynamic input dimensions
- specialised heads/losses where appropriate
- updated PINN residual handling for richer state inputs

---

## 18.13 `training/train.py`
### Required fix
- config-driven runs
- dataset version logging
- feature config logging
- benchmark vs specialised mode separation

---

## 18.14 `evaluation/metrics.py`
### Required fix
- explicit target-type awareness
- no scaled/unscaled confusion
- separate forecasting vs financial metrics

---

## 18.15 `evaluation/plots.py`
### Required addition
- cumulative return plots
- prediction vs actual
- residual diagnostics
- regime-sliced performance
- correlation heatmaps
- feature coverage plots

---

# 19. Experiment design audit

## 19.1 Required experiment structure

### Experiment 1 — core fair benchmark
Train:
- LSTM
- GRU
- BiLSTM
- Attention-LSTM
- Transformer
- baseline PINN
- GBM PINN
- OU PINN
- Black-Scholes PINN
- global PINN

All on:
- the same date ranges
- the same feature set
- the same target: SPY next-day adjusted log return
- the same preprocessing
- the same evaluation setup

### Experiment 2 — volatility extension
Train:
- vol_lstm
- vol_gru
- vol_transformer
- vol_pinn
- heston_pinn
- stacked_vol_pinn

On:
- realised-volatility or joint return-volatility target
- volatility-specific features

### Experiment 3 — advanced structured PINN extension
Train:
- residual PINN
- stacked PINN
- spectral PINN
- financial DP-PINN
- adaptive dual-phase PINN

Use:
- richer market-state context
- regime-sensitive framing
- more advanced training design

### Experiment 4 — ablations
Compare:
- single-asset vs multi-asset
- raw close vs adjusted close
- no VIX vs VIX
- no rates vs rates
- leakage-safe scaling vs improper full-data scaling
- with QA filters vs without QA filters

## 19.2 Audit action

**Mandatory dissertation structuring recommendation**.

---

# 20. Critical risk register if changes are not made

## 20.1 Risk: leakage
If preprocessing is fitted on full data, benchmark results are invalid.

## 20.2 Risk: distorted targets
If raw close is used inconsistently, returns and volatility measures may be wrong.

## 20.3 Risk: unfair comparisons
If some models see richer features or different tasks without disclosure, comparisons are not credible.

## 20.4 Risk: hidden data defects
Without QA, missing data or corporate actions may silently drive performance.

## 20.5 Risk: overfitting
Wider feature spaces without regularisation and audit controls will inflate apparent in-sample success.

## 20.6 Risk: dissertation weakness
Without reproducibility and clear benchmark separation, results will be harder to defend.

---

# 21. Priority order for implementation

## Phase 1 — mandatory foundations
1. Multi-asset downloader
2. Local cache
3. QA report generation
4. Calendar alignment
5. Adjusted-price target standardisation
6. Train-only preprocessing

## Phase 2 — benchmark dataset redesign
7. Modular feature pipeline
8. Sequence builder redesign
9. Standard benchmark config
10. Evaluation refactor

## Phase 3 — model alignment
11. Dynamic input-dimension support across baseline models
12. PINN residual redesign for richer context
13. Volatility models moved to separate task group

## Phase 4 — advanced research extensions
14. Advanced PINN variants
15. Regime-conditioned losses
16. Ablation suite
17. Walk-forward robustness testing

---

# 22. Final audit conclusion

The project requires a substantial but highly beneficial redesign. The necessary changes are not isolated bug fixes; they are a full methodological upgrade from a **single-index forecasting script** into a **research-grade market-state forecasting system**. The most important mandatory fixes are:

- stop relying on a single-series assumption
- use adjusted prices consistently
- add formal QA checks
- align all data to a master calendar
- prevent leakage by fitting preprocessing on train only
- define one shared benchmark task for fair comparisons
- separate volatility models and advanced PINN variants into their own extension experiments
- cache and version datasets locally using Parquet and JSON metadata
- refactor models so their inputs, losses, and targets remain appropriate under the richer pipeline

The cleanest benchmark remains:

- **Target**: SPY next-day adjusted log return
- **Inputs**: shared multi-asset base feature set
- **Core benchmark models**: recurrent baselines, attention/transformer baselines, and core PINN variants
- **Separate extensions**: volatility-focused models and advanced structured PINNs
- **Ablations**: multi-asset value, adjusted-price value, regime feature value, leakage-prevention value, and QA-filtering value

This is the version of the pipeline most likely to produce defendable dissertation results.
