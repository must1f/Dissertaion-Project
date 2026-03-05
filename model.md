# Model Architecture & Training Documentation (PINN Financial Forecasting)
# Author: Codex | Date: 04 March 2026 | Language: British English

----------------------------------------------------------------------
1) Change Summary
----------------------------------------------------------------------
- Clarified targets, feature scaling, shapes, causality, and λ precedence with explicit file-line citations.
- Added rigorous ASCII and Mermaid diagrams for every model (baseline, Transformer, PINN, Stacked, Residual, Volatility) with dual loss paths and masking status.
- Documented physics-loss equations, scale/units, and BS autograd path; flagged mixed-scale risks with fixes.
- Introduced leakage & causality safeguards, tensor/scale reference table, and a focused ablation plan.
- Marked unverifiable items as "Needs Verification" with file pointers.
- **[2026-03-04] Added Dual-Phase PINN (DP-PINN) for Burgers' equation**: BurgersPINN, DualPhasePINN with phase-splitting for stiff PDEs, Latin Hypercube Sampling, Adam+L-BFGS training, L2 error evaluation.

----------------------------------------------------------------------
2) Fully Revised Documentation
----------------------------------------------------------------------

## 2.1 Forecasting Task
| Item | Value | Source |
|------|-------|--------|
| Target | Next‑day **standardised close (z‑score)** | `src/data/preprocessor.py:409-449`, `scripts/train_models.py:153-206` |
| Horizon | 1 step (h=1) | `src/data/preprocessor.py:455-487` |
| Output | ŷ ∈ ℝ^(B×1) | models |
| Interpretation | z‑score: (close − μ_ticker)/σ_ticker; inverse StandardScaler required for price-level metrics | same as above |

### Features (research training, D≈14; filtered to present columns)
close, volume, log_return, simple_return, rolling_volatility_5, rolling_volatility_20, momentum_5, momentum_20, rsi_14, macd, macd_signal, bollinger_upper, bollinger_lower, atr_14.  
Normalisation: **StandardScaler per ticker**, fit on train split; applied to val/test; target shares the close scaler (`scripts/train_models.py:184-205`).

### Input / Output Contract
```
x ∈ ℝ^(B×T×D)  with T=sequence_length (research: 180; config default: 60)
ŷ ∈ ℝ^(B×1)   predicted standardised close at t+1
y ∈ ℝ^(B×1)   actual standardised close at t+1
```

## 2.2 Baseline Models
Hyperparameters shown = research mode (hidden_dim=512, num_layers=4, dropout=0.15). Class defaults are lighter (128/2/0.2).

### LSTM — `src/models/baseline.py:14-132`
- LSTM(D→512, layers=4, dropout=0.15, uni) → last h_T → Linear 512→512→ReLU→Dropout→Linear 512→1.
- Causality: causal. Init: Xavier (input), Orthogonal (hidden).

### GRU — `src/models/baseline.py:134-236`
- GRU(D→512, layers=4, dropout=0.15, uni) → last h_T → same head as LSTM.
- Causality: causal. Two-gate cell (~25% fewer params than LSTM).

### BiLSTM — `src/models/baseline.py:239-260`
- Bidirectional; concat h_T^f, h_T^b → 1024 → Linear 1024→512→ReLU→Dropout→Linear 512→1.
- Causality: **non‑causal within window** (future context used).

### Attention LSTM — `src/models/baseline.py:263-349`
- LSTM returns H ∈ ℝ^(B×T×512); attention scores via Linear→Tanh→Linear → softmax_t; context = Σ α_t h_t → Linear 512→512→ReLU→Dropout→Linear 512→1.
- Causality: causal (attention over past window only).

### Transformer — `src/models/transformer.py:57-168`
- Embed: Linear(D→d_model)×√d_model; sinusoidal PosEnc; dropout.
- Encoder × num_layers: MHA (nhead=4) + Residual + LN; FFN d_model→512→d_model + Residual + LN.
- Pool: last token → Linear d_model→(dim_ff/2)→ReLU→Dropout→Linear→1.
- Causality: **non‑causal by default** (no mask). To enforce causality, pass `generate_square_subsequent_mask(...)`.

## 2.3 PINN (price) — `src/models/pinn.py`
- Base model: LSTM / GRU / Transformer (as above).
- Loss: L_total = L_data + L_physics + L_BS (optional). dt = 1/252 years (`src/constants.py:24-36`).
- λ precedence (highest last): class defaults λ_gbm=λ_ou=λ_bs=λ_lang=0.1 → registry variant override (`src/models/model_registry.py:109-162`) → checkpoint-stored values.

### Physics losses (implemented)
1) GBM drift (diffusion ignored): residual = (S_next−S)/dt − μ·S; S=prices[:, :-1], μ=mean(returns) (`src/models/pinn.py:361-376`). **Note:** μ is estimated in log-return space but applied to price drift (units mismatch).  
2) OU drift (diffusion ignored): residual = (X_next−X)/dt − θ(μ−X); θ=softplus(θ_raw) learnable; L_OU = mean(residual²) (`src/models/pinn.py:378-397`).  
3) Langevin drift: residual = (X_next−X)/dt + γ·(−returns); γ=softplus(γ_raw); T learned but unused; L_Langevin = mean(residual²) (`src/models/pinn.py:399-421`).  
4) Black‑Scholes autograd (steady-state, ∂V/∂t omitted): bs_residual = 0.5σ²S²d²V/dS² + rS dV/dS − rV computed via a **second forward on x_grad (requires_grad)** with autograd (`src/models/pinn.py:181-266`, `547-590`). V is the price forecaster, not an option price; scales are mixed (normalised V,S with raw σ).

### Data & physics application — `src/training/trainer.py:243-276`
- L_data on ŷ,y.  
- Physics: prices (B,T, raw USD), returns (B,T, raw log), volatilities (B,T, raw) into GBM/OU/Langevin.  
- BS autograd uses normalised inputs (B,T,D) plus σ from volatilities.

## 2.4 Advanced PINNs — `src/models/stacked_pinn.py`
### StackedPINN
- PhysicsEncoder: 4× Linear→LN→GELU→Dropout (to 512) + projection.  
- Parallel heads: LSTM(512, L4) and GRU(512, L4); concat [h_lstm; h_gru] ∈ ℝ^(B×1024).  
- “Attention weights” (Linear→Tanh→Linear→Softmax) are logged only; fusion stays concatenation.  
- Head: shared MLP 1024→256→128 (LN+GELU+Dropout), regression 128→1, direction 128→2.  
- Physics: λ_gbm=λ_ou=0.1 on **returns only**, θ fixed at 1.0; no volatility input.

### ResidualPINN
- Base RNN (LSTM/GRU) last h[512]; base_pred=Linear 512→1.  
- Correction path: [h;base_pred] → 4× (Linear→256 + LN + Tanh + Dropout) → correction=Linear 256→1.  
- Final ŷ = base_pred + correction (correction bounded in z‑score units).  
- Direction head: Linear 256→32→2.  
- Physics: λ_gbm=λ_ou=0.1 on returns only, θ fixed at 1.0.

## 2.5 Volatility Models — `src/models/volatility.py`
### Baselines
- VolatilityLSTM/GRU: backbone (hidden_dim=128, layers=2, dropout=0.2) → last h → MLP → Softplus → variance forecast.  
- VolatilityTransformer: causal mask applied; encoder (nhead=4, layers=2, dim_ff=512) → last token → MLP → Softplus.

### Physics-informed
- VolatilityPINN: variance head (Softplus) + physics loss = OU mean-reversion + GARCH(1,1) consistency + Feller positivity + Leverage; learnable θ, ω, α, β (constrained).  
- HestonPINN: learnable κ, θ, ξ, ρ with Heston drift + Feller + Leverage penalties.  
- StackedVolatilityPINN: encoder+RNN with same OU/GARCH/Feller/Leverage set.  
- **Needs Verification** — dataset construction for variance targets, variance_history, returns alignment in dataloader (`src/data/dataset.py`).

## 2.6 Normalisation & Units
| Item | Scale | Source |
|------|-------|--------|
| Inputs, target (price) | StandardScaler per ticker (train-fit, reused) | `scripts/train_models.py:184-205` |
| metadata.prices | Raw USD | `scripts/train_models.py:206-232` |
| metadata.returns | Raw log returns | same |
| metadata.volatilities | Raw rolling std | same |
| BS autograd | V on normalised scale; S from normalised inputs; σ from raw vols → **mixed scale** |
| Volatility outputs | Softplus, variance units (target scale needs verification) |

## 2.7 Reproducibility & Anti-Leakage
- Temporal splits (train/val/test chronological) — `src/data/preprocessor.py:496-520`.  
- Scalers fit on train split only per ticker — `scripts/train_models.py:184-205`.  
- Research config locked: epochs=100, batch=16, lr=5e‑4, weight_decay=1e‑4, grad_clip=1.0, dropout=0.15, hidden_dim=512, num_layers=4, seq_len=180, seed=42 — `src/utils/config.py:141-232`.  
- Early stopping disabled in research mode (`src/training/trainer.py:78-118`).

## 2.8 Trained Models and Recorded Metrics
Current artefacts in `Models/` and `results/*.json`. Financial metrics prior to the latest scaling fixes may be inflated; treat Sharpe/Sortino as provisional until reevaluated with de-standardised prices and lagged positions.

| Model | Epochs (train_loss length) | RMSE | R² | Dir. Acc (%) | Sharpe | Sortino | MaxDD (%) | Source |
|-------|---------------------------|------|----|--------------|--------|---------|-----------|--------|
| lstm | 100 | 6.48 | -0.18 | 51.20 | 5.00 | 10.00 | 19.28 | `results/lstm_results.json` |
| gru | 1 | 0.87 | -0.85 | 48.53 | 0.00 | 10.00 | 0.00 | `results/gru_results.json` |
| bilstm | 1 | 2.58 | -15.43 | 47.88 | 0.00 | 10.00 | 0.00 | `results/bilstm_results.json` |
| transformer | 1 | 2.40 | -13.22 | 46.91 | 0.00 | 10.00 | 0.00 | `results/transformer_results.json` |
| attention_lstm | — (no results file) | N/A | N/A | N/A | N/A | N/A | N/A | — |
| pinn_baseline | 1 | 2.72 | -17.35 | 51.14 | 0.00 | 10.00 | 0.00 | `results/pinn_baseline_results.json` |
| pinn_gbm | 2 | 2.43 | -13.58 | 51.14 | 0.00 | 10.00 | 0.00 | `results/pinn_gbm_results.json` |
| pinn_ou | 1 | 2.36 | -12.80 | 47.88 | 0.00 | 10.00 | 0.00 | `results/pinn_ou_results.json` |
| pinn_black_scholes | — (no results file) | N/A | N/A | N/A | N/A | N/A | N/A | — |
| pinn_gbm_ou | — (no results file) | N/A | N/A | N/A | N/A | N/A | N/A | — |
| pinn_global | 1 | 2.24 | -11.41 | 47.56 | 0.00 | 10.00 | 0.00 | `results/pinn_global_results.json` |
| stacked_pinn | 1 | N/A | N/A | N/A | N/A | N/A | N/A | `results/stacked_pinn_results.json` |
| residual_pinn | 1 | N/A | N/A | N/A | N/A | N/A | N/A | `results/residual_pinn_results.json` |
| volatility models | — | N/A | N/A | N/A | N/A | N/A | N/A | `Models/volatility/vol_model_best.pt` (no metrics JSON) |

Notes:
- RMSE/R²/Directional Accuracy are on the scale recorded in the results files; rerun evaluation with de-standardisation to obtain price-scale metrics.
- Financial metrics (Sharpe/Sortino/MaxDD) were computed on normalised prices and without the corrected lagged positions; they must be recomputed after the latest fixes.

----------------------------------------------------------------------
3) Deep, More Rigorous Diagrams
----------------------------------------------------------------------

### Legend for Mermaid (use in all diagrams)
```
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#f5f5f5","primaryTextColor":"#222","primaryBorderColor":"#333","lineColor":"#555","secondaryColor":"#e8f0ff","tertiaryColor":"#fff8e6","fontSize":"13px"}}}%%
classDef input fill:#e8f0ff,stroke:#333,stroke-width:1px;
classDef model fill:#f5f5f5,stroke:#333,stroke-width:1px;
classDef loss fill:#fff8e6,stroke:#333,stroke-width:1px;
classDef physics fill:#ffe6e6,stroke:#aa0000,stroke-width:1px;
classDef output fill:#e6ffe6,stroke:#006600,stroke-width:1px;
classDef metadata fill:#f0e6ff,stroke:#550088,stroke-width:1px;
```

Below: ASCII (compact) + Mermaid for each model. Shapes assume research mode (T=180, D≈14, H=512 unless stated).

### 3.1 LSTM (ASCII, high level)
```
x[B,T,D] → LSTM(D→512,L4) → h_T[512] → Linear→ReLU→Dropout→Linear → ŷ[B,1]
```

### 3.1 LSTM (Mermaid, deep)
```
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#f5f5f5","primaryTextColor":"#222","primaryBorderColor":"#333","lineColor":"#555","secondaryColor":"#e8f0ff","tertiaryColor":"#fff8e6","fontSize":"13px"}}}%%
flowchart TD
  X[x ∈ ℝ^(B×T×D)]:::input
  L[LSTM D→512, layers=4, dropout=0.15<br/>output H ∈ ℝ^(B×T×512)]:::model
  P[Pool last h_T ∈ ℝ^(B×512)]:::model
  H1[Linear 512→512 + ReLU + Dropout]:::model
  H2[Linear 512→1]:::model
  Y[ŷ ∈ ℝ^(B×1)]:::output
  X --> L --> P --> H1 --> H2 --> Y
```

### 3.2 GRU (ASCII)
```
x → GRU(D→512,L4) → h_T[512] → Linear→ReLU→Dropout→Linear → ŷ
```

### 3.2 GRU (Mermaid)
```
%%{init: {...}}%%
flowchart TD
  X[x ∈ ℝ^(B×T×D)]:::input
  G[GRU D→512, layers=4, dropout=0.15<br/>H ∈ ℝ^(B×T×512)]:::model
  P[Pool last h_T ∈ ℝ^(B×512)]:::model
  H1[Linear 512→512 + ReLU + Dropout]:::model
  H2[Linear 512→1]:::model
  Y[ŷ ∈ ℝ^(B×1)]:::output
  X --> G --> P --> H1 --> H2 --> Y
```

### 3.3 BiLSTM (ASCII)
```
x → BiLSTM → [h_T^f; h_T^b][1024] → Linear→ReLU→Dropout→Linear → ŷ   (non‑causal)
```

### 3.4 Attention LSTM (ASCII)
```
x → LSTM → H[B,T,512] → attention(Linear→Tanh→Linear→softmax_t) → context[512] → Linear→ReLU→Dropout→Linear → ŷ
```

### 3.5 Transformer (Mermaid, deep, mask off by default)
```
%%{init: {...}}%%
flowchart TD
  X[x ∈ ℝ^(B×T×D)]:::input
  E[Embed Linear D→d_model ×√d_model + PosEnc + Dropout]:::model
  subgraph ENC[Encoder × L]
    MHA[MHA nhead=4]:::model
    LN1[Residual + LayerNorm]:::model
    FFN[Linear d_model→512→d_model + ReLU]:::model
    LN2[Residual + LayerNorm]:::model
  end
  P[Take last token x[:,T-1,:] ∈ ℝ^(B×d_model)]:::model
  H1[Linear d_model→256 + ReLU + Dropout]:::model
  H2[Linear 256→1]:::model
  Y[ŷ ∈ ℝ^(B×1)]:::output
  X --> E --> ENC --> P --> H1 --> H2 --> Y
  MHA -. "no mask by default (non‑causal)" .-> LN1
```

### 4) PINN (Mermaid, deep)
```
%%{init: {...}}%%
flowchart TD
  subgraph INPUT
    X[x ∈ ℝ^(B×T×D)]:::input
    PZ[metadata.prices ∈ ℝ^(B×T)]:::metadata
    RT[metadata.returns ∈ ℝ^(B×T)]:::metadata
    VOL[metadata.volatilities ∈ ℝ^(B×T)]:::metadata
  end

  subgraph MODEL
    M1[Base: LSTM/GRU/Transformer]:::model
    M2[Pooling / last token]:::model
    M3[Head → ŷ ∈ ℝ^(B×1)]:::output
  end

  subgraph DATA_LOSS
    Ld[MSE(ŷ, y)]:::loss
  end

  subgraph PHYSICS
    Lg[GBM drift: (S_{t+1}-S_t)/dt - μS]:::physics
    Lo[OU drift: (X_{t+1}-X_t)/dt - θ(μ-X)]:::physics
    Ll[Langevin: (X_{t+1}-X_t)/dt + γ·(-X)]:::physics
  end

  subgraph BS_AUTOGRAD
    Xg[x cloned, requires_grad=True]:::model
    V[Second forward → V (ℝ^(B×1))]:::model
    d1[dV/dS, d²V/dS² via autograd]:::model
    Lbs[BS residual: 0.5·σ²·S²·d²V + rS·dV - rV]:::physics
  end

  X --> M1 --> M2 --> M3 --> Ld
  PZ --> Lg
  RT --> Lo
  RT --> Ll
  VOL --> Lg
  VOL --> Lbs
  X --> Xg --> V --> d1 --> Lbs

  Lg --> Ltot[Total loss]:::loss
  Lo --> Ltot
  Ll --> Ltot
  Lbs --> Ltot
  Ld --> Ltot
```

### 5) StackedPINN (ASCII)
```
x → PhysicsEncoder (4× Linear→LN→GELU→Dropout, proj) → encoded[B,T,512]
    → LSTM head h_lstm[512]
    → GRU head h_gru[512]
    concat[1024] (+ logged α) → MLP 1024→256→128 → {reg 128→1, dir 128→2}; physics on returns (λ_gbm, λ_ou, θ fixed)
```

### 6) ResidualPINN (ASCII)
```
x → Base RNN → h[512], base_pred[1]
[h;base_pred] → 4×(Linear→LN→Tanh→Dropout) → correction[1]
ŷ = base_pred + correction (z-score units); dir head 256→32→2; physics on returns
```

### 7) VolatilityPINN (ASCII)
```
x → LSTM/GRU (128, L2) → h → variance head (Softplus) → σ̂²
Loss = MSE + λ_ou·OU + λ_garch·GARCH + λ_feller·pos + λ_leverage·corr
```

----------------------------------------------------------------------
4) Tensor & Scale Reference Table
----------------------------------------------------------------------
| Tensor | Shape | Scale | Used By | Notes |
|--------|-------|-------|--------|-------|
| x | (B,T,D) | z‑score | all models | StandardScaler per ticker |
| y | (B,1) | z‑score | data loss | same scaler as close |
| ŷ | (B,1) | z‑score | losses, metrics | de-standardise before price metrics |
| metadata.prices | (B,T) | raw USD | GBM | Diffusion ignored |
| metadata.returns | (B,T) | raw log | GBM/OU/Langevin | μ from returns applied to price drift |
| metadata.volatilities | (B,T) | raw | GBM, BS σ | Units mismatch with normalised ŷ |
| metadata.inputs | (B,T,D) | z‑score | BS autograd | Provides S (normalised) for BS |
| σ (BS) | (B,1) or scalar | raw vol | BS residual | Mixed with normalised S, V |
| dt | scalar | 1/252 years | all physics | May scale residuals up |

Mixed-scale issue: BS residual combines normalised V & S with raw σ.  
Fix options: (1) denormalise V and S using scaler; (2) normalise σ to same scale; (3) apply unit scaling factor.  
Pseudocode (option 1):
```
S_denorm = S_norm * close_std + close_mean
V_denorm = V_norm * close_std + close_mean
bs_residual = 0.5*σ**2*S_denorm**2*d2V_dS2 + r*S_denorm*dV_dS - r*V_denorm
```
Log: mean|residual|, scale factors, λ_bs contribution each batch.

----------------------------------------------------------------------
Dimensional Consistency & Residual Scaling
----------------------------------------------------------------------
- dt = 1/252 shrinks the time step; dividing finite differences by dt can inflate residual magnitudes by ≈252×, making λ tuning sensitive.  
- BS residual presently mixes normalised V,S with raw σ and r, so terms have inconsistent units; GBM/OU/Langevin ignore diffusion, further biasing magnitudes.
- Consistent alternatives:  
  1) **Denormalise V and S** before BS: S = S_norm·σ_close + μ_close; V = V_norm·σ_close + μ_close; compute residual in price units (recommended for unit homogeneity).  
  2) Normalise σ and r to the feature scale used for V,S (less interpretable financially).  
  3) Scale each residual by an empirical std (per-batch or running) to balance gradients when dt is small.
- Recommended: option 1 for BS (preserves financial units), plus residual standardisation for GBM/OU/Langevin to offset dt inflation; log residual RMS and λ-weighted contributions to monitor balance.

----------------------------------------------------------------------
5) Leakage & Verification Section
----------------------------------------------------------------------
Risks: bidirectional window look‑ahead (BiLSTM), unmasked Transformer, scaler fit leakage, windows crossing splits, look‑ahead features.  
Current safeguards: temporal split (`src/data/preprocessor.py:496-520`), train‑only scaler fit (`scripts/train_models.py:184-205`); masked attention only in volatility Transformer; BiLSTM allowed but marked non‑causal; Transformer unmasked by default.  
Checklist:
- Assert sequence windows do not straddle split boundaries.
- Assert scaler fitted on train only; val/test transformed with frozen params.
- If causal config enabled, assert mask passed to Transformer forward.
- Assert inverse-transform of ŷ before financial metrics (`src/evaluation/metrics.py` — Needs Verification).
Optional pseudocode assertions:
```
# windows stay within split
assert (start_indices + T <= split_points['train_end']).all()
# scaler
assert scaler_state_frozen and not scaler_fitted_on_valtest
# transformer mask
mask = generate_square_subsequent_mask(T)
assert torch.all(torch.isneginf(mask.triu(1)))
# inverse transform before metrics
y_hat_price = y_hat * close_std + close_mean
assert metrics_use_price_scale(y_hat_price)
```

----------------------------------------------------------------------
6) Ablation Plan
----------------------------------------------------------------------
| Experiment | Hypothesis | Metrics | Failure signals |
|------------|------------|---------|-----------------|
| Masked vs unmasked Transformer | Masking improves directional accuracy by removing leakage | val MSE, directional acc | Divergence, longer train time without gain |
| BS scaling variants | Unit-consistent BS residual stabilises training | loss balance, val MSE, BS residual std | Exploding/vanishing BS loss |
| λ warm‑up vs constant | Warm‑up prevents physics dominating early | training curves, final val metrics | Underfitting if λ stays low |
| Langevin diffusion term vs none | Including √(2γT) term improves volatility regimes | val MSE, residual stats | Instability or no benefit |
| Attention fusion vs concat in StackedPINN | Learned fusion reduces params, may improve generalisation | val MSE, attention entropy | Degraded accuracy or collapsed weights |

----------------------------------------------------------------------
7) Dual-Phase PINN for Burgers' Equation — `src/models/dp_pinn.py`
----------------------------------------------------------------------

### 7.1 Overview
Physics-Informed Neural Networks for solving the viscous Burgers' equation:
```
u_t + u · u_x = ν · u_xx   on x ∈ [-1,1], t ∈ [0,1]
```
With IC: u(x,0) = -sin(πx) and BC: u(-1,t) = u(1,t) = 0.

This is a stiff PDE benchmark where the solution develops steep gradients (shocks) that are challenging for standard PINNs.

### 7.2 BurgersPINN — `src/models/dp_pinn.py:40-260`
| Item | Value | Source |
|------|-------|--------|
| Architecture | 8 FC layers × 50 neurons, tanh activation | `dp_pinn.py:66-96` |
| Input | (x, t) ∈ ℝ² | `forward()` |
| Output | û(x,t) ∈ ℝ | single prediction |
| Viscosity | ν = 0.01/π ≈ 0.00318 | `dp_pinn.py:67` |
| Loss weights | λ_pde=1.0, λ_ic=100.0, λ_bc=100.0 | `dp_pinn.py:68-70` |

**Loss Function:**
```
L = λ_pde · ||u_t + u·u_x - ν·u_xx||² + λ_ic · ||u(x,0) + sin(πx)||² + λ_bc · (||u(-1,t)||² + ||u(1,t)||²)
```

**Autograd Derivatives** — `dp_pinn.py:132-175`:
- u_t via `torch.autograd.grad(u, t, create_graph=True)`
- u_x via `torch.autograd.grad(u, x, create_graph=True)`
- u_xx via `torch.autograd.grad(u_x, x, create_graph=True)`

### 7.3 DualPhasePINN — `src/models/dp_pinn.py:280-510`
Splits time domain into two phases to better handle stiff dynamics:

| Phase | Domain | Constraint | Network |
|-------|--------|------------|---------|
| Phase 1 | t ∈ [0, 0.4] | IC: u(x,0) = -sin(πx) | phase1_net |
| Phase 2 | t ∈ [0.4, 1] | Intermediate: u1(x, t_s) = u2(x, t_s) | phase2_net |

**Architecture:**
```
phase1_net: BurgersPINN (8 layers × 50, tanh) → handles t ≤ 0.4
phase2_net: BurgersPINN (8 layers × 50, tanh) → handles t > 0.4

Intermediate constraint enforces continuity at t_switch:
L_intermediate = ||u1(x, 0.4) - u2(x, 0.4)||²
```

**Training Protocol:**
1. Phase 1: Train phase1_net with IC constraint (Adam 50k iter + L-BFGS 10k iter)
2. Phase 2: Freeze phase1_net, train phase2_net with intermediate constraint

### 7.4 BurgersPINN (ASCII, deep)
```
(x,t) → Linear(2→50) → tanh → [Linear(50→50) → tanh] ×7 → Linear(50→1) → û

         ┌─────────────────────────────────────────────────────────┐
         │              Autograd Derivative Computation             │
         │  u_t = ∂u/∂t via autograd(u, t, create_graph=True)      │
         │  u_x = ∂u/∂x via autograd(u, x, create_graph=True)      │
         │  u_xx = ∂²u/∂x² via autograd(u_x, x, create_graph=True) │
         └─────────────────────────────────────────────────────────┘
                                    ↓
         ┌─────────────────────────────────────────────────────────┐
         │              PDE Residual: u_t + u·u_x - ν·u_xx         │
         │              L_pde = mean(residual²)                     │
         └─────────────────────────────────────────────────────────┘
```

### 7.5 DualPhasePINN (Mermaid)
```
%%{init: {...}}%%
flowchart TD
  subgraph INPUT
    XT["(x, t) ∈ ℝ²"]:::input
  end

  subgraph PHASE1["Phase 1: t ∈ [0, 0.4]"]
    P1[phase1_net: BurgersPINN]:::model
    IC[IC Loss: u(x,0) = -sin(πx)]:::loss
    BC1[BC Loss: u(±1,t) = 0]:::loss
    PDE1[PDE Residual]:::physics
  end

  subgraph PHASE2["Phase 2: t ∈ [0.4, 1]"]
    P2[phase2_net: BurgersPINN]:::model
    INT[Intermediate: u1(x,0.4) = u2(x,0.4)]:::loss
    BC2[BC Loss: u(±1,t) = 0]:::loss
    PDE2[PDE Residual]:::physics
  end

  XT --> P1
  XT --> P2
  P1 --> IC
  P1 --> BC1
  P1 --> PDE1
  P2 --> INT
  P2 --> BC2
  P2 --> PDE2
  P1 -. "frozen in Phase 2" .-> INT
```

### 7.6 Loss Functions — `src/losses/burgers_equation.py`
| Class | Purpose | Equation |
|-------|---------|----------|
| BurgersResidual | PDE residual via autograd | f = u_t + u·u_x - ν·u_xx |
| BurgersICLoss | Initial condition | ||u(x,0) + sin(πx)||² |
| BurgersBCLoss | Boundary conditions | ||u(-1,t)||² + ||u(1,t)||² |
| BurgersIntermediateLoss | Phase continuity | ||u1(x, t_s) - u2(x, t_s)||² |

### 7.7 Latin Hypercube Sampling — `src/utils/sampling.py`
Better space-filling than random sampling for collocation points:
```python
samples = latin_hypercube_sampling(
    n_samples=20000,
    bounds=[(-1.0, 1.0), (0.0, 1.0)],  # x, t ranges
    seed=42
)
```

**Data Generation:**
```
generate_burgers_training_data():
  - n_collocation: 20000 (LHS over full domain)
  - n_boundary: 2000 (BC points at x = ±1)
  - n_initial: 2000 (IC points at t = 0)
  - n_intermediate: 1000 (continuity points at t = t_switch)
```

### 7.8 Training — `src/training/dp_pinn_trainer.py`
| Stage | Optimizer | Iterations | Learning Rate |
|-------|-----------|------------|---------------|
| Adam | torch.optim.Adam | 50,000 | 1e-3 |
| L-BFGS | torch.optim.LBFGS | 10,000 | 1.0 |

**DPPINNTrainer Interface:**
```python
trainer = DPPINNTrainer(model, TrainingConfig(seed=42))
history = trainer.train_standard_pinn(data)  # Standard PINN
# or
history1 = trainer.train_phase1(data)  # Phase 1
history2 = trainer.train_phase2(data)  # Phase 2 (phase1 frozen)
```

### 7.9 Evaluation — `src/evaluation/pde_evaluator.py`
| Metric | Formula | Notes |
|--------|---------|-------|
| Relative L2 | ||û - u||₂ / ||u||₂ | Over full domain |
| Time-resolved L2 | L2 error at each t slice | Shows error evolution |
| Max error | max|û - u| | Point-wise maximum |

**Exact Solution:** Hopf-Cole transformation (`burgers_exact_solution_hopf_cole`)

### 7.10 Expected Results
| Model | Relative L2 Error | Notes |
|-------|------------------|-------|
| Standard PINN | ~1e-2 to 1e-1 | Error accumulates with time |
| DP-PINN | ~1e-3 to 1e-2 | Phase splitting reduces propagation |

### 7.11 Files Reference
| Component | File Path | Description |
|-----------|-----------|-------------|
| **BurgersPINN** | `src/models/dp_pinn.py:40-260` | Standard PINN for Burgers' |
| **DualPhasePINN** | `src/models/dp_pinn.py:280-510` | Two-phase architecture |
| **Burgers Losses** | `src/losses/burgers_equation.py` | PDE, IC, BC, intermediate |
| **LHS Sampling** | `src/utils/sampling.py` | Latin Hypercube Sampling |
| **Trainer** | `src/training/dp_pinn_trainer.py` | Adam + L-BFGS training |
| **Evaluator** | `src/evaluation/pde_evaluator.py` | L2 error metrics |
| **Visualization** | `src/reporting/pde_visualization.py` | 3D plots, heatmaps |
| **Config** | `configs/dp_pinn_config.yaml` | Full configuration |
| **Experiment** | `scripts/run_dp_pinn_experiment.py` | Main runner |

----------------------------------------------------------------------
8) Summary: All Model Architectures
----------------------------------------------------------------------

| Model | Type | File | Key Features |
|-------|------|------|--------------|
| LSTM | Baseline | `baseline.py:14-132` | 4-layer LSTM, causal |
| GRU | Baseline | `baseline.py:134-236` | 4-layer GRU, fewer params |
| BiLSTM | Baseline | `baseline.py:239-260` | Bidirectional, non-causal |
| Attention LSTM | Baseline | `baseline.py:263-349` | Attention over window |
| Transformer | Baseline | `transformer.py:57-168` | Encoder-only, no mask default |
| PINN | Physics | `pinn.py` | GBM/OU/BS/Langevin losses |
| StackedPINN | Advanced | `stacked_pinn.py` | Encoder + parallel LSTM/GRU |
| ResidualPINN | Advanced | `stacked_pinn.py` | Base + correction path |
| VolatilityPINN | Volatility | `volatility.py` | Variance forecast + OU/GARCH |
| HestonPINN | Volatility | `volatility.py` | Heston SDE constraints |
| **BurgersPINN** | PDE | `dp_pinn.py` | Burgers' equation, autograd |
| **DualPhasePINN** | PDE | `dp_pinn.py` | Two-phase for stiff PDEs |

----------------------------------------------------------------------
End of Document
----------------------------------------------------------------------
