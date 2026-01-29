# Black-Scholes Integration Analysis

**Date**: 2026-01-29
**Decision**: Recommendation for dissertation

---

## Executive Summary

The Black-Scholes equation was implemented as one of the physics constraints in the PINN framework. This document analyzes its theoretical appropriateness, empirical performance, and provides a recommendation for the dissertation.

---

## 1. Theoretical Analysis

### 1.1 Black-Scholes Purpose

The Black-Scholes equation was originally designed for **option pricing**:

```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

Where:
- V = Option value
- S = Underlying stock price
- σ = Volatility
- r = Risk-free rate
- t = Time to expiration

### 1.2 Application to Stock Forecasting

In this project, the Black-Scholes PDE is applied as a constraint on stock price predictions. However, there are fundamental theoretical concerns:

**Concerns**:
1. **Mismatch of purpose**: Black-Scholes models the relationship between option prices and stock prices, not stock price dynamics itself
2. **No-arbitrage assumption**: The equation enforces no-arbitrage in option markets, which doesn't directly constrain stock price movements
3. **Interpretation challenge**: What does it mean to enforce the Black-Scholes PDE on a stock price prediction?

**Potential justification**:
- Could be interpreted as enforcing consistency with derivative pricing theory
- The PDE encodes market efficiency assumptions
- The second derivative term could regularize price predictions

### 1.3 Comparison to Other Physics Equations

More appropriate equations for stock price dynamics:

| Equation | Purpose | Appropriateness for Stocks |
|----------|---------|---------------------------|
| **GBM** | Models stock price evolution | ✅ High - Industry standard |
| **Ornstein-Uhlenbeck** | Mean-reverting processes | ✅ High - Good for stationary stocks |
| **Langevin** | Momentum with friction | ⚠️ Medium - Novel application |
| **Black-Scholes** | Option pricing | ❌ Low - Designed for derivatives |

---

## 2. Empirical Performance Analysis

### 2.1 Evaluation Results

Based on rigorous evaluation results:

| Model | RMSE | R² | Sharpe | Directional Accuracy |
|-------|------|-------|---------|---------------------|
| **PINN Global** | **0.856** | **0.868** | 0.406 | 0.510 |
| **PINN OU** | 0.979 | 0.827 | **0.424** | 0.518 |
| **PINN GBM+OU** | 0.931 | 0.844 | 0.410 | 0.512 |
| **PINN Black-Scholes** | 1.053 | 0.800 | 0.425 | **0.522** |
| **PINN GBM** | 1.075 | 0.792 | 0.393 | 0.512 |
| **PINN Baseline** | 1.022 | 0.812 | 0.435 | 0.513 |

### 2.2 Performance Ranking

**By RMSE (lower is better)**:
1. PINN Global (0.856) ✅
2. PINN GBM+OU (0.931)
3. PINN OU (0.979)
4. PINN Baseline (1.022)
5. **PINN Black-Scholes (1.053)** ← 5th place
6. PINN GBM (1.075)

**By R² (higher is better)**:
1. PINN Global (0.868) ✅
2. PINN GBM+OU (0.844)
3. PINN OU (0.827)
4. PINN Baseline (0.812)
5. **PINN Black-Scholes (0.800)** ← 5th place
6. PINN GBM (0.792)

### 2.3 Key Findings

1. **Black-Scholes performs worse than Global, OU, and GBM+OU variants**
2. **Black-Scholes has highest directional accuracy (0.522)** - interesting finding
3. **Black-Scholes has good Sharpe ratio (0.425)** but not the best
4. **Overall ranking**: Middle of the pack, not among the top performers

---

## 3. Implementation Quality

### 3.1 Current Implementation

The implementation uses **automatic differentiation** correctly:

```python
def black_scholes_autograd_residual(
    self,
    model: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    ...
) -> torch.Tensor:
    # Computes ∂V/∂S via torch.autograd.grad with create_graph=True
    # Computes ∂²V/∂S² via second-order autograd
    # Correctly implements the PDE residual
```

### 3.2 Implementation Status

✅ **Correctly implemented** using automatic differentiation
✅ **Trained successfully** with checkpoints saved
✅ **Evaluated rigorously** with comprehensive metrics
⚠️ **Theoretically questionable** for stock forecasting

---

## 4. Decision Matrix

### Option A: Keep Black-Scholes

**Pros**:
- Already implemented and working
- Shows competitive performance on some metrics (directional accuracy, Sharpe)
- Demonstrates breadth of physics constraints tested
- Novel application that could be justified

**Cons**:
- Theoretically weak foundation for stock forecasting
- Underperforms compared to Global, OU, GBM+OU
- Requires extensive justification in dissertation
- Could weaken overall thesis credibility

**What would be needed**:
1. Unit tests validating derivative computation
2. Extensive theoretical justification section
3. Discussion of limitations
4. Comparison to analytical Black-Scholes solutions

### Option B: Remove Black-Scholes

**Pros**:
- Cleaner theoretical narrative
- Focus on more appropriate physics equations (GBM, OU, Langevin)
- Simplifies dissertation methodology
- Better academic rigor

**Cons**:
- Loses directional accuracy leader
- Wastes implementation and training effort
- Reduces number of PINN variants tested

**What would be needed**:
1. Remove from model registry
2. Remove from dashboards
3. Delete checkpoints
4. Add "Equations Considered and Rejected" section
5. Brief justification in dissertation

### Option C: Downgrade to "Auxiliary" or "Experimental"

**Pros**:
- Acknowledges the implementation work
- Shows thoroughness in exploration
- Can discuss in "Alternative Approaches" section

**Cons**:
- Still requires some justification
- Might confuse the narrative

---

## 5. Recommendation

### ⚠️ **RECOMMENDED: Option B - Remove Black-Scholes**

**Rationale**:

1. **Theoretical coherence**: Black-Scholes is designed for option pricing, not stock dynamics. Keeping it would require extensive justification that could distract from the main contributions.

2. **Performance**: While competitive on some metrics, it doesn't outperform the more theoretically sound variants (Global, OU, GBM+OU).

3. **Dissertation quality**: Academic rigor favors using equations with clear theoretical foundations. Reviewers would likely question the Black-Scholes application.

4. **Simplicity**: The dissertation narrative is stronger when focused on GBM, OU, and Langevin - all of which have clearer applications to stock dynamics.

5. **"Rejected approaches" section**: The implementation effort isn't wasted - it can be discussed in a methodology section showing thoroughness in equation selection.

### Alternative Recommendation

If you want to keep it: **Relegate to appendix** as an "experimental constraint" that was explored but not used in main results.

---

## 6. Implementation Actions

### If Removing (Recommended):

1. **Code cleanup**:
   - [ ] Remove `pinn_black_scholes` from model registry
   - [ ] Update dashboards to exclude Black-Scholes variant
   - [ ] Comment out (don't delete) the `black_scholes_autograd_residual()` implementation
   - [ ] Add docstring: "Historical: Tested but not used in final dissertation"

2. **File management**:
   - [ ] Move checkpoints to `Models/archive/black_scholes/`
   - [ ] Keep evaluation results for reference
   - [ ] Preserve implementation code (commented) for future reference

3. **Dissertation**:
   - [ ] Add subsection: "Physics Equations Considered and Rejected"
   - [ ] Brief explanation: "Black-Scholes PDE was explored but deemed inappropriate for stock price forecasting as it was designed for option pricing. More suitable equations (GBM, OU, Langevin) were selected."
   - [ ] Table comparing equation suitability
   - [ ] Reference in limitations/future work

### If Keeping:

1. **Validation**:
   - [ ] Create unit tests for derivative accuracy
   - [ ] Validate against analytical Black-Scholes solutions
   - [ ] Test on option pricing data to verify correctness

2. **Justification**:
   - [ ] Write extensive theoretical justification
   - [ ] Cite literature on Black-Scholes applications beyond options
   - [ ] Discuss no-arbitrage principle as regularization
   - [ ] Acknowledge limitations explicitly

3. **Documentation**:
   - [ ] Dedicate dissertation subsection to Black-Scholes rationale
   - [ ] Include caveats and assumptions
   - [ ] Explain adaptation from option pricing to stock forecasting

---

## 7. Final Decision

**DECISION**: Remove Black-Scholes from main dissertation results

**Reasoning**:
- Prioritize theoretical rigor
- Focus on equations with clear foundations
- Simplify dissertation narrative
- Preserve implementation as "rejected approach" for methodology completeness

**Next Steps**:
1. Archive Black-Scholes model files
2. Update code to exclude from main model list
3. Document decision in dissertation methodology
4. Move to next technical implementation tasks

---

**Approved by**: [To be reviewed by user]
**Status**: Recommendation pending user approval
