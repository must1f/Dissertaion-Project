# Systematic PINN Physics Configuration Comparison

## Overview

This system performs a comprehensive cross-comparison of six distinct physics-informed neural network configurations to evaluate which financial physics constraints improve forecasting performance. Each configuration embeds different mathematical assumptions about market dynamics into the neural network training process.

## Six PINN Variants

### 1. **Baseline (Data-only)**
- **Configuration**: All physics weights = 0
- **Loss**: `L = L_data (MSE only)`
- **Purpose**: Control group with no physics constraints
- **Use Case**: Pure machine learning benchmark

### 2. **Pure GBM (Geometric Brownian Motion)**
- **Configuration**: `λ_gbm = 0.1`, others = 0
- **Equation**: `dS = μS dt + σS dW`
- **Purpose**: Captures exponential trend dynamics
- **Use Case**: Bull/bear markets, growth stocks, momentum trading

### 3. **Pure OU (Ornstein-Uhlenbeck)**
- **Configuration**: `λ_ou = 0.1`, others = 0
- **Equation**: `dX = θ(μ - X)dt + σdW`
- **Purpose**: Models mean-reverting behavior
- **Use Case**: Range-bound markets, pairs trading, volatility forecasting

### 4. **Pure Black-Scholes**
- **Configuration**: `λ_bs = 0.1`, others = 0
- **Equation**: `∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0`
- **Purpose**: Enforces no-arbitrage PDE constraint
- **Use Case**: Option pricing, derivative valuation, efficient markets

### 5. **GBM+OU Hybrid**
- **Configuration**: `λ_gbm = 0.05`, `λ_ou = 0.05`
- **Purpose**: Balances trend-following and mean-reversion
- **Use Case**: Normal market conditions, general forecasting

### 6. **Global Constraint (All Equations)**
- **Configuration**: All physics weights > 0
- **Purpose**: Maximum physics regularization
- **Use Case**: Testing whether all constraints can coexist

## Key Metrics

### Violation Score
The theoretical violation score quantifies how well a model adheres to physics constraints:

$$\text{Violation Score} = \frac{\mathcal{L}_{physics}}{\mathcal{L}_{data} + \epsilon}$$

- **Lower score**: Model satisfies physics with minimal penalty
- **Higher score**: Model violates physics assumptions significantly
- **Zero score**: Baseline (no physics)

### Ranking Criteria
Models are ranked by:
1. **Primary**: Test MSE (empirical accuracy)
2. **Secondary**: Violation Score (theoretical consistency)
3. **Combined**: 0.7 × MSE_Rank + 0.3 × Violation_Rank

## How to Run

### Using run.sh (Recommended)

```bash
./run.sh
# Select option 10: Systematic PINN Physics Comparison
```

This will:
1. Prompt for number of epochs (default: 100)
2. Ask whether to train all variants or select specific ones
3. Train each variant sequentially
4. Generate comprehensive reports

### Using Python directly

```bash
# Train all variants
python3 -m src.training.train_pinn_variants --epochs 100

# Train specific variants
python3 -m src.training.train_pinn_variants --epochs 50 --variants baseline gbm ou

# Available variants: baseline, gbm, ou, black_scholes, gbm_ou, global
```

## Outputs

All results are saved to `results/pinn_comparison/`:

### 1. **README_theory.md**
- Financial justification for each physics constraint
- Mathematical formulations
- Market regime recommendations
- Empirical results for each variant
- Interpretation guide for convergence issues

### 2. **comparison_report.csv**
Ranked comparison table with columns:
- `Variant`: Configuration name
- `Test_MSE`: Mean squared error on test set
- `Test_MAE`: Mean absolute error
- `Test_RMSE`: Root mean squared error
- `Test_R2`: R-squared coefficient
- `Violation_Score`: Physics constraint violation metric
- `Lambda_*`: Physics weight configurations
- `MSE_Rank`: Ranking by empirical accuracy
- `Violation_Rank`: Ranking by theoretical consistency
- `Combined_Rank`: Overall ranking
- `Model_Path`: Location of saved checkpoint

### 3. **detailed_results.json**
Complete training history including:
- Epoch-by-epoch losses (data, physics, total)
- Validation metrics
- Test performance
- Configuration parameters
- Training timestamps

### 4. **Model Checkpoints**
Individual model files saved to `models/`:
- `pinn_baseline_best.pt` - Data-only model
- `pinn_gbm_best.pt` - Pure GBM model
- `pinn_ou_best.pt` - Pure OU model
- `pinn_black_scholes_best.pt` - Pure Black-Scholes model
- `pinn_gbm_ou_best.pt` - Hybrid model
- `pinn_global_best.pt` - All constraints model

Each checkpoint includes:
- Model state dict
- Optimizer state
- Training history
- Configuration

## Interpreting Results

### Case 1: Baseline ranks #1
**Interpretation**: Physics constraints are not helping
- Data doesn't follow theoretical assumptions
- Over-regularization occurring
- Market has non-traditional dynamics

**Action**: Use baseline for forecasting, investigate why physics doesn't help

### Case 2: GBM+OU Hybrid ranks #1
**Interpretation**: Markets exhibit balanced trend + reversion
- Best of both worlds
- Realistic model of market dynamics
- Good generalization expected

**Action**: Use hybrid model, consider it the default choice

### Case 3: Pure constraint (GBM or OU) ranks #1
**Interpretation**: Clear market regime dominance
- GBM winning → trending market
- OU winning → mean-reverting market

**Action**: Use the winning model, consider regime detection

### Case 4: Global Constraint fails to converge
**Interpretation**: Contradictory constraints
- GBM (trend) vs OU (reversion) conflict
- Black-Scholes assumptions violated
- Over-constrained optimization

**Explanation**:
1. **Opposing Forces**: GBM pushes prices to follow trends (`dS ~ μS`), while OU pulls them back to equilibrium (`dX ~ -θX`)
2. **Timescale Mismatch**: Different physics operate at different timescales
3. **Optimization Conflict**: Gradient descent cannot satisfy all constraints simultaneously
4. **Real Markets**: Don't perfectly obey all theoretical models

**Action**: Use simpler configurations (hybrid or pure), accept that real markets violate theory

## Example Workflow

```bash
# 1. Run comparison
./run.sh
# Choose option 10, enter 100 epochs, train all variants

# 2. Review results
cat results/pinn_comparison/comparison_report.csv

# 3. Read theory
cat results/pinn_comparison/README_theory.md

# 4. Load best model for forecasting
python3
>>> import torch
>>> checkpoint = torch.load('models/pinn_gbm_ou_best.pt')
>>> # Use for predictions
```

## Implementation Details

### Violation Score Calculation
```python
violation_score = L_physics / (L_data + ε)
```
- Computed at final training epoch
- Lower is better (physics satisfied with less penalty)
- Baseline always has score = 0

### Training Process
For each variant:
1. Initialize PINN model with specific λ configuration
2. Train for N epochs with early stopping
3. Track separate data_loss and physics_loss
4. Save best checkpoint based on validation loss
5. Evaluate on test set
6. Calculate violation score

### Physics Loss Components
- **GBM Loss**: `||dS/dt - μS||²`
- **Black-Scholes Loss**: `||∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV||²`
- **OU Loss**: `||dX/dt - θ(μ - X)||²`
- **Langevin Loss**: `||dX/dt + γ∇U||²`

### Data Requirements
The system uses:
- `prices`: Stock price sequences
- `returns`: Log/simple returns
- `volatilities`: Rolling volatility estimates

These are automatically computed from input data.

## Troubleshooting

### "No data fetched"
**Solution**: Run option 2 (Fetch Data) first or ensure database is populated

### "Black-Scholes loss computation failed"
**Reason**: Requires second-order derivatives (computationally expensive)
**Solution**: Black-Scholes variant may have higher computational cost, this is expected

### All variants have similar MSE
**Interpretation**: Physics constraints not strongly impacting predictions
**Possible Reasons**:
- Insufficient training epochs
- Physics weights (λ) too small
- Data doesn't exhibit strong physics patterns

**Solution**: Increase epochs or physics weights in configuration

### Global Constraint has NaN loss
**Reason**: Contradictory gradients causing numerical instability
**Solution**: Reduce physics weights or remove conflicting constraints

## Advanced Usage

### Custom Physics Weights
Modify `src/training/train_pinn_variants.py`:
```python
PINN_CONFIGURATIONS['custom'] = {
    'name': 'Custom Configuration',
    'lambda_gbm': 0.2,  # Adjust weights
    'lambda_bs': 0.0,
    'lambda_ou': 0.05,
    'lambda_langevin': 0.0,
    'enable_physics': True
}
```

### Subset Training
Train only specific variants:
```bash
python3 -m src.training.train_pinn_variants --epochs 50 --variants gbm ou gbm_ou
```

### Hyperparameter Tuning
Edit `config/config.yaml`:
- Adjust `hidden_dim`, `num_layers` for model capacity
- Modify `learning_rate` for optimization
- Change `batch_size` for training stability

## Research Questions Answered

1. **Do physics constraints improve financial forecasting?**
   → Compare baseline vs physics-enabled variants

2. **Which physics constraint is most effective?**
   → Check MSE ranking in comparison_report.csv

3. **Can all constraints coexist?**
   → Examine global constraint convergence and violation score

4. **Are markets trend-following or mean-reverting?**
   → Compare GBM vs OU performance

5. **Is there a trade-off between accuracy and theory?**
   → Plot Test_MSE vs Violation_Score

## References

- Geometric Brownian Motion: Foundation of Black-Scholes theory
- Ornstein-Uhlenbeck Process: Mean-reverting stochastic process
- Black-Scholes PDE: No-arbitrage constraint in continuous time
- Langevin Dynamics: Momentum modeling with friction

## Next Steps

After running the comparison:

1. **Select Best Model**: Based on combined rank
2. **Validate on New Data**: Test generalization
3. **Deploy for Forecasting**: Use best checkpoint
4. **Regime Detection**: Switch between models based on market conditions
5. **Ensemble Methods**: Combine predictions from multiple variants

---

For questions or issues, see the detailed error messages in the training logs or review README_theory.md for interpretation guidance.
