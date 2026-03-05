# Claude Code Instructions

## Development Preferences

**IMPORTANT: Always work on the React frontend (`frontend/`) and FastAPI backend (`backend/`).
NEVER work on the Streamlit app (`src/web/`) unless explicitly requested.**

## CRITICAL: Research Project - Model Integrity

**THIS IS A DISSERTATION/RESEARCH PROJECT. All models MUST be connected to their REAL neural network implementations.**

When working on model training or inference:
1. **NEVER use simulated/mock training** - All models must use actual PyTorch implementations
2. **Verify model connections** - Check that `HAS_SRC=True` in training_service.py
3. **Test ALL model types** - Every model (LSTM, GRU, Transformer, PINN variants) must create correctly
4. **Check backend logs** - Should show `[REAL TRAINING]` not simulated mode

### Model Registry (src/models/model_registry.py)
All 21 models must map to real implementations:
- **Baseline**: lstm, gru, bilstm, attention_lstm, transformer
- **PINN variants**: baseline_pinn, gbm, ou, black_scholes, gbm_ou, global (+ pinn_ prefixes)
- **Advanced**: stacked, residual (+ _pinn suffixes)

### Verification Command (Run Before Any Training Work)
```bash
source backend/venv/bin/activate && python -c "
import sys; sys.path.insert(0, '.')
import torch
from pathlib import Path
from src.models.model_registry import ModelRegistry

registry = ModelRegistry(Path('.'))
test_input = torch.randn(2, 30, 5)

models = ['lstm', 'gru', 'bilstm', 'transformer',
          'baseline_pinn', 'gbm', 'ou', 'black_scholes', 'gbm_ou', 'global',
          'stacked', 'residual']

for m in models:
    model = registry.create_model(m, input_dim=5)
    if model:
        out = model(test_input)
        pred = out[0] if isinstance(out, tuple) else out
        is_pinn = hasattr(model, 'compute_loss')
        print(f'✓ {m}: {model.__class__.__name__} ({\"PINN\" if is_pinn else \"Baseline\"})')
    else:
        print(f'✗ {m}: FAILED')
"
```

### Model Architecture Mapping (CRITICAL FOR RESEARCH)
| Model Key | Class | Type | Physics Constraints |
|-----------|-------|------|---------------------|
| lstm | LSTMModel | baseline | None |
| gru | GRUModel | baseline | None |
| bilstm | LSTMModel (bidirectional) | baseline | None |
| attention_lstm | LSTMModel | baseline | None |
| transformer | TransformerModel | baseline | None |
| baseline_pinn | PINNModel | pinn | λ_gbm=0, λ_ou=0, λ_bs=0 |
| gbm | PINNModel | pinn | λ_gbm=0.1 (trend) |
| ou | PINNModel | pinn | λ_ou=0.1 (mean-reversion) |
| black_scholes | PINNModel | pinn | λ_bs=0.1 (no-arbitrage) |
| gbm_ou | PINNModel | pinn | λ_gbm=0.05, λ_ou=0.05 |
| global | PINNModel | pinn | λ_gbm=0.05, λ_ou=0.05, λ_bs=0.03 |
| stacked | StackedPINN | advanced | Physics encoder + parallel LSTM/GRU |
| residual | ResidualPINN | advanced | Base + physics correction |

### Training Mode Check
GET `/api/training/mode` should return `{"mode": "real", "using_real_models": true}`

## Project Structure

- `frontend/` - React + TypeScript + Vite frontend (PRIMARY UI)
- `backend/` - FastAPI Python backend (PRIMARY API)
- `src/` - Core ML/data processing modules (shared by backend)
- `src/models/` - Neural network implementations (LSTM, GRU, Transformer, PINN)
- `src/training/` - Training logic with physics-informed losses
- `src/web/` - Legacy Streamlit dashboards (DO NOT MODIFY)

## Tech Stack

- Frontend: React 18, TypeScript, TailwindCSS, TanStack Query, Recharts
- Backend: FastAPI, Pydantic, pandas, numpy
- ML: PyTorch, PINN models with physics constraints (GBM, OU, Black-Scholes)

## Running the App

```bash
# Backend
cd backend && source venv/bin/activate && python run.py

# Frontend
cd frontend && npm run dev
```

## Key Files

- Frontend entry: `frontend/src/App.tsx`
- Backend entry: `backend/run.py`
- API routes: `backend/app/api/routes/`
- React hooks: `frontend/src/hooks/`
- Services: `frontend/src/services/`
- **Model Registry**: `src/models/model_registry.py` - Central model creation
- **Training Service**: `backend/app/services/training_service.py` - Must use HAS_SRC=True
- **PINN Models**: `src/models/pinn.py` - Physics-informed neural networks
- **Stacked/Residual PINN**: `src/models/stacked_pinn.py` - Advanced architectures
- **Project Documentation**: `DOCUMENT.md` - All changes must be documented here

## CRITICAL: Documentation Requirements

**ALL changes, bug fixes, and new features MUST be documented in `DOCUMENT.md`.**

This is a dissertation/research project. Proper documentation is essential for:
1. **Reproducibility** - Others must be able to understand and reproduce results
2. **Audit Trail** - Track what was changed, when, and why
3. **Research Integrity** - Document methodology and implementation details

### What to Document

| Change Type | Required Documentation |
|-------------|------------------------|
| Bug fixes | Issue description, root cause, solution with code snippets |
| New features | Feature description, API endpoints, usage examples |
| Architecture changes | Before/after diagrams, rationale, files modified |
| Configuration changes | Old vs new values, impact on training/results |
| Training improvements | Parameters changed, expected impact, verification steps |

### Documentation Format

Follow the existing format in `DOCUMENT.md`:
1. Add a new numbered section (e.g., `## 14. Feature Name`)
2. Include date and overview
3. Provide code snippets with before/after comparisons
4. List all files modified
5. Add verification steps where applicable

### Example Documentation Entry

```markdown
## 14. [Feature/Fix Name] (YYYY-MM-DD)

### Overview
Brief description of what was changed and why.

### Changes Made
1. **File**: `path/to/file.py`
   - What was changed
   - Code snippet if relevant

### Verification
How to verify the change works correctly.

### Files Modified
| File | Changes |
|------|---------|
| path/to/file.py | Description |
```
