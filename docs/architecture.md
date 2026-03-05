# System Architecture

## Overview

The PINN Financial Forecasting system follows a three-tier architecture with a React frontend, FastAPI backend, and PyTorch-based ML core.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Pages   │  │  Charts  │  │  Hooks   │  │  State (Zustand) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                              │                                   │
│                    HTTP/REST & WebSocket                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (FastAPI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Routes  │  │ Schemas  │  │ Services │  │   WebSocket      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                              │                                   │
│                    Python Imports                                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML Core (src/)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Models  │  │ Training │  │Evaluation│  │      Data        │ │
│  │  (PINN)  │  │ Pipeline │  │ Metrics  │  │   Processing     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │   PostgreSQL +   │  │  Model Weights   │  │   Results/     │ │
│  │   TimescaleDB    │  │    (Models/)     │  │   Metrics      │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Dissertaion-Project/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   ├── core/           # Database, exceptions
│   │   ├── schemas/        # Pydantic models
│   │   └── services/       # Business logic
│   └── run.py              # Entry point
│
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   │   ├── charts/     # Recharts wrappers
│   │   │   ├── common/     # Shared components
│   │   │   ├── layout/     # Layout components
│   │   │   └── ui/         # shadcn/ui primitives
│   │   ├── hooks/          # React Query hooks
│   │   ├── pages/          # Route pages
│   │   ├── services/       # API client
│   │   ├── stores/         # Zustand stores
│   │   └── types/          # TypeScript types
│   └── vite.config.ts
│
├── src/                    # ML Core Library
│   ├── data/               # Data fetching/processing
│   ├── models/             # PINN implementations
│   ├── training/           # Training pipelines
│   ├── evaluation/         # Metrics & backtesting
│   ├── trading/            # Trading strategies
│   └── utils/              # Utilities
│
├── docs/                   # Documentation
├── tests/                  # Test suite
├── Models/                 # Saved model weights
└── results/                # Evaluation results
```

## Component Details

### Frontend Components

| Directory | Purpose |
|-----------|---------|
| `components/charts/` | Recharts wrappers for financial visualizations |
| `components/ui/` | shadcn/ui base components (Button, Card, etc.) |
| `components/common/` | Shared components (MetricCard, LoadingSpinner) |
| `components/layout/` | Page layout (Sidebar, Header) |
| `pages/` | 14 dashboard pages |
| `hooks/` | React Query data fetching hooks |
| `stores/` | Zustand global state management |
| `services/` | Axios API client modules |
| `types/` | TypeScript interfaces |

### Backend Components

| Directory | Purpose |
|-----------|---------|
| `api/routes/` | FastAPI route handlers |
| `schemas/` | Pydantic request/response models |
| `services/` | Business logic wrapping ML core |
| `core/` | Database connection, exceptions |

### ML Core Components

| Directory | Purpose |
|-----------|---------|
| `models/` | PINN, LSTM, Transformer implementations |
| `training/` | Training loops, curriculum learning |
| `evaluation/` | Metrics, backtesting, Monte Carlo |
| `data/` | Data fetching, preprocessing |
| `trading/` | Trading agent, position sizing |

## Data Flow

### Prediction Request

```
1. User selects ticker and model in frontend
2. Frontend calls POST /api/predictions/predict
3. Backend PredictionService loads model from ModelService
4. ModelService loads weights from Models/ directory
5. Prediction runs through PINN forward pass
6. Results returned with confidence intervals
7. Frontend displays in PredictionChart
```

### Training Flow

```
1. User configures training in Training page
2. Frontend opens WebSocket to /api/training/ws/{job_id}
3. Backend TrainingService starts training loop
4. Progress updates sent via WebSocket every epoch
5. Frontend TrainingStore updates, charts re-render
6. On completion, model saved to Models/
```

## Key Design Decisions

1. **Service Layer Pattern**: Backend services wrap ML core, providing clean API for routes
2. **React Query**: Server state caching and synchronization
3. **Zustand**: Lightweight client-side state (theme, selections)
4. **Model Registry**: Centralized model loading and caching
5. **WebSocket for Training**: Real-time progress without polling
