# PINN Financial Forecasting - Frontend

React + TypeScript frontend for the PINN Financial Forecasting application.

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

App runs at http://localhost:5173

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool |
| Tailwind CSS | Styling |
| shadcn/ui | Component library |
| Recharts | Data visualization |
| Zustand | Client state |
| React Query | Server state |
| React Router | Routing |
| Axios | HTTP client |

## Directory Structure

```
frontend/
├── src/
│   ├── main.tsx              # App entry point
│   ├── App.tsx               # Router setup
│   ├── index.css             # Tailwind imports
│   │
│   ├── components/           # Reusable components
│   │   ├── ui/               # shadcn/ui primitives
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── badge.tsx
│   │   │   ├── input.tsx
│   │   │   ├── progress.tsx
│   │   │   └── index.ts      # Barrel export
│   │   │
│   │   ├── charts/           # Recharts wrappers
│   │   │   ├── PriceChart.tsx
│   │   │   ├── PredictionChart.tsx
│   │   │   ├── EquityChart.tsx
│   │   │   ├── DrawdownChart.tsx
│   │   │   ├── DistributionChart.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── common/           # Shared components
│   │   │   ├── MetricCard.tsx
│   │   │   ├── LoadingSpinner.tsx
│   │   │   ├── ErrorBoundary.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── layout/           # Layout components
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Layout.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts          # Main barrel export
│   │
│   ├── pages/                # Dashboard pages
│   │   ├── Dashboard.tsx
│   │   ├── PINNAnalysis.tsx
│   │   ├── ModelComparison.tsx
│   │   ├── Predictions.tsx
│   │   ├── Backtesting.tsx
│   │   ├── Training.tsx
│   │   ├── MonteCarlo.tsx
│   │   ├── Metrics.tsx
│   │   ├── DataExplorer.tsx
│   │   ├── Methodology.tsx
│   │   ├── Settings.tsx
│   │   ├── ComprehensiveAnalysis.tsx
│   │   ├── PhysicsParameters.tsx
│   │   └── TradingAgent.tsx
│   │
│   ├── hooks/                # React Query hooks
│   │   ├── useModels.ts
│   │   ├── usePredictions.ts
│   │   ├── useMetrics.ts
│   │   ├── useWebSocket.ts
│   │   └── index.ts
│   │
│   ├── stores/               # Zustand stores
│   │   ├── appStore.ts       # Theme, sidebar, selections
│   │   ├── trainingStore.ts  # Training job state
│   │   └── index.ts
│   │
│   ├── services/             # API client
│   │   ├── api.ts            # Axios instance
│   │   ├── modelsApi.ts
│   │   ├── dataApi.ts
│   │   ├── predictionsApi.ts
│   │   ├── metricsApi.ts
│   │   └── index.ts
│   │
│   ├── types/                # TypeScript interfaces
│   │   ├── models.ts
│   │   ├── predictions.ts
│   │   ├── metrics.ts
│   │   ├── api.ts
│   │   └── index.ts
│   │
│   └── lib/                  # Utilities
│       └── utils.ts          # cn() helper, formatters
│
├── package.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Overview with key metrics |
| PINN Analysis | `/pinn` | Physics parameters, loss curves |
| Model Comparison | `/comparison` | Side-by-side evaluation |
| Predictions | `/predictions` | Price forecasting |
| Backtesting | `/backtesting` | Historical performance |
| Training | `/training` | Real-time training |
| Monte Carlo | `/monte-carlo` | Risk simulation |
| Metrics | `/metrics` | Metrics calculator |
| Data Explorer | `/data` | Stock data exploration |
| Methodology | `/methodology` | Research documentation |
| Settings | `/settings` | App configuration |

## Component Usage

### Importing Components

```typescript
// Recommended: Use barrel exports
import { Button, Card, Badge } from "@/components/ui"
import { PriceChart, EquityChart } from "@/components/charts"
import { MetricCard, LoadingSpinner } from "@/components/common"

// Or import from main barrel
import { Button, PriceChart, MetricCard } from "@/components"
```

### Using Hooks

```typescript
import { useModels, usePredictions, useFinancialMetrics } from "@/hooks"

function MyComponent() {
  const { data: models, isLoading } = useModels()
  const { data: predictions } = usePredictions("SPY", "pinn_gbm_ou")
  const { data: metrics } = useFinancialMetrics("pinn_gbm_ou")

  // ...
}
```

### Using Stores

```typescript
import { useAppStore, useTrainingStore } from "@/stores"

function MyComponent() {
  const { selectedTicker, setSelectedTicker, theme, toggleTheme } = useAppStore()
  const { isTraining, progress, startTraining } = useTrainingStore()

  // ...
}
```

## Development

```bash
# Development server with hot reload
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## API Proxy

Vite proxies `/api` requests to the backend:

```typescript
// vite.config.ts
server: {
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## Adding shadcn/ui Components

```bash
# Add new component
npx shadcn@latest add [component-name]

# Examples
npx shadcn@latest add dialog
npx shadcn@latest add dropdown-menu
npx shadcn@latest add table
```

## Styling

Uses Tailwind CSS with custom theme:

```css
/* Custom colors available */
bg-background    /* Page background */
bg-card          /* Card background */
text-foreground  /* Primary text */
text-muted-foreground /* Secondary text */
border-border    /* Border color */
```

## Building

```bash
# Production build
npm run build

# Output in dist/
```

Bundle size warning: The build produces a ~880KB JavaScript bundle. Consider code splitting for production:

```typescript
// Lazy load pages
const Dashboard = lazy(() => import('./pages/Dashboard'))
```
