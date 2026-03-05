import { BrowserRouter, Routes, Route } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Layout } from "./components/layout/Layout"
import { ErrorBoundary } from "./components/common/ErrorBoundary"

// Pages
import Dashboard from "./pages/Dashboard"
import DataManagement from "./pages/DataManagement"
import PINNAnalysis from "./pages/PINNAnalysis"
import ModelComparison from "./pages/ModelComparison"
import ComprehensiveAnalysis from "./pages/ComprehensiveAnalysis"
import MonteCarlo from "./pages/MonteCarlo"
import Backtesting from "./pages/Backtesting"
import Methodology from "./pages/Methodology"
import Training from "./pages/Training"
import BatchTraining from "./pages/BatchTraining"
import Predictions from "./pages/Predictions"
import DataExplorer from "./pages/DataExplorer"
import PhysicsParameters from "./pages/PhysicsParameters"
import TradingAgent from "./pages/TradingAgent"
import Settings from "./pages/Settings"
import ModelManager from "./pages/ModelManager"
import Leaderboard from "./pages/Leaderboard"
import VolatilityForecasting from "./pages/VolatilityForecasting"
import DissertationAnalysis from "./pages/DissertationAnalysis"
import SpectralAnalysis from "./pages/SpectralAnalysis"

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30000,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="data-management" element={<DataManagement />} />
              <Route path="pinn" element={<PINNAnalysis />} />
              <Route path="models" element={<ModelComparison />} />
              <Route path="comprehensive" element={<ComprehensiveAnalysis />} />
              <Route path="monte-carlo" element={<MonteCarlo />} />
              <Route path="backtesting" element={<Backtesting />} />
              <Route path="methodology" element={<Methodology />} />
              <Route path="training" element={<Training />} />
              <Route path="batch-training" element={<BatchTraining />} />
              <Route path="predictions" element={<Predictions />} />
              <Route path="data" element={<DataExplorer />} />
              <Route path="physics" element={<PhysicsParameters />} />
              <Route path="trading" element={<TradingAgent />} />
              <Route path="model-manager" element={<ModelManager />} />
              <Route path="leaderboard" element={<Leaderboard />} />
              <Route path="volatility" element={<VolatilityForecasting />} />
              <Route path="dissertation" element={<DissertationAnalysis />} />
              <Route path="spectral" element={<SpectralAnalysis />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ErrorBoundary>
    </QueryClientProvider>
  )
}

export default App
