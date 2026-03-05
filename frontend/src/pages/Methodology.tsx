import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Badge } from "../components/ui/badge"
import { Book, Brain, Atom, TrendingUp, BarChart3 } from "lucide-react"

export default function Methodology() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Methodology</h1>
        <p className="text-muted-foreground">
          Research documentation and mathematical foundations
        </p>
      </div>

      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Book className="h-5 w-5" />
            Research Overview
          </CardTitle>
        </CardHeader>
        <CardContent className="prose prose-sm max-w-none dark:prose-invert">
          <p>
            This project implements Physics-Informed Neural Networks (PINNs) for financial
            time series forecasting. By incorporating physical constraints derived from
            financial mathematics into the neural network training process, we achieve
            more robust and interpretable predictions.
          </p>
        </CardContent>
      </Card>

      {/* Physics Equations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="h-5 w-5" />
            Physics Constraints
          </CardTitle>
          <CardDescription>
            Stochastic differential equations used as physics constraints
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* GBM */}
          <div className="rounded-lg border p-4">
            <h4 className="mb-2 flex items-center gap-2 font-semibold">
              Geometric Brownian Motion (GBM)
              <Badge variant="secondary">Drift-Diffusion</Badge>
            </h4>
            <div className="my-4 rounded bg-muted p-4 font-mono text-center">
              dS = μS dt + σS dW
            </div>
            <p className="text-sm text-muted-foreground">
              Models stock price dynamics with constant drift μ and volatility σ.
              Widely used in options pricing and risk management.
            </p>
          </div>

          {/* OU Process */}
          <div className="rounded-lg border p-4">
            <h4 className="mb-2 flex items-center gap-2 font-semibold">
              Ornstein-Uhlenbeck Process
              <Badge variant="secondary">Mean Reversion</Badge>
            </h4>
            <div className="my-4 rounded bg-muted p-4 font-mono text-center">
              dX = θ(μ - X) dt + σ dW
            </div>
            <p className="text-sm text-muted-foreground">
              Models mean-reverting behavior with speed θ. Useful for modeling
              interest rates, volatility, and pairs trading.
            </p>
          </div>

          {/* Black-Scholes */}
          <div className="rounded-lg border p-4">
            <h4 className="mb-2 flex items-center gap-2 font-semibold">
              Black-Scholes PDE
              <Badge variant="secondary">Options</Badge>
            </h4>
            <div className="my-4 rounded bg-muted p-4 font-mono text-center">
              ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
            </div>
            <p className="text-sm text-muted-foreground">
              Fundamental equation for option pricing. Ensures no-arbitrage condition
              in derivative pricing.
            </p>
          </div>

          {/* Langevin */}
          <div className="rounded-lg border p-4">
            <h4 className="mb-2 flex items-center gap-2 font-semibold">
              Langevin Dynamics
              <Badge variant="secondary">Temperature</Badge>
            </h4>
            <div className="my-4 rounded bg-muted p-4 font-mono text-center">
              dX = -γ∇U(X) dt + √(2γT) dW
            </div>
            <p className="text-sm text-muted-foreground">
              Models dynamics with friction γ and temperature T. Captures market
              microstructure effects and volatility clustering.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* PINN Architecture */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            PINN Architecture
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              The Physics-Informed Neural Network combines a traditional sequence model
              (LSTM/GRU) with physics-based loss functions that enforce consistency with
              known financial dynamics.
            </p>

            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border p-4">
                <h5 className="font-medium">Input Layer</h5>
                <p className="text-sm text-muted-foreground">
                  Sequence of financial features: returns, volatility, momentum, RSI, MACD
                </p>
              </div>
              <div className="rounded-lg border p-4">
                <h5 className="font-medium">Recurrent Layers</h5>
                <p className="text-sm text-muted-foreground">
                  LSTM/GRU cells with learnable physics parameters (θ, γ, T)
                </p>
              </div>
              <div className="rounded-lg border p-4">
                <h5 className="font-medium">Output Layer</h5>
                <p className="text-sm text-muted-foreground">
                  Price prediction with uncertainty estimates via MC Dropout
                </p>
              </div>
            </div>

            <div className="rounded-lg border p-4">
              <h5 className="mb-2 font-medium">Loss Function</h5>
              <div className="rounded bg-muted p-4 font-mono text-center">
                L_total = L_data + λ * L_physics
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Where L_data is MSE loss and L_physics combines residuals from all
                physics equations with learnable weighting λ.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Evaluation Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <h5 className="mb-2 font-medium">Prediction Metrics</h5>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><strong>RMSE:</strong> Root Mean Square Error</li>
                <li><strong>MAE:</strong> Mean Absolute Error</li>
                <li><strong>MAPE:</strong> Mean Absolute Percentage Error</li>
                <li><strong>R²:</strong> Coefficient of Determination</li>
                <li><strong>DA:</strong> Directional Accuracy</li>
              </ul>
            </div>
            <div>
              <h5 className="mb-2 font-medium">Financial Metrics</h5>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><strong>Sharpe Ratio:</strong> Risk-adjusted return</li>
                <li><strong>Sortino Ratio:</strong> Downside risk-adjusted</li>
                <li><strong>Max Drawdown:</strong> Peak-to-trough decline</li>
                <li><strong>Calmar Ratio:</strong> Return/Drawdown</li>
                <li><strong>Win Rate:</strong> % profitable trades</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* References */}
      <Card>
        <CardHeader>
          <CardTitle>References</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="list-decimal space-y-2 pl-6 text-sm">
            <li>
              Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
              neural networks: A deep learning framework for solving forward and inverse
              problems involving nonlinear partial differential equations.
            </li>
            <li>
              Black, F., & Scholes, M. (1973). The pricing of options and corporate
              liabilities. Journal of Political Economy.
            </li>
            <li>
              Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian
              motion. Physical Review.
            </li>
            <li>
              Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural
              Computation.
            </li>
          </ol>
        </CardContent>
      </Card>
    </div>
  )
}
