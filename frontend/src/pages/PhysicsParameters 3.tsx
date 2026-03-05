import { useState, useMemo } from "react"
import { useModels } from "../hooks/useModels"
import { usePhysicsMetrics, useMetricsComparison } from "../hooks/useMetrics"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Badge } from "../components/ui/badge"
import { MetricCard, MetricGrid } from "../components/common/MetricCard"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts"
import { Atom, Activity, AlertCircle } from "lucide-react"

export default function PhysicsParameters() {
  const { data: modelsData, isLoading: modelsLoading } = useModels()
  const [selectedModel, setSelectedModel] = useState("pinn_gbm_ou")

  const pinnModels = modelsData?.models.filter((m) => m.is_pinn) || []
  const pinnModelKeys = pinnModels.map(m => m.model_key)

  // Fetch physics metrics for selected model
  const {
    data: physicsMetrics,
    isLoading: physicsLoading,
    error: physicsError
  } = usePhysicsMetrics(selectedModel)

  // Fetch comparison metrics for all PINN models
  const {
    data: comparisonData,
    isLoading: comparisonLoading
  } = useMetricsComparison(pinnModelKeys)

  // Transform comparison data for the table
  const paramComparison = useMemo(() => {
    if (!comparisonData?.models) return []
    return comparisonData.models
      .filter(m => m.is_pinn && m.physics_metrics)
      .map(m => ({
        model: m.model_name,
        model_key: m.model_key,
        theta: m.physics_metrics?.theta ?? null,
        gamma: m.physics_metrics?.gamma ?? null,
        T: m.physics_metrics?.temperature ?? null,
        mu: m.physics_metrics?.mu ?? null,
        sigma: m.physics_metrics?.sigma ?? null,
      }))
  }, [comparisonData])

  // Generate parameter interpretation based on actual values
  const getParameterInterpretation = (paramName: string, value: number | undefined | null) => {
    if (value === undefined || value === null) return null

    switch (paramName) {
      case 'theta':
        const reversionDays = value > 0 ? (1 / value).toFixed(1) : 'N/A'
        return `Mean reversion speed in the Ornstein-Uhlenbeck process. A value of ~${value.toFixed(2)} suggests ${value > 0.5 ? 'fast' : value > 0.2 ? 'moderate' : 'slow'} mean reversion, with prices reverting to the mean over approximately ${reversionDays} days (1/\u03b8).`
      case 'gamma':
        return `Friction coefficient in Langevin dynamics. A value of ${value.toFixed(4)} indicates ${value < 0.1 ? 'low' : value < 0.2 ? 'moderate' : 'high'} market friction and ${value < 0.1 ? 'faster' : 'slower'} price adjustments to new information.`
      case 'temperature':
        return `Market temperature parameter representing overall volatility and randomness. A value of ${value.toFixed(4)} indicates ${value < 0.01 ? 'low' : value < 0.02 ? 'moderate' : 'high'} noise in price movements.`
      case 'mu':
        const annualReturn = (value * 100).toFixed(2)
        return `Drift parameter in Geometric Brownian Motion. Represents the expected annual return of ~${annualReturn}%, ${parseFloat(annualReturn) > 7 ? 'consistent with' : parseFloat(annualReturn) > 4 ? 'slightly below' : 'below'} historical equity returns.`
      case 'sigma':
        const annualVol = (value * 100).toFixed(1)
        return `Volatility parameter representing annualized price volatility of ~${annualVol}%. This is ${parseFloat(annualVol) < 15 ? 'below' : parseFloat(annualVol) < 25 ? 'typical for' : 'above typical for'} individual stocks in the S&P 500.`
      default:
        return null
    }
  }

  const isLoading = modelsLoading || physicsLoading

  if (modelsLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  // Get display values with fallback
  const theta = physicsMetrics?.theta
  const gamma = physicsMetrics?.gamma
  const temperature = physicsMetrics?.temperature
  const mu = physicsMetrics?.mu
  const sigma = physicsMetrics?.sigma

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Physics Parameters</h1>
        <p className="text-muted-foreground">
          Analyze learned physics parameters from PINN models
        </p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select PINN Model</CardTitle>
        </CardHeader>
        <CardContent>
          {pinnModels.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {pinnModels.map((model) => (
                <Button
                  key={model.model_key}
                  variant={selectedModel === model.model_key ? "default" : "outline"}
                  onClick={() => setSelectedModel(model.model_key)}
                >
                  {model.display_name}
                </Button>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-muted-foreground">
              <AlertCircle className="h-5 w-5" />
              <span>No PINN models available. Train a PINN model first.</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Current Parameters */}
      <div>
        <h2 className="mb-4 text-xl font-semibold flex items-center gap-2">
          <Atom className="h-5 w-5" />
          Learned Physics Parameters
        </h2>
        {physicsLoading ? (
          <div className="flex h-32 items-center justify-center">
            <LoadingSpinner size="md" />
          </div>
        ) : physicsError ? (
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-muted-foreground">
                <AlertCircle className="h-5 w-5" />
                <span>No physics metrics available for this model. Ensure the model has been trained.</span>
              </div>
            </CardContent>
          </Card>
        ) : (
          <MetricGrid columns={5}>
            <MetricCard
              title="\u03b8 (Theta)"
              value={theta !== undefined && theta !== null ? theta.toFixed(4) : "N/A"}
              subtitle="Mean reversion speed"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="\u03b3 (Gamma)"
              value={gamma !== undefined && gamma !== null ? gamma.toFixed(4) : "N/A"}
              subtitle="Friction coefficient"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="T (Temperature)"
              value={temperature !== undefined && temperature !== null ? temperature.toFixed(4) : "N/A"}
              subtitle="Market temperature"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="\u03bc (Mu)"
              value={mu !== undefined && mu !== null ? mu.toFixed(4) : "N/A"}
              subtitle="Drift parameter"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="\u03c3 (Sigma)"
              value={sigma !== undefined && sigma !== null ? sigma.toFixed(4) : "N/A"}
              subtitle="Volatility"
              icon={<Activity className="h-4 w-4" />}
            />
          </MetricGrid>
        )}
      </div>

      {/* Physics Loss Information */}
      {physicsMetrics && (
        <Card>
          <CardHeader>
            <CardTitle>Physics Loss Components</CardTitle>
            <CardDescription>
              Breakdown of physics-informed loss terms
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-4">
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Total Physics Loss</div>
                <div className="text-2xl font-bold">
                  {physicsMetrics.total_physics_loss?.toFixed(6) ?? "N/A"}
                </div>
              </div>
              {physicsMetrics.gbm_loss != null && (
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">GBM Loss</div>
                  <div className="text-2xl font-bold">
                    {physicsMetrics.gbm_loss.toFixed(6)}
                  </div>
                </div>
              )}
              {physicsMetrics.ou_loss != null && (
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">OU Loss</div>
                  <div className="text-2xl font-bold">
                    {physicsMetrics.ou_loss.toFixed(6)}
                  </div>
                </div>
              )}
              {physicsMetrics.langevin_loss != null && (
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Langevin Loss</div>
                  <div className="text-2xl font-bold">
                    {physicsMetrics.langevin_loss.toFixed(6)}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Comparison */}
      <Card>
        <CardHeader>
          <CardTitle>Parameter Comparison Across Models</CardTitle>
          <CardDescription>
            Compare learned parameters between different PINN variants
          </CardDescription>
        </CardHeader>
        <CardContent>
          {comparisonLoading ? (
            <div className="flex h-48 items-center justify-center">
              <LoadingSpinner size="md" />
            </div>
          ) : paramComparison.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="px-4 py-3 text-left font-medium">Model</th>
                    <th className="px-4 py-3 text-right font-medium">\u03b8 (Theta)</th>
                    <th className="px-4 py-3 text-right font-medium">\u03b3 (Gamma)</th>
                    <th className="px-4 py-3 text-right font-medium">T</th>
                    <th className="px-4 py-3 text-right font-medium">\u03bc (Mu)</th>
                    <th className="px-4 py-3 text-right font-medium">\u03c3 (Sigma)</th>
                  </tr>
                </thead>
                <tbody>
                  {paramComparison.map((row) => (
                    <tr key={row.model_key} className="border-b hover:bg-muted/50">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {row.model}
                          <Badge variant="secondary">PINN</Badge>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {row.theta?.toFixed(4) || "-"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {row.gamma?.toFixed(4) || "-"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {row.T?.toFixed(4) || "-"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {row.mu?.toFixed(4) || "-"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {row.sigma?.toFixed(4) || "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="mb-2 h-8 w-8" />
              <p>No PINN models with physics metrics available.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Parameter Interpretation */}
      <Card>
        <CardHeader>
          <CardTitle>Parameter Interpretation</CardTitle>
          <CardDescription>Understanding what the learned parameters mean</CardDescription>
        </CardHeader>
        <CardContent>
          {physicsLoading ? (
            <div className="flex h-48 items-center justify-center">
              <LoadingSpinner size="md" />
            </div>
          ) : physicsMetrics ? (
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                {theta !== undefined && theta !== null && (
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium">\u03b8 (Theta) = {theta.toFixed(4)}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {getParameterInterpretation('theta', theta)}
                    </p>
                  </div>
                )}
                {gamma !== undefined && gamma !== null && (
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium">\u03b3 (Gamma) = {gamma.toFixed(4)}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {getParameterInterpretation('gamma', gamma)}
                    </p>
                  </div>
                )}
                {temperature !== undefined && temperature !== null && (
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium">T (Temperature) = {temperature.toFixed(4)}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {getParameterInterpretation('temperature', temperature)}
                    </p>
                  </div>
                )}
              </div>
              <div className="space-y-4">
                {mu !== undefined && mu !== null && (
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium">\u03bc (Mu) = {mu.toFixed(4)}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {getParameterInterpretation('mu', mu)}
                    </p>
                  </div>
                )}
                {sigma !== undefined && sigma !== null && (
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium">\u03c3 (Sigma) = {sigma.toFixed(4)}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {getParameterInterpretation('sigma', sigma)}
                    </p>
                  </div>
                )}
                <div className="rounded-lg border p-4 bg-muted/50">
                  <h4 className="font-medium">Key Insight</h4>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {physicsMetrics.total_physics_loss != null && physicsMetrics.total_physics_loss < 0.01
                      ? "The learned parameters are well-fitted with low physics loss, indicating strong consistency with the underlying stochastic processes."
                      : "The model is learning physics parameters that balance between data fit and physical constraints. Consider adjusting physics weight for better convergence."}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="mb-2 h-8 w-8" />
              <p>Select a trained PINN model to view parameter interpretations.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
