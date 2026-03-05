import { useState, useMemo, useEffect } from "react"
import { useModels } from "../hooks/useModels"
import { useMetricsComparison } from "../hooks/useMetrics"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts"
import { Brain, AlertCircle } from "lucide-react"

export default function ModelComparison() {
  const { data: modelsData, isLoading: modelsLoading } = useModels()
  const [selectedModels, setSelectedModels] = useState<string[]>([])

  // Auto-select all models when data loads
  useEffect(() => {
    if (modelsData?.models && selectedModels.length === 0) {
      setSelectedModels(modelsData.models.map(m => m.model_key))
    }
  }, [modelsData])

  // Fetch comparison metrics for selected models
  const { data: comparisonData, isLoading: metricsLoading } = useMetricsComparison(selectedModels)

  // Transform comparison data for display
  const metricsTableData = useMemo(() => {
    if (!comparisonData?.models) return []
    return comparisonData.models.map(m => ({
      model: m.model_name,
      model_key: m.model_key,
      rmse: m.ml_metrics.rmse,
      mae: m.ml_metrics.mae,
      r2: m.ml_metrics.r2,
      da: m.ml_metrics.directional_accuracy,
      sharpe: m.financial_metrics?.sharpe_ratio ?? 0,
      is_pinn: m.is_pinn,
    }))
  }, [comparisonData])

  // Find the best model
  const bestModelKey = comparisonData?.best_by_metric?.rmse

  // Transform data for radar chart (normalize metrics to 0-100 scale)
  const radarData = useMemo(() => {
    if (metricsTableData.length === 0) return []

    // Get min/max for each metric to normalize
    const metrics = ['rmse', 'mae', 'r2', 'da', 'sharpe'] as const
    const ranges: Record<string, { min: number; max: number }> = {}
    metrics.forEach(m => {
      const values = metricsTableData.map(d => d[m]).filter(v => v !== undefined && v !== null) as number[]
      ranges[m] = { min: Math.min(...values), max: Math.max(...values) }
    })

    // Select up to 3 models for radar (first, last, and a PINN)
    const radarModels = metricsTableData.slice(0, 3)

    const normalizeMetric = (value: number, metric: string, invert = false) => {
      const { min, max } = ranges[metric]
      if (max === min) return 50
      const normalized = ((value - min) / (max - min)) * 100
      return invert ? 100 - normalized : normalized
    }

    return [
      {
        metric: "RMSE",
        ...Object.fromEntries(radarModels.map(m => [m.model, normalizeMetric(m.rmse, 'rmse', true)]))
      },
      {
        metric: "MAE",
        ...Object.fromEntries(radarModels.map(m => [m.model, normalizeMetric(m.mae, 'mae', true)]))
      },
      {
        metric: "R²",
        ...Object.fromEntries(radarModels.map(m => [m.model, normalizeMetric(m.r2, 'r2')]))
      },
      {
        metric: "Dir. Acc.",
        ...Object.fromEntries(radarModels.map(m => [m.model, normalizeMetric(m.da, 'da')]))
      },
      {
        metric: "Sharpe",
        ...Object.fromEntries(radarModels.map(m => [m.model, normalizeMetric(m.sharpe, 'sharpe')]))
      },
    ]
  }, [metricsTableData])

  const radarModels = metricsTableData.slice(0, 3)
  const radarColors = ["#6366f1", "#10b981", "#f59e0b"]

  const isLoading = modelsLoading || metricsLoading

  if (isLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  const toggleModel = (key: string) => {
    setSelectedModels((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Model Comparison</h1>
        <p className="text-muted-foreground">
          Compare performance across all models
        </p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Models to Compare</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {modelsData?.models.map((model) => (
              <Button
                key={model.model_key}
                variant={selectedModels.includes(model.model_key) ? "default" : "outline"}
                size="sm"
                onClick={() => toggleModel(model.model_key)}
              >
                {model.display_name}
                {model.is_pinn && (
                  <Brain className="ml-1 h-3 w-3" />
                )}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Metrics</CardTitle>
          <CardDescription>
            Comparison of key metrics across models
          </CardDescription>
        </CardHeader>
        <CardContent>
          {metricsLoading ? (
            <div className="flex h-48 items-center justify-center">
              <LoadingSpinner size="md" />
            </div>
          ) : metricsTableData.length === 0 ? (
            <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="mb-2 h-8 w-8" />
              <p>No metrics available. Select models to compare.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="px-4 py-3 text-left font-medium">Model</th>
                    <th className="px-4 py-3 text-right font-medium">RMSE</th>
                    <th className="px-4 py-3 text-right font-medium">MAE</th>
                    <th className="px-4 py-3 text-right font-medium">R²</th>
                    <th className="px-4 py-3 text-right font-medium">Dir. Acc.</th>
                    <th className="px-4 py-3 text-right font-medium">Sharpe</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsTableData.map((m) => (
                    <tr key={m.model_key} className="border-b hover:bg-muted/50">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {m.model}
                          {m.is_pinn && (
                            <Badge variant="secondary">PINN</Badge>
                          )}
                          {m.model_key === bestModelKey && (
                            <Badge variant="success">Best</Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {m.rmse.toFixed(4)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {m.mae.toFixed(4)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {m.r2.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {m.da.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {m.sharpe.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>RMSE Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={metricsTableData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="model" tick={{ fontSize: 12 }} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    borderColor: "hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="rmse" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Multi-Metric Radar</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid className="stroke-muted" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis tick={{ fontSize: 10 }} />
                {radarModels.map((model, index) => (
                  <Radar
                    key={model.model_key}
                    name={model.model}
                    dataKey={model.model}
                    stroke={radarColors[index % radarColors.length]}
                    fill={radarColors[index % radarColors.length]}
                    fillOpacity={0.2}
                  />
                ))}
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Sharpe Ratio Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Sharpe Ratio by Model</CardTitle>
          <CardDescription>Risk-adjusted return comparison</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricsTableData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  borderColor: "hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <Bar
                dataKey="sharpe"
                fill="hsl(var(--primary))"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
