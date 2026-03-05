import { useQuery } from "@tanstack/react-query"
import { useModels, useTrainedModels } from "../hooks/useModels"
import { useMetricsComparison } from "../hooks/useMetrics"
import { useAppStore } from "../stores/appStore"
import { MetricCard, MetricGrid } from "../components/common/MetricCard"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import {
  Brain,
  TrendingUp,
  Activity,
  BarChart3,
  ArrowRight,
  CheckCircle,
  XCircle,
} from "lucide-react"
import { Link } from "react-router-dom"
import api from "../services/api"

export default function Dashboard() {
  const { selectedTicker } = useAppStore()
  const { data: modelsData, isLoading: modelsLoading } = useModels()
  const { data: trainedModels } = useTrainedModels()

  // Fetch comparison metrics for all trained models
  const trainedModelKeys = trainedModels?.models.map((m) => m.model_key) || []
  const { data: comparisonData, isLoading: metricsLoading } = useMetricsComparison(trainedModelKeys)

  // Find best metrics across all models
  // Use metric_summary from comparison response which has sharpe_ratio keyed by model
  const sharpeValues = comparisonData?.metric_summary?.sharpe_ratio || {}
  const accuracyValues = comparisonData?.metric_summary?.directional_accuracy || {}

  // Helper to find best model for a metric
  const findBest = (values: Record<string, number>): { value: number | null; model: string | null } => {
    const entries = Object.entries(values)
    if (entries.length === 0) return { value: null, model: null }

    let bestKey = entries[0][0]
    let bestVal = entries[0][1]
    for (const [key, val] of entries) {
      if (val > bestVal) {
        bestKey = key
        bestVal = val
      }
    }
    return { value: bestVal, model: bestKey }
  }

  const bestSharpeResult = findBest(sharpeValues)
  const bestAccuracyResult = findBest(accuracyValues)

  const bestMetrics = trainedModelKeys.length > 0 ? {
    bestSharpe: bestSharpeResult.value,
    bestSharpeModel: bestSharpeResult.model,
    bestAccuracy: bestAccuracyResult.value,
    bestAccuracyModel: bestAccuracyResult.model,
  } : null

  if (modelsLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  const totalModels = modelsData?.total || 0
  const trainedCount = modelsData?.trained_count || 0
  const pinnCount = modelsData?.pinn_count || 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            PINN Financial Forecasting Overview
          </p>
        </div>
        <Badge variant="outline" className="text-lg">
          {selectedTicker}
        </Badge>
      </div>

      {/* Key Metrics */}
      <MetricGrid columns={4}>
        <MetricCard
          title="Total Models"
          value={totalModels}
          icon={<Brain className="h-4 w-4 text-muted-foreground" />}
          subtitle={`${pinnCount} PINN models`}
        />
        <MetricCard
          title="Trained Models"
          value={trainedCount}
          icon={<CheckCircle className="h-4 w-4 text-green-500" />}
          trend={trainedCount > 0 ? "up" : "neutral"}
        />
        <MetricCard
          title="Best Sharpe"
          value={
            metricsLoading
              ? "--"
              : bestMetrics?.bestSharpe
              ? bestMetrics.bestSharpe.toFixed(2)
              : trainedCount > 0
              ? "--"
              : "N/A"
          }
          icon={<TrendingUp className="h-4 w-4 text-muted-foreground" />}
          subtitle={bestMetrics?.bestSharpeModel || (trainedCount > 0 ? "Loading..." : "Train models first")}
        />
        <MetricCard
          title="Best Accuracy"
          value={
            metricsLoading
              ? "--"
              : bestMetrics?.bestAccuracy
              ? `${(bestMetrics.bestAccuracy * 100).toFixed(1)}%`
              : trainedCount > 0
              ? "--"
              : "N/A"
          }
          icon={<Activity className="h-4 w-4 text-muted-foreground" />}
          subtitle={bestMetrics?.bestAccuracyModel ? `${bestMetrics.bestAccuracyModel}` : "Directional accuracy"}
        />
      </MetricGrid>

      {/* Quick Actions */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              PINN Analysis
            </CardTitle>
            <CardDescription>
              Analyze physics-informed neural network predictions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/pinn">
              <Button className="w-full">
                View Analysis
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Model Comparison
            </CardTitle>
            <CardDescription>
              Compare performance across all models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/models">
              <Button className="w-full" variant="outline">
                Compare Models
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Predictions
            </CardTitle>
            <CardDescription>
              Generate predictions for {selectedTicker}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/predictions">
              <Button className="w-full" variant="outline">
                Make Predictions
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Models Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Available Models</CardTitle>
          <CardDescription>
            {totalModels} models available, {trainedCount} trained
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {modelsData?.models.slice(0, 6).map((model) => (
              <div
                key={model.model_key}
                className="flex items-center justify-between rounded-lg border p-4"
              >
                <div>
                  <div className="font-medium">{model.display_name}</div>
                  <div className="text-sm text-muted-foreground">
                    {model.model_type}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {model.is_pinn && (
                    <Badge variant="secondary">PINN</Badge>
                  )}
                  {model.status === "trained" ? (
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  ) : (
                    <XCircle className="h-5 w-5 text-muted-foreground" />
                  )}
                </div>
              </div>
            ))}
          </div>
          {totalModels > 6 && (
            <div className="mt-4 text-center">
              <Link to="/models">
                <Button variant="ghost">
                  View all {totalModels} models
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
