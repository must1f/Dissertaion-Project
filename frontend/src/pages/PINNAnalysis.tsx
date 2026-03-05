import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useModels } from "../hooks/useModels"
import { usePhysicsMetrics, useModelMetrics } from "../hooks/useMetrics"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
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
import { Atom, Brain, Activity, TrendingUp, AlertCircle } from "lucide-react"
import api from "../services/api"

interface TrainingHistoryResponse {
  job_id: string
  model_type: string
  epochs: Array<{
    epoch: number
    train_loss: number
    val_loss: number
    physics_loss?: number
    data_loss?: number
    learning_rate?: number
  }>
  best_epoch: number
  best_val_loss: number
  final_metrics: {
    final_train_loss: number
    final_val_loss: number
    best_val_loss: number
  }
}

interface TrainingRunSummary {
  job_id: string
  model_type: string
  status: string
  total_epochs: number
  best_val_loss?: number
}

export default function PINNAnalysis() {
  const [selectedModel, setSelectedModel] = useState("pinn_gbm_ou")
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const { data: modelsData, isLoading: modelsLoading } = useModels()
  const { data: physicsMetrics, isLoading: physicsLoading, error: physicsError } = usePhysicsMetrics(selectedModel)
  const { data: modelMetrics } = useModelMetrics(selectedModel)

  // Fetch training history for the selected model
  const { data: trainingRuns } = useQuery({
    queryKey: ["trainingRuns"],
    queryFn: async () => {
      const response = await api.get<{ runs: TrainingRunSummary[]; total: number }>("/api/training/history")
      return response.data
    },
  })

  // Fetch detailed training history when a job is selected
  const { data: trainingHistory, isLoading: historyLoading } = useQuery({
    queryKey: ["trainingHistory", selectedJobId],
    queryFn: async () => {
      if (!selectedJobId) return null
      const response = await api.get<TrainingHistoryResponse>(`/api/training/history/${selectedJobId}`)
      return response.data
    },
    enabled: !!selectedJobId,
  })

  const pinnModels = modelsData?.models.filter((m) => m.is_pinn) || []

  // Auto-select the latest training run for the selected model
  const relevantRuns = trainingRuns?.runs.filter(
    (run) => run.model_type === selectedModel && run.status === "completed"
  ) || []

  if (relevantRuns.length > 0 && !selectedJobId) {
    setSelectedJobId(relevantRuns[0].job_id)
  }

  if (modelsLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  // Transform training history for chart
  const chartData = trainingHistory?.epochs.map((epoch) => ({
    epoch: epoch.epoch,
    train_loss: epoch.train_loss,
    val_loss: epoch.val_loss,
    physics_loss: epoch.physics_loss,
    data_loss: epoch.data_loss,
  })) || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">PINN Analysis</h1>
        <p className="text-muted-foreground">
          Physics-Informed Neural Network Dashboard
        </p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select PINN Model</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {pinnModels.length > 0 ? (
              pinnModels.map((model) => (
                <Button
                  key={model.model_key}
                  variant={selectedModel === model.model_key ? "default" : "outline"}
                  onClick={() => {
                    setSelectedModel(model.model_key)
                    setSelectedJobId(null)
                  }}
                >
                  {model.display_name}
                  {model.status === "trained" && (
                    <Badge variant="secondary" className="ml-2">
                      Trained
                    </Badge>
                  )}
                </Button>
              ))
            ) : (
              <p className="text-muted-foreground">No PINN models available.</p>
            )}
          </div>

          {relevantRuns.length > 1 && (
            <div className="mt-4">
              <label className="mb-2 block text-sm font-medium">Training Run</label>
              <select
                value={selectedJobId || ""}
                onChange={(e) => setSelectedJobId(e.target.value)}
                className="w-full max-w-xs rounded-md border border-input bg-background px-3 py-2"
              >
                {relevantRuns.map((run) => (
                  <option key={run.job_id} value={run.job_id}>
                    {run.job_id} - {run.total_epochs} epochs (Best: {run.best_val_loss?.toFixed(4) || "--"})
                  </option>
                ))}
              </select>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Physics Parameters */}
      <div>
        <h2 className="mb-4 text-xl font-semibold flex items-center gap-2">
          <Atom className="h-5 w-5" />
          Learned Physics Parameters
        </h2>
        {physicsLoading ? (
          <div className="flex h-32 items-center justify-center">
            <LoadingSpinner />
          </div>
        ) : physicsError ? (
          <Card className="border-destructive">
            <CardContent className="flex items-center gap-2 pt-6">
              <AlertCircle className="h-5 w-5 text-destructive" />
              <span className="text-destructive">
                Failed to load physics metrics. The model may not be trained yet.
              </span>
            </CardContent>
          </Card>
        ) : (
          <MetricGrid columns={5}>
            <MetricCard
              title="θ (Theta)"
              value={physicsMetrics?.theta?.toFixed(4) || "--"}
              subtitle="OU Mean Reversion"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="γ (Gamma)"
              value={physicsMetrics?.gamma?.toFixed(4) || "--"}
              subtitle="Langevin Friction"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="T (Temperature)"
              value={physicsMetrics?.temperature?.toFixed(4) || "--"}
              subtitle="Langevin Temperature"
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="μ (Mu)"
              value={physicsMetrics?.mu?.toFixed(4) || "--"}
              subtitle="GBM Drift"
              icon={<TrendingUp className="h-4 w-4" />}
            />
            <MetricCard
              title="σ (Sigma)"
              value={physicsMetrics?.sigma?.toFixed(4) || "--"}
              subtitle="Volatility"
              icon={<TrendingUp className="h-4 w-4" />}
            />
          </MetricGrid>
        )}
      </div>

      {/* Training Loss Curves */}
      <Card>
        <CardHeader>
          <CardTitle>Training Loss Curves</CardTitle>
          <CardDescription>
            {trainingHistory
              ? `Training history for job ${selectedJobId} (Best epoch: ${trainingHistory.best_epoch})`
              : "Select a training run to view loss curves"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {historyLoading ? (
            <div className="flex h-96 items-center justify-center">
              <LoadingSpinner />
            </div>
          ) : chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => v != null ? v.toFixed(3) : "--"}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    borderColor: "hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Train Loss"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  name="Validation Loss"
                  stroke="hsl(var(--destructive))"
                  strokeWidth={2}
                  dot={false}
                />
                {chartData.some(d => d.physics_loss !== undefined) && (
                  <Line
                    type="monotone"
                    dataKey="physics_loss"
                    name="Physics Loss"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                  />
                )}
                {chartData.some(d => d.data_loss !== undefined) && (
                  <Line
                    type="monotone"
                    dataKey="data_loss"
                    name="Data Loss"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-96 items-center justify-center text-muted-foreground">
              {relevantRuns.length === 0
                ? "No completed training runs found for this model. Train the model first."
                : "Select a training run to view loss curves."}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Physics Constraints & Model Architecture */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Physics Loss Components</CardTitle>
            <CardDescription>
              Breakdown of physics constraint losses
            </CardDescription>
          </CardHeader>
          <CardContent>
            {modelMetrics?.physics_metrics ? (
              <div className="space-y-4">
                {Object.entries(modelMetrics.physics_metrics)
                  .filter(([key, value]) => key.endsWith('_loss') && value != null)
                  .map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="capitalize">{key.replace(/_/g, " ")}</span>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-32 rounded-full bg-muted">
                        <div
                          className="h-2 rounded-full bg-primary"
                          style={{ width: `${Math.min((value as number) * 1000, 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-sm">{(value as number).toFixed(4)}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                Physics loss components not available. Train the model to see breakdown.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Architecture</CardTitle>
            <CardDescription>PINN network structure</CardDescription>
          </CardHeader>
          <CardContent>
            {modelMetrics?.physics_metrics ? (
              <div className="space-y-3 text-sm">
                {Object.entries(modelMetrics.physics_metrics)
                  .filter(([key, value]) => !key.endsWith('_loss') && value != null)
                  .map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-muted-foreground capitalize">
                      {key.replace(/_/g, " ")}
                    </span>
                    <span className="font-mono">{typeof value === 'number' ? value.toFixed(4) : String(value)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Input Dimension</span>
                  <span className="font-mono">5</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Hidden Dimension</span>
                  <span className="font-mono">128</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Number of Layers</span>
                  <span className="font-mono">3</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dropout Rate</span>
                  <span className="font-mono">0.2</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Physics Weight</span>
                  <span className="font-mono">0.1</span>
                </div>
                <p className="mt-4 text-xs text-muted-foreground">
                  Train the model to see actual architecture details.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
