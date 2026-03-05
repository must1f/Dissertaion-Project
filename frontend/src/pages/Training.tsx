import { useState, useEffect } from "react"
import { useTrainingStore } from "../stores/trainingStore"
import { useModelTypes } from "../hooks/useModels"
import { useTrainingWebSocket } from "../hooks/useWebSocket"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { Progress } from "../components/ui/progress"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { TickerSelect } from "../components/common/TickerSelect"
import { DEFAULT_TICKER } from "../config/tickers"
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
import { Play, Square, AlertCircle, Wifi, WifiOff } from "lucide-react"
import api from "../services/api"

interface TrainingUpdate {
  type?: string
  job_id?: string
  epoch?: number
  total_epochs?: number
  train_loss?: number
  val_loss?: number
  best_val_loss?: number
  progress_percent?: number
  status?: string
  message?: string
  final_metrics?: {
    best_val_loss?: number
    total_epochs?: number
  }
}

export default function Training() {
  const { data: modelTypes } = useModelTypes()
  const { jobs, addJob, handleWsUpdate, getActiveJobs } = useTrainingStore()

  const [config, setConfig] = useState({
    modelType: "pinn_gbm_ou",
    ticker: DEFAULT_TICKER,  // Fixed to S&P 500
    epochs: 50,
    batchSize: 64,
    learningRate: 0.002,
    sequenceLength: 60,
    hiddenDim: 128,
    numLayers: 2,
    dropout: 0.2,
    enablePhysics: true,
    physicsWeight: 0.1,
  })

  const [isStarting, setIsStarting] = useState(false)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [trainingHistory, setTrainingHistory] = useState<Array<{
    epoch: number
    train_loss: number
    val_loss: number
  }>>([])

  const activeJobs = getActiveJobs()
  const currentJob = activeJobId ? jobs[activeJobId] : null

  // WebSocket connection for real-time training updates
  const { isConnected, updates, clearUpdates } = useTrainingWebSocket(activeJobId)

  // Process WebSocket updates
  useEffect(() => {
    if (updates.length === 0 || !activeJobId) return

    const latestUpdate = updates[updates.length - 1] as TrainingUpdate

    if (latestUpdate.type === "training_update") {
      // Update training store
      handleWsUpdate({
        job_id: activeJobId,
        epoch: latestUpdate.epoch,
        total_epochs: latestUpdate.total_epochs,
        train_loss: latestUpdate.train_loss,
        val_loss: latestUpdate.val_loss,
        best_val_loss: latestUpdate.best_val_loss,
        progress_percent: latestUpdate.progress_percent,
      })

      // Update training history for chart
      if (latestUpdate.epoch && latestUpdate.train_loss !== undefined && latestUpdate.val_loss !== undefined) {
        setTrainingHistory((prev) => {
          // Avoid duplicates
          if (prev.some(h => h.epoch === latestUpdate.epoch)) return prev
          return [
            ...prev,
            {
              epoch: latestUpdate.epoch!,
              train_loss: latestUpdate.train_loss!,
              val_loss: latestUpdate.val_loss!,
            },
          ]
        })
      }
    } else if (latestUpdate.type === "training_complete") {
      handleWsUpdate({
        job_id: activeJobId,
        status: latestUpdate.status,
      })
    } else if (latestUpdate.type === "error") {
      setError(latestUpdate.message || "Training error occurred")
      handleWsUpdate({
        job_id: activeJobId,
        status: "failed",
      })
    }
  }, [updates, activeJobId, handleWsUpdate])

  const handleStartTraining = async () => {
    setIsStarting(true)
    setError(null)
    setTrainingHistory([])
    clearUpdates()

    try {
      const response = await api.post("/api/training/start", {
        model_type: config.modelType,
        ticker: config.ticker,
        epochs: config.epochs,
        batch_size: config.batchSize,
        learning_rate: config.learningRate,
        sequence_length: config.sequenceLength,
        hidden_dim: config.hiddenDim,
        num_layers: config.numLayers,
        dropout: config.dropout,
        enable_physics: config.enablePhysics,
        physics_weight: config.physicsWeight,
      })

      const jobId = response.data.job_id
      setActiveJobId(jobId)

      addJob({
        jobId,
        modelType: config.modelType,
        ticker: config.ticker,
        status: "running",
        currentEpoch: 0,
        totalEpochs: config.epochs,
        progressPercent: 0,
        trainLoss: null,
        valLoss: null,
        bestValLoss: null,
        startedAt: new Date(),
        history: {
          trainLoss: [],
          valLoss: [],
          learningRate: [],
        },
      })
    } catch (err: any) {
      console.error("Failed to start training:", err)
      setError(err.response?.data?.detail || err.message || "Failed to start training")
    } finally {
      setIsStarting(false)
    }
  }

  const handleStopTraining = async () => {
    if (activeJobId) {
      try {
        await api.post(`/api/training/stop/${activeJobId}`)
        handleWsUpdate({
          job_id: activeJobId,
          status: "stopped",
        })
      } catch (err: any) {
        console.error("Failed to stop training:", err)
        setError(err.response?.data?.detail || err.message || "Failed to stop training")
      }
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Training</h1>
          <p className="text-muted-foreground">
            Train models with real-time progress updates
          </p>
        </div>
        {activeJobId && (
          <Badge variant={isConnected ? "default" : "secondary"} className="flex items-center gap-1">
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3" />
                WebSocket Connected
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3" />
                Connecting...
              </>
            )}
          </Badge>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 pt-6">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-destructive">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setError(null)}
              className="ml-auto"
            >
              Dismiss
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Training Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Model Type</label>
              <select
                value={config.modelType}
                onChange={(e) => setConfig({ ...config, modelType: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2"
                disabled={currentJob?.status === "running"}
              >
                <optgroup label="Baseline">
                  <option value="lstm">LSTM</option>
                  <option value="gru">GRU</option>
                  <option value="bilstm">BiLSTM</option>
                  <option value="attention_lstm">Attention LSTM</option>
                  <option value="transformer">Transformer</option>
                </optgroup>
                <optgroup label="PINN">
                  <option value="pinn_baseline">PINN Baseline</option>
                  <option value="pinn_gbm">PINN GBM</option>
                  <option value="pinn_ou">PINN OU</option>
                  <option value="pinn_gbm_ou">PINN GBM+OU</option>
                  <option value="pinn_global">PINN Global</option>
                </optgroup>
                <optgroup label="Advanced">
                  <option value="stacked_pinn">Stacked PINN</option>
                  <option value="residual_pinn">Residual PINN</option>
                </optgroup>
              </select>
            </div>
            <div>
              <TickerSelect
                showLabel
                label="Data Source"
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Epochs</label>
              <Input
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })}
                disabled={currentJob?.status === "running"}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Batch Size</label>
              <Input
                type="number"
                value={config.batchSize}
                onChange={(e) => setConfig({ ...config, batchSize: Number(e.target.value) })}
                disabled={currentJob?.status === "running"}
              />
            </div>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Learning Rate</label>
              <Input
                type="number"
                value={config.learningRate}
                onChange={(e) => setConfig({ ...config, learningRate: Number(e.target.value) })}
                step={0.0001}
                disabled={currentJob?.status === "running"}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Hidden Dim</label>
              <Input
                type="number"
                value={config.hiddenDim}
                onChange={(e) => setConfig({ ...config, hiddenDim: Number(e.target.value) })}
                disabled={currentJob?.status === "running"}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Num Layers</label>
              <Input
                type="number"
                value={config.numLayers}
                onChange={(e) => setConfig({ ...config, numLayers: Number(e.target.value) })}
                disabled={currentJob?.status === "running"}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Dropout</label>
              <Input
                type="number"
                value={config.dropout}
                onChange={(e) => setConfig({ ...config, dropout: Number(e.target.value) })}
                step={0.05}
                disabled={currentJob?.status === "running"}
              />
            </div>
          </div>

          {config.modelType.includes("pinn") && (
            <div className="mt-4 grid gap-4 md:grid-cols-4">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="enablePhysics"
                  checked={config.enablePhysics}
                  onChange={(e) => setConfig({ ...config, enablePhysics: e.target.checked })}
                  className="h-4 w-4"
                  disabled={currentJob?.status === "running"}
                />
                <label htmlFor="enablePhysics" className="text-sm font-medium">
                  Enable Physics Loss
                </label>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Physics Weight</label>
                <Input
                  type="number"
                  value={config.physicsWeight}
                  onChange={(e) => setConfig({ ...config, physicsWeight: Number(e.target.value) })}
                  step={0.01}
                  disabled={!config.enablePhysics || currentJob?.status === "running"}
                />
              </div>
            </div>
          )}

          <div className="mt-6 flex gap-2">
            <Button
              onClick={handleStartTraining}
              disabled={isStarting || currentJob?.status === "running"}
            >
              {isStarting ? (
                <LoadingSpinner size="sm" className="mr-2" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              Start Training
            </Button>
            <Button
              variant="destructive"
              onClick={handleStopTraining}
              disabled={currentJob?.status !== "running"}
            >
              <Square className="mr-2 h-4 w-4" />
              Stop
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Training Progress */}
      {currentJob && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Training Progress</CardTitle>
              <Badge
                variant={
                  currentJob.status === "running"
                    ? "default"
                    : currentJob.status === "completed"
                    ? "success"
                    : currentJob.status === "failed"
                    ? "destructive"
                    : "secondary"
                }
              >
                {currentJob.status}
              </Badge>
            </div>
            <CardDescription>
              {currentJob.modelType} on {currentJob.ticker}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="mb-2 flex justify-between text-sm">
                  <span>
                    Epoch {currentJob.currentEpoch} / {currentJob.totalEpochs}
                  </span>
                  <span>{currentJob.progressPercent.toFixed(1)}%</span>
                </div>
                <Progress value={currentJob.progressPercent} />
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Train Loss</div>
                  <div className="text-2xl font-bold">
                    {currentJob.trainLoss?.toFixed(4) || "--"}
                  </div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Val Loss</div>
                  <div className="text-2xl font-bold">
                    {currentJob.valLoss?.toFixed(4) || "--"}
                  </div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Best Val Loss</div>
                  <div className="text-2xl font-bold text-green-500">
                    {currentJob.bestValLoss?.toFixed(4) || "--"}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Training Chart */}
      {trainingHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Loss Curves</CardTitle>
            <CardDescription>Real-time training and validation loss</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(3)} />
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
                  name="Val Loss"
                  stroke="hsl(var(--destructive))"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Active Jobs */}
      {activeJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Active Training Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {activeJobs.map((job) => (
                <div
                  key={job.jobId}
                  className="flex items-center justify-between rounded-lg border p-4"
                >
                  <div>
                    <div className="font-medium">{job.modelType}</div>
                    <div className="text-sm text-muted-foreground">
                      {job.ticker} - Epoch {job.currentEpoch}/{job.totalEpochs}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <Progress value={job.progressPercent} className="w-32" />
                    <Badge>{job.status}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
