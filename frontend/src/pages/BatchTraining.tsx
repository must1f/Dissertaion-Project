import { useState, useEffect, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { Progress } from "../components/ui/progress"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
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
import {
  Play,
  Square,
  AlertCircle,
  Wifi,
  WifiOff,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  ChevronDown,
  ChevronUp,
} from "lucide-react"
import api from "../services/api"

// Available models with their configurations
const AVAILABLE_MODELS = {
  // Baseline models
  lstm: { name: "LSTM", type: "baseline", description: "Long Short-Term Memory network" },
  gru: { name: "GRU", type: "baseline", description: "Gated Recurrent Unit network" },
  bilstm: { name: "BiLSTM", type: "baseline", description: "Bidirectional LSTM" },
  attention_lstm: { name: "Attention LSTM", type: "baseline", description: "LSTM with attention mechanism" },
  transformer: { name: "Transformer", type: "baseline", description: "Multi-head attention transformer" },
  // PINN variants (basic)
  pinn_baseline: { name: "PINN Baseline", type: "pinn", description: "Pure data-driven (no physics)" },
  pinn_gbm: { name: "PINN GBM", type: "pinn", description: "Geometric Brownian Motion constraint" },
  pinn_ou: { name: "PINN OU", type: "pinn", description: "Ornstein-Uhlenbeck mean-reversion" },
  pinn_black_scholes: { name: "PINN Black-Scholes", type: "pinn", description: "No-arbitrage PDE constraint" },
  pinn_gbm_ou: { name: "PINN GBM+OU", type: "pinn", description: "Combined trend + mean-reversion" },
  pinn_global: { name: "PINN Global", type: "pinn", description: "All physics constraints combined" },
  // Advanced PINN architectures
  stacked_pinn: { name: "StackedPINN", type: "advanced", description: "Physics encoder + parallel LSTM/GRU" },
  residual_pinn: { name: "ResidualPINN", type: "advanced", description: "Base LSTM + physics correction" },
}

interface ModelConfig {
  model_key: string
  enabled: boolean
}

interface ModelStatus {
  model_key: string
  model_name: string
  model_type: string
  status: string
  current_epoch: number
  total_epochs: number
  // Batch-level progress within each epoch
  current_batch?: number
  total_batches?: number
  batch_loss?: number | null
  train_loss: number | null
  val_loss: number | null
  best_val_loss: number | null
  data_loss: number | null
  physics_loss: number | null
  progress_percent: number
  // Validation metrics
  val_rmse?: number | null
  val_mae?: number | null
  val_mape?: number | null
  val_r2?: number | null
  val_directional_accuracy?: number | null
}

interface TrainingHistory {
  epoch: number
  train_loss: number
  val_loss: number
  data_loss?: number
  physics_loss?: number
}

interface BatchProgress {
  batch_id: string
  status: string
  current_model: string | null
  overall_progress: number
  completed_models: number
  failed_models: number
  total_models: number
  models: ModelStatus[]
}

export default function BatchTraining() {
  // Global hyperparameters (research-grade defaults)
  const [config, setConfig] = useState({
    epochs: 50,              // Shorter by default for quicker web runs
    batchSize: 256,          // Very large batch to minimize batches/epoch
    learningRate: 0.002,     // Slightly higher LR for faster convergence
    sequenceLength: 120,     // Shorter lookback to cut compute
    hiddenDim: 512,
    numLayers: 4,
    dropout: 0.1,            // Lower dropout for deep models
    gradientClipNorm: 1.0,
    schedulerPatience: 5,    // Faster LR reductions
    earlyStoppingPatience: 10, // Re-enable early stopping for web runs
    researchMode: false,     // Default to faster, early-stopping-enabled runs
    forceRefresh: true,      // Force fresh 10-year data
  })

  // Model selection
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>(
    Object.keys(AVAILABLE_MODELS).map(key => ({
      model_key: key,
      enabled: true,
    }))
  )

  // Training state
  const [batchId, setBatchId] = useState<string | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null)
  const [trainingHistories, setTrainingHistories] = useState<Record<string, TrainingHistory[]>>({})

  // UI state
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())
  const [activeTab, setActiveTab] = useState<"config" | "progress" | "results">("config")

  // WebSocket connection with auto-reconnect
  useEffect(() => {
    if (!batchId) return

    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout | null = null
    let reconnectAttempts = 0
    const maxReconnectAttempts = 10
    const reconnectInterval = 2000
    let isCleaningUp = false

    const connect = () => {
      if (isCleaningUp) return

      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/batch-training/${batchId}`
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        setIsConnected(true)
        reconnectAttempts = 0 // Reset on successful connection
        console.log('[WS] Connected to batch training WebSocket')
      }

      ws.onclose = () => {
        setIsConnected(false)
        // Auto-reconnect if training is still running
        if (!isCleaningUp && reconnectAttempts < maxReconnectAttempts) {
          console.log(`[WS] Connection closed, reconnecting in ${reconnectInterval}ms (attempt ${reconnectAttempts + 1})`)
          reconnectTimeout = setTimeout(() => {
            reconnectAttempts++
            connect()
          }, reconnectInterval)
        }
      }

      ws.onerror = () => {
        setIsConnected(false)
      }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === "batch_progress" || data.type === "batch_status") {
          setBatchProgress({
            batch_id: data.batch_id,
            status: data.status,
            current_model: data.current_model,
            overall_progress: data.overall_progress,
            completed_models: data.completed_models,
            failed_models: data.failed_models || 0,
            total_models: data.total_models,
            models: data.models,
          })
        }

        if (data.type === "batch_training_update") {
          // Update training history for the current model (only on new epochs)
          setTrainingHistories(prev => {
            const modelKey = data.model_key
            const existing = prev[modelKey] || []

            // Avoid duplicates
            if (existing.some(h => h.epoch === data.epoch)) {
              return prev
            }

            return {
              ...prev,
              [modelKey]: [
                ...existing,
                {
                  epoch: data.epoch,
                  train_loss: data.train_loss,
                  val_loss: data.val_loss,
                  data_loss: data.data_loss,
                  physics_loss: data.physics_loss,
                },
              ],
            }
          })

          // Also update batch-level progress for real-time display
          setBatchProgress(prev => {
            if (!prev) return null
            return {
              ...prev,
              overall_progress: data.overall_progress,
              completed_models: data.completed_models,
              models: prev.models?.map(m =>
                m.model_key === data.model_key
                  ? {
                      ...m,
                      current_epoch: data.epoch,
                      current_batch: data.current_batch,
                      total_batches: data.total_batches,
                      batch_loss: data.batch_loss,
                      train_loss: data.train_loss,
                      val_loss: data.val_loss,
                      best_val_loss: data.best_val_loss,
                      data_loss: data.data_loss,
                      physics_loss: data.physics_loss,
                    }
                  : m
              ),
            }
          })
        }

        if (data.type === "batch_training_complete") {
          setBatchProgress(prev => prev ? {
            ...prev,
            status: data.status,
            completed_models: data.completed_models,
            failed_models: data.failed_models,
          } : null)
        }

        if (data.type === "error") {
          setError(data.message)
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e)
      }
    }
    } // end of connect()

    // Initial connection
    connect()

    return () => {
      isCleaningUp = true
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
      if (ws) {
        ws.close()
      }
    }
  }, [batchId])

  const toggleModel = (modelKey: string) => {
    setModelConfigs(prev =>
      prev.map(m =>
        m.model_key === modelKey ? { ...m, enabled: !m.enabled } : m
      )
    )
  }

  const selectAll = () => {
    setModelConfigs(prev => prev.map(m => ({ ...m, enabled: true })))
  }

  const deselectAll = () => {
    setModelConfigs(prev => prev.map(m => ({ ...m, enabled: false })))
  }

  const selectByType = (type: string) => {
    setModelConfigs(prev =>
      prev.map(m => ({
        ...m,
        enabled: AVAILABLE_MODELS[m.model_key as keyof typeof AVAILABLE_MODELS].type === type,
      }))
    )
  }

  const handleStartTraining = async () => {
    const enabledModels = modelConfigs.filter(m => m.enabled)

    if (enabledModels.length === 0) {
      setError("Please select at least one model to train")
      return
    }

    setIsStarting(true)
    setError(null)
    setTrainingHistories({})

    try {
      const response = await api.post("/api/training/batch/start", {
        models: enabledModels,
        ticker: DEFAULT_TICKER,
        epochs: config.epochs,
        batch_size: config.batchSize,
        learning_rate: config.learningRate,
        sequence_length: config.sequenceLength,
        hidden_dim: config.hiddenDim,
        num_layers: config.numLayers,
        dropout: config.dropout,
        gradient_clip_norm: config.gradientClipNorm,
        scheduler_patience: config.schedulerPatience,
        early_stopping_patience: config.earlyStoppingPatience,
        research_mode: config.researchMode,
        force_refresh: config.forceRefresh,
        enable_physics: true,
      })

      setBatchId(response.data.batch_id)
      setActiveTab("progress")
    } catch (err: any) {
      console.error("Failed to start batch training:", err)
      setError(err.response?.data?.detail || err.message || "Failed to start batch training")
    } finally {
      setIsStarting(false)
    }
  }

  const handleStopTraining = async () => {
    if (!batchId) return

    try {
      await api.post(`/api/training/batch/stop/${batchId}`)
    } catch (err: any) {
      console.error("Failed to stop batch training:", err)
      setError(err.response?.data?.detail || err.message || "Failed to stop batch training")
    }
  }

  const toggleModelExpand = (modelKey: string) => {
    setExpandedModels(prev => {
      const next = new Set(prev)
      if (next.has(modelKey)) {
        next.delete(modelKey)
      } else {
        next.add(modelKey)
      }
      return next
    })
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case "running":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "stopped":
        return <Square className="h-4 w-4 text-yellow-500" />
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "completed":
        return "success"
      case "running":
        return "default"
      case "failed":
        return "destructive"
      case "stopped":
        return "secondary"
      default:
        return "outline"
    }
  }

  const enabledCount = modelConfigs.filter(m => m.enabled).length
  const baselineCount = modelConfigs.filter(
    m => m.enabled && AVAILABLE_MODELS[m.model_key as keyof typeof AVAILABLE_MODELS].type === "baseline"
  ).length
  const pinnCount = modelConfigs.filter(
    m => m.enabled && AVAILABLE_MODELS[m.model_key as keyof typeof AVAILABLE_MODELS].type === "pinn"
  ).length
  const advancedCount = modelConfigs.filter(
    m => m.enabled && AVAILABLE_MODELS[m.model_key as keyof typeof AVAILABLE_MODELS].type === "advanced"
  ).length

  const isTraining = batchProgress?.status === "running"

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Batch Training</h1>
          <p className="text-muted-foreground">
            Train multiple models with configurable hyperparameters
          </p>
        </div>
        {batchId && (
          <Badge variant={isConnected ? "default" : "secondary"} className="flex items-center gap-1">
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3" />
                Connected
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

      {/* Tabs */}
      <div className="flex gap-2 border-b">
        <button
          onClick={() => setActiveTab("config")}
          className={`px-4 py-2 border-b-2 transition-colors ${
            activeTab === "config"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Configuration
        </button>
        <button
          onClick={() => setActiveTab("progress")}
          className={`px-4 py-2 border-b-2 transition-colors ${
            activeTab === "progress"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Training Progress
          {isTraining && (
            <span className="ml-2 inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          )}
        </button>
        <button
          onClick={() => setActiveTab("results")}
          className={`px-4 py-2 border-b-2 transition-colors ${
            activeTab === "results"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Results
        </button>
      </div>

      {/* Configuration Tab */}
      {activeTab === "config" && (
        <div className="space-y-6">
          {/* Global Hyperparameters */}
          <Card>
            <CardHeader>
              <CardTitle>Research-Grade Hyperparameters</CardTitle>
              <CardDescription>
                Deep model defaults for 10-year training. Early stopping disabled.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                <div>
                  <label className="mb-2 block text-sm font-medium">Epochs</label>
                  <Input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })}
                    min={10}
                    max={500}
                    disabled={isTraining}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Learning Rate</label>
                  <select
                    value={config.learningRate}
                    onChange={(e) => setConfig({ ...config, learningRate: Number(e.target.value) })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2"
                    disabled={isTraining}
                  >
                    <option value={0.0001}>0.0001</option>
                    <option value={0.0005}>0.0005</option>
                    <option value={0.001}>0.001 (research)</option>
                    <option value={0.005}>0.005</option>
                    <option value={0.01}>0.01</option>
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Batch Size</label>
                  <select
                    value={config.batchSize}
                    onChange={(e) => setConfig({ ...config, batchSize: Number(e.target.value) })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2"
                    disabled={isTraining}
                  >
                    <option value={64}>64</option>
                    <option value={128}>128</option>
                    <option value={256}>256 (fewest batches)</option>
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Sequence Length</label>
                  <select
                    value={config.sequenceLength}
                    onChange={(e) => setConfig({ ...config, sequenceLength: Number(e.target.value) })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2"
                    disabled={isTraining}
                  >
                    <option value={60}>60 (short)</option>
                    <option value={120}>120 (default)</option>
                    <option value={180}>180</option>
                    <option value={240}>240</option>
                  </select>
                </div>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-4">
                <div>
                  <label className="mb-2 block text-sm font-medium">Hidden Dimension</label>
                  <select
                    value={config.hiddenDim}
                    onChange={(e) => setConfig({ ...config, hiddenDim: Number(e.target.value) })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2"
                    disabled={isTraining}
                  >
                    <option value={128}>128</option>
                    <option value={256}>256</option>
                    <option value={512}>512 (research)</option>
                    <option value={768}>768</option>
                    <option value={1024}>1024</option>
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Number of Layers</label>
                  <select
                    value={config.numLayers}
                    onChange={(e) => setConfig({ ...config, numLayers: Number(e.target.value) })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2"
                    disabled={isTraining}
                  >
                    <option value={2}>2</option>
                    <option value={3}>3</option>
                    <option value={4}>4 (research)</option>
                    <option value={5}>5</option>
                    <option value={6}>6</option>
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Dropout</label>
                  <select
                    value={config.dropout}
                    onChange={(e) => setConfig({ ...config, dropout: Number(e.target.value) })}
                    className="w-full rounded-md border border-input bg-background px-3 py-2"
                    disabled={isTraining}
                  >
                    <option value={0.05}>0.05</option>
                    <option value={0.1}>0.1 (research)</option>
                    <option value={0.15}>0.15</option>
                    <option value={0.2}>0.2</option>
                    <option value={0.3}>0.3</option>
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Research Mode</label>
                  <div className="flex items-center gap-2 mt-2">
                    <input
                      type="checkbox"
                      checked={config.researchMode}
                      onChange={(e) => setConfig({ ...config, researchMode: e.target.checked })}
                      className="h-4 w-4"
                      disabled={isTraining}
                    />
                    <span className="text-sm text-muted-foreground">No early stopping</span>
                  </div>
                </div>
              </div>

              {/* Research Mode Info Banner */}
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="text-sm text-blue-800 dark:text-blue-200">
                  <strong>Research Mode:</strong> 10-year data, deep models (512 hidden, 4 layers),
                  no early stopping. Expect ~10-30 min per model on CPU.
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Model Selection */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Model Selection</CardTitle>
                  <CardDescription>Select models to include in batch training</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={selectAll} disabled={isTraining}>
                    Select All
                  </Button>
                  <Button variant="outline" size="sm" onClick={deselectAll} disabled={isTraining}>
                    Deselect All
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => selectByType("baseline")} disabled={isTraining}>
                    Baselines Only
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => selectByType("pinn")} disabled={isTraining}>
                    PINNs Only
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => selectByType("advanced")} disabled={isTraining}>
                    Advanced Only
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {/* Baseline Models */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold mb-3">Baseline Models</h4>
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                  {Object.entries(AVAILABLE_MODELS)
                    .filter(([_, info]) => info.type === "baseline")
                    .map(([key, info]) => {
                      const isEnabled = modelConfigs.find(m => m.model_key === key)?.enabled ?? false
                      return (
                        <div
                          key={key}
                          onClick={() => !isTraining && toggleModel(key)}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            isEnabled
                              ? "border-primary bg-primary/5"
                              : "border-muted hover:border-muted-foreground"
                          } ${isTraining ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{info.name}</span>
                            <input
                              type="checkbox"
                              checked={isEnabled}
                              onChange={() => {}}
                              className="h-4 w-4"
                              disabled={isTraining}
                            />
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{info.description}</p>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* PINN Models */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold mb-3">PINN Variants</h4>
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                  {Object.entries(AVAILABLE_MODELS)
                    .filter(([_, info]) => info.type === "pinn")
                    .map(([key, info]) => {
                      const isEnabled = modelConfigs.find(m => m.model_key === key)?.enabled ?? false
                      return (
                        <div
                          key={key}
                          onClick={() => !isTraining && toggleModel(key)}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            isEnabled
                              ? "border-primary bg-primary/5"
                              : "border-muted hover:border-muted-foreground"
                          } ${isTraining ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{info.name}</span>
                            <input
                              type="checkbox"
                              checked={isEnabled}
                              onChange={() => {}}
                              className="h-4 w-4"
                              disabled={isTraining}
                            />
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{info.description}</p>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* Advanced PINN Architectures */}
              <div>
                <h4 className="text-sm font-semibold mb-3">Advanced PINN Architectures</h4>
                <div className="grid gap-3 md:grid-cols-2">
                  {Object.entries(AVAILABLE_MODELS)
                    .filter(([_, info]) => info.type === "advanced")
                    .map(([key, info]) => {
                      const isEnabled = modelConfigs.find(m => m.model_key === key)?.enabled ?? false
                      return (
                        <div
                          key={key}
                          onClick={() => !isTraining && toggleModel(key)}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            isEnabled
                              ? "border-violet-500 bg-violet-500/10"
                              : "border-muted hover:border-muted-foreground"
                          } ${isTraining ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{info.name}</span>
                            <input
                              type="checkbox"
                              checked={isEnabled}
                              onChange={() => {}}
                              className="h-4 w-4"
                              disabled={isTraining}
                            />
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{info.description}</p>
                        </div>
                      )
                    })}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Training Summary & Start */}
          <Card>
            <CardHeader>
              <CardTitle>Training Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4 mb-6">
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Selected Models</div>
                  <div className="text-2xl font-bold">{enabledCount}</div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Total Epochs</div>
                  <div className="text-2xl font-bold">{enabledCount * config.epochs}</div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Base / PINN / Adv</div>
                  <div className="text-2xl font-bold">{baselineCount} / {pinnCount} / {advancedCount}</div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-muted-foreground">Learning Rate</div>
                  <div className="text-2xl font-bold">{config.learningRate}</div>
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={handleStartTraining}
                  disabled={isStarting || isTraining || enabledCount === 0}
                  className="flex-1"
                >
                  {isStarting ? (
                    <LoadingSpinner size="sm" className="mr-2" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  Train All Selected Models ({enabledCount})
                </Button>
                {isTraining && (
                  <Button variant="destructive" onClick={handleStopTraining}>
                    <Square className="mr-2 h-4 w-4" />
                    Stop
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Progress Tab */}
      {activeTab === "progress" && (
        <div className="space-y-6">
          {!batchProgress ? (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground">
                  No training in progress. Go to Configuration tab to start batch training.
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Overall Progress */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Overall Progress</CardTitle>
                    <Badge variant={getStatusBadgeVariant(batchProgress.status) as any}>
                      {batchProgress.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4 mb-4">
                    <div className="rounded-lg border p-4">
                      <div className="text-sm text-muted-foreground">Completed</div>
                      <div className="text-2xl font-bold text-green-500">
                        {batchProgress.completed_models} / {batchProgress.total_models}
                      </div>
                    </div>
                    <div className="rounded-lg border p-4">
                      <div className="text-sm text-muted-foreground">Failed</div>
                      <div className="text-2xl font-bold text-red-500">
                        {batchProgress.failed_models}
                      </div>
                    </div>
                    <div className="rounded-lg border p-4">
                      <div className="text-sm text-muted-foreground">Current Model</div>
                      <div className="text-lg font-medium truncate">
                        {batchProgress.current_model || "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border p-4">
                      <div className="text-sm text-muted-foreground">Progress</div>
                      <div className="text-2xl font-bold">
                        {batchProgress.overall_progress.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <Progress value={batchProgress.overall_progress} />
                </CardContent>
              </Card>

              {/* Individual Model Progress */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Training Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {batchProgress.models.map((model) => (
                      <div
                        key={model.model_key}
                        className="rounded-lg border overflow-hidden"
                      >
                        <div
                          className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/50"
                          onClick={() => toggleModelExpand(model.model_key)}
                        >
                          <div className="flex items-center gap-3">
                            {getStatusIcon(model.status)}
                            <div>
                              <div className="font-medium">{model.model_name}</div>
                              <div className="text-sm text-muted-foreground">
                                Epoch {model.current_epoch} / {model.total_epochs}
                                {model.current_batch !== undefined && model.total_batches ? (
                                  <span className="ml-2 text-xs">
                                    (Batch {model.current_batch}/{model.total_batches})
                                  </span>
                                ) : null}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="text-right">
                              <div className="text-sm">
                                Val Loss: {model.val_loss?.toFixed(4) || "—"}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                Best: {model.best_val_loss?.toFixed(4) || "—"}
                              </div>
                            </div>
                            <Progress value={model.progress_percent} className="w-24" />
                            {expandedModels.has(model.model_key) ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </div>
                        </div>

                        {expandedModels.has(model.model_key) && (
                          <div className="border-t p-4 bg-muted/30">
                            <div className="grid gap-4 md:grid-cols-4 mb-4">
                              <div>
                                <div className="text-xs text-muted-foreground">Train Loss</div>
                                <div className="font-medium">{model.train_loss?.toFixed(4) || "—"}</div>
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Val Loss</div>
                                <div className="font-medium">{model.val_loss?.toFixed(4) || "—"}</div>
                              </div>
                              {model.data_loss !== null && (
                                <div>
                                  <div className="text-xs text-muted-foreground">Data Loss</div>
                                  <div className="font-medium">{model.data_loss?.toFixed(4)}</div>
                                </div>
                              )}
                              {model.physics_loss !== null && (
                                <div>
                                  <div className="text-xs text-muted-foreground">Physics Loss</div>
                                  <div className="font-medium">{model.physics_loss?.toFixed(4)}</div>
                                </div>
                              )}
                              {model.batch_loss !== undefined && model.batch_loss !== null && (
                                <div>
                                  <div className="text-xs text-muted-foreground">Current Batch Loss</div>
                                  <div className="font-medium text-yellow-600">{model.batch_loss?.toFixed(4)}</div>
                                </div>
                              )}
                            </div>

                            {/* Mini loss chart for this model */}
                            {trainingHistories[model.model_key]?.length > 0 && (
                              <ResponsiveContainer width="100%" height={150}>
                                <LineChart data={trainingHistories[model.model_key]}>
                                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                                  <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(3)} />
                                  <Tooltip />
                                  <Line
                                    type="monotone"
                                    dataKey="train_loss"
                                    stroke="hsl(var(--primary))"
                                    strokeWidth={1.5}
                                    dot={false}
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="val_loss"
                                    stroke="hsl(var(--destructive))"
                                    strokeWidth={1.5}
                                    dot={false}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Combined Loss Chart */}
              {Object.keys(trainingHistories).length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Combined Loss Curves</CardTitle>
                    <CardDescription>Validation loss comparison across all models</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis
                          dataKey="epoch"
                          type="number"
                          domain={[1, config.epochs]}
                          tick={{ fontSize: 12 }}
                        />
                        <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(3)} />
                        <Tooltip />
                        <Legend />
                        {Object.entries(trainingHistories).map(([modelKey, history], index) => {
                          const colors = [
                            "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea",
                            "#0891b2", "#c026d3", "#ea580c", "#4f46e5", "#059669"
                          ]
                          const modelInfo = AVAILABLE_MODELS[modelKey as keyof typeof AVAILABLE_MODELS]
                          return (
                            <Line
                              key={modelKey}
                              data={history}
                              type="monotone"
                              dataKey="val_loss"
                              name={modelInfo?.name || modelKey}
                              stroke={colors[index % colors.length]}
                              strokeWidth={2}
                              dot={false}
                            />
                          )
                        })}
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      )}

      {/* Results Tab */}
      {activeTab === "results" && (
        <div className="space-y-6">
          {!batchProgress || batchProgress.status === "pending" || batchProgress.status === "running" ? (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground">
                  {batchProgress?.status === "running"
                    ? "Training in progress. Results will appear here when complete."
                    : "No training results yet. Start batch training first."}
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Results Summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Training Results Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-3 px-2">Model</th>
                          <th className="text-left py-3 px-2">Type</th>
                          <th className="text-left py-3 px-2">Status</th>
                          <th className="text-right py-3 px-2">Best Val Loss</th>
                          <th className="text-right py-3 px-2">Final Val Loss</th>
                          <th className="text-right py-3 px-2">Val RMSE</th>
                          <th className="text-right py-3 px-2">Val MAE</th>
                          <th className="text-right py-3 px-2">Val MAPE</th>
                          <th className="text-right py-3 px-2">Val R²</th>
                          <th className="text-right py-3 px-2">Dir. Acc (%)</th>
                          <th className="text-right py-3 px-2">Epochs</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...batchProgress.models]
                          .sort((a, b) => (a.best_val_loss ?? Infinity) - (b.best_val_loss ?? Infinity))
                          .map((model, index) => (
                            <tr key={model.model_key} className="border-b">
                              <td className="py-3 px-2">
                                <div className="flex items-center gap-2">
                                  {index === 0 && <span className="text-yellow-500">🥇</span>}
                                  {index === 1 && <span className="text-gray-400">🥈</span>}
                                  {index === 2 && <span className="text-amber-600">🥉</span>}
                                  {model.model_name}
                                </div>
                              </td>
                              <td className="py-3 px-2">
                                <Badge variant="outline">{model.model_type}</Badge>
                              </td>
                              <td className="py-3 px-2">
                                <Badge variant={getStatusBadgeVariant(model.status) as any}>
                                  {model.status}
                                </Badge>
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.best_val_loss?.toFixed(4) || "—"}
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.val_loss?.toFixed(4) || "—"}
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.val_rmse?.toFixed(4) || "—"}
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.val_mae?.toFixed(4) || "—"}
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.val_mape?.toFixed(2) || "—"}
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.val_r2 !== null && model.val_r2 !== undefined ? model.val_r2.toFixed(3) : "—"}
                              </td>
                              <td className="text-right py-3 px-2 font-mono">
                                {model.val_directional_accuracy !== null && model.val_directional_accuracy !== undefined
                                  ? model.val_directional_accuracy.toFixed(1)
                                  : "—"}
                              </td>
                              <td className="text-right py-3 px-2">
                                {model.current_epoch}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* Final Comparison Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Best Validation Loss Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      layout="vertical"
                      data={[...batchProgress.models]
                        .filter(m => m.best_val_loss !== null)
                        .sort((a, b) => (a.best_val_loss ?? 0) - (b.best_val_loss ?? 0))
                        .map(m => ({
                          name: m.model_name,
                          value: m.best_val_loss,
                        }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={120} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  )
}
