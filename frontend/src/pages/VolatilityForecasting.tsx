import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts"
import {
  Activity,
  TrendingUp,
  Brain,
  Settings2,
  Play,
  Info,
  AlertCircle,
  CheckCircle2,
  BarChart3,
} from "lucide-react"
import {
  useVolatilityModels,
  useVolatilityPhysicsConstraints,
  usePrepareVolatilityData,
  useTrainVolatilityModel,
  useVolatilityBacktest,
  useCompareVolatilityModels,
} from "../hooks/useVolatility"
import type {
  VolatilityTrainingResponse,
  StrategyBacktestResponse,
  ModelComparisonResponse,
} from "../types/volatility"

// Model type colors
const modelColors: Record<string, string> = {
  vol_lstm: "#6366f1",
  vol_gru: "#8b5cf6",
  vol_transformer: "#a855f7",
  vol_pinn: "#22c55e",
  heston_pinn: "#10b981",
  stacked_vol_pinn: "#14b8a6",
  rolling: "#f59e0b",
  ewma: "#f97316",
  garch: "#ef4444",
  gjr_garch: "#dc2626",
}

export default function VolatilityForecasting() {
  const [activeTab, setActiveTab] = useState("overview")

  // Data preparation state
  const [ticker, setTicker] = useState("SPY")
  const [startDate, setStartDate] = useState("2015-01-01")
  const [horizon, setHorizon] = useState(5)
  const [seqLength, setSeqLength] = useState(40)
  const [dataReady, setDataReady] = useState(false)

  // Training state
  const [selectedModel, setSelectedModel] = useState("vol_pinn")
  const [epochs, setEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(64)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [enablePhysics, setEnablePhysics] = useState(true)
  const [trainingResult, setTrainingResult] = useState<VolatilityTrainingResponse | null>(null)

  // Backtest state
  const [targetVol, setTargetVol] = useState(0.15)
  const [backtestResult, setBacktestResult] = useState<StrategyBacktestResponse | null>(null)

  // Comparison state
  const [comparisonModels, setComparisonModels] = useState<string[]>([])
  const [comparisonResult, setComparisonResult] = useState<ModelComparisonResponse | null>(null)

  // API hooks
  const { data: modelsData, isLoading: modelsLoading } = useVolatilityModels()
  const { data: physicsData } = useVolatilityPhysicsConstraints()

  const prepareDataMutation = usePrepareVolatilityData()
  const trainModelMutation = useTrainVolatilityModel()
  const backtestMutation = useVolatilityBacktest()
  const compareMutation = useCompareVolatilityModels()

  // Handle data preparation
  const handlePrepareData = () => {
    prepareDataMutation.mutate(
      { ticker, start_date: startDate, horizon, seq_length: seqLength },
      {
        onSuccess: () => setDataReady(true),
      }
    )
  }

  // Handle training
  const handleTrain = () => {
    trainModelMutation.mutate(
      {
        model_type: selectedModel,
        ticker,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        enable_physics: enablePhysics,
      },
      {
        onSuccess: (data) => setTrainingResult(data),
      }
    )
  }

  // Handle backtest
  const handleBacktest = () => {
    backtestMutation.mutate(
      { model_type: selectedModel, target_vol: targetVol },
      {
        onSuccess: (data) => setBacktestResult(data),
      }
    )
  }

  // Handle comparison
  const handleCompare = () => {
    if (comparisonModels.length > 0) {
      compareMutation.mutate(
        { model_types: comparisonModels },
        {
          onSuccess: (data) => setComparisonResult(data),
        }
      )
    }
  }

  // Toggle model for comparison
  const toggleComparisonModel = (modelKey: string) => {
    setComparisonModels(prev =>
      prev.includes(modelKey)
        ? prev.filter(k => k !== modelKey)
        : [...prev, modelKey]
    )
  }

  // Transform training history for chart
  const trainingChartData = useMemo(() => {
    if (!trainingResult?.history) return []
    return trainingResult.history.train_loss.map((loss, i) => ({
      epoch: i + 1,
      train_loss: loss,
      val_loss: trainingResult.history.val_loss[i] || null,
      val_qlike: trainingResult.history.val_qlike[i] || null,
      val_r2: trainingResult.history.val_r2[i] || null,
    }))
  }, [trainingResult])

  // Transform equity curve for chart
  const equityCurveData = useMemo(() => {
    if (!backtestResult?.equity_curve) return []
    return backtestResult.equity_curve.map((value, i) => ({
      day: i,
      equity: value,
      weight: backtestResult.weights[i] || 1,
    }))
  }, [backtestResult])

  // Get model type info
  const getModelType = (key: string): string => {
    return modelsData?.models?.[key]?.type || "unknown"
  }

  if (modelsLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Activity className="h-8 w-8 text-primary" />
          Volatility Forecasting
        </h1>
        <p className="text-muted-foreground">
          Train and evaluate volatility forecasting models with physics-informed constraints
        </p>
      </div>

      {/* Status Banner */}
      {!modelsData?.has_modules && (
        <Card className="border-amber-500/50 bg-amber-500/10">
          <CardContent className="flex items-center gap-3 py-4">
            <AlertCircle className="h-5 w-5 text-amber-500" />
            <div>
              <p className="font-medium text-amber-600 dark:text-amber-400">
                Volatility modules not available
              </p>
              <p className="text-sm text-muted-foreground">
                Install the volatility framework to enable all features
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="backtest">Backtest</TabsTrigger>
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
          <TabsTrigger value="physics">Physics</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Total Models
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{modelsData?.total || 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Neural Networks
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-indigo-500">
                  {modelsData?.by_type.neural || 0}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  PINN Models
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-emerald-500">
                  {modelsData?.by_type.pinn || 0}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Baselines
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-amber-500">
                  {modelsData?.by_type.baseline || 0}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Available Models */}
          <Card>
            <CardHeader>
              <CardTitle>Available Models</CardTitle>
              <CardDescription>
                Volatility forecasting models organized by type
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {modelsData?.models && Object.entries(modelsData.models).map(([key, model]) => (
                  <div
                    key={key}
                    className="rounded-lg border p-4 hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{model.name}</h3>
                      <Badge
                        variant={
                          model.type === "pinn"
                            ? "success"
                            : model.type === "neural"
                            ? "default"
                            : "secondary"
                        }
                      >
                        {model.type}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {model.description}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Training Tab */}
        <TabsContent value="training" className="space-y-6">
          {/* Data Preparation */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5" />
                Data Preparation
              </CardTitle>
              <CardDescription>
                Prepare data for volatility forecasting
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                <div className="space-y-2">
                  <Label htmlFor="ticker">Ticker</Label>
                  <Input
                    id="ticker"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                    placeholder="SPY"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="startDate">Start Date</Label>
                  <Input
                    id="startDate"
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="horizon">Horizon (days)</Label>
                  <Input
                    id="horizon"
                    type="number"
                    value={horizon}
                    onChange={(e) => setHorizon(parseInt(e.target.value) || 5)}
                    min={1}
                    max={60}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="seqLength">Sequence Length</Label>
                  <Input
                    id="seqLength"
                    type="number"
                    value={seqLength}
                    onChange={(e) => setSeqLength(parseInt(e.target.value) || 40)}
                    min={10}
                    max={100}
                  />
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={handlePrepareData}
                    disabled={prepareDataMutation.isPending}
                    className="w-full"
                  >
                    {prepareDataMutation.isPending ? (
                      <LoadingSpinner size="sm" className="mr-2" />
                    ) : (
                      <Play className="mr-2 h-4 w-4" />
                    )}
                    Prepare Data
                  </Button>
                </div>
              </div>
              {dataReady && (
                <div className="mt-4 flex items-center gap-2 text-sm text-emerald-600">
                  <CheckCircle2 className="h-4 w-4" />
                  Data prepared successfully
                </div>
              )}
            </CardContent>
          </Card>

          {/* Model Training */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Model Training
              </CardTitle>
              <CardDescription>
                Configure and train a volatility model
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <div className="space-y-2">
                  <Label>Model Type</Label>
                  <div className="flex flex-wrap gap-2">
                    {modelsData?.models && Object.keys(modelsData.models).map((key) => (
                      <Button
                        key={key}
                        variant={selectedModel === key ? "default" : "outline"}
                        size="sm"
                        onClick={() => setSelectedModel(key)}
                        className="text-xs"
                      >
                        {modelsData.models[key].name}
                      </Button>
                    ))}
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="epochs">Epochs</Label>
                    <Input
                      id="epochs"
                      type="number"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value) || 100)}
                      min={1}
                      max={500}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="batchSize">Batch Size</Label>
                    <Input
                      id="batchSize"
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value) || 64)}
                      min={16}
                      max={256}
                    />
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="learningRate">Learning Rate</Label>
                    <Input
                      id="learningRate"
                      type="number"
                      step="0.0001"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
                      min={0.000001}
                      max={0.1}
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="enablePhysics"
                      checked={enablePhysics}
                      onChange={(e) => setEnablePhysics(e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <Label htmlFor="enablePhysics">Enable Physics Constraints</Label>
                  </div>
                </div>
              </div>
              <div className="mt-4">
                <Button
                  onClick={handleTrain}
                  disabled={trainModelMutation.isPending || !dataReady}
                  className="w-full md:w-auto"
                >
                  {trainModelMutation.isPending ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Training...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Train Model
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Training Results */}
          {trainingResult && (
            <Card>
              <CardHeader>
                <CardTitle>Training Results</CardTitle>
                <CardDescription>
                  {trainingResult.model_type} - {trainingResult.epochs_trained} epochs
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6 lg:grid-cols-2">
                  <div>
                    <h4 className="font-medium mb-4">Training Metrics</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Best Val Loss:</span>
                        <span className="font-mono">{trainingResult.best_val_loss.toFixed(6)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Training Time:</span>
                        <span className="font-mono">{trainingResult.training_time.toFixed(2)}s</span>
                      </div>
                      {trainingResult.physics_params && (
                        <>
                          <h5 className="font-medium mt-4 mb-2">Learned Physics Parameters</h5>
                          {Object.entries(trainingResult.physics_params).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-muted-foreground">{key}:</span>
                              <span className="font-mono">{typeof value === 'number' ? value.toFixed(4) : String(value)}</span>
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-4">Loss Curve</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={trainingChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 12 }} />
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
                          stroke="#6366f1"
                          strokeWidth={2}
                          dot={false}
                          name="Train Loss"
                        />
                        <Line
                          type="monotone"
                          dataKey="val_loss"
                          stroke="#22c55e"
                          strokeWidth={2}
                          dot={false}
                          name="Val Loss"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Backtest Tab */}
        <TabsContent value="backtest" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Volatility Targeting Strategy
              </CardTitle>
              <CardDescription>
                Backtest a volatility targeting strategy using model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="targetVol">Target Volatility (annual)</Label>
                  <Input
                    id="targetVol"
                    type="number"
                    step="0.01"
                    value={targetVol}
                    onChange={(e) => setTargetVol(parseFloat(e.target.value) || 0.15)}
                    min={0.01}
                    max={0.5}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Model</Label>
                  <p className="text-lg font-medium">
                    {modelsData?.models?.[selectedModel]?.name || selectedModel}
                  </p>
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={handleBacktest}
                    disabled={backtestMutation.isPending || !trainingResult}
                    className="w-full"
                  >
                    {backtestMutation.isPending ? (
                      <LoadingSpinner size="sm" className="mr-2" />
                    ) : (
                      <Play className="mr-2 h-4 w-4" />
                    )}
                    Run Backtest
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {backtestResult && (
            <>
              {/* Backtest Metrics */}
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Total Return
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className={`text-2xl font-bold ${backtestResult.total_return >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                      {(backtestResult.total_return * 100).toFixed(2)}%
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Sharpe Ratio
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-indigo-500">
                      {backtestResult.sharpe_ratio.toFixed(2)}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Max Drawdown
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-red-500">
                      {(backtestResult.max_drawdown * 100).toFixed(2)}%
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Vol Tracking Error
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-amber-500">
                      {(backtestResult.vol_tracking_error * 100).toFixed(2)}%
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Equity Curve */}
              <Card>
                <CardHeader>
                  <CardTitle>Equity Curve & Position Weights</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={equityCurveData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                      <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
                      <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          borderColor: "hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Legend />
                      <Area
                        yAxisId="left"
                        type="monotone"
                        dataKey="equity"
                        stroke="#6366f1"
                        fill="#6366f1"
                        fillOpacity={0.3}
                        strokeWidth={2}
                        name="Equity"
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="weight"
                        stroke="#22c55e"
                        strokeWidth={1}
                        dot={false}
                        name="Position Weight"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Additional Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle>Detailed Strategy Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Annual Return:</span>
                        <span className="font-mono">{(backtestResult.annual_return * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Sortino Ratio:</span>
                        <span className="font-mono">{backtestResult.sortino_ratio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Calmar Ratio:</span>
                        <span className="font-mono">{backtestResult.calmar_ratio.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Benchmark Sharpe:</span>
                        <span className="font-mono">{backtestResult.benchmark_sharpe.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Information Ratio:</span>
                        <span className="font-mono">{backtestResult.information_ratio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Leverage:</span>
                        <span className="font-mono">{backtestResult.avg_leverage.toFixed(2)}x</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Turnover:</span>
                        <span className="font-mono">{(backtestResult.turnover * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Realized Vol:</span>
                        <span className="font-mono">{(backtestResult.realized_vol * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Target Vol:</span>
                        <span className="font-mono">{(targetVol * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Comparison Tab */}
        <TabsContent value="comparison" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Model Comparison
              </CardTitle>
              <CardDescription>
                Compare multiple volatility forecasting models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <Label>Select Models to Compare</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {modelsData?.models && Object.entries(modelsData.models).map(([key, model]) => (
                      <Button
                        key={key}
                        variant={comparisonModels.includes(key) ? "default" : "outline"}
                        size="sm"
                        onClick={() => toggleComparisonModel(key)}
                        style={{
                          backgroundColor: comparisonModels.includes(key)
                            ? modelColors[key] || undefined
                            : undefined,
                        }}
                      >
                        {model.name}
                      </Button>
                    ))}
                  </div>
                </div>
                <Button
                  onClick={handleCompare}
                  disabled={compareMutation.isPending || comparisonModels.length === 0}
                >
                  {compareMutation.isPending ? (
                    <LoadingSpinner size="sm" className="mr-2" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  Compare Models
                </Button>
              </div>
            </CardContent>
          </Card>

          {comparisonResult && (
            <>
              {/* Comparison Results */}
              <Card>
                <CardHeader>
                  <CardTitle>Comparison Results</CardTitle>
                  <CardDescription>
                    Best QLIKE: {comparisonResult.best_qlike || 'N/A'} |
                    Best R²: {comparisonResult.best_r2 || 'N/A'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="px-4 py-3 text-left font-medium">Model</th>
                          <th className="px-4 py-3 text-right font-medium">QLIKE</th>
                          <th className="px-4 py-3 text-right font-medium">R²</th>
                          <th className="px-4 py-3 text-right font-medium">RMSE</th>
                          <th className="px-4 py-3 text-right font-medium">MAE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonResult.results.map((result) => (
                          <tr key={result.model} className="border-b hover:bg-muted/50">
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <div
                                  className="w-3 h-3 rounded-full"
                                  style={{ backgroundColor: modelColors[result.model] || '#888' }}
                                />
                                {modelsData?.models?.[result.model]?.name || result.model}
                                {result.model === comparisonResult.best_qlike && (
                                  <Badge variant="success">Best QLIKE</Badge>
                                )}
                                {result.model === comparisonResult.best_r2 && (
                                  <Badge variant="default">Best R²</Badge>
                                )}
                              </div>
                            </td>
                            <td className="px-4 py-3 text-right font-mono">
                              {result.qlike?.toFixed(4) || 'N/A'}
                            </td>
                            <td className="px-4 py-3 text-right font-mono">
                              {result.r2?.toFixed(4) || 'N/A'}
                            </td>
                            <td className="px-4 py-3 text-right font-mono">
                              {result.rmse?.toFixed(6) || 'N/A'}
                            </td>
                            <td className="px-4 py-3 text-right font-mono">
                              {result.mae?.toFixed(6) || 'N/A'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* QLIKE Comparison Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>QLIKE Comparison</CardTitle>
                  <CardDescription>
                    Lower is better - quasi-likelihood loss for variance forecasting
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={comparisonResult.results.map((r) => ({
                        model: modelsData?.models?.[r.model]?.name || r.model,
                        qlike: r.qlike || 0,
                      }))}
                      layout="vertical"
                    >
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis type="number" tick={{ fontSize: 12 }} />
                      <YAxis type="category" dataKey="model" tick={{ fontSize: 12 }} width={120} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          borderColor: "hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="qlike" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Model Confidence Set */}
              {comparisonResult.mcs && (
                <Card>
                  <CardHeader>
                    <CardTitle>Model Confidence Set (MCS)</CardTitle>
                    <CardDescription>
                      Models with statistically equivalent predictive accuracy
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium mb-2">Included Models</h4>
                        <div className="flex flex-wrap gap-2">
                          {comparisonResult.mcs.included_models.map((model) => (
                            <Badge key={model} variant="success">
                              {modelsData?.models?.[model]?.name || model}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      {comparisonResult.mcs.eliminated_order.length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Elimination Order</h4>
                          <p className="text-sm text-muted-foreground">
                            {comparisonResult.mcs.eliminated_order
                              .map((m) => modelsData?.models?.[m]?.name || m)
                              .join(' → ')}
                          </p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        {/* Physics Tab */}
        <TabsContent value="physics" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Physics Constraints for Volatility PINNs
              </CardTitle>
              <CardDescription>
                Mathematical constraints that inform the neural network training
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {physicsData?.constraints && Object.entries(physicsData.constraints).map(([key, constraint]) => (
                  <div
                    key={key}
                    className="rounded-lg border p-4 space-y-2"
                  >
                    <h3 className="font-medium">{constraint.name}</h3>
                    <div className="font-mono text-sm bg-muted/50 rounded px-2 py-1 inline-block">
                      {constraint.equation}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {constraint.description}
                    </p>
                    {constraint.learnable_params && constraint.learnable_params.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {constraint.learnable_params.map((param, i) => (
                          <Badge key={i} variant="secondary" className="text-xs">
                            {param}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              {physicsData?.reference && (
                <p className="mt-4 text-sm text-muted-foreground">
                  Reference: {physicsData.reference}
                </p>
              )}
            </CardContent>
          </Card>

          {/* Motivation */}
          <Card>
            <CardHeader>
              <CardTitle>Why Physics-Informed Volatility Forecasting?</CardTitle>
            </CardHeader>
            <CardContent className="prose dark:prose-invert max-w-none">
              <p>
                Volatility exhibits well-documented statistical properties that can be encoded as
                physics-like constraints:
              </p>
              <ul>
                <li>
                  <strong>Mean Reversion (OU Process):</strong> Volatility tends to revert to a
                  long-run average, captured by the Ornstein-Uhlenbeck process.
                </li>
                <li>
                  <strong>GARCH Dynamics:</strong> Current variance depends on past returns and
                  past variance, following an autoregressive pattern.
                </li>
                <li>
                  <strong>Leverage Effect:</strong> Negative returns tend to increase future
                  volatility more than positive returns of the same magnitude.
                </li>
                <li>
                  <strong>Feller Condition:</strong> Ensures variance stays positive under
                  stochastic volatility models like Heston.
                </li>
              </ul>
              <p>
                By incorporating these constraints, PINN models can achieve better generalization
                and more interpretable predictions than pure data-driven approaches.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
