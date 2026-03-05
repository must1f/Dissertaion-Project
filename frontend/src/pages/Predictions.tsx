import { useState, useMemo, useEffect, useRef } from "react"
import { useQuery } from "@tanstack/react-query"
import { useAppStore } from "../stores/appStore"
import { useTrainedModels } from "../hooks/useModels"
import { usePredict } from "../hooks/usePredictions"
import { useModelMetrics } from "../hooks/useMetrics"
import { dataApi } from "../services/dataApi"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { TickerSelect } from "../components/common/TickerSelect"
import { DEFAULT_TICKER, getTickerInfo } from "../config/tickers"
import { MetricCard, MetricGrid } from "../components/common/MetricCard"
import { PredictionChart } from "../components/charts/PredictionChart"
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Play,
  AlertCircle,
  RefreshCw,
  Target,
} from "lucide-react"

export default function Predictions() {
  // Fixed to S&P 500 - models are trained exclusively on S&P 500 data
  const selectedTicker = DEFAULT_TICKER
  const tickerInfo = getTickerInfo(selectedTicker)
  const { data: trainedModels, isLoading: modelsLoading } = useTrainedModels()
  const predict = usePredict()

  const [selectedModel, setSelectedModel] = useState("pinn_gbm_ou")
  const [sequenceLength, setSequenceLength] = useState(60)
  const [horizon, setHorizon] = useState(5)
  const [isRealTime, setIsRealTime] = useState(false)

  // Fetch historical stock data for the chart
  const { data: stockData, isLoading: stockLoading } = useQuery({
    queryKey: ["stockData", selectedTicker],
    queryFn: () => dataApi.getStockData(selectedTicker),
    enabled: !!selectedTicker,
  })

  // Fetch model metrics for accuracy
  const { data: modelMetrics } = useModelMetrics(selectedModel)
  const historicalAccuracy = modelMetrics?.ml_metrics?.directional_accuracy ?? null

  // Get prediction result from mutation
  const predictionResult = predict.data?.prediction
  const processingTime = predict.data?.processing_time_ms

  // Extract values from prediction result or use defaults
  const currentPrice = predictionResult?.current_price ?? stockData?.data?.[stockData.data.length - 1]?.close ?? null
  const predictedPrice = predictionResult?.predicted_price ?? null
  const predictedReturn = predictionResult?.predicted_return
    ? predictionResult.predicted_return * 100
    : null
  const confidence = predictionResult?.confidence_score ?? null
  const signalAction = predictionResult?.signal_action ?? null
  const lowerBound = predictionResult?.prediction_interval?.lower ?? null
  const upperBound = predictionResult?.prediction_interval?.upper ?? null
  const uncertaintyStd = predictionResult?.uncertainty_std ?? null

  // Transform stock data for chart
  const chartData = useMemo(() => {
    if (!stockData?.data) return []

    type ChartPoint = {
      date: string
      actual?: number
      predicted?: number
      lower?: number
      upper?: number
    }

    const data: ChartPoint[] = stockData.data.slice(-90).map((item, index, arr) => {
      const isLast = index === arr.length - 1
      return {
        date: item.timestamp.split("T")[0],
        actual: item.close,
        predicted: isLast && predictedPrice ? predictedPrice : undefined,
        lower: isLast && lowerBound ? lowerBound : undefined,
        upper: isLast && upperBound ? upperBound : undefined,
      }
    })

    // Add future prediction points if we have a prediction
    if (predictedPrice && data.length > 0) {
      const lastDate = new Date(data[data.length - 1].date)
      for (let i = 1; i <= horizon; i++) {
        const futureDate = new Date(lastDate)
        futureDate.setDate(futureDate.getDate() + i)
        data.push({
          date: futureDate.toISOString().split("T")[0],
          actual: undefined,
          predicted: predictedPrice,
          lower: lowerBound ?? undefined,
          upper: upperBound ?? undefined,
        })
      }
    }

    return data
  }, [stockData, predictedPrice, lowerBound, upperBound, horizon])

  const handlePredict = () => {
    predict.mutate({
      ticker: selectedTicker,
      model_key: selectedModel,
      sequence_length: sequenceLength,
      horizon: horizon,
      estimate_uncertainty: true,
      generate_signal: true,
    })
  }

  // Real-time polling effect
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (isRealTime) {
      // Immediate first call when toggled on
      handlePredict()
      // Then set up interval (e.g. every 3 seconds for demo purposes)
      pollingRef.current = setInterval(() => {
        handlePredict()
      }, 3000)
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [isRealTime, selectedTicker, selectedModel, sequenceLength, horizon])

  const getSignalColor = (signal: string | null) => {
    switch (signal) {
      case "BUY": return "text-green-500"
      case "SELL": return "text-red-500"
      default: return "text-yellow-500"
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Predictions</h1>
          <p className="text-muted-foreground">
            Generate price predictions with uncertainty estimates
          </p>
        </div>
        <Badge variant="default" className="text-lg font-mono">
          {selectedTicker} - {tickerInfo?.name}
        </Badge>
      </div>

      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Prediction Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-5">
            <div>
              <TickerSelect
                showLabel
                label="Data Source"
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2"
              >
                {trainedModels?.models.map((m) => (
                  <option key={m.model_key} value={m.model_key}>
                    {m.display_name}
                  </option>
                )) || (
                    <>
                      <option value="pinn_gbm_ou">PINN GBM+OU</option>
                      <option value="pinn_gbm">PINN GBM</option>
                      <option value="lstm">LSTM</option>
                    </>
                  )}
              </select>
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Sequence Length</label>
              <Input
                type="number"
                value={sequenceLength}
                onChange={(e) => setSequenceLength(Number(e.target.value))}
                min={10}
                max={200}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Horizon (days)</label>
              <Input
                type="number"
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                min={1}
                max={30}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Button
                onClick={handlePredict}
                disabled={predict.isPending || isRealTime}
                className="w-full"
              >
                {predict.isPending && !isRealTime ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                Predict
              </Button>
              <Button
                variant={isRealTime ? "destructive" : "outline"}
                onClick={() => setIsRealTime(!isRealTime)}
                className="w-full transition-colors"
                title="Automatically fetch new predictions every few seconds"
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${isRealTime ? "animate-spin" : ""}`} />
                {isRealTime ? "Stop Real-Time" : "Real-Time Mode"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error State */}
      {predict.isError && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 pt-6">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-destructive">
              Prediction failed: {predict.error?.message || "Unknown error"}
            </span>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      <MetricGrid columns={5}>
        <MetricCard
          title="Current Price"
          value={currentPrice ? `$${currentPrice.toFixed(2)}` : "--"}
          icon={<Minus className="h-4 w-4" />}
        />
        <MetricCard
          title="Predicted Price"
          value={predictedPrice ? `$${predictedPrice.toFixed(2)}` : "--"}
          icon={
            predictedReturn !== null ? (
              predictedReturn > 0 ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-500" />
              )
            ) : (
              <Minus className="h-4 w-4" />
            )
          }
          change={predictedReturn ?? undefined}
          changeLabel="expected return"
        />
        <MetricCard
          title="Confidence"
          value={confidence !== null ? `${(confidence * 100).toFixed(1)}%` : "--"}
          subtitle="Prediction confidence"
        />
        <MetricCard
          title="Signal"
          value={signalAction ?? "--"}
          valueClassName={getSignalColor(signalAction)}
          subtitle="Based on expected return"
        />
        <MetricCard
          title="Historical Accuracy"
          value={historicalAccuracy !== null ? `${historicalAccuracy.toFixed(1)}%` : "--"}
          icon={<Target className="h-4 w-4 text-blue-500" />}
          subtitle="Directional correctness"
        />
      </MetricGrid>

      {/* Prediction Chart */}
      {stockLoading ? (
        <Card>
          <CardContent className="flex h-96 items-center justify-center">
            <LoadingSpinner size="lg" />
          </CardContent>
        </Card>
      ) : chartData.length > 0 ? (
        <PredictionChart
          data={chartData}
          title={`${selectedTicker} Price Prediction`}
          description={`Using ${selectedModel} model with ${sequenceLength}-day lookback`}
          showConfidenceInterval={!!lowerBound && !!upperBound}
          currentPrice={currentPrice ?? undefined}
        />
      ) : (
        <Card>
          <CardContent className="flex h-96 items-center justify-center text-muted-foreground">
            No data available. Click "Predict" to generate a prediction.
          </CardContent>
        </Card>
      )}

      {/* Prediction Details */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Prediction Interval</CardTitle>
            <CardDescription>95% confidence interval</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Lower Bound</span>
                <span className="font-mono text-red-500">
                  {lowerBound ? `$${lowerBound.toFixed(2)}` : "--"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Predicted</span>
                <span className="font-mono font-bold">
                  {predictedPrice ? `$${predictedPrice.toFixed(2)}` : "--"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Upper Bound</span>
                <span className="font-mono text-green-500">
                  {upperBound ? `$${upperBound.toFixed(2)}` : "--"}
                </span>
              </div>
              {lowerBound && upperBound && predictedPrice && (
                <div className="mt-4 h-4 rounded-full bg-muted">
                  <div
                    className="relative h-4 rounded-full bg-primary"
                    style={{
                      width: `${((upperBound - lowerBound) / (upperBound * 1.1 - lowerBound * 0.9)) * 100}%`,
                      marginLeft: `${((lowerBound - lowerBound * 0.9) / (upperBound * 1.1 - lowerBound * 0.9)) * 100}%`
                    }}
                  >
                    <div
                      className="absolute top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-white"
                      style={{
                        left: `${((predictedPrice - lowerBound) / (upperBound - lowerBound)) * 100}%`
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Information</CardTitle>
            <CardDescription>Prediction details</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span className="font-mono">{selectedModel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Uncertainty Method</span>
                <span className="font-mono">MC Dropout (50 samples)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Processing Time</span>
                <span className="font-mono">
                  {processingTime ? `${processingTime.toFixed(0)}ms` : "--"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Prediction Std</span>
                <span className="font-mono">
                  {uncertaintyStd ? `$${uncertaintyStd.toFixed(2)}` : "--"}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
