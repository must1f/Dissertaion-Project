import { useState, useMemo } from "react"
import { useMutation } from "@tanstack/react-query"
import { useAppStore } from "../stores/appStore"
import { useTrainedModels } from "../hooks/useModels"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { MetricCard, MetricGrid } from "../components/common/MetricCard"
import { EquityChart } from "../components/charts/EquityChart"
import { DrawdownChart } from "../components/charts/DrawdownChart"
import { BenchmarkComparisonChart } from "../components/charts/BenchmarkComparisonChart"
import { RollingSharpeChart } from "../components/charts/RollingSharpeChart"
import { RollingVolatilityChart } from "../components/charts/RollingVolatilityChart"
import { UnderwaterChart } from "../components/charts/UnderwaterChart"
import { ExposureChart } from "../components/charts/ExposureChart"
import { ExposureVolatilityScatter } from "../components/charts/ExposureVolatilityScatter"
import { RegimeChart } from "../components/charts/RegimeChart"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { Play, Download, AlertCircle, BarChart3, LineChart, TrendingUp } from "lucide-react"
import api from "../services/api"

interface Trade {
  id: string
  timestamp: string
  ticker: string
  action: "BUY" | "SELL"
  price: number
  quantity: number
  value: number
  pnl?: number
  pnl_percent?: number
}

interface BacktestResults {
  model_key: string
  ticker: string
  start_date: string
  end_date: string
  initial_capital: number
  final_value: number
  total_return: number
  annual_return: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  win_rate: number
  profit_factor?: number
  total_trades: number
  portfolio_history: Array<{
    timestamp: string
    portfolio_value: number
    cumulative_return: number
  }>
  equity_curve: number[]
  trades: Trade[]
  winning_trades: number
  losing_trades: number
}

interface BacktestResponse {
  success: boolean
  result_id: string
  results: BacktestResults
  processing_time_ms: number
}

export default function Backtesting() {
  const { selectedTicker } = useAppStore()
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"results" | "advanced">("results")

  // Fetch trained models for the dropdown
  const { data: trainedModelsData } = useTrainedModels()
  const trainedModels = trainedModelsData?.models || []

  const [config, setConfig] = useState({
    model: "pinn_gbm_ou",
    initialCapital: 100000,
    commission: 0.001,
    slippage: 0.0005,
    maxPosition: 0.2,
    stopLoss: 0.02,
    takeProfit: 0.05,
    signalThreshold: 0.001,
    positionSizing: "fixed",
  })

  const backtestMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post<BacktestResponse>("/api/backtest/run", {
        model_key: config.model,
        ticker: selectedTicker,
        initial_capital: config.initialCapital,
        commission_rate: config.commission,
        slippage_rate: config.slippage,
        max_position_size: config.maxPosition,
        stop_loss: config.stopLoss,
        take_profit: config.takeProfit,
        signal_threshold: config.signalThreshold,
        position_sizing_method: config.positionSizing,
      })
      return response.data
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || err.message || "Backtest failed")
    },
    onSuccess: () => {
      setError(null)
    },
  })

  const results = backtestMutation.data?.results
  const resultId = backtestMutation.data?.result_id
  const processingTime = backtestMutation.data?.processing_time_ms

  // Transform data for equity chart
  const equityData = results?.portfolio_history?.map((snapshot) => ({
    date: snapshot.timestamp.split("T")[0],
    value: snapshot.portfolio_value,
    benchmark: config.initialCapital * (1 + snapshot.cumulative_return * 0.5), // Simulated benchmark
  })) || []

  // Calculate drawdown data
  const drawdownData = results?.portfolio_history?.map((snapshot, index, arr) => {
    const peak = Math.max(
      ...arr.slice(0, index + 1).map((s) => s.portfolio_value)
    )
    return {
      date: snapshot.timestamp.split("T")[0],
      drawdown: (snapshot.portfolio_value - peak) / peak,
    }
  }) || []

  // Generate advanced analysis data from backtest results
  const advancedAnalysisData = useMemo(() => {
    if (!results || !results.portfolio_history || results.portfolio_history.length < 2) {
      return null
    }

    const history = results.portfolio_history

    // Calculate daily returns
    const dailyReturns: number[] = []
    for (let i = 1; i < history.length; i++) {
      const ret = (history[i].portfolio_value - history[i - 1].portfolio_value) / history[i - 1].portfolio_value
      dailyReturns.push(ret)
    }

    // Rolling volatility (21-day window, annualized)
    const rollingVolData = history.slice(21).map((snapshot, idx) => {
      const windowReturns = dailyReturns.slice(idx, idx + 21)
      const mean = windowReturns.reduce((a, b) => a + b, 0) / windowReturns.length
      const variance = windowReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / windowReturns.length
      const vol = Math.sqrt(variance) * Math.sqrt(252)

      // Simple regime detection based on volatility
      let regime: "low_vol" | "normal" | "high_vol" = "normal"
      if (vol < 0.12) regime = "low_vol"
      else if (vol > 0.25) regime = "high_vol"

      return {
        date: snapshot.timestamp.split("T")[0],
        volatility: vol,
        regime,
      }
    })

    // Rolling Sharpe (126-day window)
    const rollingSharpeData = history.slice(126).map((snapshot, idx) => {
      const windowReturns = dailyReturns.slice(idx, idx + 126)
      const mean = windowReturns.reduce((a, b) => a + b, 0) / windowReturns.length
      const variance = windowReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / windowReturns.length
      const std = Math.sqrt(variance)
      const sharpe = std > 0 ? (mean * 252) / (std * Math.sqrt(252)) : 0

      return {
        date: snapshot.timestamp.split("T")[0],
        sharpe,
      }
    })

    // Underwater/drawdown data with days underwater
    let peak = history[0].portfolio_value
    let daysUnderwater = 0
    const underwaterData = history.map((snapshot) => {
      const value = snapshot.portfolio_value
      if (value > peak) {
        peak = value
        daysUnderwater = 0
      } else {
        daysUnderwater++
      }
      const drawdown = (value - peak) / peak

      return {
        date: snapshot.timestamp.split("T")[0],
        drawdown,
        daysUnderwater,
        isRecovery: drawdown === 0 && daysUnderwater === 0,
      }
    })

    // Benchmark comparison data (simulated S&P 500-like benchmark)
    const benchmarkReturn = 0.10 / 252 // ~10% annual return
    let benchmarkValue = config.initialCapital
    const comparisonData = history.map((snapshot, idx) => {
      if (idx > 0) {
        benchmarkValue *= (1 + benchmarkReturn + (Math.random() - 0.5) * 0.01)
      }
      return {
        date: snapshot.timestamp.split("T")[0],
        strategy: snapshot.portfolio_value,
        benchmark: benchmarkValue,
        strategyReturn: snapshot.cumulative_return,
        benchmarkReturn: (benchmarkValue - config.initialCapital) / config.initialCapital,
      }
    })

    // Regime data
    const regimeData = rollingVolData.map((d, index) => ({
      date: d.date,
      regime: d.regime,
      probability: 0.8 + Math.random() * 0.15, // Simulated probability
      value: history[index + 21]?.portfolio_value || config.initialCapital
    }))

    // Exposure data (simulated based on volatility)
    const exposureData = rollingVolData.map(d => {
      const targetVol = 0.15
      const volScalar = targetVol / Math.max(d.volatility, 0.05)
      const grossExposure = Math.min(2.0, Math.max(0.1, volScalar))
      const netExposure = grossExposure * 0.95

      return {
        date: d.date,
        grossExposure,
        netExposure,
        targetExposure: 1.0,
        volatilityScalar: volScalar,
        regime: d.regime,
      }
    })

    // Exposure vs Volatility scatter data
    const scatterData = rollingVolData.map((d, idx) => {
      const ret = idx < dailyReturns.length - 21 ? dailyReturns[idx + 21] : 0
      const targetVol = 0.15
      const exposure = Math.min(2.0, Math.max(0.1, targetVol / Math.max(d.volatility, 0.05)))

      return {
        date: d.date,
        volatility: d.volatility,
        exposure,
        returns: ret,
        regime: d.regime,
      }
    })

    return {
      rollingVolData,
      rollingSharpeData,
      underwaterData,
      comparisonData,
      regimeData,
      exposureData,
      scatterData,
    }
  }, [results, config.initialCapital])

  const handleRunBacktest = () => {
    setError(null)
    backtestMutation.mutate()
  }

  const handleExportResults = async () => {
    if (!resultId) return
    try {
      await api.post(`/api/backtest/results/${resultId}/save`)
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to export results")
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Backtesting</h1>
          <p className="text-muted-foreground">
            Backtest trading strategies with historical data
          </p>
        </div>
        <Badge variant="outline" className="text-lg font-mono">
          {selectedTicker}
        </Badge>
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
          <CardTitle>Backtest Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Model</label>
              <select
                value={config.model}
                onChange={(e) => setConfig({ ...config, model: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2"
                disabled={backtestMutation.isPending}
              >
                {trainedModels.length > 0 ? (
                  trainedModels.map((model) => (
                    <option key={model.model_key} value={model.model_key}>
                      {model.display_name}
                    </option>
                  ))
                ) : (
                  <>
                    <option value="pinn_gbm_ou">PINN GBM+OU</option>
                    <option value="pinn_gbm">PINN GBM</option>
                    <option value="pinn_ou">PINN OU</option>
                    <option value="lstm">LSTM</option>
                    <option value="gru">GRU</option>
                    <option value="transformer">Transformer</option>
                  </>
                )}
              </select>
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Initial Capital</label>
              <Input
                type="number"
                value={config.initialCapital}
                onChange={(e) => setConfig({ ...config, initialCapital: Number(e.target.value) })}
                disabled={backtestMutation.isPending}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Commission</label>
              <Input
                type="number"
                value={config.commission}
                onChange={(e) => setConfig({ ...config, commission: Number(e.target.value) })}
                step={0.0001}
                disabled={backtestMutation.isPending}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Max Position</label>
              <Input
                type="number"
                value={config.maxPosition}
                onChange={(e) => setConfig({ ...config, maxPosition: Number(e.target.value) })}
                step={0.05}
                max={1}
                disabled={backtestMutation.isPending}
              />
            </div>
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Stop Loss</label>
              <Input
                type="number"
                value={config.stopLoss}
                onChange={(e) => setConfig({ ...config, stopLoss: Number(e.target.value) })}
                step={0.005}
                disabled={backtestMutation.isPending}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Take Profit</label>
              <Input
                type="number"
                value={config.takeProfit}
                onChange={(e) => setConfig({ ...config, takeProfit: Number(e.target.value) })}
                step={0.01}
                disabled={backtestMutation.isPending}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Position Sizing</label>
              <select
                value={config.positionSizing}
                onChange={(e) => setConfig({ ...config, positionSizing: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2"
                disabled={backtestMutation.isPending}
              >
                <option value="fixed">Fixed Risk</option>
                <option value="kelly_full">Full Kelly</option>
                <option value="kelly_half">Half Kelly</option>
                <option value="kelly_quarter">Quarter Kelly</option>
                <option value="volatility">Volatility Targeted</option>
                <option value="confidence">Model Confidence</option>
              </select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleRunBacktest}
                disabled={backtestMutation.isPending}
                className="mr-2"
              >
                {backtestMutation.isPending ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                Run Backtest
              </Button>
              <Button
                variant="outline"
                onClick={handleExportResults}
                disabled={!resultId}
              >
                <Download className="mr-2 h-4 w-4" />
                Export Results
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tab Navigation */}
      {results && (
        <div className="flex gap-2 border-b pb-2">
          <Button
            variant={activeTab === "results" ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab("results")}
          >
            <BarChart3 className="mr-2 h-4 w-4" />
            Results
          </Button>
          <Button
            variant={activeTab === "advanced" ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab("advanced")}
          >
            <TrendingUp className="mr-2 h-4 w-4" />
            Advanced Analysis
          </Button>
        </div>
      )}

      {/* Key Metrics */}
      {results ? (
        <>
          {/* Results Tab */}
          {activeTab === "results" && (
            <>
              <MetricGrid columns={6}>
                <MetricCard
                  title="Final Value"
                  value={`$${results.final_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                  change={results.total_return * 100}
                  trend={results.total_return >= 0 ? "up" : "down"}
                />
                <MetricCard
                  title="Total Return"
                  value={`${(results.total_return * 100).toFixed(2)}%`}
                  subtitle={`Annual: ${(results.annual_return * 100).toFixed(2)}%`}
                  trend={results.total_return >= 0 ? "up" : "down"}
                />
                <MetricCard
                  title="Sharpe Ratio"
                  value={results.sharpe_ratio.toFixed(2)}
                  trend={results.sharpe_ratio >= 1 ? "up" : results.sharpe_ratio >= 0 ? "neutral" : "down"}
                />
                <MetricCard
                  title="Max Drawdown"
                  value={`${(results.max_drawdown * 100).toFixed(2)}%`}
                  trend="down"
                />
                <MetricCard
                  title="Win Rate"
                  value={`${(results.win_rate * 100).toFixed(1)}%`}
                  subtitle={`${results.winning_trades}W / ${results.losing_trades}L`}
                />
                <MetricCard
                  title="Total Trades"
                  value={results.total_trades.toString()}
                  subtitle={processingTime ? `${processingTime.toFixed(0)}ms` : undefined}
                />
              </MetricGrid>

              {/* Charts */}
              {equityData.length > 0 && (
                <EquityChart
                  data={equityData}
                  title="Equity Curve"
                  description="Portfolio value over time"
                  initialCapital={results.initial_capital}
                  showBenchmark={true}
                />
              )}

              {drawdownData.length > 0 && (
                <DrawdownChart
                  data={drawdownData}
                  title="Drawdown Analysis"
                  description="Underwater curve showing peak-to-trough declines"
                />
              )}

              {/* Trade History */}
              {results.trades.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Trade History</CardTitle>
                    <CardDescription>
                      {results.trades.length} trades executed during backtest
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="px-4 py-3 text-left font-medium">Date</th>
                            <th className="px-4 py-3 text-left font-medium">Action</th>
                            <th className="px-4 py-3 text-left font-medium">Ticker</th>
                            <th className="px-4 py-3 text-right font-medium">Price</th>
                            <th className="px-4 py-3 text-right font-medium">Quantity</th>
                            <th className="px-4 py-3 text-right font-medium">P&L</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.trades.slice(0, 20).map((trade) => (
                            <tr key={trade.id} className="border-b hover:bg-muted/50">
                              <td className="px-4 py-3 font-mono text-sm">
                                {trade.timestamp.split("T")[0]}
                              </td>
                              <td className="px-4 py-3">
                                <Badge variant={trade.action === "BUY" ? "default" : "secondary"}>
                                  {trade.action}
                                </Badge>
                              </td>
                              <td className="px-4 py-3">{trade.ticker}</td>
                              <td className="px-4 py-3 text-right font-mono">
                                ${trade.price.toFixed(2)}
                              </td>
                              <td className="px-4 py-3 text-right font-mono">
                                {trade.quantity.toFixed(0)}
                              </td>
                              <td className="px-4 py-3 text-right font-mono">
                                {trade.pnl !== undefined && trade.pnl !== null ? (
                                  <span className={trade.pnl >= 0 ? "text-green-500" : "text-red-500"}>
                                    ${trade.pnl.toFixed(2)}
                                  </span>
                                ) : (
                                  "--"
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {results.trades.length > 20 && (
                        <p className="mt-4 text-center text-sm text-muted-foreground">
                          Showing 20 of {results.trades.length} trades
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {/* Advanced Analysis Tab */}
          {activeTab === "advanced" && advancedAnalysisData && (
            <div className="space-y-6">
              {/* Benchmark Comparison */}
              <BenchmarkComparisonChart
                data={advancedAnalysisData.comparisonData}
                title="Strategy vs Benchmark"
                description="Compare strategy performance against market benchmark"
                strategyName={config.model.toUpperCase()}
                benchmarkName="S&P 500 (Simulated)"
              />

              {/* Regime Detection */}
              {advancedAnalysisData.regimeData.length > 0 && (
                <RegimeChart
                  data={advancedAnalysisData.regimeData}
                  title="Market Regime Detection"
                  description="Volatility-based regime classification over time"
                />
              )}

              {/* Rolling Metrics Row */}
              <div className="grid gap-6 lg:grid-cols-2">
                {advancedAnalysisData.rollingSharpeData.length > 0 && (
                  <RollingSharpeChart
                    data={advancedAnalysisData.rollingSharpeData}
                    title="Rolling Sharpe Ratio"
                    description="126-day rolling risk-adjusted returns"
                    window={126}
                  />
                )}

                {advancedAnalysisData.rollingVolData.length > 0 && (
                  <RollingVolatilityChart
                    data={advancedAnalysisData.rollingVolData}
                    title="Rolling Volatility"
                    description="21-day rolling annualized volatility"
                    window={21}
                    targetVol={0.15}
                  />
                )}
              </div>

              {/* Drawdown Analysis */}
              {advancedAnalysisData.underwaterData.length > 0 && (
                <UnderwaterChart
                  data={advancedAnalysisData.underwaterData}
                  title="Underwater Analysis"
                  description="Drawdown depth with recovery tracking"
                />
              )}

              {/* Exposure Analysis Row */}
              <div className="grid gap-6 lg:grid-cols-2">
                {advancedAnalysisData.exposureData.length > 0 && (
                  <ExposureChart
                    data={advancedAnalysisData.exposureData}
                    title="Portfolio Exposure"
                    description="Dynamic exposure based on volatility targeting"
                    maxLeverage={2.0}
                    targetVol={0.15}
                  />
                )}

                {advancedAnalysisData.scatterData.length > 0 && (
                  <ExposureVolatilityScatter
                    data={advancedAnalysisData.scatterData}
                    title="Exposure vs Volatility"
                    description="Relationship between market volatility and portfolio exposure"
                    targetVol={0.15}
                  />
                )}
              </div>

              {/* Analysis Summary Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Advanced Analysis Summary</CardTitle>
                  <CardDescription>Key insights from the backtest analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-sm text-muted-foreground">Avg Rolling Sharpe</div>
                      <div className="text-2xl font-bold">
                        {advancedAnalysisData.rollingSharpeData.length > 0
                          ? (advancedAnalysisData.rollingSharpeData.reduce((sum, d) => sum + d.sharpe, 0) / advancedAnalysisData.rollingSharpeData.length).toFixed(2)
                          : "N/A"}
                      </div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-sm text-muted-foreground">Avg Volatility</div>
                      <div className="text-2xl font-bold">
                        {advancedAnalysisData.rollingVolData.length > 0
                          ? `${((advancedAnalysisData.rollingVolData.reduce((sum, d) => sum + d.volatility, 0) / advancedAnalysisData.rollingVolData.length) * 100).toFixed(1)}%`
                          : "N/A"}
                      </div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-sm text-muted-foreground">Time in High Vol</div>
                      <div className="text-2xl font-bold text-red-500">
                        {advancedAnalysisData.rollingVolData.length > 0
                          ? `${((advancedAnalysisData.rollingVolData.filter(d => d.regime === "high_vol").length / advancedAnalysisData.rollingVolData.length) * 100).toFixed(0)}%`
                          : "N/A"}
                      </div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-sm text-muted-foreground">Max Days Underwater</div>
                      <div className="text-2xl font-bold">
                        {advancedAnalysisData.underwaterData.length > 0
                          ? Math.max(...advancedAnalysisData.underwaterData.map(d => d.daysUnderwater))
                          : "N/A"}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      ) : (
        <Card>
          <CardContent className="flex h-64 items-center justify-center text-muted-foreground">
            {backtestMutation.isPending ? (
              <div className="flex flex-col items-center gap-4">
                <LoadingSpinner size="lg" />
                <span>Running backtest...</span>
              </div>
            ) : (
              "Configure your backtest settings and click 'Run Backtest' to see results."
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
