import { useState, useEffect, useMemo, useCallback } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useAppStore } from "../stores/appStore"
import { useTrainedModels } from "../hooks/useModels"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { MetricCard, MetricGrid } from "../components/common/MetricCard"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { RegimeChart } from "../components/charts/RegimeChart"
import { ExposureChart } from "../components/charts/ExposureChart"
import { RollingVolatilityChart } from "../components/charts/RollingVolatilityChart"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  Scatter,
  Customized,
} from "recharts"
import {
  Bot,
  Play,
  Pause,
  TrendingUp,
  TrendingDown,
  Activity,
  AlertCircle,
  AlertTriangle,
  DollarSign,
  Target,
  Shield,
  Clock,
  RefreshCw,
  Trash2,
  Plus,
  History,
  Bell,
  Settings,
  ChevronDown,
  ChevronUp,
  Wallet,
  BarChart3,
  PieChartIcon,
  Gauge,
} from "lucide-react"
import api from "../services/api"

// ============== Types ==============

interface Signal {
  time: string
  ticker: string
  action: "BUY" | "SELL" | "HOLD"
  confidence: number
  price: number
  expected_return: number
  model_key: string
  uncertainty_std?: number
}

interface Position {
  ticker: string
  quantity: number
  avg_entry_price: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  realized_pnl: number
  entry_time: string
  stop_loss?: number
  take_profit?: number
}

interface Trade {
  trade_id: string
  timestamp: string
  ticker: string
  side: string
  quantity: number
  price: number
  value: number
  commission: number
  pnl: number
  pnl_pct: number
  model_used: string
  signal_confidence: number
}

interface Alert {
  alert_id: string
  timestamp: string
  alert_type: string
  severity: string
  ticker?: string
  message: string
  data?: Record<string, unknown>
}

interface PortfolioAllocation {
  name: string
  value: number
  color: string
}

interface ConfidenceHistoryPoint {
  time: string
  confidence: number
  signal: number
}

interface PerformanceMetrics {
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  total_pnl: number
  realized_pnl: number
  unrealized_pnl: number
  largest_win: number
  largest_loss: number
  sharpe_ratio: number
  max_drawdown: number
  avg_trade_pnl: number
  profit_factor: number
}

interface MarketData {
  ticker: string
  price: number
  change: number
  change_pct: number
  volume: number
  open: number
  high: number
  low: number
  timestamp: string
  is_market_open: boolean
}

interface AgentStatus {
  is_running: boolean
  trading_mode: string
  model_key?: string
  ticker: string
  cash: number
  positions_value: number
  total_value: number
  total_pnl: number
  total_pnl_pct: number
  total_signals: number
  buy_signals: number
  sell_signals: number
  avg_confidence: number
  pnl_today: number
  config?: Record<string, unknown>
  signals: Signal[]
  positions: Position[]
  portfolio_allocation: PortfolioAllocation[]
  confidence_history: ConfidenceHistoryPoint[]
  performance?: PerformanceMetrics
  recent_alerts: Alert[]
  market_data?: MarketData
  started_at?: string
  last_update?: string
}

interface PortfolioHistoryPoint {
  timestamp: string
  total_value: number
  cash: number
  positions_value: number
  daily_return: number
  cumulative_return: number
}

// ============== Constants ==============

const ALLOCATION_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#6b7280"]

const POSITION_SIZING_OPTIONS = [
  { value: "fixed", label: "Fixed Risk" },
  { value: "kelly", label: "Kelly Criterion" },
  { value: "volatility", label: "Volatility-Based" },
  { value: "confidence", label: "Confidence-Based" },
]

const TRADING_MODE_OPTIONS = [
  { value: "paper", label: "Paper Trading" },
  { value: "simulation", label: "Historical Simulation" },
]

// ============== Component ==============

export default function TradingAgent() {
  const queryClient = useQueryClient()
  const { selectedTicker } = useAppStore()

  // State
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"overview" | "trades" | "positions" | "alerts" | "analysis">("overview")
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false)

  // Fetch trained models
  const { data: trainedModelsData } = useTrainedModels()
  const trainedModels = trainedModelsData?.models || []

  // Agent configuration
  const [config, setConfig] = useState({
    model: "pinn_gbm_ou",
    tradingMode: "paper",
    initialCapital: 100000,
    threshold: 0.005,
    maxPosition: 0.20,
    confidenceThreshold: 0.60,
    stopLoss: 0.02,
    takeProfit: 0.05,
    positionSizing: "confidence",
  })

  // Fetch agent status
  const {
    data: agentStatus,
    isLoading: statusLoading,
    refetch: refetchStatus,
  } = useQuery({
    queryKey: ["agentStatus"],
    queryFn: async () => {
      const response = await api.get<AgentStatus>("/api/trading/agent/status")
      return response.data
    },
    refetchInterval: isRunning ? 3000 : 10000,
    retry: false,
  })

  // Fetch portfolio history
  const { data: portfolioHistory } = useQuery({
    queryKey: ["portfolioHistory"],
    queryFn: async () => {
      const response = await api.get<{ history: PortfolioHistoryPoint[] }>("/api/trading/portfolio/history")
      return response.data.history || []
    },
    enabled: isRunning,
    refetchInterval: 10000,
  })

  // Fetch trade history
  const { data: tradeHistory } = useQuery({
    queryKey: ["tradeHistory"],
    queryFn: async () => {
      const response = await api.get<{ trades: Trade[] }>("/api/trading/trades")
      return response.data.trades || []
    },
    enabled: isRunning,
    refetchInterval: 5000,
  })

  // Fetch price history for market chart
  const { data: priceHistory } = useQuery({
    queryKey: ["priceHistory"],
    queryFn: async () => {
      const response = await api.get<{ prices: Array<{ timestamp: string; price: number; volume: number; open: number; high: number; low: number }>; ticker: string }>("/api/trading/price/history")
      return response.data.prices || []
    },
    enabled: isRunning,
    refetchInterval: 5000,
  })

  // Update running state from status
  useEffect(() => {
    if (agentStatus) {
      setIsRunning(agentStatus.is_running)
    }
  }, [agentStatus])

  // Start agent mutation
  const startAgentMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post("/api/trading/agent/start", {
        model_key: config.model,
        ticker: selectedTicker || "^GSPC",
        trading_mode: config.tradingMode,
        initial_capital: config.initialCapital,
        signal_threshold: config.threshold,
        max_position_size: config.maxPosition,
        min_confidence: config.confidenceThreshold,
        stop_loss_pct: config.stopLoss,
        take_profit_pct: config.takeProfit,
        position_sizing: config.positionSizing,
      })
      return response.data
    },
    onSuccess: () => {
      setIsRunning(true)
      setError(null)
      queryClient.invalidateQueries({ queryKey: ["agentStatus"] })
    },
    onError: (err: any) => {
      const detail = err.response?.data?.detail
      setError(typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : err.message || "Failed to start agent"))
    },
  })

  // Stop agent mutation
  const stopAgentMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post("/api/trading/agent/stop")
      return response.data
    },
    onSuccess: () => {
      setIsRunning(false)
      queryClient.invalidateQueries({ queryKey: ["agentStatus"] })
    },
    onError: (err: any) => {
      const detail = err.response?.data?.detail
      setError(typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : err.message || "Failed to stop agent"))
    },
  })

  // Close position mutation
  const closePositionMutation = useMutation({
    mutationFn: async (ticker: string) => {
      const response = await api.post("/api/trading/positions/close", { ticker })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agentStatus"] })
    },
  })

  const handleToggleAgent = useCallback(() => {
    if (isRunning) {
      stopAgentMutation.mutate()
    } else {
      startAgentMutation.mutate()
    }
  }, [isRunning, startAgentMutation, stopAgentMutation])

  // Computed values
  const allocationData = useMemo(() => {
    if (!agentStatus?.portfolio_allocation) return []
    return agentStatus.portfolio_allocation.map((item, index) => ({
      ...item,
      color: item.color || ALLOCATION_COLORS[index % ALLOCATION_COLORS.length],
    }))
  }, [agentStatus])

  const confidenceHistory = agentStatus?.confidence_history || []
  const signals = agentStatus?.signals || []
  const positions = agentStatus?.positions || []
  const alerts = agentStatus?.recent_alerts || []
  const performance = agentStatus?.performance
  const marketData = agentStatus?.market_data

  // Generate analysis data for regime and exposure visualization
  const analysisData = useMemo(() => {
    if (!portfolioHistory || portfolioHistory.length < 22) return null

    // Calculate daily returns
    const dailyReturns: number[] = []
    for (let i = 1; i < portfolioHistory.length; i++) {
      const ret = (portfolioHistory[i].total_value - portfolioHistory[i - 1].total_value) / portfolioHistory[i - 1].total_value
      dailyReturns.push(ret)
    }

    // Rolling volatility (21-point window, annualized)
    const rollingVolData = portfolioHistory.slice(21).map((snapshot, idx) => {
      const windowReturns = dailyReturns.slice(Math.max(0, idx), Math.min(dailyReturns.length, idx + 21))
      if (windowReturns.length < 5) return null

      const mean = windowReturns.reduce((a, b) => a + b, 0) / windowReturns.length
      const variance = windowReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / windowReturns.length
      const vol = Math.sqrt(variance) * Math.sqrt(252)

      // Simple regime detection based on volatility
      let regime: "low_vol" | "normal" | "high_vol" = "normal"
      if (vol < 0.12) regime = "low_vol"
      else if (vol > 0.25) regime = "high_vol"

      return {
        date: new Date(snapshot.timestamp).toLocaleDateString(),
        volatility: vol,
        regime,
      }
    }).filter(Boolean) as Array<{ date: string; volatility: number; regime: "low_vol" | "normal" | "high_vol" }>

    // Regime data
    const regimeData = rollingVolData.map(d => ({
      date: d.date,
      regime: d.regime,
      probability: 0.75 + Math.random() * 0.2,
      value: 0,
    }))

    // Exposure data based on volatility targeting
    const targetVol = 0.15
    const exposureData = rollingVolData.map(d => {
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

    // Current regime
    const currentRegime = rollingVolData.length > 0 ? rollingVolData[rollingVolData.length - 1].regime : "normal"
    const currentVol = rollingVolData.length > 0 ? rollingVolData[rollingVolData.length - 1].volatility : 0

    return {
      rollingVolData,
      regimeData,
      exposureData,
      currentRegime,
      currentVol,
    }
  }, [portfolioHistory])

  // Chart data: merge trade markers into price history for the price chart
  const chartData = useMemo(() => {
    if (!priceHistory || priceHistory.length === 0) return []
    const trades = tradeHistory ?? []
    const tradesByTime: Record<string, { side: string; qty: number; price: number }[]> = {}
    for (const t of trades) {
      const d = new Date(t.timestamp)
      d.setSeconds(Math.round(d.getSeconds() / 5) * 5, 0)
      const key = d.toISOString()
      if (!tradesByTime[key]) tradesByTime[key] = []
      tradesByTime[key].push({ side: t.side, qty: t.quantity, price: t.price })
    }
    return (priceHistory as any[]).map((p: any) => {
      const pTime = new Date(p.timestamp)
      let matched: { side: string; qty: number; price: number }[] | undefined
      for (let offset = -2; offset <= 2 && !matched; offset++) {
        const probe = new Date(pTime.getTime() + offset * 1000)
        probe.setSeconds(Math.round(probe.getSeconds() / 5) * 5, 0)
        matched = tradesByTime[probe.toISOString()]
      }
      if (matched) {
        const buys = matched.filter(m => m.side === 'BUY')
        const sells = matched.filter(m => m.side === 'SELL')
        return {
          ...p,
          buyPrice: buys.length > 0 ? p.price : undefined,
          sellPrice: sells.length > 0 ? p.price : undefined,
          tradeInfo: matched,
        }
      }
      return p
    })
  }, [priceHistory, tradeHistory])

  // Trade pairs: connect BUY → SELL on the price chart with dashed lines
  const tradePairs = useMemo(() => {
    if (!chartData.length) return []
    const pendingBuys: Array<{ idx: number; price: number }> = []
    const pairs: Array<{
      buyIdx: number; sellIdx: number
      buyPrice: number; sellPrice: number
      pnlPct: number; profitable: boolean
    }> = []
    chartData.forEach((d: any, i: number) => {
      if (d.buyPrice !== undefined) {
        pendingBuys.push({ idx: i, price: d.price })
      }
      if (d.sellPrice !== undefined && pendingBuys.length > 0) {
        for (const buy of pendingBuys) {
          const pnlPct = ((d.price - buy.price) / buy.price) * 100
          pairs.push({
            buyIdx: buy.idx, sellIdx: i,
            buyPrice: buy.price, sellPrice: d.price,
            pnlPct, profitable: d.price > buy.price,
          })
        }
        pendingBuys.length = 0
      }
    })
    return pairs
  }, [chartData])

  const renderTickerWatermark = useCallback((props: any) => {
    const { width, height } = props
    if (!width || !height) return null
    const tickerName = marketData?.ticker ?? selectedTicker ?? "^GSPC"
    return (
      <text
        x={width / 2 + 30}
        y={height / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="currentColor"
        opacity={0.07}
        fontSize={100}
        fontWeight="bold"
        className="font-mono tracking-tighter"
        style={{ pointerEvents: 'none' }}
      >
        {tickerName}
      </text>
    )
  }, [marketData?.ticker, selectedTicker])

  const renderTradePairLines = useCallback((props: any) => {
    const { xAxisMap, yAxisMap } = props
    if (!tradePairs.length || !xAxisMap || !yAxisMap) return null

    const xAxisId = Object.keys(xAxisMap)[0]
    const yAxisId = Object.keys(yAxisMap)[0]
    const xAxis = xAxisMap[xAxisId]
    const yAxis = yAxisMap[yAxisId]
    if (!xAxis || !yAxis) return null

    const getX = (val: string) => {
      const scaled = xAxis.scale(val)
      return scaled + (xAxis.bandwidth ? xAxis.bandwidth() / 2 : 0)
    }
    const getY = (val: number) => yAxis.scale(val)

    return (
      <g className="trade-pair-lines">
        {tradePairs.map((pair, i) => {
          const bpData = chartData[pair.buyIdx]
          const spData = chartData[pair.sellIdx]
          if (!bpData || !spData) return null

          const x1 = getX(bpData.timestamp)
          const y1 = getY(bpData.buyPrice)
          const x2 = getX(spData.timestamp)
          const y2 = getY(spData.sellPrice)

          if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) return null

          const color = pair.profitable ? '#22c55e' : '#ef4444'
          const midX = (x1 + x2) / 2
          const midY = Math.min(y1, y2) - 16
          const label = `${pair.profitable ? '+' : ''}${pair.pnlPct.toFixed(1)}%`
          const labelW = label.length * 7 + 10

          return (
            <g key={`pair-${i}`}>
              <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={color} strokeWidth={1.5}
                strokeDasharray="6 3" opacity={0.6}
              />
              <rect
                x={midX - labelW / 2} y={midY - 9}
                width={labelW} height={18} rx={4}
                fill={color} opacity={0.9}
              />
              <text
                x={midX} y={midY + 4}
                textAnchor="middle" fontSize={10}
                fontWeight="bold" fill="white"
                style={{ fontFamily: 'ui-monospace, monospace' }}
              >
                {label}
              </text>
            </g>
          )
        })}
      </g>
    )
  }, [tradePairs, chartData])

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(value)
  }

  // Format percentage
  const formatPercent = (value: number) => {
    const sign = value >= 0 ? "+" : ""
    return `${sign}${value.toFixed(2)}%`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Bot className="h-8 w-8" />
            AI Trading Agent
          </h1>
          <p className="text-muted-foreground mt-1">
            Real-time ML-powered trading signal generation and execution
          </p>
        </div>
        <div className="flex items-center gap-3">
          {marketData && (
            <div className="text-right mr-4">
              <div className="text-sm text-muted-foreground">{marketData.ticker}</div>
              <div className="text-xl font-bold">{formatCurrency(marketData.price)}</div>
              <div className={`text-sm ${marketData.change_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                {formatPercent(marketData.change_pct)}
              </div>
            </div>
          )}
          <Badge
            variant={isRunning ? "default" : "secondary"}
            className={`text-lg px-4 py-1 ${isRunning ? "bg-green-500 hover:bg-green-600" : ""}`}
          >
            {isRunning ? "LIVE" : "STOPPED"}
          </Badge>
        </div>
      </div>

      {/* Disclaimer */}
      <Card className="border-yellow-500/50 bg-yellow-500/5">
        <CardContent className="flex items-center gap-3 py-3">
          <AlertTriangle className="h-5 w-5 text-yellow-500 flex-shrink-0" />
          <p className="text-sm text-yellow-600 dark:text-yellow-400">
            <strong>Disclaimer:</strong> This is a paper trading simulation for educational purposes only.
            Not financial advice. Past performance does not guarantee future results.
          </p>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 pt-6">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-destructive">{error}</span>
            <Button variant="ghost" size="sm" onClick={() => setError(null)} className="ml-auto">
              Dismiss
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Agent Controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Agent Configuration
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
            >
              {showAdvancedConfig ? "Hide" : "Show"} Advanced
              {showAdvancedConfig ? <ChevronUp className="ml-1 h-4 w-4" /> : <ChevronDown className="ml-1 h-4 w-4" />}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-6">
            <div>
              <label className="mb-2 block text-sm font-medium">Model</label>
              <select
                value={config.model}
                onChange={(e) => setConfig({ ...config, model: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2"
                disabled={isRunning}
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
                    <option value="pinn_global">PINN Global</option>
                    <option value="lstm">LSTM</option>
                  </>
                )}
              </select>
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Trading Mode</label>
              <select
                value={config.tradingMode}
                onChange={(e) => setConfig({ ...config, tradingMode: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2"
                disabled={isRunning}
              >
                {TRADING_MODE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Initial Capital</label>
              <Input
                type="number"
                value={config.initialCapital}
                onChange={(e) => setConfig({ ...config, initialCapital: Number(e.target.value) })}
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Signal Threshold</label>
              <Input
                type="number"
                value={config.threshold}
                onChange={(e) => setConfig({ ...config, threshold: Number(e.target.value) })}
                step={0.005}
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Min Confidence</label>
              <Input
                type="number"
                value={config.confidenceThreshold}
                onChange={(e) => setConfig({ ...config, confidenceThreshold: Number(e.target.value) })}
                step={0.05}
                min={0.4}
                max={0.95}
                disabled={isRunning}
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleToggleAgent}
                className="w-full"
                variant={isRunning ? "destructive" : "default"}
                disabled={startAgentMutation.isPending || stopAgentMutation.isPending}
              >
                {startAgentMutation.isPending || stopAgentMutation.isPending ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : isRunning ? (
                  <>
                    <Pause className="mr-2 h-4 w-4" />
                    Stop Agent
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Agent
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Advanced Configuration */}
          {showAdvancedConfig && (
            <div className="mt-4 pt-4 border-t grid gap-4 md:grid-cols-5">
              <div>
                <label className="mb-2 block text-sm font-medium">Max Position Size</label>
                <Input
                  type="number"
                  value={config.maxPosition}
                  onChange={(e) => setConfig({ ...config, maxPosition: Number(e.target.value) })}
                  step={0.05}
                  min={0.05}
                  max={0.5}
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Stop Loss %</label>
                <Input
                  type="number"
                  value={config.stopLoss}
                  onChange={(e) => setConfig({ ...config, stopLoss: Number(e.target.value) })}
                  step={0.01}
                  min={0.01}
                  max={0.10}
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Take Profit %</label>
                <Input
                  type="number"
                  value={config.takeProfit}
                  onChange={(e) => setConfig({ ...config, takeProfit: Number(e.target.value) })}
                  step={0.01}
                  min={0.02}
                  max={0.20}
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Position Sizing</label>
                <select
                  value={config.positionSizing}
                  onChange={(e) => setConfig({ ...config, positionSizing: e.target.value })}
                  className="w-full rounded-md border border-input bg-background px-3 py-2"
                  disabled={isRunning}
                >
                  {POSITION_SIZING_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => refetchStatus()}
                  disabled={statusLoading}
                >
                  <RefreshCw className={`mr-2 h-4 w-4 ${statusLoading ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Regime Indicator */}
      {isRunning && analysisData && (
        <Card className="border-l-4" style={{
          borderLeftColor: analysisData.currentRegime === "low_vol" ? "#22c55e" :
            analysisData.currentRegime === "high_vol" ? "#ef4444" : "#eab308"
        }}>
          <CardContent className="flex items-center justify-between py-3">
            <div className="flex items-center gap-4">
              <Gauge className="h-5 w-5" />
              <div>
                <span className="font-medium">Current Market Regime: </span>
                <Badge
                  variant={
                    analysisData.currentRegime === "low_vol" ? "default" :
                      analysisData.currentRegime === "high_vol" ? "destructive" : "secondary"
                  }
                  className={
                    analysisData.currentRegime === "low_vol" ? "bg-green-500" :
                      analysisData.currentRegime === "high_vol" ? "bg-red-500" : "bg-yellow-500"
                  }
                >
                  {analysisData.currentRegime === "low_vol" ? "Low Volatility" :
                    analysisData.currentRegime === "high_vol" ? "High Volatility" : "Normal"}
                </Badge>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm">
              <div>
                <span className="text-muted-foreground">Current Vol: </span>
                <span className="font-mono font-medium">{(analysisData.currentVol * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Target Vol: </span>
                <span className="font-mono font-medium">15%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Vol Scalar: </span>
                <span className="font-mono font-medium">
                  {analysisData.exposureData.length > 0
                    ? analysisData.exposureData[analysisData.exposureData.length - 1].volatilityScalar.toFixed(2)
                    : "1.00"}x
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b pb-2">
        {[
          { id: "overview", label: "Overview", icon: BarChart3 },
          { id: "trades", label: "Trades", icon: History },
          { id: "positions", label: "Positions", icon: Wallet },
          { id: "alerts", label: "Alerts", icon: Bell },
          { id: "analysis", label: "Analysis", icon: Gauge },
        ].map((tab) => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
          >
            <tab.icon className="mr-2 h-4 w-4" />
            {tab.label}
            {tab.id === "alerts" && alerts.length > 0 && (
              <Badge variant="destructive" className="ml-2 h-5 w-5 p-0 text-xs">
                {alerts.length}
              </Badge>
            )}
          </Button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === "overview" && (
        <>
          {/* Portfolio Summary */}
          <MetricGrid columns={6}>
            <MetricCard
              title="Portfolio Value"
              value={formatCurrency(agentStatus?.total_value ?? config.initialCapital)}
              icon={<Wallet className="h-4 w-4" />}
            />
            <MetricCard
              title="Total P&L"
              value={formatCurrency(agentStatus?.total_pnl ?? 0)}
              change={agentStatus?.total_pnl_pct}
              trend={agentStatus?.total_pnl && agentStatus.total_pnl >= 0 ? "up" : "down"}
              icon={<DollarSign className="h-4 w-4" />}
            />
            <MetricCard
              title="Cash"
              value={formatCurrency(agentStatus?.cash ?? config.initialCapital)}
              icon={<DollarSign className="h-4 w-4 text-green-500" />}
            />
            <MetricCard
              title="Positions Value"
              value={formatCurrency(agentStatus?.positions_value ?? 0)}
              icon={<PieChartIcon className="h-4 w-4 text-blue-500" />}
            />
            <MetricCard
              title="Win Rate"
              value={`${(performance?.win_rate ?? 0).toFixed(1)}%`}
              icon={<Target className="h-4 w-4 text-purple-500" />}
            />
            <MetricCard
              title="Max Drawdown"
              value={`${(performance?.max_drawdown ?? 0).toFixed(2)}%`}
              icon={<Shield className="h-4 w-4 text-orange-500" />}
            />
          </MetricGrid>

          {/* Signal Statistics */}
          <MetricGrid columns={5}>
            <MetricCard
              title="Total Signals"
              value={(agentStatus?.total_signals ?? 0).toString()}
              subtitle="Today"
            />
            <MetricCard
              title="BUY Signals"
              value={(agentStatus?.buy_signals ?? 0).toString()}
              icon={<TrendingUp className="h-4 w-4 text-green-500" />}
            />
            <MetricCard
              title="SELL Signals"
              value={(agentStatus?.sell_signals ?? 0).toString()}
              icon={<TrendingDown className="h-4 w-4 text-red-500" />}
            />
            <MetricCard
              title="Avg Confidence"
              value={`${((agentStatus?.avg_confidence ?? 0) * 100).toFixed(1)}%`}
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="P&L Today"
              value={formatCurrency(agentStatus?.pnl_today ?? 0)}
              trend={agentStatus?.pnl_today && agentStatus.pnl_today >= 0 ? "up" : "down"}
            />
          </MetricGrid>

          {/* Market Price Chart — full width */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Market Price — {marketData?.ticker ?? selectedTicker ?? "^GSPC"}
              </CardTitle>
              <CardDescription>Real-time price collected by the trading agent</CardDescription>
            </CardHeader>
            <CardContent>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={chartData}>
                    <defs>
                      <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted opacity-20" />
                    <XAxis
                      dataKey="timestamp"
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10, fontFamily: "monospace" }}
                      axisLine={{ stroke: "hsl(var(--border))" }}
                      tickLine={{ stroke: "hsl(var(--border))" }}
                      tickFormatter={(v) => new Date(v).toLocaleTimeString()}
                    />
                    <YAxis
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10, fontFamily: "monospace" }}
                      axisLine={{ stroke: "hsl(var(--border))" }}
                      tickLine={{ stroke: "hsl(var(--border))" }}
                      domain={["auto", "auto"]}
                      tickFormatter={(v) => `$${v.toLocaleString()}`}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null
                        const d = payload[0].payload as any
                        return (
                          <div className="rounded-lg border bg-black border-amber-500/30 p-3 shadow-lg text-sm text-foreground">
                            <p className="font-medium mb-1 font-mono text-amber-500">{new Date(d.timestamp).toLocaleString()}</p>
                            <p className="font-mono">PRICE: <span className="font-bold text-white">${Number(d.price).toLocaleString(undefined, { minimumFractionDigits: 2 })}</span></p>
                            <p className="text-xs text-muted-foreground mt-1 font-mono">O: ${Number(d.open).toFixed(2)}  H: ${Number(d.high).toFixed(2)}  L: ${Number(d.low).toFixed(2)}</p>
                            <p className="text-xs text-muted-foreground font-mono">VOL: {Number(d.volume).toLocaleString()}</p>
                            {d.tradeInfo && d.tradeInfo.map((t: any, i: number) => (
                              <p key={i} className={`text-xs font-bold font-mono mt-1 ${t.side === 'BUY' ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                                {t.side === 'BUY' ? '▲' : '▼'} {t.side} {t.qty.toFixed(4)} @ ${t.price.toFixed(2)}
                              </p>
                            ))}
                          </div>
                        )
                      }}
                    />
                    <Customized component={renderTickerWatermark} />
                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke="hsl(var(--primary))"
                      fill="url(#priceGradient)"
                      strokeWidth={2}
                    />
                    <Scatter
                      dataKey="buyPrice"
                      fill="#22c55e"
                      shape={(props: any) => {
                        if (props.buyPrice == null) return null
                        return (
                          <g>
                            <polygon
                              points={`${props.cx},${props.cy - 10} ${props.cx - 6},${props.cy + 2} ${props.cx + 6},${props.cy + 2}`}
                              fill="#22c55e"
                              stroke="#16a34a"
                              strokeWidth={1}
                            />
                          </g>
                        )
                      }}
                      isAnimationActive={false}
                    />
                    <Scatter
                      dataKey="sellPrice"
                      fill="#ef4444"
                      shape={(props: any) => {
                        if (props.sellPrice == null) return null
                        return (
                          <g>
                            <polygon
                              points={`${props.cx},${props.cy + 10} ${props.cx - 6},${props.cy - 2} ${props.cx + 6},${props.cy - 2}`}
                              fill="#ef4444"
                              stroke="#dc2626"
                              strokeWidth={1}
                            />
                          </g>
                        )
                      }}
                      isAnimationActive={false}
                    />
                    <Customized component={renderTradePairLines} />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                  {isRunning ? "Collecting market prices..." : "Start the agent to see the market price chart."}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Charts Row */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Portfolio Value Over Time */}
            <Card>
              <CardHeader>
                <CardTitle>Portfolio Value</CardTitle>
              </CardHeader>
              <CardContent>
                {portfolioHistory && portfolioHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={portfolioHistory}>
                      <defs>
                        <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="timestamp"
                        tick={{ fontSize: 10 }}
                        tickFormatter={(v) => new Date(v).toLocaleTimeString()}
                      />
                      <YAxis
                        tick={{ fontSize: 10 }}
                        tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        formatter={(value: any) => [formatCurrency(value as number), "Value"]}
                        labelFormatter={(label) => new Date(label).toLocaleString()}
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          borderColor: "hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="total_value"
                        stroke="hsl(var(--primary))"
                        fill="url(#portfolioGradient)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                    {isRunning ? "Collecting data..." : "Start the agent to see portfolio history."}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Portfolio Allocation */}
            <Card>
              <CardHeader>
                <CardTitle>Portfolio Allocation</CardTitle>
              </CardHeader>
              <CardContent>
                {allocationData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={allocationData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                      >
                        {allocationData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value: any) => [`${(value as number).toFixed(1)}%`, "Allocation"]} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                    No positions. Portfolio is 100% cash.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Confidence and Signals Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Signal Confidence Over Time</CardTitle>
              <CardDescription>Confidence levels and signal types</CardDescription>
            </CardHeader>
            <CardContent>
              {confidenceHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={confidenceHistory}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
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
                      dataKey="confidence"
                      name="Confidence"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-[250px] items-center justify-center text-muted-foreground">
                  {isRunning ? "Waiting for signals..." : "Start the agent to see confidence history."}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Recent Signals */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Trading Signals</CardTitle>
              <CardDescription>Latest signals generated by the agent</CardDescription>
            </CardHeader>
            <CardContent>
              {signals.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="px-4 py-3 text-left font-medium">Time</th>
                        <th className="px-4 py-3 text-left font-medium">Ticker</th>
                        <th className="px-4 py-3 text-left font-medium">Signal</th>
                        <th className="px-4 py-3 text-right font-medium">Confidence</th>
                        <th className="px-4 py-3 text-right font-medium">Price</th>
                        <th className="px-4 py-3 text-right font-medium">Expected Return</th>
                        <th className="px-4 py-3 text-left font-medium">Model</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.slice(0, 15).map((signal, i) => (
                        <tr key={i} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3 font-mono text-sm">{signal.time}</td>
                          <td className="px-4 py-3 font-medium">{signal.ticker}</td>
                          <td className="px-4 py-3">
                            <Badge
                              variant={
                                signal.action === "BUY"
                                  ? "default"
                                  : signal.action === "SELL"
                                    ? "destructive"
                                    : "secondary"
                              }
                            >
                              {signal.action}
                            </Badge>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span
                              className={`font-mono ${signal.confidence >= 0.7
                                ? "text-green-500"
                                : signal.confidence >= 0.5
                                  ? "text-yellow-500"
                                  : "text-red-500"
                                }`}
                            >
                              {(signal.confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right font-mono">{formatCurrency(signal.price)}</td>
                          <td className="px-4 py-3 text-right">
                            <span
                              className={`font-mono ${signal.expected_return > 0 ? "text-green-500" : "text-red-500"
                                }`}
                            >
                              {formatPercent(signal.expected_return * 100)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">{signal.model_key}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex h-32 items-center justify-center text-muted-foreground">
                  {isRunning ? "Waiting for signals..." : "Start the agent to generate trading signals."}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* Trades Tab */}
      {activeTab === "trades" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Trade History
            </CardTitle>
            <CardDescription>All executed trades with P&L</CardDescription>
          </CardHeader>
          <CardContent>
            {tradeHistory && tradeHistory.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="px-4 py-3 text-left font-medium">Time</th>
                      <th className="px-4 py-3 text-left font-medium">Trade ID</th>
                      <th className="px-4 py-3 text-left font-medium">Side</th>
                      <th className="px-4 py-3 text-right font-medium">Qty</th>
                      <th className="px-4 py-3 text-right font-medium">Price</th>
                      <th className="px-4 py-3 text-right font-medium">Value</th>
                      <th className="px-4 py-3 text-right font-medium">P&L</th>
                      <th className="px-4 py-3 text-right font-medium">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tradeHistory.map((trade) => (
                      <tr key={trade.trade_id} className="border-b hover:bg-muted/50">
                        <td className="px-4 py-3 font-mono text-sm">
                          {new Date(trade.timestamp).toLocaleString()}
                        </td>
                        <td className="px-4 py-3 font-mono text-sm">{trade.trade_id}</td>
                        <td className="px-4 py-3">
                          <Badge variant={trade.side === "BUY" ? "default" : "destructive"}>
                            {trade.side}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-right font-mono">{trade.quantity}</td>
                        <td className="px-4 py-3 text-right font-mono">{formatCurrency(trade.price)}</td>
                        <td className="px-4 py-3 text-right font-mono">{formatCurrency(trade.value)}</td>
                        <td className="px-4 py-3 text-right">
                          <span className={`font-mono ${trade.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                            {formatCurrency(trade.pnl)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right font-mono">
                          {(trade.signal_confidence * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                No trades executed yet.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Positions Tab */}
      {activeTab === "positions" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wallet className="h-5 w-5" />
              Current Positions
            </CardTitle>
            <CardDescription>Open positions with unrealized P&L</CardDescription>
          </CardHeader>
          <CardContent>
            {positions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="px-4 py-3 text-left font-medium">Ticker</th>
                      <th className="px-4 py-3 text-right font-medium">Qty</th>
                      <th className="px-4 py-3 text-right font-medium">Entry Price</th>
                      <th className="px-4 py-3 text-right font-medium">Current Price</th>
                      <th className="px-4 py-3 text-right font-medium">Market Value</th>
                      <th className="px-4 py-3 text-right font-medium">Unrealized P&L</th>
                      <th className="px-4 py-3 text-right font-medium">Stop Loss</th>
                      <th className="px-4 py-3 text-right font-medium">Take Profit</th>
                      <th className="px-4 py-3 text-center font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((position) => (
                      <tr key={position.ticker} className="border-b hover:bg-muted/50">
                        <td className="px-4 py-3 font-medium">{position.ticker}</td>
                        <td className="px-4 py-3 text-right font-mono">{position.quantity}</td>
                        <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.avg_entry_price)}</td>
                        <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.current_price)}</td>
                        <td className="px-4 py-3 text-right font-mono">{formatCurrency(position.market_value)}</td>
                        <td className="px-4 py-3 text-right">
                          <div>
                            <span className={`font-mono ${position.unrealized_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                              {formatCurrency(position.unrealized_pnl)}
                            </span>
                            <span className={`ml-2 text-sm ${position.unrealized_pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                              ({formatPercent(position.unrealized_pnl_pct)})
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-red-500">
                          {position.stop_loss ? formatCurrency(position.stop_loss) : "-"}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-green-500">
                          {position.take_profit ? formatCurrency(position.take_profit) : "-"}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => closePositionMutation.mutate(position.ticker)}
                            disabled={closePositionMutation.isPending}
                          >
                            Close
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                No open positions. Portfolio is 100% cash.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Alerts Tab */}
      {activeTab === "alerts" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Recent Alerts
            </CardTitle>
            <CardDescription>System notifications and risk alerts</CardDescription>
          </CardHeader>
          <CardContent>
            {alerts.length > 0 ? (
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <div
                    key={alert.alert_id}
                    className={`flex items-start gap-3 rounded-lg border p-4 ${alert.severity === "critical"
                      ? "border-red-500/50 bg-red-500/10"
                      : alert.severity === "warning"
                        ? "border-yellow-500/50 bg-yellow-500/10"
                        : "border-border"
                      }`}
                  >
                    {alert.severity === "critical" ? (
                      <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                    ) : alert.severity === "warning" ? (
                      <AlertTriangle className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                    ) : (
                      <Bell className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {alert.alert_type}
                        </Badge>
                        {alert.ticker && (
                          <Badge variant="secondary" className="text-xs">
                            {alert.ticker}
                          </Badge>
                        )}
                        <span className="text-xs text-muted-foreground ml-auto">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="mt-1 text-sm">{alert.message}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                No alerts yet.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Analysis Tab */}
      {activeTab === "analysis" && (
        <div className="space-y-6">
          {analysisData ? (
            <>
              {/* Regime Chart */}
              {analysisData.regimeData.length > 0 && (
                <RegimeChart
                  data={analysisData.regimeData}
                  title="Market Regime History"
                  description="Volatility-based regime classification over time"
                />
              )}

              {/* Volatility and Exposure Row */}
              <div className="grid gap-6 lg:grid-cols-2">
                {analysisData.rollingVolData.length > 0 && (
                  <RollingVolatilityChart
                    data={analysisData.rollingVolData}
                    title="Rolling Volatility"
                    description="21-period rolling annualized volatility"
                    window={21}
                    targetVol={0.15}
                  />
                )}

                {analysisData.exposureData.length > 0 && (
                  <ExposureChart
                    data={analysisData.exposureData}
                    title="Dynamic Exposure"
                    description="Volatility-targeted position sizing"
                    maxLeverage={2.0}
                    targetVol={0.15}
                  />
                )}
              </div>

              {/* Regime Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle>Regime Statistics</CardTitle>
                  <CardDescription>Distribution of time spent in each market regime</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-lg bg-green-500/10 p-4 text-center">
                      <div className="text-2xl font-bold text-green-500">
                        {analysisData.regimeData.length > 0
                          ? `${((analysisData.regimeData.filter(d => d.regime === "low_vol").length / analysisData.regimeData.length) * 100).toFixed(0)}%`
                          : "0%"}
                      </div>
                      <div className="text-sm text-muted-foreground">Low Volatility</div>
                      <div className="text-xs text-muted-foreground mt-1">Vol &lt; 12%</div>
                    </div>
                    <div className="rounded-lg bg-yellow-500/10 p-4 text-center">
                      <div className="text-2xl font-bold text-yellow-500">
                        {analysisData.regimeData.length > 0
                          ? `${((analysisData.regimeData.filter(d => d.regime === "normal").length / analysisData.regimeData.length) * 100).toFixed(0)}%`
                          : "0%"}
                      </div>
                      <div className="text-sm text-muted-foreground">Normal</div>
                      <div className="text-xs text-muted-foreground mt-1">12% ≤ Vol ≤ 25%</div>
                    </div>
                    <div className="rounded-lg bg-red-500/10 p-4 text-center">
                      <div className="text-2xl font-bold text-red-500">
                        {analysisData.regimeData.length > 0
                          ? `${((analysisData.regimeData.filter(d => d.regime === "high_vol").length / analysisData.regimeData.length) * 100).toFixed(0)}%`
                          : "0%"}
                      </div>
                      <div className="text-sm text-muted-foreground">High Volatility</div>
                      <div className="text-xs text-muted-foreground mt-1">Vol &gt; 25%</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Exposure Summary */}
              <Card>
                <CardHeader>
                  <CardTitle>Exposure Summary</CardTitle>
                  <CardDescription>Dynamic exposure statistics based on volatility targeting</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-2xl font-bold">
                        {analysisData.exposureData.length > 0
                          ? `${(analysisData.exposureData.reduce((sum, d) => sum + d.grossExposure, 0) / analysisData.exposureData.length).toFixed(2)}x`
                          : "1.00x"}
                      </div>
                      <div className="text-sm text-muted-foreground">Avg Gross Exposure</div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-2xl font-bold">
                        {analysisData.exposureData.length > 0
                          ? `${Math.min(...analysisData.exposureData.map(d => d.grossExposure)).toFixed(2)}x`
                          : "0.10x"}
                      </div>
                      <div className="text-sm text-muted-foreground">Min Exposure</div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-2xl font-bold">
                        {analysisData.exposureData.length > 0
                          ? `${Math.max(...analysisData.exposureData.map(d => d.grossExposure)).toFixed(2)}x`
                          : "2.00x"}
                      </div>
                      <div className="text-sm text-muted-foreground">Max Exposure</div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-4 text-center">
                      <div className="text-2xl font-bold">
                        {analysisData.exposureData.length > 0
                          ? `${(analysisData.exposureData.reduce((sum, d) => sum + d.volatilityScalar, 0) / analysisData.exposureData.length).toFixed(2)}`
                          : "1.00"}
                      </div>
                      <div className="text-sm text-muted-foreground">Avg Vol Scalar</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="flex h-64 items-center justify-center text-muted-foreground">
                {isRunning
                  ? "Collecting data for analysis... (requires at least 22 data points)"
                  : "Start the agent to see regime and exposure analysis."}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Performance Summary Card */}
      {performance && isRunning && (
        <Card>
          <CardHeader>
            <CardTitle>Performance Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-6">
              <div className="text-center p-4 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold">{performance.total_trades}</div>
                <div className="text-sm text-muted-foreground">Total Trades</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-green-500/10">
                <div className="text-2xl font-bold text-green-500">{performance.winning_trades}</div>
                <div className="text-sm text-muted-foreground">Winning</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-red-500/10">
                <div className="text-2xl font-bold text-red-500">{performance.losing_trades}</div>
                <div className="text-sm text-muted-foreground">Losing</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold">{performance.win_rate.toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">Win Rate</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-muted/50">
                <div className={`text-2xl font-bold ${performance.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {formatCurrency(performance.total_pnl)}
                </div>
                <div className="text-sm text-muted-foreground">Total P&L</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold">{formatCurrency(performance.avg_trade_pnl)}</div>
                <div className="text-sm text-muted-foreground">Avg Trade P&L</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
