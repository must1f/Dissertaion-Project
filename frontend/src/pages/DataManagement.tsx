import { useState, useMemo, useEffect } from "react"
import { useAppStore } from "../stores/appStore"
import { useStockData, useFetchData, useFetchLatest, useStocks } from "../hooks/useData"
import { useModels, useTrainedModels, useLoadModel } from "../hooks/useModels"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { Progress } from "../components/ui/progress"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { TickerSelect } from "../components/common/TickerSelect"
import { PriceChart } from "../components/charts/PriceChart"
import {
  Database,
  Download,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Brain,
  Play,
  Upload,
  TrendingUp,
  BarChart3,
  Layers,
  Zap,
} from "lucide-react"
import { Link } from "react-router-dom"

// Quick date range presets
const DATE_PRESETS = [
  { label: "1 Year", days: 365 },
  { label: "2 Years", days: 730 },
  { label: "5 Years", days: 1825 },
  { label: "YTD", days: -1 }, // Special case
]

// Data interval options
const INTERVAL_OPTIONS = [
  { label: "Daily", value: "1d" },
  { label: "Monthly", value: "1mo" },
]

// Popular ticker groups
const TICKER_PRESETS = {
  tech: ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
  finance: ["JPM", "BAC", "GS", "V", "MA"],
  indices: ["^GSPC", "^DJI", "^IXIC"],
}

export default function DataManagement() {
  const { selectedTicker, setSelectedTicker } = useAppStore()
  const { data: stocksData } = useStocks()

  // Date range state
  const today = new Date()
  const oneYearAgo = new Date()
  oneYearAgo.setFullYear(today.getFullYear() - 1)

  const [dateRange, setDateRange] = useState({
    start: oneYearAgo.toISOString().split("T")[0],
    end: today.toISOString().split("T")[0],
  })

  // Track if user has manually set dates (to avoid overriding their selection)
  const [userSetDates, setUserSetDates] = useState(false)

  // Data interval state (daily or monthly)
  const [interval, setInterval] = useState<string>("1d")

  // Reset userSetDates when ticker changes so auto-clamping applies to new ticker
  useEffect(() => {
    setUserSetDates(false)
  }, [selectedTicker])

  // Clamp default date range to the data we actually have to avoid empty responses
  // But only on initial load, not after user interactions
  useEffect(() => {
    if (userSetDates) return // Don't override user selections
    if (!stocksData?.stocks?.length) return

    const info =
      stocksData.stocks.find((s) => s.ticker === selectedTicker) || stocksData.stocks[0]

    if (!info?.last_date) return

    const last = new Date(info.last_date)
    const first = info.first_date ? new Date(info.first_date) : undefined

    const proposedStart = new Date(last)
    proposedStart.setFullYear(proposedStart.getFullYear() - 1)

    const start = first && proposedStart < first ? first : proposedStart
    const end = last

    const startStr = start.toISOString().split("T")[0]
    const endStr = end.toISOString().split("T")[0]

    setDateRange((prev) =>
      prev.start === startStr && prev.end === endStr ? prev : { start: startStr, end: endStr }
    )
  }, [stocksData, selectedTicker, userSetDates])

  const [activeTab, setActiveTab] = useState<"fetch" | "train" | "models">("fetch")

  // Data fetching
  const {
    data: stockData,
    isLoading: stockLoading,
    error: stockError,
    refetch: refetchStock,
  } = useStockData(selectedTicker, dateRange.start, dateRange.end, interval)

  const fetchDataMutation = useFetchData()
  const fetchLatestMutation = useFetchLatest()

  // Models
  const { data: allModels, isLoading: modelsLoading } = useModels()
  const { data: trainedModels, isLoading: trainedLoading } = useTrainedModels()
  const loadModelMutation = useLoadModel()

  // Transform stock data for chart
  const chartData = useMemo(() => {
    if (!stockData?.data) return []
    return stockData.data.map((d) => ({
      date: d.timestamp.split("T")[0],
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }))
  }, [stockData])

  // Calculate price statistics
  const priceStats = useMemo(() => {
    if (!chartData.length) return null
    const first = chartData[0]
    const last = chartData[chartData.length - 1]
    const returns = ((last.close - first.close) / first.close) * 100
    const high = Math.max(...chartData.map((d) => d.high))
    const low = Math.min(...chartData.map((d) => d.low))
    const avgVolume = chartData.reduce((sum, d) => sum + d.volume, 0) / chartData.length

    return {
      startPrice: first.close,
      endPrice: last.close,
      totalReturn: returns,
      high,
      low,
      avgVolume,
    }
  }, [chartData])

  // Handle date preset selection - clamp to available data range
  const handleDatePreset = (preset: typeof DATE_PRESETS[0]) => {
    // Get the actual data bounds from stocksData
    const info =
      stocksData?.stocks?.find((s) => s.ticker === selectedTicker) ||
      stocksData?.stocks?.[0]

    const dataLastDate = info?.last_date ? new Date(info.last_date) : new Date()
    const dataFirstDate = info?.first_date ? new Date(info.first_date) : null

    // Use the latest available date as end (not today, which may be in the future)
    const end = dataLastDate
    let start: Date

    if (preset.days === -1) {
      // YTD - from Jan 1 of the end date's year
      start = new Date(end.getFullYear(), 0, 1)
    } else {
      start = new Date(end)
      start.setDate(start.getDate() - preset.days)
    }

    // Clamp start to first available date if needed
    if (dataFirstDate && start < dataFirstDate) {
      start = dataFirstDate
    }

    setUserSetDates(true)
    setDateRange({
      start: start.toISOString().split("T")[0],
      end: end.toISOString().split("T")[0],
    })
  }

  // Handle data fetch
  const handleFetchData = () => {
    fetchDataMutation.mutate(
      {
        tickers: [selectedTicker],
        start_date: dateRange.start,
        end_date: dateRange.end,
        force_refresh: true,
      },
      {
        onSuccess: () => {
          refetchStock()
        },
      }
    )
  }

  // Fetch latest data - gets 10 years of daily data with incremental updates
  const handleFetchLatest = () => {
    setUserSetDates(true)

    fetchLatestMutation.mutate(
      { ticker: selectedTicker, years: 10 },
      {
        onSuccess: (data) => {
          // Update date range to show full available data range
          if (data?.date_range?.start && data?.date_range?.end) {
            setDateRange({
              start: data.date_range.start,
              end: data.date_range.end,
            })
          }
          refetchStock()
        },
      }
    )
  }

  // Handle model load
  const handleLoadModel = (modelKey: string) => {
    loadModelMutation.mutate({ modelKey })
  }

  // Count models by category
  const modelCounts = useMemo(() => {
    if (!allModels?.models) return { total: 0, trained: 0, pinn: 0, baseline: 0 }
    return {
      total: allModels.total,
      trained: allModels.trained_count,
      pinn: allModels.pinn_count,
      baseline: allModels.total - allModels.pinn_count,
    }
  }, [allModels])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Data Management</h1>
        <p className="text-muted-foreground">
          Fetch stock data, visualize it, and manage your trained models
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b pb-2">
        <Button
          variant={activeTab === "fetch" ? "default" : "ghost"}
          onClick={() => setActiveTab("fetch")}
          className="gap-2"
        >
          <Database className="h-4 w-4" />
          Fetch & View Data
        </Button>
        <Button
          variant={activeTab === "train" ? "default" : "ghost"}
          onClick={() => setActiveTab("train")}
          className="gap-2"
        >
          <Brain className="h-4 w-4" />
          Train Models
        </Button>
        <Button
          variant={activeTab === "models" ? "default" : "ghost"}
          onClick={() => setActiveTab("models")}
          className="gap-2"
        >
          <Layers className="h-4 w-4" />
          Trained Models
        </Button>
      </div>

      {/* Tab Content */}
      {activeTab === "fetch" && (
        <div className="space-y-6">
          {/* Data Selection Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Fetch Stock Market Data
              </CardTitle>
              <CardDescription>
                Retrieve data from Yahoo Finance. Data is cached locally for faster access.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Controls Row */}
              <div className="grid gap-4 md:grid-cols-4">
                <div>
                  <TickerSelect
                    value={selectedTicker}
                    onChange={setSelectedTicker}
                    showLabel
                    label="Select Ticker"
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">Start Date</label>
                  <Input
                    type="date"
                    value={dateRange.start}
                    onChange={(e) => {
                      setUserSetDates(true)
                      setDateRange({ ...dateRange, start: e.target.value })
                    }}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">End Date</label>
                  <Input
                    type="date"
                    value={dateRange.end}
                    onChange={(e) => {
                      setUserSetDates(true)
                      setDateRange({ ...dateRange, end: e.target.value })
                    }}
                  />
                </div>
                <div className="flex flex-col items-end gap-2">
                  <div className="flex w-full gap-2">
                    <Button
                      onClick={handleFetchData}
                      disabled={fetchDataMutation.isPending || fetchLatestMutation.isPending}
                      variant="outline"
                      className="flex-1"
                    >
                      {fetchDataMutation.isPending ? (
                        <LoadingSpinner size="sm" className="mr-2" />
                      ) : (
                        <RefreshCw className="mr-2 h-4 w-4" />
                      )}
                      Fetch Range
                    </Button>
                    <Button
                      onClick={handleFetchLatest}
                      disabled={fetchDataMutation.isPending || fetchLatestMutation.isPending}
                      className="flex-1"
                      title="Fetch 10 years of daily data (incremental update)"
                    >
                      {fetchLatestMutation.isPending ? (
                        <LoadingSpinner size="sm" className="mr-2" />
                      ) : (
                        <Download className="mr-2 h-4 w-4" />
                      )}
                      Update Latest
                    </Button>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    Update Latest: fetches 10 years of daily data, only downloads missing dates
                  </span>
                </div>
              </div>

              {/* Date Presets and Interval Selector */}
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-muted-foreground">Quick select:</span>
                  {DATE_PRESETS.map((preset) => (
                    <Button
                      key={preset.label}
                      variant="outline"
                      size="sm"
                      onClick={() => handleDatePreset(preset)}
                    >
                      {preset.label}
                    </Button>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Interval:</span>
                  <div className="flex rounded-md border">
                    {INTERVAL_OPTIONS.map((opt) => (
                      <Button
                        key={opt.value}
                        variant={interval === opt.value ? "default" : "ghost"}
                        size="sm"
                        className={`rounded-none first:rounded-l-md last:rounded-r-md ${interval === opt.value ? "" : "border-0"
                          }`}
                        onClick={() => setInterval(opt.value)}
                      >
                        {opt.label}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Fetch Status */}
              {fetchDataMutation.isSuccess && fetchDataMutation.data?.success && (
                <div className="flex items-center gap-2 rounded-md bg-green-50 p-3 text-green-700 dark:bg-green-900/20 dark:text-green-400">
                  <CheckCircle2 className="h-5 w-5" />
                  <span>
                    {fetchDataMutation.data?.message || `Successfully fetched ${fetchDataMutation.data?.records_added.toLocaleString()} records`}
                  </span>
                </div>
              )}

              {fetchLatestMutation.isSuccess && fetchLatestMutation.data?.success && (
                <div className="flex items-center gap-2 rounded-md bg-green-50 p-3 text-green-700 dark:bg-green-900/20 dark:text-green-400">
                  <CheckCircle2 className="h-5 w-5" />
                  <div className="flex flex-col">
                    <span>{fetchLatestMutation.data?.message}</span>
                    {fetchLatestMutation.data?.date_range && (
                      <span className="text-sm opacity-80">
                        Data range: {fetchLatestMutation.data.date_range.start} to {fetchLatestMutation.data.date_range.end}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {fetchDataMutation.isSuccess && !fetchDataMutation.data?.success && (
                <div className="flex items-center gap-2 rounded-md bg-yellow-50 p-3 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400">
                  <AlertCircle className="h-5 w-5" />
                  <span>
                    {fetchDataMutation.data?.message || "No data returned from source"}
                  </span>
                </div>
              )}

              {fetchLatestMutation.isSuccess && !fetchLatestMutation.data?.success && (
                <div className="flex items-center gap-2 rounded-md bg-yellow-50 p-3 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400">
                  <AlertCircle className="h-5 w-5" />
                  <span>
                    {fetchLatestMutation.data?.message || "No new data available"}
                  </span>
                </div>
              )}

              {fetchDataMutation.isError && (
                <div className="flex items-center gap-2 rounded-md bg-red-50 p-3 text-red-700 dark:bg-red-900/20 dark:text-red-400">
                  <AlertCircle className="h-5 w-5" />
                  <span>Failed to fetch data. Please try again.</span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Data Summary */}
          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <div className="text-2xl font-bold">
                      {stockLoading ? <LoadingSpinner size="sm" /> : stockData?.count?.toLocaleString() ?? 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Records</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <div className="text-2xl font-bold">
                      {priceStats ? `$${priceStats.endPrice.toFixed(2)}` : "-"}
                    </div>
                    <div className="text-sm text-muted-foreground">Latest Price</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2">
                  <Zap
                    className={`h-5 w-5 ${priceStats && priceStats.totalReturn >= 0 ? "text-green-500" : "text-red-500"
                      }`}
                  />
                  <div>
                    <div
                      className={`text-2xl font-bold ${priceStats && priceStats.totalReturn >= 0 ? "text-green-600" : "text-red-600"
                        }`}
                    >
                      {priceStats ? `${priceStats.totalReturn >= 0 ? "+" : ""}${priceStats.totalReturn.toFixed(2)}%` : "-"}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Return</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <div className="text-2xl font-bold">
                      {priceStats ? `${(priceStats.avgVolume / 1000000).toFixed(1)}M` : "-"}
                    </div>
                    <div className="text-sm text-muted-foreground">Avg Volume</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Price Chart */}
          {stockLoading ? (
            <Card>
              <CardContent className="flex h-96 items-center justify-center">
                <LoadingSpinner size="lg" />
              </CardContent>
            </Card>
          ) : chartData.length > 0 ? (
            <PriceChart data={chartData} title={`${selectedTicker} Price History`} showVolume />
          ) : (
            <Card>
              <CardContent className="flex h-48 flex-col items-center justify-center text-muted-foreground">
                <AlertCircle className="mb-2 h-8 w-8" />
                <p>No data available. Click "Fetch Data" to retrieve stock data.</p>
              </CardContent>
            </Card>
          )}

          {/* Data Table Preview */}
          {chartData.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Raw Data Preview</CardTitle>
                    <CardDescription>Most recent 10 records</CardDescription>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="mr-2 h-4 w-4" />
                    Export CSV
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="px-4 py-3 text-left font-medium">Date</th>
                        <th className="px-4 py-3 text-right font-medium">Open</th>
                        <th className="px-4 py-3 text-right font-medium">High</th>
                        <th className="px-4 py-3 text-right font-medium">Low</th>
                        <th className="px-4 py-3 text-right font-medium">Close</th>
                        <th className="px-4 py-3 text-right font-medium">Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartData
                        .slice(-10)
                        .reverse()
                        .map((d) => (
                          <tr key={d.date} className="border-b hover:bg-muted/50">
                            <td className="px-4 py-3 font-mono">{d.date}</td>
                            <td className="px-4 py-3 text-right font-mono">${Number(d.open).toFixed(2)}</td>
                            <td className="px-4 py-3 text-right font-mono">${Number(d.high).toFixed(2)}</td>
                            <td className="px-4 py-3 text-right font-mono">${Number(d.low).toFixed(2)}</td>
                            <td className="px-4 py-3 text-right font-mono">${Number(d.close).toFixed(2)}</td>
                            <td className="px-4 py-3 text-right font-mono">
                              {(Number(d.volume) / 1000000).toFixed(2)}M
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {activeTab === "train" && (
        <div className="space-y-6">
          {/* Training Info */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Model Training
              </CardTitle>
              <CardDescription>
                Train neural network models on the fetched data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Data Status */}
              <div className="rounded-md bg-muted p-4">
                <h4 className="mb-2 font-medium">Data Status</h4>
                <div className="grid gap-2 md:grid-cols-3">
                  <div className="flex items-center gap-2">
                    {stockData?.count ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-yellow-500" />
                    )}
                    <span className="text-sm">
                      {stockData?.count
                        ? `${stockData.count.toLocaleString()} records loaded`
                        : "No data loaded"}
                    </span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Ticker: {selectedTicker}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Range: {dateRange.start} to {dateRange.end}
                  </div>
                </div>
              </div>

              {/* Training Options */}
              <div className="grid gap-4 md:grid-cols-2">
                <Link to="/training">
                  <Card className="cursor-pointer transition-colors hover:bg-muted/50">
                    <CardContent className="flex items-center gap-4 pt-6">
                      <div className="rounded-full bg-primary/10 p-3">
                        <Play className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <h4 className="font-semibold">Single Model Training</h4>
                        <p className="text-sm text-muted-foreground">
                          Train one model with detailed configuration
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </Link>

                <Link to="/batch-training">
                  <Card className="cursor-pointer transition-colors hover:bg-muted/50">
                    <CardContent className="flex items-center gap-4 pt-6">
                      <div className="rounded-full bg-primary/10 p-3">
                        <Layers className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <h4 className="font-semibold">Batch Training</h4>
                        <p className="text-sm text-muted-foreground">
                          Train multiple models with progress tracking
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              </div>

              {/* Model Overview */}
              <div className="grid gap-4 md:grid-cols-4">
                <Card>
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold">{modelCounts.total}</div>
                    <div className="text-sm text-muted-foreground">Total Models</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold text-green-600">{modelCounts.trained}</div>
                    <div className="text-sm text-muted-foreground">Trained</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold text-blue-600">{modelCounts.baseline}</div>
                    <div className="text-sm text-muted-foreground">Baseline</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold text-purple-600">{modelCounts.pinn}</div>
                    <div className="text-sm text-muted-foreground">PINN</div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "models" && (
        <div className="space-y-6">
          {/* Trained Models Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Trained Models
              </CardTitle>
              <CardDescription>
                View and load your pre-trained models for predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              {trainedLoading ? (
                <div className="flex h-48 items-center justify-center">
                  <LoadingSpinner size="lg" />
                </div>
              ) : trainedModels?.models && trainedModels.models.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {trainedModels.models.map((model) => (
                    <Card key={model.model_key} className="overflow-hidden">
                      <CardContent className="pt-6">
                        <div className="mb-3 flex items-center justify-between">
                          <h4 className="font-semibold">{model.display_name}</h4>
                          <Badge variant={model.is_pinn ? "default" : "secondary"}>
                            {model.is_pinn ? "PINN" : "Baseline"}
                          </Badge>
                        </div>
                        <p className="mb-4 text-sm text-muted-foreground">{model.description}</p>
                        <div className="mb-4 grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Architecture:</span>
                            <div className="font-mono text-xs overflow-hidden text-ellipsis whitespace-nowrap" title={typeof model.architecture === 'string' ? model.architecture : JSON.stringify(model.architecture)}>
                              {typeof model.architecture === 'string' ? model.architecture : JSON.stringify(model.architecture)}
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Status:</span>
                            <div className="flex items-center gap-1">
                              <CheckCircle2 className="h-4 w-4 text-green-500" />
                              <span className="text-green-600">Trained</span>
                            </div>
                          </div>
                        </div>
                        <Button
                          onClick={() => handleLoadModel(model.model_key)}
                          disabled={loadModelMutation.isPending}
                          className="w-full"
                          variant="outline"
                        >
                          {loadModelMutation.isPending &&
                            loadModelMutation.variables?.modelKey === model.model_key ? (
                            <LoadingSpinner size="sm" className="mr-2" />
                          ) : (
                            <Upload className="mr-2 h-4 w-4" />
                          )}
                          Load Model
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
                  <AlertCircle className="mb-2 h-8 w-8" />
                  <p>No trained models found.</p>
                  <p className="text-sm">Go to the Training tab to train your first model.</p>
                </div>
              )}

              {loadModelMutation.isSuccess && (
                <div className="mt-4 flex items-center gap-2 rounded-md bg-green-50 p-3 text-green-700 dark:bg-green-900/20 dark:text-green-400">
                  <CheckCircle2 className="h-5 w-5" />
                  <span>Model loaded successfully! Ready for predictions.</span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Links */}
          <div className="grid gap-4 md:grid-cols-3">
            <Link to="/predictions">
              <Card className="cursor-pointer transition-colors hover:bg-muted/50">
                <CardContent className="flex items-center gap-4 pt-6">
                  <TrendingUp className="h-8 w-8 text-primary" />
                  <div>
                    <h4 className="font-semibold">Make Predictions</h4>
                    <p className="text-sm text-muted-foreground">
                      Generate forecasts with loaded models
                    </p>
                  </div>
                </CardContent>
              </Card>
            </Link>

            <Link to="/model-manager">
              <Card className="cursor-pointer transition-colors hover:bg-muted/50">
                <CardContent className="flex items-center gap-4 pt-6">
                  <Layers className="h-8 w-8 text-primary" />
                  <div>
                    <h4 className="font-semibold">Model Manager</h4>
                    <p className="text-sm text-muted-foreground">
                      Manage checkpoints and model versions
                    </p>
                  </div>
                </CardContent>
              </Card>
            </Link>

            <Link to="/models">
              <Card className="cursor-pointer transition-colors hover:bg-muted/50">
                <CardContent className="flex items-center gap-4 pt-6">
                  <BarChart3 className="h-8 w-8 text-primary" />
                  <div>
                    <h4 className="font-semibold">Compare Models</h4>
                    <p className="text-sm text-muted-foreground">
                      Side-by-side model comparison
                    </p>
                  </div>
                </CardContent>
              </Card>
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}
