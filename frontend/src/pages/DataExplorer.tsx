import { useState, useMemo } from "react"
import { useAppStore } from "../stores/appStore"
import { useStockData, useStockFeatures, useFetchData } from "../hooks/useData"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { TickerSelect } from "../components/common/TickerSelect"
import { PriceChart } from "../components/charts/PriceChart"
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
import { Database, Download, RefreshCw, AlertCircle } from "lucide-react"

// Data interval options
const INTERVAL_OPTIONS = [
  { label: "Daily", value: "1d" },
  { label: "Monthly", value: "1mo" },
]

// Helper to calculate feature statistics
function calculateFeatureStats(features: Record<string, number>[]) {
  if (features.length === 0) return []

  const featureNames = ['log_return', 'rolling_volatility_20', 'momentum_20', 'rsi_14', 'macd']

  return featureNames.map(name => {
    const values = features
      .map(f => f[name])
      .filter(v => v !== undefined && v !== null && !isNaN(v)) as number[]

    if (values.length === 0) {
      return { name, mean: 0, std: 0, min: 0, max: 0 }
    }

    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
    const std = Math.sqrt(variance)
    const min = Math.min(...values)
    const max = Math.max(...values)

    return { name, mean, std, min, max }
  })
}

export default function DataExplorer() {
  const { selectedTicker, setSelectedTicker } = useAppStore()

  // Set default date range (last 1 year)
  const today = new Date()
  const oneYearAgo = new Date()
  oneYearAgo.setFullYear(today.getFullYear() - 1)

  const [dateRange, setDateRange] = useState({
    start: oneYearAgo.toISOString().split("T")[0],
    end: today.toISOString().split("T")[0],
  })

  // Data interval state (daily or monthly)
  const [interval, setInterval] = useState<string>("1d")

  // Fetch stock data
  const {
    data: stockData,
    isLoading: stockLoading,
    error: stockError
  } = useStockData(selectedTicker, dateRange.start, dateRange.end, interval)

  // Fetch stock features
  const {
    data: featuresData,
    isLoading: featuresLoading,
    error: featuresError
  } = useStockFeatures(selectedTicker, dateRange.start, dateRange.end, interval)

  // Mutation for fetching new data from source
  const fetchDataMutation = useFetchData()

  // Transform stock data for charts
  const chartData = useMemo(() => {
    if (!stockData?.data) return []
    return stockData.data.map(d => ({
      date: d.timestamp.split("T")[0],
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }))
  }, [stockData])

  // Calculate feature statistics
  const featureStats = useMemo(() => {
    if (!featuresData?.features) return []
    return calculateFeatureStats(featuresData.features as unknown as Record<string, number>[])
  }, [featuresData])

  // Transform features for time series chart
  const featureChartData = useMemo(() => {
    if (!featuresData?.features) return []
    return featuresData.features.map(f => ({
      date: f.timestamp.split("T")[0],
      rsi: f.rsi_14,
      momentum: f.momentum_20,
      volatility: f.rolling_volatility_20,
    }))
  }, [featuresData])

  const handleFetchData = () => {
    fetchDataMutation.mutate({
      tickers: [selectedTicker],
      start_date: dateRange.start,
      end_date: dateRange.end,
      force_refresh: true,
    })
  }

  const isLoading = stockLoading || featuresLoading
  const hasError = stockError || featuresError

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Data Explorer</h1>
          <p className="text-muted-foreground">
            Explore and analyze stock data and engineered features
          </p>
        </div>
      </div>

      {/* Search and Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Data Selection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-5">
            <div>
              <TickerSelect
                value={selectedTicker}
                onChange={setSelectedTicker}
                showLabel
                label="Ticker"
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">Start Date</label>
              <Input
                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium">End Date</label>
              <Input
                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
              />
            </div>
            <div className="flex items-end gap-2 md:col-span-2">
              <Button
                onClick={handleFetchData}
                disabled={fetchDataMutation.isPending}
              >
                {fetchDataMutation.isPending ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : (
                  <RefreshCw className="mr-2 h-4 w-4" />
                )}
                Fetch Data
              </Button>
              <Button variant="outline">
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
            </div>
          </div>
          {/* Interval Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Interval:</span>
            <div className="flex rounded-md border">
              {INTERVAL_OPTIONS.map((opt) => (
                <Button
                  key={opt.value}
                  variant={interval === opt.value ? "default" : "ghost"}
                  size="sm"
                  className={`rounded-none first:rounded-l-md last:rounded-r-md ${
                    interval === opt.value ? "" : "border-0"
                  }`}
                  onClick={() => setInterval(opt.value)}
                >
                  {opt.label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error State */}
      {hasError && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              <span>Failed to load data. Please try refreshing.</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Data Summary */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {isLoading ? <LoadingSpinner size="sm" /> : stockData?.count ?? 0}
            </div>
            <div className="text-sm text-muted-foreground">Total Records</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {stockData?.start_date ?? dateRange.start}
            </div>
            <div className="text-sm text-muted-foreground">First Date</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {stockData?.end_date ?? dateRange.end}
            </div>
            <div className="text-sm text-muted-foreground">Last Date</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {featuresData?.feature_names?.length ?? 0}
            </div>
            <div className="text-sm text-muted-foreground">Features</div>
          </CardContent>
        </Card>
      </div>

      {/* Price Chart */}
      {isLoading ? (
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
            <p>No price data available for the selected period.</p>
          </CardContent>
        </Card>
      )}

      {/* Features Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>Engineered Features</CardTitle>
          <CardDescription>Statistical summary of calculated features</CardDescription>
        </CardHeader>
        <CardContent>
          {featuresLoading ? (
            <div className="flex h-48 items-center justify-center">
              <LoadingSpinner size="md" />
            </div>
          ) : featureStats.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="px-4 py-3 text-left font-medium">Feature</th>
                    <th className="px-4 py-3 text-right font-medium">Mean</th>
                    <th className="px-4 py-3 text-right font-medium">Std Dev</th>
                    <th className="px-4 py-3 text-right font-medium">Min</th>
                    <th className="px-4 py-3 text-right font-medium">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {featureStats.map((f) => (
                    <tr key={f.name} className="border-b hover:bg-muted/50">
                      <td className="px-4 py-3 font-mono">{f.name}</td>
                      <td className="px-4 py-3 text-right font-mono">{f.mean.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono">{f.std.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono">{f.min.toFixed(4)}</td>
                      <td className="px-4 py-3 text-right font-mono">{f.max.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="mb-2 h-8 w-8" />
              <p>No feature data available. Try fetching data first.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Feature Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Time Series</CardTitle>
          <CardDescription>Visualization of key features over time</CardDescription>
        </CardHeader>
        <CardContent>
          {featuresLoading ? (
            <div className="flex h-[300px] items-center justify-center">
              <LoadingSpinner size="md" />
            </div>
          ) : featureChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={featureChartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis yAxisId="rsi" domain={[0, 100]} tick={{ fontSize: 12 }} />
                <YAxis yAxisId="momentum" orientation="right" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    borderColor: "hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
                <Line
                  yAxisId="rsi"
                  type="monotone"
                  dataKey="rsi"
                  name="RSI"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="momentum"
                  type="monotone"
                  dataKey="momentum"
                  name="Momentum"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[300px] flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="mb-2 h-8 w-8" />
              <p>No feature data available for visualization.</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Raw Data Table */}
      <Card>
        <CardHeader>
          <CardTitle>Raw OHLCV Data</CardTitle>
          <CardDescription>Last 10 records</CardDescription>
        </CardHeader>
        <CardContent>
          {stockLoading ? (
            <div className="flex h-48 items-center justify-center">
              <LoadingSpinner size="md" />
            </div>
          ) : chartData.length > 0 ? (
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
                  {chartData.slice(-10).reverse().map((d) => (
                    <tr key={d.date} className="border-b hover:bg-muted/50">
                      <td className="px-4 py-3 font-mono">{d.date}</td>
                      <td className="px-4 py-3 text-right font-mono">${d.open.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right font-mono">${d.high.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right font-mono">${d.low.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right font-mono">${d.close.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right font-mono">
                        {(d.volume / 1000000).toFixed(2)}M
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="mb-2 h-8 w-8" />
              <p>No data available.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
