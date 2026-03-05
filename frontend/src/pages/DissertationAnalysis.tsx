/**
 * Dissertation Analysis Dashboard
 *
 * Publication-quality visualizations for PINN volatility forecasting research.
 *
 * Sections:
 *   1. Core Forecast Accuracy
 *   2. Loss and Calibration Diagnostics
 *   3. Economic Performance
 *   4. Model Stability & Sensitivity
 *   5. Physics Compliance
 *   6. Model Comparison Framework
 */

import React, { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  ScatterChart, Scatter, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Cell,
  Brush, Label
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label as UILabel } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import {
  CHART_COLORS, commonXAxisProps, commonYAxisProps,
  gridConfig, tooltipContentStyle, ChartTooltip, CIBandDefs
} from "@/components/charts/chartTheme"
import {
  TrendingUp, TrendingDown, BarChart3, Activity,
  AlertTriangle, CheckCircle2, Info, Download,
  Layers, Gauge, Target, Zap
} from "lucide-react"

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

interface ForecastDataPoint {
  date: string
  realized: number
  predicted: number
  returns: number
  error: number
  regime?: "low" | "medium" | "high"
}

interface MetricResult {
  name: string
  value: number
  unit: string
  benchmark?: number
  status: "good" | "neutral" | "poor"
  description: string
}

interface ModelComparison {
  model: string
  qlike: number
  mzR2: number
  dirAcc: number
  sharpe: number
  maxDD: number
}

interface PhysicsResidual {
  date: string
  gbm: number
  ou: number
  bs: number
}

// ─────────────────────────────────────────────────────────────────────────────
// MOCK DATA GENERATORS (replace with API calls)
// ─────────────────────────────────────────────────────────────────────────────

function generateForecastData(n: number = 252): ForecastDataPoint[] {
  const data: ForecastDataPoint[] = []
  let vol = 0.15

  for (let i = 0; i < n; i++) {
    // GARCH-like volatility dynamics
    const shock = (Math.random() - 0.5) * 0.1
    vol = Math.sqrt(0.00001 + 0.1 * shock * shock + 0.85 * vol * vol)
    vol = Math.max(0.05, Math.min(0.5, vol))

    const ret = vol * (Math.random() - 0.5) * 2
    const predVol = vol * (1 + (Math.random() - 0.5) * 0.2)

    const date = new Date(2023, 0, 1)
    date.setDate(date.getDate() + i)

    data.push({
      date: date.toISOString().split('T')[0],
      realized: vol * Math.sqrt(252) * 100,
      predicted: predVol * Math.sqrt(252) * 100,
      returns: ret * 100,
      error: (predVol - vol) * Math.sqrt(252) * 100,
      regime: vol < 0.12 ? "low" : vol > 0.25 ? "high" : "medium",
    })
  }

  return data
}

function generateComparisonData(): ModelComparison[] {
  return [
    { model: "PINN-Global", qlike: 0.0823, mzR2: 0.712, dirAcc: 0.623, sharpe: 1.24, maxDD: -15.3 },
    { model: "PINN-GBM", qlike: 0.0891, mzR2: 0.685, dirAcc: 0.601, sharpe: 1.08, maxDD: -17.1 },
    { model: "PINN-OU", qlike: 0.0876, mzR2: 0.698, dirAcc: 0.612, sharpe: 1.15, maxDD: -16.2 },
    { model: "GARCH(1,1)", qlike: 0.0954, mzR2: 0.641, dirAcc: 0.572, sharpe: 0.89, maxDD: -21.5 },
    { model: "LSTM", qlike: 0.0912, mzR2: 0.668, dirAcc: 0.589, sharpe: 0.95, maxDD: -19.8 },
    { model: "EWMA", qlike: 0.1021, mzR2: 0.598, dirAcc: 0.545, sharpe: 0.72, maxDD: -24.1 },
  ]
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function MetricCard({
  title,
  value,
  unit,
  description,
  status,
  benchmark,
  icon: Icon,
}: {
  title: string
  value: number
  unit: string
  description: string
  status: "good" | "neutral" | "poor"
  benchmark?: number
  icon?: React.ComponentType<{ className?: string }>
}) {
  const statusColors = {
    good: "text-green-500 bg-green-500/10",
    neutral: "text-yellow-500 bg-yellow-500/10",
    poor: "text-red-500 bg-red-500/10",
  }

  const statusIcons = {
    good: CheckCircle2,
    neutral: Info,
    poor: AlertTriangle,
  }

  const StatusIcon = statusIcons[status]

  return (
    <Card>
      <CardContent className="pt-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold mt-1">
              {typeof value === 'number' ? value.toFixed(3) : value}
              <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>
            </p>
            {benchmark !== undefined && (
              <p className="text-xs text-muted-foreground mt-1">
                Benchmark: {benchmark.toFixed(3)}
              </p>
            )}
          </div>
          <div className={`p-2 rounded-lg ${statusColors[status]}`}>
            <StatusIcon className="h-4 w-4" />
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-2">{description}</p>
      </CardContent>
    </Card>
  )
}

function SectionHeader({
  number,
  title,
  description,
}: {
  number: number
  title: string
  description: string
}) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3">
        <Badge variant="outline" className="text-lg px-3 py-1">
          {number}
        </Badge>
        <h2 className="text-2xl font-bold">{title}</h2>
      </div>
      <p className="text-muted-foreground mt-2 ml-12">{description}</p>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1: CORE FORECAST ACCURACY
// ─────────────────────────────────────────────────────────────────────────────

function ForecastAccuracySection({ data }: { data: ForecastDataPoint[] }) {
  const [showCI, setShowCI] = useState(true)
  const [rollingWindow, setRollingWindow] = useState(21)

  // Compute rolling metrics
  const rollingData = useMemo(() => {
    return data.map((d, i) => {
      const start = Math.max(0, i - rollingWindow + 1)
      const window = data.slice(start, i + 1)
      const errors = window.map(w => w.error)
      const bias = errors.reduce((a, b) => a + b, 0) / errors.length
      const mae = errors.reduce((a, b) => a + Math.abs(b), 0) / errors.length

      return {
        ...d,
        rollingBias: bias,
        rollingMAE: mae,
      }
    })
  }, [data, rollingWindow])

  // Compute residual distribution
  const residualBins = useMemo(() => {
    const errors = data.map(d => d.error)
    const min = Math.min(...errors)
    const max = Math.max(...errors)
    const binWidth = (max - min) / 30
    const bins: { bin: number; count: number; normal: number }[] = []

    const mean = errors.reduce((a, b) => a + b, 0) / errors.length
    const std = Math.sqrt(errors.reduce((a, b) => a + (b - mean) ** 2, 0) / errors.length)

    for (let i = 0; i < 30; i++) {
      const binStart = min + i * binWidth
      const binEnd = binStart + binWidth
      const count = errors.filter(e => e >= binStart && e < binEnd).length / errors.length / binWidth

      // Normal PDF
      const binCenter = binStart + binWidth / 2
      const normal = Math.exp(-0.5 * ((binCenter - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI))

      bins.push({
        bin: binCenter,
        count,
        normal,
      })
    }

    return bins
  }, [data])

  return (
    <div className="space-y-6">
      <SectionHeader
        number={1}
        title="Core Forecast Accuracy"
        description="Evaluation of prediction accuracy through time series overlay, residual analysis, and error diagnostics"
      />

      {/* Predicted vs Realized */}
      <Card>
        <CardHeader>
          <CardTitle>Predicted vs Realized Volatility</CardTitle>
          <CardDescription>
            Time series overlay showing forecast tracking. Good forecasts closely follow realized volatility, especially during regime shifts.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-4">
            <div className="flex items-center gap-2">
              <Switch checked={showCI} onCheckedChange={setShowCI} id="show-ci" />
              <UILabel htmlFor="show-ci">Show Confidence Interval</UILabel>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={data}>
              <CartesianGrid {...gridConfig} />
              <XAxis
                dataKey="date"
                {...commonXAxisProps}
                tickFormatter={(v: any) => new Date(v).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
              />
              <YAxis {...commonYAxisProps} domain={['auto', 'auto']}>
                <Label value="Ann. Vol (%)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
              </YAxis>
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  return (
                    <ChartTooltip
                      active={active}
                      title={new Date(String(label)).toLocaleDateString()}
                      rows={[
                        { label: "Realized", value: `${payload[0]?.value?.toFixed(2)}%`, color: CHART_COLORS.actual },
                        { label: "Predicted", value: `${payload[1]?.value?.toFixed(2)}%`, color: CHART_COLORS.predicted },
                      ]}
                    />
                  )
                }}
              />
              <Line
                type="monotone"
                dataKey="realized"
                stroke={CHART_COLORS.actual}
                strokeWidth={1.5}
                dot={false}
                name="Realized"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke={CHART_COLORS.predicted}
                strokeWidth={1.5}
                dot={false}
                name="Predicted"
              />
              <Legend />
              <Brush dataKey="date" height={30} stroke={CHART_COLORS.muted} />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Rolling Forecast Error */}
      <Card>
        <CardHeader>
          <CardTitle>Rolling Forecast Error</CardTitle>
          <CardDescription>
            {rollingWindow}-day rolling bias and MAE. Stable errors indicate consistent forecast quality across market conditions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-4">
            <span className="text-sm">Window:</span>
            <Slider
              value={[rollingWindow]}
              onValueChange={([v]) => setRollingWindow(v)}
              min={5}
              max={63}
              step={1}
              className="w-48"
            />
            <span className="text-sm text-muted-foreground">{rollingWindow} days</span>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={rollingData}>
              <CartesianGrid {...gridConfig} />
              <XAxis
                dataKey="date"
                {...commonXAxisProps}
                tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short' })}
              />
              <YAxis {...commonYAxisProps}>
                <Label value="Error (%)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
              </YAxis>
              <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
              <Tooltip content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                return (
                  <ChartTooltip
                    active={active}
                    rows={[
                      { label: "Bias", value: `${payload[0]?.value?.toFixed(3)}%`, color: CHART_COLORS.predicted },
                      { label: "MAE", value: `${payload[1]?.value?.toFixed(3)}%`, color: CHART_COLORS.loss },
                    ]}
                  />
                )
              }} />
              <Area
                type="monotone"
                dataKey="rollingBias"
                fill={CHART_COLORS.confidence}
                fillOpacity={0.3}
                stroke={CHART_COLORS.predicted}
                strokeWidth={1.5}
                name="Rolling Bias"
              />
              <Line
                type="monotone"
                dataKey="rollingMAE"
                stroke={CHART_COLORS.loss}
                strokeWidth={1.5}
                dot={false}
                name="Rolling MAE"
              />
              <Legend />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Residual Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Residual Distribution</CardTitle>
          <CardDescription>
            Histogram of forecast errors vs Normal distribution. Heavy tails indicate fat-tailed errors; skewness suggests directional bias.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={residualBins}>
              <CartesianGrid {...gridConfig} />
              <XAxis
                dataKey="bin"
                {...commonXAxisProps}
                tickFormatter={(v) => v.toFixed(1)}
              >
                <Label value="Forecast Error (%)" offset={-5} position="insideBottom" />
              </XAxis>
              <YAxis {...commonYAxisProps}>
                <Label value="Density" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
              </YAxis>
              <Tooltip />
              <Bar dataKey="count" fill={CHART_COLORS.actual} fillOpacity={0.7} name="Empirical" />
              <Line
                type="monotone"
                dataKey="normal"
                stroke={CHART_COLORS.predicted}
                strokeWidth={2}
                dot={false}
                name="Normal Fit"
              />
              <Legend />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2: CALIBRATION DIAGNOSTICS
// ─────────────────────────────────────────────────────────────────────────────

function CalibrationSection({ data }: { data: ForecastDataPoint[] }) {
  // Compute QLIKE and MSE over time
  const lossData = useMemo(() => {
    const window = 21
    return data.map((d, i) => {
      const start = Math.max(0, i - window + 1)
      const windowData = data.slice(start, i + 1)

      // QLIKE: E[realized/predicted - ln(realized/predicted) - 1]
      const qlike = windowData.reduce((sum, w) => {
        const ratio = w.realized / Math.max(w.predicted, 0.01)
        return sum + (ratio - Math.log(ratio) - 1)
      }, 0) / windowData.length

      // MSE
      const mse = windowData.reduce((sum, w) => sum + w.error ** 2, 0) / windowData.length

      return {
        date: d.date,
        qlike,
        mse: mse * 10000, // Scale for visibility
      }
    })
  }, [data])

  // PIT histogram
  const pitData = useMemo(() => {
    // Compute PIT values (assuming Gaussian)
    const pitValues = data.map(d => {
      const standardized = d.returns / Math.max(d.predicted / Math.sqrt(252) / 100, 0.001)
      // CDF of standard normal (approximate)
      const z = standardized
      const cdf = 0.5 * (1 + Math.sign(z) * Math.sqrt(1 - Math.exp(-2 * z * z / Math.PI)))
      return Math.max(0, Math.min(1, cdf))
    })

    // Bin into histogram
    const nBins = 10
    const bins = Array(nBins).fill(0)
    pitValues.forEach(p => {
      const binIdx = Math.min(Math.floor(p * nBins), nBins - 1)
      bins[binIdx]++
    })

    return bins.map((count, i) => ({
      bin: `${(i * 10).toFixed(0)}-${((i + 1) * 10).toFixed(0)}%`,
      count: count / pitValues.length,
      uniform: 1 / nBins,
    }))
  }, [data])

  // VaR breach analysis
  const varData = useMemo(() => {
    const confidences = [0.90, 0.95, 0.99]
    return confidences.map(conf => {
      const zScore = -2.33 * (conf - 0.5) - 0.1 // Approximate inverse normal
      const threshold = data.map(d => zScore * d.predicted / Math.sqrt(252))
      const breaches = data.filter((d, i) => d.returns < threshold[i]).length

      return {
        confidence: `${(conf * 100).toFixed(0)}%`,
        expected: (1 - conf) * 100,
        actual: (breaches / data.length) * 100,
      }
    })
  }, [data])

  return (
    <div className="space-y-6">
      <SectionHeader
        number={2}
        title="Loss and Calibration Diagnostics"
        description="QLIKE loss (preferred for volatility), probability calibration, and VaR coverage testing"
      />

      {/* Why QLIKE */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>Why QLIKE for Volatility Forecasting?</AlertTitle>
        <AlertDescription>
          QLIKE (Quasi-Likelihood) is preferred because it is (1) scale-independent, (2) robust to heteroskedasticity,
          and (3) consistent even when realized variance is a noisy proxy (Patton, 2011).
        </AlertDescription>
      </Alert>

      {/* Loss Evolution */}
      <Card>
        <CardHeader>
          <CardTitle>Loss Evolution Over Time</CardTitle>
          <CardDescription>
            Rolling 21-day QLIKE and MSE. Stable loss indicates consistent forecast quality; spikes may indicate regime changes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={lossData}>
              <CartesianGrid {...gridConfig} />
              <XAxis
                dataKey="date"
                {...commonXAxisProps}
                tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short' })}
              />
              <YAxis yAxisId="left" {...commonYAxisProps} />
              <YAxis yAxisId="right" orientation="right" {...commonYAxisProps} />
              <Tooltip />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="qlike"
                stroke={CHART_COLORS.profit}
                strokeWidth={1.5}
                dot={false}
                name="QLIKE"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="mse"
                stroke={CHART_COLORS.loss}
                strokeWidth={1.5}
                dot={false}
                name="MSE (x10^4)"
              />
              <Legend />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* PIT Histogram */}
        <Card>
          <CardHeader>
            <CardTitle>PIT Histogram</CardTitle>
            <CardDescription>
              Probability Integral Transform. Uniform distribution indicates well-calibrated forecasts.
              U-shape = overconfident; inverted U = underconfident.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={pitData}>
                <CartesianGrid {...gridConfig} />
                <XAxis dataKey="bin" {...commonXAxisProps} />
                <YAxis {...commonYAxisProps} />
                <ReferenceLine y={0.1} stroke={CHART_COLORS.predicted} strokeDasharray="5 5" />
                <Tooltip />
                <Bar dataKey="count" fill={CHART_COLORS.actual} name="Empirical" />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* VaR Breach */}
        <Card>
          <CardHeader>
            <CardTitle>VaR Breach Rate Analysis</CardTitle>
            <CardDescription>
              Comparing actual vs expected breach rates. Good calibration: actual ≈ expected.
              Higher actual = underestimating risk.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={varData} layout="vertical">
                <CartesianGrid {...gridConfig} vertical />
                <XAxis type="number" {...commonXAxisProps} />
                <YAxis dataKey="confidence" type="category" {...commonYAxisProps} width={60} />
                <Tooltip />
                <Bar dataKey="expected" fill={CHART_COLORS.benchmark} name="Expected %" />
                <Bar dataKey="actual" fill={CHART_COLORS.profit} name="Actual %" />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3: ECONOMIC PERFORMANCE
// ─────────────────────────────────────────────────────────────────────────────

function EconomicPerformanceSection({ data }: { data: ForecastDataPoint[] }) {
  const [targetVol, setTargetVol] = useState(15)

  // Compute strategy returns
  const strategyData = useMemo(() => {
    const targetDaily = targetVol / 100 / Math.sqrt(252)
    let cumStrategy = 1
    let cumBuyHold = 1
    let runningMaxStrategy = 1
    let runningMaxBuyHold = 1

    return data.map((d, i) => {
      // Use lagged predicted vol for position sizing (avoid look-ahead)
      const laggedVol = i > 0 ? data[i - 1].predicted / 100 / Math.sqrt(252) : d.predicted / 100 / Math.sqrt(252)
      const weight = Math.max(0.25, Math.min(2.0, targetDaily / Math.max(laggedVol, 0.001)))

      const dailyReturn = d.returns / 100
      const stratReturn = weight * dailyReturn

      cumStrategy *= (1 + stratReturn)
      cumBuyHold *= (1 + dailyReturn)

      runningMaxStrategy = Math.max(runningMaxStrategy, cumStrategy)
      runningMaxBuyHold = Math.max(runningMaxBuyHold, cumBuyHold)

      return {
        date: d.date,
        strategy: (cumStrategy - 1) * 100,
        buyHold: (cumBuyHold - 1) * 100,
        ddStrategy: ((cumStrategy - runningMaxStrategy) / runningMaxStrategy) * 100,
        ddBuyHold: ((cumBuyHold - runningMaxBuyHold) / runningMaxBuyHold) * 100,
        weight,
      }
    })
  }, [data, targetVol])

  // Rolling Sharpe
  const rollingSharpe = useMemo(() => {
    const window = 63 // 3 months
    const targetDaily = targetVol / 100 / Math.sqrt(252)

    return data.map((d, i) => {
      if (i < window) return { date: d.date, strategy: null, buyHold: null }

      const windowData = data.slice(i - window + 1, i + 1)

      // Strategy returns
      const stratReturns = windowData.map((w, j) => {
        const laggedVol = j > 0 ? windowData[j - 1].predicted / 100 / Math.sqrt(252) : w.predicted / 100 / Math.sqrt(252)
        const weight = Math.max(0.25, Math.min(2.0, targetDaily / Math.max(laggedVol, 0.001)))
        return weight * w.returns / 100
      })

      // Buy & hold returns
      const bhReturns = windowData.map(w => w.returns / 100)

      const computeSharpe = (returns: number[]) => {
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length
        const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length)
        return std > 0 ? (mean / std) * Math.sqrt(252) : 0
      }

      return {
        date: d.date,
        strategy: computeSharpe(stratReturns),
        buyHold: computeSharpe(bhReturns),
      }
    })
  }, [data, targetVol])

  // Performance metrics
  const metrics = useMemo(() => {
    const final = strategyData[strategyData.length - 1]
    const annReturn = ((1 + final.strategy / 100) ** (252 / data.length) - 1) * 100
    const maxDD = Math.min(...strategyData.map(d => d.ddStrategy))

    // Approximate Sharpe from final data
    const returns = strategyData.map((d, i) =>
      i > 0 ? (1 + d.strategy / 100) / (1 + strategyData[i - 1].strategy / 100) - 1 : 0
    ).slice(1)
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length
    const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length)
    const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0

    return {
      totalReturn: final.strategy,
      annReturn,
      maxDD,
      sharpe,
      calmar: annReturn / Math.abs(maxDD),
    }
  }, [strategyData, data.length])

  return (
    <div className="space-y-6">
      <SectionHeader
        number={3}
        title="Economic Performance"
        description="Trading strategy performance using volatility-targeting. Position sizing: w_t = target_vol / predicted_vol"
      />

      {/* Controls */}
      <Card>
        <CardContent className="pt-4">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">Target Volatility:</span>
            <Slider
              value={[targetVol]}
              onValueChange={([v]) => setTargetVol(v)}
              min={5}
              max={30}
              step={1}
              className="w-48"
            />
            <span className="text-sm text-muted-foreground">{targetVol}% annual</span>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MetricCard
          title="Total Return"
          value={metrics.totalReturn}
          unit="%"
          description="Cumulative return over period"
          status={metrics.totalReturn > 0 ? "good" : "poor"}
        />
        <MetricCard
          title="Ann. Return"
          value={metrics.annReturn}
          unit="%"
          description="Annualized return"
          status={metrics.annReturn > 10 ? "good" : metrics.annReturn > 0 ? "neutral" : "poor"}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpe}
          unit=""
          description="Risk-adjusted return"
          status={metrics.sharpe > 1 ? "good" : metrics.sharpe > 0.5 ? "neutral" : "poor"}
        />
        <MetricCard
          title="Max Drawdown"
          value={metrics.maxDD}
          unit="%"
          description="Worst peak-to-trough"
          status={metrics.maxDD > -15 ? "good" : metrics.maxDD > -30 ? "neutral" : "poor"}
        />
        <MetricCard
          title="Calmar Ratio"
          value={metrics.calmar}
          unit=""
          description="Return / Max DD"
          status={metrics.calmar > 1 ? "good" : metrics.calmar > 0.5 ? "neutral" : "poor"}
        />
      </div>

      {/* Equity Curve */}
      <Card>
        <CardHeader>
          <CardTitle>Equity Curve</CardTitle>
          <CardDescription>
            Cumulative returns of volatility-targeting strategy vs buy-and-hold.
            Strategy uses lagged predictions to avoid look-ahead bias.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={strategyData}>
              <CartesianGrid {...gridConfig} />
              <XAxis
                dataKey="date"
                {...commonXAxisProps}
                tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short' })}
              />
              <YAxis {...commonYAxisProps}>
                <Label value="Cumulative Return (%)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
              </YAxis>
              <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="strategy"
                fill={CHART_COLORS.profit}
                fillOpacity={0.2}
                stroke={CHART_COLORS.profit}
                strokeWidth={2}
                name="Vol-Targeting Strategy"
              />
              <Line
                type="monotone"
                dataKey="buyHold"
                stroke={CHART_COLORS.benchmark}
                strokeWidth={1.5}
                dot={false}
                name="Buy & Hold"
              />
              <Legend />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Drawdown */}
        <Card>
          <CardHeader>
            <CardTitle>Drawdown Comparison</CardTitle>
            <CardDescription>
              Underwater equity chart showing drawdown magnitude over time.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={strategyData}>
                <CartesianGrid {...gridConfig} />
                <XAxis
                  dataKey="date"
                  {...commonXAxisProps}
                  tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short' })}
                />
                <YAxis {...commonYAxisProps} />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="ddStrategy"
                  fill={CHART_COLORS.loss}
                  fillOpacity={0.3}
                  stroke={CHART_COLORS.loss}
                  name="Strategy DD"
                />
                <Area
                  type="monotone"
                  dataKey="ddBuyHold"
                  fill={CHART_COLORS.benchmark}
                  fillOpacity={0.2}
                  stroke={CHART_COLORS.benchmark}
                  name="B&H DD"
                />
                <Legend />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Rolling Sharpe */}
        <Card>
          <CardHeader>
            <CardTitle>Rolling Sharpe Ratio</CardTitle>
            <CardDescription>
              63-day rolling Sharpe. Consistent positive values indicate stable risk-adjusted performance.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={rollingSharpe.filter(d => d.strategy !== null)}>
                <CartesianGrid {...gridConfig} />
                <XAxis
                  dataKey="date"
                  {...commonXAxisProps}
                  tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short' })}
                />
                <YAxis {...commonYAxisProps} />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                <ReferenceLine y={1} stroke={CHART_COLORS.profit} strokeDasharray="3 3" />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="strategy"
                  stroke={CHART_COLORS.profit}
                  strokeWidth={1.5}
                  dot={false}
                  name="Strategy"
                />
                <Line
                  type="monotone"
                  dataKey="buyHold"
                  stroke={CHART_COLORS.benchmark}
                  strokeWidth={1.5}
                  dot={false}
                  name="Buy & Hold"
                />
                <Legend />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4: MODEL COMPARISON
// ─────────────────────────────────────────────────────────────────────────────

function ModelComparisonSection() {
  const comparisonData = generateComparisonData()

  // Diebold-Mariano matrix (mock)
  const dmMatrix = [
    [0, 2.1, 1.8, 3.2, 2.9, 4.1],
    [-2.1, 0, -0.5, 1.2, 0.8, 2.1],
    [-1.8, 0.5, 0, 1.5, 1.1, 2.4],
    [-3.2, -1.2, -1.5, 0, -0.4, 1.0],
    [-2.9, -0.8, -1.1, 0.4, 0, 1.3],
    [-4.1, -2.1, -2.4, -1.0, -1.3, 0],
  ]

  return (
    <div className="space-y-6">
      <SectionHeader
        number={4}
        title="Model Comparison Framework"
        description="PINN vs classical benchmarks with statistical significance testing"
      />

      <Alert>
        <Target className="h-4 w-4" />
        <AlertTitle>Meaningful Improvement Thresholds</AlertTitle>
        <AlertDescription>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li><strong>QLIKE reduction &gt; 5%</strong>: Meaningful statistical improvement</li>
            <li><strong>Sharpe improvement &gt; 0.2</strong>: Economically significant</li>
            <li><strong>DM test p-value &lt; 0.05</strong>: Statistically significant (|t| &gt; 1.96)</li>
          </ul>
        </AlertDescription>
      </Alert>

      {/* Comparison Table */}
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Summary</CardTitle>
          <CardDescription>
            Comprehensive metrics across all models. Best values highlighted.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">QLIKE</TableHead>
                <TableHead className="text-right">M-Z R²</TableHead>
                <TableHead className="text-right">Dir. Acc.</TableHead>
                <TableHead className="text-right">Sharpe</TableHead>
                <TableHead className="text-right">Max DD</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {comparisonData.map((row, i) => (
                <TableRow key={row.model} className={i === 0 ? "bg-green-500/10" : ""}>
                  <TableCell className="font-medium">{row.model}</TableCell>
                  <TableCell className="text-right">
                    {row.qlike === Math.min(...comparisonData.map(d => d.qlike))
                      ? <Badge variant="outline" className="bg-green-500/20">{row.qlike.toFixed(4)}</Badge>
                      : row.qlike.toFixed(4)
                    }
                  </TableCell>
                  <TableCell className="text-right">{row.mzR2.toFixed(3)}</TableCell>
                  <TableCell className="text-right">{(row.dirAcc * 100).toFixed(1)}%</TableCell>
                  <TableCell className="text-right">
                    {row.sharpe === Math.max(...comparisonData.map(d => d.sharpe))
                      ? <Badge variant="outline" className="bg-green-500/20">{row.sharpe.toFixed(2)}</Badge>
                      : row.sharpe.toFixed(2)
                    }
                  </TableCell>
                  <TableCell className="text-right">{row.maxDD.toFixed(1)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* QLIKE Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>QLIKE Comparison</CardTitle>
            <CardDescription>Lower is better</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData} layout="vertical">
                <CartesianGrid {...gridConfig} vertical />
                <XAxis type="number" {...commonXAxisProps} />
                <YAxis dataKey="model" type="category" {...commonYAxisProps} width={100} />
                <Tooltip />
                <Bar dataKey="qlike" fill={CHART_COLORS.profit}>
                  {comparisonData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.model.startsWith("PINN") ? CHART_COLORS.profit : CHART_COLORS.benchmark}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Sharpe Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Sharpe Ratio Comparison</CardTitle>
            <CardDescription>Higher is better</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData} layout="vertical">
                <CartesianGrid {...gridConfig} vertical />
                <XAxis type="number" {...commonXAxisProps} />
                <YAxis dataKey="model" type="category" {...commonYAxisProps} width={100} />
                <Tooltip />
                <Bar dataKey="sharpe" fill={CHART_COLORS.actual}>
                  {comparisonData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.model.startsWith("PINN") ? CHART_COLORS.profit : CHART_COLORS.benchmark}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* DM Heatmap (simplified as table) */}
      <Card>
        <CardHeader>
          <CardTitle>Diebold-Mariano Test Matrix</CardTitle>
          <CardDescription>
            DM statistic comparing forecast accuracy. Positive = column model better than row model.
            Values with |t| &gt; 1.96 are statistically significant (p &lt; 0.05).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead></TableHead>
                  {comparisonData.map(d => (
                    <TableHead key={d.model} className="text-center text-xs">{d.model}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {comparisonData.map((row, i) => (
                  <TableRow key={row.model}>
                    <TableCell className="font-medium text-xs">{row.model}</TableCell>
                    {dmMatrix[i].map((val, j) => (
                      <TableCell
                        key={j}
                        className={`text-center text-xs ${i === j
                          ? "bg-muted"
                          : val > 1.96
                            ? "bg-green-500/20 text-green-700"
                            : val < -1.96
                              ? "bg-red-500/20 text-red-700"
                              : ""
                          }`}
                      >
                        {i === j ? "-" : val.toFixed(2)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5: PHYSICS COMPLIANCE
// ─────────────────────────────────────────────────────────────────────────────

function PhysicsComplianceSection({ data }: { data: ForecastDataPoint[] }) {
  // Generate mock physics residuals
  const physicsData = useMemo(() => {
    return data.map(d => ({
      date: d.date,
      gbm: Math.abs(Math.random() * 0.02 - 0.01) * (1 + d.realized / 50),
      ou: Math.abs(Math.random() * 0.015 - 0.0075) * (1 + d.realized / 50),
      bs: Math.abs(Math.random() * 0.01 - 0.005) * (1 + d.realized / 50),
    }))
  }, [data])

  // Mock learned parameters over epochs
  const parameterEvolution = useMemo(() => {
    return Array.from({ length: 100 }, (_, i) => ({
      epoch: i + 1,
      theta: 0.02 + 0.08 * (1 - Math.exp(-i / 20)) + Math.random() * 0.01,
      mu: 0.15 + 0.05 * (1 - Math.exp(-i / 15)) + Math.random() * 0.02,
      sigma: 0.20 - 0.05 * (1 - Math.exp(-i / 25)) + Math.random() * 0.01,
    }))
  }, [])

  return (
    <div className="space-y-6">
      <SectionHeader
        number={5}
        title="Physics Compliance"
        description="SDE constraint residuals and learned physics parameters"
      />

      <Alert>
        <Zap className="h-4 w-4" />
        <AlertTitle>Physics Constraints in PINN</AlertTitle>
        <AlertDescription>
          <p className="mt-2">The PINN embeds SDE structure into the loss function:</p>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li><strong>GBM</strong>: dS = μS·dt + σS·dW (geometric trend)</li>
            <li><strong>OU</strong>: dσ = θ(μ - σ)dt + η·dW (mean reversion)</li>
            <li><strong>Black-Scholes</strong>: No-arbitrage pricing PDE</li>
          </ul>
        </AlertDescription>
      </Alert>

      {/* Physics Residuals Over Time */}
      <Card>
        <CardHeader>
          <CardTitle>SDE Constraint Violations</CardTitle>
          <CardDescription>
            Physics residual magnitude over time. Low residuals indicate model satisfies physical constraints.
            Spikes during extreme events are expected.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={physicsData}>
              <CartesianGrid {...gridConfig} />
              <XAxis
                dataKey="date"
                {...commonXAxisProps}
                tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short' })}
              />
              <YAxis {...commonYAxisProps}>
                <Label value="Residual" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
              </YAxis>
              <Tooltip />
              <Line type="monotone" dataKey="gbm" stroke={CHART_COLORS.blue} strokeWidth={1} dot={false} name="GBM" />
              <Line type="monotone" dataKey="ou" stroke={CHART_COLORS.profit} strokeWidth={1} dot={false} name="OU" />
              <Line type="monotone" dataKey="bs" stroke={CHART_COLORS.predicted} strokeWidth={1} dot={false} name="Black-Scholes" />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Parameter Learning */}
      <Card>
        <CardHeader>
          <CardTitle>Learned Physics Parameters</CardTitle>
          <CardDescription>
            Evolution of learned parameters during training. Convergence to stable values indicates
            successful physics learning.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={parameterEvolution}>
              <CartesianGrid {...gridConfig} />
              <XAxis dataKey="epoch" {...commonXAxisProps}>
                <Label value="Training Epoch" offset={-5} position="insideBottom" />
              </XAxis>
              <YAxis {...commonYAxisProps} />
              <Tooltip />
              <Line type="monotone" dataKey="theta" stroke={CHART_COLORS.blue} strokeWidth={2} dot={false} name="θ (OU speed)" />
              <Line type="monotone" dataKey="mu" stroke={CHART_COLORS.profit} strokeWidth={2} dot={false} name="μ (mean level)" />
              <Line type="monotone" dataKey="sigma" stroke={CHART_COLORS.predicted} strokeWidth={2} dot={false} name="σ (diffusion)" />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Final Parameter Values */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="θ (OU Speed)"
          value={parameterEvolution[parameterEvolution.length - 1].theta}
          unit=""
          description="Mean reversion speed. Higher = faster return to mean."
          status="good"
          benchmark={0.1}
        />
        <MetricCard
          title="μ (Mean Level)"
          value={parameterEvolution[parameterEvolution.length - 1].mu}
          unit=""
          description="Long-term volatility mean. Should match historical average."
          status="good"
          benchmark={0.20}
        />
        <MetricCard
          title="σ (Diffusion)"
          value={parameterEvolution[parameterEvolution.length - 1].sigma}
          unit=""
          description="Volatility of volatility. Higher = more volatile vol dynamics."
          status="neutral"
          benchmark={0.15}
        />
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export default function DissertationAnalysis() {
  const [selectedModel, setSelectedModel] = useState("pinn_global")

  // Generate data (replace with API call)
  const data = useMemo(() => generateForecastData(252), [])

  return (
    <div className="container mx-auto py-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dissertation Analysis</h1>
          <p className="text-muted-foreground mt-1">
            Publication-quality visualizations for PINN volatility forecasting
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="pinn_global">PINN-Global</SelectItem>
              <SelectItem value="pinn_gbm">PINN-GBM</SelectItem>
              <SelectItem value="pinn_ou">PINN-OU</SelectItem>
              <SelectItem value="pinn_bs">PINN-Black-Scholes</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export Figures
          </Button>
        </div>
      </div>

      <Separator />

      {/* Navigation Tabs */}
      <Tabs defaultValue="accuracy" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="accuracy" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Accuracy
          </TabsTrigger>
          <TabsTrigger value="calibration" className="flex items-center gap-2">
            <Gauge className="h-4 w-4" />
            Calibration
          </TabsTrigger>
          <TabsTrigger value="economic" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Economic
          </TabsTrigger>
          <TabsTrigger value="comparison" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Comparison
          </TabsTrigger>
          <TabsTrigger value="physics" className="flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Physics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="accuracy">
          <ForecastAccuracySection data={data} />
        </TabsContent>

        <TabsContent value="calibration">
          <CalibrationSection data={data} />
        </TabsContent>

        <TabsContent value="economic">
          <EconomicPerformanceSection data={data} />
        </TabsContent>

        <TabsContent value="comparison">
          <ModelComparisonSection />
        </TabsContent>

        <TabsContent value="physics">
          <PhysicsComplianceSection data={data} />
        </TabsContent>
      </Tabs>

      {/* Dissertation Tips */}
      <Card className="bg-muted/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            Dissertation Presentation Tips
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-semibold mb-2">Figure Order in Main Body</h4>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Predicted vs Realized (demonstrates core capability)</li>
                <li>Model Comparison Table (positions contribution)</li>
                <li>Economic Performance (practical significance)</li>
                <li>Physics Parameters (novel contribution)</li>
              </ol>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Appendix Material</h4>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Residual diagnostics</li>
                <li>Rolling error analysis</li>
                <li>PIT histograms</li>
                <li>Sensitivity analysis</li>
              </ol>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
