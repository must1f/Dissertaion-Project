import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ReferenceArea,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card"
import { useState } from "react"
import { Button } from "../ui/button"
import {
  CHART_COLORS,
  CleanDot,
  commonXAxisProps,
  commonYAxisProps,
  gridConfig,
  cursorConfig,
  tooltipContentStyle,
  tooltipLabelStyle,
  legendConfig,
} from "./chartTheme"

interface BenchmarkData {
  date: string
  strategy: number
  benchmark: number
  regime?: "low_vol" | "normal" | "high_vol"
}

interface BenchmarkComparisonChartProps {
  data: BenchmarkData[]
  title?: string
  description?: string
  strategyName?: string
  benchmarkName?: string
  initialCapital?: number
  showRegimes?: boolean
  height?: number
}

const REGIME_FILLS = {
  low_vol: { fill: CHART_COLORS.green, opacity: 0.04 },
  normal: { fill: CHART_COLORS.orange, opacity: 0.04 },
  high_vol: { fill: CHART_COLORS.red, opacity: 0.04 },
}

export function BenchmarkComparisonChart({
  data,
  title = "Strategy vs Benchmark",
  description,
  strategyName = "Strategy",
  benchmarkName = "S&P 500",
  initialCapital = 100000,
  showRegimes = true,
  height = 400,
}: BenchmarkComparisonChartProps) {
  const [logScale, setLogScale] = useState(false)

  const strategyFinal = data[data.length - 1]?.strategy || initialCapital
  const benchmarkFinal = data[data.length - 1]?.benchmark || initialCapital
  const strategyReturn = ((strategyFinal - initialCapital) / initialCapital) * 100
  const benchmarkReturn = ((benchmarkFinal - initialCapital) / initialCapital) * 100
  const alpha = strategyReturn - benchmarkReturn

  const regimeAreas: { start: string; end: string; regime: string }[] = []
  if (showRegimes && data.some(d => d.regime)) {
    let currentRegime = data[0]?.regime
    let startDate = data[0]?.date
    for (let i = 1; i < data.length; i++) {
      if (data[i].regime !== currentRegime) {
        if (currentRegime) regimeAreas.push({ start: startDate, end: data[i - 1].date, regime: currentRegime })
        currentRegime = data[i].regime
        startDate = data[i].date
      }
    }
    if (currentRegime) regimeAreas.push({ start: startDate, end: data[data.length - 1].date, regime: currentRegime })
  }

  const fmtReturn = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`
  const retColor = (v: number) => v >= 0 ? CHART_COLORS.green : CHART_COLORS.red

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm" onClick={() => setLogScale(!logScale)} className="text-xs">
              {logScale ? "Linear" : "Log"} Scale
            </Button>
            <div className="flex items-center gap-5">
              <div className="text-right">
                <div className="text-xs text-muted-foreground">{strategyName}</div>
                <div className="text-base font-semibold font-mono" style={{ color: retColor(strategyReturn) }}>
                  {fmtReturn(strategyReturn)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-muted-foreground">{benchmarkName}</div>
                <div className="text-base font-semibold font-mono" style={{ color: retColor(benchmarkReturn) }}>
                  {fmtReturn(benchmarkReturn)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-muted-foreground">Alpha</div>
                <div className="text-base font-semibold font-mono" style={{ color: retColor(alpha) }}>
                  {fmtReturn(alpha)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data}>
            <CartesianGrid {...gridConfig} />

            {showRegimes && regimeAreas.map((area, idx) => {
              const cfg = REGIME_FILLS[area.regime as keyof typeof REGIME_FILLS]
              return cfg ? (
                <ReferenceArea key={idx} x1={area.start} x2={area.end} fill={cfg.fill} fillOpacity={cfg.opacity} />
              ) : null
            })}

            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              scale={logScale ? "log" : "auto"}
              domain={logScale ? ["auto", "auto"] : ["dataMin - 5000", "dataMax + 5000"]}
              {...commonYAxisProps}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
            />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(v: number | string | undefined, name?: string) => [
                `$${typeof v === 'number' ? v.toLocaleString() : v ?? ''}`,
                name === "strategy" ? strategyName : benchmarkName,
              ]}
            />
            <Legend {...legendConfig} />

            <ReferenceLine y={initialCapital} stroke={CHART_COLORS.muted} strokeDasharray="4 4" />

            {/* Benchmark — grey dashed */}
            <Line
              type="monotone"
              dataKey="benchmark"
              name={benchmarkName}
              stroke={CHART_COLORS.benchmark}
              strokeDasharray="5 3"
              strokeWidth={1.5}
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.benchmark} />}
            />

            {/* Strategy — solid green */}
            <Line
              type="monotone"
              dataKey="strategy"
              name={strategyName}
              stroke={CHART_COLORS.green}
              strokeWidth={2}
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.green} />}
            />
          </ComposedChart>
        </ResponsiveContainer>

        {showRegimes && data.some(d => d.regime) && (
          <div className="mt-3 flex items-center justify-center gap-6 text-xs text-muted-foreground">
            {Object.entries(REGIME_FILLS).map(([key, cfg]) => (
              <div key={key} className="flex items-center gap-1.5">
                <div className="h-2.5 w-5 rounded-sm" style={{ backgroundColor: cfg.fill, opacity: 0.3 }} />
                <span>{key === "low_vol" ? "Low Volatility" : key === "normal" ? "Normal" : "High Volatility"}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
