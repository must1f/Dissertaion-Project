import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card"
import {
  CHART_COLORS,
  CleanDot,
  commonXAxisProps,
  commonYAxisProps,
  gridConfig,
  cursorConfig,
  tooltipContentStyle,
  tooltipLabelStyle,
} from "./chartTheme"

interface RollingSharpeData {
  date: string
  sharpe: number
  regime?: "low_vol" | "normal" | "high_vol"
}

interface RollingSharpeChartProps {
  data: RollingSharpeData[]
  title?: string
  description?: string
  window?: number
  showRegimes?: boolean
  height?: number
}

const REGIME_COLORS = {
  low_vol: "rgba(0, 158, 115, 0.06)",    // green tint
  normal: "rgba(230, 159, 0, 0.06)",      // orange tint
  high_vol: "rgba(213, 94, 0, 0.06)",     // vermillion tint
}

export function RollingSharpeChart({
  data,
  title = "Rolling Sharpe Ratio",
  description,
  window = 126,
  showRegimes = false,
  height = 300,
}: RollingSharpeChartProps) {
  const currentSharpe = data[data.length - 1]?.sharpe || 0
  const avgSharpe = data.reduce((sum, d) => sum + d.sharpe, 0) / data.length
  const minSharpe = Math.min(...data.map(d => d.sharpe))
  const maxSharpe = Math.max(...data.map(d => d.sharpe))
  const above1 = (data.filter(d => d.sharpe > 1).length / data.length) * 100
  const above0 = (data.filter(d => d.sharpe > 0).length / data.length) * 100

  const regimeAreas: { start: string; end: string; regime: string }[] = []
  if (showRegimes && data.some(d => d.regime)) {
    let currentRegime = data[0]?.regime
    let startDate = data[0]?.date
    for (let i = 1; i < data.length; i++) {
      if (data[i].regime !== currentRegime) {
        if (currentRegime) {
          regimeAreas.push({ start: startDate, end: data[i - 1].date, regime: currentRegime })
        }
        currentRegime = data[i].regime
        startDate = data[i].date
      }
    }
    if (currentRegime) {
      regimeAreas.push({ start: startDate, end: data[data.length - 1].date, regime: currentRegime })
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
            <CardDescription className="mt-1">{window}-day rolling window</CardDescription>
          </div>
          <div className="flex items-center gap-5 text-right">
            <div>
              <div className="text-xs text-muted-foreground">Current</div>
              <div className="text-lg font-semibold font-mono" style={{
                color: currentSharpe >= 1 ? CHART_COLORS.green : currentSharpe >= 0 ? CHART_COLORS.orange : CHART_COLORS.red
              }}>
                {currentSharpe.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Average</div>
              <div className="text-base font-medium font-mono">{avgSharpe.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">&gt; 1.0</div>
              <div className="text-base font-medium font-mono">{above1.toFixed(0)}%</div>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data}>
            <CartesianGrid {...gridConfig} />

            {showRegimes && regimeAreas.map((area, idx) => (
              <ReferenceArea
                key={idx}
                x1={area.start}
                x2={area.end}
                fill={REGIME_COLORS[area.regime as keyof typeof REGIME_COLORS] || "transparent"}
                fillOpacity={1}
              />
            ))}

            {/* Negative zone shading */}
            <ReferenceArea y1={minSharpe} y2={0} fill={CHART_COLORS.red} fillOpacity={0.04} />

            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              domain={[Math.min(minSharpe, -1), Math.max(maxSharpe, 3)]}
              {...commonYAxisProps}
            />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(v: number | string | undefined) => [typeof v === 'number' ? v.toFixed(3) : String(v ?? ''), "Sharpe"]}
            />

            <ReferenceLine y={0} stroke={CHART_COLORS.border} strokeWidth={1} />
            <ReferenceLine
              y={1}
              stroke={CHART_COLORS.green}
              strokeDasharray="5 3"
              label={{ value: "Good (1.0)", position: "right", fill: CHART_COLORS.green, fontSize: 10 }}
            />
            <ReferenceLine
              y={2}
              stroke={CHART_COLORS.blue}
              strokeDasharray="5 3"
              label={{ value: "Excellent (2.0)", position: "right", fill: CHART_COLORS.blue, fontSize: 10 }}
            />

            <Line
              type="monotone"
              dataKey="sharpe"
              stroke={CHART_COLORS.blue}
              strokeWidth={1.5}
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.blue} />}
            />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-3 flex items-center justify-center gap-8 text-xs text-muted-foreground">
          <div>Min: <span className="font-medium font-mono text-foreground">{minSharpe.toFixed(2)}</span></div>
          <div>Max: <span className="font-medium font-mono text-foreground">{maxSharpe.toFixed(2)}</span></div>
          <div>Positive: <span className="font-medium font-mono text-foreground">{above0.toFixed(0)}%</span></div>
        </div>
      </CardContent>
    </Card>
  )
}
