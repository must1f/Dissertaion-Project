import {
  ResponsiveContainer,
  AreaChart,
  Area,
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

interface RollingVolatilityData {
  date: string
  volatility: number
  regime?: "low_vol" | "normal" | "high_vol"
}

interface RollingVolatilityChartProps {
  data: RollingVolatilityData[]
  title?: string
  description?: string
  window?: number
  targetVol?: number
  showRegimes?: boolean
  height?: number
}

const REGIME_THRESHOLDS = {
  low: 0.12,
  high: 0.25,
}

export function RollingVolatilityChart({
  data,
  title = "Rolling Volatility",
  description,
  window = 21,
  targetVol,
  showRegimes = true,
  height = 300,
}: RollingVolatilityChartProps) {
  const currentVol = data[data.length - 1]?.volatility || 0
  const avgVol = data.reduce((sum, d) => sum + d.volatility, 0) / data.length
  const minVol = Math.min(...data.map(d => d.volatility))
  const maxVol = Math.max(...data.map(d => d.volatility))

  const getCurrentRegime = (vol: number) => {
    if (vol < REGIME_THRESHOLDS.low) return "Low"
    if (vol > REGIME_THRESHOLDS.high) return "High"
    return "Normal"
  }
  const currentRegime = getCurrentRegime(currentVol)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
            <CardDescription className="mt-1">{window}-day rolling window (annualized)</CardDescription>
          </div>
          <div className="flex items-center gap-5 text-right">
            <div>
              <div className="text-xs text-muted-foreground">Current</div>
              <div className="text-lg font-semibold font-mono" style={{
                color: currentVol < REGIME_THRESHOLDS.low ? CHART_COLORS.green
                  : currentVol > REGIME_THRESHOLDS.high ? CHART_COLORS.red
                    : CHART_COLORS.orange
              }}>
                {(currentVol * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Regime</div>
              <div className="text-base font-medium">{currentRegime}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Average</div>
              <div className="text-base font-medium font-mono">{(avgVol * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data}>
            <CartesianGrid {...gridConfig} />

            {/* Regime zone shading */}
            {showRegimes && (
              <>
                <ReferenceArea y1={0} y2={REGIME_THRESHOLDS.low} fill={CHART_COLORS.green} fillOpacity={0.04} />
                <ReferenceArea
                  y1={REGIME_THRESHOLDS.high}
                  y2={Math.max(maxVol, REGIME_THRESHOLDS.high + 0.1)}
                  fill={CHART_COLORS.red}
                  fillOpacity={0.04}
                />
              </>
            )}

            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              domain={[0, Math.max(maxVol * 1.1, 0.4)]}
              {...commonYAxisProps}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(v: number | string | undefined) => [`${(typeof v === 'number' ? (v * 100).toFixed(2) : v ?? '')}%`, "Volatility"]}
            />

            {showRegimes && (
              <>
                <ReferenceLine
                  y={REGIME_THRESHOLDS.low}
                  stroke={CHART_COLORS.green}
                  strokeDasharray="5 3"
                  label={{ value: "Low Vol (12%)", position: "right", fill: CHART_COLORS.green, fontSize: 10 }}
                />
                <ReferenceLine
                  y={REGIME_THRESHOLDS.high}
                  stroke={CHART_COLORS.red}
                  strokeDasharray="5 3"
                  label={{ value: "High Vol (25%)", position: "right", fill: CHART_COLORS.red, fontSize: 10 }}
                />
              </>
            )}

            {targetVol && (
              <ReferenceLine
                y={targetVol}
                stroke={CHART_COLORS.blue}
                strokeDasharray="4 4"
                strokeWidth={1.5}
                label={{ value: `Target: ${(targetVol * 100).toFixed(0)}%`, position: "left", fill: CHART_COLORS.blue, fontSize: 10 }}
              />
            )}

            <Area
              type="monotone"
              dataKey="volatility"
              stroke={CHART_COLORS.purple}
              fill={CHART_COLORS.purple}
              fillOpacity={0.1}
              strokeWidth={1.5}
              activeDot={<CleanDot fill={CHART_COLORS.purple} />}
            />
          </AreaChart>
        </ResponsiveContainer>

        <div className="mt-3 flex items-center justify-center gap-8 text-xs text-muted-foreground">
          <div>Min: <span className="font-medium font-mono text-foreground">{(minVol * 100).toFixed(1)}%</span></div>
          <div>Max: <span className="font-medium font-mono text-foreground">{(maxVol * 100).toFixed(1)}%</span></div>
          <div>Range: <span className="font-medium font-mono text-foreground">{((maxVol - minVol) * 100).toFixed(1)}%</span></div>
        </div>
      </CardContent>
    </Card>
  )
}
