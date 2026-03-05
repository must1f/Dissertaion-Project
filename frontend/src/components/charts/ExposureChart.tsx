import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card"
import { Badge } from "../ui/badge"
import {
  CHART_COLORS,
  CleanDot,
  ChartTooltip,
  commonXAxisProps,
  commonYAxisProps,
  gridConfig,
  cursorConfig,
  legendConfig,
} from "./chartTheme"

interface ExposureData {
  date: string
  grossExposure: number
  netExposure: number
  targetExposure: number
  volatilityScalar: number
  regime?: "low_vol" | "normal" | "high_vol"
}

interface ExposureChartProps {
  data: ExposureData[]
  title?: string
  description?: string
  maxLeverage?: number
  targetVol?: number
  height?: number
}

const REGIME_BADGES = {
  low_vol: { label: "Low Vol", variant: "default" as const },
  normal: { label: "Normal", variant: "secondary" as const },
  high_vol: { label: "High Vol", variant: "destructive" as const },
}

export function ExposureChart({
  data,
  title = "Portfolio Exposure",
  description,
  maxLeverage = 2.0,
  targetVol = 0.15,
  height = 350,
}: ExposureChartProps) {
  const currentData = data[data.length - 1]
  const currentExposure = currentData?.grossExposure || 1.0
  const currentNet = currentData?.netExposure || 1.0
  const currentVolScalar = currentData?.volatilityScalar || 1.0
  const currentRegime = currentData?.regime || "normal"

  const avgExposure = data.reduce((sum, d) => sum + d.grossExposure, 0) / data.length
  const maxExposure = Math.max(...data.map(d => d.grossExposure))
  const minExposure = Math.min(...data.map(d => d.grossExposure))

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
            <CardDescription className="mt-1">
              Target Vol: {(targetVol * 100).toFixed(0)}% · Max Leverage: {maxLeverage}x
            </CardDescription>
          </div>
          <div className="flex items-center gap-5">
            <div className="text-right">
              <div className="text-xs text-muted-foreground">Gross</div>
              <div className="text-lg font-semibold font-mono" style={{
                color: currentExposure > 1.5 ? CHART_COLORS.orange : undefined
              }}>
                {currentExposure.toFixed(2)}x
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-muted-foreground">Net</div>
              <div className="text-base font-medium font-mono">{currentNet.toFixed(2)}x</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-muted-foreground">Vol Scalar</div>
              <div className="text-base font-medium font-mono">{currentVolScalar.toFixed(2)}</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-muted-foreground">Regime</div>
              <Badge variant={REGIME_BADGES[currentRegime].variant}>{REGIME_BADGES[currentRegime].label}</Badge>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data}>
            <CartesianGrid {...gridConfig} />
            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              yAxisId="exposure"
              domain={[0, Math.max(maxExposure * 1.1, maxLeverage + 0.2)]}
              {...commonYAxisProps}
              tickFormatter={(v) => `${v.toFixed(1)}x`}
            />
            <YAxis yAxisId="scalar" orientation="right" domain={[0, 3]} {...commonYAxisProps} />

            <Tooltip
              cursor={cursorConfig}
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload as ExposureData
                  return (
                    <ChartTooltip
                      active={active}
                      title={label as string}
                      rows={[
                        { label: "Gross", value: `${d.grossExposure.toFixed(2)}x`, color: CHART_COLORS.blue },
                        { label: "Net", value: `${d.netExposure.toFixed(2)}x`, color: CHART_COLORS.green },
                        { label: "Target", value: `${d.targetExposure.toFixed(2)}x`, color: CHART_COLORS.purple },
                        { label: "Vol Scalar", value: d.volatilityScalar.toFixed(2), color: CHART_COLORS.orange },
                        ...(d.regime ? [{ label: "Regime", value: REGIME_BADGES[d.regime].label }] : []),
                      ]}
                    />
                  )
                }
                return null
              }}
            />
            <Legend {...legendConfig} />

            <ReferenceLine yAxisId="exposure" y={maxLeverage} stroke={CHART_COLORS.red} strokeDasharray="5 3"
              label={{ value: `Max: ${maxLeverage}x`, position: "right", fill: CHART_COLORS.red, fontSize: 10 }}
            />
            <ReferenceLine yAxisId="exposure" y={1.0} stroke={CHART_COLORS.muted} strokeDasharray="4 4" />

            <Area
              yAxisId="exposure"
              type="monotone"
              dataKey="grossExposure"
              name="Gross Exposure"
              stroke={CHART_COLORS.blue}
              fill={CHART_COLORS.blue}
              fillOpacity={0.08}
              strokeWidth={1.5}
              activeDot={<CleanDot fill={CHART_COLORS.blue} />}
            />
            <Line
              yAxisId="exposure"
              type="monotone"
              dataKey="netExposure"
              name="Net Exposure"
              stroke={CHART_COLORS.green}
              strokeWidth={1.5}
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.green} />}
            />
            <Line
              yAxisId="exposure"
              type="monotone"
              dataKey="targetExposure"
              name="Target"
              stroke={CHART_COLORS.purple}
              strokeDasharray="5 3"
              strokeWidth={1.5}
              dot={false}
            />
            <Line
              yAxisId="scalar"
              type="monotone"
              dataKey="volatilityScalar"
              name="Vol Scalar"
              stroke={CHART_COLORS.orange}
              strokeWidth={1}
              dot={false}
              opacity={0.7}
            />
          </ComposedChart>
        </ResponsiveContainer>

        <div className="mt-3 grid grid-cols-4 gap-4 text-xs">
          <div className="text-center">
            <div className="text-muted-foreground">Avg Exposure</div>
            <div className="font-medium font-mono text-foreground">{avgExposure.toFixed(2)}x</div>
          </div>
          <div className="text-center">
            <div className="text-muted-foreground">Min Exposure</div>
            <div className="font-medium font-mono text-foreground">{minExposure.toFixed(2)}x</div>
          </div>
          <div className="text-center">
            <div className="text-muted-foreground">Max Exposure</div>
            <div className="font-medium font-mono text-foreground">{maxExposure.toFixed(2)}x</div>
          </div>
          <div className="text-center">
            <div className="text-muted-foreground">Avg Vol Scalar</div>
            <div className="font-medium font-mono text-foreground">
              {(data.reduce((sum, d) => sum + d.volatilityScalar, 0) / data.length).toFixed(2)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
