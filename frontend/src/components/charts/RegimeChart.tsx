import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
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
} from "./chartTheme"

type RegimeType = "low_vol" | "normal" | "high_vol"

interface RegimeData {
  date: string
  value: number
  regime: RegimeType
  probability?: number
  volatility?: number
}

interface RegimeChartProps {
  data: RegimeData[]
  title?: string
  description?: string
  currentRegime?: RegimeType
  height?: number
}

const REGIME_CONFIG = {
  low_vol: { label: "Low Volatility", color: CHART_COLORS.green, fillOpacity: 0.05, badgeVariant: "default" as const },
  normal: { label: "Normal", color: CHART_COLORS.orange, fillOpacity: 0.05, badgeVariant: "secondary" as const },
  high_vol: { label: "High Volatility", color: CHART_COLORS.red, fillOpacity: 0.05, badgeVariant: "destructive" as const },
}

export function RegimeChart({
  data,
  title = "Market Regime Analysis",
  description,
  currentRegime,
  height = 350,
}: RegimeChartProps) {
  const regimeDistribution = data.reduce((acc, d) => {
    acc[d.regime] = (acc[d.regime] || 0) + 1
    return acc
  }, {} as Record<RegimeType, number>)

  const total = data.length
  const regimePercentages = Object.entries(regimeDistribution).map(([regime, count]) => ({
    regime: regime as RegimeType,
    percentage: (count / total) * 100,
  }))

  const regimeAreas: { start: string; end: string; regime: RegimeType }[] = []
  if (data.length > 0) {
    let currentRegimeArea = data[0].regime
    let startIdx = 0
    for (let i = 1; i < data.length; i++) {
      if (data[i].regime !== currentRegimeArea) {
        regimeAreas.push({ start: data[startIdx].date, end: data[i - 1].date, regime: currentRegimeArea })
        currentRegimeArea = data[i].regime
        startIdx = i
      }
    }
    regimeAreas.push({ start: data[startIdx].date, end: data[data.length - 1].date, regime: currentRegimeArea })
  }

  const detectedRegime = currentRegime || (data.length > 0 ? data[data.length - 1].regime : "normal")

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="text-right">
            <div className="text-xs text-muted-foreground">Current Regime</div>
            <Badge variant={REGIME_CONFIG[detectedRegime].badgeVariant} className="mt-1">
              {REGIME_CONFIG[detectedRegime].label}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data}>
            <CartesianGrid {...gridConfig} />

            {regimeAreas.map((area, idx) => (
              <ReferenceArea
                key={idx}
                x1={area.start}
                x2={area.end}
                fill={REGIME_CONFIG[area.regime].color}
                fillOpacity={REGIME_CONFIG[area.regime].fillOpacity}
              />
            ))}

            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis {...commonYAxisProps} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`} />
            <Tooltip
              cursor={cursorConfig}
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload as RegimeData
                  return (
                    <ChartTooltip
                      active={active}
                      title={label as string}
                      rows={[
                        { label: "Value", value: `$${d.value.toLocaleString()}` },
                        { label: "Regime", value: REGIME_CONFIG[d.regime].label, color: REGIME_CONFIG[d.regime].color },
                        ...(d.volatility !== undefined ? [{ label: "Volatility", value: `${(d.volatility * 100).toFixed(1)}%` }] : []),
                        ...(d.probability !== undefined ? [{ label: "Confidence", value: `${(d.probability * 100).toFixed(0)}%` }] : []),
                      ]}
                    />
                  )
                }
                return null
              }}
            />

            <Area
              type="monotone"
              dataKey="value"
              stroke={CHART_COLORS.blue}
              fill={CHART_COLORS.blue}
              fillOpacity={0.06}
              strokeWidth={1.5}
              activeDot={<CleanDot fill={CHART_COLORS.blue} />}
            />
          </AreaChart>
        </ResponsiveContainer>

        <div className="mt-3 flex items-center justify-center gap-6">
          {regimePercentages.map(({ regime, percentage }) => (
            <div key={regime} className="flex items-center gap-1.5">
              <div className="h-2.5 w-5 rounded-sm" style={{ backgroundColor: REGIME_CONFIG[regime].color, opacity: 0.3 }} />
              <span className="text-xs text-muted-foreground">
                {REGIME_CONFIG[regime].label}: <span className="font-medium font-mono text-foreground">{percentage.toFixed(1)}%</span>
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
