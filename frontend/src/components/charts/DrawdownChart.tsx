import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
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

interface DrawdownData {
  date: string
  drawdown: number
}

interface DrawdownChartProps {
  data: DrawdownData[]
  title?: string
  description?: string
  height?: number
}

export function DrawdownChart({
  data,
  title = "Drawdown",
  description,
  height = 250,
}: DrawdownChartProps) {
  const maxDrawdown = Math.min(...data.map((d) => d.drawdown))

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="text-right">
            <div className="text-xs text-muted-foreground">Max Drawdown</div>
            <div className="text-lg font-semibold font-mono" style={{ color: CHART_COLORS.red }}>
              {(maxDrawdown * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data}>
            <CartesianGrid {...gridConfig} />
            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              domain={["auto", 0]}
              {...commonYAxisProps}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(v: any) => [`${(Number(v) * 100).toFixed(2)}%`, "Drawdown"]}
            />

            <ReferenceLine y={0} stroke={CHART_COLORS.border} />

            <ReferenceLine
              y={maxDrawdown}
              stroke={CHART_COLORS.red}
              strokeDasharray="4 4"
              label={{
                value: `Max: ${(maxDrawdown * 100).toFixed(1)}%`,
                position: "right",
                fill: CHART_COLORS.red,
                fontSize: 10,
              }}
            />

            <Area
              type="monotone"
              dataKey="drawdown"
              stroke={CHART_COLORS.red}
              fill={CHART_COLORS.red}
              fillOpacity={0.12}
              strokeWidth={1.5}
              activeDot={<CleanDot fill={CHART_COLORS.red} />}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
