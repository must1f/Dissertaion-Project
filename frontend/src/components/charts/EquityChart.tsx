import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
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
  legendConfig,
} from "./chartTheme"

interface EquityData {
  date: string
  value: number
  benchmark?: number
}

interface EquityChartProps {
  data: EquityData[]
  title?: string
  description?: string
  initialCapital?: number
  showBenchmark?: boolean
  height?: number
}

export function EquityChart({
  data,
  title = "Equity Curve",
  description,
  initialCapital = 100000,
  showBenchmark = false,
  height = 400,
}: EquityChartProps) {
  const finalValue = data[data.length - 1]?.value || initialCapital
  const totalReturn = ((finalValue - initialCapital) / initialCapital) * 100

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="text-right">
            <div className="text-xl font-semibold font-mono tracking-tight">
              ${finalValue.toLocaleString()}
            </div>
            <div className={`text-sm font-medium ${totalReturn >= 0 ? "text-[#009E73]" : "text-[#D55E00]"}`}>
              {totalReturn >= 0 ? "+" : ""}{totalReturn.toFixed(2)}%
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
              {...commonYAxisProps}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
            />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(value: any, name: any) => [`$${Number(value).toLocaleString()}`, name]}
            />
            <Legend {...legendConfig} />

            {/* Initial capital reference */}
            <ReferenceLine
              y={initialCapital}
              stroke={CHART_COLORS.muted}
              strokeDasharray="4 4"
              label={{
                value: "Initial Capital",
                position: "right",
                fill: "hsl(var(--muted-foreground))",
                fontSize: 10,
              }}
            />

            {/* Benchmark — grey dashed */}
            {showBenchmark && (
              <Area
                type="monotone"
                dataKey="benchmark"
                name="Benchmark"
                stroke={CHART_COLORS.benchmark}
                fill={CHART_COLORS.benchmark}
                fillOpacity={0.06}
                strokeWidth={1.5}
                strokeDasharray="5 3"
                dot={false}
              />
            )}

            {/* Portfolio — solid blue with light fill */}
            <Area
              type="monotone"
              dataKey="value"
              name="Portfolio"
              stroke={CHART_COLORS.blue}
              fill={CHART_COLORS.blue}
              fillOpacity={0.08}
              strokeWidth={2}
              activeDot={<CleanDot fill={CHART_COLORS.blue} />}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
