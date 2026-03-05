import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
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

interface PriceChartProps {
  data: Array<{
    date: string
    open: number
    high: number
    low: number
    close: number
    volume?: number
  }>
  title?: string
  showVolume?: boolean
  height?: number
}

export function PriceChart({
  data,
  title = "Price Chart",
  showVolume = true,
  height = 400,
}: PriceChartProps) {
  const chartData = data.map((d) => ({
    ...d,
    date: new Date(d.date).toLocaleDateString(),
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={chartData}>
            <CartesianGrid {...gridConfig} />
            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              yAxisId="price"
              domain={["auto", "auto"]}
              {...commonYAxisProps}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            {showVolume && (
              <YAxis
                yAxisId="volume"
                orientation="right"
                {...commonYAxisProps}
                tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`}
              />
            )}
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(value: any, name: any) => {
                if (name === "Volume") return [`${(Number(value) / 1e6).toFixed(2)}M`, name]
                return [`$${Number(value).toFixed(2)}`, name]
              }}
            />
            <Legend {...legendConfig} />

            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              name="Close"
              stroke={CHART_COLORS.blue}
              strokeWidth={1.5}
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.blue} />}
            />

            {showVolume && (
              <Bar
                yAxisId="volume"
                dataKey="volume"
                name="Volume"
                fill={CHART_COLORS.cyan}
                opacity={0.35}
                radius={[1, 1, 0, 0]}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
