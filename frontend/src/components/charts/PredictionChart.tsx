import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
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
  CIBandDefs,
  commonXAxisProps,
  commonYAxisProps,
  gridConfig,
  cursorConfig,
  tooltipContentStyle,
  tooltipLabelStyle,
  legendConfig,
} from "./chartTheme"

interface PredictionData {
  date: string
  actual?: number
  predicted?: number
  lower?: number
  upper?: number
}

interface PredictionChartProps {
  data: PredictionData[]
  title?: string
  description?: string
  showConfidenceInterval?: boolean
  height?: number
  currentPrice?: number
}

export function PredictionChart({
  data,
  title = "Price Predictions",
  description,
  showConfidenceInterval = true,
  height = 400,
  currentPrice,
}: PredictionChartProps) {
  const splitIndex = data.findIndex((d) => d.actual === undefined)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          {currentPrice && (
            <Badge variant="outline" className="text-base font-mono">
              ${currentPrice.toFixed(2)}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data}>
            <CIBandDefs />
            <CartesianGrid {...gridConfig} />
            <XAxis dataKey="date" {...commonXAxisProps} />
            <YAxis
              domain={["auto", "auto"]}
              {...commonYAxisProps}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
              cursor={cursorConfig}
              formatter={(value: any, name: any) => [`$${Number(value).toFixed(2)}`, name]}
            />
            <Legend {...legendConfig} />

            {/* 95% confidence band — flat semi-transparent fill */}
            {showConfidenceInterval && (
              <>
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill={CHART_COLORS.confidence}
                  fillOpacity={0.12}
                  name="95% CI Upper"
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill="hsl(var(--card))"
                  fillOpacity={1}
                  name="95% CI Lower"
                />
              </>
            )}

            {/* Actual observations — solid blue */}
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke={CHART_COLORS.actual}
              strokeWidth={1.5}
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.actual} />}
              connectNulls
            />

            {/* Model predictions — dashed orange */}
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted"
              stroke={CHART_COLORS.predicted}
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              activeDot={<CleanDot fill={CHART_COLORS.predicted} />}
            />

            {/* CI bounds — thin dashed lines */}
            {showConfidenceInterval && (
              <>
                <Line
                  type="monotone"
                  dataKey="upper"
                  name="Upper CI"
                  stroke={CHART_COLORS.confidence}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  legendType="none"
                />
                <Line
                  type="monotone"
                  dataKey="lower"
                  name="Lower CI"
                  stroke={CHART_COLORS.confidence}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  legendType="none"
                />
              </>
            )}

            {/* Split point — "Today" */}
            {splitIndex > 0 && (
              <ReferenceLine
                x={data[splitIndex - 1]?.date}
                stroke={CHART_COLORS.muted}
                strokeDasharray="4 4"
                label={{
                  value: "Today",
                  position: "top",
                  fill: "hsl(var(--muted-foreground))",
                  fontSize: 11,
                }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
