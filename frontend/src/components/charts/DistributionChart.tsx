import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card"
import {
  CHART_COLORS,
  commonXAxisProps,
  commonYAxisProps,
  gridConfig,
  cursorConfig,
  tooltipContentStyle,
  tooltipLabelStyle,
} from "./chartTheme"

interface DistributionData {
  bin: number
  count: number
}

interface DistributionChartProps {
  data: DistributionData[]
  title?: string
  description?: string
  height?: number
  mean?: number
  median?: number
  showStats?: boolean
}

export function DistributionChart({
  data,
  title = "Distribution",
  description,
  height = 300,
  mean,
  median,
  showStats = true,
}: DistributionChartProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          {showStats && (mean !== undefined || median !== undefined) && (
            <div className="flex gap-5 text-sm uppercase">
              {mean !== undefined && (
                <div className="text-right">
                  <div className="text-xs text-muted-foreground font-mono">Mean</div>
                  <div className="font-mono font-bold text-amber-500">${mean.toFixed(2)}</div>
                </div>
              )}
              {median !== undefined && (
                <div className="text-right">
                  <div className="text-xs text-muted-foreground font-mono">Median</div>
                  <div className="font-mono font-bold text-[#00ffff]">${median.toFixed(2)}</div>
                </div>
              )}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={data}>
            <CartesianGrid {...gridConfig} />
            <XAxis
              dataKey="bin"
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10, fontFamily: "monospace" }}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={{ stroke: "hsl(var(--border))" }}
            />
            <YAxis
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10, fontFamily: "monospace" }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={{ stroke: "hsl(var(--border))" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#000000",
                borderColor: "rgba(245, 158, 11, 0.3)",
                color: "#f59e0b",
                fontFamily: "monospace",
              }}
              labelStyle={{ color: "#ffffff", fontWeight: "bold" }}
              cursor={{ fill: "hsl(var(--muted-foreground) / 0.1)" }}
              formatter={(v: number | string | undefined) => [v ?? 0, "COUNT"]}
              labelFormatter={(label) => `PRICE: $${Number(label).toFixed(2)}`}
            />

            {mean !== undefined && (
              <ReferenceLine
                x={mean}
                stroke="#f59e0b"
                strokeDasharray="5 3"
                strokeWidth={1.5}
                label={{ value: "MEAN", position: "top", fill: "#f59e0b", fontSize: 10, fontFamily: "monospace" }}
              />
            )}
            {median !== undefined && (
              <ReferenceLine
                x={median}
                stroke="#00ffff"
                strokeDasharray="5 3"
                strokeWidth={1.5}
                label={{ value: "MEDIAN", position: "top", fill: "#00ffff", fontSize: 10, fontFamily: "monospace" }}
              />
            )}

            <Bar
              dataKey="count"
              fill="hsl(var(--primary))"
              opacity={0.8}
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
