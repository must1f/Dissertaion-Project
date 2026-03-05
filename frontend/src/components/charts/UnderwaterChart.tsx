import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Scatter,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card"
import {
  CHART_COLORS,
  CleanDot,
  ChartTooltip,
  commonXAxisProps,
  commonYAxisProps,
  gridConfig,
  cursorConfig,
} from "./chartTheme"

interface UnderwaterData {
  date: string
  drawdown: number
  daysUnderwater: number
  isRecovery?: boolean
}

interface DrawdownEvent {
  startDate: string
  endDate: string
  maxDrawdown: number
  duration: number
  recoveryDays?: number
}

interface UnderwaterChartProps {
  data: UnderwaterData[]
  title?: string
  description?: string
  majorDrawdowns?: DrawdownEvent[]
  height?: number
}

export function UnderwaterChart({
  data,
  title = "Underwater Chart",
  description = "Drawdown depth and recovery periods",
  majorDrawdowns,
  height = 300,
}: UnderwaterChartProps) {
  const maxDrawdown = Math.min(...data.map(d => d.drawdown))
  const maxDaysUnderwater = Math.max(...data.map(d => d.daysUnderwater))
  const currentDrawdown = data[data.length - 1]?.drawdown || 0
  const currentDaysUnderwater = data[data.length - 1]?.daysUnderwater || 0
  const recoveryPoints = data.filter(d => d.isRecovery)
  const avgDrawdown = data.reduce((sum, d) => sum + d.drawdown, 0) / data.length
  const underwaterDays = data.filter(d => d.drawdown < -0.01).length
  const underwaterPct = (underwaterDays / data.length) * 100

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="flex items-center gap-5 text-right">
            <div>
              <div className="text-xs text-muted-foreground">Current DD</div>
              <div className="text-lg font-semibold font-mono" style={{
                color: currentDrawdown < -0.05 ? CHART_COLORS.red : currentDrawdown < 0 ? CHART_COLORS.orange : CHART_COLORS.green
              }}>
                {(currentDrawdown * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Max DD</div>
              <div className="text-base font-semibold font-mono" style={{ color: CHART_COLORS.red }}>
                {(maxDrawdown * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Days Under</div>
              <div className="text-base font-medium font-mono">{currentDaysUnderwater}</div>
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
              domain={[Math.min(maxDrawdown * 1.1, -0.3), 0.02]}
              {...commonYAxisProps}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              cursor={cursorConfig}
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload as UnderwaterData
                  return (
                    <ChartTooltip
                      active={active}
                      title={label as string}
                      rows={[
                        { label: "Drawdown", value: `${(d.drawdown * 100).toFixed(2)}%`, color: CHART_COLORS.red },
                        { label: "Days underwater", value: String(d.daysUnderwater) },
                        ...(d.isRecovery ? [{ label: "Status", value: "Recovery ✓", color: CHART_COLORS.green }] : []),
                      ]}
                    />
                  )
                }
                return null
              }}
            />

            <ReferenceLine y={0} stroke={CHART_COLORS.border} strokeWidth={1} />
            <ReferenceLine y={-0.10} stroke={CHART_COLORS.orange} strokeDasharray="5 3"
              label={{ value: "-10%", position: "left", fill: CHART_COLORS.orange, fontSize: 10 }}
            />
            <ReferenceLine y={-0.20} stroke={CHART_COLORS.red} strokeDasharray="5 3"
              label={{ value: "-20%", position: "left", fill: CHART_COLORS.red, fontSize: 10 }}
            />
            <ReferenceLine y={maxDrawdown} stroke={CHART_COLORS.red} strokeDasharray="3 3" strokeWidth={1.5}
              label={{ value: `Max: ${(maxDrawdown * 100).toFixed(1)}%`, position: "right", fill: CHART_COLORS.red, fontSize: 10 }}
            />

            <Area
              type="monotone"
              dataKey="drawdown"
              stroke={CHART_COLORS.red}
              fill={CHART_COLORS.red}
              fillOpacity={0.1}
              strokeWidth={1.5}
              activeDot={<CleanDot fill={CHART_COLORS.red} />}
            />

            {/* Recovery markers */}
            <Scatter
              dataKey="drawdown"
              data={recoveryPoints}
              fill={CHART_COLORS.green}
              shape={(props: any) => {
                const { cx, cy } = props
                if (!props.payload.isRecovery) return null
                return <circle cx={cx} cy={cy} r={3} fill={CHART_COLORS.green} stroke="#fff" strokeWidth={1} />
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>

        <div className="mt-3 grid grid-cols-4 gap-4 text-xs">
          <div className="text-center">
            <div className="text-muted-foreground">Avg Drawdown</div>
            <div className="font-medium font-mono text-foreground">{(avgDrawdown * 100).toFixed(2)}%</div>
          </div>
          <div className="text-center">
            <div className="text-muted-foreground">Max Days Under</div>
            <div className="font-medium font-mono text-foreground">{maxDaysUnderwater}</div>
          </div>
          <div className="text-center">
            <div className="text-muted-foreground">Time Underwater</div>
            <div className="font-medium font-mono text-foreground">{underwaterPct.toFixed(0)}%</div>
          </div>
          <div className="text-center">
            <div className="text-muted-foreground">Recoveries</div>
            <div className="font-medium font-mono" style={{ color: CHART_COLORS.green }}>{recoveryPoints.length}</div>
          </div>
        </div>

        {majorDrawdowns && majorDrawdowns.length > 0 && (
          <div className="mt-4">
            <h4 className="mb-2 text-xs font-medium text-muted-foreground">Major Drawdowns</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/40">
                    <th className="px-2 py-1.5 text-left text-muted-foreground font-medium">Period</th>
                    <th className="px-2 py-1.5 text-right text-muted-foreground font-medium">Depth</th>
                    <th className="px-2 py-1.5 text-right text-muted-foreground font-medium">Duration</th>
                    <th className="px-2 py-1.5 text-right text-muted-foreground font-medium">Recovery</th>
                  </tr>
                </thead>
                <tbody>
                  {majorDrawdowns.slice(0, 5).map((dd, idx) => (
                    <tr key={idx} className="border-b border-border/20">
                      <td className="px-2 py-1.5 font-mono">{dd.startDate}</td>
                      <td className="px-2 py-1.5 text-right font-mono" style={{ color: CHART_COLORS.red }}>
                        {(dd.maxDrawdown * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-1.5 text-right font-mono">{dd.duration}d</td>
                      <td className="px-2 py-1.5 text-right font-mono">{dd.recoveryDays ? `${dd.recoveryDays}d` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
