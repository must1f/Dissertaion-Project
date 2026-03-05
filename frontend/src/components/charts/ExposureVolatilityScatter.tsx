import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ZAxis,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card"
import {
  CHART_COLORS,
  ChartTooltip,
  commonAxisProps,
  scatterGridConfig,
  cursorConfig,
  legendConfig,
} from "./chartTheme"

interface ScatterData {
  volatility: number
  exposure: number
  returns: number
  date: string
  regime?: "low_vol" | "normal" | "high_vol"
}

interface ExposureVolatilityScatterProps {
  data: ScatterData[]
  title?: string
  description?: string
  targetVol?: number
  height?: number
}

const REGIME_SCATTER_COLORS = {
  low_vol: CHART_COLORS.green,
  normal: CHART_COLORS.orange,
  high_vol: CHART_COLORS.red,
}

export function ExposureVolatilityScatter({
  data,
  title = "Exposure vs Volatility",
  description = "Relationship between market volatility and portfolio exposure",
  targetVol = 0.15,
  height = 400,
}: ExposureVolatilityScatterProps) {
  const lowVolData = data.filter(d => d.regime === "low_vol")
  const normalData = data.filter(d => d.regime === "normal")
  const highVolData = data.filter(d => d.regime === "high_vol")
  const noRegimeData = data.filter(d => !d.regime)

  const correlation = calculateCorrelation(
    data.map(d => d.volatility),
    data.map(d => d.exposure)
  )

  const volRange = [
    Math.min(...data.map(d => d.volatility)),
    Math.max(...data.map(d => d.volatility)),
  ]
  const expectedLineData = Array.from({ length: 20 }, (_, i) => {
    const vol = volRange[0] + (volRange[1] - volRange[0]) * (i / 19)
    return { volatility: vol, expectedExposure: Math.min(2.0, Math.max(0.1, targetVol / vol)) }
  })

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
              <div className="text-xs text-muted-foreground">Correlation</div>
              <div className="text-lg font-semibold font-mono" style={{
                color: correlation < -0.3 ? CHART_COLORS.green : correlation < 0 ? CHART_COLORS.orange : CHART_COLORS.red
              }}>
                {correlation.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Expected</div>
              <div className="text-base font-medium">Negative</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Target Vol</div>
              <div className="text-base font-medium font-mono">{(targetVol * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid {...scatterGridConfig} />

            <XAxis
              type="number"
              dataKey="volatility"
              name="Volatility"
              domain={["auto", "auto"]}
              {...commonAxisProps}
              tickMargin={8}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              label={{ value: "Realized Volatility", position: "insideBottom", offset: -10, fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="exposure"
              name="Exposure"
              domain={[0, "auto"]}
              {...commonAxisProps}
              tickMargin={6}
              width={58}
              tickFormatter={(v) => `${v.toFixed(1)}x`}
              label={{ value: "Portfolio Exposure", angle: -90, position: "insideLeft", fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
            />
            <ZAxis type="number" dataKey="returns" range={[20, 200]} name="Returns" />

            <Tooltip
              cursor={{ ...cursorConfig, strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload as ScatterData
                  return (
                    <ChartTooltip
                      active={active}
                      title={d.date}
                      rows={[
                        { label: "Volatility", value: `${(d.volatility * 100).toFixed(1)}%` },
                        { label: "Exposure", value: `${d.exposure.toFixed(2)}x` },
                        {
                          label: "Return",
                          value: `${(d.returns * 100).toFixed(2)}%`,
                          color: d.returns >= 0 ? CHART_COLORS.green : CHART_COLORS.red,
                        },
                        ...(d.regime ? [{ label: "Regime", value: d.regime.replace("_", " ") }] : []),
                      ]}
                    />
                  )
                }
                return null
              }}
            />
            <Legend {...legendConfig} />

            <ReferenceLine x={targetVol} stroke={CHART_COLORS.blue} strokeDasharray="5 3"
              label={{ value: "Target Vol", position: "top", fill: CHART_COLORS.blue, fontSize: 10 }}
            />
            <ReferenceLine y={1.0} stroke={CHART_COLORS.muted} strokeDasharray="4 4" />

            {/* Theoretical vol-targeting curve */}
            <Scatter
              name="Expected (Vol Targeting)"
              data={expectedLineData}
              dataKey="expectedExposure"
              fill="none"
              line={{ stroke: CHART_COLORS.purple, strokeDasharray: "5 3" }}
              shape={() => null}
            />

            {noRegimeData.length > 0 && (
              <Scatter name="Data Points" data={noRegimeData} fill={CHART_COLORS.blue} opacity={0.6} />
            )}
            {lowVolData.length > 0 && (
              <Scatter name="Low Vol" data={lowVolData} fill={REGIME_SCATTER_COLORS.low_vol} opacity={0.6} />
            )}
            {normalData.length > 0 && (
              <Scatter name="Normal" data={normalData} fill={REGIME_SCATTER_COLORS.normal} opacity={0.6} />
            )}
            {highVolData.length > 0 && (
              <Scatter name="High Vol" data={highVolData} fill={REGIME_SCATTER_COLORS.high_vol} opacity={0.6} />
            )}
          </ScatterChart>
        </ResponsiveContainer>

        <div className="mt-4 rounded-md border border-border/40 bg-muted/30 p-3 text-xs text-muted-foreground leading-relaxed">
          <p>
            With volatility targeting, the expected relationship is{" "}
            <strong className="text-foreground">negative</strong>:{" "}
            <em>exposure = σ<sub>target</sub> / σ<sub>realized</sub></em>.
            The purple dashed curve shows this theoretical relationship.
          </p>
          <p className="mt-1.5">
            Observed ρ ={" "}
            <span className="font-mono font-medium" style={{
              color: correlation < -0.3 ? CHART_COLORS.green : correlation < 0 ? CHART_COLORS.orange : CHART_COLORS.red
            }}>
              {correlation.toFixed(3)}
            </span>{" "}
            {correlation < -0.3
              ? "— strategy appropriately reduces exposure when volatility rises."
              : correlation < 0
                ? "— moderate volatility responsiveness."
                : "— exposure may not be responding sufficiently to volatility."}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length
  if (n === 0) return 0
  const sumX = x.reduce((a, b) => a + b, 0)
  const sumY = y.reduce((a, b) => a + b, 0)
  const sumXY = x.reduce((total, xi, i) => total + xi * y[i], 0)
  const sumX2 = x.reduce((total, xi) => total + xi * xi, 0)
  const sumY2 = y.reduce((total, yi) => total + yi * yi, 0)
  const numerator = n * sumXY - sumX * sumY
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
  return denominator === 0 ? 0 : numerator / denominator
}
