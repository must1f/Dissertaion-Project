/**
 * MonteCarloFanChart - Fan chart with regime-colored confidence bands
 *
 * Displays Monte Carlo simulation results with:
 * - Nested confidence bands (90%, 50%)
 * - Median projection line
 * - Background regions colored by dominant regime
 * - Clear percentile labels
 */
import React from "react"
import {
    ComposedChart,
    Area,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceArea,
    ReferenceLine,
    Legend,
} from "recharts"
import {
    CHART_COLORS,
    commonXAxisProps,
    commonYAxisProps,
    gridConfig,
    tooltipContentStyle,
    tooltipLabelStyle,
} from "./chartTheme"

interface PercentileBand {
    percentile: number
    values: number[]
}

interface RegimePeriod {
    start: number
    end: number
    regime: number
    regime_name: string
}

interface MonteCarloFanChartProps {
    dates: number[]
    percentileBands: PercentileBand[]
    medianPath: number[]
    regimePeriods?: RegimePeriod[]
    initialPrice: number
    title?: string
    showRegimeColors?: boolean
}

const REGIME_COLORS = {
    0: "#10b98120", // Low Vol - Green with transparency
    1: "#f59e0b15", // Normal - Amber with transparency
    2: "#ef444420", // High Vol - Red with transparency
}

const BAND_COLORS = {
    outer: "#3b82f620", // 5-95% band
    inner: "#3b82f640", // 25-75% band
    median: "#3b82f6",  // 50% line
}

export function MonteCarloFanChart({
    dates,
    percentileBands,
    medianPath,
    regimePeriods = [],
    initialPrice,
    title = "Monte Carlo Projection",
    showRegimeColors = true,
}: MonteCarloFanChartProps) {
    // Build chart data combining all bands
    const bandMap = new Map<number, number[]>()
    percentileBands.forEach((band) => {
        bandMap.set(band.percentile, band.values)
    })

    const p5 = bandMap.get(5) || []
    const p25 = bandMap.get(25) || []
    const p50 = bandMap.get(50) || medianPath
    const p75 = bandMap.get(75) || []
    const p95 = bandMap.get(95) || []

    const chartData = dates.map((day, i) => ({
        day,
        p5: p5[i],
        p25: p25[i],
        p50: p50[i],
        p75: p75[i],
        p95: p95[i],
        // For area chart, we need the range
        p5_25: p25[i] - p5[i],
        p25_75: p75[i] - p25[i],
        p75_95: p95[i] - p75[i],
    }))

    const minPrice = Math.min(...p5.filter(Boolean))
    const maxPrice = Math.max(...p95.filter(Boolean))

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload?.length) return null

        const d = chartData.find((x) => x.day === label)
        if (!d) return null

        return (
            <div style={tooltipContentStyle}>
                <div style={tooltipLabelStyle}>Day {label}</div>
                <div className="space-y-1 text-xs">
                    {d.p95 && (
                        <div className="flex justify-between gap-4 text-gray-400">
                            <span>95th %ile</span>
                            <span className="font-medium">${d.p95.toFixed(2)}</span>
                        </div>
                    )}
                    {d.p75 && (
                        <div className="flex justify-between gap-4 text-gray-400">
                            <span>75th %ile</span>
                            <span className="font-medium">${d.p75.toFixed(2)}</span>
                        </div>
                    )}
                    <div className="flex justify-between gap-4 text-blue-400">
                        <span>Median</span>
                        <span className="font-medium">${d.p50?.toFixed(2)}</span>
                    </div>
                    {d.p25 && (
                        <div className="flex justify-between gap-4 text-gray-400">
                            <span>25th %ile</span>
                            <span className="font-medium">${d.p25.toFixed(2)}</span>
                        </div>
                    )}
                    {d.p5 && (
                        <div className="flex justify-between gap-4 text-gray-400">
                            <span>5th %ile</span>
                            <span className="font-medium">${d.p5.toFixed(2)}</span>
                        </div>
                    )}
                    <div className="pt-1 border-t border-gray-600">
                        <span className="text-gray-400">From start: </span>
                        <span
                            className={`font-medium ${
                                d.p50 >= initialPrice ? "text-green-400" : "text-red-400"
                            }`}
                        >
                            {((d.p50 / initialPrice - 1) * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-foreground">{title}</h3>
                <div className="flex gap-4 text-xs text-muted-foreground">
                    <span>
                        Initial: <span className="font-medium text-foreground">${initialPrice.toFixed(2)}</span>
                    </span>
                    <span>
                        Horizon: <span className="font-medium text-foreground">{dates.length - 1} days</span>
                    </span>
                </div>
            </div>

            <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData}>
                        <defs>
                            <linearGradient id="fanOuter" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.15} />
                                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.15} />
                            </linearGradient>
                            <linearGradient id="fanInner" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.25} />
                                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.25} />
                            </linearGradient>
                        </defs>

                        <CartesianGrid {...gridConfig} />

                        {/* Regime background colors */}
                        {showRegimeColors &&
                            regimePeriods.map((period, i) => (
                                <ReferenceArea
                                    key={i}
                                    x1={period.start}
                                    x2={period.end}
                                    y1={minPrice * 0.95}
                                    y2={maxPrice * 1.05}
                                    fill={REGIME_COLORS[period.regime as keyof typeof REGIME_COLORS] || "#00000010"}
                                    stroke="none"
                                />
                            ))}

                        <XAxis
                            dataKey="day"
                            {...commonXAxisProps}
                            tickFormatter={(v) => `D${v}`}
                            label={{
                                value: "Trading Days",
                                position: "bottom",
                                offset: -5,
                                style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                            }}
                        />
                        <YAxis
                            {...commonYAxisProps}
                            tickFormatter={(v) => `$${v.toFixed(0)}`}
                            domain={[minPrice * 0.95, maxPrice * 1.05]}
                        />
                        <Tooltip content={<CustomTooltip />} />

                        {/* Initial price reference */}
                        <ReferenceLine
                            y={initialPrice}
                            stroke={CHART_COLORS.muted}
                            strokeDasharray="4 4"
                            strokeWidth={1}
                        />

                        {/* 90% CI band (p5 to p95) */}
                        <Area
                            type="monotone"
                            dataKey="p95"
                            stroke="none"
                            fill="url(#fanOuter)"
                            stackId="ci"
                        />
                        <Area
                            type="monotone"
                            dataKey="p5"
                            stroke="none"
                            fill="transparent"
                            stackId="ci"
                        />

                        {/* 50% CI band (p25 to p75) */}
                        <Area
                            type="monotone"
                            dataKey="p75"
                            stroke="none"
                            fill="url(#fanInner)"
                        />
                        <Area
                            type="monotone"
                            dataKey="p25"
                            stroke="none"
                            fill="#ffffff"
                        />

                        {/* Median line */}
                        <Line
                            type="monotone"
                            dataKey="p50"
                            stroke={BAND_COLORS.median}
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, fill: BAND_COLORS.median, stroke: "#fff" }}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 text-xs">
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-0.5" style={{ backgroundColor: BAND_COLORS.median }} />
                    <span className="text-muted-foreground">Median</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: "#3b82f6", opacity: 0.25 }}
                    />
                    <span className="text-muted-foreground">50% CI</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: "#3b82f6", opacity: 0.15 }}
                    />
                    <span className="text-muted-foreground">90% CI</span>
                </div>
                {showRegimeColors && regimePeriods.length > 0 && (
                    <>
                        <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-sm bg-green-500/20" />
                            <span className="text-muted-foreground">Low Vol</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-sm bg-red-500/20" />
                            <span className="text-muted-foreground">High Vol</span>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}

export default MonteCarloFanChart
