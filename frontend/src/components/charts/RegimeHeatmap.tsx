/**
 * RegimeHeatmap - Regime probability visualization over time
 *
 * Shows regime probabilities as a stacked area chart with:
 * - Color-coded regimes (green=low vol, yellow=normal, red=high vol)
 * - Smooth transitions between regimes
 * - Current regime indicator
 */
import React from "react"
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts"
import {
    commonXAxisProps,
    commonYAxisProps,
    gridConfig,
    tooltipContentStyle,
    tooltipLabelStyle,
} from "./chartTheme"

interface RegimeHistoryPoint {
    date: string
    probabilities: {
        "Low Volatility": number
        "Normal": number
        "High Volatility": number
    }
    dominantRegime: string
}

interface RegimeHeatmapProps {
    data: RegimeHistoryPoint[]
    title?: string
    currentRegime?: string
    currentProbability?: number
}

const REGIME_COLORS = {
    "Low Volatility": "#10b981",  // Green
    "Normal": "#f59e0b",          // Amber
    "High Volatility": "#ef4444", // Red
}

export function RegimeHeatmap({
    data,
    title = "Regime Probabilities",
    currentRegime,
    currentProbability,
}: RegimeHeatmapProps) {
    // Transform data for stacked area chart
    const chartData = data.map((point) => ({
        date: point.date,
        lowVol: point.probabilities["Low Volatility"] * 100,
        normal: point.probabilities["Normal"] * 100,
        highVol: point.probabilities["High Volatility"] * 100,
        dominant: point.dominantRegime,
    }))

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload?.length) return null

        const d = payload[0].payload
        return (
            <div style={tooltipContentStyle}>
                <div style={tooltipLabelStyle}>{label}</div>
                <div className="space-y-1 text-xs">
                    <div className="flex justify-between gap-4">
                        <span className="flex items-center gap-1.5">
                            <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: REGIME_COLORS["Low Volatility"] }}
                            />
                            Low Vol
                        </span>
                        <span className="font-medium">{d.lowVol.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="flex items-center gap-1.5">
                            <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: REGIME_COLORS["Normal"] }}
                            />
                            Normal
                        </span>
                        <span className="font-medium">{d.normal.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="flex items-center gap-1.5">
                            <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: REGIME_COLORS["High Volatility"] }}
                            />
                            High Vol
                        </span>
                        <span className="font-medium">{d.highVol.toFixed(1)}%</span>
                    </div>
                    <div className="pt-1 border-t border-gray-600">
                        <span className="text-gray-400">Dominant: </span>
                        <span className="font-medium">{d.dominant}</span>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-foreground">{title}</h3>
                {currentRegime && (
                    <div className="flex items-center gap-2 text-xs">
                        <span className="text-muted-foreground">Current:</span>
                        <span
                            className="px-2 py-0.5 rounded font-medium text-white"
                            style={{
                                backgroundColor:
                                    REGIME_COLORS[currentRegime as keyof typeof REGIME_COLORS] ||
                                    "#666",
                            }}
                        >
                            {currentRegime}
                            {currentProbability !== undefined &&
                                ` (${(currentProbability * 100).toFixed(0)}%)`}
                        </span>
                    </div>
                )}
            </div>

            <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} stackOffset="expand">
                        <CartesianGrid {...gridConfig} />
                        <XAxis
                            dataKey="date"
                            {...commonXAxisProps}
                            tickFormatter={(v) => {
                                const date = new Date(v)
                                return `${date.getMonth() + 1}/${date.getDate()}`
                            }}
                        />
                        <YAxis
                            {...commonYAxisProps}
                            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                            domain={[0, 1]}
                        />
                        <Tooltip content={<CustomTooltip />} />

                        <Area
                            type="monotone"
                            dataKey="lowVol"
                            stackId="1"
                            stroke={REGIME_COLORS["Low Volatility"]}
                            fill={REGIME_COLORS["Low Volatility"]}
                            fillOpacity={0.7}
                            name="Low Volatility"
                        />
                        <Area
                            type="monotone"
                            dataKey="normal"
                            stackId="1"
                            stroke={REGIME_COLORS["Normal"]}
                            fill={REGIME_COLORS["Normal"]}
                            fillOpacity={0.7}
                            name="Normal"
                        />
                        <Area
                            type="monotone"
                            dataKey="highVol"
                            stackId="1"
                            stroke={REGIME_COLORS["High Volatility"]}
                            fill={REGIME_COLORS["High Volatility"]}
                            fillOpacity={0.7}
                            name="High Volatility"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 text-xs">
                {Object.entries(REGIME_COLORS).map(([name, color]) => (
                    <div key={name} className="flex items-center gap-1.5">
                        <div
                            className="w-3 h-3 rounded-sm"
                            style={{ backgroundColor: color }}
                        />
                        <span className="text-muted-foreground">{name}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

export default RegimeHeatmap
