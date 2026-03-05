/**
 * SpectralAnalysisChart - Power spectrum visualization
 *
 * Displays frequency-domain analysis of financial returns:
 * - Bar chart showing power at each frequency bin
 * - Frequency band annotations (low/mid/high)
 * - Dominant frequency indicator
 */
import React from "react"
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    ReferenceArea,
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

interface PowerSpectrumData {
    frequencies: number[]
    power: number[]
    frequency_bands?: Record<string, { min_freq: number; max_freq: number; total_power: number; interpretation?: string }>
}

interface SpectralAnalysisChartProps {
    data: PowerSpectrumData
    dominantFrequency?: number
    spectralEntropy?: number
    title?: string
}

const BAND_COLORS = {
    low: "#3b82f6",   // Blue - trends
    mid: "#10b981",   // Green - cycles
    high: "#f59e0b",  // Amber - noise
}

export function SpectralAnalysisChart({
    data,
    dominantFrequency,
    spectralEntropy,
    title = "Power Spectrum",
}: SpectralAnalysisChartProps) {
    // Transform data for Recharts
    const chartData = data.frequencies.map((freq, i) => {
        // Determine frequency band
        let band: "low" | "mid" | "high" = "high"
        if (freq < 0.1) band = "low"
        else if (freq < 0.25) band = "mid"

        return {
            frequency: freq,
            power: data.power[i],
            band,
            period: freq > 0 ? (1 / freq).toFixed(1) : "inf",
        }
    }).filter((d) => d.frequency > 0) // Exclude DC component

    const maxPower = Math.max(...chartData.map((d) => d.power))

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload?.[0]) return null
        const d = payload[0].payload
        return (
            <div style={tooltipContentStyle}>
                <div style={tooltipLabelStyle}>Frequency Analysis</div>
                <div className="space-y-1 text-xs">
                    <div className="flex justify-between gap-4">
                        <span className="text-gray-400">Frequency</span>
                        <span className="font-medium">{d.frequency.toFixed(3)} cpd</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-gray-400">Period</span>
                        <span className="font-medium">{d.period} days</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-gray-400">Power</span>
                        <span className="font-medium">{(d.power * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-gray-400">Band</span>
                        <span
                            className="font-medium"
                            style={{ color: BAND_COLORS[d.band as keyof typeof BAND_COLORS] }}
                        >
                            {d.band === "low" ? "Trend" : d.band === "mid" ? "Cycle" : "Noise"}
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
                    {spectralEntropy !== undefined && (
                        <span>
                            Entropy:{" "}
                            <span className="font-medium text-foreground">
                                {(spectralEntropy * 100).toFixed(1)}%
                            </span>
                        </span>
                    )}
                    {dominantFrequency !== undefined && dominantFrequency > 0 && (
                        <span>
                            Dominant:{" "}
                            <span className="font-medium text-foreground">
                                {(1 / dominantFrequency).toFixed(1)} days
                            </span>
                        </span>
                    )}
                </div>
            </div>

            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barCategoryGap="5%">
                        <CartesianGrid {...gridConfig} />

                        {/* Frequency band backgrounds */}
                        <ReferenceArea
                            x1={0}
                            x2={0.1}
                            fill={BAND_COLORS.low}
                            fillOpacity={0.08}
                            stroke="none"
                        />
                        <ReferenceArea
                            x1={0.1}
                            x2={0.25}
                            fill={BAND_COLORS.mid}
                            fillOpacity={0.08}
                            stroke="none"
                        />
                        <ReferenceArea
                            x1={0.25}
                            x2={0.5}
                            fill={BAND_COLORS.high}
                            fillOpacity={0.08}
                            stroke="none"
                        />

                        <XAxis
                            dataKey="frequency"
                            {...commonXAxisProps}
                            tickFormatter={(v) => v.toFixed(2)}
                            label={{
                                value: "Frequency (cycles/day)",
                                position: "bottom",
                                offset: -5,
                                style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                            }}
                        />
                        <YAxis
                            {...commonYAxisProps}
                            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                            domain={[0, "auto"]}
                            label={{
                                value: "Power",
                                angle: -90,
                                position: "insideLeft",
                                style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                            }}
                        />
                        <Tooltip content={<CustomTooltip />} />

                        {/* Dominant frequency marker */}
                        {dominantFrequency && dominantFrequency > 0 && (
                            <ReferenceLine
                                x={dominantFrequency}
                                stroke={CHART_COLORS.red}
                                strokeDasharray="4 4"
                                strokeWidth={1.5}
                            />
                        )}

                        <Bar
                            dataKey="power"
                            fill={CHART_COLORS.blue}
                            radius={[2, 2, 0, 0]}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 text-xs">
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: BAND_COLORS.low, opacity: 0.6 }}
                    />
                    <span className="text-muted-foreground">Trends (&gt;10d)</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: BAND_COLORS.mid, opacity: 0.6 }}
                    />
                    <span className="text-muted-foreground">Cycles (4-10d)</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: BAND_COLORS.high, opacity: 0.6 }}
                    />
                    <span className="text-muted-foreground">Noise (&lt;4d)</span>
                </div>
            </div>
        </div>
    )
}

export default SpectralAnalysisChart
