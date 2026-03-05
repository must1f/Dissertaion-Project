/**
 * Shared chart theme for research-grade, publication-quality visualizations.
 *
 * Design principles:
 *  - No visual distortion (no glow filters, no blurry halos)
 *  - Colorblind-safe palette (Okabe-Ito inspired)
 *  - Clean precise lines (1.5–2px)
 *  - Flat CI bands (constant opacity, not gradient)
 *  - Proper axis formatting with units
 *  - Minimal chart junk
 */
import React from "react"

/* ─── Colorblind-safe palette (Okabe-Ito + scientific defaults) ─── */
export const CHART_COLORS = {
    /* UI tokens */
    foreground: "hsl(var(--foreground))",
    muted: "hsl(var(--muted-foreground))",
    border: "hsl(var(--border))",
    card: "hsl(var(--card))",

    /* Data series — Okabe-Ito palette (colorblind-safe) */
    blue: "#0072B2",        // strong blue
    orange: "#E69F00",      // amber/orange
    green: "#009E73",       // teal-green
    red: "#D55E00",         // vermillion
    purple: "#CC79A7",      // reddish-purple
    cyan: "#56B4E9",        // sky blue (light)
    yellow: "#F0E442",      // yellow

    /* Semantic aliases */
    actual: "#0072B2",      // observed data → blue
    predicted: "#E69F00",   // model output → orange
    benchmark: "#999999",   // benchmark → grey
    profit: "#009E73",      // positive return → green
    loss: "#D55E00",        // negative/drawdown → vermillion
    confidence: "#56B4E9",  // CI bands → sky blue

    /* Grid & reference */
    grid: "hsl(var(--border) / 0.35)",
}

/* ─── Common axis props ──────────────────────────────────── */
export const commonAxisProps = {
    tick: {
        fontSize: 11,
        fill: "hsl(var(--muted-foreground))",
        fontFamily: "'Inter', system-ui, sans-serif",
    },
    tickLine: false,
    axisLine: { stroke: "hsl(var(--border))", strokeWidth: 1 },
}

const axisTick = {
    fontSize: 11,
    fill: "hsl(var(--muted-foreground))",
    fontFamily: "'Inter', system-ui, sans-serif",
}

export const commonXAxisProps = {
    tick: axisTick,
    tickLine: false,
    axisLine: { stroke: "hsl(var(--border))", strokeWidth: 1 },
    tickMargin: 8,
    minTickGap: 40,
} as const

export const commonYAxisProps = {
    tick: axisTick,
    tickLine: false,
    axisLine: { stroke: "hsl(var(--border))", strokeWidth: 1 },
    tickMargin: 6,
    width: 58,
} as const

/* ─── Grid ───────────────────────────────────────────────── */
export const gridConfig = {
    strokeDasharray: "3 3",
    stroke: CHART_COLORS.grid,
    vertical: false, // horizontal only for time-series
}

/* Scatter charts need both axes */
export const scatterGridConfig = {
    ...gridConfig,
    vertical: true,
}

/* ─── Tooltip style (clean dark panel, no glass blur) ────── */
export const tooltipContentStyle: React.CSSProperties = {
    backgroundColor: "hsl(220 20% 10% / 0.95)",
    border: "1px solid hsl(220 15% 25%)",
    borderRadius: "6px",
    padding: "10px 14px",
    color: "hsl(210 40% 96%)",
    fontFamily: "'Inter', system-ui, sans-serif",
    fontSize: "12px",
    lineHeight: "1.6",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
}

export const tooltipLabelStyle: React.CSSProperties = {
    color: "hsl(215 20% 65%)",
    fontWeight: 600,
    fontSize: "11px",
    marginBottom: "3px",
}

/* ─── Cursor ─────────────────────────────────────────────── */
export const cursorConfig = {
    stroke: "hsl(var(--muted-foreground) / 0.3)",
    strokeWidth: 1,
}

/* ─── Legend ──────────────────────────────────────────────── */
export const legendConfig = {
    iconType: "line" as const,
    iconSize: 14,
    wrapperStyle: {
        fontSize: "11px",
        fontFamily: "'Inter', system-ui, sans-serif",
        paddingTop: "10px",
    },
}

/* ─── Clean active dot (small, precise, no effects) ──────── */
export function CleanDot(props: any) {
    const { cx, cy, fill = CHART_COLORS.blue } = props
    if (!cx || !cy) return null
    return (
        <circle
            cx={cx}
            cy={cy}
            r={3.5}
            fill={fill}
            stroke="#fff"
            strokeWidth={1.5}
        />
    )
}

/* ─── Custom Tooltip with labeled rows ───────────────────── */
interface ChartTooltipRow {
    label: string
    value: string
    color?: string
}

interface ChartTooltipProps {
    active?: boolean
    title?: string
    rows?: ChartTooltipRow[]
    children?: React.ReactNode
}

export function ChartTooltip({ active, title, rows, children }: ChartTooltipProps) {
    if (!active) return null

    return (
        <div style={tooltipContentStyle}>
            {title && (
                <div style={tooltipLabelStyle}>{title}</div>
            )}
            {rows?.map((row, i) => (
                <div key={i} className="flex items-center justify-between gap-5" style={{ fontSize: "12px", lineHeight: "1.7" }}>
                    <span style={{ color: "hsl(215 20% 65%)" }} className="flex items-center gap-1.5">
                        {row.color && (
                            <span
                                style={{
                                    display: "inline-block",
                                    width: 8,
                                    height: 3,
                                    borderRadius: 1,
                                    backgroundColor: row.color,
                                }}
                            />
                        )}
                        {row.label}
                    </span>
                    <span style={{ fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                        {row.value}
                    </span>
                </div>
            ))}
            {children}
        </div>
    )
}

/* ─── SVG Defs for flat CI bands (no glow, no gradient) ─── */
export function CIBandDefs() {
    return (
        <defs>
            {/* Confidence interval fill — flat sky-blue */}
            <linearGradient id="ciBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS.confidence} stopOpacity={0.15} />
                <stop offset="100%" stopColor={CHART_COLORS.confidence} stopOpacity={0.15} />
            </linearGradient>
        </defs>
    )
}
