import { useEffect, useRef, useState, useMemo } from 'react'

export type ColorMode = 'hsl' | 'regime'

export interface SpaghettiChartProps {
    paths: Float32Array[]
    regimes?: Uint8Array[]
    historicalPrice: number
    horizon: number
    colorMode?: ColorMode
    width?: number | string
    height?: number
    className?: string
    regimeColors?: string[]
}

export function SpaghettiChart({
    paths,
    regimes,
    historicalPrice,
    horizon,
    colorMode = 'hsl',
    width = '100%',
    height = 400,
    className = '',
    regimeColors = ['#f59e0b', '#ef4444', '#3b82f6', '#10b981'] // Default regime colors (Amber, Red, Blue, Green)
}: SpaghettiChartProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const [dimensions, setDimensions] = useState({ width: 0, height: height })

    // Handle Resize
    useEffect(() => {
        if (!containerRef.current) return
        const resizeObserver = new ResizeObserver((entries) => {
            for (let entry of entries) {
                setDimensions({
                    width: entry.contentRect.width,
                    height: entry.contentRect.height
                })
            }
        })
        resizeObserver.observe(containerRef.current)
        return () => resizeObserver.disconnect()
    }, [])

    // Calculate scales and render
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas || paths.length === 0 || dimensions.width === 0) return

        const ctx = canvas.getContext('2d', { alpha: false })
        if (!ctx) return

        const { width: w, height: h } = dimensions

        // Support high DPI displays
        const dpr = window.devicePixelRatio || 1
        canvas.width = w * dpr
        canvas.height = h * dpr
        ctx.scale(dpr, dpr)

        // 1. Calculate Y-axis bounds (using approx 1st and 99th percentiles to avoid massive outliers scaling the chart flat)
        let minVal = historicalPrice
        let maxVal = historicalPrice

        // Fast approx bounds: sample end prices
        const endPrices = new Float32Array(paths.length)
        for (let i = 0; i < paths.length; i++) {
            endPrices[i] = paths[i][horizon]
        }
        endPrices.sort()

        // Use 1st and 99th percentile of end prices as bounds, but ensure initial price is visible
        const p01 = endPrices[Math.floor(paths.length * 0.01)]
        const p99 = endPrices[Math.floor(paths.length * 0.99)]

        minVal = Math.min(historicalPrice * 0.95, p01 * 0.95)
        maxVal = Math.max(historicalPrice * 1.05, p99 * 1.05)

        const range = maxVal - minVal

        // Helper functions
        const getX = (t: number) => (t / horizon) * w
        const getY = (price: number) => h - ((price - minVal) / range) * h

        // 2. Clear background (Pure Black)
        ctx.fillStyle = '#0a0a0a'
        ctx.fillRect(0, 0, w, h)

        // 3. Draw Grid
        ctx.strokeStyle = '#262626'
        ctx.lineWidth = 1
        ctx.beginPath()
        // Horizontal lines
        for (let i = 1; i < 5; i++) {
            const y = (h / 5) * i
            ctx.moveTo(0, y)
            ctx.lineTo(w, y)
        }
        // Vertical lines
        for (let i = 1; i <= 10; i++) {
            const x = (w / 10) * i
            ctx.moveTo(x, 0)
            ctx.lineTo(x, h)
        }
        ctx.stroke()

        // 4. Draw Paths
        ctx.globalAlpha = Math.max(0.05, Math.min(0.25, 50 / paths.length)) // Dynamic alpha based on density
        ctx.lineWidth = 1
        ctx.lineJoin = 'round'

        for (let i = 0; i < paths.length; i++) {
            const path = paths[i]

            if (colorMode === 'hsl') {
                const hue = (i * 137.5) % 360
                ctx.strokeStyle = `hsl(${hue}, 80%, 60%)`
                ctx.beginPath()
                ctx.moveTo(getX(0), getY(path[0]))
                for (let t = 1; t <= horizon; t++) {
                    ctx.lineTo(getX(t), getY(path[t]))
                }
                ctx.stroke()
            } else if (colorMode === 'regime' && regimes) {
                // Draw in segments based on regime
                const statePath = regimes[i]
                for (let t = 1; t <= horizon; t++) {
                    ctx.beginPath()
                    const prevState = statePath[t - 1]
                    ctx.strokeStyle = regimeColors[prevState % regimeColors.length]
                    ctx.moveTo(getX(t - 1), getY(path[t - 1]))
                    ctx.lineTo(getX(t), getY(path[t]))
                    ctx.stroke()
                }
            }
        }

        // 5. Draw Baseline (Historical Price)
        ctx.globalAlpha = 1.0
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        const startY = getY(historicalPrice)
        ctx.moveTo(0, startY)
        ctx.lineTo(w, startY)
        ctx.stroke()
        ctx.setLineDash([])

        // 6. Draw Badge
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 12px monospace'
        ctx.textAlign = 'right'
        ctx.textBaseline = 'bottom'
        ctx.fillText(`${paths.length} SIMULATIONS | ${horizon} BAR FORECAST`, w - 10, h - 10)

        // Y Axis Labels
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillStyle = '#a3a3a3'
        ctx.font = '10px monospace'
        ctx.fillText(`$${maxVal.toFixed(2)}`, 5, 10)
        ctx.fillText(`$${minVal.toFixed(2)}`, 5, h - 10)
        ctx.fillText(`$${historicalPrice.toFixed(2)}`, 5, startY - 10)

    }, [paths, regimes, historicalPrice, horizon, colorMode, dimensions, regimeColors])

    return (
        <div ref={containerRef} className={`relative w-full overflow-hidden rounded-md border border-border ${className}`} style={{ height }}>
            <canvas
                ref={canvasRef}
                style={{ width: dimensions.width, height: dimensions.height, display: 'block' }}
            />
        </div>
    )
}
