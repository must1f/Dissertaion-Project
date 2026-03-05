import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ReferenceLine,
} from "recharts"
import {
    Waves,
    Play,
    Activity,
    TrendingUp,
    Gauge,
    Timer,
    BarChart3,
    ArrowRight,
} from "lucide-react"
import { SpectralAnalysisChart } from "../components/charts/SpectralAnalysisChart"
import { useSpectralAnalysis, useRegimeDetection, useFanChart } from "../hooks/useSpectral"
import type {
    SpectralAnalysisResponse,
    RegimeDetectionResponse,
    FanChartResponse,
} from "../services/spectralApi"

// Regime colors
const REGIME_COLORS: Record<number, string> = {
    0: "#3b82f6", // Blue  – Low Volatility
    1: "#10b981", // Green – Normal
    2: "#f59e0b", // Amber – High Volatility
    3: "#ef4444", // Red
    4: "#8b5cf6", // Purple
}

export default function SpectralAnalysis() {
    const [activeTab, setActiveTab] = useState("spectral")
    const [ticker, setTicker] = useState("^GSPC")
    const [windowSize, setWindowSize] = useState(64)
    const [lookbackDays, setLookbackDays] = useState(504)
    const [nRegimes, setNRegimes] = useState(3)
    const [method, setMethod] = useState("spectral_hmm")
    const [horizonDays, setHorizonDays] = useState(252)
    const [nSimulations, setNSimulations] = useState(1000)
    const [useRegimeSwitching, setUseRegimeSwitching] = useState(true)

    // Results
    const [spectralResult, setSpectralResult] = useState<SpectralAnalysisResponse | null>(null)
    const [regimeResult, setRegimeResult] = useState<RegimeDetectionResponse | null>(null)
    const [fanChartResult, setFanChartResult] = useState<FanChartResponse | null>(null)

    // Mutations
    const spectralMutation = useSpectralAnalysis()
    const regimeMutation = useRegimeDetection()
    const fanChartMutation = useFanChart()

    const handleAnalyze = () => {
        spectralMutation.mutate(
            { ticker, window_size: windowSize },
            { onSuccess: (data) => setSpectralResult(data) }
        )
    }

    const handleDetectRegimes = () => {
        regimeMutation.mutate(
            { ticker, method, n_regimes: nRegimes, lookback_days: lookbackDays },
            { onSuccess: (data) => setRegimeResult(data) }
        )
    }

    const handleGenerateFanChart = () => {
        fanChartMutation.mutate(
            {
                ticker,
                horizon_days: horizonDays,
                n_simulations: nSimulations,
                use_regime_switching: useRegimeSwitching,
                percentiles: [5, 25, 50, 75, 95],
            },
            { onSuccess: (data) => setFanChartResult(data) }
        )
    }

    // Build fan chart data for Recharts
    const fanChartData = fanChartResult
        ? fanChartResult.dates.map((day, i) => {
            const point: Record<string, number> = { day }
            fanChartResult.percentile_bands.forEach((band) => {
                point[`p${band.percentile}`] = band.values[i]
            })
            point.median = fanChartResult.median_path[i]
            return point
        })
        : []

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold flex items-center gap-2">
                    <Waves className="h-8 w-8 text-primary" />
                    Spectral Analysis
                </h1>
                <p className="text-muted-foreground">
                    Frequency-domain analysis, regime detection, and regime-aware Monte Carlo simulations
                </p>
            </div>

            {/* Ticker Input (shared across all tabs) */}
            <Card>
                <CardContent className="py-4">
                    <div className="flex items-center gap-4">
                        <div className="space-y-1">
                            <Label htmlFor="spectral-ticker">Ticker</Label>
                            <Input
                                id="spectral-ticker"
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value)}
                                placeholder="^GSPC"
                                className="w-32"
                            />
                        </div>
                        <div className="text-xs text-muted-foreground pt-5">
                            Used across all tabs. S&P 500 data is pre-loaded.
                        </div>
                    </div>
                </CardContent>
            </Card>

            <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="spectral">Power Spectrum</TabsTrigger>
                    <TabsTrigger value="regimes">Regime Detection</TabsTrigger>
                    <TabsTrigger value="fanchart">Fan Chart</TabsTrigger>
                </TabsList>

                {/* ═══════════════════════════════════════════════════════════════════
            TAB 1: Spectral Analysis
        ═══════════════════════════════════════════════════════════════════ */}
                <TabsContent value="spectral" className="space-y-6">
                    {/* Controls */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                                <BarChart3 className="h-4 w-4" />
                                Analysis Parameters
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex items-end gap-4">
                                <div className="space-y-1">
                                    <Label htmlFor="window-size">Window Size</Label>
                                    <Input
                                        id="window-size"
                                        type="number"
                                        value={windowSize}
                                        onChange={(e) => setWindowSize(parseInt(e.target.value) || 64)}
                                        min={16}
                                        max={256}
                                        className="w-28"
                                    />
                                </div>
                                <Button
                                    onClick={handleAnalyze}
                                    disabled={spectralMutation.isPending}
                                >
                                    {spectralMutation.isPending ? (
                                        <LoadingSpinner size="sm" className="mr-2" />
                                    ) : (
                                        <Play className="mr-2 h-4 w-4" />
                                    )}
                                    Analyze
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    {spectralMutation.isError && (
                        <Card className="border-red-500/50 bg-red-500/10">
                            <CardContent className="py-4 text-sm text-red-500">
                                Error: {(spectralMutation.error as Error)?.message || "Analysis failed"}
                            </CardContent>
                        </Card>
                    )}

                    {/* Results */}
                    {spectralResult && (
                        <>
                            {/* Feature Cards */}
                            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                                            <Gauge className="h-3.5 w-3.5" />
                                            Spectral Entropy
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-3xl font-bold">
                                            {(spectralResult.current_features.spectral_entropy * 100).toFixed(1)}%
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            {spectralResult.current_features.spectral_entropy > 0.85
                                                ? "High randomness — market is noise-dominated"
                                                : spectralResult.current_features.spectral_entropy > 0.6
                                                    ? "Moderate structure — some patterns present"
                                                    : "Strong structure — dominant cycles detected"}
                                        </p>
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                                            <Timer className="h-3.5 w-3.5" />
                                            Dominant Period
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-3xl font-bold">
                                            {spectralResult.current_features.dominant_period.toFixed(1)}
                                            <span className="text-base font-normal text-muted-foreground ml-1">days</span>
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Strongest cycle at {spectralResult.current_features.dominant_frequency.toFixed(3)} cpd
                                        </p>
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                                            <Activity className="h-3.5 w-3.5" />
                                            Power Ratio
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-3xl font-bold">
                                            {spectralResult.current_features.power_ratio.toFixed(2)}
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Signal-to-noise ratio
                                        </p>
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                                            <TrendingUp className="h-3.5 w-3.5" />
                                            Spectral Slope
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-3xl font-bold">
                                            {spectralResult.current_features.spectral_slope.toFixed(3)}
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            {spectralResult.current_features.spectral_slope < -1
                                                ? "Strong mean-reversion"
                                                : spectralResult.current_features.spectral_slope < -0.5
                                                    ? "Mild mean-reversion"
                                                    : "Close to random walk"}
                                        </p>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Power Spectrum Chart */}
                            <Card>
                                <CardContent className="pt-6">
                                    <SpectralAnalysisChart
                                        data={spectralResult.power_spectrum}
                                        dominantFrequency={spectralResult.current_features.dominant_frequency}
                                        spectralEntropy={spectralResult.current_features.spectral_entropy}
                                    />
                                </CardContent>
                            </Card>

                            {/* Power Band Breakdown */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-base">Frequency Band Breakdown</CardTitle>
                                    <CardDescription>Distribution of power across frequency bands</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid gap-4 md:grid-cols-3">
                                        {(["low", "mid", "high"] as const).map((band) => {
                                            const bandData = spectralResult.power_spectrum.frequency_bands[band]
                                            const totalPower = Object.values(spectralResult.power_spectrum.frequency_bands).reduce(
                                                (sum, b) => sum + b.total_power,
                                                0
                                            )
                                            const pct = totalPower > 0 ? (bandData.total_power / totalPower) * 100 : 0

                                            const bandInfo = {
                                                low: { label: "Trends", color: "#3b82f6", desc: "> 10 day cycles" },
                                                mid: { label: "Cycles", color: "#10b981", desc: "4–10 day cycles" },
                                                high: { label: "Noise", color: "#f59e0b", desc: "< 4 day cycles" },
                                            }[band]

                                            return (
                                                <div key={band} className="space-y-2">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-sm font-medium" style={{ color: bandInfo.color }}>
                                                            {bandInfo.label}
                                                        </span>
                                                        <span className="text-sm font-mono">{pct.toFixed(1)}%</span>
                                                    </div>
                                                    <div className="h-2 rounded-full bg-muted overflow-hidden">
                                                        <div
                                                            className="h-full rounded-full transition-all duration-500"
                                                            style={{ width: `${pct}%`, backgroundColor: bandInfo.color }}
                                                        />
                                                    </div>
                                                    <p className="text-xs text-muted-foreground">{bandInfo.desc}</p>
                                                </div>
                                            )
                                        })}
                                    </div>
                                </CardContent>
                            </Card>

                            <div className="text-xs text-muted-foreground text-right">
                                Processed in {spectralResult.processing_time_ms.toFixed(0)}ms
                            </div>
                        </>
                    )}
                </TabsContent>

                {/* ═══════════════════════════════════════════════════════════════════
            TAB 2: Regime Detection
        ═══════════════════════════════════════════════════════════════════ */}
                <TabsContent value="regimes" className="space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base">Detection Parameters</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex flex-wrap items-end gap-4">
                                <div className="space-y-1">
                                    <Label>Method</Label>
                                    <div className="flex gap-2">
                                        {["spectral_hmm", "hmm", "kmeans"].map((m) => (
                                            <Button
                                                key={m}
                                                variant={method === m ? "default" : "outline"}
                                                size="sm"
                                                onClick={() => setMethod(m)}
                                                className="text-xs"
                                            >
                                                {m === "spectral_hmm" ? "Spectral HMM" : m === "hmm" ? "HMM" : "K-Means"}
                                            </Button>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-1">
                                    <Label htmlFor="n-regimes">Regimes</Label>
                                    <Input
                                        id="n-regimes"
                                        type="number"
                                        value={nRegimes}
                                        onChange={(e) => setNRegimes(parseInt(e.target.value) || 3)}
                                        min={2}
                                        max={5}
                                        className="w-20"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <Label htmlFor="lookback">Lookback (days)</Label>
                                    <Input
                                        id="lookback"
                                        type="number"
                                        value={lookbackDays}
                                        onChange={(e) => setLookbackDays(parseInt(e.target.value) || 504)}
                                        min={100}
                                        max={2520}
                                        className="w-28"
                                    />
                                </div>
                                <Button
                                    onClick={handleDetectRegimes}
                                    disabled={regimeMutation.isPending}
                                >
                                    {regimeMutation.isPending ? (
                                        <LoadingSpinner size="sm" className="mr-2" />
                                    ) : (
                                        <Play className="mr-2 h-4 w-4" />
                                    )}
                                    Detect Regimes
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    {regimeMutation.isError && (
                        <Card className="border-red-500/50 bg-red-500/10">
                            <CardContent className="py-4 text-sm text-red-500">
                                Error: {(regimeMutation.error as Error)?.message || "Detection failed"}
                            </CardContent>
                        </Card>
                    )}

                    {regimeResult && (
                        <>
                            {/* Current Regime Banner */}
                            <Card
                                className="border-l-4"
                                style={{ borderLeftColor: REGIME_COLORS[regimeResult.current_regime] || "#6b7280" }}
                            >
                                <CardContent className="py-4">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="text-sm text-muted-foreground">Current Market Regime</p>
                                            <div className="flex items-center gap-3 mt-1">
                                                <h2 className="text-2xl font-bold">{regimeResult.current_regime_name}</h2>
                                                <Badge
                                                    style={{
                                                        backgroundColor: REGIME_COLORS[regimeResult.current_regime] + "20",
                                                        color: REGIME_COLORS[regimeResult.current_regime],
                                                        borderColor: REGIME_COLORS[regimeResult.current_regime],
                                                    }}
                                                    variant="outline"
                                                >
                                                    {(regimeResult.current_probability * 100).toFixed(0)}% confidence
                                                </Badge>
                                            </div>
                                        </div>
                                        <div className="text-right text-sm text-muted-foreground">
                                            <p>Method: {regimeResult.method.replace(/_/g, " ")}</p>
                                            <p>{regimeResult.processing_time_ms.toFixed(0)}ms</p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Regime Characteristics */}
                            <div className="grid gap-4 md:grid-cols-3">
                                {regimeResult.regime_characteristics.map((regime) => (
                                    <Card
                                        key={regime.regime_id}
                                        className={`border-t-4 ${regimeResult.current_regime === regime.regime_id ? "ring-1 ring-primary/30" : ""
                                            }`}
                                        style={{ borderTopColor: REGIME_COLORS[regime.regime_id] || "#6b7280" }}
                                    >
                                        <CardHeader className="pb-2">
                                            <CardTitle className="text-sm flex items-center justify-between">
                                                <span>{regime.regime_name}</span>
                                                {regimeResult.current_regime === regime.regime_id && (
                                                    <Badge variant="default" className="text-[10px]">Current</Badge>
                                                )}
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent className="space-y-2 text-sm">
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Annual Return</span>
                                                <span
                                                    className={`font-mono ${regime.mean_return_annual >= 0 ? "text-emerald-500" : "text-red-500"
                                                        }`}
                                                >
                                                    {(regime.mean_return_annual * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Annual Volatility</span>
                                                <span className="font-mono">{(regime.volatility_annual * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Expected Duration</span>
                                                <span className="font-mono">{regime.expected_duration_days.toFixed(1)} days</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-muted-foreground">Sample Count</span>
                                                <span className="font-mono">{regime.sample_count}</span>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>

                            {/* Transition Matrix */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-base">Transition Probability Matrix</CardTitle>
                                    <CardDescription>
                                        Probability of transitioning from one regime to another
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr>
                                                    <th className="text-left p-2 text-muted-foreground">From ↓ / To →</th>
                                                    {regimeResult.transition_matrix.labels.map((label) => (
                                                        <th key={label} className="p-2 text-right font-medium">{label}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {regimeResult.transition_matrix.matrix.map((row, i) => (
                                                    <tr key={i} className="border-t border-border/50">
                                                        <td className="p-2 font-medium">{regimeResult.transition_matrix.labels[i]}</td>
                                                        {row.map((prob, j) => (
                                                            <td
                                                                key={j}
                                                                className="p-2 text-right font-mono"
                                                                style={{
                                                                    backgroundColor:
                                                                        i === j
                                                                            ? `${REGIME_COLORS[i] || "#6b7280"}15`
                                                                            : prob > 0.1
                                                                                ? `${REGIME_COLORS[j] || "#6b7280"}10`
                                                                                : "transparent",
                                                                }}
                                                            >
                                                                {(prob * 100).toFixed(1)}%
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Recent Regime History */}
                            {regimeResult.recent_history.length > 0 && (
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-base">Recent Regime History</CardTitle>
                                        <CardDescription>Last {regimeResult.recent_history.length} trading days</CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="flex gap-1 items-center flex-wrap">
                                            {regimeResult.recent_history.map((point, i) => (
                                                <div
                                                    key={i}
                                                    className="group relative"
                                                    title={`${point.date}: ${point.regime_name} (${(point.probability * 100).toFixed(0)}%)`}
                                                >
                                                    <div
                                                        className="w-6 h-8 rounded-sm transition-transform hover:scale-y-110"
                                                        style={{
                                                            backgroundColor: REGIME_COLORS[point.regime] || "#6b7280",
                                                            opacity: 0.4 + point.probability * 0.6,
                                                        }}
                                                    />
                                                </div>
                                            ))}
                                            <ArrowRight className="h-4 w-4 text-muted-foreground ml-1" />
                                            <span className="text-xs text-muted-foreground ml-1">now</span>
                                        </div>
                                        <div className="flex gap-4 mt-3 text-xs text-muted-foreground">
                                            {regimeResult.regime_characteristics.map((rc) => (
                                                <div key={rc.regime_id} className="flex items-center gap-1.5">
                                                    <div
                                                        className="w-3 h-3 rounded-sm"
                                                        style={{ backgroundColor: REGIME_COLORS[rc.regime_id] || "#6b7280" }}
                                                    />
                                                    <span>{rc.regime_name}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </CardContent>
                                </Card>
                            )}
                        </>
                    )}
                </TabsContent>

                {/* ═══════════════════════════════════════════════════════════════════
            TAB 3: Fan Chart
        ═══════════════════════════════════════════════════════════════════ */}
                <TabsContent value="fanchart" className="space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base">Monte Carlo Parameters</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex flex-wrap items-end gap-4">
                                <div className="space-y-1">
                                    <Label htmlFor="horizon">Horizon (days)</Label>
                                    <Input
                                        id="horizon"
                                        type="number"
                                        value={horizonDays}
                                        onChange={(e) => setHorizonDays(parseInt(e.target.value) || 252)}
                                        min={5}
                                        max={504}
                                        className="w-28"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <Label htmlFor="n-sims">Simulations</Label>
                                    <Input
                                        id="n-sims"
                                        type="number"
                                        value={nSimulations}
                                        onChange={(e) => setNSimulations(parseInt(e.target.value) || 1000)}
                                        min={100}
                                        max={10000}
                                        className="w-28"
                                    />
                                </div>
                                <div className="flex items-center gap-2 pb-0.5">
                                    <input
                                        type="checkbox"
                                        id="regime-switch"
                                        checked={useRegimeSwitching}
                                        onChange={(e) => setUseRegimeSwitching(e.target.checked)}
                                        className="h-4 w-4 rounded border-gray-300"
                                    />
                                    <Label htmlFor="regime-switch">Regime Switching</Label>
                                </div>
                                <Button
                                    onClick={handleGenerateFanChart}
                                    disabled={fanChartMutation.isPending}
                                >
                                    {fanChartMutation.isPending ? (
                                        <LoadingSpinner size="sm" className="mr-2" />
                                    ) : (
                                        <Play className="mr-2 h-4 w-4" />
                                    )}
                                    Generate
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    {fanChartMutation.isError && (
                        <Card className="border-red-500/50 bg-red-500/10">
                            <CardContent className="py-4 text-sm text-red-500">
                                Error: {(fanChartMutation.error as Error)?.message || "Fan chart generation failed"}
                            </CardContent>
                        </Card>
                    )}

                    {fanChartResult && (
                        <>
                            {/* Summary Stats */}
                            <div className="grid gap-4 md:grid-cols-4">
                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground">
                                            Initial Price
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-2xl font-bold">
                                            ${fanChartResult.initial_price.toFixed(2)}
                                        </div>
                                    </CardContent>
                                </Card>
                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground">
                                            Expected Return
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div
                                            className={`text-2xl font-bold ${fanChartResult.expected_return >= 0 ? "text-emerald-500" : "text-red-500"
                                                }`}
                                        >
                                            {(fanChartResult.expected_return * 100).toFixed(1)}%
                                        </div>
                                    </CardContent>
                                </Card>
                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground">
                                            Value at Risk (5%)
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-2xl font-bold text-red-500">
                                            ${fanChartResult.value_at_risk_95.toFixed(2)}
                                        </div>
                                    </CardContent>
                                </Card>
                                <Card>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm font-medium text-muted-foreground">
                                            P(Loss)
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-2xl font-bold text-amber-500">
                                            {(fanChartResult.probability_of_loss * 100).toFixed(0)}%
                                        </div>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Fan Chart */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-base">
                                        Price Forecast — {fanChartResult.n_simulations.toLocaleString()} Simulations
                                    </CardTitle>
                                    <CardDescription>
                                        {fanChartResult.horizon_days}-day forecast with confidence bands
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="h-96">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={fanChartData}>
                                                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                                <XAxis
                                                    dataKey="day"
                                                    tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                                                    label={{
                                                        value: "Trading Days",
                                                        position: "bottom",
                                                        offset: -5,
                                                        style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                                                    }}
                                                />
                                                <YAxis
                                                    tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                                                    tickFormatter={(v) => `$${v.toFixed(0)}`}
                                                    domain={["auto", "auto"]}
                                                    label={{
                                                        value: "Price",
                                                        angle: -90,
                                                        position: "insideLeft",
                                                        style: { fontSize: 11, fill: "hsl(var(--muted-foreground))" },
                                                    }}
                                                />
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: "hsl(var(--card))",
                                                        borderColor: "hsl(var(--border))",
                                                        borderRadius: "8px",
                                                        fontSize: 12,
                                                    }}
                                                    formatter={((value: number, name: string) => [
                                                        `$${value.toFixed(2)}`,
                                                        name === "median"
                                                            ? "Median"
                                                            : name.replace("p", "") + "th percentile",
                                                    ]) as any}
                                                />

                                                {/* 5-95 band */}
                                                <Area
                                                    type="monotone"
                                                    dataKey="p95"
                                                    stroke="none"
                                                    fill="#6366f1"
                                                    fillOpacity={0.08}
                                                    name="p95"
                                                />
                                                <Area
                                                    type="monotone"
                                                    dataKey="p5"
                                                    stroke="none"
                                                    fill="hsl(var(--card))"
                                                    fillOpacity={1}
                                                    name="p5"
                                                />

                                                {/* 25-75 band */}
                                                <Area
                                                    type="monotone"
                                                    dataKey="p75"
                                                    stroke="none"
                                                    fill="#6366f1"
                                                    fillOpacity={0.15}
                                                    name="p75"
                                                />
                                                <Area
                                                    type="monotone"
                                                    dataKey="p25"
                                                    stroke="none"
                                                    fill="hsl(var(--card))"
                                                    fillOpacity={1}
                                                    name="p25"
                                                />

                                                {/* Median line */}
                                                <Area
                                                    type="monotone"
                                                    dataKey="median"
                                                    stroke="#6366f1"
                                                    strokeWidth={2}
                                                    fill="none"
                                                    name="median"
                                                />

                                                {/* Initial price reference */}
                                                <ReferenceLine
                                                    y={fanChartResult.initial_price}
                                                    stroke="hsl(var(--muted-foreground))"
                                                    strokeDasharray="4 4"
                                                    strokeWidth={1}
                                                />

                                                <Legend
                                                    formatter={(value) =>
                                                        value === "median"
                                                            ? "Median"
                                                            : value === "p95"
                                                                ? "5–95%"
                                                                : value === "p75"
                                                                    ? "25–75%"
                                                                    : ""
                                                    }
                                                />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </CardContent>
                            </Card>

                            <div className="text-xs text-muted-foreground text-right">
                                Processed in {fanChartResult.processing_time_ms.toFixed(0)}ms
                            </div>
                        </>
                    )}
                </TabsContent>
            </Tabs>
        </div>
    )
}
