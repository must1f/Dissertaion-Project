import { useState, useMemo, useEffect } from "react"
import { useAppStore } from "../stores/appStore"
import { useStockData } from "../hooks/useData"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { MetricCard, MetricGrid } from "../components/common/MetricCard"
import { DistributionChart } from "../components/charts/DistributionChart"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { SpaghettiChart, ColorMode } from "../components/charts/SpaghettiChart"
import { Dices, AlertCircle } from "lucide-react"
import { useSimulationWorker } from "../hooks/useSimulationWorker"
import { ModelType } from "../workers/simulationWorker"
import { RegimeConfig, estimateRegimesAndTransitionMatrix } from "../lib/simulationMath"

export default function MonteCarlo() {
  const { selectedTicker } = useAppStore()
  const { data: stockData } = useStockData(selectedTicker)
  const { runSimulation, isSimulating, results, error, processingTimeMs } = useSimulationWorker()

  const [formError, setFormError] = useState<string | null>(null)

  // Base Config
  const [modelType, setModelType] = useState<ModelType>('gbm')
  const [nSims, setNSims] = useState(1000)
  const [horizon, setHorizon] = useState(30)
  const [initialPrice, setInitialPrice] = useState(0)
  const [seed, setSeed] = useState(Date.now())
  const [colorMode, setColorMode] = useState<ColorMode>('hsl')

  // GBM Config
  const [gbmMu, setGbmMu] = useState(0.08)
  const [gbmSigma, setGbmSigma] = useState(0.20)

  // Markov Config
  const [kStates, setKStates] = useState(2)
  const [transitionMatrix, setTransitionMatrix] = useState<number[][]>([[0.9, 0.1], [0.2, 0.8]])
  const [regimesConfig, setRegimesConfig] = useState<RegimeConfig[]>([
    { mu: 0.1, sigma: 0.15 },  // Normal/Low Vol
    { mu: -0.2, sigma: 0.40 }  // Stress/High Vol
  ])
  const [initialProbabilities, setInitialProbabilities] = useState<number[]>([1.0, 0.0])

  // Merton Config
  const [mertonLambda, setMertonLambda] = useState(5.0)
  const [mertonMuJ, setMertonMuJ] = useState(0.0)
  const [mertonSigmaJ, setMertonSigmaJ] = useState(0.15)

  // Heston Config
  const [hestonInitialVar, setHestonInitialVar] = useState(0.04)
  const [hestonKappa, setHestonKappa] = useState(2.0)
  const [hestonTheta, setHestonTheta] = useState(0.04)
  const [hestonXi, setHestonXi] = useState(0.2)
  const [hestonRho, setHestonRho] = useState(-0.7)

  // Extract historical stats when data loads
  const historicalStats = useMemo(() => {
    if (!stockData?.data || stockData.data.length < 2) return null

    const prices = stockData.data.map(d => d.close)
    const returns: number[] = []
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1])
    }

    const meanDisplay = returns.reduce((a, b) => a + b, 0) / returns.length
    const variance = returns.reduce((a, b) => a + Math.pow(b - meanDisplay, 2), 0) / (returns.length - 1)

    return {
      lastPrice: prices[prices.length - 1],
      returns,
      annualizedMu: meanDisplay * 252,
      annualizedSigma: Math.sqrt(variance * 252)
    }
  }, [stockData])

  // Update initial price when stock data loads
  useEffect(() => {
    if (historicalStats) {
      setInitialPrice(historicalStats.lastPrice)
      setGbmMu(Number(historicalStats.annualizedMu.toFixed(4)))
      setGbmSigma(Number(historicalStats.annualizedSigma.toFixed(4)))
    }
  }, [historicalStats])

  const handleRunSimulation = () => {
    setFormError(null)

    if (modelType === 'markov') {
      // Validate Transition Matrix
      for (let i = 0; i < kStates; i++) {
        const sum = transitionMatrix[i].reduce((a, b) => a + b, 0)
        if (Math.abs(sum - 1.0) > 0.01) {
          setFormError(`Transition matrix row ${i + 1} does not sum to 1.0 (Sum: ${sum.toFixed(2)})`)
          return
        }
      }

      const probSum = initialProbabilities.reduce((a, b) => a + b, 0)
      if (Math.abs(probSum - 1.0) > 0.01) {
        setFormError(`Initial probabilities do not sum to 1.0 (Sum: ${probSum.toFixed(2)})`)
        return
      }
    }

    runSimulation({
      modelType,
      nSims,
      horizon,
      initialPrice,
      dt: 1 / 252, // Daily steps assumed
      seed,
      mu: gbmMu,
      sigma: gbmSigma,
      historicalReturns: historicalStats?.returns,
      regimesConfig,
      transitionMatrix,
      initialProbabilities,
      lambda: mertonLambda,
      muJ: mertonMuJ,
      sigmaJ: mertonSigmaJ,
      initialVar: hestonInitialVar,
      kappa: hestonKappa,
      theta: hestonTheta,
      xi: hestonXi,
      rho: hestonRho
    })
  }

  const handleAutoCalibrateMarkov = () => {
    if (!historicalStats?.returns) {
      setFormError("Not enough historical data to calibrate Markov regimes.")
      return
    }
    try {
      const calib = estimateRegimesAndTransitionMatrix(historicalStats.returns, kStates, 20, 252)
      setRegimesConfig(calib.regimes)
      setTransitionMatrix(calib.transitionMatrix)
      setInitialProbabilities(calib.initialProbabilities)
      setFormError(null)
    } catch (e: any) {
      setFormError(`Calibration failed: ${e.message}`)
    }
  }

  // Adjust K states matrix
  const handleKChange = (newK: number) => {
    if (newK < 1 || newK > 5) return
    const newMatrix = Array.from({ length: newK }, (_, i) => {
      const row = new Array(newK).fill(0)
      for (let j = 0; j < newK; j++) {
        if (i < transitionMatrix.length && j < transitionMatrix[i].length) {
          row[j] = transitionMatrix[i][j]
        } else {
          row[j] = (i === j) ? 1.0 : 0.0
        }
      }
      // Normalize row
      const sum = row.reduce((a, b) => a + b, 0)
      return row.map(v => v / sum)
    })

    const newRegimes = Array.from({ length: newK }, (_, i) => {
      return i < regimesConfig.length ? regimesConfig[i] : { mu: 0, sigma: 0.2 }
    })

    const newProbs = new Array(newK).fill(0)
    newProbs[0] = 1.0 // default start in state 0

    setKStates(newK)
    setTransitionMatrix(newMatrix)
    setRegimesConfig(newRegimes)
    setInitialProbabilities(newProbs)
  }

  // Calculate stats from results
  const stats = useMemo(() => {
    if (!results || results.paths.length === 0) return null

    const endPrices = new Float32Array(results.paths.length)
    for (let i = 0; i < results.paths.length; i++) {
      endPrices[i] = results.paths[i][horizon]
    }

    endPrices.sort() // Sort for percentiles and median

    const sum = endPrices.reduce((a, b) => a + b, 0)
    const mean = sum / endPrices.length
    const median = endPrices[Math.floor(endPrices.length / 2)]

    let variance = 0
    let wins = 0
    let losses = 0
    for (let i = 0; i < endPrices.length; i++) {
      variance += Math.pow(endPrices[i] - mean, 2)
      if (endPrices[i] > initialPrice) wins++
      else losses++
    }
    const std = Math.sqrt(variance / endPrices.length)

    const var95Idx = Math.floor(endPrices.length * 0.05)

    // Histogram Bins
    const minP = endPrices[0]
    const maxP = endPrices[endPrices.length - 1]
    const numBins = 50
    const binWidth = (maxP - minP) / numBins
    const histData = Array.from({ length: numBins }, (_, i) => ({
      bin: minP + (i + 0.5) * binWidth,
      count: 0
    }))
    for (let i = 0; i < endPrices.length; i++) {
      let binIdx = Math.floor((endPrices[i] - minP) / binWidth)
      if (binIdx >= numBins) binIdx = numBins - 1
      histData[binIdx].count++
    }

    // Regime stats
    let regimePercentages = null
    if (results.regimes) {
      const counts = new Array(kStates).fill(0)
      let total = 0
      for (let i = 0; i < results.regimes.length; i++) {
        for (let t = 0; t <= horizon; t++) {
          counts[results.regimes[i][t]]++
          total++
        }
      }
      regimePercentages = counts.map(c => c / total)
    }

    return {
      mean,
      median,
      std,
      pGain: wins / endPrices.length,
      pLoss: losses / endPrices.length,
      var95: endPrices[var95Idx],
      histData,
      regimePercentages
    }
  }, [results, horizon, initialPrice, kStates])

  const displayError = formError || error

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-mono text-amber-500 uppercase tracking-tight">Monte Carlo Engine</h1>
          <p className="text-muted-foreground font-mono text-sm uppercase">
            High-Performance Canvas Path Rendering & Markov Regime Switching
          </p>
        </div>
        <Badge variant="outline" className="text-lg font-mono bg-black text-amber-500 border-amber-500/50">
          {selectedTicker}
        </Badge>
      </div>

      {/* Error Display */}
      {displayError && (
        <Card className="border-[#ff0000] bg-black">
          <CardContent className="flex items-center gap-2 pt-6">
            <AlertCircle className="h-5 w-5 text-[#ff0000]" />
            <span className="text-[#ff0000] font-mono text-sm">{displayError}</span>
            <Button variant="ghost" size="sm" onClick={() => setFormError(null)} className="ml-auto text-muted-foreground">Dismiss</Button>
          </CardContent>
        </Card>
      )}

      {/* Configuration */}
      <Card className="bg-[#0a0a0a] border-[#262626]">
        <CardHeader>
          <CardTitle className="font-mono text-amber-500 uppercase">Simulation Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-5 mb-6">
            <div>
              <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Model</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value as ModelType)}
                className="w-full rounded-md border border-[#262626] bg-black text-white font-mono px-3 py-2 text-sm focus:border-amber-500"
                disabled={isSimulating}
              >
                <option value="gbm">Geometric Brownian Motion</option>
                <option value="merton">Merton Jump-Diffusion</option>
                <option value="heston">Heston Stochastic Volatility</option>
                <option value="bootstrap">Historical Bootstrap</option>
                <option value="markov">Markov Regime Switching</option>
              </select>
            </div>
            <div>
              <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Simulations (Paths)</label>
              <Input
                type="number" value={nSims} onChange={(e) => setNSims(Number(e.target.value))}
                min={10} max={100000} disabled={isSimulating}
                className="bg-black border-[#262626] text-white font-mono text-sm"
              />
            </div>
            <div>
              <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Horizon (Bars/Days)</label>
              <Input
                type="number" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}
                min={1} max={1000} disabled={isSimulating}
                className="bg-black border-[#262626] text-white font-mono text-sm"
              />
            </div>
            <div>
              <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Initial Price</label>
              <Input
                type="number" value={initialPrice} onChange={(e) => setInitialPrice(Number(e.target.value))}
                step={0.01} disabled={isSimulating}
                className="bg-black border-[#262626] text-white font-mono text-sm"
              />
            </div>
            <div>
              <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Random Seed</label>
              <Input
                type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                disabled={isSimulating}
                className="bg-black border-[#262626] text-white font-mono text-sm"
              />
            </div>
          </div>

          {/* Model Specific Configs */}
          <div className="border border-[#262626] rounded-md p-4 bg-[#050505]">
            {modelType === 'gbm' && (
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Annualized Drift (μ)</label>
                  <Input type="number" step="0.01" value={gbmMu} onChange={e => setGbmMu(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Annualized Volatility (σ)</label>
                  <Input type="number" step="0.01" value={gbmSigma} min="0.01" onChange={e => setGbmSigma(Number(e.target.value))} className="bg-black font-mono" />
                </div>
              </div>
            )}

            {modelType === 'merton' && (
              <div className="grid gap-4 md:grid-cols-5">
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Drift (μ)</label>
                  <Input type="number" step="0.01" value={gbmMu} onChange={e => setGbmMu(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Volatility (σ)</label>
                  <Input type="number" step="0.01" value={gbmSigma} min="0.01" onChange={e => setGbmSigma(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Jumps/Yr (λ)</label>
                  <Input type="number" step="0.1" value={mertonLambda} min="0" onChange={e => setMertonLambda(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Jump Mean (μ_J)</label>
                  <Input type="number" step="0.01" value={mertonMuJ} onChange={e => setMertonMuJ(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Jump Vol (σ_J)</label>
                  <Input type="number" step="0.01" value={mertonSigmaJ} min="0" onChange={e => setMertonSigmaJ(Number(e.target.value))} className="bg-black font-mono" />
                </div>
              </div>
            )}

            {modelType === 'heston' && (
              <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Drift (μ)</label>
                  <Input type="number" step="0.01" value={gbmMu} onChange={e => setGbmMu(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Init Var (v0)</label>
                  <Input type="number" step="0.01" value={hestonInitialVar} min="0" onChange={e => setHestonInitialVar(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Mean Rev (κ)</label>
                  <Input type="number" step="0.1" value={hestonKappa} min="0" onChange={e => setHestonKappa(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Long Var (θ)</label>
                  <Input type="number" step="0.01" value={hestonTheta} min="0" onChange={e => setHestonTheta(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Vol of Vol (ξ)</label>
                  <Input type="number" step="0.01" value={hestonXi} min="0" onChange={e => setHestonXi(Number(e.target.value))} className="bg-black font-mono" />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-mono text-muted-foreground uppercase">Correlation (ρ)</label>
                  <Input type="number" step="0.1" value={hestonRho} min="-1" max="1" onChange={e => setHestonRho(Number(e.target.value))} className="bg-black font-mono" />
                </div>
              </div>
            )}

            {modelType === 'bootstrap' && (
              <div className="text-sm font-mono text-muted-foreground">
                <p>Bootstrap model will uniformly sample with replacement from the {historicalStats?.returns.length || 0} historical returns of {selectedTicker}.</p>
              </div>
            )}

            {modelType === 'markov' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <label className="text-xs font-mono text-muted-foreground uppercase">Number of Regimes (K)</label>
                    <Input type="number" value={kStates} min={2} max={5} onChange={e => handleKChange(Number(e.target.value))} className="bg-black font-mono w-24 h-8" />
                  </div>
                  <Button variant="outline" size="sm" onClick={handleAutoCalibrateMarkov} className="font-mono text-xs border-amber-500/50 text-amber-500 hover:bg-amber-500/10">
                    Auto-Calibrate from Rolling Vol
                  </Button>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  {/* Regime Parameters */}
                  <div>
                    <h4 className="text-xs font-mono text-amber-500 mb-2 uppercase">Regime Parameters (Annualized)</h4>
                    {regimesConfig.map((r, i) => (
                      <div key={i} className="flex gap-2 mb-2 items-center">
                        <span className="font-mono text-xs text-muted-foreground w-16">State {i}</span>
                        <Input type="number" step="0.01" placeholder="μ" value={r.mu} onChange={e => {
                          const newConfig = [...regimesConfig]
                          newConfig[i].mu = Number(e.target.value)
                          setRegimesConfig(newConfig)
                        }} className="bg-black font-mono h-8" />
                        <Input type="number" step="0.01" placeholder="σ" value={r.sigma} onChange={e => {
                          const newConfig = [...regimesConfig]
                          newConfig[i].sigma = Number(e.target.value)
                          setRegimesConfig(newConfig)
                        }} className="bg-black font-mono h-8" />
                      </div>
                    ))}
                  </div>

                  {/* Transition Matrix & Initial Probs */}
                  <div>
                    <h4 className="text-xs font-mono text-amber-500 mb-2 uppercase">Transition Matrix & Initial Probs</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs font-mono text-center">
                        <thead>
                          <tr>
                            <th className="font-normal text-muted-foreground pb-2">From \ To</th>
                            {Array.from({ length: kStates }).map((_, i) => <th key={i} className="font-normal text-muted-foreground pb-2">S{i}</th>)}
                            <th className="font-normal text-cyan-400 pb-2">Init Prob</th>
                          </tr>
                        </thead>
                        <tbody>
                          {transitionMatrix.map((row, i) => (
                            <tr key={i}>
                              <td className="text-muted-foreground pr-2">State {i}</td>
                              {row.map((val, j) => (
                                <td key={j} className="p-1">
                                  <Input type="number" step="0.05" value={Number(val.toFixed(3))} onChange={e => {
                                    const newMatrix = [...transitionMatrix]
                                    newMatrix[i][j] = Number(e.target.value)
                                    setTransitionMatrix(newMatrix)
                                  }} className="bg-black font-mono h-7 text-center w-16 px-1" />
                                </td>
                              ))}
                              <td className="p-1 pl-4">
                                <Input type="number" step="0.05" value={Number(initialProbabilities[i].toFixed(3))} onChange={e => {
                                  const newProbs = [...initialProbabilities]
                                  newProbs[i] = Number(e.target.value)
                                  setInitialProbabilities(newProbs)
                                }} className="bg-black font-mono h-7 text-center w-16 px-1 border-cyan-500/30 text-cyan-400" />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="mt-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                onClick={handleRunSimulation}
                disabled={isSimulating}
                className="bg-amber-500 hover:bg-amber-600 text-black font-bold font-mono uppercase"
              >
                {isSimulating ? <LoadingSpinner size="sm" className="mr-2 border-black" /> : <Dices className="mr-2 h-4 w-4" />}
                Generate Paths
              </Button>
              {modelType === 'markov' && (
                <div className="flex items-center gap-2">
                  <label className="text-xs font-mono text-muted-foreground uppercase">Color paths by:</label>
                  <select
                    value={colorMode}
                    onChange={e => setColorMode(e.target.value as ColorMode)}
                    className="bg-black border border-[#262626] text-white font-mono text-xs p-1 rounded"
                  >
                    <option value="hsl">Unique ID (Spaghetti)</option>
                    <option value="regime">Active Regime (Segments)</option>
                  </select>
                </div>
              )}
            </div>

            {processingTimeMs > 0 && !isSimulating && (
              <div className="text-xs font-mono text-muted-foreground uppercase">
                Worker Exec: <span className="text-amber-500">{processingTimeMs.toFixed(0)}ms</span> |
                Paths: {(results?.paths.length || 0).toLocaleString()}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results Rendering */}
      {results && stats && (
        <>
          <MetricGrid columns={5}>
            <MetricCard title="Expected Mean" value={`$${stats.mean.toFixed(2)}`} subtitle={`${horizon} bars`} />
            <MetricCard title="Median Price" value={`$${stats.median.toFixed(2)}`} />
            <MetricCard title="P(Gain)" value={`${(stats.pGain * 100).toFixed(1)}%`} valueClassName="text-[#00ff00]" />
            <MetricCard title="P(Loss)" value={`${(stats.pLoss * 100).toFixed(1)}%`} valueClassName="text-[#ff0000]" />
            <MetricCard title="Value at Risk (95%)" value={`$${stats.var95.toFixed(2)}`} subtitle="5th Percentile" />
          </MetricGrid>

          <SpaghettiChart
            paths={results.paths}
            regimes={results.regimes}
            historicalPrice={initialPrice}
            horizon={horizon}
            colorMode={colorMode}
            height={500}
          />

          <div className="grid gap-6 lg:grid-cols-2">
            <DistributionChart
              data={stats.histData}
              title="Endpoint Density"
              description={`Distribution of terminal prices at bar ${horizon}`}
              mean={stats.mean}
              median={stats.median}
            />

            {stats.regimePercentages && (
              <Card className="bg-[#0a0a0a] border-[#262626]">
                <CardHeader>
                  <CardTitle className="font-mono text-amber-500 uppercase">Regime Occupancy</CardTitle>
                  <CardDescription className="font-mono uppercase text-xs">Total simulated time spent in each state across all paths</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {stats.regimePercentages.map((pct, i) => (
                      <div key={i}>
                        <div className="flex justify-between text-xs font-mono uppercase mb-1">
                          <span>State {i}</span>
                          <span className="text-amber-500">{(pct * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-black rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500" style={{ width: `${pct * 100}%`, backgroundColor: ['#f59e0b', '#ef4444', '#3b82f6', '#10b981'][i % 4] }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </>
      )}
    </div>
  )
}
