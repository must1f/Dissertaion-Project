import { useQuery, useMutation } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import analysisApi, { RegimeHistoryPoint, ReturnsSeries } from "../services/analysisApi"
import { Card } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { ResponsiveContainer, LineChart, Line, Tooltip } from "recharts"

export default function ComprehensiveAnalysis() {
  const [ticker, setTicker] = useState("^GSPC")
  const [method, setMethod] = useState("rolling")
  const [stressResult, setStressResult] = useState<any | null>(null)

  const { data: regimeHistory, isLoading: loadingRegime } = useQuery({
    queryKey: ["regimeHistory", ticker, method],
    queryFn: () => analysisApi.getRegimeHistory(ticker, method),
  })

  const { data: returnsSeries, isLoading: loadingReturns } = useQuery({
    queryKey: ["returnsSeries", ticker],
    queryFn: () => analysisApi.getReturns(ticker),
  })

  const stressMutation = useMutation({
    mutationFn: analysisApi.runStressTest,
    onSuccess: (data) => setStressResult(data),
  })

  const latestReturnsPayload = useMemo(() => {
    if (!returnsSeries) return { returns: [], timestamps: [] }
    return { returns: returnsSeries.returns, timestamps: returnsSeries.timestamps }
  }, [returnsSeries])

  return (
    <div className="p-6 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <input
          className="border rounded px-3 py-1"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Ticker"
        />
        <select className="border rounded px-3 py-1" value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="rolling">Rolling</option>
          <option value="hmm">HMM</option>
          <option value="kmeans">KMeans</option>
        </select>
        <Button
          disabled={loadingReturns || !returnsSeries}
          onClick={() => stressMutation.mutate(latestReturnsPayload)}
          variant="outline"
          size="sm"
        >
          {loadingReturns ? "Loading returns…" : "Run Stress"}
        </Button>
      </div>

      <Card className="p-4">
        <h2 className="font-semibold mb-2">Regime History</h2>
        {loadingRegime && <div className="text-slate-500">Loading...</div>}
        {regimeHistory && <RegimeTable points={regimeHistory.history} />}
      </Card>

      {returnsSeries && (
        <Card className="p-4">
          <h2 className="font-semibold mb-2">Recent Returns</h2>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={returnsSeries.returns.slice(-60).map((r, idx) => ({
                  idx,
                  r,
                }))}
              >
                <Tooltip
                  formatter={(value: any) => Number(value).toFixed(4)}
                  labelFormatter={(label) => `t-${60 - label}`}
                />
                <Line type="monotone" dataKey="r" stroke="#0f172a" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {stressResult && (
        <Card className="p-4">
          <h2 className="font-semibold mb-2">Stress Test</h2>
          <StressSummary result={stressResult} />
          <details className="mt-3">
            <summary className="cursor-pointer text-sm text-slate-600">Raw response</summary>
            <pre className="text-xs bg-slate-50 p-2 rounded mt-1">{JSON.stringify(stressResult, null, 2)}</pre>
          </details>
        </Card>
      )}
    </div>
  )
}

function RegimeTable({ points }: { points: RegimeHistoryPoint[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-50">
          <tr>
            <th className="px-3 py-2 text-left">Date</th>
            <th className="px-3 py-2 text-left">Regime</th>
            <th className="px-3 py-2 text-left">Vol</th>
            <th className="px-3 py-2 text-left">Stress Window</th>
          </tr>
        </thead>
        <tbody>
          {points.slice(-60).map((p) => (
            <tr key={p.timestamp} className="border-t">
              <td className="px-3 py-2">{new Date(p.timestamp).toISOString().slice(0, 10)}</td>
              <td className="px-3 py-2">{p.regime}</td>
              <td className="px-3 py-2">{p.volatility.toFixed(3)}</td>
              <td className="px-3 py-2">{p.stress_window ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SummaryCard({ title, value, sub }: { title: string; value: string; sub?: string }) {
  return (
    <div className="rounded border px-3 py-2">
      <div className="text-xs text-slate-500">{title}</div>
      <div className="text-lg font-semibold">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

function StressSummary({ result }: { result: any }) {
  const cards = [
    {
      title: "Crises Analyzed",
      value: result.crises_analyzed ?? "-",
      sub: `Outperformed: ${result.crises_outperformed ?? "-"}`,
    },
    {
      title: "Avg Alpha",
      value: formatPct(result.avg_alpha),
      sub: `Avg Return: ${formatPct(result.avg_crisis_return)}`,
    },
    {
      title: "Worst Crisis",
      value: result.worst_crisis ?? "-",
      sub: `Best: ${result.best_crisis ?? "-"}`,
    },
  ]

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        {cards.map((c) => (
          <SummaryCard key={c.title} title={c.title} value={String(c.value)} sub={c.sub} />
        ))}
      </div>
    </>
  )
}

function formatPct(x?: number | null) {
  if (x === null || x === undefined) return "-"
  return `${(x * 100).toFixed(2)}%`
}
