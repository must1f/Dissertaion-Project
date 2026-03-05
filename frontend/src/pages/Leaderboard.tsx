import { useQuery } from "@tanstack/react-query"
import metricsApi from "../services/metricsApi"
import type { LeaderboardEntry, LeaderboardResponse } from "../types/metrics"

const columns: Array<{ key: keyof LeaderboardEntry | "metric"; label: string }> = [
  { key: "rank", label: "#" },
  { key: "model_name", label: "Model" },
  { key: "metric_value", label: "Metric" },
  { key: "experiment_id", label: "Experiment" },
]

export default function Leaderboard() {
  const { data, isLoading, error } = useQuery<LeaderboardResponse>({
    queryKey: ["leaderboard", "sharpe_ratio"],
    queryFn: () => metricsApi.getLeaderboard("sharpe_ratio", 15),
  })

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Leaderboard</h1>
          <p className="text-sm text-slate-500">
            Top models by Sharpe ratio. Refresh after new evaluations complete.
          </p>
        </div>
      </div>

      {isLoading && <div className="text-slate-500">Loading leaderboard…</div>}
      {error && <div className="text-red-500">Failed to load leaderboard.</div>}

      {data && data.entries.length === 0 && (
        <div className="text-slate-500">No experiments logged yet.</div>
      )}

      {data && data.entries.length > 0 && (
        <div className="overflow-x-auto border rounded-lg bg-white shadow-sm">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                {columns.map((col) => (
                  <th key={col.key} className="px-4 py-2 text-left font-semibold text-slate-700">
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.entries.map((row) => (
                <tr key={row.experiment_id} className="border-t hover:bg-slate-50">
                  <td className="px-4 py-2 font-semibold text-slate-800">{row.rank}</td>
                  <td className="px-4 py-2">{row.model_name}</td>
                  <td className="px-4 py-2">{row.metric_value.toFixed(3)}</td>
                  <td className="px-4 py-2 text-slate-500">{row.experiment_id}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-4 py-2 text-xs text-slate-500">
            Generated at {new Date(data.generated_at).toLocaleString()} • metric: {data.metric}
          </div>
        </div>
      )}
    </div>
  )
}
