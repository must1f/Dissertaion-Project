import api from "./api"

export interface RegimeHistoryPoint {
  timestamp: string
  regime: string
  probability: number
  volatility: number
  stress_window?: string | null
}

export interface RegimeHistoryResponse {
  ticker: string
  start_date: string
  end_date: string
  total_points: number
  history: RegimeHistoryPoint[]
  regime_summary: Record<string, unknown>
}

export interface StressRunRequest {
  returns: number[]
  timestamps: string[]
}

export interface StressRunResponse {
  crises_analyzed: number
  crises_outperformed: number
  avg_alpha?: number
  avg_crisis_return: number
  avg_max_drawdown: number
  worst_crisis: string
  best_crisis: string
  crisis_results: any[]
}

export interface ReturnsSeries {
  ticker: string
  start_date: string
  end_date: string
  timestamps: string[]
  returns: number[]
}

export const analysisApi = {
  async getRegimeHistory(ticker = "^GSPC", method = "rolling") {
    const response = await api.get<RegimeHistoryResponse>("/api/analysis/regime/history", {
      params: { ticker, method },
    })
    return response.data
  },

  async getReturns(ticker = "^GSPC", start?: string, end?: string, max_points = 1500) {
    const response = await api.get<ReturnsSeries>("/api/analysis/returns", {
      params: { ticker, start_date: start, end_date: end, max_points },
    })
    return response.data
  },

  async runStressTest(payload: StressRunRequest) {
    const response = await api.post<StressRunResponse>("/api/analysis/stress/run", payload)
    return response.data
  },
}

export default analysisApi
