/**
 * Metrics API service
 */

import api from './api';
import type {
  MLMetrics,
  FinancialMetrics,
  PhysicsMetrics,
  ModelMetricsResponse,
  MetricsComparisonResponse,
  LeaderboardResponse,
} from '../types/metrics';

export interface FinancialMetricsRequest {
  returns: number[];
  risk_free_rate?: number;
  periods_per_year?: number;
  benchmark_returns?: number[];
}

export interface FinancialMetricsResponse {
  metrics: FinancialMetrics;
  input_summary: Record<string, unknown>;
}

export const metricsApi = {
  // Calculate financial metrics (GET with query params)
  async calculateFinancialMetrics(
    returns: number[],
    riskFreeRate = 0.02,
    periodsPerYear = 252
  ): Promise<FinancialMetrics> {
    const response = await api.get<FinancialMetrics>('/api/metrics/financial', {
      params: {
        returns: returns.join(','),
        risk_free_rate: riskFreeRate,
        periods_per_year: periodsPerYear,
      },
    });
    return response.data;
  },

  // Calculate financial metrics (POST with body)
  async calculateFinancialMetricsPost(
    request: FinancialMetricsRequest
  ): Promise<FinancialMetricsResponse> {
    const response = await api.post<FinancialMetricsResponse>(
      '/api/metrics/financial',
      request
    );
    return response.data;
  },

  // Calculate ML metrics
  async calculateMLMetrics(yTrue: number[], yPred: number[]): Promise<MLMetrics> {
    const response = await api.get<MLMetrics>('/api/metrics/ml', {
      params: {
        y_true: yTrue.join(','),
        y_pred: yPred.join(','),
      },
    });
    return response.data;
  },

  // Get physics metrics for a model
  async getPhysicsMetrics(modelKey: string): Promise<PhysicsMetrics> {
    const response = await api.get<PhysicsMetrics>(`/api/metrics/physics/${modelKey}`);
    return response.data;
  },

  // Get all metrics for a model
  async getModelMetrics(modelKey: string): Promise<ModelMetricsResponse> {
    const response = await api.get<ModelMetricsResponse>(`/api/metrics/model/${modelKey}`);
    return response.data;
  },

  // Compare metrics across models
  async compareMetrics(modelKeys: string[]): Promise<MetricsComparisonResponse> {
    const response = await api.get<MetricsComparisonResponse>('/api/metrics/comparison', {
      params: { model_keys: modelKeys.join(',') },
    });
    return response.data;
  },

  // Get saved metrics
  async getSavedMetrics(modelKey: string): Promise<Record<string, unknown>> {
    const response = await api.get(`/api/metrics/saved/${modelKey}`);
    return response.data;
  },

  // Leaderboard
  async getLeaderboard(metric = 'sharpe_ratio', topN = 10): Promise<LeaderboardResponse> {
    const response = await api.get<LeaderboardResponse>('/api/metrics/leaderboard', {
      params: { metric, top_n: topN },
    });
    return response.data;
  },
};

export default metricsApi;
