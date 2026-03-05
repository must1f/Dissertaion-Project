/**
 * Dissertation Analysis API Service
 *
 * Provides publication-quality metrics and data for PINN volatility forecasting research.
 */

import { api } from './api';

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface VolatilityMetrics {
  mse: number;
  mae: number;
  rmse: number;
  qlike: number;
  hmse: number;
  mz_r2: number;
  directional_accuracy: number;
  log_likelihood?: number;
}

export interface EconomicMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  win_rate: number;
  profit_factor: number;
}

export interface VaRBreachResult {
  confidence_level: number;
  expected_rate: number;
  actual_rate: number;
  breach_count: number;
  total_obs: number;
  kupiec_pvalue?: number;
  passes_kupiec?: boolean;
}

export interface PhysicsComplianceMetrics {
  gbm_residual_mean?: number;
  gbm_residual_std?: number;
  ou_residual_mean?: number;
  ou_residual_std?: number;
  bs_residual_mean?: number;
  bs_residual_std?: number;
  total_physics_loss?: number;
  data_loss?: number;
  physics_loss_ratio?: number;
}

export interface DissertationMetricsResponse {
  model_name: string;
  generated_at: string;
  volatility_metrics: VolatilityMetrics;
  economic_metrics: EconomicMetrics;
  var_analysis: VaRBreachResult[];
  physics_compliance?: PhysicsComplianceMetrics;
}

export interface ModelComparisonEntry {
  model_name: string;
  qlike: number;
  mz_r2: number;
  directional_accuracy: number;
  sharpe_ratio: number;
  max_drawdown: number;
  is_pinn: boolean;
}

export interface DieboldMarianoResult {
  model_1: string;
  model_2: string;
  dm_statistic: number;
  p_value: number;
  significant: boolean;
  better_model?: string;
}

export interface ModelComparisonResponse {
  models: ModelComparisonEntry[];
  dm_tests: DieboldMarianoResult[];
  best_qlike: string;
  best_sharpe: string;
  mcs_included: string[];
}

export interface ForecastDataResponse {
  dates: string[];
  realized_vol: number[];
  predicted_vol: number[];
  returns: number[];
  errors: number[];
  regimes: string[];
}

export interface DissertationAvailableModelsResponse {
  baseline: string[];
  pinn: string[];
  advanced: string[];
  total: number;
}

export interface DissertationSummary {
  total_models_evaluated: number;
  pinn_models: number;
  baseline_models: number;
  models: Record<string, {
    qlike?: number;
    sharpe?: number;
    dir_acc?: number;
    max_dd?: number;
  }>;
  best_qlike_model?: string;
  best_qlike_value?: number;
  best_sharpe_model?: string;
  best_sharpe_value?: number;
  pinn_qlike_improvement?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get comprehensive dissertation metrics for a model
 */
export async function getDissertationMetrics(
  modelKey: string
): Promise<DissertationMetricsResponse> {
  const response = await api.get<DissertationMetricsResponse>(
    `/api/dissertation/metrics/${modelKey}`
  );
  return response.data;
}

/**
 * Compare multiple models for dissertation analysis
 */
export async function compareModels(
  modelKeys: string[] = ['pinn_global', 'pinn_gbm', 'pinn_ou', 'garch', 'lstm']
): Promise<ModelComparisonResponse> {
  const response = await api.get<ModelComparisonResponse>(
    '/api/dissertation/comparison',
    { params: { model_keys: modelKeys.join(',') } }
  );
  return response.data;
}

/**
 * Get forecast data for visualization
 */
export async function getForecastData(
  modelKey: string,
  nPoints: number = 252
): Promise<ForecastDataResponse> {
  const response = await api.get<ForecastDataResponse>(
    `/api/dissertation/forecast-data/${modelKey}`,
    { params: { n_points: nPoints } }
  );
  return response.data;
}

/**
 * Get list of available trained models
 */
export async function getAvailableModels(): Promise<DissertationAvailableModelsResponse> {
  const response = await api.get<DissertationAvailableModelsResponse>(
    '/api/dissertation/available-models'
  );
  return response.data;
}

/**
 * Get dissertation summary statistics
 */
export async function getDissertationSummary(): Promise<DissertationSummary> {
  const response = await api.get<DissertationSummary>(
    '/api/dissertation/summary'
  );
  return response.data;
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Format metric value for display
 */
export function formatMetric(value: number, type: 'percent' | 'ratio' | 'number' = 'number'): string {
  if (value === undefined || value === null || isNaN(value)) return 'N/A';

  switch (type) {
    case 'percent':
      return `${(value * 100).toFixed(2)}%`;
    case 'ratio':
      return value.toFixed(3);
    case 'number':
    default:
      return value.toFixed(4);
  }
}

/**
 * Determine metric status for visualization
 */
export function getMetricStatus(
  metricName: string,
  value: number
): 'good' | 'neutral' | 'poor' {
  const thresholds: Record<string, { good: number; poor: number; higherBetter: boolean }> = {
    sharpe_ratio: { good: 1.0, poor: 0.3, higherBetter: true },
    sortino_ratio: { good: 1.5, poor: 0.5, higherBetter: true },
    max_drawdown: { good: -0.15, poor: -0.35, higherBetter: true },
    calmar_ratio: { good: 1.0, poor: 0.3, higherBetter: true },
    qlike: { good: 0.08, poor: 0.15, higherBetter: false },
    mz_r2: { good: 0.7, poor: 0.4, higherBetter: true },
    directional_accuracy: { good: 0.6, poor: 0.45, higherBetter: true },
    win_rate: { good: 0.55, poor: 0.45, higherBetter: true },
    profit_factor: { good: 1.5, poor: 1.0, higherBetter: true },
  };

  const config = thresholds[metricName];
  if (!config) return 'neutral';

  if (config.higherBetter) {
    if (value >= config.good) return 'good';
    if (value <= config.poor) return 'poor';
  } else {
    if (value <= config.good) return 'good';
    if (value >= config.poor) return 'poor';
  }

  return 'neutral';
}

/**
 * Get color for model based on type
 */
export function getModelColor(modelName: string): string {
  if (modelName.includes('pinn') || modelName.includes('PINN')) {
    if (modelName.includes('global') || modelName.includes('Global')) return '#009E73';
    if (modelName.includes('gbm') || modelName.includes('GBM')) return '#56B4E9';
    if (modelName.includes('ou') || modelName.includes('OU')) return '#CC79A7';
    if (modelName.includes('bs') || modelName.includes('black')) return '#E69F00';
    return '#0072B2';
  }
  if (modelName.includes('garch') || modelName.includes('GARCH')) return '#D55E00';
  if (modelName.includes('lstm') || modelName.includes('LSTM')) return '#0072B2';
  if (modelName.includes('transformer')) return '#F0E442';
  return '#999999';
}

export default {
  getDissertationMetrics,
  compareModels,
  getForecastData,
  getAvailableModels,
  getDissertationSummary,
  formatMetric,
  getMetricStatus,
  getModelColor,
};
