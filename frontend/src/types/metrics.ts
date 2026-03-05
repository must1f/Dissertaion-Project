/**
 * TypeScript types for metrics
 */

export interface MLMetrics {
  rmse: number;
  mae: number;
  mape: number;
  r2: number;
  directional_accuracy: number;
  mse?: number;
  explained_variance?: number;
}

export interface FinancialMetrics {
  total_return: number;
  annual_return: number;
  daily_return_mean: number;
  daily_return_std: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio?: number;
  information_ratio?: number;
  max_drawdown: number;
  max_drawdown_duration?: number;
  win_rate: number;
  profit_factor?: number;
  avg_win?: number;
  avg_loss?: number;
  total_trades?: number;
  winning_trades?: number;
  losing_trades?: number;
}

export interface PhysicsMetrics {
  total_physics_loss: number;
  gbm_loss?: number;
  ou_loss?: number;
  black_scholes_loss?: number;
  langevin_loss?: number;
  theta?: number;
  gamma?: number;
  temperature?: number;
  mu?: number;
  sigma?: number;
}

export interface ModelMetricsResponse {
  model_key: string;
  model_name: string;
  is_pinn: boolean;
  ml_metrics: MLMetrics;
  financial_metrics?: FinancialMetrics;
  physics_metrics?: PhysicsMetrics;
  evaluation_date?: string;
  test_period_start?: string;
  test_period_end?: string;
  sample_size?: number;
}

export interface MetricsComparisonResponse {
  models: ModelMetricsResponse[];
  metric_summary: Record<string, Record<string, number>>;
  best_by_metric: Record<string, string>;
  rankings: Record<string, string[]>;
}

export interface LeaderboardEntry {
  rank: number;
  experiment_id: string;
  model_name: string;
  metric_value: number;
  metric_name: string;
  other_metrics: Record<string, number>;
}

export interface LeaderboardResponse {
  metric: string;
  generated_at: string;
  n_experiments: number;
  entries: LeaderboardEntry[];
}
