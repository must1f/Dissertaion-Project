/**
 * TypeScript types for volatility forecasting
 */

// Model type categories
export type VolatilityModelType = 'neural' | 'pinn' | 'baseline';

// Available volatility models
export interface VolatilityModelInfo {
  name: string;
  type: VolatilityModelType;
  description: string;
}

export interface VolatilityModelsResponse {
  models: Record<string, VolatilityModelInfo>;
  total: number;
  by_type: {
    neural: number;
    pinn: number;
    baseline: number;
  };
  has_modules: boolean;
}

// Data preparation
export interface VolatilityDataRequest {
  ticker?: string;
  start_date?: string;
  end_date?: string | null;
  horizon?: number;
  seq_length?: number;
}

export interface VolatilityDataResponse {
  ticker: string;
  n_samples: number;
  n_train: number;
  n_val: number;
  n_test: number;
  n_features: number;
  feature_names: string[];
  horizon: number;
}

// Training
export interface VolatilityTrainingRequest {
  model_type: string;
  ticker?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  hidden_dim?: number;
  num_layers?: number;
  enable_physics?: boolean;
  lookback?: number;
  decay?: number;
}

export interface VolatilityTrainingHistory {
  train_loss: number[];
  val_loss: number[];
  val_qlike: number[];
  val_r2: number[];
}

export interface VolatilityPhysicsParams {
  theta?: number;
  omega?: number;
  alpha?: number;
  beta?: number;
  kappa?: number;
  xi?: number;
  rho?: number;
}

export interface VolatilityTrainingResponse {
  model_type: string;
  epochs_trained: number;
  training_time: number;
  best_val_loss: number;
  metrics: Record<string, number>;
  history: VolatilityTrainingHistory;
  physics_params?: VolatilityPhysicsParams;
}

// Prediction
export interface VolatilityPredictionRequest {
  model_type: string;
  n_steps?: number;
}

export interface VolatilityPredictionResponse {
  model_type: string;
  predictions?: number[];
  shape?: number[];
  variance?: number[];
  volatility?: number[];
}

// Strategy backtesting
export interface StrategyBacktestRequest {
  model_type: string;
  target_vol?: number;
  min_weight?: number;
  max_weight?: number;
  transaction_cost?: number;
}

export interface StrategyBacktestResponse {
  model_type: string;
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  benchmark_sharpe: number;
  information_ratio: number;
  avg_leverage: number;
  turnover: number;
  realized_vol: number;
  vol_tracking_error: number;
  equity_curve: number[];
  weights: number[];
}

// Model comparison
export interface ModelComparisonRequest {
  model_types: string[];
}

export interface ModelComparisonResult {
  model: string;
  mse?: number;
  mae?: number;
  rmse?: number;
  qlike?: number;
  hmse?: number;
  r2?: number;
  log_likelihood?: number;
}

export interface MCSResult {
  included_models: string[];
  p_values: Record<string, number>;
  eliminated_order: string[];
}

export interface ModelComparisonResponse {
  results: ModelComparisonResult[];
  mcs: MCSResult | null;
  best_qlike: string | null;
  best_r2: string | null;
}

// Metrics info
export interface MetricInfo {
  name: string;
  description: string;
}

export interface VolatilityMetricsInfo {
  statistical_metrics: MetricInfo[];
  economic_metrics: MetricInfo[];
  diagnostic_tests: MetricInfo[];
}

// Baseline models info
export interface BaselineModelInfo {
  name: string;
  description: string;
  formula: string;
  default_lambda?: number;
}

export interface BaselinesInfoResponse {
  baselines: Record<string, BaselineModelInfo>;
  reference: string;
}

// Physics constraints info
export interface PhysicsConstraintInfo {
  name: string;
  equation: string;
  description: string;
  learnable_params?: string[];
}

export interface PhysicsConstraintsResponse {
  constraints: Record<string, PhysicsConstraintInfo>;
  reference: string;
}
