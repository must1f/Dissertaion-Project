/**
 * TypeScript types for API
 */

export interface ApiError {
  error: string;
  details?: unknown;
}

export interface StockInfo {
  ticker: string;
  name?: string;
  first_date?: string;
  last_date?: string;
  record_count: number;
}

export interface OHLCVData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ticker: string;
}

export interface StockDataResponse {
  ticker: string;
  data: OHLCVData[];
  start_date?: string;
  end_date?: string;
  count: number;
}

export interface TrainingStatus {
  job_id: string;
  model_type: string;
  ticker: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  current_epoch: number;
  total_epochs: number;
  progress_percent: number;
  started_at?: string;
  completed_at?: string;
  elapsed_seconds?: number;
  current_train_loss?: number;
  current_val_loss?: number;
  best_val_loss?: number;
}

export interface TrainingRequest {
  model_type: string;
  ticker?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  sequence_length?: number;
  hidden_dim?: number;
  num_layers?: number;
  dropout?: number;
  enable_physics?: boolean;
  physics_weight?: number;
  early_stopping_patience?: number;
  use_curriculum?: boolean;
  save_checkpoints?: boolean;
}

export interface BacktestRequest {
  model_key: string;
  ticker: string;
  start_date?: string;
  end_date?: string;
  initial_capital?: number;
  commission_rate?: number;
  slippage_rate?: number;
  max_position_size?: number;
  stop_loss?: number;
  take_profit?: number;
  position_sizing_method?: string;
  signal_threshold?: number;
}

export interface Trade {
  id: string;
  timestamp: string;
  ticker: string;
  action: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  value: number;
  commission: number;
  slippage: number;
  position_before: number;
  position_after: number;
  pnl?: number;
  pnl_percent?: number;
}

export interface BacktestResults {
  model_key: string;
  ticker: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor?: number;
  total_trades: number;
  metrics: Record<string, any>;
  portfolio_history: Array<{
    timestamp: string;
    portfolio_value: number;
    cash: number;
    positions_value: number;
    daily_return?: number;
    cumulative_return: number;
  }>;
  equity_curve: number[];
  returns: number[];
  trades: Trade[];
  winning_trades: number;
  losing_trades: number;
}

export interface MonteCarloRequest {
  model_key: string;
  ticker: string;
  n_simulations?: number;
  horizon_days?: number;
  initial_price?: number;
  confidence_levels?: number[];
  random_seed?: number;
}

export interface MonteCarloResults {
  model_key: string;
  ticker: string;
  n_simulations: number;
  horizon_days: number;
  initial_price: number;
  run_date: string;
  final_price_mean: number;
  final_price_median: number;
  final_price_std: number;
  final_return_mean: number;
  final_return_std: number;
  confidence_intervals: Array<{
    level: number;
    lower: number;
    upper: number;
  }>;
  probability_of_loss: number;
  probability_of_gain: number;
  value_at_risk_95: number;
  expected_shortfall_95: number;
  sample_paths: Array<{
    path_id: number;
    prices: number[];
    final_price: number;
    total_return: number;
    max_drawdown: number;
  }>;
  histogram_bins: number[];
  histogram_counts: number[];
}
