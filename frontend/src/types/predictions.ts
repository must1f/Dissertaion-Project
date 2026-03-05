/**
 * TypeScript types for predictions
 */

export type UncertaintyMethod = 'mc_dropout' | 'ensemble' | 'both';
export type SignalAction = 'BUY' | 'SELL' | 'HOLD';

export interface PredictionInterval {
  lower: number;
  upper: number;
  confidence: number;
}

export interface PredictionResult {
  timestamp: string;
  ticker: string;
  model_key: string;
  predicted_price: number;
  predicted_return: number;
  current_price: number;
  uncertainty_std?: number;
  prediction_interval?: PredictionInterval;
  confidence_score?: number;
  signal_action?: SignalAction;
  expected_return?: number;
}

export interface PredictionRequest {
  ticker: string;
  model_key: string;
  sequence_length?: number;
  horizon?: number;
  estimate_uncertainty?: boolean;
  uncertainty_method?: UncertaintyMethod;
  n_mc_samples?: number;
  generate_signal?: boolean;
  signal_threshold?: number;
}

export interface PredictionResponse {
  success: boolean;
  prediction: PredictionResult;
  model_info: Record<string, unknown>;
  physics_parameters?: Record<string, number>;
  processing_time_ms: number;
}

export interface PredictionHistoryItem {
  id: string;
  timestamp: string;
  ticker: string;
  model_key: string;
  predicted_price: number;
  actual_price?: number;
  error?: number;
  signal_action?: SignalAction;
}

export interface PredictionHistoryResponse {
  predictions: PredictionHistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface LatestPredictionResponse {
  ticker: string;
  predictions: Record<string, PredictionResult>;
  consensus_signal?: SignalAction;
  last_updated: string;
}
