/**
 * TypeScript types for models
 */

export type ModelStatus = 'not_trained' | 'training' | 'trained' | 'failed';

export interface PhysicsParameters {
  theta?: number;
  gamma?: number;
  temperature?: number;
  mu?: number;
  sigma?: number;
}

export interface ModelInfo {
  model_key: string;
  model_type: string;
  display_name: string;
  description?: string;
  status: ModelStatus;
  is_pinn: boolean;
  architecture?: Record<string, unknown>;
  input_dim?: number;
  hidden_dim?: number;
  num_layers?: number;
  trained_at?: string;
  training_epochs?: number;
  best_val_loss?: number;
  physics_parameters?: PhysicsParameters;
  checkpoint_path?: string;
  file_size_mb?: number;
}

export interface ModelListResponse {
  models: ModelInfo[];
  total: number;
  trained_count: number;
  pinn_count: number;
}

export interface ModelWeightsInfo {
  model_key: string;
  total_parameters: number;
  trainable_parameters: number;
  layer_info: Array<{
    name: string;
    shape: number[];
    parameters: number;
    trainable: boolean;
  }>;
}

export interface ModelComparisonItem {
  model_key: string;
  display_name: string;
  is_pinn: boolean;
  metrics: Record<string, number>;
  physics_parameters?: PhysicsParameters;
}

export interface ModelComparisonResponse {
  models: ModelComparisonItem[];
  metric_names: string[];
  best_by_metric: Record<string, string>;
}
