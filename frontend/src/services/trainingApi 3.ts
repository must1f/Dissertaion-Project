/**
 * Training API service
 */

import api from "./api"

export interface TrainingModeInfo {
  mode: "real" | "simulated"
  using_real_models: boolean
  import_error: string | null
  message: string
}

export interface ResearchMetrics {
  // Prediction metrics
  rmse?: number | null
  mae?: number | null
  mape?: number | null
  mse?: number | null
  r2?: number | null
  directional_accuracy?: number | null

  // Financial metrics
  sharpe_ratio?: number | null
  sortino_ratio?: number | null
  max_drawdown?: number | null
  calmar_ratio?: number | null
  annualized_return?: number | null
  total_return?: number | null
  volatility?: number | null
  win_rate?: number | null
}

export interface ModelConfig {
  model_key: string
  enabled: boolean
  epochs?: number
  learning_rate?: number
  batch_size?: number
  hidden_dim?: number
  num_layers?: number
  dropout?: number
}

export interface BatchTrainingRequest {
  models: ModelConfig[]
  ticker: string
  epochs: number
  batch_size: number
  learning_rate: number
  sequence_length: number
  hidden_dim: number
  num_layers: number
  dropout: number
  gradient_clip_norm: number
  scheduler_patience: number
  early_stopping_patience: number
  research_mode: boolean
  force_refresh: boolean
  enable_physics: boolean
}

export interface BatchTrainingStartResponse {
  success: boolean
  batch_id: string
  message: string
  total_models: number
  model_keys: string[]
  websocket_url: string
}

export interface ModelStatus {
  model_key: string
  model_name: string
  model_type: string
  status: string
  current_epoch: number
  total_epochs: number
  train_loss: number | null
  val_loss: number | null
  best_val_loss: number | null
  val_rmse: number | null
  val_mae: number | null
  val_mape: number | null
  val_r2: number | null
  val_directional_accuracy: number | null
  data_loss: number | null
  physics_loss: number | null
  progress_percent: number
  test_metrics?: ResearchMetrics | null
}

export interface BatchTrainingStatus {
  batch_id: string
  status: string
  total_models: number
  completed_models: number
  failed_models: number
  current_model: string | null
  models: ModelStatus[]
  overall_progress: number
  config: Record<string, any>
}

export interface AvailableModel {
  name: string
  type: string
  description: string
  physics_constraints?: Record<string, number>
}

export interface AvailableModelsResponse {
  models: Record<string, AvailableModel>
  total: number
  by_type: Record<string, number>
}

// --- Single training job types ---

export interface TrainingJobInfo {
  job_id: string
  model_type: string
  ticker: string
  status: string
  current_epoch: number
  total_epochs: number
  progress_percent: number
  started_at?: string | null
  completed_at?: string | null
  elapsed_seconds?: number | null
  current_train_loss?: number | null
  current_val_loss?: number | null
  best_val_loss?: number | null
  saved_model_name?: string | null
  config: Record<string, unknown>
  final_metrics?: Record<string, unknown> | null
}

export interface EpochMetrics {
  epoch: number
  train_loss: number
  val_loss: number
  learning_rate: number
  data_loss?: number | null
  physics_loss?: number | null
  theta?: number | null
  gamma?: number | null
  temperature?: number | null
  epoch_time_seconds: number
}

export interface TrainingStatusResponse {
  job: TrainingJobInfo
  history?: Record<string, number[]> | null
}

export interface TrainingHistoryResponse {
  job_id: string
  model_type: string
  epochs: EpochMetrics[]
  best_epoch: number
  best_val_loss: number
  final_metrics: Record<string, number>
}

export interface TrainingRunListResponse {
  runs: TrainingJobInfo[]
  total: number
  page: number
  page_size: number
}

export const trainingApi = {
  // Get current training mode (real vs simulated)
  getTrainingMode: async (): Promise<TrainingModeInfo> => {
    const response = await api.get("/api/training/mode")
    return response.data
  },

  // Get available models for batch training
  getAvailableModels: async (): Promise<AvailableModelsResponse> => {
    const response = await api.get("/api/training/batch/models")
    return response.data
  },

  // Start batch training
  startBatchTraining: async (request: BatchTrainingRequest): Promise<BatchTrainingStartResponse> => {
    const response = await api.post("/api/training/batch/start", request)
    return response.data
  },

  // Stop batch training
  stopBatchTraining: async (batchId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/api/training/batch/stop/${batchId}`)
    return response.data
  },

  // Get batch training status
  getBatchStatus: async (batchId: string): Promise<BatchTrainingStatus> => {
    const response = await api.get(`/api/training/batch/status/${batchId}`)
    return response.data.batch
  },

  // List batch jobs
  listBatchJobs: async (params?: {
    status?: string
    page?: number
    page_size?: number
  }): Promise<{
    batches: BatchTrainingStatus[]
    total: number
    page: number
    page_size: number
  }> => {
    const response = await api.get("/api/training/batch/list", { params })
    return response.data
  },

  // Single model training
  startTraining: async (config: {
    model_type: string
    ticker: string
    epochs: number
    batch_size: number
    learning_rate: number
    hidden_dim: number
    num_layers: number
    dropout: number
    enable_physics?: boolean
    physics_weight?: number
  }): Promise<{
    success: boolean
    job_id: string
    message: string
    websocket_url: string
  }> => {
    const response = await api.post("/api/training/start", config)
    return response.data
  },

  // Stop training
  stopTraining: async (jobId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/api/training/stop/${jobId}`)
    return response.data
  },

  // Get training status
  getTrainingStatus: async (jobId: string): Promise<TrainingStatusResponse> => {
    const response = await api.get(`/api/training/status/${jobId}`)
    return response.data
  },

  // Get training history
  getTrainingHistory: async (jobId: string): Promise<TrainingHistoryResponse> => {
    const response = await api.get(`/api/training/history/${jobId}`)
    return response.data
  },

  // List training runs
  listTrainingRuns: async (params?: {
    status?: string
    page?: number
    page_size?: number
  }): Promise<TrainingRunListResponse> => {
    const response = await api.get("/api/training/history", { params })
    return response.data
  },

  // Get active jobs
  getActiveJobs: async (): Promise<{
    active_jobs: TrainingJobInfo[]
    count: number
  }> => {
    const response = await api.get("/api/training/active")
    return response.data
  },
}

export default trainingApi
