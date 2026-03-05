/**
 * Volatility Forecasting API service
 */

import api from './api';
import type {
  VolatilityModelsResponse,
  VolatilityModelInfo,
  VolatilityDataRequest,
  VolatilityDataResponse,
  VolatilityTrainingRequest,
  VolatilityTrainingResponse,
  VolatilityPredictionRequest,
  VolatilityPredictionResponse,
  StrategyBacktestRequest,
  StrategyBacktestResponse,
  ModelComparisonRequest,
  ModelComparisonResponse,
  VolatilityMetricsInfo,
  BaselinesInfoResponse,
  PhysicsConstraintsResponse,
} from '../types/volatility';

export const volatilityApi = {
  /**
   * Get all available volatility forecasting models
   */
  async getModels(): Promise<VolatilityModelsResponse> {
    const response = await api.get<VolatilityModelsResponse>('/api/volatility/models');
    return response.data;
  },

  /**
   * Get information about a specific model
   */
  async getModelInfo(modelType: string): Promise<{ model_type: string } & VolatilityModelInfo> {
    const response = await api.get<{ model_type: string } & VolatilityModelInfo>(
      `/api/volatility/models/${modelType}`
    );
    return response.data;
  },

  /**
   * Prepare data for volatility forecasting
   */
  async prepareData(request: VolatilityDataRequest): Promise<VolatilityDataResponse> {
    const response = await api.post<VolatilityDataResponse>(
      '/api/volatility/data/prepare',
      request
    );
    return response.data;
  },

  /**
   * Train a volatility forecasting model
   */
  async trainModel(request: VolatilityTrainingRequest): Promise<VolatilityTrainingResponse> {
    const response = await api.post<VolatilityTrainingResponse>(
      '/api/volatility/train',
      request
    );
    return response.data;
  },

  /**
   * Make volatility predictions
   */
  async predict(request: VolatilityPredictionRequest): Promise<VolatilityPredictionResponse> {
    const response = await api.post<VolatilityPredictionResponse>(
      '/api/volatility/predict',
      request
    );
    return response.data;
  },

  /**
   * Backtest volatility targeting strategy
   */
  async backtestStrategy(request: StrategyBacktestRequest): Promise<StrategyBacktestResponse> {
    const response = await api.post<StrategyBacktestResponse>(
      '/api/volatility/backtest',
      request
    );
    return response.data;
  },

  /**
   * Compare multiple volatility forecasting models
   */
  async compareModels(request: ModelComparisonRequest): Promise<ModelComparisonResponse> {
    const response = await api.post<ModelComparisonResponse>(
      '/api/volatility/compare',
      request
    );
    return response.data;
  },

  /**
   * Get information about available volatility metrics
   */
  async getMetricsInfo(): Promise<VolatilityMetricsInfo> {
    const response = await api.get<VolatilityMetricsInfo>('/api/volatility/metrics');
    return response.data;
  },

  /**
   * Get information about baseline volatility models
   */
  async getBaselinesInfo(): Promise<BaselinesInfoResponse> {
    const response = await api.get<BaselinesInfoResponse>('/api/volatility/baselines');
    return response.data;
  },

  /**
   * Get information about physics constraints for volatility PINNs
   */
  async getPhysicsConstraintsInfo(): Promise<PhysicsConstraintsResponse> {
    const response = await api.get<PhysicsConstraintsResponse>('/api/volatility/physics-constraints');
    return response.data;
  },
};

export default volatilityApi;
