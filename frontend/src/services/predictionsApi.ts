/**
 * Predictions API service
 */

import api from './api';
import type {
  PredictionRequest,
  PredictionResponse,
  PredictionHistoryResponse,
  LatestPredictionResponse,
} from '../types/predictions';

export interface BatchPredictionRequest {
  tickers: string[];
  model_key: string;
  sequence_length?: number;
  estimate_uncertainty?: boolean;
}

export interface BatchPredictionResponse {
  success: boolean;
  predictions: PredictionResponse[];
  failed_tickers: string[];
  total_processing_time_ms: number;
}

export const predictionsApi = {
  // Run single prediction
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await api.post<PredictionResponse>('/api/predictions/predict', request);
    return response.data;
  },

  // Run batch prediction
  async batchPredict(request: BatchPredictionRequest): Promise<BatchPredictionResponse> {
    const response = await api.post<BatchPredictionResponse>(
      '/api/predictions/predict/batch',
      request
    );
    return response.data;
  },

  // Get prediction history
  async getHistory(
    ticker?: string,
    modelKey?: string,
    page = 1,
    pageSize = 50
  ): Promise<PredictionHistoryResponse> {
    const response = await api.get<PredictionHistoryResponse>('/api/predictions/history', {
      params: {
        ticker,
        model_key: modelKey,
        page,
        page_size: pageSize,
      },
    });
    return response.data;
  },

  // Get latest predictions for a ticker
  async getLatest(ticker: string): Promise<LatestPredictionResponse> {
    const response = await api.get<LatestPredictionResponse>(
      `/api/predictions/${ticker}/latest`
    );
    return response.data;
  },

  // Clear cache
  async clearCache(modelKey?: string): Promise<{ message: string }> {
    const response = await api.delete('/api/predictions/cache', {
      params: { model_key: modelKey },
    });
    return response.data;
  },
};

export default predictionsApi;
