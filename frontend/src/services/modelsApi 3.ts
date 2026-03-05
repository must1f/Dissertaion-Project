/**
 * Models API service
 */

import api from './api';
import type { ModelInfo, ModelListResponse, ModelWeightsInfo, ModelComparisonResponse } from '../types/models';

export const modelsApi = {
  // List all models
  async listModels(): Promise<ModelListResponse> {
    const response = await api.get<ModelListResponse>('/api/models/');
    return response.data;
  },

  // List trained models
  async listTrainedModels(): Promise<ModelListResponse> {
    const response = await api.get<ModelListResponse>('/api/models/trained');
    return response.data;
  },

  // Get model types
  async getModelTypes(): Promise<{
    model_types: string[];
    categories: Record<string, string[]>;
  }> {
    const response = await api.get('/api/models/types');
    return response.data;
  },

  // Get single model
  async getModel(modelKey: string): Promise<ModelInfo> {
    const response = await api.get<ModelInfo>(`/api/models/${modelKey}`);
    return response.data;
  },

  // Get model weights info
  async getModelWeights(modelKey: string): Promise<ModelWeightsInfo> {
    const response = await api.get<ModelWeightsInfo>(`/api/models/${modelKey}/weights`);
    return response.data;
  },

  // Compare models
  async compareModels(modelKeys: string[]): Promise<ModelComparisonResponse> {
    const response = await api.get<ModelComparisonResponse>('/api/models/compare', {
      params: { model_keys: modelKeys.join(',') },
    });
    return response.data;
  },

  // Load model into memory
  async loadModel(modelKey: string, device?: string): Promise<{ message: string }> {
    const response = await api.post(`/api/models/${modelKey}/load`, null, {
      params: { device },
    });
    return response.data;
  },

  // Unload model from memory
  async unloadModel(modelKey: string): Promise<{ message: string }> {
    const response = await api.post(`/api/models/${modelKey}/unload`);
    return response.data;
  },
};

export default modelsApi;
