/**
 * React Query hooks for models
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelsApi } from '../services/modelsApi';

// Query keys
export const modelKeys = {
  all: ['models'] as const,
  list: () => [...modelKeys.all, 'list'] as const,
  trained: () => [...modelKeys.all, 'trained'] as const,
  types: () => [...modelKeys.all, 'types'] as const,
  detail: (key: string) => [...modelKeys.all, 'detail', key] as const,
  weights: (key: string) => [...modelKeys.all, 'weights', key] as const,
  comparison: (keys: string[]) => [...modelKeys.all, 'comparison', keys] as const,
};

// List all models
export function useModels() {
  return useQuery({
    queryKey: modelKeys.list(),
    queryFn: modelsApi.listModels,
    staleTime: 30000, // 30 seconds
  });
}

// List trained models only
export function useTrainedModels() {
  return useQuery({
    queryKey: modelKeys.trained(),
    queryFn: modelsApi.listTrainedModels,
    staleTime: 30000,
  });
}

// Get model types
export function useModelTypes() {
  return useQuery({
    queryKey: modelKeys.types(),
    queryFn: modelsApi.getModelTypes,
    staleTime: 60000, // 1 minute (rarely changes)
  });
}

// Get single model
export function useModel(modelKey: string) {
  return useQuery({
    queryKey: modelKeys.detail(modelKey),
    queryFn: () => modelsApi.getModel(modelKey),
    enabled: !!modelKey,
  });
}

// Get model weights
export function useModelWeights(modelKey: string) {
  return useQuery({
    queryKey: modelKeys.weights(modelKey),
    queryFn: () => modelsApi.getModelWeights(modelKey),
    enabled: !!modelKey,
  });
}

// Compare models
export function useModelComparison(modelKeys: string[]) {
  return useQuery({
    queryKey: modelKeys.length > 0 ? ['models', 'comparison', modelKeys] : ['models', 'comparison'],
    queryFn: () => modelsApi.compareModels(modelKeys),
    enabled: modelKeys.length > 0,
  });
}

// Load model mutation
export function useLoadModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ modelKey, device }: { modelKey: string; device?: string }) =>
      modelsApi.loadModel(modelKey, device),
    onSuccess: (_, { modelKey }) => {
      queryClient.invalidateQueries({ queryKey: modelKeys.detail(modelKey) });
    },
  });
}

// Unload model mutation
export function useUnloadModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelKey: string) => modelsApi.unloadModel(modelKey),
    onSuccess: (_, modelKey) => {
      queryClient.invalidateQueries({ queryKey: modelKeys.detail(modelKey) });
    },
  });
}
