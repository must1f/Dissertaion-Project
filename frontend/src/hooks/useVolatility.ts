/**
 * React Query hooks for volatility forecasting
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { volatilityApi } from '../services/volatilityApi';
import type {
  VolatilityDataRequest,
  VolatilityTrainingRequest,
  VolatilityPredictionRequest,
  StrategyBacktestRequest,
  ModelComparisonRequest,
} from '../types/volatility';

// Query keys
export const volatilityKeys = {
  all: ['volatility'] as const,
  models: () => [...volatilityKeys.all, 'models'] as const,
  modelInfo: (modelType: string) => [...volatilityKeys.all, 'model', modelType] as const,
  data: () => [...volatilityKeys.all, 'data'] as const,
  metricsInfo: () => [...volatilityKeys.all, 'metricsInfo'] as const,
  baselines: () => [...volatilityKeys.all, 'baselines'] as const,
  physicsConstraints: () => [...volatilityKeys.all, 'physicsConstraints'] as const,
  comparison: (modelTypes: string[]) => [...volatilityKeys.all, 'comparison', modelTypes] as const,
};

/**
 * Get all available volatility forecasting models
 */
export function useVolatilityModels() {
  return useQuery({
    queryKey: volatilityKeys.models(),
    queryFn: volatilityApi.getModels,
    staleTime: 60000, // Cache for 1 minute
  });
}

/**
 * Get information about a specific volatility model
 */
export function useVolatilityModelInfo(modelType: string) {
  return useQuery({
    queryKey: volatilityKeys.modelInfo(modelType),
    queryFn: () => volatilityApi.getModelInfo(modelType),
    enabled: !!modelType,
  });
}

/**
 * Get information about available volatility metrics
 */
export function useVolatilityMetricsInfo() {
  return useQuery({
    queryKey: volatilityKeys.metricsInfo(),
    queryFn: volatilityApi.getMetricsInfo,
    staleTime: Infinity, // Static info, never stale
  });
}

/**
 * Get information about baseline volatility models
 */
export function useVolatilityBaselines() {
  return useQuery({
    queryKey: volatilityKeys.baselines(),
    queryFn: volatilityApi.getBaselinesInfo,
    staleTime: Infinity,
  });
}

/**
 * Get information about physics constraints for volatility PINNs
 */
export function useVolatilityPhysicsConstraints() {
  return useQuery({
    queryKey: volatilityKeys.physicsConstraints(),
    queryFn: volatilityApi.getPhysicsConstraintsInfo,
    staleTime: Infinity,
  });
}

/**
 * Prepare data for volatility forecasting
 */
export function usePrepareVolatilityData() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: VolatilityDataRequest) => volatilityApi.prepareData(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: volatilityKeys.data() });
    },
  });
}

/**
 * Train a volatility forecasting model
 */
export function useTrainVolatilityModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: VolatilityTrainingRequest) => volatilityApi.trainModel(request),
    onSuccess: (_, variables) => {
      // Invalidate relevant queries after training
      queryClient.invalidateQueries({ queryKey: volatilityKeys.models() });
      queryClient.invalidateQueries({
        queryKey: volatilityKeys.modelInfo(variables.model_type),
      });
    },
  });
}

/**
 * Make volatility predictions
 */
export function useVolatilityPredict() {
  return useMutation({
    mutationFn: (request: VolatilityPredictionRequest) => volatilityApi.predict(request),
  });
}

/**
 * Backtest volatility targeting strategy
 */
export function useVolatilityBacktest() {
  return useMutation({
    mutationFn: (request: StrategyBacktestRequest) => volatilityApi.backtestStrategy(request),
  });
}

/**
 * Compare multiple volatility forecasting models
 */
export function useCompareVolatilityModels() {
  return useMutation({
    mutationFn: (request: ModelComparisonRequest) => volatilityApi.compareModels(request),
  });
}
