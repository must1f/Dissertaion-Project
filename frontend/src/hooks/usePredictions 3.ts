/**
 * React Query hooks for predictions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionsApi } from '../services/predictionsApi';
import type { PredictionRequest } from '../types/predictions';

// Query keys
export const predictionKeys = {
  all: ['predictions'] as const,
  history: (ticker?: string, model?: string) =>
    [...predictionKeys.all, 'history', { ticker, model }] as const,
  latest: (ticker: string) => [...predictionKeys.all, 'latest', ticker] as const,
};

// Get prediction history
export function usePredictionHistory(
  ticker?: string,
  modelKey?: string,
  page = 1,
  pageSize = 50
) {
  return useQuery({
    queryKey: [...predictionKeys.history(ticker, modelKey), { page, pageSize }],
    queryFn: () => predictionsApi.getHistory(ticker, modelKey, page, pageSize),
  });
}

// Get latest predictions for a ticker
export function useLatestPredictions(ticker: string) {
  return useQuery({
    queryKey: predictionKeys.latest(ticker),
    queryFn: () => predictionsApi.getLatest(ticker),
    enabled: !!ticker,
    refetchInterval: 60000, // Refetch every minute
  });
}

// Predict mutation
export function usePredict() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: PredictionRequest) => predictionsApi.predict(request),
    onSuccess: (data) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({
        queryKey: predictionKeys.latest(data.prediction.ticker),
      });
      queryClient.invalidateQueries({
        queryKey: predictionKeys.history(data.prediction.ticker),
      });
    },
  });
}

// Batch predict mutation
export function useBatchPredict() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: predictionsApi.batchPredict,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.all });
    },
  });
}

// Clear cache mutation
export function useClearPredictionCache() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelKey?: string) => predictionsApi.clearCache(modelKey),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.all });
    },
  });
}
