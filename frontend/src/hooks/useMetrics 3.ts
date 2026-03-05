/**
 * React Query hooks for metrics
 */

import { useQuery, useMutation } from '@tanstack/react-query';
import { metricsApi, type FinancialMetricsRequest } from '../services/metricsApi';

// Query keys
export const metricsKeys = {
  all: ['metrics'] as const,
  physics: (modelKey: string) => [...metricsKeys.all, 'physics', modelKey] as const,
  model: (modelKey: string) => [...metricsKeys.all, 'model', modelKey] as const,
  comparison: (modelKeys: string[]) => [...metricsKeys.all, 'comparison', modelKeys] as const,
  saved: (modelKey: string) => [...metricsKeys.all, 'saved', modelKey] as const,
};

// Get physics metrics for a model
export function usePhysicsMetrics(modelKey: string) {
  return useQuery({
    queryKey: metricsKeys.physics(modelKey),
    queryFn: () => metricsApi.getPhysicsMetrics(modelKey),
    enabled: !!modelKey,
  });
}

// Get all metrics for a model
export function useModelMetrics(modelKey: string) {
  return useQuery({
    queryKey: metricsKeys.model(modelKey),
    queryFn: () => metricsApi.getModelMetrics(modelKey),
    enabled: !!modelKey,
  });
}

// Compare metrics across models
export function useMetricsComparison(modelKeys: string[]) {
  return useQuery({
    queryKey: metricsKeys.comparison(modelKeys),
    queryFn: () => metricsApi.compareMetrics(modelKeys),
    enabled: modelKeys.length > 0,
  });
}

// Get saved metrics
export function useSavedMetrics(modelKey: string) {
  return useQuery({
    queryKey: metricsKeys.saved(modelKey),
    queryFn: () => metricsApi.getSavedMetrics(modelKey),
    enabled: !!modelKey,
  });
}

// Calculate financial metrics mutation
export function useCalculateFinancialMetrics() {
  return useMutation({
    mutationFn: (request: FinancialMetricsRequest) =>
      metricsApi.calculateFinancialMetricsPost(request),
  });
}

// Calculate ML metrics mutation
export function useCalculateMLMetrics() {
  return useMutation({
    mutationFn: ({ yTrue, yPred }: { yTrue: number[]; yPred: number[] }) =>
      metricsApi.calculateMLMetrics(yTrue, yPred),
  });
}
