/**
 * React Query hooks for data operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { dataApi, type FetchDataRequest, type FetchLatestResponse } from '../services/dataApi';

// Query keys
export const dataKeys = {
  all: ['data'] as const,
  stocks: () => [...dataKeys.all, 'stocks'] as const,
  stock: (ticker: string) => [...dataKeys.all, 'stock', ticker] as const,
  stockData: (ticker: string, startDate?: string, endDate?: string, interval?: string) =>
    [...dataKeys.all, 'stockData', ticker, startDate, endDate, interval] as const,
  features: (ticker: string, startDate?: string, endDate?: string, interval?: string) =>
    [...dataKeys.all, 'features', ticker, startDate, endDate, interval] as const,
};

// List available stocks
export function useStocks() {
  return useQuery({
    queryKey: dataKeys.stocks(),
    queryFn: dataApi.listStocks,
    staleTime: 60000, // 1 minute
  });
}

// Get stock data
export function useStockData(
  ticker: string,
  startDate?: string,
  endDate?: string,
  interval: string = "1d"
) {
  return useQuery({
    queryKey: dataKeys.stockData(ticker, startDate, endDate, interval),
    queryFn: () => dataApi.getStockData(ticker, startDate, endDate, interval),
    enabled: !!ticker,
    staleTime: 30000, // 30 seconds
  });
}

// Get stock features
export function useStockFeatures(
  ticker: string,
  startDate?: string,
  endDate?: string,
  interval: string = "1d"
) {
  return useQuery({
    queryKey: dataKeys.features(ticker, startDate, endDate, interval),
    queryFn: () => dataApi.getStockFeatures(ticker, startDate, endDate, interval),
    enabled: !!ticker,
    staleTime: 30000,
  });
}

// Fetch new data mutation
export function useFetchData() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: FetchDataRequest) => dataApi.fetchData(request),
    onSuccess: (_, variables) => {
      // Invalidate relevant queries
      variables.tickers.forEach((ticker) => {
        queryClient.invalidateQueries({ queryKey: dataKeys.stock(ticker) });
      });
      queryClient.invalidateQueries({ queryKey: dataKeys.stocks() });
    },
  });
}

// Fetch latest data with incremental updates (10 years)
export function useFetchLatest() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ ticker, years = 10 }: { ticker: string; years?: number }) =>
      dataApi.fetchLatest(ticker, years),
    onSuccess: (data) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: dataKeys.stock(data.ticker) });
      queryClient.invalidateQueries({ queryKey: dataKeys.stocks() });
      // Clear all stockData queries for this ticker
      queryClient.invalidateQueries({
        queryKey: dataKeys.all,
        predicate: (query) =>
          query.queryKey.includes('stockData') && query.queryKey.includes(data.ticker),
      });
    },
  });
}

// Clear cache mutation
export function useClearCache() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (ticker?: string) => dataApi.clearCache(ticker),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dataKeys.all });
    },
  });
}
