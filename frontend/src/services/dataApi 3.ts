/**
 * Data API service
 */

import api from './api';
import type { StockInfo, StockDataResponse } from '../types/api';

export interface StockListResponse {
  stocks: StockInfo[];
  total: number;
}

export interface FetchDataRequest {
  tickers: string[];
  start_date?: string;
  end_date?: string;
  interval?: string;
  force_refresh?: boolean;
}

export interface FetchDataResponse {
  success: boolean;
  tickers_fetched: string[];
  records_added: number;
  message: string;
  errors?: Record<string, string>;
}

export interface FetchLatestResponse {
  success: boolean;
  ticker: string;
  records_fetched: number;
  message: string;
  date_range: {
    start: string | null;
    end: string | null;
  } | null;
}

export interface FeatureData {
  timestamp: string;
  ticker: string;
  log_return?: number;
  simple_return?: number;
  rolling_volatility_5?: number;
  rolling_volatility_20?: number;
  rolling_volatility_60?: number;
  momentum_5?: number;
  momentum_10?: number;
  momentum_20?: number;
  momentum_60?: number;
  rsi_14?: number;
  macd?: number;
  macd_signal?: number;
  bollinger_upper?: number;
  bollinger_lower?: number;
  atr_14?: number;
}

export interface FeaturesResponse {
  ticker: string;
  features: FeatureData[];
  feature_names: string[];
  count: number;
}

export const dataApi = {
  // List available stocks
  async listStocks(): Promise<StockListResponse> {
    const response = await api.get<StockListResponse>('/api/data/stocks');
    return response.data;
  },

  // Get stock data
  async getStockData(
    ticker: string,
    startDate?: string,
    endDate?: string,
    interval: string = "1d"
  ): Promise<StockDataResponse> {
    const response = await api.get<StockDataResponse>(`/api/data/stocks/${ticker}`, {
      params: {
        start_date: startDate,
        end_date: endDate,
        interval,
      },
    });
    return response.data;
  },

  // Get stock features
  async getStockFeatures(
    ticker: string,
    startDate?: string,
    endDate?: string,
    interval: string = "1d"
  ): Promise<FeaturesResponse> {
    const response = await api.get<FeaturesResponse>(
      `/api/data/stocks/${ticker}/features`,
      {
        params: {
          start_date: startDate,
          end_date: endDate,
          interval,
        },
      }
    );
    return response.data;
  },

  // Fetch new data
  async fetchData(request: FetchDataRequest): Promise<FetchDataResponse> {
    const response = await api.post<FetchDataResponse>('/api/data/fetch', request);
    return response.data;
  },

  // Fetch latest data with incremental updates (10 years of daily data)
  async fetchLatest(ticker: string, years: number = 10): Promise<FetchLatestResponse> {
    const response = await api.post<FetchLatestResponse>(`/api/data/fetch-latest/${ticker}?years=${years}`);
    return response.data;
  },

  // Clear cache
  async clearCache(ticker?: string): Promise<{ message: string }> {
    const response = await api.delete('/api/data/cache', {
      params: { ticker },
    });
    return response.data;
  },
};

export default dataApi;
