import { afterEach, describe, expect, it, vi } from 'vitest';
import api from '../api';
import { dataApi } from '../dataApi';

// Helper to build Axios-like responses without importing types
const asResponse = <T>(data: T) => ({ data } as const);

describe('dataApi service', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
    localStorage.clear();
  });

  it('lists stocks through the /api/data/stocks endpoint', async () => {
    const payload = { stocks: [{ ticker: '^GSPC', record_count: 10 }], total: 1 };
    const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

    const result = await dataApi.listStocks();

    expect(spy).toHaveBeenCalledWith('/api/data/stocks');
    expect(result).toEqual(payload);
  });

  it('passes date params when requesting stock data', async () => {
    const response = { ticker: '^GSPC', data: [], start_date: null, end_date: null, count: 0 };
    const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(response));

    await dataApi.getStockData('^GSPC', '2024-01-01', '2024-02-01');

    expect(spy).toHaveBeenCalledWith('/api/data/stocks/^GSPC', {
      params: { start_date: '2024-01-01', end_date: '2024-02-01', interval: '1d' },
    });
  });

  it('posts fetchData requests with provided payload', async () => {
    const request = { tickers: ['^GSPC'], start_date: '2024-01-01', force_refresh: true };
    const response = { success: true, tickers_fetched: ['^GSPC'], records_added: 5, message: 'ok' };
    const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(response));

    const result = await dataApi.fetchData(request);

    expect(spy).toHaveBeenCalledWith('/api/data/fetch', request);
    expect(result).toEqual(response);
  });

  it('clears cache with optional ticker parameter', async () => {
    const response = { message: 'Cache cleared for ^GSPC' };
    const spy = vi.spyOn(api, 'delete').mockResolvedValue(asResponse(response));

    const result = await dataApi.clearCache('^GSPC');

    expect(spy).toHaveBeenCalledWith('/api/data/cache', { params: { ticker: '^GSPC' } });
    expect(result).toEqual(response);
  });
});
