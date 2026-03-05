import { afterEach, describe, expect, it, vi } from 'vitest';
import api from '../api';
import { predictionsApi } from '../predictionsApi';

const asResponse = <T>(data: T) => ({ data }) as const;

describe('predictionsApi service', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('predict posts to /api/predictions/predict', async () => {
        const request = { ticker: '^GSPC', model_key: 'lstm' };
        const payload = { success: true, prediction: {}, model_info: {}, processing_time_ms: 10 };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await predictionsApi.predict(request as any);

        expect(spy).toHaveBeenCalledWith('/api/predictions/predict', request);
        expect(result).toEqual(payload);
    });

    it('batchPredict posts to /api/predictions/predict/batch', async () => {
        const request = { tickers: ['^GSPC'], model_key: 'lstm' };
        const payload = { success: true, predictions: [], failed_tickers: [], total_processing_time_ms: 50 };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await predictionsApi.batchPredict(request);

        expect(spy).toHaveBeenCalledWith('/api/predictions/predict/batch', request);
        expect(result).toEqual(payload);
    });

    it('getHistory passes all params to /api/predictions/history', async () => {
        const payload = { predictions: [], total: 0, page: 2, page_size: 10 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await predictionsApi.getHistory('^GSPC', 'lstm', 2, 10);

        expect(spy).toHaveBeenCalledWith('/api/predictions/history', {
            params: { ticker: '^GSPC', model_key: 'lstm', page: 2, page_size: 10 },
        });
        expect(result).toEqual(payload);
    });

    it('getHistory uses default pagination', async () => {
        const payload = { predictions: [], total: 0, page: 1, page_size: 50 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        await predictionsApi.getHistory();

        expect(spy).toHaveBeenCalledWith('/api/predictions/history', {
            params: { ticker: undefined, model_key: undefined, page: 1, page_size: 50 },
        });
    });

    it('getLatest calls GET /api/predictions/{ticker}/latest', async () => {
        const payload = { ticker: '^GSPC', predictions: {}, consensus_signal: null, last_updated: '' };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await predictionsApi.getLatest('^GSPC');

        expect(spy).toHaveBeenCalledWith('/api/predictions/^GSPC/latest');
        expect(result).toEqual(payload);
    });

    it('clearCache calls DELETE /api/predictions/cache with model key', async () => {
        const payload = { message: 'Cache cleared for lstm' };
        const spy = vi.spyOn(api, 'delete').mockResolvedValue(asResponse(payload));

        const result = await predictionsApi.clearCache('lstm');

        expect(spy).toHaveBeenCalledWith('/api/predictions/cache', {
            params: { model_key: 'lstm' },
        });
        expect(result).toEqual(payload);
    });
});
