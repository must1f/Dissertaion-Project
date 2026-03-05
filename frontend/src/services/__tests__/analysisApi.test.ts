import { afterEach, describe, expect, it, vi } from 'vitest';
import api from '../api';
import { analysisApi } from '../analysisApi';

const asResponse = <T>(data: T) => ({ data }) as const;

describe('analysisApi service', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('getRegimeHistory passes ticker and method params', async () => {
        const payload = {
            ticker: '^GSPC',
            start_date: '2024-01-01',
            end_date: '2024-06-01',
            total_points: 100,
            history: [],
            regime_summary: {},
        };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await analysisApi.getRegimeHistory('^GSPC', 'rolling');

        expect(spy).toHaveBeenCalledWith('/api/analysis/regime/history', {
            params: { ticker: '^GSPC', method: 'rolling' },
        });
        expect(result).toEqual(payload);
    });

    it('getRegimeHistory uses default params', async () => {
        const payload = { ticker: '^GSPC', history: [], total_points: 0, regime_summary: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        await analysisApi.getRegimeHistory();

        expect(spy).toHaveBeenCalledWith('/api/analysis/regime/history', {
            params: { ticker: '^GSPC', method: 'rolling' },
        });
    });

    it('runStressTest posts payload to /api/analysis/stress/run', async () => {
        const request = { returns: [0.01, -0.02], timestamps: ['2024-01-01', '2024-01-02'] };
        const payload = {
            crises_analyzed: 5,
            crises_outperformed: 3,
            avg_crisis_return: -0.05,
            avg_max_drawdown: -0.10,
            worst_crisis: 'GFC',
            best_crisis: 'COVID Recovery',
            crisis_results: [],
        };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await analysisApi.runStressTest(request);

        expect(spy).toHaveBeenCalledWith('/api/analysis/stress/run', request);
        expect(result).toEqual(payload);
    });
});
