import { afterEach, describe, expect, it, vi } from 'vitest';
import api from '../api';
import { metricsApi } from '../metricsApi';

const asResponse = <T>(data: T) => ({ data }) as const;

describe('metricsApi service', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('calculateFinancialMetrics joins returns and calls GET /api/metrics/financial', async () => {
        const payload = { sharpe_ratio: 1.5, total_return: 0.1 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.calculateFinancialMetrics([0.01, -0.02], 0.03, 252);

        expect(spy).toHaveBeenCalledWith('/api/metrics/financial', {
            params: { returns: '0.01,-0.02', risk_free_rate: 0.03, periods_per_year: 252 },
        });
        expect(result).toEqual(payload);
    });

    it('calculateFinancialMetricsPost posts to /api/metrics/financial', async () => {
        const request = { returns: [0.01, -0.02], risk_free_rate: 0.02, periods_per_year: 252 };
        const payload = { metrics: { sharpe_ratio: 1.5 }, input_summary: {} };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.calculateFinancialMetricsPost(request);

        expect(spy).toHaveBeenCalledWith('/api/metrics/financial', request);
        expect(result).toEqual(payload);
    });

    it('calculateMLMetrics joins arrays and calls GET /api/metrics/ml', async () => {
        const payload = { rmse: 0.05, mae: 0.03, mape: 2.5, r2: 0.85, directional_accuracy: 0.6 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.calculateMLMetrics([1.0, 2.0], [1.1, 2.1]);

        expect(spy).toHaveBeenCalledWith('/api/metrics/ml', {
            params: { y_true: '1,2', y_pred: '1.1,2.1' },
        });
        expect(result).toEqual(payload);
    });

    it('getPhysicsMetrics calls GET /api/metrics/physics/{key}', async () => {
        const payload = { total_physics_loss: 0.001 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.getPhysicsMetrics('pinn_gbm');

        expect(spy).toHaveBeenCalledWith('/api/metrics/physics/pinn_gbm');
        expect(result).toEqual(payload);
    });

    it('getModelMetrics calls GET /api/metrics/model/{key}', async () => {
        const payload = { model_key: 'lstm', model_name: 'LSTM', is_pinn: false, ml_metrics: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.getModelMetrics('lstm');

        expect(spy).toHaveBeenCalledWith('/api/metrics/model/lstm');
        expect(result).toEqual(payload);
    });

    it('compareMetrics joins keys and calls GET /api/metrics/comparison', async () => {
        const payload = { models: [], metric_summary: {}, best_by_metric: {}, rankings: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.compareMetrics(['lstm', 'gru']);

        expect(spy).toHaveBeenCalledWith('/api/metrics/comparison', {
            params: { model_keys: 'lstm,gru' },
        });
        expect(result).toEqual(payload);
    });

    it('getSavedMetrics calls GET /api/metrics/saved/{key}', async () => {
        const payload = { rmse: 0.05 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.getSavedMetrics('lstm');

        expect(spy).toHaveBeenCalledWith('/api/metrics/saved/lstm');
        expect(result).toEqual(payload);
    });

    it('getLeaderboard passes metric and topN params', async () => {
        const payload = { metric: 'rmse', entries: [], n_experiments: 0, generated_at: '' };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await metricsApi.getLeaderboard('rmse', 5);

        expect(spy).toHaveBeenCalledWith('/api/metrics/leaderboard', {
            params: { metric: 'rmse', top_n: 5 },
        });
        expect(result).toEqual(payload);
    });
});
