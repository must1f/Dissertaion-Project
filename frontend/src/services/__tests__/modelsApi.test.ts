import { afterEach, describe, expect, it, vi } from 'vitest';
import api from '../api';
import { modelsApi } from '../modelsApi';

const asResponse = <T>(data: T) => ({ data }) as const;

describe('modelsApi service', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('listModels calls GET /api/models/', async () => {
        const payload = { models: [], total: 0, trained_count: 0, pinn_count: 0 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.listModels();

        expect(spy).toHaveBeenCalledWith('/api/models/');
        expect(result).toEqual(payload);
    });

    it('listTrainedModels calls GET /api/models/trained', async () => {
        const payload = { models: [], total: 0, trained_count: 0, pinn_count: 0 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.listTrainedModels();

        expect(spy).toHaveBeenCalledWith('/api/models/trained');
        expect(result).toEqual(payload);
    });

    it('getModelTypes calls GET /api/models/types', async () => {
        const payload = { model_types: ['lstm'], categories: { baseline: ['lstm'] } };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.getModelTypes();

        expect(spy).toHaveBeenCalledWith('/api/models/types');
        expect(result).toEqual(payload);
    });

    it('getModel calls GET /api/models/{key}', async () => {
        const payload = { model_key: 'lstm', model_type: 'lstm', display_name: 'LSTM' };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.getModel('lstm');

        expect(spy).toHaveBeenCalledWith('/api/models/lstm');
        expect(result).toEqual(payload);
    });

    it('getModelWeights calls GET /api/models/{key}/weights', async () => {
        const payload = { model_key: 'lstm', total_parameters: 100000, trainable_parameters: 100000, layer_info: [] };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.getModelWeights('lstm');

        expect(spy).toHaveBeenCalledWith('/api/models/lstm/weights');
        expect(result).toEqual(payload);
    });

    it('compareModels joins keys and calls GET /api/models/compare', async () => {
        const payload = { models: [], metric_names: [], best_by_metric: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.compareModels(['lstm', 'gru']);

        expect(spy).toHaveBeenCalledWith('/api/models/compare', {
            params: { model_keys: 'lstm,gru' },
        });
        expect(result).toEqual(payload);
    });

    it('loadModel calls POST /api/models/{key}/load with device param', async () => {
        const payload = { message: 'Model loaded' };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.loadModel('lstm', 'cpu');

        expect(spy).toHaveBeenCalledWith('/api/models/lstm/load', null, {
            params: { device: 'cpu' },
        });
        expect(result).toEqual(payload);
    });

    it('unloadModel calls POST /api/models/{key}/unload', async () => {
        const payload = { message: 'Model unloaded' };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await modelsApi.unloadModel('lstm');

        expect(spy).toHaveBeenCalledWith('/api/models/lstm/unload');
        expect(result).toEqual(payload);
    });
});
