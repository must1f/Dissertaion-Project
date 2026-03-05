import { afterEach, describe, expect, it, vi } from 'vitest';
import api from '../api';
import { trainingApi } from '../trainingApi';

const asResponse = <T>(data: T) => ({ data }) as const;

describe('trainingApi service', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('getTrainingMode calls GET /api/training/mode', async () => {
        const payload = { mode: 'real', using_real_models: true, import_error: null, message: 'ok' };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.getTrainingMode();

        expect(spy).toHaveBeenCalledWith('/api/training/mode');
        expect(result).toEqual(payload);
    });

    it('getAvailableModels calls GET /api/training/batch/models', async () => {
        const payload = { models: {}, total: 0, by_type: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.getAvailableModels();

        expect(spy).toHaveBeenCalledWith('/api/training/batch/models');
        expect(result).toEqual(payload);
    });

    it('startBatchTraining posts to /api/training/batch/start', async () => {
        const request = {
            models: [{ model_key: 'lstm', enabled: true }],
            ticker: '^GSPC',
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.001,
            sequence_length: 60,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.1,
            gradient_clip_norm: 1.0,
            scheduler_patience: 5,
            early_stopping_patience: 10,
            research_mode: false,
            force_refresh: false,
            enable_physics: false,
        };
        const payload = { success: true, batch_id: 'batch-1', message: 'ok', total_models: 1, model_keys: ['lstm'], websocket_url: '/ws' };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.startBatchTraining(request);

        expect(spy).toHaveBeenCalledWith('/api/training/batch/start', request);
        expect(result).toEqual(payload);
    });

    it('stopBatchTraining posts to /api/training/batch/stop/{id}', async () => {
        const payload = { success: true, message: 'Stopped' };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.stopBatchTraining('batch-1');

        expect(spy).toHaveBeenCalledWith('/api/training/batch/stop/batch-1');
        expect(result).toEqual(payload);
    });

    it('getBatchStatus extracts .batch from response', async () => {
        const batchData = { batch_id: 'batch-1', status: 'running', total_models: 1, completed_models: 0, failed_models: 0, current_model: 'lstm', models: [], overall_progress: 0.5, config: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse({ batch: batchData }));

        const result = await trainingApi.getBatchStatus('batch-1');

        expect(spy).toHaveBeenCalledWith('/api/training/batch/status/batch-1');
        expect(result).toEqual(batchData);
    });

    it('listBatchJobs passes params to /api/training/batch/list', async () => {
        const payload = { batches: [], total: 0, page: 1, page_size: 10 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.listBatchJobs({ page: 1, page_size: 10 });

        expect(spy).toHaveBeenCalledWith('/api/training/batch/list', { params: { page: 1, page_size: 10 } });
        expect(result).toEqual(payload);
    });

    it('startTraining posts config to /api/training/start', async () => {
        const config = {
            model_type: 'lstm',
            ticker: '^GSPC',
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.001,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.1,
        };
        const payload = { success: true, job_id: 'job-1', message: 'Started', websocket_url: '/ws' };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.startTraining(config);

        expect(spy).toHaveBeenCalledWith('/api/training/start', config);
        expect(result).toEqual(payload);
    });

    it('stopTraining posts to /api/training/stop/{id}', async () => {
        const payload = { success: true, message: 'Stopped' };
        const spy = vi.spyOn(api, 'post').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.stopTraining('job-1');

        expect(spy).toHaveBeenCalledWith('/api/training/stop/job-1');
        expect(result).toEqual(payload);
    });

    it('getTrainingStatus calls GET /api/training/status/{id}', async () => {
        const payload = { job: { job_id: 'job-1', status: 'running' }, history: null };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.getTrainingStatus('job-1');

        expect(spy).toHaveBeenCalledWith('/api/training/status/job-1');
        expect(result).toEqual(payload);
    });

    it('getTrainingHistory calls GET /api/training/history/{id}', async () => {
        const payload = { job_id: 'job-1', model_type: 'lstm', epochs: [], best_epoch: 1, best_val_loss: 0.01, final_metrics: {} };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.getTrainingHistory('job-1');

        expect(spy).toHaveBeenCalledWith('/api/training/history/job-1');
        expect(result).toEqual(payload);
    });

    it('listTrainingRuns passes params to GET /api/training/history', async () => {
        const payload = { runs: [], total: 0, page: 1, page_size: 20 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.listTrainingRuns({ status: 'completed', page: 1, page_size: 20 });

        expect(spy).toHaveBeenCalledWith('/api/training/history', {
            params: { status: 'completed', page: 1, page_size: 20 },
        });
        expect(result).toEqual(payload);
    });

    it('getActiveJobs calls GET /api/training/active', async () => {
        const payload = { active_jobs: [], count: 0 };
        const spy = vi.spyOn(api, 'get').mockResolvedValue(asResponse(payload));

        const result = await trainingApi.getActiveJobs();

        expect(spy).toHaveBeenCalledWith('/api/training/active');
        expect(result).toEqual(payload);
    });
});
