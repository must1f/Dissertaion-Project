/**
 * Training state store using Zustand
 */

import { create } from 'zustand';

interface TrainingJob {
  jobId: string;
  modelType: string;
  ticker: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  currentEpoch: number;
  totalEpochs: number;
  progressPercent: number;
  trainLoss: number | null;
  valLoss: number | null;
  bestValLoss: number | null;
  startedAt: Date | null;
  history: {
    trainLoss: number[];
    valLoss: number[];
    learningRate: number[];
  };
}

interface TrainingState {
  // Current jobs
  jobs: Record<string, TrainingJob>;

  // Actions
  addJob: (job: TrainingJob) => void;
  updateJob: (jobId: string, updates: Partial<TrainingJob>) => void;
  removeJob: (jobId: string) => void;

  // Update from WebSocket
  handleWsUpdate: (data: {
    job_id: string;
    epoch?: number;
    total_epochs?: number;
    train_loss?: number;
    val_loss?: number;
    best_val_loss?: number;
    progress_percent?: number;
    status?: string;
  }) => void;

  // Getters
  getActiveJobs: () => TrainingJob[];
  getJob: (jobId: string) => TrainingJob | undefined;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  jobs: {},

  addJob: (job) =>
    set((state) => ({
      jobs: { ...state.jobs, [job.jobId]: job },
    })),

  updateJob: (jobId, updates) =>
    set((state) => {
      const job = state.jobs[jobId];
      if (!job) return state;

      return {
        jobs: {
          ...state.jobs,
          [jobId]: { ...job, ...updates },
        },
      };
    }),

  removeJob: (jobId) =>
    set((state) => {
      const { [jobId]: _, ...rest } = state.jobs;
      return { jobs: rest };
    }),

  handleWsUpdate: (data) =>
    set((state) => {
      const job = state.jobs[data.job_id];
      if (!job) return state;

      const updates: Partial<TrainingJob> = {};

      if (data.epoch !== undefined) {
        updates.currentEpoch = data.epoch;
      }
      if (data.total_epochs !== undefined) {
        updates.totalEpochs = data.total_epochs;
      }
      if (data.train_loss !== undefined) {
        updates.trainLoss = data.train_loss;
        updates.history = {
          ...job.history,
          trainLoss: [...job.history.trainLoss, data.train_loss],
        };
      }
      if (data.val_loss !== undefined) {
        updates.valLoss = data.val_loss;
        updates.history = {
          ...job.history,
          valLoss: [...(updates.history?.valLoss || job.history.valLoss), data.val_loss],
        };
      }
      if (data.best_val_loss !== undefined) {
        updates.bestValLoss = data.best_val_loss;
      }
      if (data.progress_percent !== undefined) {
        updates.progressPercent = data.progress_percent;
      }
      if (data.status !== undefined) {
        updates.status = data.status as TrainingJob['status'];
      }

      return {
        jobs: {
          ...state.jobs,
          [data.job_id]: { ...job, ...updates },
        },
      };
    }),

  getActiveJobs: () => {
    const state = get();
    return Object.values(state.jobs).filter(
      (job) => job.status === 'pending' || job.status === 'running'
    );
  },

  getJob: (jobId) => get().jobs[jobId],
}));
