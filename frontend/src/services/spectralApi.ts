/**
 * Spectral Analysis API service
 */

import api from './api';

// ── Types ────────────────────────────────────────────────────────────────────

export interface SpectralAnalysisRequest {
    ticker: string;
    window_size?: number;
}

export interface SpectralSnapshot {
    date: string;
    spectral_entropy: number;
    dominant_frequency: number;
    dominant_period: number;
    power_low: number;
    power_mid: number;
    power_high: number;
    power_ratio: number;
    autocorrelation_lag1: number;
    spectral_slope: number;
}

export interface PowerSpectrumData {
    frequencies: number[];
    power: number[];
    frequency_bands: Record<string, { min_freq: number; max_freq: number; total_power: number }>;
}

export interface SpectralAnalysisResponse {
    success: boolean;
    ticker: string;
    analysis_date: string;
    window_size: number;
    current_features: SpectralSnapshot;
    power_spectrum: PowerSpectrumData;
    processing_time_ms: number;
}

export interface RegimeDetectionRequest {
    ticker: string;
    method?: string;
    n_regimes?: number;
    lookback_days?: number;
}

export interface RegimeCharacteristics {
    regime_id: number;
    regime_name: string;
    mean_return_annual: number;
    volatility_annual: number;
    stationary_probability: number;
    expected_duration_days: number;
    spectral_entropy: number | null;
    dominant_frequency: number | null;
    power_ratio: number | null;
    sample_count: number;
}

export interface TransitionMatrix {
    matrix: number[][];
    labels: string[];
}

export interface RegimeHistoryPoint {
    date: string;
    regime: number;
    regime_name: string;
    probability: number;
    all_probabilities: Record<string, number>;
}

export interface RegimeDetectionResponse {
    success: boolean;
    ticker: string;
    method: string;
    n_regimes: number;
    current_regime: number;
    current_regime_name: string;
    current_probability: number;
    regime_characteristics: RegimeCharacteristics[];
    transition_matrix: TransitionMatrix;
    recent_history: RegimeHistoryPoint[];
    processing_time_ms: number;
}

export interface FanChartRequest {
    ticker: string;
    initial_price?: number;
    horizon_days?: number;
    n_simulations?: number;
    percentiles?: number[];
    use_regime_switching?: boolean;
}

export interface PercentileBand {
    percentile: number;
    values: number[];
}

export interface RegimePeriod {
    start: number;
    end: number;
    regime: number;
    regime_name: string;
}

export interface FanChartResponse {
    success: boolean;
    ticker: string;
    initial_price: number;
    horizon_days: number;
    n_simulations: number;
    dates: number[];
    percentile_bands: PercentileBand[];
    median_path: number[];
    regime_periods: RegimePeriod[];
    regime_probabilities: Record<string, number>[];
    expected_return: number;
    value_at_risk_95: number;
    probability_of_loss: number;
    processing_time_ms: number;
}

// ── API Functions ────────────────────────────────────────────────────────────

export const spectralApi = {
    /** Run spectral analysis on a ticker */
    async analyzeSpectral(request: SpectralAnalysisRequest): Promise<SpectralAnalysisResponse> {
        const response = await api.post<SpectralAnalysisResponse>('/api/spectral/analyze', request);
        return response.data;
    },

    /** Detect market regimes using HMM or spectral-enhanced HMM */
    async detectRegimes(request: RegimeDetectionRequest): Promise<RegimeDetectionResponse> {
        const response = await api.post<RegimeDetectionResponse>('/api/spectral/regimes/detect', request);
        return response.data;
    },

    /** Generate regime-aware Monte Carlo fan chart */
    async generateFanChart(request: FanChartRequest): Promise<FanChartResponse> {
        const response = await api.post<FanChartResponse>('/api/spectral/fan-chart', request);
        return response.data;
    },
};

export default spectralApi;
