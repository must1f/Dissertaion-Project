/**
 * React Query hooks for spectral analysis and regime detection
 */

import { useMutation } from '@tanstack/react-query';
import { spectralApi } from '../services/spectralApi';
import type {
    SpectralAnalysisRequest,
    RegimeDetectionRequest,
    FanChartRequest,
} from '../services/spectralApi';

/**
 * Run spectral analysis on a ticker
 */
export function useSpectralAnalysis() {
    return useMutation({
        mutationFn: (request: SpectralAnalysisRequest) => spectralApi.analyzeSpectral(request),
    });
}

/**
 * Detect market regimes
 */
export function useRegimeDetection() {
    return useMutation({
        mutationFn: (request: RegimeDetectionRequest) => spectralApi.detectRegimes(request),
    });
}

/**
 * Generate fan chart
 */
export function useFanChart() {
    return useMutation({
        mutationFn: (request: FanChartRequest) => spectralApi.generateFanChart(request),
    });
}
