/**
 * API client configuration
 */

import axios, { AxiosError, AxiosInstance } from 'axios';

// Use empty string to leverage Vite's dev server proxy for /api routes
// This avoids CORS issues during development
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Create axios instance
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor (auth can be added here when needed)
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.message);
    } else {
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// WebSocket connection helper
export function createWebSocket(path: string): WebSocket {
  // If API_BASE_URL is set, use it; otherwise construct from window.location
  let wsUrl: string;
  if (API_BASE_URL) {
    wsUrl = API_BASE_URL.replace('http', 'ws') + path;
  } else {
    // Use current host with ws/wss protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    wsUrl = `${protocol}//${window.location.host}${path}`;
  }
  return new WebSocket(wsUrl);
}

export default api;
