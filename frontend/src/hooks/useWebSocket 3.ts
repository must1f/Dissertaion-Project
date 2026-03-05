/**
 * WebSocket hook for real-time updates
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: unknown) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: unknown | null;
  sendMessage: (data: unknown) => void;
  disconnect: () => void;
  reconnect: () => void;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnect: shouldReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<unknown | null>(null);

  const connect = useCallback(() => {
    // Don't connect if URL is empty or invalid
    if (!url) {
      return;
    }

    // Don't connect if already connected
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Create WebSocket connection
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      onOpen?.();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        onMessage?.(data);
      } catch {
        // Handle non-JSON messages
        setLastMessage(event.data);
        onMessage?.(event.data);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      onClose?.();

      // Attempt to reconnect
      if (
        shouldReconnect &&
        reconnectAttemptsRef.current < maxReconnectAttempts
      ) {
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttemptsRef.current += 1;
          connect();
        }, reconnectInterval);
      }
    };

    ws.onerror = (error) => {
      onError?.(error);
    };

    wsRef.current = ws;
  }, [
    url,
    onMessage,
    onOpen,
    onClose,
    onError,
    shouldReconnect,
    reconnectInterval,
    maxReconnectAttempts,
  ]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    reconnectAttemptsRef.current = maxReconnectAttempts; // Prevent auto-reconnect
    wsRef.current?.close();
  }, [maxReconnectAttempts]);

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected');
    }
  }, []);

  const reconnectManual = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect]);

  // Connect on mount
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    disconnect,
    reconnect: reconnectManual,
  };
}

/**
 * Specialized hook for training WebSocket
 */
export function useTrainingWebSocket(jobId: string | null) {
  const [updates, setUpdates] = useState<unknown[]>([]);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);

  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const wsUrl = jobId && !isTrainingComplete
    ? `${baseUrl.replace('http', 'ws')}/api/ws/training/${jobId}`
    : '';

  const ws = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      setUpdates((prev) => [...prev, data]);

      // Stop reconnecting when training is complete
      const update = data as { type?: string; status?: string };
      if (
        update.type === 'training_complete' ||
        update.status === 'completed' ||
        update.status === 'failed' ||
        update.status === 'stopped'
      ) {
        setIsTrainingComplete(true);
      }
    },
    reconnect: true,
    maxReconnectAttempts: 3,
  });

  // Reset completion state when jobId changes
  useEffect(() => {
    if (jobId) {
      setIsTrainingComplete(false);
    }
  }, [jobId]);

  return {
    ...ws,
    updates,
    clearUpdates: () => {
      setUpdates([]);
      setIsTrainingComplete(false);
    },
    isTrainingComplete,
  };
}
