/**
 * Main application store using Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_TICKER } from '../config/tickers';

interface AppState {
  // Theme
  theme: 'light' | 'dark' | 'system';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;

  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Selected ticker - fixed to S&P 500, cannot be changed
  selectedTicker: string;
  setSelectedTicker: (ticker: string) => void;

  // Selected model
  selectedModel: string | null;
  setSelectedModel: (model: string | null) => void;

  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Theme
      theme: 'dark',
      setTheme: (theme) => set({ theme }),

      // Sidebar
      sidebarOpen: true,
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      // Selected ticker - fixed to S&P 500 Index (^GSPC)
      // Models are trained exclusively on S&P 500 data
      selectedTicker: DEFAULT_TICKER,
      // setSelectedTicker is kept for backwards compatibility but always sets to DEFAULT_TICKER
      setSelectedTicker: () => set({ selectedTicker: DEFAULT_TICKER }),

      // Selected model
      selectedModel: null,
      setSelectedModel: (model) => set({ selectedModel: model }),

      // Notifications
      notifications: [],
      addNotification: (notification) =>
        set((state) => ({
          notifications: [
            ...state.notifications,
            {
              ...notification,
              id: crypto.randomUUID(),
              timestamp: new Date(),
            },
          ],
        })),
      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),
      clearNotifications: () => set({ notifications: [] }),
    }),
    {
      name: 'pinn-app-storage',
      partialize: (state) => ({
        theme: state.theme,
        sidebarOpen: state.sidebarOpen,
        // Always persist the default ticker
        selectedTicker: DEFAULT_TICKER,
      }),
    }
  )
);
