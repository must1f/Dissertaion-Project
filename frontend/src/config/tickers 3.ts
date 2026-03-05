/**
 * Ticker configuration
 * Restricted to S&P 500 data only for model training and predictions
 */

export interface TickerOption {
  symbol: string;
  name: string;
  category: 'index' | 'stock';
}

// Only S&P 500 is supported - models are trained exclusively on S&P 500 data
export const AVAILABLE_TICKERS: TickerOption[] = [
  { symbol: '^GSPC', name: 'S&P 500 Index', category: 'index' },
];

// Default and only ticker - S&P 500 Index
export const DEFAULT_TICKER = '^GSPC';

// Allowed tickers for validation
export const ALLOWED_TICKERS = ['^GSPC'];

// Get ticker by symbol
export const getTickerInfo = (symbol: string): TickerOption | undefined => {
  return AVAILABLE_TICKERS.find(t => t.symbol === symbol);
};

// Get display name for a ticker
export const getTickerDisplayName = (symbol: string): string => {
  const ticker = getTickerInfo(symbol);
  return ticker ? `${ticker.symbol} - ${ticker.name}` : symbol;
};

// Group tickers by category (kept for backwards compatibility)
export const getTickersByCategory = () => {
  const indices = AVAILABLE_TICKERS.filter(t => t.category === 'index');
  const stocks = AVAILABLE_TICKERS.filter(t => t.category === 'stock');
  return { indices, stocks };
};

// Validate if a ticker is allowed
export const isTickerAllowed = (symbol: string): boolean => {
  return ALLOWED_TICKERS.includes(symbol);
};
