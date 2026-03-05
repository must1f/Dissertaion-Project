/**
 * Ticker display component
 * Shows S&P 500 as the only supported ticker (no selection allowed)
 */

import { DEFAULT_TICKER, getTickerInfo } from "../../config/tickers"
import { cn } from "../../lib/utils"

interface TickerSelectProps {
  value?: string;
  onChange?: (ticker: string) => void;
  className?: string;
  showLabel?: boolean;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * Displays the S&P 500 ticker (read-only)
 * All models are trained exclusively on S&P 500 data
 */
export function TickerSelect({
  className,
  showLabel = false,
  label = "Data Source",
  size = 'md',
}: TickerSelectProps) {
  const ticker = getTickerInfo(DEFAULT_TICKER);

  const sizeClasses = {
    sm: 'h-8 text-xs px-2',
    md: 'h-10 text-sm px-3',
    lg: 'h-12 text-base px-4',
  };

  return (
    <div className={cn("flex flex-col", className)}>
      {showLabel && (
        <label className="mb-2 block text-sm font-medium">{label}</label>
      )}
      <div
        className={cn(
          "w-full rounded-md border border-input bg-muted py-2 font-mono",
          "flex items-center justify-between",
          sizeClasses[size]
        )}
      >
        <span className="font-semibold">{ticker?.symbol}</span>
        <span className="text-muted-foreground ml-2">{ticker?.name}</span>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        Models are trained exclusively on S&P 500 data
      </p>
    </div>
  );
}

/**
 * Compact version for header - displays S&P 500 badge
 */
export function TickerSelectCompact({
  className,
}: Omit<TickerSelectProps, 'showLabel' | 'label' | 'size'>) {
  return (
    <div
      className={cn(
        "h-9 rounded-md border border-input bg-muted px-3 py-1 text-sm font-mono",
        "flex items-center gap-2",
        className
      )}
    >
      <span className="font-semibold">^GSPC</span>
      <span className="text-muted-foreground">S&P 500</span>
    </div>
  );
}
