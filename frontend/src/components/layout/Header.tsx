import { useAppStore } from "../../stores/appStore"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Badge } from "../ui/badge"
import { DEFAULT_TICKER, getTickerInfo } from "../../config/tickers"
import { TrainingModeIndicator } from "../common/TrainingModeIndicator"
import {
  Sun,
  Moon,
  Search,
  Bell,
  RefreshCw,
  TrendingUp,
} from "lucide-react"

interface HeaderProps {
  title?: string
}

export function Header({ title }: HeaderProps) {
  const { theme, setTheme } = useAppStore()
  const tickerInfo = getTickerInfo(DEFAULT_TICKER)

  const toggleTheme = () => {
    if (theme === "light") {
      setTheme("dark")
      document.documentElement.classList.add("dark")
    } else {
      setTheme("light")
      document.documentElement.classList.remove("dark")
    }
  }

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border/50 px-6
      bg-background/70 backdrop-blur-xl
      supports-[backdrop-filter]:bg-background/50
      dark:bg-[hsl(222,47%,6%)/0.7]"
    >
      {/* Left side */}
      <div className="flex items-center gap-4">
        {title && <h1 className="text-xl font-semibold tracking-tight">{title}</h1>}
      </div>

      {/* Center - Search and data source */}
      <div className="flex items-center gap-4">
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary" />
          <Input
            type="text"
            placeholder="Search..."
            className="w-64 pl-9 bg-muted/50 border-border/50 focus:bg-muted/80 transition-all"
          />
        </div>

        {/* S&P 500 Data Source Badge (read-only) */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">Data:</span>
          <Badge variant="default" className="gap-1 font-mono text-xs" title={tickerInfo?.name}>
            <TrendingUp className="h-3 w-3" />
            {DEFAULT_TICKER}
          </Badge>
          <span className="text-xs text-muted-foreground/70 hidden lg:inline">{tickerInfo?.name}</span>
        </div>

        {/* Training Mode Indicator */}
        <div className="flex items-center gap-2 border-l border-border/50 pl-4 ml-2">
          <span className="text-xs font-medium text-muted-foreground">Training:</span>
          <TrainingModeIndicator />
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="icon" title="Refresh data" className="h-9 w-9 rounded-lg hover:bg-accent/80">
          <RefreshCw className="h-4 w-4 transition-transform hover:rotate-90 duration-500" />
        </Button>

        <Button variant="ghost" size="icon" title="Notifications" className="h-9 w-9 rounded-lg hover:bg-accent/80">
          <Bell className="h-4 w-4" />
        </Button>

        <Button variant="ghost" size="icon" onClick={toggleTheme} title="Toggle theme" className="h-9 w-9 rounded-lg hover:bg-accent/80">
          {theme === "dark" ? (
            <Sun className="h-4 w-4 transition-transform hover:rotate-45 duration-300" />
          ) : (
            <Moon className="h-4 w-4 transition-transform hover:-rotate-12 duration-300" />
          )}
        </Button>
      </div>
    </header>
  )
}
