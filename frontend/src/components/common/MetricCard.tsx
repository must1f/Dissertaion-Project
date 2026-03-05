import { cn } from "../../lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  change?: number
  changeLabel?: string
  icon?: React.ReactNode
  trend?: "up" | "down" | "neutral"
  className?: string
  valueClassName?: string
}

export function MetricCard({
  title,
  value,
  subtitle,
  change,
  changeLabel,
  icon,
  trend,
  className,
  valueClassName,
}: MetricCardProps) {
  const getTrendIcon = () => {
    if (trend === "up") return <TrendingUp className="h-3.5 w-3.5" />
    if (trend === "down") return <TrendingDown className="h-3.5 w-3.5" />
    return <Minus className="h-3.5 w-3.5" />
  }

  const getTrendColor = () => {
    if (trend === "up") return "text-emerald-500 bg-emerald-500/10"
    if (trend === "down") return "text-red-400 bg-red-400/10"
    return "text-muted-foreground bg-muted/50"
  }

  return (
    <Card className={cn("group", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {icon && (
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary transition-transform duration-200 group-hover:scale-110">
            {icon}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className={cn("text-2xl font-bold tracking-tight", valueClassName)}>
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>
        {(subtitle || change !== undefined || trend) && (
          <div className="mt-1 flex items-center gap-1.5 text-xs">
            {(change !== undefined || trend) && (
              <span className={cn("inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 font-medium", getTrendColor())}>
                {getTrendIcon()}
                {change !== undefined && (
                  <span>
                    {change > 0 ? "+" : ""}
                    {change.toFixed(2)}%
                  </span>
                )}
              </span>
            )}
            {changeLabel && <span className="text-muted-foreground">{changeLabel}</span>}
            {subtitle && <span className="text-muted-foreground">{subtitle}</span>}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

interface MetricGridProps {
  children: React.ReactNode
  columns?: 2 | 3 | 4 | 5 | 6
  className?: string
}

export function MetricGrid({
  children,
  columns = 4,
  className,
}: MetricGridProps) {
  const gridCols = {
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-3",
    4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
    5: "grid-cols-1 md:grid-cols-3 lg:grid-cols-5",
    6: "grid-cols-1 md:grid-cols-3 lg:grid-cols-6",
  }

  return (
    <div className={cn("grid gap-4", gridCols[columns], className)}>
      {children}
    </div>
  )
}
