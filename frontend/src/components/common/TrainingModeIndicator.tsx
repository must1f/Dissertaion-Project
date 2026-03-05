import { useEffect, useState, useRef } from "react"
import { Badge } from "../ui/badge"
import { trainingApi, TrainingModeInfo } from "../../services/trainingApi"

interface TrainingModeIndicatorProps {
  className?: string
  showDetails?: boolean
}

// Cache the mode info globally to prevent redundant fetches across component instances
let cachedModeInfo: TrainingModeInfo | null = null
let lastFetchTime = 0
const CACHE_TTL_MS = 60000 // 60 seconds cache

export function TrainingModeIndicator({
  className = "",
  showDetails = false,
}: TrainingModeIndicatorProps) {
  const [modeInfo, setModeInfo] = useState<TrainingModeInfo | null>(cachedModeInfo)
  const [loading, setLoading] = useState(!cachedModeInfo)
  const [error, setError] = useState<string | null>(null)
  const isMounted = useRef(true)

  useEffect(() => {
    isMounted.current = true

    const fetchMode = async () => {
      // Skip if we have recent cached data
      const now = Date.now()
      if (cachedModeInfo && (now - lastFetchTime) < CACHE_TTL_MS) {
        if (isMounted.current) {
          setModeInfo(cachedModeInfo)
          setLoading(false)
        }
        return
      }

      try {
        const info = await trainingApi.getTrainingMode()
        cachedModeInfo = info
        lastFetchTime = Date.now()
        if (isMounted.current) {
          setModeInfo(info)
          setError(null)
        }
      } catch (err) {
        if (isMounted.current) {
          setError("Failed to fetch training mode")
          console.error("Training mode fetch error:", err)
        }
      } finally {
        if (isMounted.current) {
          setLoading(false)
        }
      }
    }

    fetchMode()
    // Refresh every 60 seconds (reduced from 30s to minimize log clutter)
    const interval = setInterval(fetchMode, 60000)
    return () => {
      isMounted.current = false
      clearInterval(interval)
    }
  }, [])

  if (loading) {
    return (
      <Badge variant="secondary" className={className}>
        Checking...
      </Badge>
    )
  }

  if (error || !modeInfo) {
    return (
      <Badge variant="destructive" className={className}>
        Mode Unknown
      </Badge>
    )
  }

  const isReal = modeInfo.mode === "real"

  if (showDetails) {
    return (
      <div className={`flex flex-col gap-1 ${className}`}>
        <div className="flex items-center gap-2">
          <Badge variant={isReal ? "success" : "warning"}>
            {isReal ? "Real Training" : "Simulated"}
          </Badge>
          {isReal && (
            <span className="text-xs text-green-600 dark:text-green-400">
              Neural networks active
            </span>
          )}
        </div>
        {!isReal && modeInfo.import_error && (
          <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
            {modeInfo.import_error}
          </p>
        )}
      </div>
    )
  }

  return (
    <Badge
      variant={isReal ? "success" : "warning"}
      className={className}
      title={modeInfo.message}
    >
      {isReal ? "Real Training" : "Simulated Mode"}
    </Badge>
  )
}

export default TrainingModeIndicator
