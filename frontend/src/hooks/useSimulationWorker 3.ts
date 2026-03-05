import { useState, useEffect, useRef, useCallback } from 'react'
import { SimulationRequest, SimulationResponse } from '../workers/simulationWorker'

export function useSimulationWorker() {
    const [isSimulating, setIsSimulating] = useState(false)
    const [results, setResults] = useState<{ paths: Float32Array[], regimes?: Uint8Array[] } | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [processingTimeMs, setProcessingTimeMs] = useState(0)
    const workerRef = useRef<Worker | null>(null)

    useEffect(() => {
        // Initialize Web Worker
        workerRef.current = new Worker(new URL('../workers/simulationWorker.ts', import.meta.url), {
            type: 'module'
        })

        workerRef.current.onmessage = (e: MessageEvent<SimulationResponse>) => {
            setIsSimulating(false)
            if (e.data.error) {
                setError(e.data.error)
                return
            }
            setResults({ paths: e.data.paths, regimes: e.data.regimes })
        }

        workerRef.current.onerror = (e) => {
            setIsSimulating(false)
            setError(`Worker error: ${e.message}`)
        }

        return () => {
            if (workerRef.current) {
                workerRef.current.terminate()
            }
        }
    }, [])

    const runSimulation = useCallback((req: SimulationRequest) => {
        if (!workerRef.current || isSimulating) return
        setIsSimulating(true)
        setError(null)

        const startTime = performance.now()

        // Intercept to measure time on receipt
        const originalOnMessage = workerRef.current!.onmessage
        workerRef.current!.onmessage = (e: MessageEvent<SimulationResponse>) => {
            setProcessingTimeMs(performance.now() - startTime)
            if (originalOnMessage) {
                originalOnMessage.call(workerRef.current!, e)
            }
        }

        workerRef.current.postMessage(req)
    }, [isSimulating])

    return { runSimulation, isSimulating, results, error, processingTimeMs }
}
