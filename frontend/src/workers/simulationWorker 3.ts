import {
    simulateGBM,
    simulateBootstrap,
    simulateMarkovRegimeSwitching,
    simulateMertonJumpDiffusion,
    simulateHestonModel,
    RegimeConfig
} from '../lib/simulationMath'

export type ModelType = 'gbm' | 'bootstrap' | 'markov' | 'merton' | 'heston'

export interface SimulationRequest {
    modelType: ModelType
    nSims: number
    horizon: number
    initialPrice: number
    dt: number
    seed: number

    // GBM Params
    mu?: number
    sigma?: number

    // Bootstrap Params
    historicalReturns?: number[]

    // Markov Params
    regimesConfig?: RegimeConfig[]
    transitionMatrix?: number[][]
    initialProbabilities?: number[]

    // Merton Params
    lambda?: number
    muJ?: number
    sigmaJ?: number

    // Heston Params
    initialVar?: number
    kappa?: number
    theta?: number
    xi?: number
    rho?: number
}

export interface SimulationResponse {
    paths: Float32Array[]
    regimes?: Uint8Array[]
    error?: string
}

self.onmessage = (e: MessageEvent<SimulationRequest>) => {
    try {
        const req = e.data
        let result: { paths: Float32Array[], regimes?: Uint8Array[] }

        switch (req.modelType) {
            case 'gbm':
                if (req.mu === undefined || req.sigma === undefined) {
                    throw new Error('GBM requires mu and sigma')
                }
                result = simulateGBM(
                    req.nSims,
                    req.horizon,
                    req.initialPrice,
                    req.mu,
                    req.sigma,
                    req.dt,
                    req.seed
                )
                break

            case 'bootstrap':
                if (!req.historicalReturns) {
                    throw new Error('Bootstrap requires historicalReturns')
                }
                result = simulateBootstrap(
                    req.nSims,
                    req.horizon,
                    req.initialPrice,
                    req.historicalReturns,
                    req.seed
                )
                break

            case 'markov':
                if (!req.regimesConfig || !req.transitionMatrix || !req.initialProbabilities) {
                    throw new Error('Markov regime switching requires regimesConfig, transitionMatrix, and initialProbabilities')
                }
                result = simulateMarkovRegimeSwitching(
                    req.nSims,
                    req.horizon,
                    req.initialPrice,
                    req.regimesConfig,
                    req.transitionMatrix,
                    req.initialProbabilities,
                    req.dt,
                    req.seed
                )
                break

            case 'merton':
                if (req.mu === undefined || req.sigma === undefined || req.lambda === undefined || req.muJ === undefined || req.sigmaJ === undefined) {
                    throw new Error('Merton requires mu, sigma, lambda, muJ, sigmaJ')
                }
                result = simulateMertonJumpDiffusion(
                    req.nSims,
                    req.horizon,
                    req.initialPrice,
                    req.mu,
                    req.sigma,
                    req.lambda,
                    req.muJ,
                    req.sigmaJ,
                    req.dt,
                    req.seed
                )
                break

            case 'heston':
                if (req.initialVar === undefined || req.mu === undefined || req.kappa === undefined || req.theta === undefined || req.xi === undefined || req.rho === undefined) {
                    throw new Error('Heston requires initialVar, mu, kappa, theta, xi, rho')
                }
                result = simulateHestonModel(
                    req.nSims,
                    req.horizon,
                    req.initialPrice,
                    req.initialVar,
                    req.mu,
                    req.kappa,
                    req.theta,
                    req.xi,
                    req.rho,
                    req.dt,
                    req.seed
                )
                break

            default:
                throw new Error(`Unknown model type: ${req.modelType}`)
        }

        // Prepare transferables for zero-copy postMessage
        const transferables: Transferable[] = []
        for (const path of result.paths) {
            transferables.push(path.buffer)
        }
        if (result.regimes) {
            for (const reg of result.regimes) {
                transferables.push(reg.buffer)
            }
        }

        // TypeScript sees 'self' as Window without webworker lib, yielding overload errors for postMessage.
        ; (self as any).postMessage(
            { paths: result.paths, regimes: result.regimes } as SimulationResponse,
            transferables
        )
    } catch (error: any) {
        self.postMessage({ error: error.message } as SimulationResponse)
    }
}
