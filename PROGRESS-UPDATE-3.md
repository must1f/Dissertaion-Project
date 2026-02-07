# Section 3: Current Stage and Methodology

## Physics-Informed Neural Networks for Financial Forecasting

---

```latex
\section{Current Stage and Methodology}

This section presents the technical implementation of the research system, reviews progress against the established timetable and documents the engineering challenges encountered alongside their respective mitigation strategies.

\subsection{Technical Implementation Overview}

The development of a rigorous experimental framework for evaluating Physics-Informed Neural Networks (PINNs) in financial forecasting necessitated the construction of three interconnected subsystems: a hybrid data infrastructure capable of handling temporal financial data at scale, a modular neural architecture framework embedding quantitative finance principles and a comprehensive evaluation engine that applies both statistical and financial performance measures.

\subsubsection{Hybrid Data Infrastructure}

Financial time-series data presents unique engineering challenges that distinguish it from conventional machine learning datasets. The temporal dependencies inherent in price sequences, combined with the necessity of preventing data leakage during model evaluation, demanded a purpose-built data architecture rather than reliance upon standard file-based storage.

\paragraph{TimescaleDB Implementation.} The primary storage layer utilises PostgreSQL extended with TimescaleDB, a time-series database optimised for temporal queries. This architectural decision was motivated by the requirement to execute complex windowed operations---such as computing rolling volatilities and momentum indicators across multiple time horizons---without incurring the memory overhead of loading entire datasets into RAM. The implementation employs \textit{hypertables} with a seven-day chunk interval, enabling efficient temporal partitioning that accelerates range queries by orders of magnitude compared to standard relational tables.

The database schema (\texttt{src/utils/database.py}) comprises four principal tables:
\begin{itemize}
    \item \texttt{finance.stock\_prices}: Raw OHLCV data with composite primary key on (time, ticker)
    \item \texttt{finance.features}: Engineered features including log returns, rolling volatilities (5, 20, 60-day windows), momentum indicators, RSI, MACD and Bollinger Bands
    \item \texttt{finance.predictions}: Model outputs with confidence intervals and metadata stored as JSONB
    \item \texttt{finance.model\_metrics}: Comprehensive performance metrics with hyperparameter tracking
\end{itemize}

Connection pooling with health checks (\texttt{pool\_pre\_ping=True}) ensures resilience during extended training runs, whilst upsert operations (\texttt{ON CONFLICT DO UPDATE}) maintain data integrity during incremental updates.

\paragraph{Parquet Storage Layer.} To prevent GPU starvation during training---a phenomenon whereby data loading becomes the bottleneck rather than computation---the system implements a secondary storage layer using Apache Parquet files with Snappy compression. This columnar format, integrated via DuckDB for SQL-like queries, provides read throughput exceeding that achievable through database connections. The \texttt{PhysicsAwareDataset} class (\texttt{src/data/dataset.py}) extends PyTorch's Dataset interface to provide physics-relevant tensors (price sequences, return sequences, volatility estimates) alongside standard feature matrices, ensuring the physics loss functions receive appropriately structured inputs.

\paragraph{Feature Engineering Pipeline.} The preprocessing module (\texttt{src/data/preprocessor.py}) implements a comprehensive feature engineering pipeline encompassing:
\begin{itemize}
    \item Return computation (log returns: $r_t = \log(P_t/P_{t-1})$; simple returns)
    \item Rolling volatility estimation with annualisation factor $\sqrt{252}$
    \item Momentum indicators across multiple horizons (5, 10, 20, 60 days)
    \item Technical indicators: RSI(14), MACD, Bollinger Bands, ATR(14), OBV
    \item Stationarity testing via Augmented Dickey-Fuller with automatic differencing
    \item Normalisation options: z-score, min-max and robust (median/IQR-based)
\end{itemize}

\subsubsection{Physics-Informed Architecture}

The central contribution of this research lies in embedding quantitative finance principles directly into neural network training through differentiable physics constraints. Unlike conventional supervised learning, which treats price prediction as a pure pattern-recognition task, the PINN framework imposes soft constraints derived from stochastic differential equations that have long characterised financial asset dynamics.

\paragraph{Governing Equations.} The system implements four governing equations, each capturing distinct aspects of market behaviour:

\begin{enumerate}
    \item \textbf{Geometric Brownian Motion (GBM):}
    \begin{equation}
        dS = \mu S \, dt + \sigma S \, dW
    \end{equation}
    This models trending behaviour in asset prices, with drift coefficient $\mu$ estimated from historical returns. The physics loss penalises predictions inconsistent with the GBM drift term.

    \item \textbf{Black-Scholes Partial Differential Equation:}
    \begin{equation}
        \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
    \end{equation}
    This fundamental no-arbitrage condition constrains model outputs to respect option pricing theory. The implementation uses a steady-state simplification given the single-step forecasting horizon.

    \item \textbf{Ornstein-Uhlenbeck (Mean Reversion):}
    \begin{equation}
        dX = \theta(\mu - X) \, dt + \sigma \, dW
    \end{equation}
    Applied to return sequences, this captures the empirically observed tendency of returns to revert toward a long-run mean, particularly relevant for value investing strategies.

    \item \textbf{Langevin Dynamics (Momentum):}
    \begin{equation}
        dX = -\gamma \nabla U(X) \, dt + \sqrt{2\gamma T} \, dW
    \end{equation}
    This physics-inspired equation models momentum effects through a potential function gradient, capturing the tendency of trending assets to continue their trajectory.
\end{enumerate}

\paragraph{Automatic Differentiation for Exact Derivatives.} A key innovation distinguishing this implementation from prior PINN research is the use of PyTorch's automatic differentiation (\texttt{torch.autograd.grad}) for computing exact derivatives required by the physics constraints. Conventional approaches employ numerical approximations (finite differences), which introduce discretisation errors and require careful tuning of step sizes. By setting \texttt{create\_graph=True}, the implementation computes true analytical derivatives through the computational graph, enabling:

\begin{verbatim}
# First derivative: dV/dS
dV_dS = torch.autograd.grad(V, x_grad, grad_outputs,
                            create_graph=True, retain_graph=True)[0]

# Second derivative: d²V/dS²
d2V_dS2 = torch.autograd.grad(dV_dS, x_grad, grad_outputs,
                              create_graph=True, retain_graph=True)[0]
\end{verbatim}

This approach ensures that physics residuals backpropagate correctly into network parameters, maintaining end-to-end differentiability.

\paragraph{Learnable Physics Parameters.} Rather than assuming fixed values for physics parameters---an assumption that would impose potentially incorrect priors on the model---the implementation treats key parameters as learnable variables optimised during training:

\begin{itemize}
    \item $\theta$ (OU mean-reversion speed): Initialised at 1.0, constrained positive via softplus
    \item $\gamma$ (Langevin friction coefficient): Initialised at 0.5, constrained positive
    \item $T$ (Langevin temperature): Initialised at 0.1, constrained positive
\end{itemize}

This design allows the network to ``discover'' asset-specific dynamics from data rather than relying upon assumed constants, addressing the No Free Lunch theorem's implication that optimal parameters vary across instruments and market regimes.

\paragraph{Architecture Variants.} To rigorously isolate the contribution of individual physics constraints, the system implements fourteen distinct neural architectures organised into three categories:

\textbf{Baseline Models (5):}
\begin{itemize}
    \item LSTM (Long Short-Term Memory) with configurable layers and bidirectional variant
    \item GRU (Gated Recurrent Unit) for computational efficiency
    \item BiLSTM (Bidirectional LSTM) for enhanced context capture
    \item AttentionLSTM incorporating self-attention mechanisms
    \item Transformer (encoder-only) with multi-head self-attention
\end{itemize}

\textbf{PINN Variants (7):}
\begin{itemize}
    \item Baseline (data loss only, $\lambda_{physics} = 0$)
    \item Pure GBM (trend-focused)
    \item Pure OU (mean-reversion-focused)
    \item Pure Black-Scholes (no-arbitrage constraint)
    \item Hybrid GBM+OU (combining trend and reversion)
    \item Global (all four physics constraints active)
    \item Langevin-only (momentum-focused)
\end{itemize}

\textbf{Advanced Architectures (2):}
\begin{itemize}
    \item \textbf{StackedPINN} (\texttt{src/models/stacked\_pinn.py}): A three-stage architecture comprising a PhysicsEncoder (dense layers with LayerNorm and GELU activation), ParallelHeads (simultaneous LSTM and GRU processing with attention-based combination) and a PredictionHead with dual regression/classification outputs
    \item \textbf{ResidualPINN}: A correction-based approach where a base LSTM/GRU produces an initial estimate, subsequently refined by a physics-informed residual network
\end{itemize}

This granularity enables systematic ablation studies isolating the contribution of each physical law to forecasting accuracy across different market conditions.

\paragraph{Total Loss Function.} The training objective combines data fidelity with physics constraints through weighted summation:

\begin{equation}
    \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{gbm}\mathcal{L}_{gbm} + \lambda_{bs}\mathcal{L}_{bs} + \lambda_{ou}\mathcal{L}_{ou} + \lambda_{langevin}\mathcal{L}_{langevin}
\end{equation}

where $\lambda$ coefficients (default 0.1 each) control the relative importance of each constraint.

\subsubsection{Advanced Evaluation Engine}

Rigorous model assessment in financial forecasting requires metrics that extend beyond conventional machine learning measures. The evaluation subsystem (\texttt{src/evaluation/}) implements a comprehensive framework spanning predictive accuracy, financial performance and statistical validation.

\paragraph{Predictive Metrics.} Standard regression metrics implemented include:
\begin{itemize}
    \item Root Mean Squared Error (RMSE)
    \item Mean Absolute Error (MAE)
    \item Mean Absolute Percentage Error (MAPE) with epsilon protection against division by zero
    \item Coefficient of Determination ($R^2$)
    \item Directional Accuracy with configurable significance thresholds
\end{itemize}

\paragraph{Financial Performance Metrics.} The \texttt{financial\_metrics.py} module (1,200+ lines) implements industry-standard portfolio performance measures:

\begin{itemize}
    \item \textbf{Sharpe Ratio}: $(r_p - r_f) / \sigma_p \times \sqrt{252}$, clipped to $[-5, 5]$
    \item \textbf{Sortino Ratio}: Downside-deviation variant, clipped to $[-10, 10]$
    \item \textbf{Maximum Drawdown}: Peak-to-trough decline, bounded at $[-1.0, 0.0]$
    \item \textbf{Calmar Ratio}: Annualised return per unit drawdown
    \item \textbf{Win Rate}: Proportion of profitable trading periods
    \item \textbf{Profit Factor}: Gross profit divided by gross loss
    \item \textbf{Information Coefficient}: Correlation between predicted and realised returns
    \item \textbf{Value at Risk (VaR)}: 5\% quantile of return distribution
    \item \textbf{Conditional VaR (CVaR)}: Expected shortfall beyond VaR threshold
\end{itemize}

Extensive bounds-checking and NaN/Infinity protection ensure numerical stability across all market conditions.

\paragraph{Monte Carlo Uncertainty Quantification.} The \texttt{monte\_carlo.py} module implements MC Dropout for uncertainty estimation, executing 100+ forward passes with dropout enabled during inference. This Bayesian approximation yields:
\begin{itemize}
    \item Mean and median prediction paths
    \item Confidence intervals (default 95\%)
    \item VaR and CVaR risk measures
    \item Prediction variance decomposition
\end{itemize}

\paragraph{Walk-Forward Validation.} To prevent look-ahead bias---a critical concern in financial time-series research---the system implements walk-forward validation (\texttt{src/training/walk\_forward.py}) with both expanding-window (train on $[0, t]$, test on $[t, t+h]$) and sliding-window (train on $[t-w, t]$, test on $[t, t+h]$) variants. This ensures that no future information contaminates model training.

\paragraph{Statistical Significance Testing.} The evaluation engine includes:
\begin{itemize}
    \item \textbf{Diebold-Mariano Test}: Compares forecast accuracy between model pairs with Newey-West correction for autocorrelated errors
    \item \textbf{Bootstrap Confidence Intervals}: Resampling-based uncertainty quantification for all metrics
\end{itemize}

\paragraph{Backtesting Platform.} The comprehensive backtesting module (\texttt{src/evaluation/backtesting\_platform.py}) simulates realistic trading conditions:
\begin{itemize}
    \item Position sizing: Fixed, Kelly Criterion, volatility-scaled and confidence-based methods
    \item Transaction costs: Commission (0.1\%) and slippage (0.05\%)
    \item Risk management: Stop-loss (2\%), take-profit (5\%), maximum position limits (20\%)
    \item Portfolio tracking: Cash, positions, entry prices, holding periods
    \item Benchmark strategies: Buy-and-hold, SMA crossover, random walk
\end{itemize}

\subsection{Progress Review Against Timetable}

Table~\ref{tab:progress} summarises the project status at Day 77 against the original timetable.

\begin{table}[h]
\centering
\caption{Progress Against Original Timetable}
\label{tab:progress}
\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Phase} & \textbf{Description} & \textbf{Planned} & \textbf{Status} \\
\hline
1 & Literature Review & Weeks 1--3 & \textcolor{green}{Complete} \\
2 & Environment Setup & Weeks 4--5 & \textcolor{green}{Complete} \\
3 & Baseline Benchmarks & Weeks 6--8 & \textcolor{green}{Complete} \\
4 & Core Physics Integration & Weeks 9--11 & \textcolor{green}{Complete} \\
5 & Evaluation Framework & Weeks 11--13 & \textcolor{orange}{95\% Complete} \\
6 & Analysis \& Write-up & Weeks 14--16 & Pending \\
\hline
\end{tabular}
\end{table}

\paragraph{Phase 1--4 Completion.} The foundational phases have been completed on schedule. The literature review established the theoretical grounding in stochastic calculus and PINN methodology. Environment setup delivered the Docker-containerised infrastructure with CI/CD pipelines via GitHub Actions. Baseline benchmarks produced trained LSTM, GRU, BiLSTM, AttentionLSTM and Transformer models. Core physics integration delivered all four governing equations with automatic differentiation and learnable parameters.

\paragraph{Phase 5 Status.} The evaluation framework is 95\% complete. All 22 metrics are implemented and validated. Walk-forward validation and Monte Carlo uncertainty quantification are operational. The backtesting platform is feature-complete. Remaining work involves:
\begin{itemize}
    \item Final validation of cross-model metric consistency
    \item Dashboard refinements for comparative visualisation
    \item Documentation of evaluation methodology for dissertation
\end{itemize}

\subsection{Engineering Challenges and Mitigation Strategies}

The development process encountered several non-trivial engineering challenges requiring bespoke solutions. This section documents these challenges and the mitigation strategies employed, demonstrating adaptive project management and technical problem-solving.

\subsubsection{The Stiff PDE Problem}

\paragraph{Challenge.} During initial PINN training, physics gradients dominated the loss landscape, causing the model to satisfy physics constraints whilst ignoring data fidelity. This ``stiff PDE'' phenomenon---wherein physics terms have significantly larger gradient magnitudes than data terms---prevented effective learning.

\paragraph{Symptoms.} Models converged to physically plausible but predictively useless solutions. Training loss decreased (physics satisfied) whilst validation loss increased (predictions deteriorated).

\paragraph{Mitigation: Curriculum Learning.} The solution implemented a curriculum learning scheduler that ``warm-starts'' the model on data before introducing physics:

\begin{enumerate}
    \item \textbf{Phase 1 (Epochs 1--10):} Train with $\lambda_{physics} = 0$ (pure data loss)
    \item \textbf{Phase 2 (Epochs 11--30):} Linearly ramp physics weights from 0 to target values
    \item \textbf{Phase 3 (Epochs 31+):} Train with full physics constraints at target weights
\end{enumerate}

This approach allows the network to first learn meaningful representations from data before physics constraints guide refinement, avoiding the local minima associated with premature physics enforcement.

\subsubsection{Hardware Acceleration on Apple Silicon}

\paragraph{Challenge.} Development occurred on macOS with Apple Silicon (M-series chips), which lack CUDA support. Standard PyTorch code paths defaulted to CPU execution, resulting in prohibitively slow training times.

\paragraph{Mitigation: Metal Performance Shaders (MPS).} The training infrastructure was optimised to leverage Apple's Metal Performance Shaders backend:

\begin{verbatim}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
\end{verbatim}

Additional MPS-specific optimisations included:
\begin{itemize}
    \item Disabling \texttt{pin\_memory} (incompatible with MPS)
    \item Setting \texttt{num\_workers=0} in DataLoaders (MPS multiprocessing issues)
    \item Explicit tensor device placement before operations
\end{itemize}

These modifications achieved training throughput comparable to entry-level CUDA GPUs whilst maintaining development on the primary workstation.

\subsubsection{Reproducibility Under Non-Determinism}

\paragraph{Challenge.} Both CUDA and MPS exhibit inherent non-determinism in certain operations (e.g., atomic additions in reduction operations), causing identical code to produce different results across runs. This impedes scientific reproducibility and complicates debugging.

\paragraph{Mitigation: Rigorous Seeding Protocol.} The reproducibility module (\texttt{src/utils/reproducibility.py}) implements comprehensive seeding:

\begin{verbatim}
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
\end{verbatim}

Whilst this does not eliminate all non-determinism (some operations remain irreducibly stochastic), it reduces variance to acceptable levels for research purposes. Experimental results report mean and standard deviation across multiple seeds.

\subsubsection{Financial Metrics Numerical Stability}

\paragraph{Challenge.} During evaluation of ResidualPINN and StackedPINN models, financial metrics exhibited catastrophic numerical instability: infinite Sharpe ratios, NaN drawdowns and physically impossible values (e.g., maximum drawdown of $-6696.99\%$, exceeding the theoretical bound of $-100\%$).

\paragraph{Root Causes.} Investigation (documented in \texttt{BUG\_DOCUMENTATION.md}) revealed multiple interacting issues:
\begin{enumerate}
    \item Cumulative return computation via \texttt{np.cumprod(1 + returns)} overflowed for extreme predictions
    \item No lower bound on cumulative equity (allowed negative portfolio values)
    \item Maximum drawdown calculated without physical bounds
    \item Inconsistent annualisation approaches across metric functions
\end{enumerate}

\paragraph{Mitigation: Defensive Bounds and Validation.} The financial metrics module was extensively refactored:
\begin{itemize}
    \item Return clipping: \texttt{np.clip(returns, -0.99, 1.0)} prevents $>100\%$ single-period losses
    \item Equity floor: Cumulative returns $\geq 10^{-10}$
    \item Drawdown bounds: Clamped to $[-1.0, 0.0]$
    \item Sharpe bounds: Clamped to $[-5.0, 5.0]$
    \item Comprehensive NaN/Inf replacement with bounded defaults
    \item Unified evaluation pipeline ensuring consistent computation paths
\end{itemize}

These defensive measures ensure that all models, regardless of prediction quality, produce interpretable metrics suitable for comparative analysis.

\subsubsection{Black-Scholes Computational Overhead}

\paragraph{Challenge.} Computing second-order derivatives via automatic differentiation for the Black-Scholes constraint incurs approximately 20\% training overhead compared to first-order-only physics constraints.

\paragraph{Mitigation: Optional Constraint Activation.} The physics weight configuration allows selective constraint activation:
\begin{itemize}
    \item Setting $\lambda_{bs} = 0$ disables Black-Scholes computation entirely
    \item Enables rapid prototyping with cheaper constraints (GBM, OU)
    \item Full constraint suite activated for final evaluation runs
\end{itemize}

This flexibility balances experimental iteration speed against evaluation rigour.

\subsection{Infrastructure and DevOps}

\paragraph{Containerisation.} The entire system is containerised via Docker, with \texttt{docker-compose.yml} orchestrating three services:
\begin{itemize}
    \item \texttt{timescaledb}: PostgreSQL 15 with TimescaleDB extension
    \item \texttt{pinn-app}: Training container with GPU passthrough (when available)
    \item \texttt{web}: Streamlit dashboard on ports 5000/8501
\end{itemize}

Health checks ensure service dependencies are satisfied before training commences.

\paragraph{Continuous Integration.} GitHub Actions workflow (\texttt{.github/workflows/ci.yml}) executes on all pushes and pull requests:
\begin{itemize}
    \item Dependency installation (PyTorch CPU, project requirements)
    \item Unit test execution via pytest
    \item Code quality checks via flake8
\end{itemize}

\paragraph{Web Interface.} Six Streamlit dashboards (\texttt{src/web/}) provide interactive visualisation:
\begin{itemize}
    \item Main application with project overview
    \item PINN variant comparison dashboard
    \item All-models performance leaderboard
    \item Monte Carlo simulation visualiser
    \item Backtesting results explorer
    \item Prediction visualiser with residual analysis
\end{itemize}

\subsection{Summary}

The technical implementation represents a mature, research-grade system comprising 42 Python modules totalling approximately 1.8MB of source code. The hybrid data infrastructure, physics-informed architectures with learnable parameters and comprehensive evaluation engine collectively provide the experimental apparatus necessary for rigorous empirical investigation of PINNs in financial forecasting. The engineering challenges encountered---and successfully mitigated---demonstrate both the non-trivial nature of this research domain and the adaptive problem-solving essential to scientific computing.
```

---

## Complete System Architecture and Implementation Details

This section provides exhaustive documentation of every component implemented in the PINN financial forecasting system, explaining not just what was built but how it works and why specific design decisions were made.

---

## 1. Data Pipeline: From Raw Market Data to Model-Ready Sequences

### 1.1 Data Retrieval Infrastructure

The system implements a robust multi-source data acquisition pipeline designed for reliability and redundancy.

#### Primary Data Source: Yahoo Finance (yfinance)

The `DataFetcher` class (`src/data/fetcher.py`) serves as the primary interface to financial market data:

```python
# Key configuration
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', ...]  # 50 S&P 500 constituents
date_range = '2014-01-01' to '2024-01-01'         # 10 years of history
interval = '1d'                                     # Daily OHLCV data
```

**Why yfinance?**
- No API key required for basic functionality
- Provides adjusted close prices (accounting for splits/dividends)
- Includes volume data essential for technical indicators (OBV)
- Rate limiting is generous for research use

**Implementation Details:**
- Fetches OHLCV data: Open, High, Low, Close, Volume
- Automatic retry logic with exponential backoff for transient failures
- Multi-ticker batching to minimize API calls
- Timestamp standardization to UTC for consistency

#### Backup Data Source: Alpha Vantage

When yfinance encounters rate limits or data gaps, the system falls back to Alpha Vantage:

```python
# Requires API key in .env file
ALPHA_VANTAGE_API_KEY=your_key_here
```

**Alpha Vantage Limitations:**
- 5 API calls per minute (free tier)
- 500 calls per day maximum
- The fetcher implements a rate-limiter respecting these constraints

#### Data Storage Strategy

The system employs a dual-storage architecture for different access patterns:

**1. TimescaleDB (Primary Storage)**

TimescaleDB extends PostgreSQL with time-series optimizations:

```sql
-- Hypertable creation with automatic partitioning
SELECT create_hypertable('finance.stock_prices', 'time', chunk_time_interval => INTERVAL '7 days');

-- Schema structure
CREATE TABLE finance.stock_prices (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (time, ticker)
);
```

**Why TimescaleDB?**
- Transparent compression (90%+ reduction in storage)
- Automatic chunking enables parallel query execution
- Native support for continuous aggregates (pre-computed rollups)
- Standard SQL interface (no new query language to learn)

**2. Parquet Files (Backup/Fast Access)**

When the database is unavailable or for faster batch reads:

```python
# Parquet storage with Snappy compression
df.to_parquet('data/parquet/stock_prices.parquet', compression='snappy')
```

**Why Parquet?**
- Columnar format optimized for analytics queries
- Read only the columns needed (no full table scan)
- Snappy compression provides good speed/size tradeoff
- Works offline without database connection

### 1.2 Feature Engineering Pipeline

The `DataPreprocessor` class (`src/data/preprocessor.py`) transforms raw OHLCV data into model-ready features through a deterministic pipeline.

#### Step 1: Return Calculations

```python
# Log returns (preferred for financial modeling)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Simple returns (for interpretability)
df['simple_return'] = df['close'].pct_change()
```

**Why Log Returns?**
- Additive across time (can sum log returns for cumulative)
- Approximately normally distributed (enables statistical tests)
- Symmetric: +10% and -10% log returns are equidistant from 0
- Required by physics equations (GBM, OU operate on log prices)

#### Step 2: Volatility Estimation

Rolling volatility captures time-varying risk:

```python
# Multiple windows for different trading horizons
for window in [5, 20, 60]:
    df[f'rolling_volatility_{window}'] = df.groupby('ticker')['log_return'].transform(
        lambda x: x.rolling(window=window).std()
    )
```

**Window Interpretations:**
- **5-day**: Weekly volatility (short-term traders)
- **20-day**: Monthly volatility (swing traders)
- **60-day**: Quarterly volatility (position traders)

#### Step 3: Momentum Indicators

```python
for window in [5, 10, 20, 60]:
    # Rate of change
    df[f'momentum_{window}'] = (df['close'] / df['close'].shift(window)) - 1

    # Simple Moving Average
    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
```

**Momentum Rationale:**
- Captures trending behavior (GBM physics constraint relates to drift)
- Multiple horizons detect trends at different scales
- SMA crossovers are classic trading signals

#### Step 4: Technical Indicators (via pandas_ta)

The system calculates 13 technical indicators using the pandas_ta library:

| Indicator | Calculation | Trading Signal |
|-----------|-------------|----------------|
| **RSI(14)** | Relative Strength Index | Overbought (>70) / Oversold (<30) |
| **MACD** | 12-period EMA - 26-period EMA | Bullish/Bearish crossovers |
| **MACD Signal** | 9-period EMA of MACD | Confirmation of MACD signals |
| **MACD Histogram** | MACD - Signal | Momentum strength |
| **Bollinger Upper** | SMA(20) + 2*std(20) | Resistance / Overbought |
| **Bollinger Middle** | SMA(20) | Trend direction |
| **Bollinger Lower** | SMA(20) - 2*std(20) | Support / Oversold |
| **ATR(14)** | Average True Range | Volatility measure |
| **OBV** | On-Balance Volume | Volume-based momentum |
| **Stochastic %K** | (Close - Low14) / (High14 - Low14) | Price position in range |
| **Stochastic %D** | 3-period SMA of %K | Smoothed stochastic |

**Implementation Robustness:**
```python
# Handle different pandas_ta versions with dynamic column name detection
macd = ta.macd(ticker_df['close'])
macd_col = [c for c in macd.columns if c.startswith('MACD_') and not c.startswith(('MACDs_', 'MACDh_'))][0]
```

#### Step 5: Stationarity Testing

Financial time series must be stationary for many statistical tests. The system performs Augmented Dickey-Fuller (ADF) tests:

```python
def test_stationarity(series, significance_level=0.05):
    result = adfuller(series, autolag='AIC')
    return {
        'is_stationary': result[1] < significance_level,  # p-value < 0.05
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4]
    }
```

**Key Finding:** Raw prices are typically non-stationary (random walk), but log returns are stationary. This validates the use of returns (not prices) as model inputs.

#### Step 6: Normalization

Per-ticker normalization prevents cross-asset scale contamination:

```python
# Z-score normalization per ticker
for ticker in df['ticker'].unique():
    scaler = StandardScaler()
    df.loc[df['ticker'] == ticker, feature_cols] = scaler.fit_transform(
        df.loc[df['ticker'] == ticker, feature_cols]
    )
    scalers[ticker] = scaler  # Save for inverse transform during backtesting
```

**Why Per-Ticker?**
- AAPL price ~$150 vs BRK-A price ~$500,000
- Without per-ticker scaling, high-priced stocks dominate gradients
- Scalers are preserved for inverse transformation during evaluation

### 1.3 Sequence Creation for Time Series Models

The `FinancialDataset` class (`src/data/dataset.py`) creates sequences suitable for RNN/Transformer models:

```python
# Configuration
sequence_length = 60    # 60 trading days (~3 months)
forecast_horizon = 1    # Predict 1 day ahead

# Sequence creation
for i in range(len(data) - sequence_length - forecast_horizon + 1):
    X.append(features[i:i + sequence_length])      # Input: 60 days of features
    y.append(targets[i + sequence_length])          # Target: next day's return
```

**PhysicsAwareDataset Extension:**

The `PhysicsAwareDataset` class provides additional tensors required by physics loss functions:

```python
def __getitem__(self, idx):
    return {
        'features': self.features[idx],           # (seq_len, num_features)
        'target': self.targets[idx],              # Scalar return
        'prices': self.prices[idx],               # (seq_len,) price sequence
        'returns': self.returns[idx],             # (seq_len,) return sequence
        'volatilities': self.volatilities[idx],   # (seq_len,) volatility sequence
        'ticker': self.tickers[idx],              # Ticker identifier
        'timestamp': self.timestamps[idx]         # Timestamp for backtesting
    }
```

### 1.4 Temporal Train/Val/Test Split

To prevent data leakage (using future information to predict past), the system implements strict temporal splitting:

```python
# Chronological split (NOT random shuffle)
train_end = int(n * 0.70)    # First 70% of data
val_end = int(n * 0.85)       # Next 15% of data

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]   # Final 15% held out
```

**Critical Principle:** The test set contains ONLY data from after the training period, simulating real-world deployment where future data is genuinely unavailable.

---

## 2. Model Architectures: From Baseline to Physics-Informed

### 2.1 Baseline Models (`src/models/baseline.py`)

The system implements five baseline architectures as performance benchmarks:

#### LSTM (Long Short-Term Memory)

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
```

**Why LSTM?**
- Designed to capture long-term dependencies (financial trends)
- Forget gates prevent vanishing gradients
- Industry standard for sequence modeling

#### GRU (Gated Recurrent Unit)

Simpler alternative to LSTM with fewer parameters:
- ~33% fewer parameters than LSTM
- Comparable performance in many tasks
- Faster training and inference

#### BiLSTM (Bidirectional LSTM)

Processes sequences in both directions:
```python
self.lstm = nn.LSTM(..., bidirectional=True)
# Output dimension doubles: hidden_dim * 2
```

**Trade-off:** Better context capture vs. not applicable for real-time prediction (requires future data).

#### AttentionLSTM

Combines LSTM with self-attention mechanism:
```python
# Attention weights computed from LSTM outputs
attention_weights = F.softmax(self.attention_layer(lstm_output), dim=1)
context = torch.sum(attention_weights * lstm_output, dim=1)
```

**Benefit:** Learns which timesteps are most relevant for prediction.

#### Transformer

Encoder-only Transformer architecture:
```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

**Positional Encoding:** Adds temporal position information since Transformers have no inherent notion of sequence order.

### 2.2 PINN Architecture (`src/models/pinn.py`)

The Physics-Informed Neural Network is the core innovation of this research.

#### PhysicsLoss Module

The `PhysicsLoss` class computes physics constraint violations:

```python
class PhysicsLoss(nn.Module):
    def __init__(self,
                 lambda_gbm=0.1,      # Weight for GBM constraint
                 lambda_bs=0.1,       # Weight for Black-Scholes
                 lambda_ou=0.1,       # Weight for Ornstein-Uhlenbeck
                 lambda_langevin=0.1, # Weight for Langevin
                 risk_free_rate=0.02, # Annual risk-free rate
                 dt=1.0/252.0):       # Time step (1 trading day)

        # LEARNABLE physics parameters (NOT hardcoded!)
        self.theta_raw = nn.Parameter(torch.tensor(1.0))   # OU mean reversion
        self.gamma_raw = nn.Parameter(torch.tensor(0.5))   # Langevin friction
        self.temperature_raw = nn.Parameter(torch.tensor(0.1))  # Langevin temperature
```

**Why Learnable Parameters?**
- Different assets have different dynamics (tech stocks vs. utilities)
- Market regimes change over time (bull vs. bear markets)
- The network learns optimal physics parameters from data

**Softplus Constraint:** Parameters must be positive:
```python
@property
def theta(self):
    return F.softplus(self.theta_raw)  # Always > 0
```

#### Physics Constraint Implementations

**1. Geometric Brownian Motion (GBM):**
```python
def gbm_residual(self, S, dS_dt, mu, sigma):
    # dS/dt should equal μS (drift term)
    residual = dS_dt - mu * S
    return torch.mean(residual ** 2)
```

**2. Black-Scholes PDE (with Automatic Differentiation):**
```python
def black_scholes_autograd_residual(self, model, x, sigma, price_feature_idx=0):
    # Enable gradient tracking
    x_grad = x.clone().detach().requires_grad_(True)
    V = model(x_grad)

    # First derivative via autograd
    dV_dS = torch.autograd.grad(V, x_grad, grad_outputs=torch.ones_like(V),
                                 create_graph=True)[0][:, -1, price_feature_idx]

    # Second derivative via autograd
    d2V_dS2 = torch.autograd.grad(dV_dS, x_grad, grad_outputs=torch.ones_like(dV_dS),
                                   create_graph=True)[0][:, -1, price_feature_idx]

    # Black-Scholes PDE: ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV ≈ 0
    bs_residual = 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
    return torch.mean(bs_residual ** 2)
```

**3. Ornstein-Uhlenbeck (Mean Reversion):**
```python
def ou_residual(self, X, dX_dt, theta, mu, sigma):
    # dX/dt should equal θ(μ - X) (mean reversion)
    residual = dX_dt - theta * (mu - X)
    return torch.mean(residual ** 2)
```

**4. Langevin Dynamics (Momentum):**
```python
def langevin_residual(self, X, dX_dt, grad_U, gamma, T):
    # dX/dt should equal -γ∇U(X) (momentum/friction)
    residual = dX_dt + gamma * grad_U
    return torch.mean(residual ** 2)
```

#### PINNModel Class

The complete PINN model combines a base neural network with physics constraints:

```python
class PINNModel(nn.Module):
    def __init__(self, input_dim, base_model='lstm', lambda_gbm=0.1, ...):
        # Base neural network (LSTM, GRU, or Transformer)
        if base_model == 'lstm':
            self.base_model = LSTMModel(input_dim, ...)

        # Physics loss module
        self.physics_loss = PhysicsLoss(lambda_gbm=lambda_gbm, ...)

    def compute_loss(self, predictions, targets, metadata, enable_physics=True):
        # Data loss (MSE)
        data_loss = F.mse_loss(predictions, targets)

        if not enable_physics:
            return data_loss

        # Physics losses
        physics_loss = self.physics_loss(predictions, targets,
                                          metadata['prices'],
                                          metadata['returns'],
                                          metadata['volatilities'])

        # Combined loss
        return data_loss + physics_loss
```

### 2.3 PINN Variants

To isolate the contribution of each physics constraint, seven variants are implemented:

| Variant | λ_GBM | λ_BS | λ_OU | λ_Langevin | Purpose |
|---------|-------|------|------|------------|---------|
| Baseline | 0.0 | 0.0 | 0.0 | 0.0 | Control (no physics) |
| Pure GBM | 0.1 | 0.0 | 0.0 | 0.0 | Trend-following only |
| Pure OU | 0.0 | 0.0 | 0.1 | 0.0 | Mean-reversion only |
| Pure BS | 0.0 | 0.1 | 0.0 | 0.0 | No-arbitrage only |
| Langevin | 0.0 | 0.0 | 0.0 | 0.1 | Momentum only |
| GBM+OU | 0.05 | 0.0 | 0.05 | 0.0 | Trend + Reversion |
| Global | 0.1 | 0.03 | 0.1 | 0.1 | All constraints |

### 2.4 Advanced Architectures (`src/models/stacked_pinn.py`)

#### StackedPINN Architecture

A three-stage architecture with parallel processing:

```
Input Features (batch, seq_len, input_dim)
         ↓
    PhysicsEncoder
    ┌────────────────────────────────┐
    │ Linear → LayerNorm → GELU     │
    │ Linear → LayerNorm → GELU     │
    │ Physics-aware Projection      │
    └────────────────────────────────┘
         ↓
    ParallelHeads
    ┌────────────────────────────────┐
    │     ┌──────┐    ┌─────┐       │
    │     │ LSTM │    │ GRU │       │
    │     └──┬───┘    └──┬──┘       │
    │        │           │          │
    │     Concatenate [hidden_dim * 2] │
    │        │                      │
    │   Attention Weights (softmax) │
    └────────────────────────────────┘
         ↓
    PredictionHead
    ┌────────────────────────────────┐
    │ Shared Dense Layers           │
    │      ↓           ↓            │
    │ Regression    Classification  │
    │ (return)      (direction)     │
    └────────────────────────────────┘
```

**PhysicsEncoder:** Learns physics-aware feature representations before sequence processing.

**ParallelHeads:** LSTM captures long-term dependencies while GRU provides efficiency; attention dynamically weights their contributions.

**Dual Outputs:** Returns both:
- Regression output: Predicted return magnitude
- Classification output: Predicted direction (up/down)

#### ResidualPINN Architecture

A correction-based approach:

```
Input Features
         ↓
    Base Model (LSTM or GRU)
         ↓
    Base Prediction (initial estimate)
         ↓                    ↓
         │            Correction Network
         │            ┌────────────────┐
         │            │ Hidden + Pred  │
         │            │ Dense + Tanh   │
         │            │ Residual       │
         │            └───────┬────────┘
         │                    │
         └───────────────────┘
                    ↓
    Final = Base + Correction
```

**Why Residual Learning?**
- Corrections are typically smaller than raw predictions (easier to learn)
- Base model provides reasonable defaults
- Gradient flow improved through residual connections
- Physics constraints refine rather than replace predictions

---

## 3. Training Framework

### 3.1 Trainer Class (`src/training/trainer.py`)

The `Trainer` class manages the complete training lifecycle:

```python
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)
        self.early_stopping = EarlyStopping(patience=10)

    def train_epoch(self, dataloader):
        self.model.train()
        for batch in dataloader:
            # Forward pass
            predictions = self.model(batch['features'])

            # Compute loss (including physics if enabled)
            loss, loss_dict = self.model.compute_loss(
                predictions, batch['target'],
                metadata=batch,
                enable_physics=self.enable_physics
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
```

**Key Features:**
- Gradient clipping prevents exploding gradients
- Learning rate scheduling adapts to loss plateaus
- Early stopping prevents overfitting
- Checkpoint saving preserves best models

### 3.2 Curriculum Learning (`src/training/curriculum.py`)

Curriculum learning gradually introduces physics constraints:

```python
class CurriculumScheduler:
    def __init__(self, total_epochs, warmup_epochs=10, strategy='cosine'):
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy

    def get_physics_scale(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.0  # Pure data loss during warmup

        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

        if self.strategy == 'linear':
            return progress
        elif self.strategy == 'cosine':
            return 0.5 * (1 - math.cos(math.pi * progress))
        elif self.strategy == 'exponential':
            return progress ** 2
```

**Available Strategies:**

| Strategy | Formula | Characteristic |
|----------|---------|----------------|
| Linear | `scale = progress` | Steady increase |
| Cosine | `scale = 0.5*(1-cos(π*progress))` | Slow start/end, fast middle |
| Exponential | `scale = progress²` | Slow start, fast finish |
| Step | Discrete jumps at 25/50/75% | Controlled phases |

### 3.3 Walk-Forward Validation (`src/training/walk_forward.py`)

Simulates realistic model deployment over time:

```python
def walk_forward_validation(data, model, window_type='expanding'):
    results = []

    for t in range(initial_train_end, len(data) - horizon, step_size):
        if window_type == 'expanding':
            train_data = data[0:t]           # All historical data
        else:  # sliding
            train_data = data[t-window:t]    # Fixed-size window

        test_data = data[t:t+horizon]

        # Retrain or fine-tune model
        model.fit(train_data)

        # Evaluate on out-of-sample data
        predictions = model.predict(test_data)
        results.append(evaluate(predictions, test_data))

    return aggregate_results(results)
```

**Window Types:**
- **Expanding:** Uses all historical data (more data, but older patterns may be irrelevant)
- **Sliding:** Fixed-size window (more recent data only)

---

## 4. Evaluation Framework

### 4.1 Prediction Metrics (`src/evaluation/metrics.py`)

Standard machine learning metrics:

```python
class MetricsCalculator:
    @staticmethod
    def rmse(predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

    @staticmethod
    def mae(predictions, targets):
        return np.mean(np.abs(predictions - targets))

    @staticmethod
    def mape(predictions, targets, epsilon=1e-8):
        return np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100

    @staticmethod
    def r2(predictions, targets):
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def directional_accuracy(predictions, targets):
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(targets))
        return np.mean(pred_direction == actual_direction)
```

### 4.2 Financial Metrics (`src/evaluation/financial_metrics.py`)

Industry-standard portfolio performance measures:

#### Sharpe Ratio

```python
def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    # Annualized risk-free rate to period rate
    rf_per_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_per_period

    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return np.clip(sharpe, -5.0, 5.0)  # Bound to realistic range
```

**Interpretation:**
- < 0: Underperforming risk-free rate
- 0-1: Mediocre risk-adjusted returns
- 1-2: Good risk-adjusted returns
- 2+: Excellent risk-adjusted returns

#### Sortino Ratio

```python
def sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    rf_per_period = risk_free_rate / 252
    excess_returns = returns - rf_per_period

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < target_return]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0

    if downside_std == 0:
        return 10.0 if np.mean(excess_returns) > 0 else 0.0

    return np.mean(excess_returns) / downside_std * np.sqrt(252)
```

**Why Sortino?**
- Sharpe penalizes both upside and downside volatility
- Sortino only penalizes downside (losses)
- Better for asymmetric return distributions

#### Maximum Drawdown

```python
def max_drawdown(returns, return_series=False):
    # Cumulative returns (equity curve)
    cum_returns = np.cumprod(1 + np.clip(returns, -0.99, 1.0))
    cum_returns = np.maximum(cum_returns, 1e-10)  # Floor at near-zero

    # Running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Drawdown at each point
    drawdown = (cum_returns - running_max) / running_max

    max_dd = np.min(drawdown)
    max_dd = np.clip(max_dd, -1.0, 0.0)  # Cannot lose more than 100%

    return max_dd
```

**Critical Bounds:**
- Drawdown cannot exceed -100% (total loss)
- Implemented defensive bounds after numerical overflow bugs

#### Additional Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Calmar Ratio** | Annual Return / \|Max DD\| | Return per unit of max loss |
| **Win Rate** | Winning Trades / Total Trades | Percentage profitable |
| **Profit Factor** | Gross Profit / Gross Loss | Profitability ratio |
| **Information Coefficient** | corr(predicted_returns, actual_returns) | Forecast skill |
| **VaR (5%)** | 5th percentile of returns | Worst expected loss (95% confidence) |
| **CVaR (5%)** | Mean of returns below VaR | Expected shortfall |

### 4.3 Strategy Return Computation

The system simulates a trading strategy based on model predictions:

```python
def compute_strategy_returns(predictions, targets, transaction_cost=0.003):
    # Generate signals: long if predicted return > 0, else short/cash
    positions = np.sign(predictions)

    # Actual returns achieved
    strategy_returns = positions[:-1] * targets[1:]

    # Deduct transaction costs on position changes
    position_changes = np.abs(np.diff(positions))
    costs = position_changes * transaction_cost

    strategy_returns -= costs

    return strategy_returns
```

**Transaction Cost Components:**
- Commission: 0.1% per trade
- Slippage: 0.05% per trade (price impact)
- Bid-ask spread: ~0.05-0.15%
- Total: ~0.3% assumed conservatively

---

## 5. Backtesting Platform (`src/evaluation/backtesting_platform.py`)

### 5.1 Platform Architecture

The backtesting platform simulates realistic trading:

```python
class BacktestingPlatform:
    def __init__(self, config: BacktestConfig):
        self.initial_capital = config.initial_capital  # $100,000
        self.commission = config.commission_rate       # 0.1%
        self.slippage = config.slippage_rate           # 0.05%
        self.max_position = config.max_position_size   # 20%
        self.stop_loss = config.stop_loss              # 2%
        self.take_profit = config.take_profit          # 5%
```

### 5.2 Position Sizing Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **Fixed** | Constant percentage of capital | Conservative baseline |
| **Kelly Criterion** | f* = (bp - q) / b | Optimal growth rate |
| **Volatility-Scaled** | size ∝ 1/σ | Larger in calm markets |
| **Confidence-Based** | size ∝ \|prediction\| | Scale with signal strength |

**Kelly Criterion Implementation:**
```python
def kelly_size(win_rate, avg_win, avg_loss, fractional=0.5):
    b = avg_win / avg_loss  # Win/loss ratio
    p = win_rate            # Probability of win
    q = 1 - p               # Probability of loss

    kelly_fraction = (b * p - q) / b

    # Half-Kelly for reduced volatility
    return max(0, min(kelly_fraction * fractional, 0.25))
```

### 5.3 Risk Management

```python
def apply_risk_management(self, position, entry_price, current_price):
    # Stop-loss check
    if position > 0 and current_price < entry_price * (1 - self.stop_loss):
        return 'close'  # Exit long position

    # Take-profit check
    if position > 0 and current_price > entry_price * (1 + self.take_profit):
        return 'close'  # Lock in profits

    return 'hold'
```

### 5.4 Benchmark Strategies

| Strategy | Logic | Purpose |
|----------|-------|---------|
| **Buy & Hold** | Always long | Market benchmark |
| **SMA Crossover** | Long when SMA(10) > SMA(50) | Trend-following benchmark |
| **Momentum** | Long when 20-day return > 0 | Momentum benchmark |
| **Mean Reversion** | Long when price < Bollinger Lower | Contrarian benchmark |
| **Random Walk** | Random positions | Sanity check |

---

## 6. Web Dashboards (`src/web/`)

### 6.1 Main Dashboard (`app.py`)

The central Streamlit interface:

```python
def main():
    st.set_page_config(page_title="PINN Financial Forecasting", layout="wide")

    # Prominent disclaimer
    st.warning("""
    **DISCLAIMER:** This is for ACADEMIC RESEARCH ONLY.
    NOT financial advice. Past performance does not guarantee future results.
    """)

    # Navigation
    page = st.sidebar.radio("Navigation", [
        "Overview",
        "Model Comparison",
        "PINN Analysis",
        "Backtesting",
        "Monte Carlo"
    ])
```

### 6.2 PINN Dashboard (`pinn_dashboard.py`)

Specialized dashboard for PINN variant comparison:

**Features:**
- Side-by-side comparison of all 8 PINN variants
- Four metric category tabs:
  1. Risk-Adjusted (Sharpe, Sortino, Calmar)
  2. Capital Preservation (Max DD, DD Duration)
  3. Trading Viability (Win Rate, Profit Factor)
  4. Signal Quality (Directional Accuracy, IC)
- Training history visualization with curriculum learning phases
- Learned physics parameter display

### 6.3 Backtesting Dashboard (`backtesting_dashboard.py`)

**Features:**
- Strategy configuration sidebar
- Equity curve visualization
- Drawdown chart
- Trade-by-trade breakdown
- Comparison against benchmarks
- Walk-forward validation results

### 6.4 Monte Carlo Dashboard (`monte_carlo_dashboard.py`)

**Features:**
- Price path simulations
- 95% confidence intervals
- VaR/CVaR analysis
- Stress test scenarios
- Ensemble uncertainty visualization

---

## 7. Monte Carlo Simulation (`src/evaluation/monte_carlo.py`)

### 7.1 MC Dropout for Uncertainty

```python
class MCDropoutPredictor:
    def __init__(self, model, n_samples=100):
        self.model = model
        self.n_samples = n_samples

    def predict_with_uncertainty(self, x):
        self.model.train()  # Keep dropout active during inference

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        return {
            'mean': predictions.mean(dim=0),
            'std': predictions.std(dim=0),
            'lower_ci': predictions.quantile(0.025, dim=0),
            'upper_ci': predictions.quantile(0.975, dim=0)
        }
```

### 7.2 Path Simulation

```python
def simulate_paths(self, initial_price, mu, sigma, horizon, n_paths=1000):
    dt = 1/252
    paths = np.zeros((n_paths, horizon))
    paths[:, 0] = initial_price

    for t in range(1, horizon):
        # GBM simulation
        dW = np.random.randn(n_paths) * np.sqrt(dt)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

    return {
        'paths': paths,
        'mean_path': paths.mean(axis=0),
        'lower_ci': np.percentile(paths, 2.5, axis=0),
        'upper_ci': np.percentile(paths, 97.5, axis=0),
        'var_5': np.percentile(paths[:, -1], 5),
        'cvar_5': paths[:, -1][paths[:, -1] <= np.percentile(paths[:, -1], 5)].mean()
    }
```

### 7.3 Stress Test Scenarios

| Scenario | Volatility Multiplier | Drift Adjustment | Description |
|----------|----------------------|------------------|-------------|
| Base | 1.0x | 0% | Normal conditions |
| High Volatility | 2.0x | 0% | Elevated uncertainty |
| Market Crash | 3.0x | -2% | Severe downturn |
| Bull Market | 0.8x | +1% | Low volatility rally |
| Black Swan | 5.0x | -5% | Extreme tail event |

---

## 8. Bug Fixes and Quality Assurance

### 8.1 Critical Bugs Fixed

| Bug | Severity | Root Cause | Fix |
|-----|----------|------------|-----|
| Infinite Sharpe Ratio | Critical | Cumulative return overflow | Return clipping to [-0.99, 1.0] |
| Max DD > -100% | Critical | No physical bounds | Clamping to [-1.0, 0.0] |
| NaN in metrics | Critical | Division by zero | Epsilon protection |
| ImportError in backtesting | Critical | Missing standalone functions | Added wrapper functions |
| IC computed on prices | High | Used levels not returns | Added `use_returns` parameter |
| Inconsistent DA scale | Medium | 0-100 vs 0-1 | Standardized to 0-1 |

### 8.2 Defensive Programming

All financial metric functions now include:
- Input validation (NaN/Inf checks)
- Return clipping to realistic bounds
- Output validation before returning
- Warning logs for edge cases

---

## 9. Configuration and Reproducibility

### 9.1 Configuration Management (`src/utils/config.py`)

Pydantic-based configuration with validation:

```python
class ModelConfig(BaseModel):
    input_dim: int = 13
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lambda_gbm: float = 0.1
    lambda_ou: float = 0.1

class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
```

### 9.2 Reproducibility (`src/utils/reproducibility.py`)

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 10. Codebase Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 42 |
| Lines of Code | ~15,000 |
| Models Implemented | 14 |
| Physics Equations | 4 |
| Financial Metrics | 22 |
| Web Dashboards | 6 |
| Unit Tests | 15+ |

---

## Appendix: Codebase Audit Summary

### Models Implemented

| Category | Model | File | Status |
|----------|-------|------|--------|
| Baseline | LSTM | `src/models/baseline.py` | Complete |
| Baseline | GRU | `src/models/baseline.py` | Complete |
| Baseline | BiLSTM | `src/models/baseline.py` | Complete |
| Baseline | AttentionLSTM | `src/models/baseline.py` | Complete |
| Baseline | Transformer | `src/models/transformer.py` | Complete |
| PINN | PhysicsLoss Module | `src/models/pinn.py` | Complete |
| PINN | 7 Constraint Variants | `src/models/pinn.py` | Complete |
| Advanced | StackedPINN | `src/models/stacked_pinn.py` | Complete |
| Advanced | ResidualPINN | `src/models/stacked_pinn.py` | Complete |

### Physics Constraints Implemented

| Equation | Mathematical Form | Implementation | Learnable Params |
|----------|------------------|----------------|------------------|
| GBM | $dS = \mu S \, dt + \sigma S \, dW$ | MSE on drift residual | None (estimated from data) |
| Black-Scholes | $\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$ | AutoGrad exact derivatives | None |
| Ornstein-Uhlenbeck | $dX = \theta(\mu - X) \, dt + \sigma \, dW$ | MSE on mean-reversion residual | $\theta$ (softplus constrained) |
| Langevin | $dX = -\gamma \nabla U(X) \, dt + \sqrt{2\gamma T} \, dW$ | MSE on momentum residual | $\gamma$, $T$ (softplus constrained) |

### Evaluation Metrics (22 Total)

**Predictive (6):** RMSE, MAE, MAPE, R², MSE, Directional Accuracy

**Financial (16):** Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Total Return, Annualised Return, Volatility, Win Rate, Profit Factor, Information Coefficient, VaR (5%), CVaR (5%), Omega Ratio, Tail Ratio, Skewness, Kurtosis

### Issues Documented and Resolved

| Issue | Severity | Status | Resolution |
|-------|----------|--------|------------|
| Infinity in ResidualPINN metrics | Critical | Fixed | Return clipping, equity floor |
| Max drawdown > -100% | Critical | Fixed | Physical bounds clamping |
| Inconsistent Sharpe annualisation | High | Fixed | Standardised formula |
| Physics gradient domination | High | Fixed | Curriculum learning |
| MPS training failures | Medium | Fixed | Device-specific DataLoader config |
| CUDA/MPS non-determinism | Medium | Mitigated | Comprehensive seeding protocol |

### Test Coverage

| Test Module | Focus | File |
|------------|-------|------|
| Black-Scholes | AutoGrad derivatives, Greeks | `tests/test_black_scholes.py` |
| Models | Instantiation, forward pass | `tests/test_models.py` |
| Uncertainty | MC Dropout, confidence intervals | `tests/test_uncertainty.py` |

---

*Document generated: Day 77 of project timeline*

*Codebase: 42 Python modules, ~1.8MB source code*

*Infrastructure: Docker (3 services), GitHub Actions CI/CD, TimescaleDB + Parquet storage*
