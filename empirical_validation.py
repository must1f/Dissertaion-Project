#!/usr/bin/env python3
"""
Physics Equation Suitability Analysis for Financial Time Series

Empirically validates whether GBM, OU, and Langevin equations are appropriate
for modeling S&P 500 stock price dynamics.

Tests:
1. GBM (Geometric Brownian Motion)
   - Normality of log-returns
   - Constant volatility assumption
   - Independence of returns

2. OU (Ornstein-Uhlenbeck)
   - Stationarity (ADF test)
   - Mean reversion (half-life)
   - Appropriate for mean-reverting assets

3. Sector-specific analysis
   - Compare equation fit across sectors (Tech, Utilities, Finance, Consumer)

Outputs:
- Statistical test results (CSV)
- Suitability scores for each equation-stock pair
- Sector-level aggregated statistics
- Visualization figures for dissertation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pathlib import Path
import json
from typing import Dict, Tuple, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhysicsEquationValidator:
    """Empirically validate physics equations for financial time series"""

    def __init__(self, data_dir: Path = Path("data/parquet")):
        """
        Initialize validator

        Args:
            data_dir: Directory containing stock price data (parquet files)
        """
        self.data_dir = data_dir
        self.results = []

        # Sector classifications (S&P 500 examples)
        self.sectors = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Utilities': ['DUK', 'SO', 'NEE', 'D', 'AEP'],
            'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Consumer': ['WMT', 'PG', 'KO', 'MCD', 'NKE'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
        }

    def load_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Load historical price data for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with price data
        """
        parquet_path = self.data_dir / f"{ticker}.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {len(df)} rows for {ticker}")
            return df
        else:
            logger.warning(f"Data not found for {ticker}")
            return None

    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute log returns

        Args:
            prices: Price series

        Returns:
            Log returns
        """
        return np.log(prices / prices.shift(1)).dropna()

    def test_gbm_normality(self, returns: pd.Series) -> Dict:
        """
        Test if log-returns are normally distributed (GBM assumption)

        Args:
            returns: Log return series

        Returns:
            Dictionary with normality test results
        """
        # Shapiro-Wilk test for normality
        stat_sw, p_sw = stats.shapiro(returns)

        # Jarque-Bera test
        stat_jb, p_jb = stats.jarque_bera(returns)

        # Anderson-Darling test
        result_ad = stats.anderson(returns, dist='norm')

        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Q-Q plot correlation
        qq = stats.probplot(returns, dist='norm')
        qq_corr = np.corrcoef(qq[0][0], qq[0][1])[0, 1]

        result = {
            'test': 'GBM_normality',
            'shapiro_stat': stat_sw,
            'shapiro_pvalue': p_sw,
            'jarque_bera_stat': stat_jb,
            'jarque_bera_pvalue': p_jb,
            'anderson_darling_stat': result_ad.statistic,
            'anderson_critical_5pct': result_ad.critical_values[2],  # 5% level
            'skewness': skewness,
            'kurtosis': kurtosis,
            'qq_correlation': qq_corr,
            'is_normal': p_sw > 0.05 and p_jb > 0.05  # Conservative: both tests
        }

        return result

    def test_gbm_constant_volatility(self, returns: pd.Series, window: int = 20) -> Dict:
        """
        Test if volatility is constant (GBM assumption)

        Args:
            returns: Log return series
            window: Rolling window for volatility calculation

        Returns:
            Dictionary with volatility stability test results
        """
        # Rolling volatility
        rolling_vol = returns.rolling(window).std()

        # Remove NaN values
        rolling_vol_clean = rolling_vol.dropna()

        # Coefficient of variation of volatility
        mean_vol = rolling_vol_clean.mean()
        std_vol = rolling_vol_clean.std()
        cv_vol = std_vol / mean_vol if mean_vol != 0 else np.inf

        # Test for heteroskedasticity using Breusch-Pagan test
        # Regress squared returns on time
        time_index = np.arange(len(returns))
        X = sm.add_constant(time_index)
        y = returns ** 2

        try:
            model = sm.OLS(y, X).fit()
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
        except:
            bp_stat, bp_pvalue = np.nan, np.nan

        result = {
            'test': 'GBM_constant_volatility',
            'mean_volatility': mean_vol,
            'std_volatility': std_vol,
            'coefficient_of_variation': cv_vol,
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_pvalue': bp_pvalue,
            'is_constant_vol': cv_vol < 0.5  # CV < 0.5 suggests reasonable stability
        }

        return result

    def test_ou_stationarity(self, returns: pd.Series) -> Dict:
        """
        Test if series is stationary (OU process assumption)

        Args:
            returns: Return series (or price level for OU)

        Returns:
            Dictionary with stationarity test results
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(returns, autolag='AIC')

        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical = adf_result[4]

        result = {
            'test': 'OU_stationarity',
            'adf_statistic': adf_stat,
            'adf_pvalue': adf_pvalue,
            'adf_critical_1pct': adf_critical['1%'],
            'adf_critical_5pct': adf_critical['5%'],
            'adf_critical_10pct': adf_critical['10%'],
            'is_stationary': adf_pvalue < 0.05  # Reject null (unit root) at 5% level
        }

        return result

    def test_ou_mean_reversion(self, returns: pd.Series) -> Dict:
        """
        Test for mean reversion and estimate half-life

        Args:
            returns: Return series

        Returns:
            Dictionary with mean reversion test results
        """
        # AR(1) model: r_t = α + β*r_{t-1} + ε_t
        # If β < 1 and significant, suggests mean reversion

        lagged_returns = returns.shift(1).dropna()
        current_returns = returns[1:]

        # Ensure same length
        min_len = min(len(lagged_returns), len(current_returns))
        lagged_returns = lagged_returns[-min_len:]
        current_returns = current_returns[-min_len:]

        # Regression
        X = sm.add_constant(lagged_returns)
        y = current_returns

        try:
            model = sm.OLS(y, X).fit()

            alpha = model.params[0]
            beta = model.params[1]
            beta_pvalue = model.pvalues[1]

            # Half-life of mean reversion
            # If β < 1: half_life = -ln(2) / ln(β)
            if 0 < beta < 1:
                half_life = -np.log(2) / np.log(beta)
            else:
                half_life = np.inf

            # Mean reversion speed (OU parameter θ)
            # θ ≈ -ln(β) / Δt (assuming Δt = 1 day)
            if beta > 0:
                theta = -np.log(beta)
            else:
                theta = np.nan

        except:
            alpha, beta, beta_pvalue, half_life, theta = np.nan, np.nan, np.nan, np.nan, np.nan

        result = {
            'test': 'OU_mean_reversion',
            'ar1_alpha': alpha,
            'ar1_beta': beta,
            'ar1_beta_pvalue': beta_pvalue,
            'half_life_days': half_life,
            'theta_estimate': theta,
            'has_mean_reversion': (beta < 1 and beta_pvalue < 0.05) if not np.isnan(beta_pvalue) else False
        }

        return result

    def test_independence(self, returns: pd.Series, max_lag: int = 20) -> Dict:
        """
        Test for independence of returns (GBM assumption)

        Args:
            returns: Return series
            max_lag: Maximum lag for autocorrelation test

        Returns:
            Dictionary with independence test results
        """
        # Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_result = acorr_ljungbox(returns, lags=[max_lag], return_df=True)

        lb_stat = lb_result['lb_stat'].values[0]
        lb_pvalue = lb_result['lb_pvalue'].values[0]

        # Autocorrelation at lag 1
        acf_lag1 = returns.autocorr(lag=1)

        result = {
            'test': 'independence',
            'ljung_box_stat': lb_stat,
            'ljung_box_pvalue': lb_pvalue,
            'acf_lag1': acf_lag1,
            'is_independent': lb_pvalue > 0.05  # Fail to reject independence
        }

        return result

    def comprehensive_validation(self, ticker: str) -> Dict:
        """
        Run all validation tests for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with all test results
        """
        logger.info(f"Validating physics equations for {ticker}...")

        # Load data
        df = self.load_price_data(ticker)

        if df is None or 'close' not in df.columns:
            logger.error(f"Failed to load data for {ticker}")
            return None

        # Compute returns
        prices = df['close']
        returns = self.compute_returns(prices)

        if len(returns) < 100:
            logger.warning(f"Insufficient data for {ticker} ({len(returns)} points)")
            return None

        # Run all tests
        result = {
            'ticker': ticker,
            'n_observations': len(returns),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'sharpe_annualized': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        }

        # GBM tests
        gbm_normality = self.test_gbm_normality(returns)
        gbm_volatility = self.test_gbm_constant_volatility(returns)
        independence = self.test_independence(returns)

        # OU tests
        ou_stationarity = self.test_ou_stationarity(returns)
        ou_mean_reversion = self.test_ou_mean_reversion(returns)

        # Combine results
        result.update(gbm_normality)
        result.update(gbm_volatility)
        result.update(independence)
        result.update(ou_stationarity)
        result.update(ou_mean_reversion)

        # Suitability scores (0-100)
        # GBM suitability: based on normality, constant vol, independence
        gbm_score = 0
        if gbm_normality['is_normal']:
            gbm_score += 40
        gbm_score += max(0, 40 * (1 - gbm_normality['kurtosis'] / 10))  # Penalize excess kurtosis
        if gbm_volatility['is_constant_vol']:
            gbm_score += 20

        # OU suitability: based on stationarity and mean reversion
        ou_score = 0
        if ou_stationarity['is_stationary']:
            ou_score += 50
        if ou_mean_reversion['has_mean_reversion']:
            ou_score += 50

        result['gbm_suitability_score'] = max(0, min(100, gbm_score))
        result['ou_suitability_score'] = max(0, min(100, ou_score))

        # Overall recommendation
        if gbm_score > ou_score:
            result['recommended_equation'] = 'GBM'
        elif ou_score > gbm_score:
            result['recommended_equation'] = 'OU'
        else:
            result['recommended_equation'] = 'Mixed'

        logger.info(f"{ticker}: GBM={gbm_score:.0f}, OU={ou_score:.0f}, Rec={result['recommended_equation']}")

        return result

    def validate_all_tickers(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Run validation for all tickers

        Args:
            tickers: List of ticker symbols (None = use all sectors)

        Returns:
            DataFrame with results for all tickers
        """
        if tickers is None:
            # Use all tickers from sector definitions
            tickers = []
            for sector_tickers in self.sectors.values():
                tickers.extend(sector_tickers)

        logger.info(f"Validating {len(tickers)} tickers...")

        results = []
        for ticker in tickers:
            try:
                result = self.comprehensive_validation(ticker)
                if result:
                    # Add sector
                    for sector, sector_tickers in self.sectors.items():
                        if ticker in sector_tickers:
                            result['sector'] = sector
                            break
                    results.append(result)
            except Exception as e:
                logger.error(f"Error validating {ticker}: {e}")

        df = pd.DataFrame(results)

        logger.info(f"Validation complete: {len(df)} tickers processed")

        return df

    def generate_sector_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results by sector

        Args:
            df: Results DataFrame

        Returns:
            Sector-level summary DataFrame
        """
        if 'sector' not in df.columns:
            logger.warning("No sector information in results")
            return None

        summary = df.groupby('sector').agg({
            'gbm_suitability_score': ['mean', 'std'],
            'ou_suitability_score': ['mean', 'std'],
            'is_normal': 'mean',
            'is_constant_vol': 'mean',
            'is_stationary': 'mean',
            'has_mean_reversion': 'mean',
            'ticker': 'count'
        })

        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={'ticker_count': 'n_tickers'})

        return summary

    def save_results(self, df: pd.DataFrame, output_dir: Path = Path("results")):
        """
        Save validation results

        Args:
            df: Results DataFrame
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        csv_path = output_dir / "physics_equation_validation.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # Save sector summary
        sector_summary = self.generate_sector_summary(df)
        if sector_summary is not None:
            summary_path = output_dir / "physics_equation_sector_summary.csv"
            sector_summary.to_csv(summary_path)
            logger.info(f"Sector summary saved to {summary_path}")

        # Save JSON summary
        summary_dict = {
            'n_tickers': len(df),
            'gbm_mean_score': df['gbm_suitability_score'].mean(),
            'ou_mean_score': df['ou_suitability_score'].mean(),
            'recommendations': df['recommended_equation'].value_counts().to_dict(),
            'normality_pct': (df['is_normal'].sum() / len(df)) * 100,
            'constant_vol_pct': (df['is_constant_vol'].sum() / len(df)) * 100,
            'stationary_pct': (df['is_stationary'].sum() / len(df)) * 100,
            'mean_reversion_pct': (df['has_mean_reversion'].sum() / len(df)) * 100
        }

        json_path = output_dir / "physics_equation_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(f"Summary saved to {json_path}")

    def visualize_results(self, df: pd.DataFrame, output_dir: Path = Path("dissertation/figures")):
        """
        Create visualization figures for dissertation

        Args:
            df: Results DataFrame
            output_dir: Output directory for figures
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: Suitability scores by sector
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if 'sector' in df.columns:
            sector_summary = df.groupby('sector').agg({
                'gbm_suitability_score': 'mean',
                'ou_suitability_score': 'mean'
            }).reset_index()

            axes[0].bar(sector_summary['sector'], sector_summary['gbm_suitability_score'], color='#3498db')
            axes[0].set_ylabel('GBM Suitability Score')
            axes[0].set_xlabel('Sector')
            axes[0].set_title('GBM Suitability by Sector')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)

            axes[1].bar(sector_summary['sector'], sector_summary['ou_suitability_score'], color='#e74c3c')
            axes[1].set_ylabel('OU Suitability Score')
            axes[1].set_xlabel('Sector')
            axes[1].set_title('OU Suitability by Sector')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "physics_suitability_by_sector.pdf", dpi=300, bbox_inches='tight')
        logger.info(f"Sector suitability chart saved")
        plt.close()

        # Figure 2: Test pass rates
        fig, ax = plt.subplots(figsize=(10, 6))

        test_pass_rates = {
            'Normal Returns': (df['is_normal'].sum() / len(df)) * 100,
            'Constant Volatility': (df['is_constant_vol'].sum() / len(df)) * 100,
            'Stationary': (df['is_stationary'].sum() / len(df)) * 100,
            'Mean Reverting': (df['has_mean_reversion'].sum() / len(df)) * 100
        }

        colors = ['#27ae60' if rate > 50 else '#e74c3c' for rate in test_pass_rates.values()]

        ax.barh(list(test_pass_rates.keys()), list(test_pass_rates.values()), color=colors)
        ax.set_xlabel('Pass Rate (%)')
        ax.set_title('Physics Equation Assumptions: Test Pass Rates')
        ax.axvline(x=50, color='black', linestyle='--', linewidth=0.8, label='50% threshold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "physics_test_pass_rates.pdf", dpi=300, bbox_inches='tight')
        logger.info(f"Test pass rates chart saved")
        plt.close()


def main():
    """Run physics equation validation"""

    print("=" * 60)
    print("Physics Equation Suitability Analysis")
    print("=" * 60)

    # Initialize validator
    validator = PhysicsEquationValidator()

    # Validate all tickers
    results_df = validator.validate_all_tickers()

    if results_df.empty:
        print("\n⚠️  No results generated. Check data availability.")
        return

    # Display summary
    print(f"\n📊 Validation Summary")
    print(f"   Tickers analyzed: {len(results_df)}")
    print(f"   GBM avg score: {results_df['gbm_suitability_score'].mean():.1f}/100")
    print(f"   OU avg score: {results_df['ou_suitability_score'].mean():.1f}/100")
    print(f"\n   Normality: {(results_df['is_normal'].sum() / len(results_df)) * 100:.1f}% pass")
    print(f"   Constant Vol: {(results_df['is_constant_vol'].sum() / len(results_df)) * 100:.1f}% pass")
    print(f"   Stationarity: {(results_df['is_stationary'].sum() / len(results_df)) * 100:.1f}% pass")
    print(f"   Mean Reversion: {(results_df['has_mean_reversion'].sum() / len(results_df)) * 100:.1f}% pass")

    # Recommendations
    print(f"\n📌 Equation Recommendations:")
    for eq, count in results_df['recommended_equation'].value_counts().items():
        print(f"   {eq}: {count} tickers ({count/len(results_df)*100:.1f}%)")

    # Save results
    validator.save_results(results_df)

    # Generate visualizations
    validator.visualize_results(results_df)

    print(f"\n✅ Analysis complete!")
    print(f"   Results: results/physics_equation_validation.csv")
    print(f"   Figures: dissertation/figures/")


if __name__ == "__main__":
    main()
