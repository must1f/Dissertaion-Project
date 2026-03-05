# Alternative Physics Equations Considered

- **Heston Stochastic Volatility:** Not adopted due to lack of implied vol surface inputs; would require option data and adds two extra latent factors.
- **Variance Gamma / Jump-Diffusion:** Rejected for dissertation scope; jump calibration unstable on daily equity closes.
- **Rough Volatility (RFSV):** Excluded because fractional Brownian motion terms are not supported in current autograd implementation.

Current choice of GBM, OU, Black–Scholes, and Langevin covers diffusion with drift, mean reversion, pricing PDE consistency, and stochastic friction; empirical validation favored OU overwhelmingly, justifying its prominence.
