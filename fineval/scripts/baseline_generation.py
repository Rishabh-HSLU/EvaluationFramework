"""
Generate the baseline synthetic datasets: GBM and FIGARCH.

The two baselines anchor opposite ends of the score range:

- **GBM** (Geometric Brownian Motion) is the canonical "wrong"
  generator: Gaussian at every scale, no volatility clustering, no
  regime-conditional tail structure. Every metric should separate it
  cleanly from real data — it anchors the low end the same way the
  real-vs-real baseline anchors the high end.
- **MSV** (multi-scale stochastic volatility, in the two-timescale
  spirit of Fouque, Papanicolaou & Sircar) is the positive control
  for M2. Log-volatility is the sum of a *slow* factor — long-memory
  fractional Gaussian noise across sessions, constant within each
  session, reproducing the day-level volatility persistence that
  dominates real's within-session |r| ACF — and a *fast* intraday
  OU/AR(1) factor reproducing the short-lag decay. A model *designed*
  to satisfy the fact M2 measures should score well on M2, validating
  the metric's top end with an independent generator (a
  resampled-real control would be tautological; single-factor
  FIGARCH and LMSV attempts decayed far too steeply — all documented
  in scripts/reasoning.md).

Both baselines are deliberately matched to the curated real dataset on
everything *except* the return-generating process:

- Same market clock (the curated real parquet's index).
- Same ticker universe and column order.
- Per-ticker scale calibrated to real's overnight-masked 1-minute log
  returns, so marginal scale is comparable.
- Real's NaN mask imposed on the output, so both sides carry identical
  missingness and no metric can score coverage instead of dynamics.

Output is one wide-format parquet per baseline, identical in shape to
the curated real file, loadable through ``GBMBaselineLoader`` /
``MSVBaselineLoader`` (fineval/data/curate.py). No CurationPipeline
pass is needed: the data is curated by construction.

Run from the repository root:

    uv run python -m fineval.scripts.baseline_generation
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from fineval.config import SEED

CURATED_DIR = Path(__file__).resolve().parent / "data" / "curated"
REAL_PATH = CURATED_DIR / "real_prices.parquet"
GBM_OUTPUT_PATH = CURATED_DIR / "gbm_prices.parquet"
MSV_OUTPUT_PATH = CURATED_DIR / "msv_prices.parquet"

# MSV parameters, selected by sweeping the simulated |r| ACF curve
# against real's (see scripts/reasoning.md, MSV baseline section).
MSV_NU_SLOW = 0.9  # amplitude of the slow (per-session) log-vol factor
MSV_NU_FAST = 0.3  # amplitude of the fast intraday log-vol factor
MSV_TAU_FAST = 20.0  # fast-factor autocorrelation time in minutes
MSV_H_SLOW = 0.90  # Hurst exponent of the across-session fGn factor
MSV_SEED = SEED + 1  # decoupled from GBM's stream so the two
# baselines don't share innovation draws


def _masked_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns with the overnight (09:31 ET) bar masked.

    Mirrors PreprocessingPipeline._compute_log_returns so that GBM
    calibration sees exactly the return population the metrics see.
    """
    log_returns = np.log(prices).diff()
    ny_index = log_returns.index.tz_convert("America/New_York")
    session_start = (ny_index.hour == 9) & (ny_index.minute == 31)
    log_returns.loc[session_start] = np.nan
    log_returns.iloc[0] = np.nan
    return log_returns


def generate_gbm_prices(real_prices: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Simulate one GBM price path per ticker on the real market clock.

    Each ticker's 1-minute log-return increments are drawn i.i.d.
    Normal(mu_i, sigma_i), with mu_i and sigma_i estimated from that
    ticker's overnight-masked real log returns. Estimating mu_i on log
    returns directly absorbs the usual -sigma^2/2 Ito correction, so
    exp(cumsum(increments)) is a discretely-sampled GBM path. Paths are
    anchored at each ticker's first observed real price, and real's NaN
    mask is copied onto the output.

    Args:
        real_prices: Curated wide-format real prices (T, N).
        seed: RNG seed; the output is fully reproducible.

    Returns:
        pd.DataFrame of GBM prices with the same index, columns and
        NaN structure as ``real_prices``.
    """
    rng = np.random.default_rng(seed)

    log_returns = _masked_log_returns(real_prices)
    mu = log_returns.mean().to_numpy()
    sigma = log_returns.std().to_numpy()

    t_len, n_tickers = real_prices.shape
    increments = mu + sigma * rng.standard_normal((t_len, n_tickers))
    increments[0, :] = 0.0  # anchor: first bar is the starting price itself

    p0 = real_prices.bfill().iloc[0].to_numpy()  # first observed price per ticker
    paths = p0 * np.exp(np.cumsum(increments, axis=0))
    paths[real_prices.isna().to_numpy()] = np.nan

    return pd.DataFrame(paths, index=real_prices.index, columns=real_prices.columns)


def _fgn_eigenvalues(h: float, n_steps: int) -> np.ndarray:
    """Circulant-embedding eigenvalues for exact fGn sampling.

    Builds the autocovariance of fractional Gaussian noise,

        rho(k) = 0.5 * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}),

    embeds it in a circulant of size 2*n_steps, and returns the FFT
    eigenvalues (Davies & Harte, 1987). Tiny negative eigenvalues from
    floating-point round-off are clipped to zero.
    """
    k = np.arange(n_steps + 1, dtype=float)
    rho = 0.5 * ((k + 1) ** (2 * h) - 2 * k ** (2 * h) + np.abs(k - 1) ** (2 * h))
    row = np.concatenate([rho, rho[-2:0:-1]])  # length 2 * n_steps
    return np.clip(np.fft.fft(row).real, 0.0, None)


def _sample_fgn(rng: np.random.Generator, eigenvalues: np.ndarray, n_paths: int) -> np.ndarray:
    """Draw exact fGn paths via circulant embedding, two per FFT.

    With w_k = sqrt(lambda_k / m) * (A_k + i B_k) for i.i.d. standard
    normal A, B and X = fft(w), the real and imaginary parts of the
    first n entries of X are two independent fGn samples with the
    embedded autocovariance (Dieker, 2004).

    Returns:
        (n_steps, n_paths) array of standard-normal-marginal fGn.
    """
    m = len(eigenvalues)
    n_steps = m // 2
    scale = np.sqrt(eigenvalues / m)
    paths = np.empty((n_steps, n_paths))
    for j in range(0, n_paths, 2):
        w = scale * (rng.standard_normal(m) + 1j * rng.standard_normal(m))
        x = np.fft.fft(w)
        paths[:, j] = x.real[:n_steps]
        if j + 1 < n_paths:
            paths[:, j + 1] = x.imag[:n_steps]
    return paths


def generate_msv_prices(
    real_prices: pd.DataFrame,
    nu_slow: float = MSV_NU_SLOW,
    nu_fast: float = MSV_NU_FAST,
    tau_fast: float = MSV_TAU_FAST,
    h_slow: float = MSV_H_SLOW,
    seed: int = MSV_SEED,
) -> pd.DataFrame:
    """Simulate one multi-scale SV price path per ticker on the real market clock.

    Two-factor log-volatility in the two-timescale spirit of Fouque,
    Papanicolaou & Sircar:

        r_t = c_i * exp(nu_slow * s_{d(t)} + nu_fast * f_t) * z_t

    - s_d is the *slow* factor: exact fractional Gaussian noise with
      Hurst exponent ``h_slow`` across the sample's sessions, held
      constant within each session. It reproduces the day-level
      volatility persistence that dominates real's within-session |r|
      ACF (a volatile day stays volatile all session).
    - f_t is the *fast* factor: a stationary AR(1)/OU process in
      minutes with autocorrelation time ``tau_fast``, reproducing the
      short-lag ACF decay.
    - z_t is i.i.d. standard Gaussian, and c_i rescales each ticker so
      its simulated return standard deviation exactly matches its real
      counterpart (from overnight-masked real log returns).

    Both factors are stationary and sampled exactly, so no burn-in is
    needed. Prices are anchored at each ticker's first observed real
    price and real's NaN mask is copied onto the output, exactly as
    for GBM.

    Args:
        real_prices: Curated wide-format real prices (T, N).
        nu_slow: Amplitude of the per-session log-vol factor.
        nu_fast: Amplitude of the intraday log-vol factor.
        tau_fast: Fast-factor autocorrelation time in minutes.
        h_slow: Hurst exponent of the across-session fGn factor.
        seed: RNG seed; the output is fully reproducible.

    Returns:
        pd.DataFrame of MSV prices with the same index, columns and
        NaN structure as ``real_prices``.
    """
    rng = np.random.default_rng(seed)

    log_returns = _masked_log_returns(real_prices)
    target_std = log_returns.std().to_numpy()

    t_len, n_tickers = real_prices.shape
    day_index = pd.factorize(real_prices.index.normalize())[0]
    n_days = int(day_index[-1]) + 1

    # Slow factor: long-memory fGn across sessions, constant within each
    slow = _sample_fgn(rng, _fgn_eigenvalues(h_slow, n_days), n_tickers)[day_index]

    # Fast factor: stationary AR(1) with unit marginal variance
    phi = np.exp(-1.0 / tau_fast)
    shocks = rng.standard_normal((t_len, n_tickers)) * np.sqrt(1.0 - phi**2)
    shocks[0] /= np.sqrt(1.0 - phi**2)  # draw x_0 from the stationary law
    fast = lfilter([1.0], [1.0, -phi], shocks, axis=0)

    log_vol = nu_slow * slow + nu_fast * fast
    returns = np.exp(log_vol) * rng.standard_normal((t_len, n_tickers))
    returns *= target_std / returns.std(axis=0)

    returns[0, :] = 0.0  # anchor: first bar is the starting price itself

    p0 = real_prices.bfill().iloc[0].to_numpy()
    paths = p0 * np.exp(np.cumsum(returns, axis=0))
    paths[real_prices.isna().to_numpy()] = np.nan

    return pd.DataFrame(paths, index=real_prices.index, columns=real_prices.columns)


def main() -> None:
    print(f"Loading curated real prices: {REAL_PATH}")
    real_prices = pd.read_parquet(REAL_PATH)
    print(f"Real shape: {real_prices.shape}")

    gbm_prices = generate_gbm_prices(real_prices, seed=SEED)
    assert gbm_prices.shape == real_prices.shape
    assert gbm_prices.isna().to_numpy().sum() == real_prices.isna().to_numpy().sum()
    gbm_prices.to_parquet(GBM_OUTPUT_PATH)
    print(f"Saved: {GBM_OUTPUT_PATH} ({gbm_prices.shape}), seed={SEED}")

    msv_prices = generate_msv_prices(real_prices)
    assert msv_prices.shape == real_prices.shape
    assert msv_prices.isna().to_numpy().sum() == real_prices.isna().to_numpy().sum()
    msv_prices.to_parquet(MSV_OUTPUT_PATH)
    print(
        f"Saved: {MSV_OUTPUT_PATH} ({msv_prices.shape}), "
        f"nu_slow={MSV_NU_SLOW}, nu_fast={MSV_NU_FAST}, "
        f"tau={MSV_TAU_FAST}, H={MSV_H_SLOW}, seed={MSV_SEED}"
    )


if __name__ == "__main__":
    main()
