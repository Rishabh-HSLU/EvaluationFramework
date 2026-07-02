"""
Generate the baseline synthetic datasets: GBM and FIGARCH.

The two baselines anchor opposite ends of the score range:

- **GBM** (Geometric Brownian Motion) is the canonical "wrong"
  generator: Gaussian at every scale, no volatility clustering, no
  regime-conditional tail structure. Every metric should separate it
  cleanly from real data — it anchors the low end the same way the
  real-vs-real baseline anchors the high end.
- **FIGARCH** (Fractionally Integrated GARCH; Baillie, Bollerslev &
  Mikkelsen, 1996) is the positive control for M2. Its ARCH weights
  decay hyperbolically (~k^-(1+d)) rather than exponentially, giving
  genuine long-memory volatility clustering — the one stylized fact a
  short-memory GARCH cannot fake. A model *designed* to satisfy the
  fact M2 measures should score well on M2, validating the metric's
  top end with an independent generator (a resampled-real control
  would be tautological; see scripts/reasoning.md).

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
``FIGARCHBaselineLoader`` (fineval/data/curate.py). No CurationPipeline
pass is needed: the data is curated by construction.

Run from the repository root:

    uv run python -m fineval.scripts.baseline_generation
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fineval.config import SEED

CURATED_DIR = Path(__file__).resolve().parent / "data" / "curated"
REAL_PATH = CURATED_DIR / "real_prices.parquet"
GBM_OUTPUT_PATH = CURATED_DIR / "gbm_prices.parquet"
FIGARCH_OUTPUT_PATH = CURATED_DIR / "figarch_prices.parquet"

# FIGARCH(0, d, 0) parameters. d controls the hyperbolic decay rate of
# the ARCH(inf) weights; the truncation length bounds how far back the
# conditional variance looks (2 full sessions of memory at 1min bars).
FIGARCH_D = 0.45
FIGARCH_TRUNCATION = 780
FIGARCH_BURN_IN = 2000
FIGARCH_SEED = SEED + 1  # decoupled from GBM's stream so the two
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


def _figarch_weights(d: float, truncation: int) -> np.ndarray:
    """ARCH(inf) weights of FIGARCH(0, d, 0), truncated.

    lambda(L) = 1 - (1-L)^d, so lambda_k = -delta_k where delta_k are
    the binomial expansion coefficients of (1-L)^d:

        delta_0 = 1,  delta_k = delta_{k-1} * (k - 1 - d) / k

    All lambda_k are positive for d in (0, 1) and decay hyperbolically
    as k^-(1+d) — the long-memory signature. The untruncated weights
    sum to exactly 1 (IGARCH-like unit persistence); truncation leaves
    the sum slightly below 1, which is what makes the simulated
    variance finite and lets omega set its level.
    """
    delta = np.empty(truncation + 1)
    delta[0] = 1.0
    for k in range(1, truncation + 1):
        delta[k] = delta[k - 1] * (k - 1 - d) / k
    return -delta[1:]


def generate_figarch_prices(
    real_prices: pd.DataFrame,
    d: float = FIGARCH_D,
    truncation: int = FIGARCH_TRUNCATION,
    burn_in: int = FIGARCH_BURN_IN,
    seed: int = FIGARCH_SEED,
) -> pd.DataFrame:
    """Simulate one FIGARCH(0, d, 0) price path per ticker on the real market clock.

    Conditional variance follows the truncated ARCH(inf) form

        sigma^2_t = omega_i + sum_{k=1}^{K} lambda_k * r^2_{t-k}

    with per-ticker omega_i = var_i * (1 - sum(lambda)) so each
    ticker's unconditional return variance matches its real
    counterpart (var_i from overnight-masked real log returns).
    Innovations are standard Gaussian; a burn-in period lets the
    variance process forget its flat initial state before the
    market-clock sample begins. Prices are anchored at each ticker's
    first observed real price and real's NaN mask is copied onto the
    output, exactly as for GBM.

    Args:
        real_prices: Curated wide-format real prices (T, N).
        d: Fractional integration order in (0, 1); higher d means
            more persistent volatility clustering.
        truncation: ARCH(inf) truncation length K in bars.
        burn_in: Pre-sample steps discarded before the clock starts.
        seed: RNG seed; the output is fully reproducible.

    Returns:
        pd.DataFrame of FIGARCH prices with the same index, columns
        and NaN structure as ``real_prices``.
    """
    rng = np.random.default_rng(seed)

    log_returns = _masked_log_returns(real_prices)
    var = log_returns.var().to_numpy()

    lam = _figarch_weights(d, truncation)
    lam_rev = lam[::-1].copy()  # lam_rev @ window == sum_k lambda_k * r2_{t-k}
    omega = var * (1.0 - lam.sum())

    t_len, n_tickers = real_prices.shape
    total = burn_in + t_len
    z = rng.standard_normal((total, n_tickers))

    r2 = np.empty((total + truncation, n_tickers))
    r2[:truncation] = var  # pre-history at the unconditional variance
    returns = np.empty((total, n_tickers))

    for t in tqdm(range(total), desc="FIGARCH simulation"):
        sigma2 = omega + lam_rev @ r2[t : t + truncation]
        r_t = np.sqrt(sigma2) * z[t]
        returns[t] = r_t
        r2[truncation + t] = r_t**2

    increments = returns[burn_in:]
    increments[0, :] = 0.0  # anchor: first bar is the starting price itself

    p0 = real_prices.bfill().iloc[0].to_numpy()
    paths = p0 * np.exp(np.cumsum(increments, axis=0))
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

    figarch_prices = generate_figarch_prices(real_prices)
    assert figarch_prices.shape == real_prices.shape
    assert figarch_prices.isna().to_numpy().sum() == real_prices.isna().to_numpy().sum()
    figarch_prices.to_parquet(FIGARCH_OUTPUT_PATH)
    print(
        f"Saved: {FIGARCH_OUTPUT_PATH} ({figarch_prices.shape}), "
        f"d={FIGARCH_D}, K={FIGARCH_TRUNCATION}, seed={FIGARCH_SEED}"
    )


if __name__ == "__main__":
    main()
