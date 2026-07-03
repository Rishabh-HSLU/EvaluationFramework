"""M6 — Regime-conditional tail heaviness.

Measures whether the generator reproduces the empirical relationship
between local volatility regime and tail thickness. A realistic
generator should produce heavier tails during turbulent periods and
thinner tails during calm periods — a conditional property invisible
to unconditional marginal metrics (M1).

See preprocessing/reasoning.md § "Regime-Conditional Tail Design"
for the full derivation, proxy selection (rolling60 > EWMA),
min_periods validation, and GBM baseline discrimination results.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from fineval.metrics.base import BaseMetric


class RegimeConditionalTails(BaseMetric):
    """Regime-conditional tail heaviness via rolling-volatility quintiles.

    Labels each valid (ticker, time) observation into one of n_regimes
    volatility quintiles using a rolling window std of deseasonalized
    returns, pooled across tickers, self-labeled per sample (no
    anchoring to a reference distribution's cut points). Within each
    regime cell, pools returns across tickers and fits a GPD via MLE
    to the upper-tail exceedances of |returns|, extracting the shape
    parameter ξ. Captures whether tail heaviness responds correctly
    to volatility regime — a property invisible to unconditional
    marginal metrics.

    Args:
        name: Metric label, e.g. "regime_conditional_tails".
        window: Rolling window length in bars for the volatility proxy.
        min_periods: Minimum valid observations required within window
            before a std estimate is emitted; empirically tuned to 20
            (33% of window) to balance estimator stability against
            coverage loss in calm-regime cells.
        n_regimes: Number of volatility quintiles (regime cells).
        tail_quantile: Fixed quantile defining the GPD exceedance
            threshold within each regime cell (e.g. 0.05 = top 5% by
            |returns|). Fixed by config, not derived per-draw, so
            features stay comparable across bootstrap resamples.
        regime_weights: 1-D array of length n_regimes, normalized to
            sum to 1. Controls stress-testing emphasis in
            compute_distance (heavier weight on turbulent quintiles).

    Example::

        from fineval.config import (
            ROLLING_VOL_WINDOW, ROLLING_VOL_MIN_PERIODS,
            N_REGIME_QUINTILES, TAIL_QUANTILE, REGIME_WEIGHTS,
        )

        metric = RegimeConditionalTails(
            name="regime_conditional_tails",
            window=ROLLING_VOL_WINDOW,
            min_periods=ROLLING_VOL_MIN_PERIODS,
            n_regimes=N_REGIME_QUINTILES,
            tail_quantile=TAIL_QUANTILE,
            regime_weights=REGIME_WEIGHTS,
        )
    """

    # Minimum exceedances for a stable GPD MLE fit.
    _MIN_EXCEEDANCES = 500

    def __init__(
        self,
        name: str,
        window: int,
        min_periods: int,
        n_regimes: int,
        tail_quantile: float,
        regime_weights: np.ndarray,
    ) -> None:
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods
        self.n_regimes = n_regimes
        self.tail_quantile = tail_quantile
        self.regime_weights = np.asarray(regime_weights, dtype=float)

    # ── internal helpers ─────────────────────────────────────────────

    def _rolling_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-ticker causal rolling std, confined within each session.

        Grouping by calendar date (not a fixed row stride, since the
        sample includes variable-length half-day sessions) prevents the
        window from reaching into the prior session's returns near the
        open — the same session-boundary discipline enforced in M2.
        """
        session = df.index.normalize()
        # One vectorized rolling per session block (not per column via
        # transform): identical output, ~10x faster on wide panels.
        # Concatenation doesn't need to reproduce the original row order:
        # every downstream consumer joins this result back by its
        # (time, ticker) index label, not by row position.
        pieces = [
            group.rolling(window=self.window, min_periods=self.min_periods).std()
            for _, group in df.groupby(session)
        ]
        return pd.concat(pieces)

    @staticmethod
    def _fit_gpd_xi(exceedances: np.ndarray) -> float:
        """MLE shape parameter ξ, or NaN if too few exceedances."""
        if len(exceedances) < RegimeConditionalTails._MIN_EXCEEDANCES:
            return np.nan
        xi, _, _ = genpareto.fit(exceedances, floc=0)
        return xi

    # ── public interface ─────────────────────────────────────────────

    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        """Compute per-regime GPD shape parameters for one panel sample.

        Pipeline:
            1. Per-ticker rolling std as the volatility proxy.
            2. Pool all valid (σ, r) pairs across tickers.
            3. Self-labeled quintile cut points from pooled σ.
            4. Within each regime cell, fit GPD on the top
               ``tail_quantile`` fraction of |returns|.
            5. Extract ξ per regime.

        Args:
            sample: Panel of deseasonalized returns, DataFrame with
                tickers as columns and a time index.

        Returns:
            1-D ndarray of length n_regimes. A regime cell with too
            few exceedances for stable MLE emits NaN rather than a
            forced estimate.
        """
        vol = self._rolling_vol(sample)

        # Pool (volatility, return) pairs across all tickers
        v = vol.stack()
        r = sample.stack()
        paired = pd.DataFrame({"vol": v, "ret": r}).dropna()

        # Self-labeled quintile boundaries
        paired["regime"] = pd.qcut(paired["vol"], q=self.n_regimes, labels=False, duplicates="drop")

        # Per-regime GPD ξ
        xis = np.full(self.n_regimes, np.nan)
        tail_q_upper = 1.0 - self.tail_quantile  # e.g. 0.95

        for reg in range(self.n_regimes):
            cell = paired.loc[paired["regime"] == reg, "ret"].values
            if len(cell) == 0:
                continue
            abs_ret = np.abs(cell)
            thresh = np.quantile(abs_ret, tail_q_upper)
            exc = abs_ret[abs_ret > thresh] - thresh
            xis[reg] = self._fit_gpd_xi(exc)

        return xis

    def compute_distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        """Weighted masked mean absolute difference between ξ vectors.

        Args:
            fa: Regime ξ vector from sample A.
            fb: Regime ξ vector from sample B.

        Returns:
            Non-negative scalar. NaN if every regime is masked in
            both vectors.
        """
        valid = np.isfinite(fa) & np.isfinite(fb)
        if not valid.any():
            return np.nan

        abs_diff = np.abs(fa[valid] - fb[valid])
        w = self.regime_weights[valid]
        w = w / w.sum()
        return float(np.dot(w, abs_diff))

    def normalize(self, g_rr: np.ndarray, g_sr: np.ndarray) -> float:
        """Ratio-of-means similarity score, matching M1/M2's array convention.

        Args:
            g_rr: Array of real-vs-real self-distances across bootstrap draws.
            g_sr: Array of synthetic-vs-real distances across bootstrap draws.

        Returns:
            Similarity score in [0, 1].
        """
        rr_mean = float(np.nanmean(g_rr))
        sr_mean = float(np.nanmean(g_sr))
        if rr_mean + sr_mean == 0.0:
            return 1.0
        return rr_mean / (rr_mean + sr_mean)
