"""M4 — Regime-conditional tail shape.

Measures whether a generator reproduces how the GPD tail-shape
parameter varies across relative local-volatility regimes.

The metric does not measure:
- regime-specific exceedance frequencies,
- absolute tail thresholds,
- GPD scale parameters, or
- tail asymmetry.

See preprocessing/reasoning.md § "Regime-Conditional Tail Design".
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from fineval.metrics.base import BaseMetric


class RegimeConditionalTails(BaseMetric):
    """Regime-conditional tail shape across volatility quantile bins.

    Each return r_t is labeled using a strictly lagged rolling-volatility
    estimate computed only from returns preceding t within the same
    session. Volatility observations are pooled across tickers and
    divided into self-labeled quantile regimes.

    Within each regime, a GPD is fitted to exceedances of absolute
    returns over a regime-specific high threshold. The resulting feature
    vector contains the GPD shape parameter xi for each regime.

    Args:
        name: Metric label, e.g. "regime_conditional_tails".
        window: Rolling window length in bars for the volatility proxy.
        min_periods: Minimum number of prior observations required for
            the rolling standard deviation.
        n_regimes: Number of volatility quantile bins.
        tail_fraction: Exceedance fraction within each regime. For
            example, 0.05 selects observations above the empirical
            95th percentile of absolute returns.
        regime_weights: Non-negative weights of length n_regimes.
            These are renormalized over the regimes available in both
            feature vectors.

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
            tail_fraction=TAIL_FRACTION,
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
        tail_fraction: float,
        regime_weights: np.ndarray,
    ) -> None:
        super().__init__(name)
        if window < 2:
            raise ValueError("window must be at least 2.")
        if not 2 <= min_periods <= window:
            raise ValueError("min_periods must be between 2 and window.")
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2.")
        if not 0.0 < tail_fraction < 1.0:
            raise ValueError("tail_fraction must be in (0, 1).")
        weights = np.asarray(regime_weights, dtype=float)
        if weights.shape != (n_regimes,):
            raise ValueError(f"regime_weights must have shape ({n_regimes},), got {weights.shape}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("regime_weights must be finite and non-negative.")
        if weights.sum() <= 0.0:
            raise ValueError("At least one regime weight must be positive.")
        self.window = window
        self.min_periods = min_periods
        self.n_regimes = n_regimes
        self.tail_fraction = tail_fraction
        self.regime_weights = weights / weights.sum()

    def _rolling_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strictly lagged rolling std computed within each session.

        For return r_t, the volatility proxy uses only returns up to
        r_{t-1}. The shift is performed separately within every session,
        so neither the previous session nor the current return can
        influence the regime label.
        """
        session = df.index.normalize()
        pieces = [
            group.shift(1).rolling(window=self.window, min_periods=self.min_periods).std()
            for _, group in df.groupby(session)
        ]
        # Concatenation doesn't need to reproduce the original row order:
        # every downstream consumer joins this result back by its
        # (time, ticker) index label, not by row position.
        return pd.concat(pieces)

    @classmethod
    def _fit_gpd_xi(cls, exceedances: np.ndarray) -> float:
        """MLE shape parameter ξ, or NaN if too few exceedances."""
        exc = np.asarray(exceedances, dtype=float)
        exc = exc[np.isfinite(exc) & (exc > 0.0)]
        if exc.size < cls._MIN_EXCEEDANCES:
            return np.nan
        # A constant exceedance sample cannot support a stable
        # shape-parameter estimate.
        if np.ptp(exc) == 0.0:
            return np.nan
        try:
            xi, _, _ = genpareto.fit(exc, floc=0.0)
        except (ValueError, RuntimeError, FloatingPointError):
            return np.nan
        return float(xi) if np.isfinite(xi) else np.nan

    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        """Compute per-regime GPD shape parameters for one panel sample.

        Pipeline:
            1. Compute a strictly lagged rolling-volatility proxy.
            2. Pool valid volatility-return (σ, r) pairs across tickers.
            3. Construct self-labeled volatility quantile regimes.
            4. Select the largest absolute returns within each regime.
            5. Fit a GPD to the excesses and retain its shape parameter ξ.

        Args:
            sample: Panel of deseasonalized returns, DataFrame with
                tickers as columns and a time index.

        Returns:
            Array of length n_regimes. A regime is NaN when it cannot
            support a stable estimate.
        """
        xis = np.full(self.n_regimes, np.nan, dtype=float)
        vol = self._rolling_vol(sample)

        # Pool (volatility, return) pairs across all tickers
        paired = pd.concat(
            [
                vol.stack().rename("vol"),
                sample.stack().rename("ret"),
            ],
            axis=1,
        )
        finite = np.isfinite(paired["vol"].to_numpy()) & np.isfinite(paired["ret"].to_numpy())
        paired = paired.loc[finite].copy()
        if paired.empty:
            return xis

        # Do not silently drop duplicate quantile boundaries. Doing so
        # would change the number and meaning of the regime indices,
        # while regime_weights retains its original fixed length.
        try:
            paired["regime"] = pd.qcut(
                paired["vol"],
                q=self.n_regimes,
                labels=False,
                duplicates="raise",
            )
        except ValueError:
            # The requested quantile partition is not identifiable,
            # usually because the volatility proxy has too many ties.
            return xis

        threshold_quantile = 1.0 - self.tail_fraction
        for regime in range(self.n_regimes):
            cell = paired.loc[paired["regime"] == regime, "ret"].to_numpy(dtype=float)
            if cell.size == 0:
                continue
            abs_returns = np.abs(cell)
            threshold = np.quantile(abs_returns, threshold_quantile)
            exceedances = abs_returns[abs_returns > threshold] - threshold
            xis[regime] = self._fit_gpd_xi(exceedances)
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

        weights = self.regime_weights[valid]
        weight_sum = weights.sum()
        # Relevant only if zero weights are permitted and all positively
        # weighted regimes have been masked.
        if weight_sum <= 0.0:
            return np.nan

        normalized_weights = weights / weight_sum
        absolute_difference = np.abs(fa[valid] - fb[valid])
        return float(np.dot(normalized_weights, absolute_difference))
