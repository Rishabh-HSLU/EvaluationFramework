"""
b6_tail_regime.py
=================
Bucket 6 — Conditional GPD Tail-Index Curve.

Measures whether the tail index of returns depends on the volatility regime in
the same way as real data. In real markets: high-vol regimes have heavier
tails than low-vol regimes.

Aggregation strategy
--------------------
Distributional. Pool returns across all N paths before computing vol regimes and GPD fits.
The vol regime boundaries are computed ONCE on the full real corpus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from ..bucket import Bucket


class BucketTailRegime(Bucket):
    """
    Conditional GPD Tail-Index Curve Gap.

    Parameters
    ----------
    n_regimes : int, default 3
        Number of volatility regimes (e.g. low / mid / high vol terciles).
    tail_q : float, default 0.05
        Fit GPD to top 5% of |returns| in each regime.
    vol_lookback : int, default 60
        Rolling window for local vol estimate.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        tail_q: float = 0.05,
        vol_lookback: int = 60,
    ) -> None:
        self.n_regimes = n_regimes
        self.tail_q = tail_q
        self.vol_lookback = vol_lookback
        self.fixed_thresholds = None

    def fit(self, real_corpus: np.ndarray) -> None:
        """
        Precompute vol regime boundaries on the full real corpus.
        Must be called ONCE before synthetic evaluation loop.
        """
        _, real_vols = self._get_returns_and_vols(real_corpus)
        if len(real_vols) == 0:
            raise ValueError("Real corpus too small to compute volatility regimes.")
        
        self.fixed_thresholds = [
            float(np.percentile(real_vols, 100 * i / self.n_regimes))
            for i in range(1, self.n_regimes)
        ]

    def compute_gap(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        self._validate_input(real, synthetic)

        real_returns, real_vols = self._get_returns_and_vols(real)
        syn_returns, syn_vols = self._get_returns_and_vols(synthetic)

        if self.fixed_thresholds is not None:
            thresholds = self.fixed_thresholds
        else:
            if len(real_vols) == 0:
                return float('nan')
            thresholds = [
                float(np.percentile(real_vols, 100 * i / self.n_regimes))
                for i in range(1, self.n_regimes)
            ]

        bounds = [-np.inf] + thresholds + [np.inf]
        
        gap_sum = 0.0
        valid_regimes = 0

        for i in range(self.n_regimes):
            low, high = bounds[i], bounds[i+1]

            mask_real = (real_vols >= low) & (real_vols < high)
            mask_syn = (syn_vols >= low) & (syn_vols < high)

            ret_real = real_returns[mask_real]
            ret_syn = syn_returns[mask_syn]

            xi_real = self._fit_gpd_xi(ret_real)
            xi_syn = self._fit_gpd_xi(ret_syn)

            if not np.isnan(xi_real) and not np.isnan(xi_syn):
                gap_sum += abs(xi_real - xi_syn)
                valid_regimes += 1

        if valid_regimes == 0:
            return float('nan')

        return gap_sum / valid_regimes

    def _get_returns_and_vols(self, corpus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N_paths, T = corpus.shape
        if T <= self.vol_lookback:
            return np.array([]), np.array([])

        # Use pandas for fast rolling std calculation across columns (paths)
        df = pd.DataFrame(corpus.T)
        vols = df.rolling(window=self.vol_lookback, min_periods=self.vol_lookback).std().values

        # vols has shape (T, N_paths). First `vol_lookback - 1` are NaN.
        # local_vol(t) is std(r[t-lookback:t]).
        # df.rolling(window=lookback).std() at index `t-1` gives std(r[t-lookback:t]).
        # Thus, we align return at `t` (corpus[:, t]) with vol ending at `t-1` (vols[t-1, :]).
        
        valid_returns = corpus[:, self.vol_lookback:]            # shape (N, T - lookback)
        valid_vols = vols[self.vol_lookback - 1 : T - 1, :].T    # shape (N, T - lookback)

        return valid_returns.ravel(), valid_vols.ravel()

    def _fit_gpd_xi(self, returns: np.ndarray) -> float:
        abs_r = np.abs(returns)
        if len(abs_r) == 0:
            return float('nan')

        u = np.percentile(abs_r, 100 * (1 - self.tail_q))
        exceedances = abs_r[abs_r > u] - u

        if len(exceedances) < 50:
            return float('nan')

        params = genpareto.fit(exceedances, floc=0)
        return float(params[0])

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        raise NotImplementedError("Sanity checks not yet implemented for BucketTailRegime.")

    @property
    def name(self) -> str:
        return "B6_tail_regime"

    @property
    def description(self) -> str:
        return "Conditional GPD tail-index gap across volatility regimes"
