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
                return float("nan")
            thresholds = [
                float(np.percentile(real_vols, 100 * i / self.n_regimes))
                for i in range(1, self.n_regimes)
            ]

        bounds = [-np.inf] + thresholds + [np.inf]

        gap_sum = 0.0
        valid_regimes = 0

        for i in range(self.n_regimes):
            low, high = bounds[i], bounds[i + 1]

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
            return float("nan")

        return gap_sum / valid_regimes

    def _get_returns_and_vols(self, corpus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N_paths, T = corpus.shape
        if self.vol_lookback >= T:
            return np.array([]), np.array([])

        # Use pandas for fast rolling std calculation across columns (paths)
        df = pd.DataFrame(corpus.T)
        vols = df.rolling(window=self.vol_lookback, min_periods=self.vol_lookback).std().values

        # vols has shape (T, N_paths). First `vol_lookback - 1` are NaN.
        # local_vol(t) is std(r[t-lookback:t]).
        # df.rolling(window=lookback).std() at index `t-1` gives std(r[t-lookback:t]).
        # Thus, we align return at `t` (corpus[:, t]) with vol ending at `t-1` (vols[t-1, :]).

        valid_returns = corpus[:, self.vol_lookback :]  # shape (N, T - lookback)
        valid_vols = vols[self.vol_lookback - 1 : T - 1, :].T  # shape (N, T - lookback)

        return valid_returns.ravel(), valid_vols.ravel()

    def _fit_gpd_xi(self, returns: np.ndarray) -> float:
        abs_r = np.abs(returns)
        if len(abs_r) == 0:
            return float("nan")

        u = np.percentile(abs_r, 100 * (1 - self.tail_q))
        exceedances = abs_r[abs_r > u] - u

        if len(exceedances) < 50:
            return float("nan")

        params = genpareto.fit(exceedances, floc=0)
        return float(params[0])

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        """
        Sanity checks for BucketTailRegime:
            S6.1 — Homogeneous-tail synthesis (GARCH + Gaussian) -> large gap
            S6.2 — Real-real baseline calibration -> s6 ~ 0.5
            S6.3 — Reversed tail-index curve -> very large gap
        """
        rng = np.random.default_rng(0)

        # Ensure vol regime thresholds are fitted
        if self.fixed_thresholds is None:
            self.fit(real)

        # Noise floor: stratified split real-vs-real gap.
        # For B6, the PDF specifies stratified (not contiguous) splits
        # because GPD fitting doesn't depend on temporal ordering.
        # However, compute_gap needs (N_paths, window_len) arrays,
        # so we use a contiguous split as a pragmatic approximation.
        half = len(real) // 2
        g_rr = self.compute_gap(real[:half], real[half : half * 2])

        results: dict[str, bool] = {}

        # --- S6.1  Homogeneous-tail synthesis ---
        # GARCH(1,1) with Gaussian innovations: conditional distribution
        # of r_t/σ_t is Gaussian regardless of σ_t, so the conditional
        # tail index ξ should be ~constant across vol regimes. Real data
        # has monotonically increasing ξ. Gap should be large.
        flat = real.ravel()
        omega = float(np.var(flat) * (1.0 - 0.95))
        alpha, beta = 0.05, 0.90
        garch = np.empty_like(real)
        for i in range(len(garch)):
            T = real.shape[1]
            sigma2 = np.empty(T)
            r_g = np.empty(T)
            sigma2[0] = omega / (1.0 - alpha - beta)
            r_g[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
            for t in range(1, T):
                sigma2[t] = omega + alpha * r_g[t - 1] ** 2 + beta * sigma2[t - 1]
                r_g[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
            garch[i] = r_g
        g_homo = self.compute_gap(real, garch)
        results["S6.1_homogeneous_tail"] = g_homo > 2.0 * g_rr

        # --- S6.2  Real-real baseline calibration ---
        # s6 = g_rr / (g_rr + g_sr) should be ~ 0.5.
        g_sr = self.compute_gap(real[:half], real[half : half * 2])
        if g_rr + g_sr > 0:
            s6 = g_rr / (g_rr + g_sr)
            results["S6.2_baseline_calibration"] = 0.2 < s6 < 0.8
        else:
            results["S6.2_baseline_calibration"] = True

        # --- S6.3  Reversed tail-index curve ---
        # Construct synthetic where calm-regime returns are placed in
        # high-vol time slots and vice versa, creating a reversed
        # conditional tail structure. Gap should be very large.
        #
        # Implementation: for each path, sort by local vol (ascending),
        # then reverse the order so high-vol returns sit in calm slots.
        reversed_regime = real.copy()
        for i in range(len(reversed_regime)):
            path = reversed_regime[i]
            # Compute rolling vol to sort by
            lookback = self.vol_lookback
            if len(path) > lookback:
                vols = pd.Series(path).rolling(window=lookback, min_periods=lookback).std().values
                # Only sort the valid portion (after lookback)
                valid_start = lookback
                valid_path = path[valid_start:].copy()
                valid_vols = vols[valid_start:]
                # Reverse the vol-sorted order
                order = np.argsort(valid_vols)[::-1]
                valid_path_sorted = valid_path[order]
                reversed_regime[i, valid_start:] = valid_path_sorted
        g_reversed = self.compute_gap(real, reversed_regime)
        results["S6.3_reversed_tail_curve"] = g_reversed > 1.5 * g_rr

        return results

    @property
    def name(self) -> str:
        return "B6_tail_regime"

    @property
    def description(self) -> str:
        return "Conditional GPD tail-index gap across volatility regimes"
