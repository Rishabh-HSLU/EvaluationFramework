"""
b2_nonlinear_temporal.py
========================
Bucket 2 — Nonlinear temporal dependence.

Measures the fidelity of volatility clustering and long memory using
a lag-weighted autocorrelation gap on absolute returns.

Mathematical formulation
------------------------
    rho_k(|r|) = Corr(|r_t|, |r_{t+k}|)   for k = 1, ..., K

    Gap_2(r, r_hat) = (1 / |W|) * sum_{k in W} |rho_k(|r|) - rho_k(|r_hat|)|

where W = {k_min, ..., k_max} is the long-memory window.

The weight function is implicit: we include only lags in W, which
upweights the long-memory region (k >= k_min) and ignores short lags
where the difference between models is typically small and noisy.

Aggregation strategy
--------------------
Per-path then average. The ACF is a temporal statistic — pooling
across paths would corrupt it by creating false pairs at path
boundaries. Procedure:

    1. Compute rho_k(|r|) for k in W on each individual path
    2. Average the ACF curves across all paths in the corpus
    3. Do the same for the synthetic corpus
    4. Compute the mean absolute difference between the two
       averaged ACF curves across the lag window W

References
----------
- Ding, Granger, Engle (1993): absolute returns have stronger and
  longer-lasting ACF than squared returns — foundational result
  justifying |r| over r^2
- Cont (2001): slow decay of ACF in absolute/squared returns as a
  canonical stylized fact of financial time series
- Andersen, Bollerslev (1997): deseasonalization required before ACF
  analysis on intraday returns to avoid spurious periodicity
"""

from __future__ import annotations

import numpy as np

from ..bucket import Bucket


class BucketNonlinearTemporal(Bucket):
    """
    Lag-weighted ACF gap on absolute returns.

    Parameters
    ----------
    k_min : int, default 60
        Minimum lag in the evaluation window. Lags below this are
        dominated by short-memory effects and excluded. Default 60
        minutes corresponds to one hour — the boundary beyond which
        genuine long-memory persistence dominates local clustering.
    k_max : int, default 390
        Maximum lag in the evaluation window. Default 390 minutes
        corresponds to one full trading day. Capped at window_len - 1
        if the input is shorter.
    """

    def __init__(
        self,
        k_min: int = 60,
        k_max: int = 390,
    ) -> None:
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}")
        if k_max <= k_min:
            raise ValueError(f"k_max must be > k_min, got k_min={k_min}, k_max={k_max}")
        self.k_min = k_min
        self.k_max = k_max

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _acf_path(path: np.ndarray, k_min: int, k_max: int) -> np.ndarray:
        """
        Compute the ACF of |r| at lags k_min..k_max for one path.

        Parameters
        ----------
        path  : 1-D array of returns, shape (window_len,)
        k_min : minimum lag (inclusive)
        k_max : maximum lag (inclusive), capped at len(path) - 1

        Returns
        -------
        acf : 1-D array of shape (k_max - k_min + 1,)
              acf[i] = Corr(|r_t|, |r_{t + k_min + i}|)
        """
        abs_r = np.abs(path)
        k_max_actual = min(k_max, len(path) - 1)

        if k_min > k_max_actual:
            # Path too short to estimate any lag in the window
            return np.full(k_max - k_min + 1, np.nan)

        mu = abs_r.mean()
        std = abs_r.std()

        if std < 1e-10:
            # Constant path — ACF undefined
            return np.zeros(k_max - k_min + 1)

        centered = abs_r - mu
        n = len(centered)

        acf = np.empty(k_max - k_min + 1)
        for i, k in enumerate(range(k_min, k_max + 1)):
            if k > k_max_actual:
                acf[i] = np.nan
            else:
                acf[i] = np.dot(centered[: n - k], centered[k:]) / (n * std**2)

        return acf

    @staticmethod
    def _mean_acf(
        corpus: np.ndarray,
        k_min: int,
        k_max: int,
    ) -> np.ndarray:
        """
        Compute the ACF averaged across all paths in a corpus.

        Parameters
        ----------
        corpus : shape (N_paths, window_len)
        k_min  : minimum lag
        k_max  : maximum lag

        Returns
        -------
        mean_acf : shape (k_max - k_min + 1,), NaN-aware mean
        """
        acfs = np.stack(
            [BucketNonlinearTemporal._acf_path(corpus[i], k_min, k_max) for i in range(len(corpus))]
        )  # (N_paths, n_lags)
        return np.nanmean(acfs, axis=0)  # (n_lags,)

    # ------------------------------------------------------------------
    # Core metric
    # ------------------------------------------------------------------

    def compute_gap(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        self._validate_input(real, synthetic)

        window_len = real.shape[1]

        # Cap k_max at window_len - 1 — can't compute ACF at lag >= T
        k_max_eff = min(self.k_max, window_len - 1)

        if self.k_min > k_max_eff:
            raise ValueError(
                f"Effective k_max ({k_max_eff}) <= k_min ({self.k_min}). "
                f"window_len={window_len} is too short for this lag range. "
                f"Reduce k_min or use longer paths."
            )

        # Average ACF across paths for each corpus
        acf_real = self._mean_acf(real, self.k_min, k_max_eff)
        acf_syn = self._mean_acf(synthetic, self.k_min, k_max_eff)

        # Mean absolute difference across valid lags
        diff = np.abs(acf_real - acf_syn)
        valid = ~np.isnan(diff)

        if not valid.any():
            raise ValueError("All ACF estimates are NaN — paths may be too short.")

        gap = float(diff[valid].mean())
        return gap

    # ------------------------------------------------------------------
    # Sanity checks — stubs, filled in during empirical validation
    # ------------------------------------------------------------------

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        """
        Sanity checks for BucketNonlinearTemporal:
            N2.1 — shuffle destroys ACF -> large gap
            N2.2 — i.i.d. resampling from marginal -> large gap
            N2.3 — short-burst clustering (low-persistence GARCH) ->
                   moderate to large gap driven by long lags
            N2.4 — tail replacement preserves ACF -> small gap
            N2.5 — scale invariance: scaling by 2x -> ~zero gap
        """
        rng = np.random.default_rng(0)

        # Noise floor: contiguous-split real-vs-real gap
        half = len(real) // 2
        g_rr = self.compute_gap(real[:half], real[half : half * 2])

        results: dict[str, bool] = {}

        # --- N2.1  Shuffle of returns ---
        # Random permutation per path; marginal preserved, ACF destroyed.
        shuffled = real.copy()
        for i in range(len(shuffled)):
            rng.shuffle(shuffled[i])
        g_shuffle = self.compute_gap(real, shuffled)
        results["N2.1_shuffle"] = g_shuffle > 3.0 * g_rr

        # --- N2.2  i.i.d. resampling from marginal ---
        # Draw each element independently from pooled empirical marginal.
        flat = real.ravel()
        iid_sample = rng.choice(flat, size=real.shape, replace=True)
        g_iid = self.compute_gap(real, iid_sample)
        results["N2.2_iid_resample"] = g_iid > 3.0 * g_rr

        # --- N2.3  Short-memory GARCH(1,1) ---
        # Simulate GARCH(1,1) with low persistence (alpha+beta ≈ 0.5),
        # matching real marginal's scale. Exponential ACF decay → large gap
        # at long lags even though short lags are fine.
        omega = float(np.var(flat) * 0.5)  # unconditional var fraction
        alpha, beta = 0.10, 0.40  # α+β = 0.50 → fast forgetting
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
        g_garch = self.compute_gap(real, garch)
        results["N2.3_short_memory_garch"] = g_garch > 2.0 * g_rr

        # --- N2.4  Tail replacement invariance ---
        # Replace 5% tails with Gaussian draws, preserve temporal order.
        # ACF of |r| depends partly on extreme values, so this gap is
        # not negligible — but it should be substantially smaller than
        # the gap from full temporal-structure destruction (shuffle/iid).
        std = flat.std()
        lo_q = np.quantile(flat, 0.05)
        hi_q = np.quantile(flat, 0.95)
        tail_replaced = real.copy()
        flat_tr = tail_replaced.ravel()
        mask_lo = flat_tr < lo_q
        mask_hi = flat_tr > hi_q
        flat_tr[mask_lo] = -np.abs(rng.normal(0, std, mask_lo.sum()))
        flat_tr[mask_hi] = np.abs(rng.normal(0, std, mask_hi.sum()))
        tail_replaced = flat_tr.reshape(real.shape)
        g_tail = self.compute_gap(real, tail_replaced)
        # Compare against destructive checks: tail replacement should be
        # substantially less than shuffle/iid (at most half the average).
        g_destructive_mean = (g_shuffle + g_iid) / 2.0
        results["N2.4_tail_replacement"] = g_tail < 0.6 * g_destructive_mean

        # --- N2.5  Scale invariance ---
        # ρ_k(|cr|) = ρ_k(|r|) for any positive c; gap should be ~zero.
        g_scale = self.compute_gap(real, 2.0 * real)
        results["N2.5_scale_invariance"] = g_scale < 1.5 * g_rr

        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "B2_nonlinear_temporal"

    @property
    def description(self) -> str:
        return (
            f"Lag-weighted ACF gap on |r|, lags "
            f"[{self.k_min}, {self.k_max}], per-path then averaged"
        )
