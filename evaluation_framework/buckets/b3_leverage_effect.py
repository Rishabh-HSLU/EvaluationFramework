"""
b3_leverage_effect.py
=====================
Bucket 3 — Cross-quantity dependence (Leverage Effect).

Measures the asymmetric relationship between past returns and future volatility
using a lag-weighted cross-correlation gap.

Mathematical formulation
------------------------
    L(k) = Corr(r_t, |r_{t+k}|)   for k > 0

    Gap_3(r, r_hat) = (1 / |W|) * sum_{k in W} |L_r(k) - L_{r_hat}(k)|

where W = {k_min, ..., k_max} is the lag window.

Aggregation strategy
--------------------
Per-path then average. The cross-correlation is a temporal statistic — pooling
across paths would corrupt it by creating false pairs at path boundaries.
"""

from __future__ import annotations

import numpy as np

from ..bucket import Bucket


class BucketLeverageEffect(Bucket):
    """
    Lag-weighted leverage-curve gap.

    Parameters
    ----------
    k_min : int, default 1
        Minimum lag in the evaluation window. The leverage effect is
        typically strongest immediately after a shock, so we start at lag 1.
    k_max : int, default 390
        Maximum lag in the evaluation window. Default 390 minutes
        corresponds to one full trading day. Capped at window_len - 1
        if the input is shorter.
    """

    def __init__(
        self,
        k_min: int = 1,
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
    def _leverage_path(path: np.ndarray, k_min: int, k_max: int) -> np.ndarray:
        """
        Compute the leverage curve L(k) = Corr(r_t, |r_{t+k}|) at lags k_min..k_max for one path.

        Parameters
        ----------
        path  : 1-D array of returns, shape (window_len,)
        k_min : minimum lag (inclusive)
        k_max : maximum lag (inclusive), capped at len(path) - 1

        Returns
        -------
        leverage : 1-D array of shape (k_max - k_min + 1,)
                   leverage[i] = Corr(r_t, |r_{t + k_min + i}|)
        """
        r = path
        abs_r = np.abs(path)
        k_max_actual = min(k_max, len(path) - 1)

        if k_min > k_max_actual:
            return np.full(k_max - k_min + 1, np.nan)

        mu_r = r.mean()
        std_r = r.std()

        mu_abs = abs_r.mean()
        std_abs = abs_r.std()

        if std_r < 1e-10 or std_abs < 1e-10:
            # Constant path — correlation undefined
            return np.zeros(k_max - k_min + 1)

        centered_r = r - mu_r
        centered_abs = abs_r - mu_abs
        n = len(path)

        leverage = np.empty(k_max - k_min + 1)
        for i, k in enumerate(range(k_min, k_max + 1)):
            if k > k_max_actual:
                leverage[i] = np.nan
            else:
                # Standard biased cross-correlation estimator uses n in denominator
                leverage[i] = np.dot(centered_r[: n - k], centered_abs[k:]) / (n * std_r * std_abs)

        return leverage

    @staticmethod
    def _mean_leverage(
        corpus: np.ndarray,
        k_min: int,
        k_max: int,
    ) -> np.ndarray:
        """
        Compute the leverage curve averaged across all paths in a corpus.
        """
        curves = np.stack(
            [
                BucketLeverageEffect._leverage_path(corpus[i], k_min, k_max)
                for i in range(len(corpus))
            ]
        )  # (N_paths, n_lags)
        return np.nanmean(curves, axis=0)  # (n_lags,)

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
        k_max_eff = min(self.k_max, window_len - 1)

        if self.k_min > k_max_eff:
            raise ValueError(
                f"Effective k_max ({k_max_eff}) <= k_min ({self.k_min}). "
                f"window_len={window_len} is too short."
            )

        # Average leverage curves across paths
        lev_real = self._mean_leverage(real, self.k_min, k_max_eff)
        lev_syn = self._mean_leverage(synthetic, self.k_min, k_max_eff)

        # Mean absolute difference across valid lags
        diff = np.abs(lev_real - lev_syn)
        valid = ~np.isnan(diff)

        if not valid.any():
            raise ValueError("All leverage estimates are NaN — paths may be too short.")

        gap = float(diff[valid].mean())
        return gap

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        """
        Sanity checks for BucketLeverageEffect:
            N3.1 — Time reversal flips the leverage effect -> large gap
            N3.2 — Sign symmetrization (randomly flipping r_t sign) -> large gap
            N3.3 — Symmetric GARCH simulation -> large gap
            N3.4 — Shuffle invariance -> large gap
            N3.5 — Scale invariance -> ~zero gap

        Note: leverage at 1-minute frequency is genuinely weak, so we use
        a multi-split averaged baseline to stabilise the noise floor and
        use moderate thresholds.
        """
        rng = np.random.default_rng(0)

        # Multi-split averaged noise floor: 8 random splits to get a
        # more stable g_rr (single-split g_rr for leverage is noisy
        # because the signal is weak at 1-minute frequency).
        n_splits = 8
        g_rr_samples = []
        indices = np.arange(len(real))
        for _ in range(n_splits):
            rng.shuffle(indices)
            half = len(real) // 2
            a_idx, b_idx = indices[:half], indices[half : half * 2]
            g_rr_samples.append(self.compute_gap(real[a_idx], real[b_idx]))
        g_rr = float(np.mean(g_rr_samples))

        results: dict[str, bool] = {}

        # --- N3.1  Time reversal ---
        # Reverse time order per path; flips causal direction of
        # leverage (drop → future vol becomes future vol → drop).
        reversed_real = real[:, ::-1].copy()
        g_rev = self.compute_gap(real, reversed_real)
        results["N3.1_time_reversal"] = g_rev > 1.5 * g_rr

        # --- N3.2  Sign symmetrization ---
        # Randomly flip the sign of each r_t independently.
        # Preserves |r| exactly (B2 invariant), destroys leverage → 0.
        signs = rng.choice([-1.0, 1.0], size=real.shape)
        symmetrized = real * signs
        g_sym = self.compute_gap(real, symmetrized)
        results["N3.2_sign_symmetrization"] = g_sym > 1.5 * g_rr

        # --- N3.3  Symmetric GARCH simulation ---
        # GARCH(1,1) with symmetric innovations: conditional variance
        # depends on r², not r, so L(k)=0 at every lag.
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
        g_garch = self.compute_gap(real, garch)
        results["N3.3_symmetric_garch"] = g_garch > 1.5 * g_rr

        # --- N3.4  Shuffle ---
        # Random permutation per path destroys all cross-lag
        # relationships, including leverage.
        shuffled = real.copy()
        for i in range(len(shuffled)):
            rng.shuffle(shuffled[i])
        g_shuffle = self.compute_gap(real, shuffled)
        results["N3.4_shuffle"] = g_shuffle > 1.5 * g_rr

        # --- N3.5  Scale invariance ---
        # Corr(c·r_t, c·|r_{t+k}|) = Corr(r_t, |r_{t+k}|); gap ~ 0.
        g_scale = self.compute_gap(real, 2.0 * real)
        results["N3.5_scale_invariance"] = g_scale < 1.5 * g_rr

        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "B3_leverage_effect"

    @property
    def description(self) -> str:
        return (
            f"Lag-weighted leverage curve gap on Corr(r_t, |r_{{t+k}}|), lags "
            f"[{self.k_min}, {self.k_max}], per-path averaged"
        )
