"""
b4_kurtosis.py
==============
Bucket 4 — Scale-Weighted Kurtosis Gap.

Measures the tail heaviness of returns at multiple aggregation horizons
and how it converges toward Gaussianity under temporal aggregation
(aggregational Gaussianity — Cont 2001 stylized fact #4).

Aggregation strategy
--------------------
Distributional. Pool all returns across N paths into one 1D vector before
aggregating. Kurtosis is a population-level property of the marginal
distribution at each scale.

Estimator choice: L-kurtosis
-----------------------------
Standard moment-based excess kurtosis requires a finite eighth moment for
stable estimation. Financial returns — especially during crisis regimes —
have tail indices ξ ≈ 0.3–0.5, meaning the eighth moment does not exist.
Using scipy.stats.kurtosis on such data produces CV > 0.8 at N=200 paths,
rendering the metric uninformative.

L-kurtosis (fourth L-moment ratio τ_4) is defined via linear combinations
of order statistics and requires only that E[|X|] < ∞ (first moment finite),
which holds for all financial returns. It measures the same shape property —
tail heaviness relative to the center — with dramatically better finite-sample
stability for heavy-tailed distributions.

L-kurtosis for a Gaussian distribution is τ_4 = 0.1226.
Heavy-tailed distributions have τ_4 > 0.1226.
As aggregation horizon grows, returns approach Gaussian and τ_4 → 0.1226.

References
----------
- Hosking (1990): L-moments — analysis and estimation of distributions
  using linear combinations of order statistics
- Cont (2001): aggregational Gaussianity as stylized fact #4
- Zhang et al. (2026, SFAG): scale-dependent tail behavior in synthetic data
"""

from __future__ import annotations

import numpy as np

from ..bucket import Bucket


# ---------------------------------------------------------------------------
# L-kurtosis estimator
# ---------------------------------------------------------------------------

def _l_kurtosis(x: np.ndarray) -> float:
    """
    Compute sample L-kurtosis (fourth L-moment ratio τ_4).

    τ_4 = λ_4 / λ_2

    where λ_r is the r-th L-moment, computed from order statistics via
    the unbiased PWM (probability-weighted moment) estimator.

    Parameters
    ----------
    x : 1-D array of observations (need not be sorted)

    Returns
    -------
    tau4 : float, the L-kurtosis. Returns nan if n < 4.

    Notes
    -----
    Uses the unbiased PWM estimator b_r = (1/n) * sum_j C(j-1, r)/C(n-1, r) * x_(j)
    where x_(j) are order statistics and C is the binomial coefficient.
    L-moments: λ_1 = b_0
                λ_2 = 2*b_1 - b_0
                λ_4 = 20*b_3 - 30*b_2 + 12*b_1 - b_0
    τ_4 = λ_4 / λ_2
    """
    n = len(x)
    if n < 4:
        return float("nan")

    x_sorted = np.sort(x)
    j = np.arange(1, n + 1, dtype=float)   # 1-indexed order statistic positions

    # Unbiased PWM estimators b_0, b_1, b_2, b_3
    # b_r = (1/n) * sum_j [C(j-1, r) / C(n-1, r)] * x_(j)
    # Only defined for j-1 >= r, i.e. j >= r+1

    def pwm(r: int) -> float:
        # C(j-1, r) / C(n-1, r) for j = r+1 ... n
        # = product_{k=0}^{r-1} (j-1-k) / (n-1-k)
        idx   = j[r:]           # j values from r+1 to n (0-indexed slice from r)
        coeff = np.ones(n - r)
        for k in range(r):
            coeff *= (idx - 1 - k) / (n - 1 - k)
        return float(coeff @ x_sorted[r:]) / n

    b0 = pwm(0)   # = mean(x)
    b1 = pwm(1)
    b2 = pwm(2)
    b3 = pwm(3)

    lam2 = 2.0 * b1 - b0
    lam4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0

    if abs(lam2) < 1e-12:
        return float("nan")

    return lam4 / lam2


# ---------------------------------------------------------------------------
# Bucket class
# ---------------------------------------------------------------------------

class BucketKurtosis(Bucket):
    """
    Scale-Weighted L-Kurtosis Gap.

    Measures the tail heaviness of pooled returns at multiple aggregation
    horizons via L-kurtosis (τ_4 = λ_4 / λ_2), and the weighted absolute
    gap between real and synthetic L-kurtosis curves across horizons.

    L-kurtosis is used in place of moment-based excess kurtosis because
    financial returns have infinite eighth moments during crisis regimes,
    making the moment estimator unstable at N=200 paths (CV > 0.8).
    L-kurtosis requires only E[|X|] < ∞ and is stable for all financial
    return distributions.

    Parameters
    ----------
    horizons : list[int], default [1, 5, 30, 60, 390]
        Aggregation horizons in minutes. Each horizon h aggregates returns
        into non-overlapping h-minute blocks by summing h consecutive returns.
    weights : list[float] | None, default None
        Per-horizon weights. If None, uses w_h = 1/h so that fine-scale
        horizons (where the stylized fact is strongest) receive more weight.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        weights:  list[float] | None = None,
    ) -> None:
        if horizons is None:
            self.horizons = [1, 5, 30, 60, 390]
        else:
            self.horizons = list(horizons)

        if weights is None:
            self.weights = [1.0 / h for h in self.horizons]
        else:
            if len(weights) != len(self.horizons):
                raise ValueError("weights and horizons must have the same length")
            self.weights = list(weights)

    # ------------------------------------------------------------------
    # Core metric
    # ------------------------------------------------------------------

    def compute_gap(
        self,
        real:      np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        """
        Compute the scale-weighted L-kurtosis gap between real and synthetic.

        For each horizon h:
          1. Pool all returns across N paths into one 1-D vector
          2. Aggregate into non-overlapping h-minute blocks (sum h returns)
          3. Compute L-kurtosis τ_4 on the block distribution
          4. gap_h = |τ_4(real) - τ_4(synthetic)|

        Final gap = sum(w_h * gap_h) / sum(w_h)

        Parameters
        ----------
        real      : (N_real, window_len)
        synthetic : (N_syn, window_len)

        Returns
        -------
        gap : non-negative scalar, or nan if all horizons are degenerate
        """
        self._validate_input(real, synthetic)

        real_pooled = real.ravel()
        syn_pooled  = synthetic.ravel()

        total_gap    = 0.0
        total_weight = 0.0

        for h, w in zip(self.horizons, self.weights):
            # Aggregate into h-minute blocks
            n_real = (len(real_pooled) // h) * h
            n_syn  = (len(syn_pooled)  // h) * h
            blocks_real = real_pooled[:n_real].reshape(-1, h).sum(axis=1)
            blocks_syn  = syn_pooled[:n_syn].reshape(-1, h).sum(axis=1)

            # Need at least 4 observations for L-kurtosis (PWM b_3 requires n >= 4)
            # In practice we want many more — skip only truly degenerate cases
            if len(blocks_real) < 30 or len(blocks_syn) < 30:
                continue

            tau4_real = _l_kurtosis(blocks_real)
            tau4_syn  = _l_kurtosis(blocks_syn)

            if np.isnan(tau4_real) or np.isnan(tau4_syn):
                continue

            total_gap    += w * abs(tau4_real - tau4_syn)
            total_weight += w

        if total_weight == 0.0:
            return float("nan")

        return total_gap / total_weight

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        """
        Sanity checks for BucketKurtosis:
            N4.1 — Gaussian noise replacement should reduce L-kurtosis gap
                    (Gaussian has τ_4 = 0.1226, real is higher)
            N4.2 — Temporal shuffle should not change gap
                    (L-kurtosis is a marginal property, order-invariant)
            N4.3 — Scale perturbation (2x vol) should change gap at h=1
                    but not qualitatively at large h
        """
        raise NotImplementedError(
            "Sanity checks not yet implemented for BucketKurtosis."
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "B4_kurtosis"

    @property
    def description(self) -> str:
        horizons_str = str(self.horizons)
        return (
            f"Scale-weighted L-kurtosis gap on pooled returns "
            f"(horizons={horizons_str}, weights=1/h)"
        )