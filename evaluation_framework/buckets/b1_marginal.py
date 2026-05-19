"""
b1_marginal.py
==============
Bucket 1 — Marginal distribution.

Measures the fidelity of the unconditional marginal return distribution
using tail-weighted Wasserstein-1 distance.

Mathematical formulation
------------------------
    Gap_1(F_r, F_r_hat) = integral_0^1 w(u) * |F_r^{-1}(u) - F_r_hat^{-1}(u)| du

with weight function:

    w(u) = 1  if u in [0, tail_q] or u in [1 - tail_q, 1]
    w(u) = 0  otherwise

where tail_q defaults to 0.05 (VaR-95 alignment).

Aggregation strategy
--------------------
Pooled. All returns across all paths in each corpus are flattened into
a single 1-D vector. The marginal distribution is a property of the
return-generating process at the population level, so the estimator
must pool — this is dictated by the statistical object being measured,
not a design choice.

References
----------
- Cont (2001): heavy tails and gain/loss asymmetry as defining stylized
  facts of the marginal distribution
- Zhang et al. (2026, SFAG): tail-emphasized distributional matching for
  stress-testing relevance
"""

from __future__ import annotations

import numpy as np

from ..bucket import Bucket


class BucketMarginal(Bucket):
    """
    Tail-weighted Wasserstein-1 between the pooled marginal distributions
    of real and synthetic return corpora.

    Parameters
    ----------
    tail_q : float, default 0.05
        Tail quantile threshold. The weight function is 1 on
        u in [0, tail_q] union [1 - tail_q, 1] and 0 elsewhere.
        Default 0.05 aligns with VaR-95 industry practice. Set to 0.01
        for VaR-99 alignment.
    n_quantile_grid : int, default 1000
        Number of points on the common quantile grid used for numerical
        integration. Larger gives finer resolution at higher cost.
    """

    def __init__(
        self,
        tail_q:          float = 0.05,
        n_quantile_grid: int   = 1000,
    ) -> None:
        if not (0.0 < tail_q < 0.5):
            raise ValueError(f"tail_q must be in (0, 0.5), got {tail_q}")
        if n_quantile_grid < 100:
            raise ValueError(
                f"n_quantile_grid must be at least 100, got {n_quantile_grid}"
            )
        self.tail_q          = tail_q
        self.n_quantile_grid = n_quantile_grid

    # ------------------------------------------------------------------
    # Core metric
    # ------------------------------------------------------------------

    def compute_gap(
        self,
        real:      np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        self._validate_input(real, synthetic)

        # Pool: flatten both corpora into 1-D vectors
        real_pooled = real.ravel()
        syn_pooled  = synthetic.ravel()

        # Common quantile grid u in [1/(n+1), n/(n+1)] avoiding 0 and 1
        n = self.n_quantile_grid
        u = (np.arange(1, n + 1) - 0.5) / n        # midpoint rule

        # Empirical quantile functions
        q_real = np.quantile(real_pooled, u)
        q_syn  = np.quantile(syn_pooled,  u)

        # Pointwise absolute difference
        diff = np.abs(q_real - q_syn)

        # Tail region mask
        mask = (u < self.tail_q) | (u > 1.0 - self.tail_q)

        # Average over tail region only (mean = numerical integral of
        # the indicator weight, normalized by tail region width)
        gap = float(diff[mask].mean())
        return gap

    # ------------------------------------------------------------------
    # Sanity checks — stubs for now, fill in during empirical validation
    # ------------------------------------------------------------------

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        """
        Sanity checks for BucketMarginal:
            N1.1 — tail replacement should produce large gap
            N1.2 — skew flip should produce large gap
            N1.3 — temporal shuffle should produce ~zero gap
            N1.4 — bulk perturbation (tails preserved) should produce small gap
            N1.5 — variance scaling by 2x should produce large gap
        """
        raise NotImplementedError(
            "Sanity checks not yet implemented for BucketMarginal."
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "B1_marginal"

    @property
    def description(self) -> str:
        return (
            f"Tail-weighted Wasserstein-1 on pooled returns "
            f"(tail_q={self.tail_q}, grid={self.n_quantile_grid})"
        )