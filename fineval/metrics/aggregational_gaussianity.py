"""M4 — Aggregational Gaussianity.

Measures whether the generator reproduces the convergence of return
distributions toward Gaussianity as the aggregation scale increases.
Real financial returns are leptokurtic at fine scales but converge
toward Gaussian at coarser scales — the rate of this convergence is
a stylized fact (Cont, 2001 §2.5) that naive generators cannot
reproduce. GBM is Gaussian at every scale (flat ratio curve);
GARCH converges too fast; a good generator should track the
empirical decay rate.

See preprocessing/reasoning.md for full metric derivation context.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from fineval.metrics.base import BaseMetric


class AggregationalGaussianity(BaseMetric):
    """Normalized excess kurtosis decay across aggregation scales.

    For each scale k in the configured scale set, aggregates k
    consecutive 1-minute log returns within each session into
    non-overlapping k-minute returns, pools across tickers, and
    computes excess kurtosis. The feature vector is the normalized
    ratio κ(k)/κ(1), capturing the rate at which the distribution
    converges toward Gaussian as the scale increases.

    Aggregation is confined within session boundaries (calendar-date
    grouping) so that no aggregated return ever spans an overnight
    gap — the same session-boundary discipline enforced in M2 and M6.

    Attributes:
        name: Metric label, e.g. "aggregational_gaussianity".
        scales: Sorted list of aggregation scales in minutes. Every
            scale must divide TRADING_MINUTES (390).
        min_obs: Minimum pooled observations at a given scale for
            a reliable kurtosis estimate. Scales with fewer valid
            aggregated returns emit NaN.
        use_moors: If True, use Moors (1988) octile kurtosis
            (quantile-based, outlier-resistant) instead of standard
            excess kurtosis. Default False — classical kurtosis is
            the quantity the stylized fact describes, and pooled
            sample sizes are large enough to estimate it reliably.

    Example::

        from fineval.config import M4_SCALES, M4_MIN_OBS

        metric = AggregationalGaussianity(
            name="aggregational_gaussianity",
            scales=M4_SCALES,
            min_obs=M4_MIN_OBS,
        )
    """

    def __init__(
        self,
        name: str,
        scales: list[int],
        min_obs: int,
        use_moors: bool = False,
    ) -> None:
        super().__init__(name)
        self.scales = sorted(scales)
        self.min_obs = min_obs
        self.use_moors = use_moors

        if 1 not in self.scales:
            raise ValueError(
                "Scale 1 must be in the scale set — κ(k)/κ(1) normalization requires it."
            )
        from fineval.config import TRADING_MINUTES

        bad = [k for k in self.scales if TRADING_MINUTES % k != 0]
        if bad:
            raise ValueError(
                f"Scales {bad} do not divide {TRADING_MINUTES}. "
                f"Non-divisible scales cause systematic data loss on "
                f"full sessions."
            )

    # ── internal helpers ─────────────────────────────────────────────

    def _aggregate_scale(self, returns: pd.DataFrame, scale: int) -> np.ndarray:
        """Aggregate returns to a given scale within session boundaries.

        For scale 1, pools and drops NaN directly. For scale > 1,
        groups by calendar date, splits each session into
        non-overlapping blocks of exactly ``scale`` bars, sums log
        returns within each block (NaN propagates naturally — any
        block with a missing bar becomes NaN and is dropped).

        Returns:
            1-D array of valid aggregated returns, NaN-free.
        """
        if scale == 1:
            flat = returns.values.flatten()
            return flat[~np.isnan(flat)]

        session_ids = returns.index.normalize()
        pooled = []

        for _, group in returns.groupby(session_ids):
            vals = group.values  # (T_session, N_tickers)
            n_blocks = vals.shape[0] // scale
            if n_blocks == 0:
                continue
            trimmed = vals[: n_blocks * scale, :]
            blocks = trimmed.reshape(n_blocks, scale, -1)
            agg = np.sum(blocks, axis=1)  # NaN propagates honestly
            pooled.append(agg.flatten())

        if not pooled:
            return np.array([])

        all_agg = np.concatenate(pooled)
        return all_agg[~np.isnan(all_agg)]

    @staticmethod
    def _moors_kurtosis(x: np.ndarray) -> float:
        """Moors (1988) octile kurtosis — quantile-based, outlier-resistant.

        K_M = ((E_7 - E_5) + (E_3 - E_1)) / (E_6 - E_2)

        where E_i is the i/8 quantile. For a Gaussian, K_M ≈ 1.233.
        """
        e1, e2, e3, e5, e6, e7 = np.quantile(x, [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8])
        denom = e6 - e2
        if denom == 0.0:
            return np.nan
        return ((e7 - e5) + (e3 - e1)) / denom

    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Excess kurtosis via standard or Moors estimator."""
        if len(x) < self.min_obs:
            return np.nan
        if self.use_moors:
            return self._moors_kurtosis(x)
        return float(scipy_kurtosis(x, fisher=True))

    # ── public interface ─────────────────────────────────────────────

    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        """Compute normalized kurtosis ratio curve κ(k)/κ(1).

        Pipeline:
            1. For each scale k, aggregate within-session returns
               into non-overlapping k-minute blocks, pool across
               tickers, drop NaN.
            2. Compute excess kurtosis (or Moors) at each scale.
            3. Normalize by κ(1) to produce the decay curve.

        Args:
            sample: Panel of deseasonalized returns, DataFrame with
                tickers as columns and a time index.

        Returns:
            1-D ndarray of length len(scales). κ(1)/κ(1) = 1.0
            by construction. Scales with too few observations or
            a degenerate κ(1) emit NaN.
        """
        raw_kurtosis = np.full(len(self.scales), np.nan)

        for i, scale in enumerate(self.scales):
            agg = self._aggregate_scale(sample, scale)
            raw_kurtosis[i] = self._compute_kurtosis(agg)

        kappa_1 = raw_kurtosis[0]  # scales is sorted; index 0 = scale 1
        if not np.isfinite(kappa_1) or kappa_1 == 0.0:
            return np.full(len(self.scales), np.nan)

        return raw_kurtosis / kappa_1

    def compute_distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        """Masked mean absolute difference between kurtosis ratio curves.

        Args:
            fa: Normalized kurtosis curve from sample A.
            fb: Normalized kurtosis curve from sample B.

        Returns:
            Non-negative scalar. NaN if all scales are masked.
        """
        valid = np.isfinite(fa) & np.isfinite(fb)
        if not valid.any():
            return np.nan
        return float(np.mean(np.abs(fa[valid] - fb[valid])))
