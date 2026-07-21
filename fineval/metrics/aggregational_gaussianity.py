"""M3 — Aggregational Gaussianity.

Measures whether the generator reproduces the evolution of return
kurtosis as the aggregation scale increases.

Real financial returns are typically leptokurtic at fine scales and
become more Gaussian at coarser scales. The metric compares normalized
Pearson-kurtosis curves across aggregation horizons.

See preprocessing/reasoning.md for the full metric derivation.
"""

from math import lcm
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from fineval.metrics.base import BaseMetric


class AggregationalGaussianity(BaseMetric):
    """Normalized kurtosis evolution across aggregation scales.

    For each configured scale, consecutive one-minute log returns are
    aggregated within each trading session into non-overlapping blocks.
    Aggregated returns are then pooled across assets and sessions.
    All aggregation scales use identical intraday support. With the canonical
    390-row market clock and scales [1, 5, 15, 30], the structural opening row
    is removed and every scale is evaluated on a common 360-return segment.

    With classical kurtosis, the feature at scale ``k`` is

        K(k) / K(1),

    where ``K`` is Pearson kurtosis. Pearson kurtosis is used instead
    of excess kurtosis because its Gaussian reference value is 3,
    avoiding unstable normalization when base-scale excess kurtosis is
    close to zero.

    If ``use_moors`` is enabled, the same base-scale normalization is
    applied to the Moors octile-kurtosis coefficient.

    Attributes:
        name: Metric label, e.g. "aggregational_gaussianity".
        scales:
            Sorted list of positive aggregation scales in minutes. Scale 1 must
            be included. All scales are evaluated on the largest common
            within-session support divisible by every configured scale.

        support_anchor:
            Determines whether the common support is anchored at the beginning
            or end of the valid within-session return sequence. ``"close"`` retains
            the market close; ``"open"`` retains the market open.
        min_obs: Minimum pooled observations at a given scale for
            a reliable kurtosis estimate. Scales with fewer valid
            aggregated returns emit NaN.
        use_moors: If True, use Moors (1988) octile kurtosis
            (quantile-based, outlier-resistant) instead of
            classical Pearson kurtosis.

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
        support_anchor: Literal["open", "close"] = "close",
    ) -> None:
        super().__init__(name)
        self.scales = sorted(scales)
        self.min_obs = min_obs
        self.use_moors = use_moors
        self.support_anchor = support_anchor

        if 1 not in self.scales:
            raise ValueError("Scale 1 must be included for base-scale normalization.")
        if len(self.scales) < 2:
            raise ValueError("At least one aggregation scale greater than 1 is required.")
        if any(not isinstance(scale, int) or scale <= 0 for scale in self.scales):
            raise ValueError("All aggregation scales must be positive integers.")
        if support_anchor not in {"open", "close"}:
            raise ValueError("support_anchor must be either 'open' or 'close'.")
        from fineval.config import TRADING_MINUTES

        # The first clock row is structurally unavailable because no preceding
        # close exists within the session.
        valid_returns_per_session = TRADING_MINUTES - 1
        # Use the largest session segment that is divisible by every configured
        # scale, so all scales are estimated on identical intraday support.
        scale_multiple = lcm(*self.scales)
        self.common_support = (valid_returns_per_session // scale_multiple) * scale_multiple
        if self.common_support < max(self.scales):
            raise ValueError(
                "Configured scales do not fit within the valid session support. "
                f"Valid returns per session: {valid_returns_per_session}; "
                f"scales: {self.scales}."
            )

    def _aggregate_scale(
        self,
        returns: pd.DataFrame,
        scale: int,
    ) -> np.ndarray:
        """Aggregate returns on identical within-session support.

        The structural opening row is removed first. Every scale is then
        evaluated on the same contiguous session segment of length
        ``common_support``. The segment may be anchored at the market open or
        close, as configured by ``support_anchor``.

        A ticker-block is retained only when its aggregated return is finite.

        Returns:
            One-dimensional array of valid aggregated returns pooled across
            tickers and sessions.
        """
        session_ids = returns.index.normalize()
        pooled: list[np.ndarray] = []

        for _, group in returns.groupby(session_ids):
            # Remove the structural opening row, whose return cannot be formed
            # without a preceding within-session close.
            vals = group.iloc[1:].to_numpy(dtype=float, copy=False)

            if vals.shape[0] < self.common_support:
                # Exclude sessions that cannot provide the complete common
                # support, such as shortened trading sessions.
                continue

            if self.support_anchor == "open":
                support = vals[: self.common_support, :]
            else:
                support = vals[-self.common_support :, :]

            n_blocks = self.common_support // scale
            blocks = support.reshape(n_blocks, scale, -1)

            # NaN and infinite inputs propagate through the sum and are removed
            # after pooling.
            aggregated = np.sum(blocks, axis=1)
            pooled.append(aggregated.reshape(-1))

        if not pooled:
            return np.array([], dtype=float)

        all_aggregated = np.concatenate(pooled)
        return all_aggregated[np.isfinite(all_aggregated)]

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
        """Compute Pearson kurtosis or the Moors coefficient."""
        if len(x) < self.min_obs:
            return np.nan
        if self.use_moors:
            return self._moors_kurtosis(x)
        return float(scipy_kurtosis(x, fisher=False, bias=False))

    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        """Compute the normalized aggregation curve.

        Returns:
            One-dimensional array with one entry per configured scale.
            The base-scale entry equals 1. Scales with insufficient
            observations emit NaN.
        """
        raw_kurtosis = np.full(len(self.scales), np.nan)
        for i, scale in enumerate(self.scales):
            agg = self._aggregate_scale(sample, scale)
            raw_kurtosis[i] = self._compute_kurtosis(agg)
        base_kurtosis = raw_kurtosis[self.scales.index(1)]
        if not np.isfinite(base_kurtosis) or base_kurtosis == 0.0:
            return np.full(len(self.scales), np.nan)
        return raw_kurtosis / base_kurtosis

    def compute_distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        """Compute the masked mean absolute curve difference.

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
