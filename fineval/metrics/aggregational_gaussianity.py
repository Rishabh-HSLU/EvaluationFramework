"""M3 — Aggregational Gaussianity.

Measures whether the generator reproduces the evolution of return
kurtosis as the aggregation scale increases.

Real financial returns are typically leptokurtic at fine scales and
become more Gaussian at coarser scales. The metric compares normalized
Pearson-kurtosis curves across aggregation horizons.

See preprocessing/reasoning.md for the full metric derivation.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from fineval.metrics.base import BaseMetric


class AggregationalGaussianity(BaseMetric):
    """Normalized kurtosis evolution across aggregation scales.

    For each configured scale, consecutive one-minute log returns are
    aggregated within each trading session into non-overlapping blocks.
    Aggregated returns are then pooled across assets and sessions.

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
        scales: Sorted list of aggregation scales in minutes. Every
            scale must divide TRADING_MINUTES (390).
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
    ) -> None:
        super().__init__(name)
        self.scales = sorted(scales)
        self.min_obs = min_obs
        self.use_moors = use_moors

        if 1 not in self.scales:
            raise ValueError("Scale 1 must be included for base-scale normalization.")
        if len(self.scales) < 2:
            raise ValueError("At least one aggregation scale greater than 1 is required.")
        from fineval.config import TRADING_MINUTES

        bad = [k for k in self.scales if TRADING_MINUTES % k != 0]
        if bad:
            raise ValueError(
                f"Scales {bad} do not divide {TRADING_MINUTES}. "
                "Using only exact divisors avoids systematically "
                "discarding observations at the end of each session."
            )

    def _aggregate_scale(self, returns: pd.DataFrame, scale: int) -> np.ndarray:
        """Aggregate returns within trading-session boundaries.

        Aggregation is performed separately for each asset. A given
        asset-block is retained only when all returns in that block are
        finite.

        Returns:
            1-D array of valid aggregated returns pooled across assets and sessions.
        """
        if scale == 1:
            flat = returns.values.flatten()
            return flat[~np.isnan(flat)]

        session_ids = returns.index.normalize()
        pooled = []

        for _, group in returns.groupby(session_ids):
            # Remove first row -09:31— as it is structurally NaN
            vals = group.iloc[1:].values  # (T_session, N_tickers)
            n_blocks = vals.shape[0] // scale
            if n_blocks == 0:
                continue
            trimmed = vals[: n_blocks * scale, :]
            blocks = trimmed.reshape(n_blocks, scale, -1)
            # np.sum propagates NaN or infinite values, causing the
            # corresponding asset-block to be removed below.
            agg = np.sum(blocks, axis=1)
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
