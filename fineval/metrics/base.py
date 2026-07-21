"""Abstract base class for all fineval fidelity metrics.

Design principles (see preprocessing/reasoning.md for full rationale):

1. **Stateless.** No fit() step. extract_features takes a single panel
   sample and returns a feature vector in one pass. Regime boundaries,
   quantile cuts, etc. are computed internally from the sample itself
   (self-labeled), not anchored to a reference distribution.

2. **Honest NaN policy.** extract_features emits NaN for any feature
   dimension where the data is insufficient for a stable estimate,
   rather than imputing or padding. compute_distance handles NaN via
   masking — comparing only the dimensions where both vectors are
   finite. This mirrors the framework's no-imputation principle:
   absence is preserved, never fabricated over.

3. **Normalization on the base.** The normalization turned out
   identical across all implemented metrics, so normalize() is a
   single concrete method here: a nanmean ratio-of-means score with
   a 0.5 convention in the degenerate all-zero case.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseMetric(ABC):
    """Abstract base for a single fidelity metric.

    Subclasses implement two methods:
        extract_features — panel sample → fixed-length feature vector
        compute_distance — two feature vectors → non-negative scalar

    normalize() — per-draw distance arrays → [0, 1] score — is
    provided concretely here and shared by all metrics.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        """Extract a feature vector from a single panel sample.

        Args:
            sample: Panel of deseasonalized returns. Shape and
                semantics are metric-specific (e.g. tickers × time).

        Returns:
            1-D ndarray. Entries may be NaN where the data is
            insufficient for stable estimation.
        """

    @abstractmethod
    def compute_distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        """Distance between two feature vectors.

        Must handle NaN entries in either vector by masking — computing
        the distance only over dimensions where both fa and fb are
        finite.

        Args:
            fa: Feature vector from sample A.
            fb: Feature vector from sample B.

        Returns:
            Non-negative scalar. NaN only if every dimension is masked
            in both vectors.
        """

    def normalize(self, g_rr: np.ndarray, g_sr: np.ndarray) -> float:
        """Normalize per-draw distances into a [0, 1] similarity score.

        Ratio of means: s_b = mean(g_rr) / (mean(g_rr) + mean(g_sr)),
        with NaN draws ignored via nanmean.

        Args:
            g_rr: Per-draw self-distances (real vs real bootstrap
                draws), 1-D ndarray.
            g_sr: Per-draw cross-distances (synthetic vs real draws),
                1-D ndarray.

        Returns:
            Similarity score in [0, 1]. s_b ≈ 0.5 means the
            synthetic–real gap matches the real–real noise floor —
            indistinguishable at this metric's resolution. s_b → 0
            means the synthetic gap dominates the noise floor.
            s_b → 1 means the synthetic–real gap is *smaller* than
            the real–real noise floor — a red flag for memorization
            or copying, not a good score. In the degenerate case
            where both means are zero (synthetic exactly as close to
            real as real is to itself), returns 0.5.
        """
        valid = np.isfinite(g_rr) & np.isfinite(g_sr)
        rr_mean = np.mean(g_rr[valid])
        sr_mean = np.mean(g_sr[valid])
        if rr_mean + sr_mean == 0.0:
            return 0.5
        return rr_mean / (rr_mean + sr_mean)
