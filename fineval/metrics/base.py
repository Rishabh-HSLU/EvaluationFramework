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

3. **Normalization on the metric.** Each subclass owns its normalize()
   logic. This may be refactored to the coordinator once all six
   metrics are implemented and their normalization turns out identical.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    """Abstract base for a single fidelity metric.

    Subclasses implement three methods:
        extract_features — panel sample → fixed-length feature vector
        compute_distance — two feature vectors → non-negative scalar
        normalize — (self-distance, cross-distance) → [0, 1] score
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def extract_features(self, sample: np.ndarray) -> np.ndarray:
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

    @abstractmethod
    def normalize(self, g_rr: float, g_sr: float) -> float:
        """Normalize raw distances into a [0, 1] similarity score.

        Args:
            g_rr: Mean self-distance (real vs real bootstrap draws).
            g_sr: Mean cross-distance (synthetic vs real draws).

        Returns:
            Similarity score in [0, 1], where 1 means the synthetic
            generator is indistinguishable from real data at this
            metric's resolution.
        """
