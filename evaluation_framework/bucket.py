"""
bucket.py
=========
Abstract base class for fidelity buckets.

A bucket measures one orthogonal dimension of fidelity between a real
return corpus and a synthetic return corpus. Each subclass implements
exactly one bucket from the six-bucket framework.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Bucket(ABC):
    """
    Abstract base for fidelity buckets.

    Input convention
    ----------------
    Each bucket receives two corpora as 2-D numpy arrays of shape
    (N_paths, window_len). Both corpora must have the same window_len.
    N_paths can differ between corpora.

    Output convention
    -----------------
    compute_gap() returns a non-negative scalar (float). 0 means perfect
    match. The scale and interpretation of the value is bucket-specific
    and only becomes interpretable after normalization against a
    real-real baseline.
    """

    @abstractmethod
    def compute_gap(
        self,
        real:      np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        """
        Compute the raw gap between two corpora for this bucket.

        Parameters
        ----------
        real      : shape (N_real, window_len)
        synthetic : shape (N_syn,  window_len)

        Returns
        -------
        gap : non-negative scalar
        """
        ...

    @abstractmethod
    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        """
        Run the bucket's sanity checks against constructed perturbations
        of the real corpus. Each check returns True if the gap responded
        as expected, False otherwise.

        Parameters
        ----------
        real : shape (N_real, window_len)

        Returns
        -------
        results : dict mapping check_name -> passed
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short bucket identifier, e.g. 'B1_marginal'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this bucket measures."""
        ...

    def _validate_input(
        self,
        real:      np.ndarray,
        synthetic: np.ndarray,
    ) -> None:
        """Shared input validation — called by subclasses at top of compute_gap."""
        if real.ndim != 2 or synthetic.ndim != 2:
            raise ValueError(
                f"Expected 2-D arrays (N_paths, window_len), got "
                f"real.shape={real.shape}, synthetic.shape={synthetic.shape}"
            )
        if real.shape[1] != synthetic.shape[1]:
            raise ValueError(
                f"window_len mismatch: real={real.shape[1]}, "
                f"synthetic={synthetic.shape[1]}"
            )
        if np.isnan(real).any() or np.isnan(synthetic).any():
            raise ValueError("Input contains NaNs")
        if np.isinf(real).any() or np.isinf(synthetic).any():
            raise ValueError("Input contains Infs")