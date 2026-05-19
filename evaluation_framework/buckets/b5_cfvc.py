"""
b5_cfvc.py
==========
Bucket 5 — CFVC Frobenius Gap (Cross-scale Volatility Correlation).

Measures whether synthetic data reproduces the hierarchical correlation structure
of realized volatility across time scales.

Aggregation strategy
--------------------
Temporal. Compute per-path correlation matrix, then average across N paths.
The cross-scale correlation structure is a temporal property of each path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..bucket import Bucket


class BucketCFVC(Bucket):
    """
    Cross-scale Volatility Correlation Gap using Frobenius norm.

    Parameters
    ----------
    scales : list[int], default [5, 20, 60, 120]
        Rolling window sizes in minutes for realized volatility computation.
    """

    def __init__(
        self,
        scales: list[int] | None = None,
    ) -> None:
        if scales is None:
            self.scales = [5, 20, 60, 120]
        else:
            self.scales = scales

    def compute_gap(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        self._validate_input(real, synthetic)

        c_real = self._compute_mean_corr(real)
        c_syn = self._compute_mean_corr(synthetic)

        if c_real is None or c_syn is None:
            return float('nan')

        return float(np.linalg.norm(c_real - c_syn, ord='fro'))

    def _compute_mean_corr(self, corpus: np.ndarray) -> np.ndarray | None:
        T = corpus.shape[1]
        max_w = max(self.scales)
        T_valid = T - max_w + 1

        if T_valid <= 0:
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # corpus: (N_paths, T)
        r = torch.tensor(corpus, dtype=torch.float32, device=device)
        r_sq = r ** 2

        sigmas = []
        for w in self.scales:
            # F.avg_pool1d expects input of shape (batch_size, channels, seq_len)
            # r_sq.unsqueeze(1) gives (N, 1, T)
            rv_sq = F.avg_pool1d(r_sq.unsqueeze(1), kernel_size=w, stride=1).squeeze(1)
            # rv_sq has length T - w + 1
            # We align all scales to the shortest valid length (last T_valid elements)
            rv = torch.sqrt(rv_sq[:, -T_valid:])
            sigmas.append(rv)

        # sigma_stack: (N, S, T_valid) where S is len(scales)
        sigma_stack = torch.stack(sigmas, dim=1)

        # Compute per-path correlation matrix
        # Mean over time: (N, S, 1)
        mean = sigma_stack.mean(dim=2, keepdim=True)
        # Centered: (N, S, T_valid)
        centered = sigma_stack - mean
        # Variance: (N, S, 1)
        var = (centered ** 2).sum(dim=2, keepdim=True)

        # Valid paths have non-zero variance across all scales
        valid_paths = (var.squeeze(2) > 1e-8).all(dim=1)
        if not valid_paths.any():
            return None

        centered = centered[valid_paths]
        var = var[valid_paths]
        std = torch.sqrt(var)

        # Normalized: (N_valid, S, T_valid)
        normalized = centered / std

        # Correlation matrix: (N_valid, S, S)
        corr = torch.matmul(normalized, normalized.transpose(1, 2))

        # Average across valid paths
        mean_corr = corr.mean(dim=0).cpu().numpy()
        return mean_corr

    def sanity_checks(self, real: np.ndarray) -> dict[str, bool]:
        raise NotImplementedError("Sanity checks not yet implemented for BucketCFVC.")

    @property
    def name(self) -> str:
        return "B5_cfvc"

    @property
    def description(self) -> str:
        return "Frobenius gap of cross-scale volatility correlation (CFVC)"
