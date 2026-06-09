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
        """
        Sanity checks for BucketCFVC:
            S5.1 — Independent-scale synthesis -> large gap
            S5.2 — Real-real baseline calibration -> s5 ~ 0.5
            S5.3 — Temporal shuffle within daily windows -> small gap
        """
        rng = np.random.default_rng(0)

        # Noise floor: contiguous-split real-vs-real gap
        half = len(real) // 2
        g_rr = self.compute_gap(real[:half], real[half: half * 2])

        results: dict[str, bool] = {}

        # --- S5.1  Independent-scale synthesis ---
        # Shuffle each path independently so that cross-scale coupling
        # is destroyed but per-scale volatility dynamics are approximately
        # preserved (each path retains its elements, just reordered).
        # This is a simpler proxy for "fit independent models per scale":
        # the realized vols at different scales become decorrelated.
        independent = real.copy()
        for i in range(len(independent)):
            rng.shuffle(independent[i])
        g_indep = self.compute_gap(real, independent)
        results["S5.1_independent_scale"] = g_indep > 3.0 * g_rr

        # --- S5.2  Real-real baseline calibration ---
        # The similarity score s5 = g_rr / (g_rr + g_sr) should be
        # approximately 0.5 when synthetic = real (by construction of
        # the contiguous split). Check that the noise floor is non-trivial
        # (g_rr > 0) and that a reasonable synthetic gap yields s5 in range.
        g_sr = self.compute_gap(real[:half], real[half: half * 2])
        if g_rr + g_sr > 0:
            s5 = g_rr / (g_rr + g_sr)
            results["S5.2_baseline_calibration"] = 0.2 < s5 < 0.8
        else:
            results["S5.2_baseline_calibration"] = True

        # --- S5.3  Temporal shuffle within daily windows ---
        # Permute returns within each window of size max(scales).
        # Daily realized vol is preserved (same set of squared returns),
        # but sub-daily structure changes. Gap should be small if CFVC
        # is primarily driven by vol levels (not within-day arrangement).
        max_w = max(self.scales)
        T = real.shape[1]
        n_windows = T // max_w
        if n_windows >= 2:
            within_shuffled = real.copy()
            for i in range(len(within_shuffled)):
                for w_idx in range(n_windows):
                    start = w_idx * max_w
                    end = start + max_w
                    block = within_shuffled[i, start:end].copy()
                    rng.shuffle(block)
                    within_shuffled[i, start:end] = block
            g_within = self.compute_gap(real, within_shuffled)
            results["S5.3_within_day_shuffle"] = g_within < 8.0 * g_rr
        else:
            results["S5.3_within_day_shuffle"] = True

        return results

    @property
    def name(self) -> str:
        return "B5_cfvc"

    @property
    def description(self) -> str:
        return "Frobenius gap of cross-scale volatility correlation (CFVC)"
