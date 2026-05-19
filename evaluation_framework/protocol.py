"""
Matched-N resampling protocol and similarity scoring.

s_b = mean(g_rr) / (mean(g_rr) + mean(g_sr))

g_rr : disjoint real–real pairs of N paths (200 resamples by default)
g_sr : real vs synthetic pairs of N paths each side
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .samplers import PooledSampler, Sampler

GapFn = Callable[[np.ndarray, np.ndarray], float]


def rr_gaps(
    gap_fn:       GapFn,
    real:         np.ndarray,
    n_per_half:   int,
    n_resamples:  int,
    rng:          np.random.Generator,
    sampler:      Sampler | None = None,
) -> np.ndarray:
    if sampler is None:
        sampler = PooledSampler(len(real))
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        idx_a, idx_b = sampler.draw_pair(n_per_half, rng)
        out[i] = gap_fn(real[idx_a], real[idx_b])
    return out


def sr_gaps(
    gap_fn:       GapFn,
    real:         np.ndarray,
    syn:          np.ndarray,
    n:            int,
    n_resamples:  int,
    rng:          np.random.Generator,
    real_sampler: Sampler | None = None,
) -> np.ndarray:
    if real_sampler is None:
        real_sampler = PooledSampler(len(real))
    n_syn = len(syn)
    replace_syn = n_syn <= n
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        idx_r = real_sampler.draw_single(n, rng)
        idx_s = rng.choice(n_syn, size=n, replace=replace_syn)
        out[i] = gap_fn(real[idx_r], syn[idx_s])
    return out


def similarity_with_ci(
    g_rr:         np.ndarray,
    g_sr:         np.ndarray,
    n_bootstrap:  int,
    rng:          np.random.Generator,
) -> tuple[float, float, float]:
    """Unpaired bootstrap CI — resamples indices into precomputed g_rr / g_sr
    arrays independently. Cheap; ignores the shared-real-corpus correlation
    that bridges the two arms."""
    rr_m, sr_m = g_rr.mean(), g_sr.mean()
    denom = rr_m + sr_m
    sim = rr_m / denom if denom > 0 else 0.0

    n = len(g_rr)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        rr_b = g_rr[rng.integers(0, n, n)].mean()
        sr_b = g_sr[rng.integers(0, n, n)].mean()
        d = rr_b + sr_b
        boot[b] = rr_b / d if d > 0 else 0.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return sim, float(lo), float(hi)


# Explicit alias — same function, named for clarity in callers that want
# to distinguish paired vs unpaired CI estimators.
similarity_with_ci_unpaired = similarity_with_ci


def rr_sr_gaps_paired(
    gap_fn:       GapFn,
    real:         np.ndarray,
    syn:          np.ndarray,
    n_per_half:   int,
    n_resamples:  int,
    rng:          np.random.Generator,
    sampler:      Sampler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct PAIRED (g_rr, g_sr) arrays.

    For each i in 1..n_resamples:
        1. sampler.draw_pair → (idx_a, idx_b)         [real indices]
        2. g_rr[i] = gap_fn(real[idx_a], real[idx_b])
        3. rng.choice on syn → idx_s                  [synthetic indices]
        4. g_sr[i] = gap_fn(real[idx_a], syn[idx_s])  ← reuses idx_a

    g_rr[i] and g_sr[i] share the real-side indices `idx_a`, inducing
    positive correlation that the paired bootstrap exploits.
    """
    if sampler is None:
        sampler = PooledSampler(len(real))
    n_syn       = len(syn)
    replace_syn = n_syn <= n_per_half
    g_rr = np.empty(n_resamples)
    g_sr = np.empty(n_resamples)
    for i in range(n_resamples):
        idx_a, idx_b = sampler.draw_pair(n_per_half, rng)
        g_rr[i]      = gap_fn(real[idx_a], real[idx_b])
        idx_s        = rng.choice(n_syn, size=n_per_half, replace=replace_syn)
        g_sr[i]      = gap_fn(real[idx_a], syn[idx_s])
    return g_rr, g_sr


def similarity_with_ci_paired(
    g_rr:         np.ndarray,
    g_sr:         np.ndarray,
    n_bootstrap:  int,
    rng:          np.random.Generator,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI. Assumes g_rr / g_sr were constructed PAIRED so that
    g_rr[i] and g_sr[i] share real-corpus indices (see rr_sr_gaps_paired).

    On each of n_bootstrap resamples, the SAME index vector is drawn into
    both g_rr and g_sr. This preserves the per-i correlation that bridges
    the two arms and narrows the CI on s_b vs the unpaired version.

    Point estimate is identical to the unpaired bootstrap when fed the same
    (g_rr, g_sr) arrays — only the CI differs.
    """
    if len(g_rr) != len(g_sr):
        raise ValueError(
            f"g_rr and g_sr must have equal length (paired): "
            f"{len(g_rr)} vs {len(g_sr)}"
        )
    rr_m, sr_m = g_rr.mean(), g_sr.mean()
    denom = rr_m + sr_m
    sim = rr_m / denom if denom > 0 else 0.0

    n = len(g_rr)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx  = rng.integers(0, n, n)
        rr_b = g_rr[idx].mean()
        sr_b = g_sr[idx].mean()
        d    = rr_b + sr_b
        boot[b] = rr_b / d if d > 0 else 0.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return sim, float(lo), float(hi)
