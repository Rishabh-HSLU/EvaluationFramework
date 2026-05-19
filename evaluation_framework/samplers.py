"""
samplers.py
===========
Index-only path samplers for the matched-N resampling protocol.

A Sampler's job is to choose path INDICES — never path arrays — so that
gap functions remain completely untouched by the diagnostic refactor.
The caller writes:

    idx_a, idx_b = sampler.draw_pair(n_per_half, rng)
    gap = gap_fn(real[idx_a], real[idx_b])

Three concrete samplers correspond to the variance-decomposition arms:

  - PooledSampler         : draw from full corpus (reproduces existing behavior)
  - WithinTickerSampler   : both halves from a single (randomly picked) ticker
  - WithinRegimeSampler   : both halves from one fixed regime

In addition to draw_pair (used by rr_gaps), the ABC exposes draw_single
(used by sr_gaps for the real-side draw). PooledSampler.draw_single is
written so the rng consumption pattern is bit-identical to the legacy
sr_gaps real-side `rng.choice(n_real, N, replace=False)` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np


class Sampler(ABC):
    """Abstract base."""

    @abstractmethod
    def draw_pair(
        self,
        n_per_half: int,
        rng:        np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return two DISJOINT integer index arrays into the real corpus,
        each of length n_per_half.

        Raises
        ------
        ValueError if the eligible pool is smaller than 2 * n_per_half.
        """

    @abstractmethod
    def draw_single(
        self,
        n:   int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Return one integer index array of length n into the real corpus,
        drawn without replacement from this sampler's eligible pool.
        Used by sr_gaps for the real-side draw.
        """


# ---------------------------------------------------------------------------
# PooledSampler
# ---------------------------------------------------------------------------

class PooledSampler(Sampler):
    """
    Draw from the full corpus, without replacement.

    Bit-identity contract: with the same seed,
        draw_pair(N, rng) ≡ rng.choice(n_real, 2N, replace=False) split at N
        draw_single(N, rng) ≡ rng.choice(n_real, N, replace=False)
    """

    def __init__(self, n_real: int) -> None:
        if n_real < 2:
            raise ValueError(f"n_real must be ≥ 2, got {n_real}")
        self._n_real = int(n_real)

    def draw_pair(
        self,
        n_per_half: int,
        rng:        np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if 2 * n_per_half > self._n_real:
            raise ValueError(
                f"Pooled corpus has {self._n_real} paths; cannot draw "
                f"2×{n_per_half} disjoint indices."
            )
        idx = rng.choice(self._n_real, size=2 * n_per_half, replace=False)
        return idx[:n_per_half], idx[n_per_half:]

    def draw_single(
        self,
        n:   int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n > self._n_real:
            raise ValueError(f"Requested {n} > corpus size {self._n_real}")
        return rng.choice(self._n_real, size=n, replace=False)


# ---------------------------------------------------------------------------
# WithinTickerSampler
# ---------------------------------------------------------------------------

class WithinTickerSampler(Sampler):
    """
    Pick one ticker uniformly at random per call, then draw both halves from
    that ticker's windows only. Different ticker per draw — pooling within-
    ticker variance across the ticker population.

    Eligibility: a ticker is eligible iff it has ≥ 2 * max_n_per_half windows.
    Construction raises if no tickers meet the bar.
    """

    def __init__(
        self,
        ticker_labels:  np.ndarray,
        max_n_per_half: int,
    ) -> None:
        if max_n_per_half < 1:
            raise ValueError(f"max_n_per_half must be ≥ 1, got {max_n_per_half}")

        counts = Counter(ticker_labels.tolist())
        eligible = sorted(t for t, c in counts.items()
                          if c >= 2 * max_n_per_half)

        if not eligible:
            top = counts.most_common(3)
            raise ValueError(
                f"No tickers have ≥ {2 * max_n_per_half} windows. "
                f"Largest tickers: {top}"
            )

        # Pre-compute per-ticker index arrays for O(1) lookup at draw time.
        self._eligible_tickers   = np.array(eligible)
        self._ticker_to_indices  = {
            t: np.where(ticker_labels == t)[0] for t in eligible
        }
        self._max_n_per_half     = int(max_n_per_half)

    @property
    def eligible_tickers(self) -> np.ndarray:
        return self._eligible_tickers

    def _pick_ticker(self, rng: np.random.Generator) -> str:
        return self._eligible_tickers[rng.integers(len(self._eligible_tickers))]

    def draw_pair(
        self,
        n_per_half: int,
        rng:        np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_per_half > self._max_n_per_half:
            raise ValueError(
                f"n_per_half={n_per_half} exceeds max_n_per_half="
                f"{self._max_n_per_half} fixed at construction."
            )
        ticker = self._pick_ticker(rng)
        pool   = self._ticker_to_indices[ticker]
        idx    = rng.choice(pool, size=2 * n_per_half, replace=False)
        return idx[:n_per_half], idx[n_per_half:]

    def draw_single(
        self,
        n:   int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n > self._max_n_per_half:
            raise ValueError(
                f"n={n} exceeds max_n_per_half={self._max_n_per_half}."
            )
        ticker = self._pick_ticker(rng)
        return rng.choice(self._ticker_to_indices[ticker],
                          size=n, replace=False)


# ---------------------------------------------------------------------------
# WithinRegimeSampler
# ---------------------------------------------------------------------------

class WithinRegimeSampler(Sampler):
    """
    Draw both halves from a single (fixed) regime.
    target_regime is set at construction; one instance ≡ one regime arm.
    """

    def __init__(
        self,
        regime_labels:  np.ndarray,
        target_regime:  int,
        max_n_per_half: int,
    ) -> None:
        if max_n_per_half < 1:
            raise ValueError(f"max_n_per_half must be ≥ 1, got {max_n_per_half}")

        pool = np.where(regime_labels == target_regime)[0]
        if len(pool) < 2 * max_n_per_half:
            raise ValueError(
                f"Regime {target_regime} has {len(pool)} windows, "
                f"need ≥ {2 * max_n_per_half}."
            )
        self._pool            = pool
        self._target_regime   = int(target_regime)
        self._max_n_per_half  = int(max_n_per_half)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def target_regime(self) -> int:
        return self._target_regime

    def draw_pair(
        self,
        n_per_half: int,
        rng:        np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_per_half > self._max_n_per_half:
            raise ValueError(
                f"n_per_half={n_per_half} exceeds max_n_per_half="
                f"{self._max_n_per_half} fixed at construction."
            )
        idx = rng.choice(self._pool, size=2 * n_per_half, replace=False)
        return idx[:n_per_half], idx[n_per_half:]

    def draw_single(
        self,
        n:   int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n > self._max_n_per_half:
            raise ValueError(
                f"n={n} exceeds max_n_per_half={self._max_n_per_half}."
            )
        return rng.choice(self._pool, size=n, replace=False)
