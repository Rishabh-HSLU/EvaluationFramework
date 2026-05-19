"""Regression: PooledSampler bit-identity and protocol reproducibility."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation_framework.buckets import BucketMarginal
from evaluation_framework.io import load_corpus
from evaluation_framework.paths import REAL_CORPUS
from evaluation_framework.protocol import rr_gaps
from evaluation_framework.samplers import PooledSampler

SEED = 42
N = 200
N_RESAMPLES = 5

_b1_gap = BucketMarginal(tail_q=0.05, n_quantile_grid=1000).compute_gap
_HAS_CORPUS = REAL_CORPUS().exists()


def _legacy_draw_pair(n_real: int, n_per_half: int, rng: np.random.Generator):
    idx = rng.choice(n_real, size=2 * n_per_half, replace=False)
    return idx[:n_per_half], idx[n_per_half:]


@unittest.skipUnless(_HAS_CORPUS, "eval corpus missing")
class TestPooledSampler(unittest.TestCase):
    def test_draw_pair_matches_rng_choice(self) -> None:
        real = load_corpus(REAL_CORPUS())
        n_real = len(real)
        rng_a = np.random.default_rng(SEED)
        rng_b = np.random.default_rng(SEED)
        sampler = PooledSampler(n_real)
        for _ in range(N_RESAMPLES):
            la, lb = _legacy_draw_pair(n_real, N, rng_a)
            sa, sb = sampler.draw_pair(N, rng_b)
            self.assertTrue(np.array_equal(la, sa))
            self.assertTrue(np.array_equal(lb, sb))
            self.assertEqual(len(np.intersect1d(la, lb)), 0)

    def test_rr_gaps_reproducible(self) -> None:
        real = load_corpus(REAL_CORPUS())
        a = rr_gaps(_b1_gap, real, N, N_RESAMPLES, np.random.default_rng(SEED))
        b = rr_gaps(_b1_gap, real, N, N_RESAMPLES, np.random.default_rng(SEED))
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
