"""Unit tests for the aggregate fidelity score G on MatchedTickerBootstrap."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from fineval.bootstrap import MatchedTickerBootstrap
from fineval.metrics.base import BaseMetric

B = 64
SEED = 42


class _StubMetric(BaseMetric):
    """Name-only stub; the aggregate reads stored arrays, never these."""

    def extract_features(self, sample):
        raise NotImplementedError

    def compute_distance(self, fa, fb):
        raise NotImplementedError


def _engine(g_rr: dict, g_sr: dict) -> MatchedTickerBootstrap:
    """Engine with pre-filled per-draw arrays, as if run() had happened."""
    engine = MatchedTickerBootstrap(metrics=[_StubMetric(name) for name in g_rr], seed=SEED)
    engine.g_rr = g_rr
    engine.g_sr = g_sr
    return engine


class TestAggregatePointEstimate(unittest.TestCase):
    def test_recovers_known_geometric_mean(self) -> None:
        # r_M1 = 2/1 = 2, r_M2 = 4/0.5 = 8 -> G = sqrt(2 * 8) = 4
        engine = _engine(
            g_rr={"M1": np.full(B, 1.0), "M2": np.full(B, 0.5)},
            g_sr={"M1": {"GEN": np.full(B, 2.0)}, "M2": {"GEN": np.full(B, 4.0)}},
        )
        df = engine.compute_aggregate()
        row = df.iloc[0]
        self.assertEqual(row["generator"], "GEN")
        self.assertAlmostEqual(row["G"], 4.0)
        self.assertEqual(row["k_used"], 2)
        self.assertEqual(row["k_total"], 2)
        # Constant arrays: every joint resample reproduces G exactly.
        self.assertAlmostEqual(row["ci_low"], 4.0)
        self.assertAlmostEqual(row["ci_high"], 4.0)

    def test_deterministic_under_fixed_seed(self) -> None:
        rng = np.random.default_rng(0)
        engine = _engine(
            g_rr={"M1": rng.lognormal(size=B), "M2": rng.lognormal(size=B)},
            g_sr={
                "M1": {"GEN": rng.lognormal(size=B)},
                "M2": {"GEN": rng.lognormal(size=B)},
            },
        )
        first = engine.compute_aggregate()
        second = engine.compute_aggregate()
        self.assertTrue(first.equals(second))

    def test_requires_run_first(self) -> None:
        engine = MatchedTickerBootstrap(metrics=[_StubMetric("M1")], seed=SEED)
        with self.assertRaises(RuntimeError):
            engine.compute_aggregate()


class TestAggregateDegenerateMetrics(unittest.TestCase):
    def test_zero_and_nan_ratios_are_excluded_not_fatal(self) -> None:
        # M1 healthy (r = 2); MZ has r = 0 (zero synthetic gap);
        # MN has r = 0/0 = NaN. Both must drop out of the product.
        engine = _engine(
            g_rr={
                "M1": np.full(B, 1.0),
                "MZ": np.full(B, 1.0),
                "MN": np.zeros(B),
            },
            g_sr={
                "M1": {"GEN": np.full(B, 2.0)},
                "MZ": {"GEN": np.zeros(B)},
                "MN": {"GEN": np.zeros(B)},
            },
        )
        df = engine.compute_aggregate()
        row = df.iloc[0]
        self.assertAlmostEqual(row["G"], 2.0)
        self.assertEqual(row["k_used"], 1)
        self.assertEqual(row["k_total"], 3)
        self.assertAlmostEqual(row["ci_low"], 2.0)
        self.assertAlmostEqual(row["ci_high"], 2.0)

    def test_all_degenerate_yields_nan_not_raise(self) -> None:
        engine = _engine(
            g_rr={"MZ": np.full(B, 1.0), "MN": np.zeros(B)},
            g_sr={"MZ": {"GEN": np.zeros(B)}, "MN": {"GEN": np.zeros(B)}},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN reductions
            df = engine.compute_aggregate()
        row = df.iloc[0]
        self.assertTrue(np.isnan(row["G"]))
        self.assertEqual(row["k_used"], 0)
        self.assertTrue(np.isnan(row["ci_low"]))
        self.assertTrue(np.isnan(row["ci_high"]))


class TestJointCI(unittest.TestCase):
    def test_joint_ci_narrower_than_independent_under_shared_shock(self) -> None:
        """The joint index draw must preserve inter-metric correlation.

        A shared per-draw shock s_b enters M1's noise floor and M2's
        generator gap (corr = +1 between those arrays, the analogue of
        the shared real draw A_b). In log space the two gap ratios then
        cancel exactly under a *joint* index draw (ln r_1 + ln r_2 = 0,
        so G = 1 identically), while independent per-metric draws leave
        residual variance. A joint CI strictly narrower than the
        independent one proves the mechanism is not silently
        degenerating to independent resampling.
        """
        shock = np.random.default_rng(0).lognormal(mean=0.0, sigma=1.0, size=B)
        ones = np.ones(B)
        g_rr = {"M1": shock.copy(), "M2": ones.copy()}
        g_sr = {"M1": {"GEN": ones.copy()}, "M2": {"GEN": shock.copy()}}
        # The known positive inter-metric correlation this test relies on:
        self.assertAlmostEqual(np.corrcoef(g_rr["M1"], g_sr["M2"]["GEN"])[0, 1], 1.0)

        engine = _engine(g_rr, g_sr)
        lo, hi = engine._joint_paired_ci_aggregate("GEN", engine._stream(3, "GEN"))
        joint_width = hi - lo

        # Independent comparator: same estimator, but each metric gets
        # its own resample index, destroying the cross-metric pairing.
        rng = np.random.default_rng(SEED)
        idx1 = rng.integers(0, B, size=(engine.n_ci_boot, B))
        idx2 = rng.integers(0, B, size=(engine.n_ci_boot, B))
        r1 = np.mean(ones[idx1], axis=1) / np.mean(shock[idx1], axis=1)
        r2 = np.mean(shock[idx2], axis=1) / np.mean(ones[idx2], axis=1)
        boot = np.exp((np.log(r1) + np.log(r2)) / 2.0)
        indep_lo, indep_hi = np.percentile(boot, [2.5, 97.5])
        indep_width = indep_hi - indep_lo

        self.assertGreater(indep_width, 0.01)
        self.assertLess(joint_width, indep_width)
        # And the joint distribution is degenerate at the true G = 1.
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main()
