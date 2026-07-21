"""Unit tests for the symmetric aggregate deviation score G_dev."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fineval.bootstrap import MatchedTickerBootstrap
from fineval.metrics.base import BaseMetric

N_DRAWS = 64
SEED = 42


class _StubMetric(BaseMetric):
    """Name-only metric; aggregate tests use precomputed gap arrays."""

    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def compute_distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        raise NotImplementedError


def _engine(
    g_rr: dict[str, np.ndarray],
    g_sr: dict[str, dict[str, np.ndarray]],
) -> MatchedTickerBootstrap:
    """Create an engine with stored per-draw gap arrays."""
    engine = MatchedTickerBootstrap(
        metrics=[_StubMetric(name) for name in g_rr],
        n_resamples=N_DRAWS,
        seed=SEED,
    )
    engine.g_rr = g_rr
    engine.g_sr = g_sr
    return engine


def test_recovers_known_aggregate_deviation() -> None:
    # M1: r = 2 / 1 = 2
    # M2: r = 4 / 0.5 = 8
    #
    # G_dev = exp((|log 2| + |log 8|) / 2)
    #       = sqrt(2 * 8)
    #       = 4.
    engine = _engine(
        g_rr={
            "M1": np.full(N_DRAWS, 1.0),
            "M2": np.full(N_DRAWS, 0.5),
        },
        g_sr={
            "M1": {"GEN": np.full(N_DRAWS, 2.0)},
            "M2": {"GEN": np.full(N_DRAWS, 4.0)},
        },
    )

    result = engine.compute_aggregate()
    row = result.iloc[0]

    assert row["generator"] == "GEN"
    assert row["G_dev"] == pytest.approx(4.0)
    assert row["k_used"] == 2
    assert row["k_total"] == 2

    # Confidence intervals are computed by the outer-bootstrap module,
    # not by compute_aggregate().
    assert "ci_low" not in result.columns
    assert "ci_high" not in result.columns


def test_opposite_direction_ratios_do_not_cancel() -> None:
    # M1: r = 2
    # M2: r = 0.5
    #
    # Both have symmetric deviation 2, so G_dev = 2.
    engine = _engine(
        g_rr={
            "M1": np.full(N_DRAWS, 1.0),
            "M2": np.full(N_DRAWS, 2.0),
        },
        g_sr={
            "M1": {"GEN": np.full(N_DRAWS, 2.0)},
            "M2": {"GEN": np.full(N_DRAWS, 1.0)},
        },
    )

    row = engine.compute_aggregate().iloc[0]

    assert row["G_dev"] == pytest.approx(2.0)


def test_equals_one_only_when_all_ratios_equal_one() -> None:
    engine = _engine(
        g_rr={
            "M1": np.full(N_DRAWS, 1.0),
            "M2": np.full(N_DRAWS, 3.0),
        },
        g_sr={
            "M1": {"GEN": np.full(N_DRAWS, 1.0)},
            "M2": {"GEN": np.full(N_DRAWS, 3.0)},
        },
    )

    row = engine.compute_aggregate().iloc[0]

    assert row["G_dev"] == pytest.approx(1.0)


def test_aggregate_uses_jointly_valid_draws() -> None:
    engine = _engine(
        g_rr={
            "M1": np.array([1.0, 100.0, np.nan]),
        },
        g_sr={
            "M1": {
                "GEN": np.array([2.0, np.nan, 3.0]),
            },
        },
    )

    # Only index 0 is jointly valid:
    # r = 2 / 1 = 2, hence G_dev = 2.
    row = engine.compute_aggregate().iloc[0]

    assert row["G_dev"] == pytest.approx(2.0)
    assert row["k_used"] == 1
    assert row["k_total"] == 1


def test_invalid_metric_makes_aggregate_undefined() -> None:
    engine = _engine(
        g_rr={
            "M1": np.full(N_DRAWS, 1.0),
            "M2": np.full(N_DRAWS, 1.0),
        },
        g_sr={
            "M1": {"GEN": np.full(N_DRAWS, 2.0)},
            # Ratio zero is invalid for the log-deviation aggregate.
            "M2": {"GEN": np.zeros(N_DRAWS)},
        },
    )

    row = engine.compute_aggregate().iloc[0]

    assert np.isnan(row["G_dev"])
    assert row["k_used"] == 1
    assert row["k_total"] == 2


def test_all_invalid_metrics_yield_nan() -> None:
    engine = _engine(
        g_rr={
            "M1": np.zeros(N_DRAWS),
            "M2": np.full(N_DRAWS, 1.0),
        },
        g_sr={
            "M1": {"GEN": np.zeros(N_DRAWS)},
            "M2": {"GEN": np.zeros(N_DRAWS)},
        },
    )

    row = engine.compute_aggregate().iloc[0]

    assert np.isnan(row["G_dev"])
    assert row["k_used"] == 0
    assert row["k_total"] == 2


def test_multiple_generators_are_computed_independently() -> None:
    engine = _engine(
        g_rr={
            "M1": np.full(N_DRAWS, 1.0),
        },
        g_sr={
            "M1": {
                "GEN_A": np.full(N_DRAWS, 2.0),
                "GEN_B": np.full(N_DRAWS, 4.0),
            },
        },
    )

    result = engine.compute_aggregate().set_index("generator")

    assert result.loc["GEN_A", "G_dev"] == pytest.approx(2.0)
    assert result.loc["GEN_B", "G_dev"] == pytest.approx(4.0)


def test_compute_aggregate_is_deterministic() -> None:
    rng = np.random.default_rng(0)

    engine = _engine(
        g_rr={
            "M1": rng.lognormal(size=N_DRAWS),
            "M2": rng.lognormal(size=N_DRAWS),
        },
        g_sr={
            "M1": {"GEN": rng.lognormal(size=N_DRAWS)},
            "M2": {"GEN": rng.lognormal(size=N_DRAWS)},
        },
    )

    first = engine.compute_aggregate()
    second = engine.compute_aggregate()

    pd.testing.assert_frame_equal(first, second)


def test_requires_stored_gap_arrays() -> None:
    engine = MatchedTickerBootstrap(
        metrics=[_StubMetric("M1")],
        seed=SEED,
    )

    with pytest.raises(
        RuntimeError,
        match=r"compute_aggregate\(\) requires run\(\)",
    ):
        engine.compute_aggregate()
