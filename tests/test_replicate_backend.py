"""Backend-invariance tests for the inner matched-ticker bootstrap.

The inner pool inside outer/Monte Carlo replicates runs on processes rather
than threads. That is a performance choice only: it must not move a single
number, and it must not oversubscribe the worker budget now that the two
nesting levels are both real processes. These tests pin both properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fineval.benchmark.config import build_metrics
from fineval.bootstrap.engine import MatchedTickerBootstrap
from fineval.bootstrap.execution import nested_worker_plan

MINUTES_PER_SESSION = 390

#: Twelve sessions is not arbitrary: M4 assigns regimes from a 60-minute
#: rolling volatility and then reads a 5% tail within each regime, and it
#: returns all-NaN on shorter panels. A smaller fixture would still pass
#: every equality assertion below while comparing NaN to NaN, so the
#: finiteness check in ``_assert_gaps_equal`` guards the fixture size.
N_SESSIONS = 12


def _session_index(n_sessions: int) -> pd.DatetimeIndex:
    days = [
        pd.date_range(f"2026-01-{day + 1:02d} 09:31", periods=MINUTES_PER_SESSION, freq="min")
        for day in range(n_sessions)
    ]
    return pd.DatetimeIndex(np.concatenate([day.values for day in days]))


def _panel(index: pd.DatetimeIndex, n_tickers: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(index), n_tickers)),
        index=index,
        columns=[f"T{i:04d}" for i in range(n_tickers)],
    )


@pytest.fixture(scope="module")
def corpora() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    index = _session_index(N_SESSIONS)
    return _panel(index, 24, 1), {"ail": _panel(index, 24, 2), "gbm": _panel(index, 24, 3)}


def _engine(n_jobs: int) -> MatchedTickerBootstrap:
    return MatchedTickerBootstrap(
        metrics=build_metrics(),
        n_resamples=8,
        tickers_per_draw=12,
        seed=42,
        n_jobs=n_jobs,
    )


def _assert_gaps_equal(left: MatchedTickerBootstrap, right: MatchedTickerBootstrap) -> None:
    """Compare the raw per-draw gaps, and refuse to pass on all-NaN input.

    The summary frames reduce these arrays with means, so a per-draw
    difference could average away; comparing before the reduction is what
    makes the equality claim load-bearing.
    """
    for metric in ("M1", "M2", "M3", "M4"):
        assert np.isfinite(left.g_rr[metric]).all(), f"{metric} produced no valid real-real gaps"
        np.testing.assert_array_equal(left.g_rr[metric], right.g_rr[metric])
        for generator in left.g_sr[metric]:
            assert np.isfinite(left.g_sr[metric][generator]).all()
            np.testing.assert_array_equal(
                left.g_sr[metric][generator],
                right.g_sr[metric][generator],
            )


def test_inner_backends_agree_bitwise(corpora) -> None:
    real, synthetics = corpora

    threaded = _engine(n_jobs=4)
    threaded_result = threaded.run(
        real, synthetics, show_progress=False, parallel_backend="threads"
    )
    threaded_aggregate = threaded.compute_aggregate()

    processes = _engine(n_jobs=4)
    process_result = processes.run(
        real, synthetics, show_progress=False, parallel_backend="processes"
    )
    process_aggregate = processes.compute_aggregate()

    _assert_gaps_equal(threaded, processes)
    pd.testing.assert_frame_equal(threaded_result, process_result, check_exact=True)
    pd.testing.assert_frame_equal(threaded_aggregate, process_aggregate, check_exact=True)


def test_nested_process_pools_reproduce_serial_outer_bootstrap(corpora) -> None:
    """Replicate processes each opening an inner process pool must not drift.

    ``n_jobs=1`` runs both levels sequentially in this process; ``n_jobs=8``
    plans 4 replicate processes each owning a 2-process inner pool, so this
    is also the check that nesting two ``ProcessPoolExecutor`` levels works
    at all rather than deadlocking on daemonic children.
    """
    real, synthetics = corpora
    assert nested_worker_plan(8, 4, 8) == (4, 2)

    serial = _engine(n_jobs=1).run_outer_bootstrap(
        real, synthetics, n_outer_resamples=4, show_progress=False
    )
    nested = _engine(n_jobs=8).run_outer_bootstrap(
        real, synthetics, n_outer_resamples=4, show_progress=False
    )

    assert serial.metric_replicates["score"].notna().all()
    pd.testing.assert_frame_equal(
        serial.metric_replicates, nested.metric_replicates, check_exact=True
    )
    pd.testing.assert_frame_equal(
        serial.aggregate_replicates, nested.aggregate_replicates, check_exact=True
    )
    pd.testing.assert_frame_equal(serial.metric_summary, nested.metric_summary, check_exact=True)


def test_nested_process_pools_reproduce_serial_mc_stability(corpora) -> None:
    real, synthetics = corpora

    serial = _engine(n_jobs=1).run_monte_carlo_stability(
        real,
        synthetics,
        n_repeats=4,
        base_metric_result=None,
        base_aggregate_result=None,
        show_progress=False,
    )
    nested = _engine(n_jobs=8).run_monte_carlo_stability(
        real,
        synthetics,
        n_repeats=4,
        base_metric_result=None,
        base_aggregate_result=None,
        show_progress=False,
    )

    assert serial.metric_replicates["score"].notna().all()
    pd.testing.assert_frame_equal(
        serial.metric_replicates, nested.metric_replicates, check_exact=True
    )
    pd.testing.assert_frame_equal(
        serial.aggregate_replicates, nested.aggregate_replicates, check_exact=True
    )


@pytest.mark.parametrize(
    ("total_workers", "n_replicates", "n_inner"),
    [(8, 4, 8), (8, 4, 12), (4, 2, 2), (16, 8, 8), (3, 4, 12), (8, 1, 8)],
)
def test_nested_plan_never_oversubscribes(total_workers, n_replicates, n_inner) -> None:
    """Both nesting levels are processes now, so the product is real CPUs."""
    replicate_workers, inner_workers = nested_worker_plan(total_workers, n_replicates, n_inner)
    assert replicate_workers >= 1 and inner_workers >= 1
    assert replicate_workers * inner_workers <= total_workers
