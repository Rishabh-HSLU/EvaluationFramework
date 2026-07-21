"""Tests for aggregate outer-bootstrap and Monte Carlo summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from fineval.bootstrap.uncertainty import _summarize_mc, _summarize_outer


def _metric_replicates(id_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_column: [0, 1, 2],
            "metric": ["M1", "M1", "M1"],
            "generator": ["GEN", "GEN", "GEN"],
            "score": [0.4, 0.5, 0.6],
        }
    )


def _aggregate_replicates(id_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_column: [0, 1, 2],
            "generator": ["GEN", "GEN", "GEN"],
            "G_dev": [1.0, 2.0, 4.0],
            "k_used": [4, 4, 4],
            "k_total": [4, 4, 4],
        }
    )


def test_outer_summary_uses_g_dev_column() -> None:
    metric_replicates = _metric_replicates("outer_id")
    aggregate_replicates = _aggregate_replicates("outer_id")

    metric_summary, aggregate_summary = _summarize_outer(
        metric_replicates,
        aggregate_replicates,
    )

    metric_row = metric_summary.iloc[0]
    aggregate_row = aggregate_summary.iloc[0]

    expected_g_dev = aggregate_replicates["G_dev"]

    assert metric_row["outer_mean"] == pytest.approx(0.5)
    assert metric_row["n_outer_valid"] == 3

    assert aggregate_row["outer_mean"] == pytest.approx(expected_g_dev.mean())
    assert aggregate_row["ci_low"] == pytest.approx(expected_g_dev.quantile(0.025))
    assert aggregate_row["ci_high"] == pytest.approx(expected_g_dev.quantile(0.975))
    assert aggregate_row["n_outer_valid"] == 3


def test_monte_carlo_summary_uses_g_dev_column() -> None:
    metric_replicates = _metric_replicates("repeat_id")
    aggregate_replicates = _aggregate_replicates("repeat_id")

    metric_summary, aggregate_summary = _summarize_mc(
        metric_replicates,
        aggregate_replicates,
    )

    metric_row = metric_summary.iloc[0]
    aggregate_row = aggregate_summary.iloc[0]

    assert metric_row["mc_mean"] == pytest.approx(0.5)
    assert metric_row["mc_min"] == pytest.approx(0.4)
    assert metric_row["mc_max"] == pytest.approx(0.6)
    assert metric_row["n_mc_valid"] == 3

    assert aggregate_row["mc_mean"] == pytest.approx(7.0 / 3.0)
    assert aggregate_row["mc_min"] == pytest.approx(1.0)
    assert aggregate_row["mc_max"] == pytest.approx(4.0)
    assert aggregate_row["n_mc_valid"] == 3
