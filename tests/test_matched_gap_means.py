"""Tests for jointly valid matched-gap handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fineval.bootstrap import MatchedTickerBootstrap
from fineval.metrics.base import BaseMetric, matched_gap_means


class _StubMetric(BaseMetric):
    """Minimal metric used to exercise shared normalization logic."""

    def extract_features(self, sample: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def compute_distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        raise NotImplementedError


def test_matched_gap_means_uses_joint_mask() -> None:
    g_rr = np.array([1.0, 2.0, np.nan, 100.0])
    g_sr = np.array([3.0, np.nan, 5.0, 200.0])

    rr_mean, sr_mean, n_valid = matched_gap_means(g_rr, g_sr)

    # Only indices 0 and 3 are finite in both arrays.
    assert n_valid == 2
    assert rr_mean == pytest.approx(50.5)
    assert sr_mean == pytest.approx(101.5)


def test_matched_gap_means_returns_nan_without_jointly_valid_draws() -> None:
    g_rr = np.array([1.0, np.nan])
    g_sr = np.array([np.nan, 2.0])

    rr_mean, sr_mean, n_valid = matched_gap_means(g_rr, g_sr)

    assert n_valid == 0
    assert np.isnan(rr_mean)
    assert np.isnan(sr_mean)


def test_matched_gap_means_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        matched_gap_means(
            np.array([1.0, 2.0]),
            np.array([1.0]),
        )


def test_matched_gap_means_rejects_non_vector_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        matched_gap_means(
            np.ones((2, 2)),
            np.ones((2, 2)),
        )


def test_metric_normalization_uses_joint_mask() -> None:
    metric = _StubMetric("M1")

    g_rr = np.array([1.0, 100.0, np.nan])
    g_sr = np.array([3.0, np.nan, 5.0])

    # Only index 0 is jointly valid:
    # score = 1 / (1 + 3) = 0.25.
    score = metric.normalize(g_rr, g_sr)

    assert score == pytest.approx(0.25)


def test_metric_normalization_returns_nan_without_valid_draws() -> None:
    metric = _StubMetric("M1")

    score = metric.normalize(
        np.array([1.0, np.nan]),
        np.array([np.nan, 2.0]),
    )

    assert np.isnan(score)


def test_engine_summary_uses_same_draws_for_means_and_score() -> None:
    metric = _StubMetric("M1")
    engine = MatchedTickerBootstrap(metrics=[metric], n_resamples=4)

    engine.g_rr = {
        "M1": np.array([1.0, 2.0, np.nan, 100.0]),
    }
    engine.g_sr = {
        "M1": {
            "GEN": np.array([3.0, np.nan, 5.0, 200.0]),
        },
    }

    result = engine._summarize(["GEN"])
    row = result.iloc[0]

    assert row["g_rr_mean"] == pytest.approx(50.5)
    assert row["g_sr_mean"] == pytest.approx(101.5)
    assert row["score"] == pytest.approx(50.5 / (50.5 + 101.5))
    assert row["n_valid_draws"] == 2
    assert row["n_total_draws"] == 4
