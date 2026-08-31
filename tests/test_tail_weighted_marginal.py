import numpy as np
import pandas as pd
import pytest

from fineval.metrics import TailWeightedMarginal

# M1 pools every observation and ignores temporal ordering, so the panels
# below carry a plain RangeIndex rather than a market clock.
N_GRID = 101


def _metric() -> TailWeightedMarginal:
    return TailWeightedMarginal(
        name="M1",
        n_grid=N_GRID,
        tail_alpha=0.3,
        tail_lambda=1.0,
    )


def _panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((200, 3)),
        columns=["alpha", "beta", "gamma"],
    )


def test_gap_is_invariant_to_rescaling_a_panel() -> None:
    metric = _metric()
    features_real = metric.extract_features(_panel(0))
    synthetic = _panel(1)

    baseline = metric.compute_distance(
        features_real,
        metric.extract_features(synthetic),
    )
    rescaled = metric.compute_distance(
        features_real,
        metric.extract_features(synthetic * 7.5),
    )

    # Guard against the invariance holding only because the gap is zero.
    assert baseline > 0.0
    np.testing.assert_allclose(rescaled, baseline)


def test_panels_differing_only_in_per_column_scale_have_zero_gap() -> None:
    metric = _metric()
    panel = _panel(0)

    # Each column is scaled by a different factor, so only a per-column
    # standardization can collapse the two panels onto each other.
    rescaled = panel * np.array([1.0, 10.0, 0.1])

    gap = metric.compute_distance(
        metric.extract_features(panel),
        metric.extract_features(rescaled),
    )

    np.testing.assert_allclose(gap, 0.0, atol=1e-12)


def test_degenerate_columns_are_named_in_the_error() -> None:
    metric = _metric()

    constant = _panel(0)
    constant["beta"] = 4.0
    with pytest.raises(ValueError, match=r"\['beta'\]"):
        metric.extract_features(constant)

    missing = _panel(0)
    missing["gamma"] = np.nan
    with pytest.raises(ValueError, match=r"\['gamma'\]"):
        metric.extract_features(missing)


def test_missing_entries_are_dropped_not_imputed() -> None:
    metric = _metric()

    # The gaps sit in the same rows of both columns, so dropping those rows
    # leaves each column's observed values untouched.
    scattered = pd.DataFrame(
        {
            "alpha": [1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0, 6.0],
            "beta": [2.0, 4.0, np.nan, 8.0, 3.0, np.nan, 7.0, 5.0],
        }
    )

    np.testing.assert_allclose(
        metric.extract_features(scattered),
        metric.extract_features(scattered.dropna()),
    )

    # Filling the gaps instead of dropping them would move the quantiles.
    assert not np.allclose(
        metric.extract_features(scattered),
        metric.extract_features(scattered.fillna(0.0)),
    )


def test_feature_vector_has_grid_length() -> None:
    metric = _metric()

    features = metric.extract_features(_panel(0))
    assert features.shape == (N_GRID,)
    assert np.isfinite(features).all()

    # A panel with no columns pools no observations at all, so the
    # all-invalid short circuit returns an all-NaN vector of the same
    # length rather than raising.
    empty = metric.extract_features(pd.DataFrame(index=range(8)))
    assert empty.shape == (N_GRID,)
    assert np.isnan(empty).all()
