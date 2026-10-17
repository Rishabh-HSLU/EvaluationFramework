"""Keep published diagnostic curves consistent with benchmark feature maps."""

import numpy as np
import pandas as pd

from fineval.benchmark.config import build_metrics
from scripts.plot_metric_profiles import regime_features


def test_memory_efficient_regime_pairing_matches_canonical_metric_exactly():
    rng = np.random.default_rng(72)
    index = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(
                    day + pd.Timedelta(hours=9, minutes=31), periods=390, freq="min"
                ).values
                for day in pd.bdate_range("2020-01-02", periods=12)
            ]
        )
    )
    returns = pd.DataFrame(rng.standard_t(7, size=(len(index), 24)) * 0.001, index=index)
    returns.iloc[::390] = np.nan
    returns.iloc[41::113, 3] = np.nan
    metric = next(m for m in build_metrics() if m.name == "M4")
    expected = metric.extract_features(returns)
    actual = regime_features(metric, returns)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(actual.xi, expected)
    assert (actual.exceedances >= metric._MIN_EXCEEDANCES).all()
    assert (actual.vol_lower < actual.vol_upper).all()
