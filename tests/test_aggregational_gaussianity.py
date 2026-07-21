import numpy as np
import pandas as pd

from fineval.metrics import AggregationalGaussianity


def _full_session(day: str) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{day} 09:31",
        periods=390,
        freq="min",
    )


def test_all_scales_use_same_close_anchored_support() -> None:
    index = _full_session("2020-01-02")

    # Structural opening row, followed by 29 deliberately distinctive
    # observations and then the 360 observations that should be retained.
    values = np.concatenate(
        [
            [np.nan],
            np.full(29, 100.0),
            np.ones(360),
        ]
    )
    returns = pd.DataFrame({"ticker": values}, index=index)

    metric = AggregationalGaussianity(
        name="M3",
        scales=[1, 5, 15, 30],
        min_obs=1,
        support_anchor="close",
    )

    scale_1 = metric._aggregate_scale(returns, 1)
    scale_5 = metric._aggregate_scale(returns, 5)
    scale_15 = metric._aggregate_scale(returns, 15)
    scale_30 = metric._aggregate_scale(returns, 30)

    assert metric.common_support == 360

    assert scale_1.shape == (360,)
    assert scale_5.shape == (72,)
    assert scale_15.shape == (24,)
    assert scale_30.shape == (12,)

    np.testing.assert_allclose(scale_1, 1.0)
    np.testing.assert_allclose(scale_5, 5.0)
    np.testing.assert_allclose(scale_15, 15.0)
    np.testing.assert_allclose(scale_30, 30.0)


def test_open_anchor_retains_opening_support() -> None:
    index = _full_session("2020-01-02")
    values = np.concatenate(
        [
            [np.nan],
            np.full(360, 1.0),
            np.full(29, 100.0),
        ]
    )
    returns = pd.DataFrame({"ticker": values}, index=index)

    metric = AggregationalGaussianity(
        name="M3",
        scales=[1, 5, 15, 30],
        min_obs=1,
        support_anchor="open",
    )

    np.testing.assert_allclose(
        metric._aggregate_scale(returns, 1),
        1.0,
    )
