"""Contract test for the SBBTS baseline loader.

``SBBTSBaselineLoader`` is a thin binding over ``CuratedParquetLoader``:
it fixes the dataset name and the synthetic flag and inherits the
``BaseLoader`` validation contract. This pins that binding against a
temporary parquet file rather than the real curated panel, so the test
runs without ``data/curated`` present.
"""

from __future__ import annotations

import pandas as pd

from fineval.data import SBBTSBaselineLoader


def test_sbbts_loader_round_trips_a_curated_parquet(tmp_path) -> None:
    index = pd.date_range("2026-01-02 09:31", periods=5, freq="min", tz="UTC")
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 10.1, 10.2, 10.3, 10.4],
            "BBB": [20.0, 19.9, 19.8, 19.7, 19.6],
        },
        index=index,
    )
    parquet_path = tmp_path / "sbbts_prices.parquet"
    prices.to_parquet(parquet_path)

    dataset = SBBTSBaselineLoader(parquet_path=str(parquet_path)).load()

    assert dataset.name == "SBBTS"
    assert dataset.is_synthetic is True
    # check_freq=False: a parquet file stores index values, not the
    # DatetimeIndex freq attribute, so the reloaded index is freq=None.
    pd.testing.assert_frame_equal(dataset.prices, prices, check_freq=False)
