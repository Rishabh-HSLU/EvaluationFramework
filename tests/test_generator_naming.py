"""The generator rename must preserve input access and historical RNG streams."""

import zlib
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from fineval.bootstrap.engine import MatchedTickerBootstrap
from fineval.data import AILSyntheticLoader, InDiGOSyntheticLoader
from fineval.naming import resolve_indigo_path
from fineval.scripts.sensitivity import run_sweep
from scripts.check_reproduction import compare_tables, read_csv


@pytest.mark.parametrize("current,original", [("InDiGO", "AIL"), ("indigo", "ail")])
def test_sampling_streams_preserve_original_entropy(current, original):
    engine = MatchedTickerBootstrap([], seed=42, n_resamples=3, tickers_per_draw=4)
    panel = pd.DataFrame(np.zeros((2, 8)))
    _, _, indices = engine._prepare_inner_indices(panel, {current: panel})
    expected = np.random.default_rng([42, 1, zlib.crc32(original.encode())])
    np.testing.assert_array_equal(indices[current], expected.integers(0, 8, size=(3, 4)))
    expected_outer = np.random.default_rng([42, 5, zlib.crc32(original.encode())])
    np.testing.assert_array_equal(
        engine._stream(5, current).integers(0, 600, size=(200, 600)),
        expected_outer.integers(0, 600, size=(200, 600)),
    )


def test_aliases_cannot_be_evaluated_as_distinct_generators():
    with pytest.raises(ValueError, match="collision"):
        MatchedTickerBootstrap._check_no_crc32_collision(["InDiGO", "AIL"])


def test_original_panel_names_remain_readable(tmp_path):
    current = tmp_path / "indigo_prices.parquet"
    original = tmp_path / "ail_prices.parquet"
    original.touch()
    assert resolve_indigo_path(current) == original
    current.touch()
    assert resolve_indigo_path(current) == current
    legacy_input = tmp_path / "ail_synthetic_data"
    legacy_input.mkdir()
    (legacy_input / "input.parquet").touch()
    assert resolve_indigo_path(tmp_path / "indigo_synthetic_data/input.parquet") == (
        legacy_input / "input.parquet"
    )
    assert AILSyntheticLoader is InDiGOSyntheticLoader


def test_sensitivity_hashes_accept_original_archive_name(tmp_path, monkeypatch):
    content = b"original panel bytes"
    (tmp_path / "ail_prices.parquet").write_bytes(content)
    monkeypatch.setattr(run_sweep, "CURATED_DIR", tmp_path)
    monkeypatch.setattr(run_sweep, "CURATED_FILES", ("indigo_prices.parquet",))
    assert run_sweep.dataset_hashes() == {"indigo_prices.parquet": sha256(content).hexdigest()}


def test_checker_compares_original_and_current_labels(tmp_path):
    original = tmp_path / "original.csv"
    current = tmp_path / "current.csv"
    original.write_text("metric,generator,score\nM1,AIL,0.343590\n")
    current.write_text("metric,generator,score\nM1,InDiGO,0.343590\n")
    assert compare_tables(read_csv(original), read_csv(current), ("metric", "generator"), 0, 0)[
        "passed"
    ]
