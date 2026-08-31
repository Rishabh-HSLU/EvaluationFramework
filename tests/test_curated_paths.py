"""Curated-data path resolution tests.

Every reader and writer must agree on one curated directory, and that
directory must be derived from the repository root rather than the
process working directory. These assertions pin the location that
commit b5dfef1 moved and that the benchmark then failed to find.
"""

from __future__ import annotations

from fineval.benchmark import datasets
from fineval.benchmark.config import CURATED_DIR, ROOT_DIR
from scripts import baseline_generation, build_curated_datasets


def test_curated_dir_is_the_repository_data_directory() -> None:
    assert CURATED_DIR == ROOT_DIR / "data" / "curated"


def test_every_script_resolves_the_same_curated_directory() -> None:
    assert datasets.CURATED_DIR == CURATED_DIR
    assert build_curated_datasets.OUTPUT_DIR == CURATED_DIR
    assert baseline_generation.REAL_PATH.parent == CURATED_DIR
    assert baseline_generation.GBM_OUTPUT_PATH.parent == CURATED_DIR
    assert baseline_generation.MSV_OUTPUT_PATH.parent == CURATED_DIR


def test_raw_curation_inputs_are_repository_relative() -> None:
    assert build_curated_datasets.REAL_DIR == ROOT_DIR / "data" / "raw_intraday"
    assert build_curated_datasets.AIL_PATH.parent == ROOT_DIR / "data" / "ail_synthetic_data"


def test_curated_paths_do_not_depend_on_the_working_directory(monkeypatch, tmp_path) -> None:
    """A path built from __file__ is stable; one built from "../" is not."""
    monkeypatch.chdir(tmp_path)
    assert CURATED_DIR == ROOT_DIR / "data" / "curated"
    assert CURATED_DIR.is_absolute()
    assert build_curated_datasets.OUTPUT_DIR.is_absolute()
