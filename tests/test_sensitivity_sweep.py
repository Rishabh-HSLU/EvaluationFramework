"""Focused tests for the exploratory sensitivity-sweep harness."""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from fineval.scripts.sensitivity import run_sweep


@pytest.mark.parametrize(
    "arguments",
    [
        ["--n-resamples", "0"],
        ["--tickers-per-draw", "0"],
        ["--n-jobs", "-2"],
    ],
)
def test_invalid_execution_arguments_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        run_sweep.parse_args(arguments)


def test_failed_specs_produce_a_failing_exit(monkeypatch, tmp_path) -> None:
    """A partial result set must not be reported as a successful sweep."""
    monkeypatch.setattr(run_sweep, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_sweep, "write_grid_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        run_sweep,
        "load_default_datasets",
        lambda _log: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        run_sweep,
        "preprocess_pairs",
        lambda *_args: (pd.DataFrame(), {"synthetic": pd.DataFrame()}),
    )
    monkeypatch.setattr(
        run_sweep,
        "MatchedTickerBootstrap",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("deliberate failure")),
    )

    args = argparse.Namespace(
        n_resamples=1,
        tickers_per_draw=1,
        seed=42,
        n_jobs=1,
        fast=False,
    )
    with pytest.raises(RuntimeError, match=r"failed for 13 spec\(s\)"):
        run_sweep.run_sweep(args)

    manifest = pd.read_csv(tmp_path / "sweep_manifest.csv")
    assert manifest["status"].eq("failed").all()
    assert manifest["error"].str.contains("deliberate failure").all()
