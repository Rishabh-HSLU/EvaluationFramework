"""Interrupted sweeps must preserve specs, reject changed inputs and stay exact."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from fineval.bootstrap import MatchedTickerBootstrap
from fineval.scripts.sensitivity import run_sweep
from fineval.scripts.sensitivity.grid import GRID
from fineval.scripts.sensitivity.persistence import atomic_json, save_snapshot

GENERATORS = ("AIL", "GBM", "MSV", "SBBTS")


@pytest.mark.parametrize(
    "changed", ["seed", "dataset_sha256", "specs", "source_sha256", "numerical_environment"]
)
def test_resume_rejects_changed_inputs(tmp_path, changed):
    fields = [
        "run_kind",
        "n_resamples",
        "tickers_per_draw",
        "seed",
        "mode",
        "dataset_sha256",
        "specs",
        "source_sha256",
        "numerical_environment",
    ]
    snapshot = {key: str(key) for key in fields}
    path = tmp_path / "grid.json"
    save_snapshot(path, snapshot, resume=False)
    assert save_snapshot(path, snapshot, resume=True) == snapshot
    with pytest.raises(ValueError, match=changed):
        save_snapshot(path, {**snapshot, changed: "different"}, resume=True)


def test_all_frozen_specs_fast_and_full_agree_exactly():
    index = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"2026-01-{i + 1:02d} 09:31", periods=390, freq="min").values
                for i in range(12)
            ]
        )
    )

    def panel(seed):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(rng.normal(0, 0.01, (len(index), 40)), index=index)

    real = panel(101)
    synthetics = {name: panel(i + 102) for i, name in enumerate(GENERATORS)}
    primary = None
    for spec in GRID:

        def run(fast, spec=spec):
            engine = MatchedTickerBootstrap(
                metrics=run_sweep.metrics_for(spec, fast),
                n_resamples=1,
                tickers_per_draw=40,
                seed=42,
                n_jobs=1,
            )
            result = engine.run(real, synthetics, show_progress=False)
            return result, engine

        full, engine = run(False)
        fast, _ = run(True)
        if spec.spec_id == "primary":
            primary = full
        else:
            fast = pd.concat(
                [fast, primary.loc[~primary.metric.isin(spec.affected_metrics)]], ignore_index=True
            )
        for frame in (fast, full):
            assert np.isfinite(frame.score).all(), spec.spec_id

        def ordered(frame):
            return frame.sort_values(["metric", "generator"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(ordered(fast), ordered(full), check_exact=True)
        pd.testing.assert_frame_equal(
            run_sweep.aggregate_from_metric_rows(ordered(fast))
            .sort_values("generator")
            .reset_index(drop=True),
            engine.compute_aggregate().sort_values("generator").reset_index(drop=True),
            check_exact=True,
        )


def test_interrupted_sweep_resumes_without_recomputing_primary(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sweep, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_sweep, "GRID", GRID[:2])

    def snapshot(directory, args, stamp):
        if not getattr(args, "resume", False):
            atomic_json(directory / "grid_spec.json", {"created_at": stamp})

    monkeypatch.setattr(run_sweep, "write_grid_snapshot", snapshot)
    monkeypatch.setattr(run_sweep, "load_default_datasets", lambda _: (None, None))
    monkeypatch.setattr(
        run_sweep,
        "preprocess_pairs",
        lambda *_: (pd.DataFrame(), {g: pd.DataFrame() for g in GENERATORS}),
    )
    diagnostics = {
        g: {"valid_pairs": 1000, "valid_label_sessions": 12, "regime_fits": 5}
        for g in ("Real", *GENERATORS)
    }
    monkeypatch.setattr(run_sweep, "spec_diagnostics", lambda *_: diagnostics)
    calls = []

    class Engine:
        def __init__(self, **_):
            pass

        def run(self, *_, **__):
            calls.append(len(calls))
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated shutdown")
            return pd.DataFrame(
                [
                    {
                        "metric": m,
                        "generator": g,
                        "score": 1 / 3,
                        "g_rr_mean": 1.0,
                        "g_sr_mean": 2.0,
                        "n_valid_draws": 1,
                        "n_total_draws": 1,
                    }
                    for m in run_sweep.ALL_METRICS
                    for g in GENERATORS
                ]
            )

        def compute_aggregate(self):
            return pd.DataFrame(
                [
                    {"generator": g, "G_dev": 2.0, "k_used": 4, "k_total": 4, "min_valid_draws": 1}
                    for g in GENERATORS
                ]
            )

    monkeypatch.setattr(run_sweep, "MatchedTickerBootstrap", Engine)
    args = argparse.Namespace(
        n_resamples=1,
        tickers_per_draw=1,
        seed=42,
        n_jobs=1,
        fast=False,
        resume=False,
        run_dir=tmp_path / "seed42",
    )
    with pytest.raises(KeyboardInterrupt, match="simulated shutdown"):
        run_sweep.run_sweep(args)
    assert (args.run_dir / "checkpoints/01.json").exists()
    args.resume = True
    run_sweep.run_sweep(args)
    assert len(calls) == 3
    status = json.loads((args.run_dir / "status.json").read_text())
    assert status["status"] == "completed" and len(status["completed_specs"]) == 2
    assert len(pd.read_csv(args.run_dir / "sweep_metric_results.csv")) == 32
