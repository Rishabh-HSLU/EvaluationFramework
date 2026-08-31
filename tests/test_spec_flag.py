"""Acceptance tests for the --spec pre-specified spec selection flag."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from fineval.benchmark.artifacts import run_tag
from fineval.benchmark.config import build_metrics
from fineval.scripts.sensitivity.grid import GRID, METRIC_PARAM_KEYS, PRIMARY_PARAMS
from scripts.run_benchmark import SPECS, parse_args


def _metric_params(params) -> dict:
    """Project a full parameter set onto what ``build_metrics`` accepts.

    Mirrors ``Spec.metric_params``. ``PRIMARY_PARAMS`` is a bare mapping
    rather than a ``Spec``, so it carries no such property of its own, but
    it has the same superset problem: ``PARAM_KEYS`` includes the
    grid-level ``min_frac`` dial and ``build_metrics`` does not accept it.
    """
    return {key: params[key] for key in METRIC_PARAM_KEYS}


def test_specs_match_grid_ids() -> None:
    assert sorted(SPECS) == sorted(spec.spec_id for spec in GRID)
    assert len(SPECS) == 19
    for required in ("primary", "tail_alpha=0.75", "regime_weights=flat"):
        assert required in SPECS


def test_default_spec_is_primary() -> None:
    args = parse_args([])
    assert args.spec == "primary"
    assert args.spec_params == _metric_params(PRIMARY_PARAMS)


def test_primary_spec_reproduces_canonical_metrics() -> None:
    """Guard the drift that ``--spec primary`` would otherwise hide.

    ``PRIMARY_PARAMS`` is a hand-maintained copy of the canonical
    ``fineval.config`` constants, kept separate so the grid module stays
    import-free. Routing the canonical run through it means an edit to
    either side that is not mirrored on the other silently changes the
    numbers a ``--spec primary`` run produces, under an unchanged label.
    Compare the constructed metrics rather than the parameter dicts: the
    two differ by design on ``regime_weights`` (raw here, pre-normalized in
    ``fineval.config``) and only agree once the metric has normalized them.
    """
    canonical = build_metrics()
    primary = build_metrics(**_metric_params(PRIMARY_PARAMS))

    assert [metric.name for metric in canonical] == [metric.name for metric in primary]
    for expected, actual in zip(canonical, primary, strict=True):
        assert type(expected) is type(actual)
        assert expected.__dict__.keys() == actual.__dict__.keys()
        for attribute, value in expected.__dict__.items():
            np.testing.assert_array_equal(
                np.asarray(actual.__dict__[attribute]),
                np.asarray(value),
                err_msg=f"{expected.name}.{attribute} drifted between config and grid",
            )


def test_tail_alpha_spec_resolves_oat_params() -> None:
    args = parse_args(["--spec", "tail_alpha=0.75"])
    params = dict(SPECS[args.spec].params)
    assert params["tail_alpha"] == 0.75
    for key, value in PRIMARY_PARAMS.items():
        if key != "tail_alpha":
            assert params[key] == value


def test_flat_weights_spec_resolves_raw_weights() -> None:
    args = parse_args(["--spec", "regime_weights=flat"])
    params = dict(SPECS[args.spec].params)
    assert params["regime_weights"] == (1.0,) * 5


def test_update_canonical_rejected_for_non_primary_spec() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--spec",
                "tail_alpha=0.75",
                "--n-outer-resamples",
                "40",
                "--update-canonical",
            ]
        )


def test_unregistered_spec_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--spec", "tail_alpha=0.9"])


def test_build_metrics_receives_spec_params() -> None:
    metrics = build_metrics(**SPECS["tail_alpha=0.75"].metric_params)
    assert [metric.name for metric in metrics] == ["M1", "M2", "M3", "M4"]
    assert metrics[0].tail_alpha == 0.75

    flat_metrics = build_metrics(**SPECS["regime_weights=flat"].metric_params)
    m4 = flat_metrics[3]
    # RegimeConditionalTails normalizes raw weights to sum to one.
    assert np.allclose(m4.regime_weights, np.full(5, 0.2))


def test_run_tag_encodes_spec_without_equals() -> None:
    args = argparse.Namespace(
        spec="tail_alpha=0.75",
        n_resamples=100,
        tickers_per_draw=200,
        seed=42,
        n_outer_resamples=200,
        n_mc_repeats=0,
    )
    tag = run_tag(args, "20260802-000000-000000", 8)
    assert "spec-tail_alpha-0.75" in tag
    assert "=" not in tag
