"""Structural tests for the pre-specified sensitivity grid.

The grid is a pre-registration artifact, so its shape is part of what is
being committed to. These tests pin the spec count, the one-at-a-time
guarantee, the opt-in interaction specs, and the derivation rule that
links min_periods to window and min_frac.
"""

from __future__ import annotations

import inspect
import math

import pytest

from fineval.benchmark.config import build_metrics
from fineval.config import (
    M1_TAIL_ALPHA,
    ROLLING_VOL_MIN_FRAC,
    ROLLING_VOL_MIN_PERIODS,
    ROLLING_VOL_WINDOW,
)
from fineval.scripts.sensitivity.grid import (
    DERIVED_PARAMS,
    DIAL_TO_METRIC,
    GRID,
    METRIC_PARAM_KEYS,
    PARAM_KEYS,
    PRIMARY_PARAMS,
)
from fineval.scripts.sensitivity.run_sweep import (
    assert_grid_integrity,
    assert_primary_matches_config,
)

EXPECTED_SPEC_COUNT = 19


def test_grid_has_the_declared_number_of_specs() -> None:
    assert len(GRID) == EXPECTED_SPEC_COUNT


def test_spec_ids_are_unique() -> None:
    ids = [spec.spec_id for spec in GRID]
    assert len(set(ids)) == len(ids)


def test_spec_ids_are_usable_as_path_components() -> None:
    for spec in GRID:
        assert "/" not in spec.spec_id
        assert spec.spec_id == spec.spec_id.strip()


def test_every_spec_carries_the_complete_parameter_set() -> None:
    for spec in GRID:
        assert set(spec.params) == set(PARAM_KEYS), spec.spec_id


def test_metric_params_match_the_build_metrics_signature() -> None:
    accepted = set(inspect.signature(build_metrics).parameters)
    assert set(METRIC_PARAM_KEYS) == accepted
    for spec in GRID:
        build_metrics(**spec.metric_params)


def test_min_periods_follows_the_derivation_rule() -> None:
    for spec in GRID:
        expected = math.ceil(spec.params["window"] * spec.params["min_frac"])
        assert spec.params["min_periods"] == expected, spec.spec_id


def test_primary_spec_reproduces_the_canonical_configuration() -> None:
    assert PRIMARY_PARAMS["tail_alpha"] == M1_TAIL_ALPHA
    assert PRIMARY_PARAMS["window"] == ROLLING_VOL_WINDOW
    assert PRIMARY_PARAMS["min_frac"] == ROLLING_VOL_MIN_FRAC
    assert PRIMARY_PARAMS["min_periods"] == ROLLING_VOL_MIN_PERIODS


@pytest.mark.parametrize("spec", [s for s in GRID if s.dial not in ("none", "corner")])
def test_one_at_a_time_specs_vary_exactly_one_dial(spec) -> None:
    assert spec.dials == (spec.dial,)
    allowed = {spec.dial, *DERIVED_PARAMS.get(spec.dial, ())}
    varying = {k for k in PARAM_KEYS if spec.params[k] != PRIMARY_PARAMS[k]}
    assert varying <= allowed, spec.spec_id
    assert spec.dial in varying


@pytest.mark.parametrize("spec", [s for s in GRID if s.dial == "corner"])
def test_corner_specs_vary_every_declared_dial_and_nothing_else(spec) -> None:
    assert len(spec.dials) >= 2
    allowed = set(spec.dials)
    for dial in spec.dials:
        allowed.update(DERIVED_PARAMS.get(dial, ()))
        assert spec.params[dial] != PRIMARY_PARAMS[dial], (spec.spec_id, dial)
    varying = {k for k in PARAM_KEYS if spec.params[k] != PRIMARY_PARAMS[k]}
    assert varying <= allowed, spec.spec_id


def test_every_corner_has_its_single_dial_counterparts() -> None:
    """An interaction term is only identified if both main effects are run."""
    for spec in (s for s in GRID if s.dial == "corner"):
        for dial in spec.dials:
            assert any(
                other.dials == (dial,) and other.params[dial] == spec.params[dial] for other in GRID
            ), f"{spec.spec_id}: no single-dial spec for {dial}={spec.params[dial]!r}"


def test_affected_metrics_are_derived_from_every_dial() -> None:
    for spec in GRID:
        if spec.dial == "none":
            assert spec.affected_metrics == ()
            continue
        assert set(spec.affected_metrics) == {DIAL_TO_METRIC[d] for d in spec.dials}


def test_startup_guards_pass_on_the_shipped_grid() -> None:
    assert_grid_integrity()
    assert_primary_matches_config()
