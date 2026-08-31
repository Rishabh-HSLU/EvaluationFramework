"""Pre-specified one-at-a-time (OAT) sensitivity grid for the FinEval benchmark.

Pure data — this module performs no execution and imports nothing from the
benchmark, metrics, or bootstrap packages. Each spec carries the COMPLETE
resolved parameter set (not a delta from the primary), so this file is
self-contained as a future pre-registration artifact.

Design commitments encoded here:

- Most specs are one-at-a-time around the primary specification, and OAT
  measures main effects only. Specs with ``dial == "corner"`` are the
  deliberate exception: they move two dials at once so the no-interaction
  assumption can be tested rather than asserted. ``Spec.dials`` is the
  authoritative list of what a spec varies; ``dial`` is its label.
- The grid is one-at-a-time around the primary specification. Every sweep
  list contains the primary value; those entries collapse onto the single
  ``primary`` spec and are not re-run, giving 19 distinct specs:
  1 primary + 5 tail_alpha + 3 tail_lambda + 2 window + 2 min_frac
  + 2 tail_fraction + 1 regime_weights + 3 corners.
- ``min_periods`` is a DERIVED parameter:
  ``min_periods = ceil(window * min_frac)``. Both factors are dials, so the
  derivation holds when either varies. It is written out explicitly per spec
  rather than computed at run time, because the derivation itself is a
  pre-registration commitment (window=30 -> 15, window=60 -> 30,
  window=120 -> 60 at the primary min_frac = 1/2).
- ``regime_weights`` are stored RAW (e.g. ``(1, 1, 1, 2, 3)``); the metric
  constructor normalizes them to sum to one, exactly as
  ``fineval/config.py`` does for the canonical run.

This grid is exploratory machinery. Nothing in it is frozen, tagged, or
selected; freezing later means committing/hashing this file (and the
``grid_spec.json`` snapshot emitted by ``run_sweep.py``) as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

RUN_KIND = "exploratory"

#: The eight parameters accepted by ``fineval.benchmark.config.build_metrics``.
#: Only these may be splatted into that call.
METRIC_PARAM_KEYS = (
    "n_grid",
    "tail_alpha",
    "tail_lambda",
    "window",
    "min_periods",
    "n_regimes",
    "tail_fraction",
    "regime_weights",
)

#: Every key a spec carries. ``min_frac`` is a grid-level dial rather than a
#: metric argument: the metric takes ``min_periods``, but the parameter whose
#: provenance is AIL-informed is the *coefficient* that produces it, so the
#: coefficient is what a spec has to be able to vary.
PARAM_KEYS = (*METRIC_PARAM_KEYS, "min_frac")

#: Which metric each dial feeds. ``run_sweep`` uses this both for the fast
#: mode metric subset and for the OAT integrity check.
DIAL_TO_METRIC = MappingProxyType(
    {
        "tail_alpha": "M1",
        "tail_lambda": "M1",
        "window": "M4",
        "min_frac": "M4",
        "tail_fraction": "M4",
        "regime_weights": "M4",
    }
)

#: Dials whose variation legitimately moves a second, derived parameter.
#: ``min_periods = ceil(window * min_frac)``, so either factor moves it.
DERIVED_PARAMS = MappingProxyType(
    {
        "window": ("min_periods",),
        "min_frac": ("min_periods",),
    }
)

PRIMARY_PARAMS = MappingProxyType(
    {
        "n_grid": 5001,
        "tail_alpha": 0.3,
        "tail_lambda": 1.0,
        "window": 60,
        "min_periods": 30,  # = ceil(60 / 2)
        "n_regimes": 5,
        "tail_fraction": 0.05,
        "regime_weights": (1.0, 1.0, 1.0, 2.0, 3.0),  # raw; normalized by the metric
        "min_frac": 0.5,  # = ROLLING_VOL_MIN_FRAC; derives min_periods
    }
)


@dataclass(frozen=True)
class Spec:
    """One fully-resolved benchmark specification.

    Attributes:
        spec_id: Unique human-readable identifier.
        dial: The parameter this spec varies, or ``"none"`` for the primary.
        dial_value: The varied parameter's value (the raw tuple for
            ``regime_weights``); ``None`` for the primary.
        params: Complete parameter set — always all of ``PARAM_KEYS``,
            never a delta. Includes the grid-level ``min_frac`` dial, so it
            is a superset of what ``build_metrics`` accepts.
    """

    spec_id: str
    dial: str
    dial_value: Any
    params: dict[str, Any] = field(repr=False)
    dials: tuple[str, ...] = ()

    @property
    def metric_params(self) -> dict[str, Any]:
        """The subset of ``params`` that ``build_metrics`` accepts."""
        return {key: self.params[key] for key in METRIC_PARAM_KEYS}

    @property
    def affected_metrics(self) -> tuple[str, ...]:
        """Metrics this spec's dials can move, in M1..M4 order."""
        affected = {DIAL_TO_METRIC[dial] for dial in self.dials}
        return tuple(sorted(affected))


def _spec(dial: str, dial_value: Any, spec_id: str, **overrides: Any) -> Spec:
    params = dict(PRIMARY_PARAMS)
    params.update(overrides)
    return Spec(spec_id=spec_id, dial=dial, dial_value=dial_value, params=params, dials=(dial,))


def _corner(dials: tuple[str, ...], spec_id: str, **overrides: Any) -> Spec:
    """A deliberate off-axis spec that moves more than one dial at once.

    The grid is one-at-a-time, which measures main effects only. A claim
    that the ranking is insensitive to the parameters is a claim about the
    joint surface, and OAT cannot see interaction. These specs sit at the
    corners of the already-swept ranges so the interaction term can be read
    off directly against the corresponding single-dial specs.
    """
    params = dict(PRIMARY_PARAMS)
    params.update(overrides)
    dial_value = tuple(params[dial] for dial in dials)
    return Spec(spec_id=spec_id, dial="corner", dial_value=dial_value, params=params, dials=dials)


GRID: tuple[Spec, ...] = (
    Spec(spec_id="primary", dial="none", dial_value=None, params=dict(PRIMARY_PARAMS), dials=()),
    # tail_alpha sweep: {0, 0.15, 0.3*, 0.5, 0.75, 0.9}  (* = primary)
    _spec("tail_alpha", 0.0, "tail_alpha=0.00", tail_alpha=0.0),
    _spec("tail_alpha", 0.15, "tail_alpha=0.15", tail_alpha=0.15),
    _spec("tail_alpha", 0.5, "tail_alpha=0.50", tail_alpha=0.5),
    _spec("tail_alpha", 0.75, "tail_alpha=0.75", tail_alpha=0.75),
    # Item 5 — 0.75 was the largest value ever swept, so any claim about the
    # behaviour of tail_alpha "up to 1" rests on an untested boundary. 0.9
    # probes it: the weight stays integrable and normalizes to unit mean
    # (max weight 284 vs 119 at 0.75), so the boundary is reachable in
    # practice and its effect is measurable rather than asserted.
    _spec("tail_alpha", 0.9, "tail_alpha=0.90", tail_alpha=0.9),
    # tail_lambda sweep: {0, 0.5, 1.0*, 2.0}
    _spec("tail_lambda", 0.0, "tail_lambda=0.0", tail_lambda=0.0),
    _spec("tail_lambda", 0.5, "tail_lambda=0.5", tail_lambda=0.5),
    _spec("tail_lambda", 2.0, "tail_lambda=2.0", tail_lambda=2.0),
    # window sweep: {30, 60*, 120}; min_periods derived as ceil(window / 2)
    _spec("window", 30, "window=30", window=30, min_periods=15),
    _spec("window", 120, "window=120", window=120, min_periods=60),
    # tail_fraction sweep: {0.025, 0.05*, 0.10}
    _spec("tail_fraction", 0.025, "tail_fraction=0.025", tail_fraction=0.025),
    _spec("tail_fraction", 0.10, "tail_fraction=0.100", tail_fraction=0.10),
    # min_frac sweep: {1/3, 1/2*, 2/3} at window=60; min_periods follows.
    # Item 2 — the coefficient chosen by two AIL-scored sweeps, previously
    # pinned at 1/2 in every spec. 1/3 is the superseded optimum (mp=20) and
    # 2/3 the runner-up the re-sweep rejected on AIL-coverage grounds (mp=40).
    _spec("min_frac", 1 / 3, "min_frac=0.333", min_frac=1 / 3, min_periods=20),
    _spec("min_frac", 2 / 3, "min_frac=0.667", min_frac=2 / 3, min_periods=40),
    # Item 3 — M1 corner. Both single-dial counterparts already exist
    # (tail_alpha=0.75, tail_lambda=2.0), so the interaction term reads off
    # directly: corner - primary, minus the two main effects.
    _corner(
        ("tail_alpha", "tail_lambda"),
        "corner=alpha0.75-lambda2.0",
        tail_alpha=0.75,
        tail_lambda=2.0,
    ),
    # Item 3 — M4 corners. The two volatility-proxy dials interact through
    # min_periods = ceil(120 * 2/3) = 80, and the two tail dials both reweight
    # the same regime cells, so these are the two places an M4 interaction is
    # most likely to be non-negligible. Every single-dial counterpart exists.
    _corner(
        ("window", "min_frac"),
        "corner=win120-frac0.667",
        window=120,
        min_frac=2 / 3,
        min_periods=80,
    ),
    _corner(
        ("tail_fraction", "regime_weights"),
        "corner=tailf0.10-flat",
        tail_fraction=0.10,
        regime_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    # regime_weights sweep: {(1,1,1,2,3)*, flat}
    _spec(
        "regime_weights",
        (1.0, 1.0, 1.0, 1.0, 1.0),
        "regime_weights=flat",
        regime_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    ),
)
