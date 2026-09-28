"""Beta calibration: beta * Delta_t_i > 1 must hold for the actual corpus.

Fixtures are directly-constructed `SessionExample` records (see
test_sbbts_batching.py's fixture idiom) rather than a standalone synthetic
corpus (see scripts/reasoning.md for why that was deleted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbbts.calibration import check_beta_constraint, derive_beta
from sbbts.data import corpus_min_delta_t
from scripts.sbbts_panel import GAP_DTYPE, VALUE_DTYPE, SessionExample


def _example(index: int, length: int, gaps: np.ndarray | None = None) -> SessionExample:
    rng = np.random.default_rng(index)
    values = np.empty(length, dtype=VALUE_DTYPE)
    values[0] = 0.0
    values[1:] = np.cumsum(rng.normal(0.0, 1.0, size=length - 1))
    if gaps is None:
        gaps = rng.integers(1, 5, size=length).astype(GAP_DTYPE)
    return SessionExample(
        ticker=f"T{index:04d}",
        session=pd.Timestamp("2026-01-05", tz="UTC"),
        values=values,
        gaps=gaps,
    )


def test_derived_beta_satisfies_the_corpus_constraint() -> None:
    examples = [_example(i, length) for i, length in enumerate([10, 15, 20, 8, 12, 30])]

    min_dt = corpus_min_delta_t(examples)
    beta = derive_beta(examples, safety_factor=2.0)

    assert beta == 2.0 / min_dt
    assert beta * min_dt > 1.0
    assert check_beta_constraint(examples, beta)


def test_beta_at_the_bare_floor_violates_the_constraint() -> None:
    """Sanity check on the constraint direction: beta == 1/min_dt exactly must fail."""
    examples = [_example(i, length) for i, length in enumerate([10, 20, 15, 25])]
    min_dt = corpus_min_delta_t(examples)

    floor_beta = 1.0 / min_dt
    assert not check_beta_constraint(examples, floor_beta)


def test_min_delta_t_excludes_the_leading_distance_from_open() -> None:
    """gaps[0] is the distance from session open, not a between-points gap."""
    # gaps[0] = 100 (far from open); every real inter-point gap is 7.
    gaps = np.array([100, 7, 7, 7], dtype=GAP_DTYPE)
    example = _example(0, length=4, gaps=gaps)

    assert corpus_min_delta_t([example]) == 7.0
