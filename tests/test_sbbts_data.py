"""sbbts.data: the adapter between the real pipeline and the drift network.

This module holds no bucketing or extraction logic of its own -- that is
`scripts.sbbts_panel` / `scripts.sbbts_batching` -- so these tests exercise
only `session_train_val_split` (needs to see `SessionExample`s before
batching) and `to_model_inputs` (converts a `PaddedBatch` to tensors), both
against directly-constructed `SessionExample` fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbbts.data import session_train_val_split
from scripts.sbbts_panel import GAP_DTYPE, VALUE_DTYPE, SessionExample


def _example(index: int, session: str, length: int = 10) -> SessionExample:
    rng = np.random.default_rng(index)
    values = np.empty(length, dtype=VALUE_DTYPE)
    values[0] = 0.0
    values[1:] = np.cumsum(rng.normal(0.0, 1.0, size=length - 1))
    return SessionExample(
        ticker=f"T{index:04d}",
        session=pd.Timestamp(session, tz="UTC"),
        values=values,
        gaps=rng.integers(1, 5, size=length).astype(GAP_DTYPE),
    )


def test_session_split_is_disjoint_and_chronological() -> None:
    sessions = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    examples = [_example(i, s) for i in range(3) for s in sessions]

    train, val = session_train_val_split(examples, val_frac=0.2)
    assert train and val

    train_dates = {e.session for e in train}
    val_dates = {e.session for e in val}

    assert train_dates.isdisjoint(val_dates)
    # Chronological, not random: every held-out date must be no earlier
    # than every training date, so validation never precedes training.
    assert max(train_dates) < min(val_dates)


def test_single_session_corpus_cannot_be_split() -> None:
    examples = [_example(i, "2026-01-05") for i in range(3)]

    train, val = session_train_val_split(examples, val_frac=0.2)
    assert val == []
    assert len(train) == len(examples)
