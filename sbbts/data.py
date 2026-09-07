"""Adapter between the real SBBTS pipeline and the drift network's tensors.

This module does not build a data pipeline of its own. Trajectory
extraction and length-bucketed batching are `scripts.sbbts_panel` and
`scripts.sbbts_batching` -- they turn the curated real panel into
`SessionExample` records and then `PaddedBatch` objects. This module only
adapts a `PaddedBatch`'s (values, gaps) representation into the
(y_seq, times, valid_mask) tensors `sbbts.model.SBBTSDrift` expects, and
holds the one piece of corpus-level logic that has to see `SessionExample`
records before they are batched: the validation split and beta's minimum
elapsed-time gap.

An earlier pass built a standalone synthetic corpus and its own bucketing
logic here, before `scripts.sbbts_panel`/`scripts.sbbts_batching` existed.
That has been deleted outright rather than kept alongside the real
pipeline; see scripts/reasoning.md for the record.
"""

from __future__ import annotations

import torch

from scripts.sbbts_batching import PaddedBatch
from scripts.sbbts_panel import SessionExample


def to_model_inputs(batch: PaddedBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a real `PaddedBatch` into (y_seq, times, valid_mask).

    `PaddedBatch.gaps` holds the elapsed minutes since the *previous*
    surviving bar (`gaps[0]` is the first survivor's distance from the
    session open), not an absolute elapsed time. The cumulative sum over
    each row recovers actual elapsed time since the session open -- the
    representation `sbbts.model.TrajectoryEncoder` and `SBBTSDrift.drift`
    expect. Padded gap entries are 1 (never 0, see sbbts_batching), so the
    cumulative sum keeps climbing through padding; that is harmless, since
    padded positions never reach a masked attention output or an
    unmasked loss term.

    `PaddedBatch.values` has no trailing feature axis (state_dim=1,
    implicit); `SBBTSDrift` expects one, so it is added here.
    """
    y_seq = batch.values.unsqueeze(-1).to(torch.float32)
    times = torch.cumsum(batch.gaps.to(torch.float32), dim=-1)
    return y_seq, times, batch.valid_mask


def corpus_min_delta_t(examples: list[SessionExample]) -> float:
    """Minimum elapsed-time gap between consecutive training points.

    Used to derive beta: Theorem 3.2 / Algorithm 1 require
    `beta * Delta_t_i > 1` for every step. A `SessionExample`'s `gaps[i]`
    for `i >= 1` *is* `Delta_t_{i-1}` (the elapsed time between its i-1th
    and ith surviving bars); `gaps[0]` is the distance from the session
    open to the first survivor, not a between-training-points gap, and is
    excluded.
    """
    min_dt = float("inf")
    for example in examples:
        if len(example) < 2:
            continue
        min_dt = min(min_dt, float(example.gaps[1:].min()))
    if not (min_dt < float("inf")):
        raise ValueError("No example in the corpus has more than one surviving bar.")
    return min_dt


def session_train_val_split(
    examples: list[SessionExample], val_frac: float
) -> tuple[list[SessionExample], list[SessionExample]]:
    """Hold out the most recent `val_frac` fraction of session dates.

    Splitting by session date -- not by individual example -- keeps every
    ticker's example from a given trading day on one side of the split, so
    no minute-level information from a held-out session leaks into
    training. The split is chronological (the latest dates are held out),
    not a random shuffle: validation sessions are always in the future
    relative to training sessions, avoiding lookahead leakage.

    A corpus with only one distinct session date cannot be split this way;
    the entire corpus is returned as training data with an empty
    validation set in that case (expected for small synthetic smoke tests
    only -- the real corpus spans many sessions).
    """
    unique_dates = sorted({example.session for example in examples})
    if len(unique_dates) < 2:
        return list(examples), []

    n_val_dates = max(1, round(len(unique_dates) * val_frac))
    n_val_dates = min(n_val_dates, len(unique_dates) - 1)  # keep at least one training date
    val_dates = set(unique_dates[len(unique_dates) - n_val_dates :])

    train = [e for e in examples if e.session not in val_dates]
    val = [e for e in examples if e.session in val_dates]
    return train, val
