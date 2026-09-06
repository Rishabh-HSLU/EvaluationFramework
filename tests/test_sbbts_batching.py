"""Tests for the SBBTS padded-batch builder.

Padding is the step where a silent bug is most expensive: a mask that is off
by one, padding placed at the front, or an identity that drifts out of
alignment with its row would all train quietly and wrongly. These pin the
shape contract, the mask, the padding side, exact recoverability, bucket
conservation, and identity alignment.

Fixtures are synthetic SessionExample records, so nothing here reads the
curated panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.sbbts_batching import (
    GAP_PADDING,
    VALUE_PADDING,
    PaddedBatch,
    assign_buckets,
    bucket_boundaries,
    build_batches,
)
from scripts.sbbts_panel import GAP_DTYPE, VALUE_DTYPE, SessionExample

SESSIONS = ["2026-01-05", "2026-01-06", "2026-01-07"]


def _example(index: int, length: int) -> SessionExample:
    """One synthetic record with a unique ticker and a known length."""
    rng = np.random.default_rng(index)
    values = np.empty(length, dtype=VALUE_DTYPE)
    values[0] = 0.0
    values[1:] = np.cumsum(rng.normal(0.0, 1.0, size=length - 1))
    return SessionExample(
        ticker=f"T{index:04d}",
        session=pd.Timestamp(SESSIONS[index % len(SESSIONS)], tz="UTC"),
        values=values,
        gaps=rng.integers(1, 5, size=length).astype(GAP_DTYPE),
    )


def _examples(lengths: list[int]) -> list[SessionExample]:
    return [_example(index, length) for index, length in enumerate(lengths)]


def _spread_lengths(count: int = 97) -> list[int]:
    """Lengths spanning the real panel's 90..389 range, deliberately uneven."""
    rng = np.random.default_rng(0)
    skewed = 389 - (rng.beta(1.0, 5.0, size=count) * 299)
    return sorted(int(round(value)) for value in skewed)


def test_batch_shapes_are_internally_consistent() -> None:
    """values, gaps and valid_mask agree on shape; identities agree on count."""
    examples = _examples(_spread_lengths())

    batches = build_batches(examples, batch_size=16)

    assert batches
    for batch in batches:
        assert isinstance(batch, PaddedBatch)
        rows, width = batch.values.shape
        assert batch.gaps.shape == (rows, width)
        assert batch.valid_mask.shape == (rows, width)
        assert batch.lengths.shape == (rows,)
        assert len(batch.tickers) == len(batch.sessions) == rows == len(batch)
        assert batch.values.dtype == torch.float32
        assert batch.valid_mask.dtype == torch.bool


def test_valid_mask_marks_exactly_the_real_positions() -> None:
    """The mask is True on the first `length` positions and nowhere else."""
    lengths = [90, 91, 200, 201, 388, 389, 389, 120]
    examples = _examples(lengths)

    batches = build_batches(examples, batch_size=3, n_buckets=3)

    seen = {}
    for batch in batches:
        width = batch.values.shape[1]
        expected = torch.arange(width)[None, :] < batch.lengths[:, None]
        torch.testing.assert_close(batch.valid_mask, expected)
        assert torch.equal(batch.valid_mask.sum(dim=1), batch.lengths)
        for row, ticker in enumerate(batch.tickers):
            seen[ticker] = int(batch.lengths[row])

    assert seen == {example.ticker: len(example) for example in examples}


def test_padding_sits_at_the_end_only() -> None:
    """No padding before or inside a sequence — mask never goes False then True."""
    examples = _examples(_spread_lengths())

    for batch in build_batches(examples, batch_size=8):
        mask = batch.valid_mask.int()
        # Monotone non-increasing along each row: post-padding, never pre.
        assert torch.all(mask[:, :-1] >= mask[:, 1:]), "padding is not end-aligned"

        for row in range(len(batch)):
            length = int(batch.lengths[row])
            assert batch.valid_mask[row, :length].all()
            assert not batch.valid_mask[row, length:].any()
            assert torch.all(batch.values[row, length:] == VALUE_PADDING)
            assert torch.all(batch.gaps[row, length:] == GAP_PADDING)


def test_unmasking_recovers_the_originals_exactly() -> None:
    """Selecting the masked positions returns the input values and gaps."""
    examples = _examples(_spread_lengths())
    by_ticker = {example.ticker: example for example in examples}

    for batch in build_batches(examples, batch_size=8):
        for row, ticker in enumerate(batch.tickers):
            original = by_ticker[ticker]
            keep = batch.valid_mask[row]

            np.testing.assert_array_equal(batch.values[row][keep].numpy(), original.values)
            np.testing.assert_array_equal(batch.gaps[row][keep].numpy(), original.gaps)


def test_bucketing_neither_drops_nor_duplicates() -> None:
    """Every example appears in exactly one batch row."""
    examples = _examples(_spread_lengths())

    batches = build_batches(examples, batch_size=8)

    assert sum(len(batch) for batch in batches) == len(examples)
    produced = [
        (ticker, session)
        for batch in batches
        for ticker, session in zip(batch.tickers, batch.sessions, strict=True)
    ]
    assert len(produced) == len(set(produced)) == len(examples)
    assert set(produced) == {(example.ticker, example.session) for example in examples}


def test_identity_stays_aligned_with_its_row() -> None:
    """Row i's ticker and session belong to the record in row i, not a neighbour."""
    examples = _examples(_spread_lengths())
    by_key = {(example.ticker, example.session): example for example in examples}

    for batch in build_batches(examples, batch_size=8):
        for row in range(len(batch)):
            key = (batch.tickers[row], batch.sessions[row])
            assert key in by_key
            original = by_key[key]
            assert int(batch.lengths[row]) == len(original)
            np.testing.assert_array_equal(
                batch.values[row, : len(original)].numpy(), original.values
            )


def test_each_bucket_is_padded_to_its_own_maximum() -> None:
    """Batch width equals the longest example in that bucket, not the global max."""
    examples = _examples(_spread_lengths())

    batches = build_batches(examples, batch_size=8)

    widths = {}
    maxima = {}
    for batch in batches:
        widths.setdefault(batch.bucket, set()).add(int(batch.values.shape[1]))
        maxima[batch.bucket] = max(maxima.get(batch.bucket, 0), int(batch.lengths.max()))

    for bucket, bucket_widths in widths.items():
        assert bucket_widths == {maxima[bucket]}

    global_max = max(len(example) for example in examples)
    assert min(maxima.values()) < global_max, "bucketing gained nothing"


def test_quantile_boundaries_collapse_on_a_point_mass() -> None:
    """A heavy repeated length yields fewer buckets, never empty ones."""
    lengths = np.array([389] * 90 + [90, 120, 200, 250, 300], dtype=np.int64)

    boundaries = bucket_boundaries(lengths, n_buckets=8)
    assignment = assign_buckets(lengths, boundaries)

    assert boundaries.size == np.unique(boundaries).size
    counts = np.bincount(assignment, minlength=boundaries.size + 1)
    assert (counts > 0).all(), "an empty bucket survived"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"n_buckets": 0}, "n_buckets must be at least 1"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_batches(_examples([100, 200]), **kwargs)
