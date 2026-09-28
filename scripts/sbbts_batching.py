"""
Turn the variable-length SBBTS training examples into padded batches.

``scripts/sbbts_panel.build_training_examples`` produces one
``SessionExample`` per (ticker, session), each with its own length because
missing bars are dropped rather than filled. This module groups those
records into rectangular tensors a model can consume, and hands back the
mask that says which positions are real.

Nothing here is wired into the benchmark, and nothing under ``fineval/`` is
touched; this module is standalone.

Why bucket rather than pad everything to the maximum
----------------------------------------------------
Lengths on the real panel run from 90 to 389 with a median of 368, so the
distribution is heavily concentrated near the top. Padding every example to
the global maximum would be cheap for the bulk and ruinous for the tail, but
it still wastes work on every example that is merely close to full length.
Examples are therefore sorted into a small number of **quantile** buckets —
equal-count, not equal-width, so no bucket is nearly empty — and each bucket
is padded only to its own longest member.

Padding placement
-----------------
Padding goes at the **end** of every sequence. A causal attention mask
assumes position *i* may attend to positions <= *i*; pre-padding would shift
each real sequence to a different offset and misalign that structure. Post-
padding keeps position 0 as every sequence's true first observation.

``gaps`` pads with ``GAP_PADDING_VALUE = 1`` rather than 0, because 0 is not
a valid elapsed-minute count and a downstream consumer that forgets the mask
would silently ingest an impossible gap. A padded 1 is inert wherever the
mask is applied and merely plausible where it is not.

The model and its causal mask are deliberately not built here. This module
stops at correctly padded, correctly masked batches.

Run from the repository root:

    uv run python -m scripts.sbbts_batching
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts.sbbts_panel import REAL_PATH, SessionExample, build_training_examples

DEFAULT_N_BUCKETS = 8
DEFAULT_BATCH_SIZE = 64

VALUE_PADDING = 0.0
#: 0 is not a valid gap, so padding with it would be indistinguishable from a
#: corrupt observation. 1 is the smallest legal gap and inert under the mask.
GAP_PADDING = 1


@dataclass(frozen=True)
class PaddedBatch:
    """One rectangular batch of padded sequences.

    Attributes:
        values: (B, L) float32 padded cumulative standardized paths.
        gaps: (B, L) int32 padded elapsed-minute gaps.
        valid_mask: (B, L) bool, True at real positions and False at padding.
        lengths: (B,) int64 true length of each row.
        tickers: Ticker per row, in row order.
        sessions: Session per row, in row order.
        bucket: Index of the length bucket this batch came from.
    """

    values: torch.Tensor
    gaps: torch.Tensor
    valid_mask: torch.Tensor
    lengths: torch.Tensor
    tickers: tuple[str, ...]
    sessions: tuple[pd.Timestamp, ...]
    bucket: int

    def __len__(self) -> int:
        return int(self.values.shape[0])


def bucket_boundaries(lengths: np.ndarray, n_buckets: int) -> np.ndarray:
    """Interior quantile edges splitting ``lengths`` into ~equal-count buckets.

    Duplicate edges are collapsed, so a length distribution with a large
    point mass — the real panel has 13% of examples at exactly 389 — yields
    fewer buckets than requested rather than empty ones.

    Args:
        lengths: Per-example sequence lengths.
        n_buckets: Requested bucket count; at least 1.

    Returns:
        Sorted, unique interior boundaries. ``len(result) + 1`` buckets
        result from them.

    Raises:
        ValueError: If ``n_buckets`` is below 1 or ``lengths`` is empty.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be at least 1, got {n_buckets}")
    if lengths.size == 0:
        raise ValueError("cannot bucket an empty set of lengths")
    if n_buckets == 1:
        return np.empty(0, dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    edges = np.quantile(lengths, quantiles, method="linear")
    return np.unique(np.ceil(edges).astype(np.int64))


def assign_buckets(lengths: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Map each length to its bucket index, ``0 .. len(boundaries)``."""
    return np.searchsorted(boundaries, lengths, side="right").astype(np.int64)


def _pad_batch(
    members: list[SessionExample],
    target_length: int,
    bucket: int,
) -> PaddedBatch:
    """Pad one group of examples out to ``target_length`` at the end."""
    values = pad_sequence(
        [torch.from_numpy(example.values) for example in members],
        batch_first=True,
        padding_value=VALUE_PADDING,
    )
    gaps = pad_sequence(
        [torch.from_numpy(example.gaps) for example in members],
        batch_first=True,
        padding_value=GAP_PADDING,
    )

    # pad_sequence stops at the longest member of this group; extend to the
    # bucket's own maximum so every batch in a bucket is the same width.
    shortfall = target_length - values.shape[1]
    if shortfall > 0:
        values = F.pad(values, (0, shortfall), value=VALUE_PADDING)
        gaps = F.pad(gaps, (0, shortfall), value=GAP_PADDING)

    lengths = torch.tensor([len(example) for example in members], dtype=torch.int64)
    valid_mask = torch.arange(target_length)[None, :] < lengths[:, None]

    return PaddedBatch(
        values=values,
        gaps=gaps,
        valid_mask=valid_mask,
        lengths=lengths,
        tickers=tuple(example.ticker for example in members),
        sessions=tuple(example.session for example in members),
        bucket=bucket,
    )


def build_batches(
    examples: list[SessionExample],
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> list[PaddedBatch]:
    """Group examples into length buckets and pad each bucket to its own max.

    Every input example appears in exactly one batch; order within a bucket
    follows the input order, and buckets are emitted shortest first.

    Args:
        examples: Variable-length records from ``build_training_examples``.
        batch_size: Maximum rows per batch. The last batch of a bucket may
            be smaller.
        n_buckets: Requested number of quantile length buckets.

    Returns:
        Batches covering every input example exactly once.

    Raises:
        ValueError: If ``examples`` is empty or ``batch_size`` is below 1.
    """
    if not examples:
        raise ValueError("cannot batch an empty example list")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    lengths = np.array([len(example) for example in examples], dtype=np.int64)
    boundaries = bucket_boundaries(lengths, n_buckets)
    assignment = assign_buckets(lengths, boundaries)

    batches: list[PaddedBatch] = []
    for bucket in range(len(boundaries) + 1):
        members = np.flatnonzero(assignment == bucket)
        if members.size == 0:
            continue
        target_length = int(lengths[members].max())
        for start in range(0, members.size, batch_size):
            chunk = members[start : start + batch_size]
            batches.append(_pad_batch([examples[index] for index in chunk], target_length, bucket))
    return batches


def bucket_summary(
    examples: list[SessionExample],
    batches: list[PaddedBatch],
) -> pd.DataFrame:
    """Per-bucket length range, example count, and padding overhead."""
    rows = []
    for bucket in sorted({batch.bucket for batch in batches}):
        members = [batch for batch in batches if batch.bucket == bucket]
        lengths = torch.cat([batch.lengths for batch in members])
        real = int(lengths.sum())
        padded = sum(batch.values.numel() for batch in members)
        rows.append(
            {
                "bucket": bucket,
                "examples": int(lengths.numel()),
                "batches": len(members),
                "min_len": int(lengths.min()),
                "max_len": int(lengths.max()),
                "padded_width": int(members[0].values.shape[1]),
                "real_points": real,
                "padded_points": padded,
                "waste_pct": 100.0 * (padded - real) / padded,
            }
        )
    return pd.DataFrame(rows)


def padding_overhead(
    examples: list[SessionExample],
    batches: list[PaddedBatch],
) -> dict[str, float]:
    """Compare bucketed padding against padding everything to the global max."""
    real = sum(len(example) for example in examples)
    bucketed = sum(batch.values.numel() for batch in batches)
    global_width = max(len(example) for example in examples)
    naive = len(examples) * global_width
    return {
        "real_points": real,
        "bucketed_points": bucketed,
        "naive_points": naive,
        "global_width": global_width,
        "bucketed_waste_pct": 100.0 * (bucketed - real) / bucketed,
        "naive_waste_pct": 100.0 * (naive - real) / naive,
        "padding_avoided": naive - bucketed,
        "padding_avoided_pct": 100.0 * (naive - bucketed) / (naive - real),
    }


def main() -> None:
    print(f"Loading curated real prices: {REAL_PATH}")
    prices = pd.read_parquet(REAL_PATH)

    examples, _, dropped = build_training_examples(prices)
    print(f"Examples: {len(examples):,} ({dropped:,} dropped upstream)")

    batches = build_batches(examples)
    assert sum(len(batch) for batch in batches) == len(examples)
    print(
        f"Batches: {len(batches):,} across "
        f"{len({batch.bucket for batch in batches})} buckets "
        f"(batch_size={DEFAULT_BATCH_SIZE}, requested n_buckets={DEFAULT_N_BUCKETS})"
    )

    print()
    print(bucket_summary(examples, batches).to_string(index=False))

    print()
    overhead = padding_overhead(examples, batches)
    print(
        f"Real points        : {overhead['real_points']:,}\n"
        f"Bucketed padded    : {overhead['bucketed_points']:,} "
        f"({overhead['bucketed_waste_pct']:.1f}% padding)\n"
        f"Pad-all-to-{overhead['global_width']}    : {overhead['naive_points']:,} "
        f"({overhead['naive_waste_pct']:.1f}% padding)\n"
        f"Padding avoided    : {overhead['padding_avoided']:,} positions "
        f"({overhead['padding_avoided_pct']:.1f}% of the naive scheme's waste)"
    )


if __name__ == "__main__":
    main()
