"""
Read- and validate-side of the benchmark corpus contract.

The write side (BenchmarkReference, save_benchmark_corpus, align_to_reference)
lives in `data_pipeline.canonical`. This module re-exports the dataclass and
manifest constants so callers in the evaluation package can stay decoupled
from the data-prep package's internals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_pipeline.canonical import (
    MANIFEST_NAME,
    BenchmarkReference,
    load_reference as load_manifest,
)

STD_TOLERANCE = 0.20  # warn if |corpus_std - ref_std| / ref_std exceeds this

__all__ = [
    "MANIFEST_NAME",
    "STD_TOLERANCE",
    "BenchmarkReference",
    "load_manifest",
    "validate_corpus",
    "format_corpus_line",
]


def validate_corpus(
    arr:       np.ndarray,
    reference: BenchmarkReference,
    name:      str,
) -> dict[str, float]:
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected (N, T), got {arr.shape}")
    if arr.shape[1] != reference.window_len:
        raise ValueError(
            f"{name}: T={arr.shape[1]} != {reference.window_len}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite values")
    stats = {"mean": float(arr.mean()), "std": float(arr.std()), "n": float(len(arr))}
    rel = abs(stats["std"] - reference.std) / max(reference.std, 1e-12)
    if rel > STD_TOLERANCE:
        print(
            f"  WARN {name}: std={stats['std']:.4f} vs reference "
            f"{reference.std:.4f} (rel err {rel:.1%})"
        )
    return stats


def format_corpus_line(name: str, stats: dict[str, float]) -> str:
    return (
        f"  {name:<8} n={int(stats['n']):>4}  "
        f"mean={stats['mean']:+.5f}  std={stats['std']:.4f}"
    )
