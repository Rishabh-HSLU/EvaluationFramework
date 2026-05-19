"""Benchmark corpus contract (mirrors SyntheticGenerators/data/canonical.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANIFEST_NAME = "benchmark_manifest.json"
STD_TOLERANCE = 0.20  # warn if |syn_std - ref_std| / ref_std exceeds this


@dataclass(frozen=True)
class BenchmarkReference:
    window_len: int
    n_paths: int
    seed: int
    mean: float
    std: float


def load_manifest(output_dir: Path) -> BenchmarkReference:
    p = output_dir / MANIFEST_NAME
    if not p.exists():
        eval_path = output_dir / "eval_deseasonalized.npy"
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Missing {MANIFEST_NAME} and eval corpus under {output_dir}"
            )
        arr = np.load(eval_path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return BenchmarkReference(
            window_len=int(arr.shape[1]),
            n_paths=200,
            seed=42,
            mean=float(arr.mean()),
            std=float(arr.std()),
        )
    d = json.loads(p.read_text())
    return BenchmarkReference(
        window_len=int(d["window_len"]),
        n_paths=int(d["benchmark_n_paths"]),
        seed=int(d["benchmark_seed"]),
        mean=float(d["reference_mean"]),
        std=float(d["reference_std"]),
    )


def validate_corpus(
    arr: np.ndarray,
    reference: BenchmarkReference,
    name: str,
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
    return f"  {name:<8} n={int(stats['n']):>4}  mean={stats['mean']:+.5f}  std={stats['std']:.4f}"
