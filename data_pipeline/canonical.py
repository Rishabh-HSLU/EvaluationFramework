"""
Benchmark I/O contract shared with EvaluationFramework.

All generator outputs: deseasonalized 1-min log returns, shape (N, T, 1),
aligned to the real eval corpus volatility (pooled std), N = BENCHMARK_N_PATHS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW_LEN = 2520
BENCHMARK_N_PATHS = 200
BENCHMARK_SEED = 42
MANIFEST_NAME = "benchmark_manifest.json"


@dataclass(frozen=True)
class BenchmarkReference:
    window_len: int
    n_paths: int
    seed: int
    mean: float
    std: float

    @classmethod
    def from_eval(cls, eval_path: Path) -> BenchmarkReference:
        arr = np.load(eval_path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return cls(
            window_len=int(arr.shape[1]),
            n_paths=BENCHMARK_N_PATHS,
            seed=BENCHMARK_SEED,
            mean=float(arr.mean()),
            std=float(arr.std()),
        )

    def write_manifest(self, output_dir: Path) -> None:
        path = output_dir / MANIFEST_NAME
        path.write_text(
            json.dumps(
                {
                    "window_len": self.window_len,
                    "benchmark_n_paths": self.n_paths,
                    "benchmark_seed": self.seed,
                    "reference_mean": self.mean,
                    "reference_std": self.std,
                    "units": "deseasonalized_log_returns",
                },
                indent=2,
            )
        )


def load_reference(output_dir: str | Path) -> BenchmarkReference:
    output_dir = Path(output_dir)
    manifest = output_dir / MANIFEST_NAME
    if manifest.exists():
        d = json.loads(manifest.read_text())
        return BenchmarkReference(
            window_len=int(d["window_len"]),
            n_paths=int(d["benchmark_n_paths"]),
            seed=int(d["benchmark_seed"]),
            mean=float(d["reference_mean"]),
            std=float(d["reference_std"]),
        )
    return BenchmarkReference.from_eval(output_dir / "eval_deseasonalized.npy")


def _squeeze_paths(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"expected (N, T) or (N, T, 1), got {arr.shape}")
    return arr


def subsample_paths(
    windows: np.ndarray,
    n_paths: int,
    seed: int = BENCHMARK_SEED,
) -> np.ndarray:
    arr = _squeeze_paths(windows)
    if len(arr) < n_paths:
        raise ValueError(f"need at least {n_paths} paths, got {len(arr)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=n_paths, replace=False)
    return arr[idx]


def align_to_reference(
    windows: np.ndarray,
    reference: BenchmarkReference,
) -> np.ndarray:
    """Zero mean and match pooled std to the real eval corpus."""
    arr = _squeeze_paths(windows).copy()
    arr -= arr.mean()
    std = arr.std()
    if std > 1e-12:
        arr *= reference.std / std
    return arr


def prepare_benchmark_corpus(
    windows: np.ndarray,
    reference: BenchmarkReference,
    *,
    align_volatility: bool = True,
    n_paths: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    n_paths = reference.n_paths if n_paths is None else n_paths
    seed = reference.seed if seed is None else seed
    arr = subsample_paths(windows, n_paths, seed)
    if align_volatility:
        arr = align_to_reference(arr, reference)
    else:
        arr = arr - arr.mean()
    if arr.shape[1] != reference.window_len:
        raise ValueError(
            f"window_len {arr.shape[1]} != reference {reference.window_len}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in corpus")
    return arr


def save_benchmark_corpus(
    windows: np.ndarray,
    path: str | Path,
    output_dir: str | Path,
    *,
    align_volatility: bool = True,
) -> np.ndarray:
    """Write (N, T, 1) float32 for EvaluationFramework."""
    output_dir = Path(output_dir)
    reference = load_reference(output_dir)
    arr = prepare_benchmark_corpus(
        windows, reference, align_volatility=align_volatility
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr[:, :, np.newaxis].astype(np.float32))
    return arr
