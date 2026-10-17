"""Persistent per-specification recovery and exact sweep source provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(clean_json(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_identity() -> dict:
    paths = sorted((ROOT / "fineval").rglob("*.py"))
    paths.append(ROOT / "scripts/reproduce.py")
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def numerical_environment() -> dict:
    return {
        "python": sys.version,
        "architecture": platform.machine(),
        "libc": platform.libc_ver(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "pyarrow")
        },
        "settings": {
            name: os.environ.get(name, "")
            for name in (
                "NPY_DISABLE_CPU_FEATURES",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def save_snapshot(path: Path, snapshot: dict, *, resume: bool) -> dict:
    if resume:
        previous = json.loads(path.read_text())
        for key in (
            "run_kind",
            "n_resamples",
            "tickers_per_draw",
            "seed",
            "mode",
            "dataset_sha256",
            "specs",
            "source_sha256",
            "numerical_environment",
            "primary_reference_sha256",
        ):
            if previous.get(key) != clean_json(snapshot.get(key)):
                raise ValueError(f"Resume refused: {key} changed or was not recorded")
        return previous
    atomic_json(path, snapshot)
    return clean_json(snapshot)


def save_spec(path: Path, spec_id: str, metric_rows, aggregate_rows, diagnostics) -> None:
    atomic_json(
        path,
        {
            "spec_id": spec_id,
            "metric_rows": metric_rows.to_dict("records"),
            "aggregate_rows": aggregate_rows.to_dict("records"),
            "diagnostics": diagnostics,
        },
    )


def load_spec(path: Path, spec_id: str, generators: set[str], n_draws: int):
    if not path.exists():
        return None
    import pandas as pd

    payload = json.loads(path.read_text())
    if payload["spec_id"] != spec_id:
        raise ValueError("Checkpoint specification identity differs")
    metric = pd.DataFrame(payload["metric_rows"])
    aggregate = pd.DataFrame(payload["aggregate_rows"])
    expected = {(m, g) for m in ("M1", "M2", "M3", "M4") for g in generators}
    if (
        len(metric) != len(expected)
        or set(zip(metric.metric, metric.generator, strict=True)) != expected
    ):
        raise ValueError("Checkpoint metric cells are incomplete")
    if len(aggregate) != len(generators) or set(aggregate.generator) != generators:
        raise ValueError("Checkpoint aggregate cells are incomplete")
    if not metric.n_total_draws.eq(n_draws).all():
        raise ValueError("Checkpoint matched-draw count differs")
    return metric, aggregate, payload["diagnostics"]


def write_tables(run_dir: Path, metric_frames, aggregate_frames) -> None:
    import pandas as pd

    for name, frames in [
        ("sweep_metric_results.csv", metric_frames),
        ("sweep_aggregate_results.csv", aggregate_frames),
    ]:
        temporary = run_dir / (name + ".tmp")
        pd.concat(frames, ignore_index=True).to_csv(temporary, index=False)
        temporary.replace(run_dir / name)
