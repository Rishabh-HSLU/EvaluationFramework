"""
Paths into the local data directory.

By default the framework reads from `<repo_root>/data/output_data/`. Override
with the `EVALFRAMEWORK_DATA_DIR` environment variable to point at a different
location (e.g. a shared mount or a custom build of the canonical corpus).
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "output_data"


def output_dir() -> Path:
    raw = os.environ.get("EVALFRAMEWORK_DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_OUTPUT_DIR


def real_corpus() -> Path:
    return output_dir() / "eval_deseasonalized.npy"


def manifest_path() -> Path:
    return output_dir() / "benchmark_manifest.json"


def real_ticker_labels() -> Path:
    return output_dir() / "eval_ticker_labels.npy"


def real_regime_labels() -> Path:
    return output_dir() / "eval_regime_labels.npy"


def generator_paths() -> dict[str, Path]:
    """Synthetic generator outputs alongside the real eval corpus."""
    out = output_dir()
    return {
        "AIL":    out / "ail_synthetic.npy",
        "GARCH":  out / "garch_synthetic.npy",
        "SFAGan": out / "sfagan_synthetic.npy",
        "SBBTS":  out / "sbbts_synthetic.npy",
    }


REAL_CORPUS = real_corpus  # legacy alias for test imports
