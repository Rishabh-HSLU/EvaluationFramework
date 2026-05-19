"""
Paths into SyntheticGenerators artifacts.

Set SYNTHGEN_ROOT if the repo is not at ~/PycharmProjects/SyntheticGenerators.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_SYNTHGEN = Path.home() / "PycharmProjects" / "SyntheticGenerators"


def synthgen_root() -> Path:
    raw = os.environ.get("SYNTHGEN_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_SYNTHGEN.resolve()


def output_dir() -> Path:
    return synthgen_root() / "data" / "output_data"


def real_corpus() -> Path:
    return output_dir() / "eval_deseasonalized.npy"


def manifest_path() -> Path:
    return output_dir() / "benchmark_manifest.json"


def real_ticker_labels() -> Path:
    return output_dir() / "eval_ticker_labels.npy"


def real_regime_labels() -> Path:
    return output_dir() / "eval_regime_labels.npy"


def generator_paths() -> dict[str, Path]:
    """All benchmark synthetics live next to the real eval corpus."""
    out = output_dir()
    return {
        "AIL": out / "ail_synthetic.npy",
        "GARCH": out / "garch_synthetic.npy",
        "SFAGan": out / "sfagan_synthetic.npy",
        "SBBTS": out / "sbbts_synthetic.npy",
    }


REAL_CORPUS = real_corpus
