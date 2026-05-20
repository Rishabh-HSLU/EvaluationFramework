"""
Data preparation pipeline for the evaluation framework.

`data_prep` cleans raw 1-minute CSVs into the canonical windowed format used
by the benchmark. `canonical` defines the manifest contract every corpus
must obey. `regen_eval_labels` rebuilds the label companion arrays when only
they need updating.
"""

from .canonical import (
    BENCHMARK_N_PATHS,
    BENCHMARK_SEED,
    MANIFEST_NAME,
    WINDOW_LEN,
    BenchmarkReference,
    align_to_reference,
    load_reference,
    prepare_benchmark_corpus,
    save_benchmark_corpus,
    subsample_paths,
)
from .data_prep import (
    CRASH_CUTOFF,
    build_dataset,
    clean_ticker,
    deseasonalize,
    fit_fff,
    make_windows,
)

__all__ = [
    "BENCHMARK_N_PATHS",
    "BENCHMARK_SEED",
    "MANIFEST_NAME",
    "WINDOW_LEN",
    "BenchmarkReference",
    "CRASH_CUTOFF",
    "align_to_reference",
    "build_dataset",
    "clean_ticker",
    "deseasonalize",
    "fit_fff",
    "load_reference",
    "make_windows",
    "prepare_benchmark_corpus",
    "save_benchmark_corpus",
    "subsample_paths",
]
