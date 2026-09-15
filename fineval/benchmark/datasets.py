"""Loading and preprocessing for the full benchmark workflow."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import pandas as pd

from fineval.data import (
    CuratedParquetLoader,
    GBMBaselineLoader,
    MSVBaselineLoader,
    SBBTSBaselineLoader,
)
from fineval.preprocessing import PreprocessingPipeline

from .config import CURATED_DIR
from .reporting import log


def _dataset_shape(dataset) -> str:
    rows, columns = dataset.prices.shape
    return f"{rows:,} timestamps × {columns:,} paths"


def load_default_datasets(progress_callback: Callable[[int], None]) -> tuple:
    """Load the curated Real panel and every available synthetic panel.

    Returns ``(real, synthetics)`` where ``synthetics`` is a list of
    ``MarketDataset`` in the fixed order AIL, GBM, MSV, SBBTS. AIL, GBM and
    MSV are required; SBBTS is included when ``sbbts_prices.parquet`` exists
    (it is produced by ``scripts.sbbts_generate`` from a trained checkpoint
    and is absent until then). Which generators were loaded is logged so a
    run's generator set is never implicit.
    """
    loaders = [
        CuratedParquetLoader(
            parquet_path=str(CURATED_DIR / "real_prices.parquet"),
            name="Real",
            is_synthetic=False,
        ),
        CuratedParquetLoader(
            parquet_path=str(CURATED_DIR / "ail_prices.parquet"),
            name="AIL",
            is_synthetic=True,
        ),
        GBMBaselineLoader(parquet_path=str(CURATED_DIR / "gbm_prices.parquet")),
        MSVBaselineLoader(parquet_path=str(CURATED_DIR / "msv_prices.parquet")),
    ]
    sbbts_path = CURATED_DIR / "sbbts_prices.parquet"
    if sbbts_path.exists():
        loaders.append(SBBTSBaselineLoader(parquet_path=str(sbbts_path)))
    else:
        log(f"SBBTS panel not found at {sbbts_path}; running without it.")

    datasets = []
    for loader in loaders:
        started = time.monotonic()
        dataset = loader.load()
        datasets.append(dataset)
        progress_callback(1)
        log(
            f"Loaded {dataset.name}: {_dataset_shape(dataset)} "
            f"in {time.monotonic() - started:.1f}s."
        )
    real, synthetics = datasets[0], datasets[1:]
    log(f"Generators: {', '.join(d.name for d in synthetics)}.")
    return real, synthetics


def preprocess_pairs(
    real,
    synthetic_datasets: Iterable,
    progress_callback: Callable[[int], None],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Preprocess each real-synthetic pair into deseasonalized returns."""
    deseas_real: pd.DataFrame | None = None
    synthetics: dict[str, pd.DataFrame] = {}

    for dataset in synthetic_datasets:
        started = time.monotonic()
        log(f"Preprocessing Real vs {dataset.name}...")
        pipeline = PreprocessingPipeline().run(real.prices, dataset.prices)
        synthetics[dataset.name] = pipeline.deseas_synthetic
        if deseas_real is None:
            deseas_real = pipeline.deseas_real
        progress_callback(1)
        log(
            f"Preprocessed {dataset.name}: real={pipeline.deseas_real.shape}, "
            f"synthetic={pipeline.deseas_synthetic.shape} "
            f"in {time.monotonic() - started:.1f}s."
        )

    if deseas_real is None:
        raise RuntimeError("no synthetic datasets were provided")
    return deseas_real, synthetics
