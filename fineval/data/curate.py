"""
Overview:

This module takes loaded MarketDataset objects (one real, one or more
synthetic) and produces clean, aligned, wide-format files ready for the
preprocessing layer. It performs three operations:

1. **Ticker intersection.** Only tickers present in real *and* every
   synthetic dataset are retained. Loaders are intentionally dumb about
   each other's contents; the intersection happens here.

2. **Market clock alignment.** Both sides are reindexed onto the NYSE
   regular-session 1-minute clock. Bars outside the regular session
   (extended hours, off-calendar minutes) are dropped. Missing market-clock
   minutes become NaN — preserved, not imputed.

3. **Joint coverage filter.** For each ticker, the worse-side coverage
   `min(real_cov, synthetic_cov_1, ..., synthetic_cov_k)` must meet the
   COVERAGE_FLOOR threshold (default 70%). Tickers below the floor are
   dropped from all datasets.

The output is one parquet file per dataset plus a metadata JSON. All
parquet files share the same DatetimeIndex (the market clock) and the
same column order (sorted final ticker universe).

Reproducibility
---------------
A metadata JSON is written alongside the parquet files recording every
parameter that affected curation, summary statistics of the curation
process, and SHA-256 hashes of the raw source files. A reviewer can
verify they are working with the same inputs by re-hashing their copies
of the source files.

Example
-------
    real = RealDataLoader(directory=REAL_DIR).load()
    ail  = AILSyntheticLoader(parquet_path=AIL_PATH).load()

    pipeline = CurationPipeline(
        real_dataset=real,
        synthetic_datasets=[ail],
        start_date="2019-09-03",
        end_date="2020-03-20",
        output_dir="data/curated",
    )
    pipeline.run()
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pyarrow.parquet as pq
from tqdm import tqdm

from ..config import (
    BAR_FREQUENCY,
    COVERAGE_FLOOR,
    MARKET_CALENDAR,
)
from .loader import BaseLoader, MarketDataset


@dataclass
class CurationPipeline:
    """
    End-to-end curation from loaded MarketDataset objects to evaluation-ready files.

    Attributes
    ----------
    real_dataset : MarketDataset
        The loaded real dataset. Acts as the anchor for cross-dataset
        alignment; all synthetic datasets are filtered to match its
        timestamps and ticker universe.
    synthetic_datasets : list[MarketDataset]
        One or more loaded synthetic datasets to curate alongside the
        real data. Listed to support multi-generator evaluation
        (AIL, SFAGan, SBBTS, ...).
    start_date : str
        Inclusive start of the evaluation window, ISO format (YYYY-MM-DD).
    end_date : str
        Inclusive end of the evaluation window, ISO format (YYYY-MM-DD).
    output_dir : str
        Directory where the curated parquet files and metadata JSON
        are written. Created if it does not exist.
    coverage_floor : float
        Joint-coverage threshold below which a ticker is dropped.
        Defaults to the framework-wide COVERAGE_FLOOR from config.
    source_paths : dict[str, str] | None
        Optional mapping of dataset name to source file/directory path,
        used for SHA-256 provenance hashing in the metadata. If None,
        hashes are skipped and the metadata records that fact.
    """

    real_dataset: MarketDataset
    synthetic_datasets: list[MarketDataset]
    start_date: str
    end_date: str
    output_dir: str
    coverage_floor: float = COVERAGE_FLOOR
    source_paths: dict | None = None

    def run(self) -> None:
        self._validate_inputs()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        market_clock = self._build_market_clock()

        intersection = self._intersect_tickers()
        real_subset, synth_subsets = self._subset_to_intersection(intersection)

        real_aligned = real_subset.reindex(market_clock)
        synth_aligned = {name: df.reindex(market_clock) for name, df in synth_subsets.items()}

        coverage = self._compute_joint_coverage(real_aligned, synth_aligned)
        final_universe = sorted(coverage.index[coverage["joint_min"] >= self.coverage_floor])

        real_final = real_aligned[final_universe]
        synth_final = {name: df[final_universe] for name, df in synth_aligned.items()}

        self._validate_outputs(real_final, synth_final, market_clock)
        self._save_datasets(real_final, synth_final)
        self._save_metadata(
            market_clock=market_clock,
            intersection=intersection,
            final_universe=final_universe,
            real_aligned=real_aligned,
            synth_aligned=synth_aligned,
            coverage=coverage,
        )

        print(
            f"\nCuration complete. {len(final_universe)} tickers retained "
            f"at {self.coverage_floor:.0%} joint coverage floor."
        )
        print(f"Output written to: {self.output_dir}")

    def _validate_inputs(self) -> None:
        if self.real_dataset.prices is None:
            raise ValueError("real_dataset.prices is None; load() must be called first.")
        for ds in self.synthetic_datasets:
            if ds.prices is None:
                raise ValueError(
                    f"synthetic_dataset '{ds.name}'.prices is None; load() must be called first."
                )
        if not 0 < self.coverage_floor <= 1:
            raise ValueError(f"coverage_floor must be in (0, 1], got {self.coverage_floor}")
        if not self.synthetic_datasets:
            raise ValueError("At least one synthetic dataset is required.")

    def _build_market_clock(self) -> pd.DatetimeIndex:
        calendar = mcal.get_calendar(MARKET_CALENDAR)
        schedule = calendar.schedule(start_date=self.start_date, end_date=self.end_date)
        clock = mcal.date_range(schedule, frequency=BAR_FREQUENCY)
        print(f"Market clock: {len(clock)} minutes, {clock[0]} → {clock[-1]} ({clock.tz})")
        return clock

    def _intersect_tickers(self) -> list[str]:
        real_tickers = set(self.real_dataset.prices.columns)
        intersection = real_tickers.copy()
        for ds in self.synthetic_datasets:
            intersection &= set(ds.prices.columns)

        print("\nTicker intersection:")
        print(f"  Real: {len(real_tickers)}")
        for ds in self.synthetic_datasets:
            print(f"  {ds.name}: {len(ds.prices.columns)}")
        print(f"  Intersection: {len(intersection)}")

        return sorted(intersection)

    def _subset_to_intersection(self, tickers: list[str]) -> tuple[pd.DataFrame, dict]:
        real_subset = self.real_dataset.prices[tickers]
        synth_subset = {ds.name: ds.prices[tickers] for ds in self.synthetic_datasets}
        return real_subset, synth_subset

    def _compute_joint_coverage(
        self, real_aligned: pd.DataFrame, synth_aligned: dict
    ) -> pd.DataFrame:
        coverage = pd.DataFrame({"real": real_aligned.notna().mean()})
        for name, df in synth_aligned.items():
            coverage[name] = df.notna().mean()
        coverage["joint_min"] = coverage.min(axis=1)

        print("\nJoint coverage distribution (worse side per ticker):")
        print(coverage["joint_min"].describe().to_string())

        return coverage

    def _validate_outputs(
        self, real_final: pd.DataFrame, synth_final: dict, market_clock: pd.DatetimeIndex
    ) -> None:
        for name, df in {"Real": real_final, **synth_final}.items():
            assert df.index.equals(market_clock), f"{name} index does not match market clock"
            assert df.index.tz is not None, f"{name} index must be tz-aware"
            assert list(df.columns) == sorted(df.columns), f"{name} columns must be sorted"

        first_cols = list(real_final.columns)
        for name, df in synth_final.items():
            assert list(df.columns) == first_cols, f"{name} column order differs from Real"

    def _save_datasets(self, real_final: pd.DataFrame, synth_final: dict) -> None:
        real_path = Path(self.output_dir) / "real_prices.parquet"
        real_final.to_parquet(real_path)
        print(f"Saved: {real_path} ({real_final.shape})")

        for name, df in synth_final.items():
            path = Path(self.output_dir) / f"{name.lower()}_prices.parquet"
            df.to_parquet(path)
            print(f"Saved: {path} ({df.shape})")

    def _save_metadata(
        self, market_clock, intersection, final_universe, real_aligned, synth_aligned, coverage
    ) -> None:

        def _stats(series):
            return {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std()),
            }

        metadata = {
            "build_timestamp_utc": datetime.now(UTC).isoformat(),
            "parameters": {
                "calendar": MARKET_CALENDAR,
                "frequency": BAR_FREQUENCY,
                "coverage_floor": self.coverage_floor,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
            "clock": {
                "length": len(market_clock),
                "first": str(market_clock[0]),
                "last": str(market_clock[-1]),
                "tz": str(market_clock.tz),
            },
            "ticker_counts": {
                "real_raw": self.real_dataset.prices.shape[1],
                "synthetic_raw": {ds.name: ds.prices.shape[1] for ds in self.synthetic_datasets},
                "intersection": len(intersection),
                "final_universe": len(final_universe),
            },
            "coverage_stats": {
                "real": _stats(coverage["real"]),
                **{name: _stats(coverage[name]) for name in synth_aligned},
                "joint_min": _stats(coverage["joint_min"]),
            },
            "source_file_sha256": self._hash_sources(),
        }

        meta_path = Path(self.output_dir) / "curation_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved: {meta_path}")

    def _hash_sources(self) -> dict:
        if self.source_paths is None:
            return {"note": "source_paths not provided; hashes skipped"}

        hashes = {}
        for name, path in self.source_paths.items():
            p = Path(path)
            if p.is_dir():
                hashes[name] = self._hash_directory(p)
            elif p.is_file():
                hashes[name] = self._hash_file(p)
            else:
                hashes[name] = f"path not found: {path}"
        return hashes

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _hash_directory(path: Path) -> str:
        files = sorted(path.glob("**/*"))
        combined = hashlib.sha256()
        for f in files:
            if f.is_file():
                combined.update(CurationPipeline._hash_file(f).encode())
        return combined.hexdigest()


class RealDataLoader(BaseLoader):
    """
    Loads real market data from a directory of per-ticker CSV files.

    Expects one CSV per ticker, named {ticker}.csv, each containing
    at minimum a 'timestamp' column and a 'close' column. Loads the
    full ticker universe present in the directory; cross-dataset
    alignment (intersection with synthetic tickers) is handled
    downstream, not here.

    Attributes:

        directory : str
            Path to the folder containing the CSV files.

    Example::

        loader  = RealDataLoader(directory="/path/to/raw_intraday")
        dataset = loader.load()
    """

    def __init__(self, directory: str):
        super().__init__(name="Real", is_synthetic=False)
        self.directory = directory

    def _load_raw(self) -> pd.DataFrame:
        csv_files = sorted([f for f in os.listdir(self.directory) if f.endswith(".csv")])

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.directory}")

        series_list = []
        skipped = []
        for filename in tqdm(csv_files, desc=f"Loading {self.name}"):
            ticker = filename.replace(".csv", "")
            path = os.path.join(self.directory, filename)

            cols = pd.read_csv(path, nrows=0).columns.tolist()
            if "timestamp" not in cols or "close" not in cols:
                skipped.append(ticker)
                continue

            df = pd.read_csv(
                path, usecols=["timestamp", "close"], index_col="timestamp", parse_dates=True
            )
            series_list.append(df["close"].rename(ticker))

        if skipped:
            print(
                f"Skipped {len(skipped)} files (missing 'timestamp' or 'close'): "
                f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}"
            )

        prices = pd.concat(series_list, axis=1)
        prices.index = pd.to_datetime(prices.index, utc=True)
        prices = prices.sort_index()

        return prices


class CuratedParquetLoader(BaseLoader):
    """
    Loads an evaluation-ready wide-format parquet file.

    For datasets that are already on the curated market clock — either
    the output of CurationPipeline (real_prices.parquet,
    ail_prices.parquet) or a baseline generated directly on that clock
    (gbm_prices.parquet). No reshaping is needed; the file is read,
    contract-validated by BaseLoader, and wrapped in a MarketDataset.

    Attributes:

        parquet_path : str
            Path to the wide-format parquet file.

    Example::

        loader  = CuratedParquetLoader(
            parquet_path="data/curated/real_prices.parquet",
            name="Real",
            is_synthetic=False,
        )
        dataset = loader.load()
    """

    def __init__(self, parquet_path: str, name: str, is_synthetic: bool):
        super().__init__(name=name, is_synthetic=is_synthetic)
        self.parquet_path = parquet_path

    def _load_raw(self) -> pd.DataFrame:
        prices = pd.read_parquet(self.parquet_path)
        prices.index = pd.to_datetime(prices.index, utc=True)
        return prices.sort_index()


class GBMBaselineLoader(CuratedParquetLoader):
    """
    Loads the GBM baseline dataset produced by
    fineval/scripts/baseline_generation.py.

    The GBM baseline is generated directly on the curated market clock
    with real's NaN mask imposed, so it is curated by construction and
    never passes through CurationPipeline.

    Attributes:

        parquet_path : str
            Path to the GBM baseline parquet file.

    Example::

        loader  = GBMBaselineLoader(parquet_path="data/curated/gbm_prices.parquet")
        dataset = loader.load()
    """

    def __init__(self, parquet_path: str):
        super().__init__(parquet_path=parquet_path, name="GBM", is_synthetic=True)


class AILSyntheticLoader(BaseLoader):
    """
    Loads AIL synthetic data from a long-format parquet file.

    Expects columns: tr_ric, timestamp, close. Tickers in tr_ric
    carry exchange suffixes that are stripped to plain symbols to
    match the real data's convention. Loads the full ticker universe
    present in the file; cross-dataset alignment is handled downstream.

    Attributes:

        parquet_path : str
            Path to the AIL synthetic parquet file.

    Example::

        loader  = AILSyntheticLoader(parquet_path="path/to/ail.parquet")
        dataset = loader.load()
    """

    def __init__(self, parquet_path: str):
        super().__init__(name="AIL", is_synthetic=True)
        self.parquet_path = parquet_path

    def _load_raw(self) -> pd.DataFrame:
        print("Reading parquet...")
        table = pq.read_table(self.parquet_path, columns=["tr_ric", "timestamp", "close"])
        raw = table.to_pandas()

        print("Stripping RIC suffixes...")
        # Strip any trailing exchange suffix (.N, .OQ, .A, .K, ...) generically
        raw["ticker"] = raw["tr_ric"].str.replace(r"\.\w+$", "", regex=True)

        print("Pivoting to wide format...")
        prices = raw.pivot_table(index="timestamp", columns="ticker", values="close")
        prices.index = pd.to_datetime(prices.index).tz_localize("UTC")
        prices = prices.sort_index()

        print(f"Done: {prices.shape}")
        return prices
