"""Process workers for outer-bootstrap and Monte Carlo replicates.

Each replicate here opens its own inner matched-ticker pool, so execution is
two levels of processes deep. Three properties make that safe:

- ``ProcessPoolExecutor`` workers are non-daemonic, so a worker may create a
  pool of its own (unlike ``multiprocessing.Pool``, whose daemonic children
  cannot).
- ``nested_worker_plan`` picks the two pool sizes from one budget under
  ``replicate_workers * inner_workers <= total_workers``, so the two levels
  together never exceed the requested worker count.
- Results stay deterministic regardless of pool type or completion order:
  every draw index is generated before any work is submitted, and each chunk
  is written back at its own offset rather than appended on arrival.

The inner pool uses the process backend rather than threads because the
per-draw work is NumPy/pandas feature extraction that holds the GIL, which
caps thread scaling well below the worker budget. The cost is that the inner
pool is rebuilt once per replicate and its initializer re-serializes the
resampled panels to each inner worker, so the trade is only favourable while
a replicate's ``n_resamples`` draws outweigh that fixed per-replicate setup.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .engine import MatchedTickerBootstrap
from .execution import subsample_panel

_OUTER_CONTEXT: dict = {}
_MC_CONTEXT: dict = {}


def init_outer_worker(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics,
    n_resamples: int,
    tickers_per_draw: int,
    seed: int,
    inner_workers: int,
    inner_chunk_size: int | None,
) -> None:
    global _OUTER_CONTEXT
    _OUTER_CONTEXT = {
        "real": real,
        "synthetics": synthetics,
        "metrics": metrics,
        "n_resamples": n_resamples,
        "tickers_per_draw": tickers_per_draw,
        "seed": seed,
        "inner_workers": inner_workers,
        "inner_chunk_size": inner_chunk_size,
    }


def evaluate_outer_chunk(
    payload: list[tuple[int, np.ndarray, dict[str, np.ndarray]]],
) -> tuple[list[dict], list[dict]]:
    """Evaluate a chunk of complete-corpus bootstrap replicates."""
    context = _OUTER_CONTEXT
    metric_records: list[dict] = []
    aggregate_records: list[dict] = []

    for outer_id, real_idx, synth_idx in payload:
        outer_real = subsample_panel(context["real"], real_idx)
        outer_synthetics = {
            generator: subsample_panel(synthetic, synth_idx[generator])
            for generator, synthetic in context["synthetics"].items()
        }
        child = MatchedTickerBootstrap(
            metrics=context["metrics"],
            n_resamples=context["n_resamples"],
            tickers_per_draw=context["tickers_per_draw"],
            seed=context["seed"],
            n_jobs=context["inner_workers"],
            inner_chunk_size=context["inner_chunk_size"],
        )
        metric_result = child.run(
            outer_real,
            outer_synthetics,
            show_progress=False,
            parallel_backend="processes",
        )
        aggregate_result = child.compute_aggregate()
        metric_records.extend(
            {"outer_id": outer_id, "inner_seed": context["seed"], **record}
            for record in metric_result.to_dict("records")
        )
        aggregate_records.extend(
            {"outer_id": outer_id, "inner_seed": context["seed"], **record}
            for record in aggregate_result.to_dict("records")
        )

    return metric_records, aggregate_records


def init_mc_worker(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics,
    n_resamples: int,
    tickers_per_draw: int,
    inner_workers: int,
    inner_chunk_size: int | None,
) -> None:
    global _MC_CONTEXT
    _MC_CONTEXT = {
        "real": real,
        "synthetics": synthetics,
        "metrics": metrics,
        "n_resamples": n_resamples,
        "tickers_per_draw": tickers_per_draw,
        "inner_workers": inner_workers,
        "inner_chunk_size": inner_chunk_size,
    }


def evaluate_mc_chunk(
    payload: list[tuple[int, int]],
) -> tuple[list[dict], list[dict]]:
    """Evaluate a chunk of independent-seed repeats."""
    context = _MC_CONTEXT
    metric_records: list[dict] = []
    aggregate_records: list[dict] = []

    for repeat_id, repeat_seed in payload:
        child = MatchedTickerBootstrap(
            metrics=context["metrics"],
            n_resamples=context["n_resamples"],
            tickers_per_draw=context["tickers_per_draw"],
            seed=repeat_seed,
            n_jobs=context["inner_workers"],
            inner_chunk_size=context["inner_chunk_size"],
        )
        metric_result = child.run(
            context["real"],
            context["synthetics"],
            show_progress=False,
            parallel_backend="processes",
        )
        aggregate_result = child.compute_aggregate()
        metric_records.extend(
            {"repeat_id": repeat_id, "seed": repeat_seed, **record}
            for record in metric_result.to_dict("records")
        )
        aggregate_records.extend(
            {"repeat_id": repeat_id, "seed": repeat_seed, **record}
            for record in aggregate_result.to_dict("records")
        )

    return metric_records, aggregate_records


def run_replicate_chunks(
    *,
    payloads: list[list],
    n_replicates: int,
    workers: int,
    initializer,
    initargs: tuple,
    worker,
    description: str,
    id_column: str,
    inner_draws: int,
    show_progress: bool,
    progress_callback: Callable[[int], None] | None,
    progress_position: int,
    leave_progress: bool,
) -> tuple[list[dict], list[dict]]:
    """Run replicate chunks sequentially or in a process pool."""
    metric_records: list[dict] = []
    aggregate_records: list[dict] = []

    def store(result: tuple[list[dict], list[dict]]) -> int:
        metric_chunk, aggregate_chunk = result
        metric_records.extend(metric_chunk)
        aggregate_records.extend(aggregate_chunk)
        completed = len({record[id_column] for record in metric_chunk})
        if progress_callback is not None:
            progress_callback(completed * inner_draws)
        return completed

    with tqdm(
        total=n_replicates,
        desc=description,
        disable=not show_progress,
        position=progress_position,
        leave=leave_progress,
        unit="replicate",
    ) as progress:
        if workers == 1:
            initializer(*initargs)
            for payload in payloads:
                progress.update(store(worker(payload)))
            return metric_records, aggregate_records

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            futures = [executor.submit(worker, payload) for payload in payloads]
            for future in as_completed(futures):
                progress.update(store(future.result()))

    return metric_records, aggregate_records
