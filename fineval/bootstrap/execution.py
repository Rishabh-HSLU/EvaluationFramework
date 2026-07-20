"""Deterministic execution helpers for matched ticker draws.

This module owns multiprocessing, chunking, and low-level draw evaluation.
It intentionally contains no uncertainty-analysis or reporting logic.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..metrics.base import BaseMetric

_INNER_PROCESS_CONTEXT: dict = {}


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve an explicit or conservative automatic worker budget."""
    cpu_count = os.cpu_count() or 1
    if n_jobs in (None, 0):
        return max(1, min(8, cpu_count - 1 if cpu_count > 1 else 1))
    if n_jobs == -1:
        return cpu_count
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1, 0, None, or a positive integer")
    return n_jobs


def chunk_bounds(
    n_items: int,
    n_workers: int,
    chunk_size: int | None,
) -> list[tuple[int, int]]:
    """Return deterministic contiguous chunks covering ``n_items``."""
    if n_items < 1:
        return []
    if chunk_size is None:
        chunk_size = max(1, math.ceil(n_items / max(1, n_workers * 4)))
    return [(start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)]


def nested_worker_plan(
    total_workers: int,
    n_replicates: int,
    n_inner: int,
) -> tuple[int, int]:
    """Split a bounded worker budget across replicates and inner draws."""
    total_workers = max(1, total_workers)
    if n_replicates <= 1:
        return 1, min(total_workers, max(1, n_inner))

    if total_workers >= 4 and n_replicates >= 2 and n_inner >= 2:
        candidates: list[tuple[int, int, int, int]] = []
        for replicate_workers in range(2, min(total_workers, n_replicates) + 1):
            for inner_workers in range(2, min(total_workers, n_inner) + 1):
                product = replicate_workers * inner_workers
                if product <= total_workers:
                    candidates.append(
                        (
                            product,
                            min(replicate_workers, inner_workers),
                            replicate_workers,
                            inner_workers,
                        )
                    )
        if candidates:
            _, _, replicate_workers, inner_workers = max(candidates)
            return replicate_workers, inner_workers

    return min(total_workers, max(1, n_replicates)), 1


def subsample_panel(panel: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Column-subsample a panel and relabel duplicate columns positionally."""
    sub = panel.iloc[:, idx]
    sub.columns = [f"T{i:04d}" for i in range(sub.shape[1])]
    return sub


def evaluate_inner_chunk(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
    payload: tuple[int, np.ndarray, np.ndarray, dict[str, np.ndarray]],
) -> tuple[int, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Evaluate one contiguous chunk of pre-generated matched draws."""
    start, idx_a_chunk, idx_b_chunk, idx_s_chunk = payload
    chunk_len = len(idx_a_chunk)
    g_rr = {metric.name: np.empty(chunk_len) for metric in metrics}
    g_sr = {
        metric.name: {generator: np.empty(chunk_len) for generator in synthetics}
        for metric in metrics
    }

    for local_id in range(chunk_len):
        panel_a = subsample_panel(real, idx_a_chunk[local_id])
        panel_b = subsample_panel(real, idx_b_chunk[local_id])
        features_a = {metric.name: metric.extract_features(panel_a) for metric in metrics}
        features_b = {metric.name: metric.extract_features(panel_b) for metric in metrics}

        for metric in metrics:
            g_rr[metric.name][local_id] = metric.compute_distance(
                features_a[metric.name],
                features_b[metric.name],
            )

        for generator, synthetic in synthetics.items():
            panel_s = subsample_panel(synthetic, idx_s_chunk[generator][local_id])
            for metric in metrics:
                features_s = metric.extract_features(panel_s)
                g_sr[metric.name][generator][local_id] = metric.compute_distance(
                    features_a[metric.name],
                    features_s,
                )

    return start, g_rr, g_sr


def _init_inner_process_worker(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
) -> None:
    global _INNER_PROCESS_CONTEXT
    _INNER_PROCESS_CONTEXT = {
        "real": real,
        "synthetics": synthetics,
        "metrics": metrics,
    }


def _evaluate_inner_process_chunk(payload):
    context = _INNER_PROCESS_CONTEXT
    return evaluate_inner_chunk(
        context["real"],
        context["synthetics"],
        context["metrics"],
        payload,
    )


def execute_inner_draws(
    *,
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    idx_s: dict[str, np.ndarray],
    n_jobs: int,
    chunk_size: int | None,
    backend: Literal["processes", "threads"],
    show_progress: bool,
    progress_callback: Callable[[int], None] | None,
    progress_position: int,
    leave_progress: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Execute all matched draws and return gap arrays in draw order."""
    n_draws = len(idx_a)
    workers = min(max(1, n_jobs), n_draws)
    bounds = chunk_bounds(n_draws, workers, chunk_size)
    payloads = [
        (
            start,
            idx_a[start:stop],
            idx_b[start:stop],
            {generator: values[start:stop] for generator, values in idx_s.items()},
        )
        for start, stop in bounds
    ]

    g_rr = {metric.name: np.empty(n_draws) for metric in metrics}
    g_sr = {
        metric.name: {generator: np.empty(n_draws) for generator in synthetics}
        for metric in metrics
    }

    def store(result) -> int:
        start, chunk_rr, chunk_sr = result
        chunk_len = len(next(iter(chunk_rr.values())))
        stop = start + chunk_len
        for metric in metrics:
            g_rr[metric.name][start:stop] = chunk_rr[metric.name]
            for generator in synthetics:
                g_sr[metric.name][generator][start:stop] = chunk_sr[metric.name][generator]
        if progress_callback is not None:
            progress_callback(chunk_len)
        return chunk_len

    description = "Matched ticker resamples"
    if workers > 1:
        description += f" ({workers} {backend})"

    with tqdm(
        total=n_draws,
        desc=description,
        disable=not show_progress,
        position=progress_position,
        leave=leave_progress,
        unit="draw",
    ) as progress:
        if workers == 1:
            for payload in payloads:
                progress.update(store(evaluate_inner_chunk(real, synthetics, metrics, payload)))
            return g_rr, g_sr

        executor_class = ProcessPoolExecutor if backend == "processes" else ThreadPoolExecutor
        executor_kwargs: dict = {"max_workers": workers}
        if backend == "processes":
            executor_kwargs.update(
                initializer=_init_inner_process_worker,
                initargs=(real, synthetics, metrics),
            )
            worker_fn = _evaluate_inner_process_chunk
        else:

            def worker_fn(payload):
                return evaluate_inner_chunk(
                    real,
                    synthetics,
                    metrics,
                    payload,
                )

        with executor_class(**executor_kwargs) as executor:
            futures = [executor.submit(worker_fn, payload) for payload in payloads]
            for future in as_completed(futures):
                progress.update(store(future.result()))

    return g_rr, g_sr
