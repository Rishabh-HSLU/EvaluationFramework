"""Corpus-bootstrap intervals and independent-seed stability analyses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .engine import MatchedTickerBootstrap
from .execution import chunk_bounds, nested_worker_plan
from .models import ReplicateAnalysis
from .replicate_execution import (
    evaluate_mc_chunk,
    evaluate_outer_chunk,
    init_mc_worker,
    init_outer_worker,
    run_replicate_chunks,
)


def _replicate_frames(
    metric_records: list[dict],
    aggregate_records: list[dict],
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = (
        pd.DataFrame(metric_records)
        .sort_values([id_column, "metric", "generator"], kind="stable")
        .reset_index(drop=True)
    )
    aggregate = (
        pd.DataFrame(aggregate_records)
        .sort_values([id_column, "generator"], kind="stable")
        .reset_index(drop=True)
    )
    return metric, aggregate


def _summarize_outer(
    metric_replicates: pd.DataFrame,
    aggregate_replicates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_summary = metric_replicates.groupby(
        ["metric", "generator"], as_index=False, sort=False
    ).agg(
        outer_mean=("score", "mean"),
        outer_sd=("score", "std"),
        ci_low=("score", lambda values: values.quantile(0.025)),
        ci_high=("score", lambda values: values.quantile(0.975)),
        n_outer_valid=("score", "count"),
    )
    aggregate_summary = aggregate_replicates.groupby("generator", as_index=False, sort=False).agg(
        outer_mean=("G", "mean"),
        outer_sd=("G", "std"),
        ci_low=("G", lambda values: values.quantile(0.025)),
        ci_high=("G", lambda values: values.quantile(0.975)),
        n_outer_valid=("G", "count"),
    )
    return metric_summary, aggregate_summary


def _summarize_mc(
    metric_replicates: pd.DataFrame,
    aggregate_replicates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_summary = metric_replicates.groupby(
        ["metric", "generator"], as_index=False, sort=False
    ).agg(
        mc_mean=("score", "mean"),
        mc_sd=("score", "std"),
        mc_min=("score", "min"),
        mc_max=("score", "max"),
        mc_q025=("score", lambda values: values.quantile(0.025)),
        mc_q975=("score", lambda values: values.quantile(0.975)),
        n_mc_valid=("score", "count"),
    )
    aggregate_summary = aggregate_replicates.groupby("generator", as_index=False, sort=False).agg(
        mc_mean=("G", "mean"),
        mc_sd=("G", "std"),
        mc_min=("G", "min"),
        mc_max=("G", "max"),
        mc_q025=("G", lambda values: values.quantile(0.025)),
        mc_q975=("G", lambda values: values.quantile(0.975)),
        n_mc_valid=("G", "count"),
    )
    return metric_summary, aggregate_summary


def run_outer_bootstrap(
    engine: MatchedTickerBootstrap,
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    *,
    n_outer_resamples: int,
    show_progress: bool,
    replicate_chunk_size: int | None,
    progress_callback: Callable[[int], None] | None,
    progress_position: int,
    leave_progress: bool,
) -> ReplicateAnalysis:
    """Estimate complete-corpus percentile intervals."""
    if n_outer_resamples < 2:
        raise ValueError("n_outer_resamples must be at least 2")
    if replicate_chunk_size is not None and replicate_chunk_size < 1:
        raise ValueError("replicate_chunk_size must be at least 1")

    engine._check_no_crc32_collision(list(synthetics))
    outer_real_idx = engine._stream(4).integers(
        0,
        real.shape[1],
        size=(n_outer_resamples, real.shape[1]),
    )
    outer_synth_idx = {
        generator: engine._stream(5, generator).integers(
            0,
            synthetic.shape[1],
            size=(n_outer_resamples, synthetic.shape[1]),
        )
        for generator, synthetic in synthetics.items()
    }

    outer_workers, inner_workers = nested_worker_plan(
        engine.n_jobs,
        n_outer_resamples,
        engine.n_resamples,
    )
    bounds = chunk_bounds(n_outer_resamples, outer_workers, replicate_chunk_size)
    payloads = [
        [
            (
                outer_id,
                outer_real_idx[outer_id],
                {generator: indices[outer_id] for generator, indices in outer_synth_idx.items()},
            )
            for outer_id in range(start, stop)
        ]
        for start, stop in bounds
    ]

    if show_progress:
        tqdm.write(
            "Outer-bootstrap worker plan: "
            f"{outer_workers} process(es) × {inner_workers} inner thread(s), "
            f"{len(payloads)} chunk(s), {engine.n_resamples} inner draws/replicate."
        )

    metric_records, aggregate_records = run_replicate_chunks(
        payloads=payloads,
        n_replicates=n_outer_resamples,
        workers=outer_workers,
        initializer=init_outer_worker,
        initargs=(
            real,
            synthetics,
            engine.metrics,
            engine.n_resamples,
            engine.tickers_per_draw,
            engine.seed,
            inner_workers,
            engine.inner_chunk_size,
        ),
        worker=evaluate_outer_chunk,
        description="Outer corpus bootstrap",
        id_column="outer_id",
        inner_draws=engine.n_resamples,
        show_progress=show_progress,
        progress_callback=progress_callback,
        progress_position=progress_position,
        leave_progress=leave_progress,
    )
    metric_replicates, aggregate_replicates = _replicate_frames(
        metric_records,
        aggregate_records,
        "outer_id",
    )
    metric_summary, aggregate_summary = _summarize_outer(
        metric_replicates,
        aggregate_replicates,
    )
    return ReplicateAnalysis(
        metric_summary,
        aggregate_summary,
        metric_replicates,
        aggregate_replicates,
    )


def run_monte_carlo_stability(
    engine: MatchedTickerBootstrap,
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    *,
    n_repeats: int,
    base_metric_result: pd.DataFrame | None,
    base_aggregate_result: pd.DataFrame | None,
    show_progress: bool,
    replicate_chunk_size: int | None,
    progress_callback: Callable[[int], None] | None,
    progress_position: int,
    leave_progress: bool,
) -> ReplicateAnalysis:
    """Assess numerical stability across independent inner seeds."""
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2")
    if replicate_chunk_size is not None and replicate_chunk_size < 1:
        raise ValueError("replicate_chunk_size must be at least 1")

    seeds = np.empty(n_repeats, dtype=np.uint32)
    seeds[0] = np.uint32(engine.seed)
    seeds[1:] = engine._stream(7).integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_repeats - 1,
        dtype=np.uint32,
    )

    metric_records: list[dict] = []
    aggregate_records: list[dict] = []
    start_repeat = 0
    if base_metric_result is not None and base_aggregate_result is not None:
        metric_records.extend(
            {"repeat_id": 0, "seed": int(seeds[0]), **record}
            for record in base_metric_result.to_dict("records")
        )
        aggregate_records.extend(
            {"repeat_id": 0, "seed": int(seeds[0]), **record}
            for record in base_aggregate_result.to_dict("records")
        )
        start_repeat = 1

    pending = n_repeats - start_repeat
    repeat_workers, inner_workers = nested_worker_plan(
        engine.n_jobs,
        max(1, pending),
        engine.n_resamples,
    )
    repeat_workers = min(repeat_workers, max(1, pending))
    repeat_items = [
        (repeat_id, int(seeds[repeat_id])) for repeat_id in range(start_repeat, n_repeats)
    ]
    bounds = chunk_bounds(pending, repeat_workers, replicate_chunk_size)
    payloads = [repeat_items[start:stop] for start, stop in bounds]

    if show_progress:
        tqdm.write(
            "Monte Carlo worker plan: "
            f"{repeat_workers} process(es) × {inner_workers} inner thread(s), "
            f"{len(payloads)} chunk(s), {engine.n_resamples} inner draws/repeat."
        )

    if pending:
        metric_chunk, aggregate_chunk = run_replicate_chunks(
            payloads=payloads,
            n_replicates=pending,
            workers=repeat_workers,
            initializer=init_mc_worker,
            initargs=(
                real,
                synthetics,
                engine.metrics,
                engine.n_resamples,
                engine.tickers_per_draw,
                inner_workers,
                engine.inner_chunk_size,
            ),
            worker=evaluate_mc_chunk,
            description="Monte Carlo seed repeats",
            id_column="repeat_id",
            inner_draws=engine.n_resamples,
            show_progress=show_progress,
            progress_callback=progress_callback,
            progress_position=progress_position,
            leave_progress=leave_progress,
        )
        metric_records.extend(metric_chunk)
        aggregate_records.extend(aggregate_chunk)

    metric_replicates, aggregate_replicates = _replicate_frames(
        metric_records,
        aggregate_records,
        "repeat_id",
    )
    metric_summary, aggregate_summary = _summarize_mc(
        metric_replicates,
        aggregate_replicates,
    )
    return ReplicateAnalysis(
        metric_summary,
        aggregate_summary,
        metric_replicates,
        aggregate_replicates,
    )
