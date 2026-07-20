"""High-level orchestration for one complete benchmark run."""

from __future__ import annotations

import time

import pandas as pd
from tqdm import tqdm

from fineval.bootstrap import MatchedTickerBootstrap
from fineval.bootstrap.models import ReplicateAnalysis

from .artifacts import run_tag, save_outputs
from .config import CURATED_DIR, RESULTS_DIR, build_metrics
from .datasets import load_default_datasets, preprocess_pairs
from .reporting import (
    BenchmarkProgress,
    attach_outer_diagnostics,
    format_aggregate_table,
    format_mc_table,
    format_metric_table,
    interval_label,
    log,
    log_outer_diagnostics,
)


def _merge_outer_results(
    point_results: pd.DataFrame,
    point_aggregate: pd.DataFrame,
    outer: ReplicateAnalysis,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = point_results.merge(
        outer.metric_summary,
        on=["metric", "generator"],
        how="left",
        validate="one_to_one",
    )
    aggregate = point_aggregate.merge(
        outer.aggregate_summary,
        on="generator",
        how="left",
        validate="one_to_one",
    )
    return attach_outer_diagnostics(results, aggregate)


def _merge_mc_results(
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    mc: ReplicateAnalysis,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        results.merge(
            mc.metric_summary,
            on=["metric", "generator"],
            how="left",
            validate="one_to_one",
        ),
        aggregate.merge(
            mc.aggregate_summary,
            on="generator",
            how="left",
            validate="one_to_one",
        ),
    )


def _print_tables(
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    mc: ReplicateAnalysis | None,
    outer_label: str | None,
) -> None:
    if outer_label is None:
        log("Per-metric table: matched ticker point estimates; no corpus interval requested.")
    else:
        log(f"Per-metric table: point estimate [lower, upper], using {outer_label}.")
    tqdm.write("\n" + format_metric_table(results) + "\n")

    log(
        "Aggregate table: G = exp(mean_k log(mean(g_sr)/mean(g_rr))). "
        "G=1 is real-real noise-floor parity."
    )
    tqdm.write("\n" + format_aggregate_table(aggregate, outer_label) + "\n")

    if mc is not None:
        log(
            "Per-metric Monte Carlo stability: independent inner seeds on fixed "
            "corpora; this is not a confidence interval."
        )
        tqdm.write("\n" + format_mc_table(mc.metric_summary, "score") + "\n")
        log("Aggregate Monte Carlo stability for G on the fixed observed corpora.")
        tqdm.write("\n" + format_mc_table(mc.aggregate_summary, "G") + "\n")


def run_benchmark(args, stamp: str, record: dict) -> None:
    """Execute one complete benchmark run and update its manifest record."""
    planned_draws = args.n_resamples * (1 + args.n_outer_resamples + max(0, args.n_mc_repeats - 1))
    log(
        f"Starting benchmark: B={args.n_resamples}, m={args.tickers_per_draw}, "
        f"seed={args.seed}, outer={args.n_outer_resamples}, "
        f"MC repeats={args.n_mc_repeats}, requested workers={args.n_jobs}. "
        f"Planned matched draws: {planned_draws:,}."
    )

    with BenchmarkProgress(args) as progress:
        started = progress.stage("Load curated datasets", str(CURATED_DIR))
        real, ail, gbm, msv = load_default_datasets(progress.update)
        log(f"Dataset loading completed in {time.monotonic() - started:.1f}s.")

        started = progress.stage(
            "Preprocess real-synthetic pairs",
            "log returns, overnight mask and conditional FFF",
        )
        deseas_real, synthetics = preprocess_pairs(
            real,
            (ail, gbm, msv),
            progress.update,
        )
        log(f"Preprocessing completed in {time.monotonic() - started:.1f}s.")

        started = progress.stage("Configure metrics and parallel engine")
        metrics = build_metrics()
        engine = MatchedTickerBootstrap(
            metrics=metrics,
            n_resamples=args.n_resamples,
            tickers_per_draw=args.tickers_per_draw,
            seed=args.seed,
            n_jobs=args.n_jobs,
            inner_chunk_size=args.inner_chunk_size or None,
        )
        record["n_jobs_resolved"] = engine.n_jobs
        progress.update()
        log(
            f"Configured metrics: {', '.join(metric.name for metric in metrics)}. "
            f"Worker budget={engine.n_jobs}; inner chunks="
            f"{args.inner_chunk_size or 'automatic'}; replicate chunks="
            f"{args.replicate_chunk_size or 'automatic'}."
        )
        log(f"Engine setup completed in {time.monotonic() - started:.1f}s.")

        started = progress.stage(
            "Matched ticker point estimate",
            f"{args.n_resamples} draws using up to {engine.n_jobs} workers",
        )
        point_results = engine.run(
            deseas_real,
            synthetics,
            progress_callback=progress.update,
            progress_position=1,
            leave_progress=False,
        )
        point_aggregate = engine.compute_aggregate()
        results = point_results.copy()
        aggregate = point_aggregate.copy()
        log(
            f"Point estimation completed in {time.monotonic() - started:.1f}s; "
            f"{len(results)} metric-generator cells computed."
        )

        outer: ReplicateAnalysis | None = None
        outer_label = interval_label(args.n_outer_resamples)
        if args.n_outer_resamples:
            started = progress.stage(
                "Outer ticker-corpus bootstrap",
                f"{args.n_outer_resamples} complete corpus replicates",
            )
            outer = engine.run_outer_bootstrap(
                deseas_real,
                synthetics,
                n_outer_resamples=args.n_outer_resamples,
                replicate_chunk_size=args.replicate_chunk_size or None,
                progress_callback=progress.update,
                progress_position=1,
                leave_progress=False,
            )
            results, aggregate = _merge_outer_results(
                point_results,
                point_aggregate,
                outer,
            )
            record["outer_interval_label"] = outer_label
            record["n_point_outside_outer_interval"] = int(
                (~results["point_in_outer_interval"]).sum()
            )
            record["n_aggregate_outside_outer_interval"] = int(
                (~aggregate["point_in_outer_interval"]).sum()
            )
            log(
                f"Outer bootstrap completed in {time.monotonic() - started:.1f}s; "
                "valid metric replicate counts range from "
                f"{int(outer.metric_summary['n_outer_valid'].min())} to "
                f"{int(outer.metric_summary['n_outer_valid'].max())}."
            )
            log_outer_diagnostics(results, aggregate, args.n_outer_resamples)

        mc: ReplicateAnalysis | None = None
        if args.n_mc_repeats:
            started = progress.stage(
                "Monte Carlo numerical stability",
                f"{args.n_mc_repeats} seeds; repeat zero reuses the point estimate",
            )
            mc = engine.run_monte_carlo_stability(
                deseas_real,
                synthetics,
                n_repeats=args.n_mc_repeats,
                base_metric_result=point_results,
                base_aggregate_result=point_aggregate,
                replicate_chunk_size=args.replicate_chunk_size or None,
                progress_callback=progress.update,
                progress_position=1,
                leave_progress=False,
            )
            results, aggregate = _merge_mc_results(results, aggregate, mc)
            log(
                f"Monte Carlo stability completed in {time.monotonic() - started:.1f}s; "
                f"metric seed SD range={mc.metric_summary['mc_sd'].min():.4g}-"
                f"{mc.metric_summary['mc_sd'].max():.4g}."
            )

        tag = run_tag(args, stamp, engine.n_jobs)
        started = progress.stage("Save results", str(RESULTS_DIR))
        results_path, aggregate_path = save_outputs(
            results=results,
            aggregate=aggregate,
            outer=outer,
            mc=mc,
            tag=tag,
            update_canonical=args.update_canonical,
            record=record,
        )
        progress.update(2)
        log(
            f"Saved results in {time.monotonic() - started:.1f}s: "
            f"{results_path.name}, {aggregate_path.name}."
        )

        progress.stage("Print benchmark tables and run summary")
        _print_tables(results, aggregate, mc, outer_label)
        log(f"Tidy benchmark results: {results_path}")
        log(f"Aggregate results: {aggregate_path}")
        progress.update()
        progress.complete()
        record["status"] = "completed"
