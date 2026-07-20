"""Matched-N ticker resampling engine.

The primary point estimate uses repeated matched ticker draws:

    g_rr[b] = distance(real draw A_b, real draw B_b)
    g_sr[b] = distance(real draw A_b, synthetic draw S_b)

The shared A_b creates a matched design between the real-real and
synthetic-real gaps. The score for metric k is

    s_k = mean(g_rr) / (mean(g_rr) + mean(g_sr)).

Two optional uncertainty analyses are implemented separately:

1. ``run_outer_bootstrap`` resamples the complete observed ticker
   corpora and reruns the inner matched estimator. Percentiles of the
   resulting complete scores form corpus-resampling confidence
   intervals, conditional on the fitted preprocessing and generated
   synthetic panels supplied to this engine.
2. ``run_monte_carlo_stability`` reruns the inner estimator on the
   fixed corpora under independent deterministic seeds. Its spread is
   a numerical-stability diagnostic, not a confidence interval.

Parallel execution is deterministic. Every inner index, outer corpus
index and Monte Carlo seed is generated before work is submitted, and
results are written back by replicate id rather than completion order.
The point estimator parallelizes inner chunks with processes. Nested
analyses split one total worker budget between outer/repeat processes
and inner threads, preventing nested CPU oversubscription.
"""

from __future__ import annotations

import math
import os
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import SEED
from ..metrics.base import BaseMetric

# Process-worker contexts. Initializers transfer the large, read-only panels
# once per worker instead of serializing them again for every chunk.
_INNER_PROCESS_CONTEXT: dict = {}
_OUTER_PROCESS_CONTEXT: dict = {}
_MC_PROCESS_CONTEXT: dict = {}


def _subsample_panel(panel: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Column-subsample a panel and relabel columns positionally."""
    sub = panel.iloc[:, idx]
    sub.columns = [f"T{i:04d}" for i in range(sub.shape[1])]
    return sub


def _evaluate_inner_chunk(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
    payload: tuple[int, np.ndarray, np.ndarray, dict[str, np.ndarray]],
) -> tuple[int, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Evaluate one contiguous chunk of pre-generated inner draws."""
    start, idx_a_chunk, idx_b_chunk, idx_s_chunk = payload
    chunk_len = len(idx_a_chunk)
    g_rr = {metric.name: np.empty(chunk_len) for metric in metrics}
    g_sr = {metric.name: {gen: np.empty(chunk_len) for gen in synthetics} for metric in metrics}

    for local_id in range(chunk_len):
        panel_a = _subsample_panel(real, idx_a_chunk[local_id])
        panel_b = _subsample_panel(real, idx_b_chunk[local_id])

        features_a = {metric.name: metric.extract_features(panel_a) for metric in metrics}
        features_b = {metric.name: metric.extract_features(panel_b) for metric in metrics}

        for metric in metrics:
            g_rr[metric.name][local_id] = metric.compute_distance(
                features_a[metric.name],
                features_b[metric.name],
            )

        for gen, synth in synthetics.items():
            panel_s = _subsample_panel(synth, idx_s_chunk[gen][local_id])
            for metric in metrics:
                features_s = metric.extract_features(panel_s)
                g_sr[metric.name][gen][local_id] = metric.compute_distance(
                    features_a[metric.name],
                    features_s,
                )

    return start, g_rr, g_sr


def _init_inner_process_worker(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
) -> None:
    """Initialize one process used for point-estimate inner chunks."""
    global _INNER_PROCESS_CONTEXT
    _INNER_PROCESS_CONTEXT = {
        "real": real,
        "synthetics": synthetics,
        "metrics": metrics,
    }


def _evaluate_inner_process_chunk(payload):
    """Process-pool adapter for ``_evaluate_inner_chunk``."""
    return _evaluate_inner_chunk(
        _INNER_PROCESS_CONTEXT["real"],
        _INNER_PROCESS_CONTEXT["synthetics"],
        _INNER_PROCESS_CONTEXT["metrics"],
        payload,
    )


def _init_outer_process_worker(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
    n_resamples: int,
    tickers_per_draw: int,
    seed: int,
    inner_workers: int,
    inner_chunk_size: int | None,
) -> None:
    """Initialize one process used for outer-bootstrap chunks."""
    global _OUTER_PROCESS_CONTEXT
    _OUTER_PROCESS_CONTEXT = {
        "real": real,
        "synthetics": synthetics,
        "metrics": metrics,
        "n_resamples": n_resamples,
        "tickers_per_draw": tickers_per_draw,
        "seed": seed,
        "inner_workers": inner_workers,
        "inner_chunk_size": inner_chunk_size,
    }


def _evaluate_outer_process_chunk(
    payload: list[tuple[int, np.ndarray, dict[str, np.ndarray]]],
) -> tuple[list[dict], list[dict]]:
    """Evaluate several outer replicates inside one process."""
    context = _OUTER_PROCESS_CONTEXT
    metric_records: list[dict] = []
    aggregate_records: list[dict] = []

    for outer_id, real_idx, synth_idx in payload:
        outer_real = _subsample_panel(context["real"], real_idx)
        outer_synthetics = {
            gen: _subsample_panel(synth, synth_idx[gen])
            for gen, synth in context["synthetics"].items()
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
            parallel_backend="threads",
        )
        aggregate_result = child.compute_aggregate()

        for record in metric_result.to_dict("records"):
            metric_records.append(
                {
                    "outer_id": outer_id,
                    "inner_seed": context["seed"],
                    **record,
                }
            )
        for record in aggregate_result.to_dict("records"):
            aggregate_records.append(
                {
                    "outer_id": outer_id,
                    "inner_seed": context["seed"],
                    **record,
                }
            )

    return metric_records, aggregate_records


def _init_mc_process_worker(
    real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    metrics: list[BaseMetric],
    n_resamples: int,
    tickers_per_draw: int,
    inner_workers: int,
    inner_chunk_size: int | None,
) -> None:
    """Initialize one process used for Monte Carlo repeat chunks."""
    global _MC_PROCESS_CONTEXT
    _MC_PROCESS_CONTEXT = {
        "real": real,
        "synthetics": synthetics,
        "metrics": metrics,
        "n_resamples": n_resamples,
        "tickers_per_draw": tickers_per_draw,
        "inner_workers": inner_workers,
        "inner_chunk_size": inner_chunk_size,
    }


def _evaluate_mc_process_chunk(
    payload: list[tuple[int, int]],
) -> tuple[list[dict], list[dict]]:
    """Evaluate several independent-seed repeats inside one process."""
    context = _MC_PROCESS_CONTEXT
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
            parallel_backend="threads",
        )
        aggregate_result = child.compute_aggregate()

        for record in metric_result.to_dict("records"):
            metric_records.append({"repeat_id": repeat_id, "seed": repeat_seed, **record})
        for record in aggregate_result.to_dict("records"):
            aggregate_records.append({"repeat_id": repeat_id, "seed": repeat_seed, **record})

    return metric_records, aggregate_records


@dataclass(frozen=True)
class ReplicateAnalysis:
    """Summary and replicate-level outputs from a repeated analysis."""

    metric_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    metric_replicates: pd.DataFrame
    aggregate_replicates: pd.DataFrame


class MatchedTickerBootstrap:
    """Run the matched-N ticker estimator for a set of metrics.

    Args:
        metrics: Metric instances implementing extract_features,
            compute_distance, and normalize.
        n_resamples: Number of inner matched ticker draws B.
        tickers_per_draw: Number of tickers sampled with replacement in
            each inner draw. Capped at the real panel width.
        seed: Root seed. Every stochastic role uses an independent,
            reproducible stream derived from this seed.
        n_jobs: Total worker budget. ``None`` or ``0`` selects a
            conservative automatic value: all but one logical CPU,
            capped at eight workers. ``-1`` uses every logical CPU.
        inner_chunk_size: Number of inner draws submitted per task.
            ``None`` selects approximately four tasks per worker.
    """

    def __init__(
        self,
        metrics: list[BaseMetric],
        n_resamples: int = 100,
        tickers_per_draw: int = 200,
        seed: int = SEED,
        n_jobs: int | None = None,
        inner_chunk_size: int | None = None,
    ) -> None:
        self.metrics = metrics
        self.n_resamples = n_resamples
        self.tickers_per_draw = tickers_per_draw
        self.seed = seed
        self.n_jobs = self.resolve_n_jobs(n_jobs)
        if inner_chunk_size is not None and inner_chunk_size < 1:
            raise ValueError("inner_chunk_size must be at least 1")
        self.inner_chunk_size = inner_chunk_size
        self.g_rr: dict[str, np.ndarray] = {}
        self.g_sr: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def resolve_n_jobs(n_jobs: int | None) -> int:
        """Resolve an explicit or automatic total worker budget."""
        cpu_count = os.cpu_count() or 1
        if n_jobs in (None, 0):
            return max(1, min(8, cpu_count - 1 if cpu_count > 1 else 1))
        if n_jobs == -1:
            return cpu_count
        if n_jobs < 1:
            raise ValueError("n_jobs must be -1, 0, None, or a positive integer")
        return n_jobs

    @staticmethod
    def _subsample(panel: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        """Column-subsample a panel and relabel columns positionally."""
        return _subsample_panel(panel, idx)

    def _stream(self, role: int, *names: str) -> np.random.Generator:
        """Return an independent RNG stream for one stochastic role."""
        entropy = [self.seed, role, *(zlib.crc32(name.encode()) for name in names)]
        return np.random.default_rng(entropy)

    @staticmethod
    def _check_no_crc32_collision(names: list[str]) -> None:
        """Guard against two names silently sharing a keyed RNG stream."""
        digests = [zlib.crc32(name.encode()) for name in names]
        if len(set(digests)) != len(names):
            raise ValueError(f"crc32 collision among generator names {names!r}")

    @staticmethod
    def _chunk_bounds(
        n_items: int,
        n_workers: int,
        chunk_size: int | None,
    ) -> list[tuple[int, int]]:
        """Return deterministic contiguous chunks covering ``n_items``."""
        if n_items < 1:
            return []
        if chunk_size is None:
            chunk_size = max(1, math.ceil(n_items / max(1, n_workers * 4)))
        return [
            (start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)
        ]

    @staticmethod
    def _nested_worker_plan(
        total_workers: int,
        n_replicates: int,
        n_inner: int,
    ) -> tuple[int, int]:
        """Split one worker budget across replicate processes and inner threads.

        When at least four workers and two units exist at both levels,
        the selected factor pair parallelizes both loops while never
        exceeding ``total_workers``. Ties favor more replicate workers,
        because each replicate is a larger and more memory-isolated task.
        """
        total_workers = max(1, total_workers)
        if n_replicates <= 1:
            return 1, min(total_workers, max(1, n_inner))
        if total_workers >= 4 and n_replicates >= 2 and n_inner >= 2:
            candidates = []
            for outer_workers in range(2, min(total_workers, n_replicates) + 1):
                for inner_workers in range(2, min(total_workers, n_inner) + 1):
                    product = outer_workers * inner_workers
                    if product <= total_workers:
                        candidates.append(
                            (
                                product,
                                min(outer_workers, inner_workers),
                                outer_workers,
                                inner_workers,
                            )
                        )
            if candidates:
                _, _, outer_workers, inner_workers = max(candidates)
                return outer_workers, inner_workers

        outer_workers = min(total_workers, max(1, n_replicates))
        return outer_workers, 1

    def _prepare_inner_indices(
        self,
        real: pd.DataFrame,
        synthetics: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Pre-generate every inner draw index for order-independent execution.

        The real index tensor has shape (B, 2, m), preserving the exact
        sequential RNG order of drawing A_b and then B_b in each legacy
        loop iteration. Each generator receives its own independent
        (B, m) index matrix.
        """
        n_real = real.shape[1]
        if n_real == 0:
            raise ValueError("real panel has no ticker columns")

        m = min(self.tickers_per_draw, n_real)
        real_idx = self._stream(0).integers(
            0,
            n_real,
            size=(self.n_resamples, 2, m),
        )
        idx_a = real_idx[:, 0, :]
        idx_b = real_idx[:, 1, :]

        idx_s = {}
        for gen, synth in synthetics.items():
            if synth.shape[1] == 0:
                raise ValueError(f"synthetic panel {gen!r} has no ticker columns")
            idx_s[gen] = self._stream(1, gen).integers(
                0,
                synth.shape[1],
                size=(self.n_resamples, m),
            )
        return idx_a, idx_b, idx_s

    def run(
        self,
        real: pd.DataFrame,
        synthetics: dict[str, pd.DataFrame],
        *,
        show_progress: bool = True,
        parallel_backend: Literal["processes", "threads"] = "processes",
        n_jobs: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
        progress_position: int = 0,
        leave_progress: bool = True,
    ) -> pd.DataFrame:
        """Run the inner matched ticker estimator on fixed corpora.

        The default process backend is intended for the top-level point
        estimate. Nested outer or Monte Carlo workers call this method
        with threads so the total worker budget remains bounded.

        Returns one point estimate per (metric, generator). No interval
        is attached here: corpus uncertainty and Monte Carlo stability
        are computed by separate, explicitly named methods.
        """
        names = list(synthetics)
        self._check_no_crc32_collision(names)
        idx_a_all, idx_b_all, idx_s_all = self._prepare_inner_indices(real, synthetics)

        b_total = self.n_resamples
        self.g_rr = {metric.name: np.empty(b_total) for metric in self.metrics}
        self.g_sr = {
            metric.name: {gen: np.empty(b_total) for gen in synthetics} for metric in self.metrics
        }

        workers = min(
            self.resolve_n_jobs(n_jobs) if n_jobs is not None else self.n_jobs,
            b_total,
        )
        chunks = self._chunk_bounds(b_total, workers, self.inner_chunk_size)
        payloads = [
            (
                start,
                idx_a_all[start:stop],
                idx_b_all[start:stop],
                {gen: indices[start:stop] for gen, indices in idx_s_all.items()},
            )
            for start, stop in chunks
        ]

        def store_chunk(result) -> int:
            start, chunk_rr, chunk_sr = result
            chunk_len = len(next(iter(chunk_rr.values())))
            stop = start + chunk_len
            for metric in self.metrics:
                self.g_rr[metric.name][start:stop] = chunk_rr[metric.name]
                for gen in synthetics:
                    self.g_sr[metric.name][gen][start:stop] = chunk_sr[metric.name][gen]
            if progress_callback is not None:
                progress_callback(chunk_len)
            return chunk_len

        if workers == 1:
            with tqdm(
                total=b_total,
                desc="Matched ticker resamples",
                disable=not show_progress,
                position=progress_position,
                leave=leave_progress,
                unit="draw",
            ) as progress:
                for payload in payloads:
                    progress.update(
                        store_chunk(_evaluate_inner_chunk(real, synthetics, self.metrics, payload))
                    )
        else:
            executor_class = (
                ProcessPoolExecutor if parallel_backend == "processes" else ThreadPoolExecutor
            )
            executor_kwargs = {"max_workers": workers}
            submit_fn = None
            if parallel_backend == "processes":
                executor_kwargs.update(
                    {
                        "initializer": _init_inner_process_worker,
                        "initargs": (real, synthetics, self.metrics),
                    }
                )
                submit_fn = _evaluate_inner_process_chunk
            else:
                submit_fn = lambda payload: _evaluate_inner_chunk(  # noqa: E731
                    real,
                    synthetics,
                    self.metrics,
                    payload,
                )

            with executor_class(**executor_kwargs) as executor:
                futures = [executor.submit(submit_fn, payload) for payload in payloads]
                with tqdm(
                    total=b_total,
                    desc=f"Matched ticker resamples ({workers} {parallel_backend})",
                    disable=not show_progress,
                    position=progress_position,
                    leave=leave_progress,
                    unit="draw",
                ) as progress:
                    for future in as_completed(futures):
                        progress.update(store_chunk(future.result()))

        return self._summarize(synthetics.keys())

    def _summarize(self, generators) -> pd.DataFrame:
        """Build per-metric point estimates from stored gap arrays."""
        rows = []
        for metric in self.metrics:
            g_rr = self.g_rr[metric.name]
            for gen in generators:
                g_sr = self.g_sr[metric.name][gen]
                rows.append(
                    {
                        "metric": metric.name,
                        "generator": gen,
                        "score": metric.normalize(g_rr, g_sr),
                        "g_rr_mean": float(np.nanmean(g_rr)),
                        "g_sr_mean": float(np.nanmean(g_sr)),
                    }
                )
        return pd.DataFrame(rows)

    def compute_aggregate(self, generators: list[str] | None = None) -> pd.DataFrame:
        """Compute aggregate gap ratio G for each generator.

        For metric k,

            r_k = mean(g_sr,k) / mean(g_rr,k)

        and

            G = exp(mean_k(log(r_k))).

        G=1 is real-real noise-floor parity; larger values indicate a
        larger aggregate synthetic-real gap. All configured metrics are
        part of the statistic. If any ratio is non-finite or non-positive,
        G is reported as NaN rather than silently changing the metric set.
        """
        if not self.g_rr:
            raise RuntimeError("compute_aggregate() requires run() to have been called")
        if generators is None:
            generators = list(self.g_sr[self.metrics[0].name])

        rows = []
        for gen in generators:
            log_ratios = []
            for metric in self.metrics:
                rr = float(np.nanmean(self.g_rr[metric.name]))
                sr = float(np.nanmean(self.g_sr[metric.name][gen]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = sr / rr
                if np.isfinite(ratio) and ratio > 0:
                    log_ratios.append(float(np.log(ratio)))

            k_used = len(log_ratios)
            score = (
                float(np.exp(np.mean(log_ratios))) if k_used == len(self.metrics) else float("nan")
            )
            rows.append(
                {
                    "generator": gen,
                    "G": score,
                    "k_used": k_used,
                    "k_total": len(self.metrics),
                }
            )
        return pd.DataFrame(rows)

    def _prepare_outer_inputs(
        self,
        real: pd.DataFrame,
        synthetics: dict[str, pd.DataFrame],
        n_outer_resamples: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Pre-generate outer corpus indices for order-independent execution.

        Every outer replicate uses the same inner seed as the reported
        point estimate (common random numbers), so its spread isolates
        corpus resampling rather than mixing in avoidable inner-seed noise.
        """
        outer_real_idx = self._stream(4).integers(
            0,
            real.shape[1],
            size=(n_outer_resamples, real.shape[1]),
        )
        outer_synth_idx = {
            gen: self._stream(5, gen).integers(
                0,
                synth.shape[1],
                size=(n_outer_resamples, synth.shape[1]),
            )
            for gen, synth in synthetics.items()
        }
        return outer_real_idx, outer_synth_idx

    def run_outer_bootstrap(
        self,
        real: pd.DataFrame,
        synthetics: dict[str, pd.DataFrame],
        *,
        n_outer_resamples: int,
        show_progress: bool = True,
        replicate_chunk_size: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
        progress_position: int = 0,
        leave_progress: bool = True,
    ) -> ReplicateAnalysis:
        """Estimate corpus-resampling percentile intervals.

        Each outer replicate resamples the complete real corpus and
        each complete synthetic corpus with replacement. The full inner
        matched estimator is then rerun on those bootstrap corpora.

        The total worker budget is split between outer processes and
        inner threads. All replicate inputs are fixed before submission,
        so changing worker counts or chunk sizes does not change results.

        The resulting intervals are conditional on the already-fitted
        preprocessing and synthetic panels passed to this method. They
        do not include preprocessing refit, generator refit, or generator
        retraining uncertainty.
        """
        if n_outer_resamples < 2:
            raise ValueError("n_outer_resamples must be at least 2")
        if replicate_chunk_size is not None and replicate_chunk_size < 1:
            raise ValueError("replicate_chunk_size must be at least 1")

        names = list(synthetics)
        self._check_no_crc32_collision(names)
        outer_real_idx, outer_synth_idx = self._prepare_outer_inputs(
            real,
            synthetics,
            n_outer_resamples,
        )
        outer_workers, inner_workers = self._nested_worker_plan(
            self.n_jobs,
            n_outer_resamples,
            self.n_resamples,
        )
        bounds = self._chunk_bounds(
            n_outer_resamples,
            outer_workers,
            replicate_chunk_size,
        )
        payloads = [
            [
                (
                    outer_id,
                    outer_real_idx[outer_id],
                    {gen: indices[outer_id] for gen, indices in outer_synth_idx.items()},
                )
                for outer_id in range(start, stop)
            ]
            for start, stop in bounds
        ]

        metric_records: list[dict] = []
        aggregate_records: list[dict] = []

        if show_progress:
            tqdm.write(
                "Outer-bootstrap worker plan: "
                f"{outer_workers} process(es) × {inner_workers} inner thread(s), "
                f"{len(payloads)} submitted chunk(s), "
                f"{self.n_resamples} inner draws/replicate "
                f"(<= {self.n_jobs} total workers)."
            )

        if outer_workers == 1:
            _init_outer_process_worker(
                real,
                synthetics,
                self.metrics,
                self.n_resamples,
                self.tickers_per_draw,
                self.seed,
                inner_workers,
                self.inner_chunk_size,
            )
            with tqdm(
                total=n_outer_resamples,
                desc="Outer corpus bootstrap",
                disable=not show_progress,
                position=progress_position,
                leave=leave_progress,
                unit="replicate",
            ) as progress:
                for payload in payloads:
                    metric_chunk, aggregate_chunk = _evaluate_outer_process_chunk(payload)
                    metric_records.extend(metric_chunk)
                    aggregate_records.extend(aggregate_chunk)
                    completed = len({record["outer_id"] for record in metric_chunk})
                    progress.update(completed)
                    if progress_callback is not None:
                        progress_callback(completed * self.n_resamples)
        else:
            with ProcessPoolExecutor(
                max_workers=outer_workers,
                initializer=_init_outer_process_worker,
                initargs=(
                    real,
                    synthetics,
                    self.metrics,
                    self.n_resamples,
                    self.tickers_per_draw,
                    self.seed,
                    inner_workers,
                    self.inner_chunk_size,
                ),
            ) as executor:
                futures = [
                    executor.submit(_evaluate_outer_process_chunk, payload) for payload in payloads
                ]
                with tqdm(
                    total=n_outer_resamples,
                    desc="Outer corpus bootstrap",
                    disable=not show_progress,
                    position=progress_position,
                    leave=leave_progress,
                    unit="replicate",
                ) as progress:
                    for future in as_completed(futures):
                        metric_chunk, aggregate_chunk = future.result()
                        metric_records.extend(metric_chunk)
                        aggregate_records.extend(aggregate_chunk)
                        completed_ids = {record["outer_id"] for record in metric_chunk}
                        completed = len(completed_ids)
                        progress.update(completed)
                        if progress_callback is not None:
                            progress_callback(completed * self.n_resamples)

        metric_replicates = (
            pd.DataFrame(metric_records)
            .sort_values(["outer_id", "metric", "generator"], kind="stable")
            .reset_index(drop=True)
        )
        aggregate_replicates = (
            pd.DataFrame(aggregate_records)
            .sort_values(["outer_id", "generator"], kind="stable")
            .reset_index(drop=True)
        )

        metric_summary = metric_replicates.groupby(
            ["metric", "generator"], as_index=False, sort=False
        ).agg(
            outer_mean=("score", "mean"),
            outer_sd=("score", "std"),
            ci_low=("score", lambda values: values.quantile(0.025)),
            ci_high=("score", lambda values: values.quantile(0.975)),
            n_outer_valid=("score", "count"),
        )
        aggregate_summary = aggregate_replicates.groupby(
            "generator", as_index=False, sort=False
        ).agg(
            outer_mean=("G", "mean"),
            outer_sd=("G", "std"),
            ci_low=("G", lambda values: values.quantile(0.025)),
            ci_high=("G", lambda values: values.quantile(0.975)),
            n_outer_valid=("G", "count"),
        )

        return ReplicateAnalysis(
            metric_summary=metric_summary,
            aggregate_summary=aggregate_summary,
            metric_replicates=metric_replicates,
            aggregate_replicates=aggregate_replicates,
        )

    def _prepare_monte_carlo_seeds(self, n_repeats: int) -> np.ndarray:
        """Return deterministic independent seeds, including the base seed."""
        seeds = np.empty(n_repeats, dtype=np.uint32)
        seeds[0] = np.uint32(self.seed)
        if n_repeats > 1:
            seeds[1:] = self._stream(7).integers(
                0,
                np.iinfo(np.uint32).max,
                size=n_repeats - 1,
                dtype=np.uint32,
            )
        return seeds

    def run_monte_carlo_stability(
        self,
        real: pd.DataFrame,
        synthetics: dict[str, pd.DataFrame],
        *,
        n_repeats: int,
        base_metric_result: pd.DataFrame | None = None,
        base_aggregate_result: pd.DataFrame | None = None,
        show_progress: bool = True,
        replicate_chunk_size: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
        progress_position: int = 0,
        leave_progress: bool = True,
    ) -> ReplicateAnalysis:
        """Assess numerical stability across independent inner seeds.

        The observed real and synthetic corpora remain fixed. Reported
        spread therefore quantifies Monte Carlo sensitivity only and
        must not be described as corpus-level confidence coverage.

        Repeat zero may reuse the already-computed point estimate. The
        remaining repeats are split across processes, and each repeat's
        inner draws use the bounded nested thread allocation.
        """
        if n_repeats < 2:
            raise ValueError("n_repeats must be at least 2")
        if replicate_chunk_size is not None and replicate_chunk_size < 1:
            raise ValueError("replicate_chunk_size must be at least 1")

        seeds = self._prepare_monte_carlo_seeds(n_repeats)
        metric_records: list[dict] = []
        aggregate_records: list[dict] = []

        start_repeat = 0
        if base_metric_result is not None and base_aggregate_result is not None:
            for record in base_metric_result.to_dict("records"):
                metric_records.append({"repeat_id": 0, "seed": int(seeds[0]), **record})
            for record in base_aggregate_result.to_dict("records"):
                aggregate_records.append({"repeat_id": 0, "seed": int(seeds[0]), **record})
            start_repeat = 1

        pending_repeats = n_repeats - start_repeat
        repeat_workers, inner_workers = self._nested_worker_plan(
            self.n_jobs,
            max(1, pending_repeats),
            self.n_resamples,
        )
        repeat_workers = min(repeat_workers, max(1, pending_repeats))
        bounds = self._chunk_bounds(
            pending_repeats,
            repeat_workers,
            replicate_chunk_size,
        )
        repeat_items = [
            (repeat_id, int(seeds[repeat_id])) for repeat_id in range(start_repeat, n_repeats)
        ]
        payloads = [repeat_items[start:stop] for start, stop in bounds]

        if show_progress:
            tqdm.write(
                "Monte Carlo worker plan: "
                f"{repeat_workers} process(es) × {inner_workers} inner thread(s), "
                f"{len(payloads)} submitted chunk(s), "
                f"{self.n_resamples} inner draws/repeat "
                f"(<= {self.n_jobs} total workers)."
            )

        if pending_repeats:
            if repeat_workers == 1:
                _init_mc_process_worker(
                    real,
                    synthetics,
                    self.metrics,
                    self.n_resamples,
                    self.tickers_per_draw,
                    inner_workers,
                    self.inner_chunk_size,
                )
                with tqdm(
                    total=pending_repeats,
                    desc="Monte Carlo seed repeats",
                    disable=not show_progress,
                    position=progress_position,
                    leave=leave_progress,
                    unit="repeat",
                ) as progress:
                    for payload in payloads:
                        metric_chunk, aggregate_chunk = _evaluate_mc_process_chunk(payload)
                        metric_records.extend(metric_chunk)
                        aggregate_records.extend(aggregate_chunk)
                        completed = len({record["repeat_id"] for record in metric_chunk})
                        progress.update(completed)
                        if progress_callback is not None:
                            progress_callback(completed * self.n_resamples)
            else:
                with ProcessPoolExecutor(
                    max_workers=repeat_workers,
                    initializer=_init_mc_process_worker,
                    initargs=(
                        real,
                        synthetics,
                        self.metrics,
                        self.n_resamples,
                        self.tickers_per_draw,
                        inner_workers,
                        self.inner_chunk_size,
                    ),
                ) as executor:
                    futures = [
                        executor.submit(_evaluate_mc_process_chunk, payload) for payload in payloads
                    ]
                    with tqdm(
                        total=pending_repeats,
                        desc="Monte Carlo seed repeats",
                        disable=not show_progress,
                        position=progress_position,
                        leave=leave_progress,
                        unit="repeat",
                    ) as progress:
                        for future in as_completed(futures):
                            metric_chunk, aggregate_chunk = future.result()
                            metric_records.extend(metric_chunk)
                            aggregate_records.extend(aggregate_chunk)
                            completed_ids = {record["repeat_id"] for record in metric_chunk}
                            completed = len(completed_ids)
                            progress.update(completed)
                            if progress_callback is not None:
                                progress_callback(completed * self.n_resamples)

        metric_replicates = (
            pd.DataFrame(metric_records)
            .sort_values(["repeat_id", "metric", "generator"], kind="stable")
            .reset_index(drop=True)
        )
        aggregate_replicates = (
            pd.DataFrame(aggregate_records)
            .sort_values(["repeat_id", "generator"], kind="stable")
            .reset_index(drop=True)
        )

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
        aggregate_summary = aggregate_replicates.groupby(
            "generator", as_index=False, sort=False
        ).agg(
            mc_mean=("G", "mean"),
            mc_sd=("G", "std"),
            mc_min=("G", "min"),
            mc_max=("G", "max"),
            mc_q025=("G", lambda values: values.quantile(0.025)),
            mc_q975=("G", lambda values: values.quantile(0.975)),
            n_mc_valid=("G", "count"),
        )

        return ReplicateAnalysis(
            metric_summary=metric_summary,
            aggregate_summary=aggregate_summary,
            metric_replicates=metric_replicates,
            aggregate_replicates=aggregate_replicates,
        )
