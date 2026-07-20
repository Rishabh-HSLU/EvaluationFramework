"""Matched-N ticker resampling engine.

The primary point estimate uses repeated matched ticker draws:

    g_rr[b] = distance(real draw A_b, real draw B_b)      — noise floor
    g_sr[b] = distance(real draw A_b, synthetic draw S_b) — generator gap

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

All inner indices, outer corpus indices, and replicate seeds are
pre-generated from role-specific RNG streams. Therefore each replicate
is fixed before execution and can later be distributed across workers
or chunks without making results depend on scheduling order.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import SEED
from ..metrics.base import BaseMetric


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
    """

    def __init__(
        self,
        metrics: list[BaseMetric],
        n_resamples: int = 100,
        tickers_per_draw: int = 200,
        seed: int = SEED,
    ) -> None:
        self.metrics = metrics
        self.n_resamples = n_resamples
        self.tickers_per_draw = tickers_per_draw
        self.seed = seed
        self.g_rr: dict[str, np.ndarray] = {}
        self.g_sr: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _subsample(panel: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        """Column-subsample a panel and relabel columns positionally."""
        sub = panel.iloc[:, idx]
        sub.columns = [f"T{i:04d}" for i in range(sub.shape[1])]
        return sub

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
    ) -> pd.DataFrame:
        """Run the inner matched ticker estimator on fixed corpora.

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

        iterator = tqdm(
            range(b_total),
            desc="Matched ticker resamples",
            disable=not show_progress,
        )
        for b in iterator:
            panel_a = self._subsample(real, idx_a_all[b])
            panel_b = self._subsample(real, idx_b_all[b])

            features_a = {metric.name: metric.extract_features(panel_a) for metric in self.metrics}
            features_b = {metric.name: metric.extract_features(panel_b) for metric in self.metrics}

            for metric in self.metrics:
                self.g_rr[metric.name][b] = metric.compute_distance(
                    features_a[metric.name],
                    features_b[metric.name],
                )

            for gen, synth in synthetics.items():
                panel_s = self._subsample(synth, idx_s_all[gen][b])
                for metric in self.metrics:
                    features_s = metric.extract_features(panel_s)
                    self.g_sr[metric.name][gen][b] = metric.compute_distance(
                        features_a[metric.name],
                        features_s,
                    )

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

        These arrays can later be sliced into arbitrary worker chunks.
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
    ) -> ReplicateAnalysis:
        """Estimate corpus-resampling 95% confidence intervals.

        Each outer replicate resamples the complete real corpus and
        each complete synthetic corpus with replacement. The full inner
        matched estimator is then rerun on those bootstrap corpora.

        The resulting intervals are conditional on the already-fitted
        preprocessing and synthetic panels passed to this method. They
        do not include preprocessing refit, generator refit, or generator
        retraining uncertainty.
        """
        if n_outer_resamples < 2:
            raise ValueError("n_outer_resamples must be at least 2")

        names = list(synthetics)
        self._check_no_crc32_collision(names)
        outer_real_idx, outer_synth_idx = self._prepare_outer_inputs(
            real,
            synthetics,
            n_outer_resamples,
        )

        metric_runs = []
        aggregate_runs = []
        iterator = tqdm(
            range(n_outer_resamples),
            desc="Outer corpus bootstrap",
            disable=not show_progress,
        )
        for outer_id in iterator:
            outer_real = self._subsample(real, outer_real_idx[outer_id])
            outer_synthetics = {
                gen: self._subsample(synth, outer_synth_idx[gen][outer_id])
                for gen, synth in synthetics.items()
            }
            child_seed = self.seed
            child = MatchedTickerBootstrap(
                metrics=self.metrics,
                n_resamples=self.n_resamples,
                tickers_per_draw=self.tickers_per_draw,
                seed=child_seed,
            )
            metric_result = child.run(
                outer_real,
                outer_synthetics,
                show_progress=False,
            )
            metric_result.insert(0, "outer_id", outer_id)
            metric_result.insert(1, "inner_seed", child_seed)
            metric_runs.append(metric_result)

            aggregate_result = child.compute_aggregate()
            aggregate_result.insert(0, "outer_id", outer_id)
            aggregate_result.insert(1, "inner_seed", child_seed)
            aggregate_runs.append(aggregate_result)

        metric_replicates = pd.concat(metric_runs, ignore_index=True)
        aggregate_replicates = pd.concat(aggregate_runs, ignore_index=True)

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
    ) -> ReplicateAnalysis:
        """Assess numerical stability across independent inner seeds.

        The observed real and synthetic corpora remain fixed. Reported
        spread therefore quantifies Monte Carlo sensitivity only and
        must not be described as corpus-level confidence coverage.
        """
        if n_repeats < 2:
            raise ValueError("n_repeats must be at least 2")

        seeds = self._prepare_monte_carlo_seeds(n_repeats)
        metric_runs = []
        aggregate_runs = []

        iterator = tqdm(
            range(n_repeats),
            desc="Monte Carlo seed repeats",
            disable=not show_progress,
        )
        for repeat_id in iterator:
            repeat_seed = int(seeds[repeat_id])
            if (
                repeat_id == 0
                and base_metric_result is not None
                and base_aggregate_result is not None
            ):
                metric_result = base_metric_result.copy()
                aggregate_result = base_aggregate_result.copy()
            else:
                child = MatchedTickerBootstrap(
                    metrics=self.metrics,
                    n_resamples=self.n_resamples,
                    tickers_per_draw=self.tickers_per_draw,
                    seed=repeat_seed,
                )
                metric_result = child.run(real, synthetics, show_progress=False)
                aggregate_result = child.compute_aggregate()

            metric_result.insert(0, "repeat_id", repeat_id)
            metric_result.insert(1, "seed", repeat_seed)
            metric_runs.append(metric_result)

            aggregate_result.insert(0, "repeat_id", repeat_id)
            aggregate_result.insert(1, "seed", repeat_seed)
            aggregate_runs.append(aggregate_result)

        metric_replicates = pd.concat(metric_runs, ignore_index=True)
        aggregate_replicates = pd.concat(aggregate_runs, ignore_index=True)

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
