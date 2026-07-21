"""Matched-N ticker estimator.

The engine computes point estimates on fixed real and synthetic panels. Corpus
bootstrap intervals and independent-seed stability analyses live in
``uncertainty.py`` and are exposed here as thin convenience methods.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from ..config import SEED
from ..metrics.base import BaseMetric
from .execution import execute_inner_draws, resolve_n_jobs, subsample_panel
from .models import ReplicateAnalysis


class MatchedTickerBootstrap:
    """Run the matched ticker estimator for a set of metrics.

    Args:
        metrics: Metrics implementing ``extract_features``,
            ``compute_distance``, and ``normalize``.
        n_resamples: Number of matched ticker draws.
        tickers_per_draw: Tickers sampled with replacement per draw.
        seed: Root seed used to derive independent deterministic streams.
        n_jobs: Total worker budget. ``None`` or ``0`` uses a conservative
            automatic value; ``-1`` uses every logical CPU.
        inner_chunk_size: Inner draws per submitted task. ``None`` chooses
            approximately four tasks per worker.
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
        if n_resamples < 1:
            raise ValueError("n_resamples must be at least 1")
        if tickers_per_draw < 1:
            raise ValueError("tickers_per_draw must be at least 1")
        if inner_chunk_size is not None and inner_chunk_size < 1:
            raise ValueError("inner_chunk_size must be at least 1")

        self.metrics = metrics
        self.n_resamples = n_resamples
        self.tickers_per_draw = tickers_per_draw
        self.seed = seed
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.inner_chunk_size = inner_chunk_size
        self.g_rr: dict[str, np.ndarray] = {}
        self.g_sr: dict[str, dict[str, np.ndarray]] = {}

    resolve_n_jobs = staticmethod(resolve_n_jobs)
    _subsample = staticmethod(subsample_panel)

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
        """Pre-generate all draw indices for order-independent execution."""
        n_real = real.shape[1]
        if n_real == 0:
            raise ValueError("real panel has no ticker columns")

        m = min(self.tickers_per_draw, n_real)
        real_idx = self._stream(0).integers(
            0,
            n_real,
            size=(self.n_resamples, 2, m),
        )
        idx_s: dict[str, np.ndarray] = {}
        for generator, synthetic in synthetics.items():
            if synthetic.shape[1] == 0:
                raise ValueError(f"synthetic panel {generator!r} has no ticker columns")
            idx_s[generator] = self._stream(1, generator).integers(
                0,
                synthetic.shape[1],
                size=(self.n_resamples, m),
            )
        return real_idx[:, 0, :], real_idx[:, 1, :], idx_s

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
        """Compute per-metric point estimates on fixed corpora."""
        names = list(synthetics)
        self._check_no_crc32_collision(names)
        idx_a, idx_b, idx_s = self._prepare_inner_indices(real, synthetics)

        resolved_jobs = self.n_jobs if n_jobs is None else resolve_n_jobs(n_jobs)
        self.g_rr, self.g_sr = execute_inner_draws(
            real=real,
            synthetics=synthetics,
            metrics=self.metrics,
            idx_a=idx_a,
            idx_b=idx_b,
            idx_s=idx_s,
            n_jobs=resolved_jobs,
            chunk_size=self.inner_chunk_size,
            backend=parallel_backend,
            show_progress=show_progress,
            progress_callback=progress_callback,
            progress_position=progress_position,
            leave_progress=leave_progress,
        )
        return self._summarize(names)

    def _summarize(self, generators: list[str]) -> pd.DataFrame:
        """Build point estimates from the stored gap arrays."""
        rows = []
        for metric in self.metrics:
            g_rr = self.g_rr[metric.name]
            for generator in generators:
                g_sr = self.g_sr[metric.name][generator]
                rows.append(
                    {
                        "metric": metric.name,
                        "generator": generator,
                        "score": metric.normalize(g_rr, g_sr),
                        "g_rr_mean": float(np.nanmean(g_rr)),
                        "g_sr_mean": float(np.nanmean(g_sr)),
                    }
                )
        return pd.DataFrame(rows)

    def compute_aggregate(
        self,
        generators: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute the symmetric aggregate deviation score ``G_dev``.

        For each metric, the gap ratio is ``r = mean(g_sr) / mean(g_rr)``.
        The aggregate is

            G_dev = exp(mean(abs(log(r)))).

        ``G_dev`` is at least one and equals one only when every included
        metric-specific ratio equals one. Ratios above and below one therefore
        cannot cancel.
        """
        if not self.g_rr:
            raise RuntimeError("compute_aggregate() requires run() to have been called")
        if generators is None:
            generators = list(self.g_sr[self.metrics[0].name])
        rows = []
        for generator in generators:
            log_deviations: list[float] = []
            for metric in self.metrics:
                rr = float(np.nanmean(self.g_rr[metric.name]))
                sr = float(np.nanmean(self.g_sr[metric.name][generator]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = sr / rr
                if np.isfinite(ratio) and ratio > 0:
                    log_deviations.append(abs(float(np.log(ratio))))
            k_used = len(log_deviations)
            score = (
                float(np.exp(np.mean(log_deviations)))
                if k_used == len(self.metrics)
                else float("nan")
            )
            rows.append(
                {
                    "generator": generator,
                    "G_dev": score,
                    "k_used": k_used,
                    "k_total": len(self.metrics),
                }
            )
        return pd.DataFrame(rows)

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
        """Run the complete-corpus outer bootstrap."""
        from .uncertainty import run_outer_bootstrap

        return run_outer_bootstrap(
            self,
            real,
            synthetics,
            n_outer_resamples=n_outer_resamples,
            show_progress=show_progress,
            replicate_chunk_size=replicate_chunk_size,
            progress_callback=progress_callback,
            progress_position=progress_position,
            leave_progress=leave_progress,
        )

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
        """Run independent-seed numerical-stability repeats."""
        from .uncertainty import run_monte_carlo_stability

        return run_monte_carlo_stability(
            self,
            real,
            synthetics,
            n_repeats=n_repeats,
            base_metric_result=base_metric_result,
            base_aggregate_result=base_aggregate_result,
            show_progress=show_progress,
            replicate_chunk_size=replicate_chunk_size,
            progress_callback=progress_callback,
            progress_position=progress_position,
            leave_progress=leave_progress,
        )
