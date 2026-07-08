"""Matched-N ticker bootstrap engine.

Produces, for every (metric, generator) pair, the two distance arrays
the scoring rule needs:

    g_rr[b] = distance(real draw A_b, real draw B_b)      — noise floor
    g_sr[b] = distance(real draw A_b, synthetic draw S_b) — generator gap

Each resample b draws `tickers_per_draw` tickers with replacement,
independently for A, B and each synthetic S. The real draw A_b is
shared between g_rr and every generator's g_sr (matched design), so
per-draw sampling noise partially cancels out of the score.

RNG streams are independent per role: the real draws come from one
stream and each generator's draws from its own stream keyed by the
generator's name. Adding or removing a generator therefore never
changes the subsamples drawn for the others — every (metric,
generator) cell is reproducible in isolation under a fixed seed.

The similarity score is the metric's own normalize():

    s = mean(g_rr) / (mean(g_rr) + mean(g_sr))

and its confidence interval comes from a paired bootstrap over the
resample index — the same index re-drawn into both arrays per
iteration, preserving the per-b correlation created by sharing A_b.

The aggregate fidelity score G (compute_aggregate) condenses one
generator's row into a single number: the geometric mean of the
per-metric gap ratios r_k = mean(g_sr) / mean(g_rr), computed in log
space. Its CI is a *joint* paired bootstrap — one index set per
iteration, shared across every metric's arrays — preserving the
inter-metric correlation that sharing A_b creates within each draw.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import SEED
from ..metrics.base import BaseMetric


class MatchedTickerBootstrap:
    """Runs the matched-N ticker bootstrap for a set of metrics.

    Attributes:
        metrics : list[BaseMetric]
            Metric instances to evaluate. Each must implement the
            extract_features / compute_distance / normalize contract.
        n_resamples : int
            Number of bootstrap resamples B.
        tickers_per_draw : int
            Tickers drawn (with replacement) per subsample. Capped at
            the panel width if the panel has fewer tickers.
        seed : int
            RNG seed; the full run is reproducible.
        n_ci_boot : int
            Iterations of the paired bootstrap used for the CI.
        g_rr : dict[str, np.ndarray]
            After run(): metric name -> (B,) real-vs-real distances.
        g_sr : dict[str, dict[str, np.ndarray]]
            After run(): metric name -> generator name -> (B,)
            synthetic-vs-real distances.

    Example::

        engine = MatchedTickerBootstrap(metrics=[m1, m2, m4, m6])
        results = engine.run(deseas_real, {"AIL": deseas_ail, "GBM": deseas_gbm})
        aggregate = engine.compute_aggregate()
    """

    def __init__(
        self,
        metrics: list[BaseMetric],
        n_resamples: int = 100,
        tickers_per_draw: int = 200,
        seed: int = SEED,
        n_ci_boot: int = 2000,
    ) -> None:
        self.metrics = metrics
        self.n_resamples = n_resamples
        self.tickers_per_draw = tickers_per_draw
        self.seed = seed
        self.n_ci_boot = n_ci_boot
        self.g_rr: dict[str, np.ndarray] = {}
        self.g_sr: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _subsample(panel: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        """Column-subsample a panel and re-label positionally.

        Sampling with replacement duplicates columns; positional
        relabeling keeps column names unique strings so downstream
        stack/groupby operations behave, without touching the values.
        """
        sub = panel.iloc[:, idx]
        sub.columns = [f"T{i:04d}" for i in range(sub.shape[1])]
        return sub

    def _stream(self, role: int, *names: str) -> np.random.Generator:
        """Independent RNG stream for one (role, *names) key.

        role 0: real draws. role 1: one generator's draws, keyed by its
        name. role 2: one (metric, generator) cell's CI bootstrap, keyed
        by both names. role 3: one generator's joint aggregate-CI
        bootstrap, keyed by its name only — it iterates across all
        metrics at once, so it must not reuse any per-cell role-2
        stream. A stable digest of each name (not its position) means
        adding, removing or reordering generators never perturbs any
        other stream.
        """
        return np.random.default_rng([self.seed, role, *(zlib.crc32(n.encode()) for n in names)])

    @staticmethod
    def _check_no_crc32_collision(names: list[str]) -> None:
        """Guard against two names silently sharing an RNG stream.

        crc32 collisions are astronomically unlikely for a handful of
        generator names, but a collision would otherwise correlate two
        supposedly-independent streams with no error at all.
        """
        digests = [zlib.crc32(n.encode()) for n in names]
        if len(set(digests)) != len(names):
            raise ValueError(f"crc32 collision among generator names {names!r}")

    def run(self, real: pd.DataFrame, synthetics: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute the bootstrap and return the benchmark results.

        Args:
            real: Deseasonalized real returns, wide format (T, N).
            synthetics: Generator name -> deseasonalized synthetic
                returns, each wide format on the same market clock.

        Returns:
            Tidy DataFrame with one row per (metric, generator):
            columns [metric, generator, score, ci_low, ci_high,
            g_rr_mean, g_sr_mean].
        """
        # Independent streams keyed by role + a stable digest of each name
        # (not position), so every cell's draws are fixed under any
        # addition, removal or reordering of generators. Collision-checked
        # below: two names hashing to the same crc32 would otherwise
        # silently correlate their streams with no error.
        names = list(synthetics)
        self._check_no_crc32_collision(names)
        rng_real = self._stream(0)
        rng_synth = {gen: self._stream(1, gen) for gen in names}

        n_real = real.shape[1]
        m = min(self.tickers_per_draw, n_real)
        b_total = self.n_resamples

        self.g_rr = {metric.name: np.empty(b_total) for metric in self.metrics}
        self.g_sr = {
            metric.name: {gen: np.empty(b_total) for gen in synthetics} for metric in self.metrics
        }

        for b in tqdm(range(b_total), desc="Bootstrap resamples"):
            idx_a = rng_real.integers(0, n_real, m)
            idx_b = rng_real.integers(0, n_real, m)
            panel_a = self._subsample(real, idx_a)
            panel_b = self._subsample(real, idx_b)

            features_a = {mt.name: mt.extract_features(panel_a) for mt in self.metrics}
            features_b = {mt.name: mt.extract_features(panel_b) for mt in self.metrics}
            for mt in self.metrics:
                self.g_rr[mt.name][b] = mt.compute_distance(
                    features_a[mt.name], features_b[mt.name]
                )

            for gen, synth in synthetics.items():
                idx_s = rng_synth[gen].integers(0, synth.shape[1], m)
                panel_s = self._subsample(synth, idx_s)
                for mt in self.metrics:
                    features_s = mt.extract_features(panel_s)
                    self.g_sr[mt.name][gen][b] = mt.compute_distance(
                        features_a[mt.name], features_s
                    )

        return self._summarize(synthetics.keys())

    def _summarize(self, generators) -> pd.DataFrame:
        rows = []
        for mt in self.metrics:
            g_rr = self.g_rr[mt.name]
            for gen in generators:
                g_sr = self.g_sr[mt.name][gen]
                rng = self._stream(2, mt.name, gen)
                ci_low, ci_high = self._paired_ci(g_rr, g_sr, rng)
                rows.append(
                    {
                        "metric": mt.name,
                        "generator": gen,
                        "score": mt.normalize(g_rr, g_sr),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "g_rr_mean": float(np.nanmean(g_rr)),
                        "g_sr_mean": float(np.nanmean(g_sr)),
                    }
                )
        return pd.DataFrame(rows)

    def _paired_ci(
        self, g_rr: np.ndarray, g_sr: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, float]:
        """95% paired-bootstrap CI on the similarity score.

        The same resample index is drawn into both arrays per
        iteration, preserving the per-b correlation from the matched
        design; this yields narrower, honest CIs versus resampling the
        two arrays independently.
        """
        b_total = len(g_rr)
        idx = rng.integers(0, b_total, size=(self.n_ci_boot, b_total))
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.nanmean(g_rr[idx], axis=1)
            sr = np.nanmean(g_sr[idx], axis=1)
            boot = rr / (rr + sr)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return float(lo), float(hi)

    def compute_aggregate(self, generators: list[str] | None = None) -> pd.DataFrame:
        """Aggregate fidelity score G per generator, with a joint CI.

        For each metric k the gap ratio r_k = mean(g_sr) / mean(g_rr)
        compares the generator gap to the noise floor; G is the
        geometric mean of the r_k, computed in log space:

            G = exp(mean_k(ln r_k))

        over the K metrics whose r_k is finite and positive. A metric
        with a NaN, zero or negative ratio (degenerate — not expected
        on real data) is dropped rather than raised on; k_used vs
        k_total records how many metrics survived, and a generator
        with no usable metric gets G = NaN. G = 1 means the generator
        sits at the noise floor on the geometric-average metric;
        larger G means a larger aggregate gap.

        Reads the g_rr / g_sr arrays stored by run() — call it after
        run(); it never re-runs the bootstrap.

        Args:
            generators: Generator names to aggregate. Defaults to
                every generator present in the stored g_sr arrays.

        Returns:
            Tidy DataFrame with one row per generator: columns
            [generator, G, ci_low, ci_high, k_used, k_total], the CI
            being the 95% joint paired bootstrap over the resample
            index (role-3 RNG stream, keyed by generator name).
        """
        if not self.g_rr:
            raise RuntimeError("compute_aggregate() requires run() to have been called")
        if generators is None:
            generators = list(self.g_sr[self.metrics[0].name])

        rows = []
        for gen in generators:
            log_ratios = []
            for mt in self.metrics:
                rr = np.nanmean(self.g_rr[mt.name])
                sr = np.nanmean(self.g_sr[mt.name][gen])
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = sr / rr
                if np.isfinite(r) and r > 0:
                    log_ratios.append(np.log(r))
            score = float(np.exp(np.mean(log_ratios))) if log_ratios else float("nan")
            ci_low, ci_high = self._joint_paired_ci_aggregate(gen, self._stream(3, gen))
            rows.append(
                {
                    "generator": gen,
                    "G": score,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "k_used": len(log_ratios),
                    "k_total": len(self.metrics),
                }
            )
        return pd.DataFrame(rows)

    def _joint_paired_ci_aggregate(self, gen: str, rng: np.random.Generator) -> tuple[float, float]:
        """95% joint paired-bootstrap CI on the aggregate score G.

        Mirrors _paired_ci with one extra level of sharing: per
        iteration a single resample index is drawn into *every*
        metric's g_rr and g_sr arrays, preserving both the per-b
        rr/sr correlation of the matched design and the inter-metric
        correlation created by the shared real draw A_b. Each
        iteration recomputes every metric's mean gap ratio and
        aggregates exactly as the point estimate does — non-finite or
        non-positive ratios drop out of that iteration's average, and
        an iteration with no usable metric contributes NaN.
        """
        b_total = len(next(iter(self.g_rr.values())))
        idx = rng.integers(0, b_total, size=(self.n_ci_boot, b_total))
        log_ratios = np.full((len(self.metrics), self.n_ci_boot), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            for i, mt in enumerate(self.metrics):
                rr = np.nanmean(self.g_rr[mt.name][idx], axis=1)
                sr = np.nanmean(self.g_sr[mt.name][gen][idx], axis=1)
                r = sr / rr
                valid = np.isfinite(r) & (r > 0)
                log_ratios[i, valid] = np.log(r[valid])
            boot = np.exp(np.nanmean(log_ratios, axis=0))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return float(lo), float(hi)
