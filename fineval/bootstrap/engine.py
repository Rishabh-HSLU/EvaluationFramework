"""Matched-N ticker bootstrap engine.

Produces, for every (metric, generator) pair, the two distance arrays
the scoring rule needs:

    g_rr[b] = distance(real draw A_b, real draw B_b)      — noise floor
    g_sr[b] = distance(real draw A_b, synthetic draw S_b) — generator gap

Each resample b draws `tickers_per_draw` tickers with replacement,
independently for A, B and each synthetic S. The real draw A_b is
shared between g_rr and every generator's g_sr (matched design), so
per-draw sampling noise partially cancels out of the score.

The similarity score is the metric's own normalize():

    s = mean(g_rr) / (mean(g_rr) + mean(g_sr))

and its confidence interval comes from a paired bootstrap over the
resample index — the same index re-drawn into both arrays per
iteration, preserving the per-b correlation created by sharing A_b.
"""

from __future__ import annotations

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
        rng = np.random.default_rng(self.seed)
        n_real = real.shape[1]
        m = min(self.tickers_per_draw, n_real)
        b_total = self.n_resamples

        self.g_rr = {metric.name: np.empty(b_total) for metric in self.metrics}
        self.g_sr = {
            metric.name: {gen: np.empty(b_total) for gen in synthetics} for metric in self.metrics
        }

        for b in tqdm(range(b_total), desc="Bootstrap resamples"):
            idx_a = rng.integers(0, n_real, m)
            idx_b = rng.integers(0, n_real, m)
            panel_a = self._subsample(real, idx_a)
            panel_b = self._subsample(real, idx_b)

            features_a = {mt.name: mt.extract_features(panel_a) for mt in self.metrics}
            features_b = {mt.name: mt.extract_features(panel_b) for mt in self.metrics}
            for mt in self.metrics:
                self.g_rr[mt.name][b] = mt.compute_distance(
                    features_a[mt.name], features_b[mt.name]
                )

            for gen, synth in synthetics.items():
                idx_s = rng.integers(0, synth.shape[1], m)
                panel_s = self._subsample(synth, idx_s)
                for mt in self.metrics:
                    features_s = mt.extract_features(panel_s)
                    self.g_sr[mt.name][gen][b] = mt.compute_distance(
                        features_a[mt.name], features_s
                    )

        return self._summarize(synthetics.keys(), rng)

    def _summarize(self, generators, rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for mt in self.metrics:
            g_rr = self.g_rr[mt.name]
            for gen in generators:
                g_sr = self.g_sr[mt.name][gen]
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
