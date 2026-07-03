"""
M2: Nonlinear Temporal Dependence (Volatility Clustering)

Measures the long-memory volatility clustering of returns by evaluating the
mean absolute gap in the empirical autocorrelation function (ACF) of absolute
returns (|r|) across long lags (e.g., 60 to 390 minutes).
"""

import numpy as np
import pandas as pd

from fineval.metrics.base import BaseMetric


class VolatilityClustering(BaseMetric):
    """
    Evaluates the volatility clustering fidelity of synthetic financial returns.

    The metric computes the autocorrelation function (ACF) of absolute returns (|r|)
    over a designated long-memory lag window. It specifically targets the long-lag
    behavior (e.g., k=60 to k=390) because short-lag autocorrelation is easily
    reproduced by naive short-memory generators (like standard GARCH). Furthermore,
    it utilizes |r| instead of r^2 to prevent multiplicative outlier amplification,
    which severely degrades the signal-to-noise ratio of the estimator at long lags.

    The final distance is the summed absolute difference between the real and
    synthetic cross-sectional average ACF curves over the defined lag window.

    Attributes:
        name (str): Identifier for the metric instance.
        lag_min (int): The starting lag (inclusive) for the evaluation window.
        lag_max (int): The maximum lag (inclusive) for the evaluation window.
    """

    def __init__(self, name: str, lag_min: int, lag_max: int):
        """
        Initializes the Nonlinear Temporal Dependence metric.

        Args:
            name (str): Identifier for the metric.
            lag_min (int): Minimum lag to include in the ACF gap.
            lag_max (int): Maximum lag to include in the ACF gap.
        """
        super().__init__(name)
        self.lag_min = lag_min
        self.lag_max = lag_max

    def extract_features(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Extracts the empirical ACF of absolute returns for the panel.

        This method operates per-path (per-ticker) to strictly preserve the temporal
        ordering of the sequence. It computes the per-ticker ACF for all lags up to
        `lag_max`, utilizing pairwise masking to gracefully ignore `NaN` values without
        artificially stitching non-adjacent timestamps. The resulting per-ticker ACF
        curves are then averaged cross-sectionally.

        Args:
            returns (pd.DataFrame): A wide matrix of deseasonalized log returns
                (T timestamps x N tickers).

        Returns:
            np.ndarray: A 1D array of shape `(lag_max,)` representing the
                cross-sectionally averaged ACF values for lags 1 through `lag_max`.
        """

        a = returns.abs().values
        means = np.nanmean(a, axis=0)
        centered = a - means
        denominator = np.nansum(centered**2, axis=0)

        # Session boundaries from the calendar clock, not a fixed 390 stride,
        # because the sample contains variable-length half-day sessions.
        session_ids = returns.index.normalize()
        bounds = np.flatnonzero(session_ids[1:].values != session_ids[:-1].values) + 1
        blocks = np.split(centered, bounds, axis=0)

        # Per-block lag-k cross-product sums via FFT autocorrelation.
        # NaN -> 0 is exact here: the original nansum skipped any pair
        # with a NaN member, and a zero factor contributes zero to the
        # sum. Zero-padding to >= 2n makes the circular correlation
        # linear, so s[k] = sum_t c[t] * c[t+k] within the block.
        num = np.zeros((self.lag_max, a.shape[1]))
        for blk in blocks:
            n = blk.shape[0]
            if n < 2:
                continue
            c = np.nan_to_num(blk, nan=0.0)
            pad = 1 << int(np.ceil(np.log2(2 * n)))
            f = np.fft.rfft(c, n=pad, axis=0)
            s = np.fft.irfft(f * np.conj(f), n=pad, axis=0)
            kmax = min(self.lag_max, n - 1)
            num[:kmax] += s[1 : kmax + 1]

        with np.errstate(invalid="ignore", divide="ignore"):
            ticker_acfs = num / denominator
        return np.nanmean(ticker_acfs, axis=1)

    def compute_distance(self, features_real: np.ndarray, features_synth: np.ndarray) -> float:
        """
        Computes the lag-weighted absolute ACF gap.

        Differences in the short-memory regime (lags < lag_min) are strictly excluded,
        focusing the metric completely on the long-memory persistence.

        Args:
            features_real (np.ndarray): The empirical ACF curve of the real data.
            features_synth (np.ndarray): The empirical ACF curve of the synthetic data.

        Returns:
            float: The sum of the absolute differences over the lag window.
        """
        # Arrays are 0-indexed where index 0 represents lag 1.
        # So lag_min corresponds to index lag_min - 1.
        start_idx = self.lag_min - 1
        end_idx = self.lag_max

        rho_real = features_real[start_idx:end_idx]
        rho_synth = features_synth[start_idx:end_idx]

        return float(np.nansum(np.abs(rho_real - rho_synth)))

    def normalize(self, g_rr: np.ndarray, g_sr: np.ndarray) -> float:
        """
        Converts raw ACF gaps into a bounded similarity score.

        Args:
            g_rr (np.ndarray): Array of real-vs-real baseline distances.
            g_sr (np.ndarray): Array of synthetic-vs-real distances.

        Returns:
            float: A similarity score bounded in [0, 1].
        """
        rr_mean = float(g_rr.mean())
        sr_mean = float(g_sr.mean())

        if rr_mean + sr_mean == 0.0:
            return 0.0

        return rr_mean / (rr_mean + sr_mean)
