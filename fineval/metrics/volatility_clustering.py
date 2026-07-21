"""
M2: Nonlinear Temporal Dependence (Volatility Clustering)

Measures long-memory volatility clustering through the empirical
autocorrelation function (ACF) of absolute returns.

The distance is the mean absolute gap between the real and synthetic
cross-sectionally averaged ACF curves over [lag_min, lag_max].
"""

import numpy as np
import pandas as pd

from fineval.metrics.base import BaseMetric


class VolatilityClustering(BaseMetric):
    """
    Evaluates the volatility clustering fidelity of synthetic financial returns.

    For each ticker, the metric computes the ACF of absolute returns while
    excluding all pairs that cross a trading-session boundary. The resulting
    ticker-level ACF curves are averaged cross-sectionally.

    The final distance is the mean absolute difference between two averaged
    ACF curves over the configured long-lag window.

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
        """Extract the cross-sectionally averaged absolute-return ACF.

        The mean and variance normalization are computed over the complete
        sample of each ticker. Lagged cross-products are accumulated only
        within trading sessions.

        Unsupported lags and tickers with zero variance are represented by
        NaN rather than being treated as zero.

        Args:
            returns (pd.DataFrame): Deseasonalized log returns with shape
                ``(timestamps, tickers)`` and a DatetimeIndex.

        Returns:
            np.ndarray: Array of shape ``(lag_max,)``. Index zero represents lag one.
        """
        absolute_returns = returns.abs().to_numpy(dtype=float)
        means = np.nanmean(absolute_returns, axis=0)
        centered = absolute_returns - means
        # Full-sample variance denominator. Keeping this denominator fixed
        # gives the intended biased-style shrinkage at long lags.
        denominator = np.nansum(centered**2, axis=0)

        # Identify sessions from calendar dates rather than assuming a fixed
        # number of observations per session.
        session_ids = returns.index.normalize()
        # Find every position where the date changes.
        boundaries = np.flatnonzero(np.asarray(session_ids[1:] != session_ids[:-1])) + 1
        blocks = np.split(centered, boundaries, axis=0)

        n_tickers = absolute_returns.shape[1]
        numerator = np.zeros((self.lag_max, n_tickers), dtype=float)
        # Track the number of genuinely available pairs so that structurally
        # unsupported lags are emitted as NaN rather than zero.
        pair_counts = np.zeros((self.lag_max, n_tickers), dtype=int)
        for block in blocks:
            n_observations = block.shape[0]
            if n_observations < 2:
                continue
            supported_lag_max = min(self.lag_max, n_observations - 1)
            fft_length = 1 << int(np.ceil(np.log2(2 * n_observations)))

            # Replacing NaN by zero is exact for the numerator because any
            # product containing a missing value must contribute nothing.
            values = np.nan_to_num(block, nan=0.0)
            value_fft = np.fft.rfft(values, n=fft_length, axis=0)
            autocovariance_sums = np.fft.irfft(value_fft * np.conj(value_fft), n=fft_length, axis=0)
            numerator[:supported_lag_max] += autocovariance_sums[1 : supported_lag_max + 1]

            # Compute the number of finite observation pairs at each lag.
            valid = np.isfinite(block).astype(float)
            valid_fft = np.fft.rfft(valid, n=fft_length, axis=0)
            valid_pair_sums = np.fft.irfft(valid_fft * np.conj(valid_fft), n=fft_length, axis=0)
            block_pair_counts = np.rint(valid_pair_sums[1 : supported_lag_max + 1])
            block_pair_counts = np.maximum(
                block_pair_counts,
                0,
            ).astype(int)
            pair_counts[:supported_lag_max] += block_pair_counts

        with np.errstate(invalid="ignore", divide="ignore"):
            ticker_acfs = numerator / denominator[None, :]

        # Without this line, unsupported lags incorrectly appear as an
        # estimated autocorrelation of zero.
        ticker_acfs[pair_counts == 0] = np.nan
        # Equal-weight average over tickers with finite estimates.
        finite = np.isfinite(ticker_acfs)
        finite_counts = finite.sum(axis=1)

        average_acf = np.full(self.lag_max, np.nan, dtype=float)
        supported = finite_counts > 0
        average_acf[supported] = (
            np.nansum(ticker_acfs[supported], axis=1) / finite_counts[supported]
        )
        return average_acf

    def compute_distance(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Compute the mean absolute ACF gap over the selected lag window.

        Only lags for which both feature vectors contain finite estimates
        are compared. If no common finite lag exists, the distance is NaN.

        Args:
            features_a (np.ndarray): The empirical ACF curve of the real data.
            features_b (np.ndarray): The empirical ACF curve of the synthetic data.

        Returns:
            float: Mean absolute difference over common finite lags, or NaN if
                no lag can be compared.
        """
        # Arrays are 0-indexed where index 0 represents lag 1.
        # So lag_min corresponds to index lag_min - 1.
        start_idx = self.lag_min - 1
        end_idx = self.lag_max

        acf_a = features_a[start_idx:end_idx]
        acf_b = features_b[start_idx:end_idx]

        finite = np.isfinite(acf_a) & np.isfinite(acf_b)
        if not np.any(finite):
            return float("nan")
        return float(np.mean(np.abs(acf_a[finite] - acf_b[finite])))
