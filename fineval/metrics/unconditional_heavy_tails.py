"""
M1: Unconditional Heavy Tails

Measures the mismatch in the marginal distribution of returns by evaluating the
tail-weighted Wasserstein-1 distance. This metric isolates the absolute tail thickness
across the entire sample, ignoring temporal ordering and conditionally filtering
out the bulk of the distribution where operational risk is negligible.
"""

import numpy as np
import pandas as pd

from fineval.metrics.base import BaseMetric


class UnconditionalHeavyTails(BaseMetric):
    """
    Evaluates the marginal tail distribution fidelity of synthetic financial returns.

    The metric implements a tail-weighted Wasserstein-1 (W1) distance using the
    quantile representation:
        W1_tail = \int_0^1 w(u) |F^{-1}(u) - \hat{F}^{-1}(u)| du

    Standard metrics like the Kolmogorov-Smirnov test are geometrically bounded in
    the tails and thus inherently bulk-biased. Moment-matching approaches (e.g.,
    kurtosis) suffer from multiplicative outlier amplification, leading to extreme
    sample noise. This metric resolves both defects by directly integrating the
    absolute difference between the empirical quantile functions strictly in the
    regions of interest, defined by the `tail_theta` threshold.

    Attributes:
        name (str): Identifier for the metric instance.
        n_grid (int): Resolution of the uniform grid used to evaluate the quantile function.
        tail_theta (float): The fraction of the distribution designated as the tail
            on each side (e.g., 0.05 isolates the top 5% and bottom 5% quantiles).
    """

    def __init__(self, name: str, n_grid: int, tail_alpha: float, tail_lambda: float):
        """
        Initializes the Unconditional Heavy Tails metric.

        Args:
            name (str): Identifier for the metric.
            n_grid (int): Number of evaluation points for the quantile grid.
            tail_theta (float): Threshold determining the tail region.
        """

        super().__init__(name)
        self.n_grid = n_grid
        self.tail_alpha = tail_alpha
        self.tail_lambda = tail_lambda
        self._grid = np.linspace(0.0, 1.0, n_grid, endpoint=False) + (0.5 / n_grid)
        u = self._grid
        self._weights = 1.0 + tail_lambda * (u**-tail_alpha + (1.0 - u) ** -tail_alpha)

    def extract_features(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Extracts the empirical quantile function for the fully pooled return series.

        This method strictly adheres to the framework's non-imputation policy.
        It flattens the wide return matrix and aggressively drops any `NaN` values
        (such as structural overnight gaps or illiquidity periods), ensuring the
        marginal distribution is estimated exclusively from realized market prints.

        Args:
            returns (pd.DataFrame): A wide matrix of deseasonalized log returns
                (T timestamps x N tickers).

        Returns:
            np.ndarray: A 1D array of shape `(n_grid,)` representing the quantile
                values evaluated over a uniform grid `u \\in (0, 1)`.
        """
        flat = returns.values.flatten()
        valid = flat[~np.isnan(flat)]
        return np.quantile(valid, self._grid)

    def compute_distance(self, features_real: np.ndarray, features_synth: np.ndarray) -> float:
        """
        Computes the tail-weighted Wasserstein-1 integrated gap.

        The distance is strictly localized to the extreme quantiles. Differences
        in the bulk of the distribution (between `tail_theta` and `1 - tail_theta`)
        are masked out via the indicator weight function, focusing the metric exclusively
        on systemic operational risk regions.

        Args:
            features_real (np.ndarray): The empirical quantile function of the real data.
            features_synth (np.ndarray): The empirical quantile function of the synthetic data.

        Returns:
            float: The weighted integral of the absolute quantile differences.
        """
        integrand = self._weights * np.abs(features_real - features_synth)
        return float(np.nanmean(integrand))

    def normalize(self, g_rr: np.ndarray, g_sr: np.ndarray) -> float:
        """
        Converts raw Wasserstein gaps into a bounded similarity score.

        The normalization utilizes the real-real baseline noise floor to interpret
        the magnitude of the synthetic deviation.

        Args:
            g_rr (np.ndarray): Array of real-vs-real baseline distances.
            g_sr (np.ndarray): Array of synthetic-vs-real distances.

        Returns:
            float: A similarity score bounded in [0, 1], where 1.0 indicates perfect
                alignment and 0.5 denotes parity with empirical sampling noise.
        """
        rr_mean = float(g_rr.mean())
        sr_mean = float(g_sr.mean())

        if rr_mean + sr_mean == 0.0:
            return 0.0

        return rr_mean / (rr_mean + sr_mean)
