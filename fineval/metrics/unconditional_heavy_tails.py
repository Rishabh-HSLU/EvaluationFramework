"""
M1: Unconditional Heavy Tails

Measures the mismatch in the marginal distribution of returns by evaluating a
tail-emphasized (soft-weighted) Wasserstein-1 distance in quantile
representation. This metric targets absolute tail thickness across the entire
sample, ignoring temporal ordering: the bulk of the distribution enters at
weight ≈ 1 while the tails are progressively up-weighted.
"""

import numpy as np
import pandas as pd

from fineval.metrics.base import BaseMetric


class UnconditionalHeavyTails(BaseMetric):
    r"""
    Evaluates the marginal tail distribution fidelity of synthetic financial returns.

    The metric implements a tail-emphasized (soft-weighted) Wasserstein-1 (W1)
    distance using the quantile representation:
        W1_tail = \int_0^1 w(u) |F^{-1}(u) - \hat{F}^{-1}(u)| du
    with the smooth, strictly positive weight
        w(u) = 1 + tail_lambda * (u^{-tail_alpha} + (1 - u)^{-tail_alpha}),
    evaluated on a midpoint grid excluding u ∈ {0, 1}. The bulk is included at
    weight ≈ 1; the tails are progressively up-weighted as u approaches 0 or 1,
    so this is a tail-emphasized distance, not a tail-only one.

    Standard metrics like the Kolmogorov-Smirnov test are geometrically bounded in
    the tails and thus inherently bulk-biased. Moment-matching approaches (e.g.,
    kurtosis) suffer from multiplicative outlier amplification, leading to extreme
    sample noise. This metric resolves both defects by integrating the absolute
    difference between the empirical quantile functions under a weight that
    emphasizes the tail regions without discarding the bulk.

    Attributes:
        name (str): Identifier for the metric instance.
        n_grid (int): Resolution of the uniform grid used to evaluate the quantile function.
        tail_alpha (float): Exponent controlling how sharply the weight grows
            as u approaches 0 or 1 (larger values concentrate the emphasis
            deeper in the tails).
        tail_lambda (float): Multiplier on the tail term, setting the strength
            of tail emphasis relative to the baseline bulk weight of 1.
    """

    def __init__(self, name: str, n_grid: int, tail_alpha: float, tail_lambda: float):
        """
        Initializes the Unconditional Heavy Tails metric.

        Args:
            name (str): Identifier for the metric.
            n_grid (int): Number of evaluation points for the quantile grid.
            tail_alpha (float): Exponent of the tail up-weighting term.
            tail_lambda (float): Strength of the tail emphasis relative to
                the unit bulk weight.
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
        Computes the tail-emphasized Wasserstein-1 integrated gap.

        The absolute quantile gap is integrated under the smooth, strictly
        positive weight w(u): differences in the bulk of the distribution
        contribute at weight ≈ 1, while differences in the extreme quantiles
        are progressively up-weighted.

        Args:
            features_real (np.ndarray): The empirical quantile function of the real data.
            features_synth (np.ndarray): The empirical quantile function of the synthetic data.

        Returns:
            float: The weighted integral of the absolute quantile differences.
        """
        integrand = self._weights * np.abs(features_real - features_synth)
        return float(np.nanmean(integrand))
