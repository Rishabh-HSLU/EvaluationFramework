"""M1 — Tail-weighted marginal distribution.

Compares the complete pooled marginal return distributions through a
tail-weighted L1 distance between empirical quantile functions.

The metric emphasizes tail discrepancies but retains positive weight
throughout the distribution. It is therefore sensitive to location, scale,
bulk shape, and tail behavior and should not be interpreted as an isolated
measure of tail thickness.
"""

import numpy as np
import pandas as pd

from fineval.metrics.base import BaseMetric


class TailWeightedMarginal(BaseMetric):
    r"""Evaluate marginal-distribution fidelity with tail emphasis.

    The metric compares pooled empirical quantile functions using a smooth,
    strictly positive weight over the entire quantile grid. Tail observations
    receive greater weight, but bulk discrepancies remain part of the metric.
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
        raw_weights = 1.0 + tail_lambda * (u**-tail_alpha + (1.0 - u) ** -tail_alpha)
        self._weights = raw_weights / raw_weights.mean()

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
        flat = returns.to_numpy().ravel()
        valid = flat[np.isfinite(flat)]
        if valid.size == 0:
            return np.full(self.n_grid, np.nan)
        return np.quantile(
            valid,
            self._grid,
            method="linear",
        )

    def compute_distance(self, features_real: np.ndarray, features_synth: np.ndarray) -> float:
        """Compute the normalized tail-weighted quantile gap.

        Args:
            features_real (np.ndarray): The empirical quantile function of the real data.
            features_synth (np.ndarray): The empirical quantile function of the synthetic data.

        Returns:
            float: The weighted integral of the absolute quantile differences.
        """
        gap = np.abs(features_real - features_synth)
        valid = np.isfinite(gap)
        if not np.any(valid):
            return float("nan")
        weights = self._weights[valid]
        return np.sum(weights * gap[valid]) / np.sum(weights)
