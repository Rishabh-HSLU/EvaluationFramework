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
        Extracts the empirical quantile function for the pooled standardized returns.

        Every column is divided by its own standard deviation before the panel is
        pooled. Pooling raw returns across tickers with unequal variances makes the
        marginal a scale mixture, which is leptokurtic even when each individual
        series is Gaussian, so the pooled tail would partly reflect cross-sectional
        volatility dispersion rather than distributional shape. Standardizing per
        ticker removes that artifact.

        The standard deviation is used rather than a robust width such as the MAD
        because it gives every column unit variance, which makes the pooled kurtosis
        the mean of the per-ticker kurtoses. The MAD estimates bulk width only and
        inflates pooled kurtosis on heavy-tailed panels.

        Scales are computed from the panel as passed in, never from a stored
        reference, so the metric remains stateless in the sense required by
        `BaseMetric`. Column subsampling retains every row, so a column's scale does
        not depend on which draw it appears in.

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

        Raises:
            ValueError: If any column has a zero or non-finite standard deviation.
                Such a column is named rather than dropped, so that a data problem
                cannot silently reduce the effective ticker count.
        """
        observed = returns.to_numpy(dtype=float)
        observed = np.where(np.isfinite(observed), observed, np.nan)
        scales = np.nanstd(observed, axis=0, ddof=1)
        degenerate = ~np.isfinite(scales) | (scales <= 0.0)
        if degenerate.any():
            raise ValueError(
                "Cannot standardize columns with zero or non-finite standard "
                f"deviation: {returns.columns[degenerate].tolist()}"
            )
        flat = (observed / scales).ravel()
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
