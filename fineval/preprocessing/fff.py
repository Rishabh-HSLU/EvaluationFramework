"""Flexible Fourier Form (FFF) intraday volatility deseasonalizer.

Intraday equity returns exhibit a pronounced U-shaped variance profile
across the trading session — high at the open, low at midday, rising
again toward the close. This diurnal pattern is a nuisance for every
metric : without removal it dominates the marginal distribution,
inflates absolute-return ACF at session-periodic lags, and
contaminates the leverage curve.

The FFF models log intraday variance as a Fourier series over the
trading session and divides each return by the corresponding minute's
estimated seasonal volatility, producing approximately unit-variance
returns whose remaining structure is the signal each metric targets.

Reference
---------
Andersen, T. G., & Bollerslev, T. (1997). Intraday periodicity and
volatility persistence in financial markets. *Journal of Empirical
Finance*, 4(2–3), 115–158.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .session_clock import session_minute_position


class FFFDeseasonalizer:
    """Estimates and removes the U-shaped intraday volatility smile.

    The log-variance profile is represented as:

        log V(τ) = c₀ + Σⱼ₌₁ᴷ [ aⱼ cos(2πjτ/M) + bⱼ sin(2πjτ/M) ]

    where τ ∈ {0, …, M−1} is the zero-based return-bar position
    within the session and M = 390 is the number of one-minute returns.
    Thus τ = 0 corresponds to the return timestamped 09:31 ET and
    τ = 389 corresponds to the return timestamped 16:00 ET.

    For each series independently, fitting averages squared returns across
    available sessions at each minute slot and fits a separate Fourier
    variance profile.

    Attributes:
        num_harmonics
            Number of Fourier harmonics K. Total free parameters in
            the model is 1 + 2K. K = 4 follows Andersen & Bollerslev
            (1997) and captures the open/close peaks plus the midday
            trough without overfitting intraday microstructure.
        trading_minutes
            Session length in minutes. 390 for NYSE (09:30–16:00 ET).
        coefficients
            Fitted FFF coefficients (c₀, a₁, b₁, …, aK, bK). Only
            available after ``fit`` has been called.
        vol_smile
            Estimated σ(τ) for τ = 0, …, M−1, with one column per
            fitted return series.

    Example::

        fff = FFFDeseasonalizer(num_harmonics=4)
        fff.fit(log_returns_real)

        deseas_real = fff.transform(log_returns_real)
        deseas_ail  = fff.transform(log_returns_ail)
    """

    def __init__(self, num_harmonics: int = 4, trading_minutes: int = 390) -> None:
        self.num_harmonics = num_harmonics
        self.trading_minutes = trading_minutes
        # One coefficient vector and one volatility profile per series.
        self._coefficients: dict[object, np.ndarray] | None = None
        self._vol_smile: pd.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        """True after fit() has completed successfully."""
        return self._coefficients is not None and self._vol_smile is not None

    @property
    def coefficients(self) -> dict[object, np.ndarray]:
        """FFF coefficients (c₀, a₁, b₁, …, aK, bK) for each fitted series."""
        self._require_fitted()
        return self._coefficients

    @property
    def vol_smile(self) -> pd.DataFrame:
        """Estimated σ(τ), indexed by minute slot and with one column per series."""
        self._require_fitted()
        return self._vol_smile

    def _session_slots(self, index: pd.DatetimeIndex) -> pd.Index:
        """Return zero-based return-bar slots: 09:31 -> 0, 16:00 -> 389.

        session_minute_position() measures elapsed minutes from 09:30,
        producing positions 1, ..., 390. FFF slots instead use
        zero-based return-bar positions 0, ..., 389.
        """
        raw_positions = pd.Series(
            np.asarray(session_minute_position(index)),
            copy=False,
        )
        slots = raw_positions.astype(np.int64) - 1

        invalid = (slots < 0) | (slots >= self.trading_minutes)
        if invalid.any():
            invalid_values = sorted(slots.loc[invalid].unique().tolist())
            raise ValueError(
                "Found session positions outside the expected FFF range "
                f"0, ..., {self.trading_minutes - 1}: {invalid_values}"
            )

        return pd.Index(slots.to_numpy(), name="fff_slot")

    def fit(self, log_returns: pd.DataFrame) -> FFFDeseasonalizer:
        """Estimate one intraday volatility profile per real return series.

        For each series, squared returns are averaged across available
        sessions at each zero-based return-bar slot. A separate Fourier
        variance profile is then fitted to each series.

        The resulting profiles may subsequently be applied to both the
        real reference data and corresponding synthetic series.
        """
        minute_slots = self._session_slots(log_returns.index)
        expected_slots = pd.RangeIndex(self.trading_minutes, name="fff_slot")

        # One empirical variance estimate per series and minute slot.
        var_profile = (
            log_returns.pow(2).groupby(minute_slots, sort=True).mean().reindex(expected_slots)
        )

        # Do not silently fit an incomplete intraday profile.
        missing_slots = {
            column: var_profile.index[var_profile[column].isna()].tolist()
            for column in var_profile.columns
            if var_profile[column].isna().any()
        }
        if missing_slots:
            raise ValueError(
                f"Missing empirical variance estimates for one or more series: {missing_slots}"
            )

        time_steps = expected_slots.to_numpy(dtype=float)
        coefficients: dict[object, np.ndarray] = {}
        vol_smile = pd.DataFrame(
            index=expected_slots,
            columns=log_returns.columns,
            dtype=float,
        )

        for column in log_returns.columns:
            log_variance = np.log(var_profile[column].to_numpy(dtype=float) + 1e-12)

            p0 = np.zeros(1 + 2 * self.num_harmonics)
            p0[0] = log_variance.mean()

            fitted_coefficients, _ = curve_fit(
                self._fourier_log_variance,
                time_steps,
                log_variance,
                p0=p0,
                maxfev=10_000,
            )

            fitted_log_variance = self._fourier_log_variance(
                time_steps,
                *fitted_coefficients,
            )

            coefficients[column] = fitted_coefficients
            vol_smile[column] = np.clip(
                np.exp(0.5 * fitted_log_variance),
                a_min=1e-8,
                a_max=None,
            )

        self._coefficients = coefficients
        self._vol_smile = vol_smile
        return self

    def transform(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Divide each return by its series-specific seasonal volatility."""
        self._require_fitted()

        missing_columns = [
            column for column in log_returns.columns if column not in self._vol_smile.columns
        ]
        if missing_columns:
            raise ValueError(f"No fitted FFF profile is available for columns: {missing_columns}")

        minute_slots = self._session_slots(log_returns.index)

        sigma = self._vol_smile.loc[
            minute_slots.to_numpy(),
            log_returns.columns,
        ].to_numpy(dtype=float)

        transformed = log_returns.to_numpy(dtype=float, copy=False) / sigma

        return pd.DataFrame(
            transformed,
            index=log_returns.index,
            columns=log_returns.columns,
        )

    def fit_transform(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Fit on log_returns, then return the transformed version."""
        return self.fit(log_returns).transform(log_returns)

    def _fourier_log_variance(self, time_steps: np.ndarray, *coeffs: float) -> np.ndarray:
        """Evaluate the Fourier log-variance model at given minute slots."""
        est = np.full_like(time_steps, coeffs[0], dtype=float)
        for j in range(1, self.num_harmonics + 1):
            angle = 2 * np.pi * j * time_steps / self.trading_minutes
            est += coeffs[2 * j - 1] * np.cos(angle) + coeffs[2 * j] * np.sin(angle)
        return est

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("FFFDeseasonalizer has not been fitted. Call fit() first.")
