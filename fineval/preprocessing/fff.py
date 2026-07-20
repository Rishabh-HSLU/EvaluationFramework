"""Flexible Fourier Form (FFF) intraday volatility deseasonalizer.

Intraday equity returns exhibit a pronounced U-shaped variance profile
across the trading session — high at the open, low at midday, rising
again toward the close. This diurnal pattern is a nuisance for every
metric: without removal it dominates the marginal distribution,
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
    """Estimates and removes intraday volatility seasonality.

    The log-variance profile is represented as:

        log V(τ) = c₀ + Σⱼ₌₁ᴷ [aⱼ cos(2πjτ/M) + bⱼ sin(2πjτ/M)]

    where M = 390 is the number of one-minute intervals in the regular
    NYSE session and τ ∈ {0, ..., 389} is the zero-based interval slot.

    With end-of-bar close timestamps:

        τ = 0   corresponds to 09:30–09:31, timestamped 09:31
        τ = 1   corresponds to 09:31–09:32, timestamped 09:32
        ...
        τ = 389 corresponds to 15:59–16:00, timestamped 16:00

    Because the input contains close prices starting at 09:31, the
    return for τ = 0 cannot be calculated without the 09:30 opening
    price. It is therefore a structural NaN. The FFF is fitted on the
    389 valid close-to-close return slots τ = 1, ..., 389.

    For each series independently, fitting averages squared returns
    across available sessions at each valid slot and fits a separate
    Fourier variance profile.

    Attributes:
        num_harmonics
            Number of Fourier harmonics K. The model has 1 + 2K free
            parameters.
        trading_minutes
            Length of the regular trading session in minutes. This
            remains 390 even though only 389 close-to-close returns
            are available.
        coefficients
            Fitted FFF coefficients for each series.
        vol_smile
            Estimated seasonal volatility for valid FFF slots
            1, ..., 389, with one column per fitted series.

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
        assert self._coefficients is not None
        return self._coefficients

    @property
    def vol_smile(self) -> pd.DataFrame:
        """Seasonal volatility indexed by valid FFF slot."""
        self._require_fitted()
        assert self._vol_smile is not None
        return self._vol_smile

    def _session_slots(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return zero-based return-bar slots: 09:31 -> 0, 16:00 -> 389.

        session_minute_position() measures elapsed minutes from 09:30,
        producing positions 1, ..., 390. FFF slots instead use
        zero-based return-bar positions 0, ..., 389.
        """
        raw_positions = pd.Series(
            np.asarray(session_minute_position(index)),
            index=index,
            name="session_minute_position",
        )
        slots = raw_positions.astype(np.int64) - 1
        slots.name = "fff_slot"

        invalid = (slots < 0) | (slots >= self.trading_minutes)
        if invalid.any():
            invalid_values = sorted(slots.loc[invalid].unique().tolist())
            raise ValueError(
                "Found FFF slots outside the expected range "
                f"0, ..., {self.trading_minutes - 1}: {invalid_values}"
            )

        return slots

    def fit(self, log_returns: pd.DataFrame) -> FFFDeseasonalizer:
        """Estimate one FFF profile per real return series.

        Slot 0, corresponding to the return timestamped 09:31, is
        structurally unavailable and excluded. The model is fitted on
        slots 1, ..., 389, corresponding to 09:32 through 16:00.
        """
        minute_slots = self._session_slots(log_returns.index)

        # Slot 0 is unavailable with close-only prices starting at 09:31.
        valid_mask = minute_slots.gt(0)
        valid_returns = log_returns.loc[valid_mask]
        valid_slots = minute_slots.loc[valid_mask]

        expected_slots = pd.RangeIndex(
            start=1,
            stop=self.trading_minutes,
            name="fff_slot",
        )

        var_profile = (
            valid_returns.pow(2).groupby(valid_slots, sort=True).mean().reindex(expected_slots)
        )

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
        """Remove series-specific intraday volatility seasonality.

        Slot 0 is preserved unchanged. It should be NaN because the
        corresponding 09:31 return cannot be computed from close-only
        prices starting at 09:31. All other structural NaNs are also
        preserved.
        """
        self._require_fitted()
        assert self._vol_smile is not None

        missing_columns = [
            column for column in log_returns.columns if column not in self._vol_smile.columns
        ]
        if missing_columns:
            raise ValueError(f"No fitted FFF profile is available for columns {missing_columns}.")

        minute_slots = self._session_slots(log_returns.index)

        valid_mask = minute_slots.gt(0)
        valid_returns = log_returns.loc[valid_mask]
        valid_slots = minute_slots.loc[valid_mask]

        sigma = self._vol_smile.loc[
            valid_slots.to_numpy(),
            log_returns.columns,
        ].to_numpy(dtype=float)

        transformed = log_returns.copy()

        transformed.loc[valid_mask, :] = valid_returns.to_numpy(dtype=float, copy=False) / sigma

        return transformed

    def fit_transform(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Fit on log_returns and return the transformed data."""
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
