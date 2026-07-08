"""Flexible Fourier Form (FFF) intraday volatility deseasonalizer.

Intraday equity returns exhibit a pronounced U-shaped variance profile
across the trading session — high at the open, low at midday, rising
again toward the close. This diurnal pattern is a nuisance for every
metric : without removal it dominates the marginal distribution
, inflates absolute-return ACF at session-periodic lags , and
contaminates the leverage curve .

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

    where τ ∈ {0, …, M−1} is the minute position within the session and
    M = 390 is the number of NYSE trading minutes. In practice τ = 1
    corresponds to 09:31 ET (the first return after the open); τ = 0 is
    never populated and τ = M (16:00, the session close) falls outside
    this range and is dropped by transform() — a known off-by-one, see
    todo.md, "FFF drops every session's close bar".

    Fitting pools squared returns across all available days and tickers
    to produce a single cross-sectional variance profile, then fits
    the Fourier model via nonlinear least squares.

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
            Estimated σ(τ) for τ = 0, …, M−1 indexed by integer
            minute slot. The per-minute seasonal volatility used to
            divide returns in ``transform``.

    Example::

        fff = FFFDeseasonalizer(num_harmonics=4)
        fff.fit(log_returns_real)

        deseas_real = fff.transform(log_returns_real)
        deseas_ail  = fff.transform(log_returns_ail)
    """

    def __init__(self, num_harmonics: int = 4, trading_minutes: int = 390) -> None:
        self.num_harmonics = num_harmonics
        self.trading_minutes = trading_minutes
        self._coefficients: np.ndarray | None = None
        self._vol_smile: pd.Series | None = None

    @property
    def is_fitted(self) -> bool:
        """True after ``fit`` has been called successfully."""
        return self._coefficients is not None

    @property
    def coefficients(self) -> np.ndarray:
        """Fitted FFF coefficients (c₀, a₁, b₁, …, aK, bK)."""
        self._require_fitted()
        return self._coefficients

    @property
    def vol_smile(self) -> pd.Series:
        """Estimated σ(τ) for each minute slot in the session."""
        self._require_fitted()
        return self._vol_smile

    def fit(self, log_returns: pd.DataFrame) -> FFFDeseasonalizer:
        """Estimate the intraday volatility profile from real log-returns.

        Pools squared returns across all days and tickers per minute
        slot to produce a single cross-sectional variance profile,
        then fits the Fourier model via nonlinear least squares.

        The input must carry a tz-aware DatetimeIndex (UTC or
        America/New_York) so that session-relative minute positions
        can be computed unambiguously.
        """
        minute_pos = session_minute_position(log_returns.index)

        avg_var = (log_returns**2).groupby(minute_pos).mean().mean(axis=1)

        valid = avg_var.loc[0 : self.trading_minutes - 1].dropna()
        m_arr = valid.index.values.astype(float)
        lv_tgt = np.log(valid.values + 1e-12)

        p0 = np.zeros(1 + 2 * self.num_harmonics)
        p0[0] = lv_tgt.mean()
        coefficients, _ = curve_fit(self._fourier_log_variance, m_arr, lv_tgt, p0=p0, maxfev=10_000)
        self._coefficients = coefficients

        all_slots = np.arange(self.trading_minutes, dtype=float)
        log_var_hat = self._fourier_log_variance(all_slots, *coefficients)
        self._vol_smile = pd.Series(
            np.exp(0.5 * log_var_hat), index=np.arange(self.trading_minutes)
        )
        return self

    def transform(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Divide each return by its minute slot's seasonal volatility.

        Structural NaN values (session boundaries, intraday gaps) pass
        through unchanged — dividing NaN by a scalar is still NaN, so
        the NaN structure of the input is exactly preserved in the
        output.
        """
        self._require_fitted()
        minute_pos = session_minute_position(log_returns.index)
        smile_map = dict(enumerate(self._vol_smile))
        sigma_vec = minute_pos.map(smile_map).to_numpy().reshape(-1, 1)
        return log_returns / sigma_vec

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
