"""Preprocessing pipeline for paired real–synthetic return data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import NUM_HARMONICS, TRADING_MINUTES
from .fff import FFFDeseasonalizer


class PreprocessingPipeline:
    """Transforms paired (real, synthetic) prices into deseasonalised
    log-returns on a shared NYSE market clock.

    Attributes:
        log_returns_real
            (T, N) overnight-masked log returns for real, before FFF.
            Set by run(). Diagnostic access to the pre-FFF state.
        log_returns_synthetic
            Same for synthetic.
        fff_real
            Fitted FFFDeseasonalizer for real (None if no smile).
        fff_synthetic
            Fitted FFFDeseasonalizer for synthetic (None if no smile).
        deseas_real
            Final deseasonalised real returns.
        deseas_synthetic
            Final deseasonalised synthetic returns.
    """

    SESSION_OFFSET = 570  # 09:30 ET = 9*60 + 30

    def __init__(
        self, num_harmonics: int = NUM_HARMONICS, trading_minutes: int = TRADING_MINUTES
    ) -> None:
        self.num_harmonics = num_harmonics
        self.trading_minutes = trading_minutes
        self.log_returns_real: pd.DataFrame | None = None
        self.log_returns_synthetic: pd.DataFrame | None = None
        self.fff_real: FFFDeseasonalizer | None = None
        self.fff_synthetic: FFFDeseasonalizer | None = None
        self.deseas_real: pd.DataFrame | None = None
        self.deseas_synthetic: pd.DataFrame | None = None

    def _compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert prices to log-returns with overnight masking."""
        log_returns = np.log(prices).diff()
        ny_index = log_returns.index.tz_convert("America/New_York")
        session_start = (ny_index.hour == 9) & (ny_index.minute == 31)
        log_returns.loc[session_start] = np.nan
        log_returns.iloc[0] = np.nan
        return log_returns

    def _has_seasonality(self, log_returns: pd.DataFrame, threshold: float = 0.3) -> bool:
        """Detect whether a return series has a genuine intraday smile.

        Uses the coefficient of variation of the pooled per-minute
        variance profile: a flat (non-seasonal) series has near-zero
        CV, while real equities show a large CV driven by the open
        spike. Threshold of 0.3 separates GBM-like flat baselines
        (CV ~0.05) from real/AIL-like seasonal series (CV ~0.6+).
        """
        ny_index = log_returns.index.tz_convert("America/New_York")
        minute_pos = ny_index.hour * 60 + ny_index.minute - self.SESSION_OFFSET
        profile = (log_returns**2).groupby(minute_pos).mean().mean(axis=1)
        profile = profile.loc[0 : self.trading_minutes - 1].dropna()
        cv = profile.std() / profile.mean()
        return cv > threshold

    def run(
        self, real_prices: pd.DataFrame, synthetic_prices: pd.DataFrame
    ) -> PreprocessingPipeline:
        """Run prices → log returns → conditional per-series FFF."""
        self.log_returns_real = self._compute_log_returns(real_prices)
        self.log_returns_synthetic = self._compute_log_returns(synthetic_prices)

        self.deseas_real, self.fff_real = self._maybe_deseasonalize(self.log_returns_real)
        self.deseas_synthetic, self.fff_synthetic = self._maybe_deseasonalize(
            self.log_returns_synthetic
        )
        return self

    def _maybe_deseasonalize(
        self, log_returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, FFFDeseasonalizer | None]:
        """Apply FFF only if the series shows a genuine intraday smile.

        A flat series (e.g. GBM) is returned unchanged rather than
        having a smile injected by dividing by someone else's profile.
        """
        if not self._has_seasonality(log_returns):
            return log_returns, None
        fff = FFFDeseasonalizer(
            num_harmonics=self.num_harmonics, trading_minutes=self.trading_minutes
        )
        deseas = fff.fit_transform(log_returns)
        return deseas, fff
