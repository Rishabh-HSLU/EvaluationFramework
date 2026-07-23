"""Preprocessing pipeline for paired real–synthetic return data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import NUM_HARMONICS, SEASONALITY_CV_THRESHOLD, TRADING_MINUTES
from .fff import FFFDeseasonalizer
from .session_clock import overnight_masked_log_returns, session_minute_position


class PreprocessingPipeline:
    """Transform paired real and synthetic prices on a shared market clock.

    Overnight-masked log returns are computed for both datasets.
    Intraday seasonality is then detected independently in each dataset.

    When a dataset exhibits meaningful intraday seasonality, an FFF
    profile is fitted to that dataset and used to deseasonalize it.
    Flat datasets such as GBM and MSV are left unchanged to avoid
    injecting an artificial inverse seasonal pattern.

    Attributes:
        log_returns_real
            (T, N) overnight-masked real log returns before FFF.
        log_returns_synthetic
            Overnight-masked synthetic log returns before FFF.
        fff_real
            FFF fitted to the real returns, or None if no meaningful
            seasonality was detected.
        fff_synthetic
            FFF fitted to the synthetic returns, or None if no meaningful
            seasonality was detected.
        deseas_real
            Final real returns after conditional deseasonalization.
        deseas_synthetic
            Final synthetic returns after conditional deseasonalization.
    """

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

    def _has_seasonality(self, log_returns: pd.DataFrame) -> bool:
        """Detect whether returns have meaningful intraday seasonality.

        The statistic is the coefficient of variation of the pooled
        per-slot variance profile.

        With close prices beginning at 09:31, the return timestamped
        09:31 is structurally unavailable. Seasonality detection
        therefore uses the 389 valid returns timestamped 09:32–16:00.
        """
        session_positions = pd.Series(
            np.asarray(session_minute_position(log_returns.index)),
            index=log_returns.index,
            name="session_minute_position",
        )

        valid_mask = session_positions.gt(1)
        valid_returns = log_returns.loc[valid_mask]
        valid_positions = session_positions.loc[valid_mask]

        expected_positions = pd.RangeIndex(
            start=2,
            stop=self.trading_minutes + 1,
            name="session_minute_position",
        )

        profile = (
            valid_returns.pow(2)
            .groupby(valid_positions, sort=True)
            .mean()
            .mean(axis=1)
            .reindex(expected_positions)
        )

        if profile.isna().any():
            missing_positions = profile.index[profile.isna()].tolist()
            raise ValueError(
                f"Missing pooled variance estimates for session positions {missing_positions}."
            )

        mean_variance = float(profile.mean())
        cv = float(profile.std() / mean_variance)
        return bool(cv > SEASONALITY_CV_THRESHOLD)

    def run(
        self, real_prices: pd.DataFrame, synthetic_prices: pd.DataFrame
    ) -> PreprocessingPipeline:
        """Compute returns and conditionally deseasonalize each dataset."""
        self.log_returns_real = overnight_masked_log_returns(real_prices)
        self.log_returns_synthetic = overnight_masked_log_returns(synthetic_prices)

        self.deseas_real, self.fff_real = self._maybe_deseasonalize(self.log_returns_real)
        self.deseas_synthetic, self.fff_synthetic = self._maybe_deseasonalize(
            self.log_returns_synthetic
        )
        return self

    def _maybe_deseasonalize(
        self, log_returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, FFFDeseasonalizer | None]:
        """Deseasonalize only when meaningful seasonality is detected.

        Flat datasets are returned unchanged. This avoids injecting an
        inverse seasonal profile into generators such as GBM and MSV.
        """
        if not self._has_seasonality(log_returns):
            return log_returns, None
        fff = FFFDeseasonalizer(
            num_harmonics=self.num_harmonics, trading_minutes=self.trading_minutes
        )
        deseas = fff.fit_transform(log_returns)
        return deseas, fff
