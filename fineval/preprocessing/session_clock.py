"""Shared session-clock utilities.

`session_minute_position` and `overnight_masked_log_returns` were
independently duplicated across `FFFDeseasonalizer`, `PreprocessingPipeline`,
and `scripts/baseline_generation.py`. This module is their single source
of truth.

Known issue (see todo.md, "FFF drops every session's close bar"):
SESSION_OFFSET = 570 maps the first bar of a session (09:31 ET) to
minute position 1, not 0, so a slot range of [0, TRADING_MINUTES - 1]
excludes minute position TRADING_MINUTES (16:00, the session close).
Preserved as-is here — this refactor deduplicates the three prior
copies without changing any numeric output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_OFFSET = 570  # 09:30 ET in minutes since midnight (9*60 + 30)


def session_minute_position(index: pd.DatetimeIndex) -> pd.Index:
    """Map a tz-aware DatetimeIndex to session-relative minute slots.

    Converts to America/New_York, then computes τ = hour*60 + minute − 570.
    Off-hours timestamps produce values with no corresponding session
    slot; downstream lookups against those values propagate as NaN.
    """
    ny = index.tz_convert("America/New_York")
    return ny.hour * 60 + ny.minute - SESSION_OFFSET


def overnight_masked_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from wide-format prices, with the overnight bar masked.

    The first bar of each session (09:31 ET, minute position 1) is a
    diff against the prior session's last bar rather than a genuine
    1-minute return, so it is masked to NaN — along with the very
    first row of the whole series, which has no prior bar at all.
    """
    log_returns = np.log(prices).diff()
    session_start = session_minute_position(log_returns.index) == 1
    log_returns.loc[session_start] = np.nan
    log_returns.iloc[0] = np.nan
    return log_returns
