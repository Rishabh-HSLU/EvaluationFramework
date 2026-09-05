"""Tests for the standalone SBBTS panel bridge.

The script has two directions — curated real panel to training tensor, and
generated output back to a price panel — and these pin the four properties
that a mistake in either direction would silently break: the standardization
round-trips, the tensor carries SBBTS's declared shape, the early-close
sessions are excluded by bar count rather than by date, and an early-close
session is reconstructed by plain truncation.

Fixtures are synthetic panels built here, never the real curated parquet, so
the tests run without ``data/curated`` present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.sbbts_panel import (
    EARLY_CLOSE_BARS,
    PATH_FEATURES,
    SESSION_BARS,
    assemble_price_panel,
    build_training_tensor,
    invert_standardization,
    per_ticker_scales,
    split_sessions,
    standardize_returns,
)

TICKERS = ["AAA", "BBB", "CCC"]


def _session_index(day: str, n_bars: int) -> pd.DatetimeIndex:
    """One session's bars, 09:31 ET onward, returned on the UTC clock."""
    start = pd.Timestamp(f"{day} 09:31", tz="America/New_York")
    return pd.date_range(start, periods=n_bars, freq="min").tz_convert("UTC")


def _panel(session_lengths: dict[str, int], seed: int = 0) -> pd.DataFrame:
    """A synthetic curated-style panel: UTC index, string columns, float64."""
    parts = [_session_index(day, bars) for day, bars in session_lengths.items()]
    index = parts[0].append(parts[1:])
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.001, size=(len(index), len(TICKERS)))
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=index, columns=TICKERS)


def _default_lengths() -> dict[str, int]:
    """Four full sessions plus exactly two early closes, as in the real window."""
    return {
        "2026-01-05": SESSION_BARS,
        "2026-01-06": SESSION_BARS,
        "2026-01-07": EARLY_CLOSE_BARS,
        "2026-01-08": SESSION_BARS,
        "2026-01-09": EARLY_CLOSE_BARS,
        "2026-01-12": SESSION_BARS,
    }


def test_standardization_round_trips() -> None:
    """invert_standardization undoes standardize_returns to floating tolerance."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.002, size=(500, len(TICKERS)))
    scales = np.array([0.0013, 0.0009, 0.0021])

    recovered = invert_standardization(standardize_returns(returns, scales), scales)

    np.testing.assert_allclose(recovered, returns, rtol=1e-12, atol=1e-15)


def test_training_tensor_has_the_sbbts_shape() -> None:
    """The tensor is (M, N + 1, d) with N = 390 and d = 1, in float32."""
    prices = _panel(_default_lengths())

    tensor, scales, index = build_training_tensor(prices)

    n_full_sessions = sum(1 for bars in _default_lengths().values() if bars == SESSION_BARS)
    assert tensor.shape == (len(TICKERS) * n_full_sessions, SESSION_BARS + 1, PATH_FEATURES)
    assert tensor.shape[1] == SESSION_BARS + 1
    assert tensor.shape[2] == PATH_FEATURES == 1
    assert tensor.dtype == torch.float32
    assert len(index) == tensor.shape[0]
    assert set(scales) == set(TICKERS)


def test_exactly_two_sessions_are_excluded_by_bar_count() -> None:
    """Early closes are found by length, excluded from scales, and counted."""
    prices = _panel(_default_lengths())

    full_sessions, early_sessions = split_sessions(prices.index)

    assert len(early_sessions) == 2
    assert len(full_sessions) == 4
    # Detected by bar count, not by date: every excluded session is short.
    assert all(bars == EARLY_CLOSE_BARS for bars in (2 * [EARLY_CLOSE_BARS]))
    assert set(early_sessions).isdisjoint(set(full_sessions))

    # The scale pool sees only full-length sessions.
    returns_all = np.log(prices).diff()
    pooled_rows = returns_all.index.normalize().isin(full_sessions).sum()
    assert pooled_rows == 4 * SESSION_BARS

    scales = per_ticker_scales(prices, full_sessions)
    assert scales.shape == (len(TICKERS),)
    assert np.isfinite(scales).all() and (scales > 0).all()


@pytest.mark.parametrize("n_early", [0, 1, 3])
def test_a_different_early_close_count_raises(n_early: int) -> None:
    """Any count other than two is a changed window and must not pass silently."""
    lengths = {f"2026-02-{2 + day:02d}": SESSION_BARS for day in range(5)}
    for day in range(n_early):
        lengths[f"2026-03-{2 + day:02d}"] = EARLY_CLOSE_BARS

    prices = _panel(lengths)

    with pytest.raises(ValueError, match="Expected exactly 2 early-close sessions"):
        split_sessions(prices.index)


def test_early_close_session_is_a_plain_truncation() -> None:
    """The 210-bar output is the first 210 bars of the full 390-bar path."""
    prices = _panel(_default_lengths())
    _, scales, _ = build_training_tensor(prices)

    sessions = pd.DatetimeIndex(sorted(set(prices.index.normalize())))
    counts = pd.Series(1, index=prices.index.normalize()).groupby(level=0).size()
    early_session = sessions[counts.reindex(sessions).to_numpy() == EARLY_CLOSE_BARS][0]

    # A distinctive, strictly non-constant path so truncation cannot be
    # confused with resampling, interpolation or padding.
    rng = np.random.default_rng(11)
    path = np.cumsum(rng.normal(0.0, 0.05, size=SESSION_BARS)).astype(np.float32)
    generated = torch.from_numpy(path.reshape(1, SESSION_BARS, PATH_FEATURES))

    ticker = TICKERS[0]
    panel = assemble_price_panel(
        generated=generated,
        path_tickers=[ticker],
        path_sessions=[early_session],
        scales=scales,
        real_prices=prices,
    )

    produced = panel.loc[panel.index.normalize() == early_session, ticker].to_numpy()
    assert produced.shape == (EARLY_CLOSE_BARS,)

    session_rows = prices.loc[prices.index.normalize() == early_session, ticker]
    anchor = float(session_rows.iloc[0])
    full_reconstruction = anchor * np.exp(invert_standardization(path, scales[ticker]))

    # Identical to the first 210 entries of the untruncated reconstruction.
    np.testing.assert_allclose(produced, full_reconstruction[:EARLY_CLOSE_BARS], rtol=1e-12)
    assert not np.allclose(produced, full_reconstruction[-EARLY_CLOSE_BARS:])
