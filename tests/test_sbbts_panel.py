"""Tests for the standalone SBBTS panel bridge.

The script has two directions — curated real panel to variable-length
training examples, and generated output back to a price panel — and these pin
the properties a mistake in either direction would silently break: the
standardization round-trips, examples are well-formed and free of NaNs,
minute position 390 never reaches the output, gaps count skipped minutes
honestly, the per-ticker scale still follows M1's convention, early closes
stay out of training, and an early-close session is reconstructed by plain
truncation.

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
    GAP_DTYPE,
    MIN_SURVIVING_BARS,
    PATH_FEATURES,
    SESSION_BARS,
    VALUE_DTYPE,
    WORKING_BARS,
    assemble_price_panel,
    build_training_examples,
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


def _session_rows(prices: pd.DataFrame, session: str) -> np.ndarray:
    """Positional row numbers belonging to one session date."""
    # Sessions are keyed by the UTC-normalized index, and a 09:31-16:00 ET
    # session lies wholly inside one UTC date.
    return np.flatnonzero(prices.index.normalize() == pd.Timestamp(session, tz="UTC"))


def _by_key(examples: list) -> dict[tuple[str, pd.Timestamp], object]:
    return {(example.ticker, example.session): example for example in examples}


def test_standardization_round_trips() -> None:
    """invert_standardization undoes standardize_returns to floating tolerance."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.002, size=(500, len(TICKERS)))
    scales = np.array([0.0013, 0.0009, 0.0021])

    recovered = invert_standardization(standardize_returns(returns, scales), scales)

    np.testing.assert_allclose(recovered, returns, rtol=1e-12, atol=1e-15)


def test_examples_are_variable_length_and_well_formed() -> None:
    """Dense input gives one WORKING_BARS-long example per ticker-session."""
    prices = _panel(_default_lengths())

    examples, scales, dropped = build_training_examples(prices)

    n_full = sum(1 for bars in _default_lengths().values() if bars == SESSION_BARS)
    assert len(examples) == len(TICKERS) * n_full
    assert dropped == 0
    assert set(scales) == set(TICKERS)

    for example in examples:
        assert example.values.ndim == example.gaps.ndim == 1
        assert example.values.shape == example.gaps.shape
        assert example.values.dtype == VALUE_DTYPE
        assert example.gaps.dtype == GAP_DTYPE
        assert example.values[0] == 0.0
        # Nothing is missing here, so every bar in the working range survives.
        assert len(example) == WORKING_BARS
        assert np.array_equal(example.gaps, np.ones(WORKING_BARS, dtype=GAP_DTYPE))


def test_lengths_vary_once_bars_are_missing() -> None:
    """Dropping bars shortens the affected example and leaves others alone."""
    prices = _panel(_default_lengths())
    rows = _session_rows(prices, "2026-01-05")
    prices.iloc[rows[10:40], 0] = np.nan  # 30 bars of AAA on one session

    examples = _by_key(build_training_examples(prices)[0])
    session = prices.index[rows[0]].normalize()

    assert len(examples[("AAA", session)]) == WORKING_BARS - 30
    assert len(examples[("BBB", session)]) == WORKING_BARS
    lengths = {len(example) for example in examples.values()}
    assert len(lengths) > 1, "expected variable-length output"


def test_no_record_contains_a_nan() -> None:
    """Missing bars are removed, never carried through as NaN."""
    rng = np.random.default_rng(3)
    prices = _panel(_default_lengths())
    holes = rng.random(prices.shape) < 0.25
    prices = prices.mask(holes)

    examples, _, _ = build_training_examples(prices)

    assert examples
    for example in examples:
        assert np.isfinite(example.values).all()
        assert np.isfinite(example.gaps).all()


def test_position_390_never_reaches_the_output() -> None:
    """The 16:00 bar is dropped even when it carries a price."""
    prices = _panel(_default_lengths())
    baseline, baseline_scales, _ = build_training_examples(prices)

    # Corrupt every full-length session's last bar beyond recognition. If it
    # were consumed anywhere, values or scales would move.
    perturbed = prices.copy()
    for day, bars in _default_lengths().items():
        if bars != SESSION_BARS:
            continue
        perturbed.iloc[_session_rows(prices, day)[-1], :] *= 5.0

    examples, scales, _ = build_training_examples(perturbed)

    assert len(examples) == len(baseline)
    for produced, expected in zip(examples, baseline, strict=True):
        assert produced.ticker == expected.ticker
        assert produced.session == expected.session
        np.testing.assert_array_equal(produced.values, expected.values)
        np.testing.assert_array_equal(produced.gaps, expected.gaps)
    assert scales == baseline_scales


def test_gaps_count_skipped_minutes() -> None:
    """A gap of n means n - 1 bars were skipped; the first gap counts from the open."""
    prices = _panel(_default_lengths())
    rows = _session_rows(prices, "2026-01-05")
    session = prices.index[rows[0]].normalize()

    # AAA: minutes 5, 6, 7 removed -> the survivor at minute 8 has gap 4.
    prices.iloc[rows[4:7], 0] = np.nan
    # BBB: minutes 1 and 2 removed -> the first survivor is minute 3, gap 3.
    prices.iloc[rows[0:2], 1] = np.nan

    examples = _by_key(build_training_examples(prices)[0])

    aaa = examples[("AAA", session)]
    assert len(aaa) == WORKING_BARS - 3
    assert aaa.gaps[0] == 1
    np.testing.assert_array_equal(aaa.gaps[:4], [1, 1, 1, 1])
    assert aaa.gaps[4] == 4, "minute 8 follows minute 4"
    np.testing.assert_array_equal(aaa.gaps[5:], np.ones(len(aaa) - 5, dtype=GAP_DTYPE))

    bbb = examples[("BBB", session)]
    assert len(bbb) == WORKING_BARS - 2
    assert bbb.gaps[0] == 3, "first survivor sits at minute position 3"
    np.testing.assert_array_equal(bbb.gaps[1:], np.ones(len(bbb) - 1, dtype=GAP_DTYPE))

    for example in examples.values():
        assert (example.gaps >= 1).all()
        # Gaps span the working range exactly: the last survivor's position is
        # the cumulative sum of every gap before and including it.
        assert example.gaps.sum() <= WORKING_BARS


def test_pairs_below_two_surviving_bars_are_dropped_and_counted() -> None:
    """A session with one surviving bar yields no return, so it is skipped."""
    prices = _panel(_default_lengths())
    rows = _session_rows(prices, "2026-01-05")
    prices.iloc[rows[1:], 0] = np.nan  # AAA keeps exactly one bar that session

    examples, _, dropped = build_training_examples(prices)

    assert dropped == 1
    assert MIN_SURVIVING_BARS == 2
    session = prices.index[rows[0]].normalize()
    assert ("AAA", session) not in _by_key(examples)


def test_scale_follows_the_m1_convention() -> None:
    """Scales equal nanstd(ddof=1) over the pooled working-range returns."""
    prices = _panel(_default_lengths())
    full_sessions, _ = split_sessions(prices.index)

    scales = per_ticker_scales(prices, full_sessions)

    pooled = []
    for day, bars in _default_lengths().items():
        if bars != SESSION_BARS:
            continue
        rows = _session_rows(prices, day)[:WORKING_BARS]
        block = np.log(prices.iloc[rows].to_numpy(dtype=float))
        pooled.append(np.diff(block, axis=0))
    expected = np.nanstd(np.concatenate(pooled, axis=0), axis=0, ddof=1)

    np.testing.assert_allclose(scales, expected, rtol=1e-12)
    assert (scales > 0).all() and np.isfinite(scales).all()


def test_a_degenerate_column_is_named_rather_than_dropped() -> None:
    """M1 refuses to standardize a zero-variance column; so does this."""
    prices = _panel(_default_lengths())
    prices["BBB"] = 50.0
    full_sessions, _ = split_sessions(prices.index)

    with pytest.raises(ValueError, match="BBB"):
        per_ticker_scales(prices, full_sessions)


def test_exactly_two_sessions_are_excluded_by_bar_count() -> None:
    """Early closes are found by length and kept out of the scale pool."""
    prices = _panel(_default_lengths())

    full_sessions, early_sessions = split_sessions(prices.index)

    assert len(early_sessions) == 2
    assert len(full_sessions) == 4
    assert set(early_sessions).isdisjoint(set(full_sessions))

    scales = per_ticker_scales(prices, full_sessions)
    assert scales.shape == (len(TICKERS),)
    assert np.isfinite(scales).all() and (scales > 0).all()


def test_early_close_sessions_are_absent_from_examples() -> None:
    """No training example may come from a short session."""
    prices = _panel(_default_lengths())
    _, early_sessions = split_sessions(prices.index)

    examples, _, _ = build_training_examples(prices)

    produced = {example.session for example in examples}
    assert produced.isdisjoint(set(early_sessions))
    assert len(produced) == 4


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
    _, scales, _ = build_training_examples(prices)

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

    np.testing.assert_allclose(produced, full_reconstruction[:EARLY_CLOSE_BARS], rtol=1e-12)
    assert not np.allclose(produced, full_reconstruction[-EARLY_CLOSE_BARS:])
