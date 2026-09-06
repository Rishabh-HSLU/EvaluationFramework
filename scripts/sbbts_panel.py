"""
Bridge the curated real panel and the SBBTS generator, in both directions.

Direction 1 (``build_training_examples``) turns the curated real prices into
the variable-length sequences SBBTS trains on. Direction 2
(``assemble_price_panel``) turns SBBTS's raw output back into a wide price
panel shaped exactly like the curated real file, so it can be written to
parquet and read back through ``SBBTSBaselineLoader``
(fineval/data/curate.py).

Nothing here is wired into the benchmark. ``load_default_datasets`` and its
call sites are untouched; this module is standalone.

Conventions reused, not reinvented
----------------------------------
- **Per-ticker scale.** One volatility scalar per ticker, computed as
  ``np.nanstd(returns, axis=0, ddof=1)`` over that ticker's own pooled
  returns. This is M1's convention (fineval/metrics/tail_weighted_marginal.py),
  including its refusal to silently drop a degenerate column: a zero or
  non-finite scale names the offending tickers and raises. The pool is
  whatever returns survive the gap logic below.
- **Output panel format.** Wide DataFrame, curated market-clock index,
  same column order, float64, real's NaN mask imposed — identical to what
  ``CurationPipeline._save_datasets`` writes.

Session geometry and the gap representation
-------------------------------------------
Minute position ``SESSION_BARS`` (390, the 16:00 close) is dropped before
anything else, unconditionally, whether or not it carries a price in a given
session. It is missing for roughly half of all (ticker, session) cells for
structural reasons unrelated to ordinary illiquidity, so it is not treated as
data. The working range is minutes 1..389, ``WORKING_BARS``.

Within that range a (ticker, session) example keeps **whatever bars actually
exist**, in order, and drops the rest — no filling, no flagging. The example
is the cumulative standardized log-return path over the surviving bars,
anchored at 0 on the first of them, so ``values[0]`` is always 0.0 and a
return spans however many minutes separate two consecutive survivors.

``gaps`` runs alongside ``values``, one entry per point, holding the minutes
elapsed since the previous surviving bar. A gap of 1 means consecutive
minutes; a gap of 3 means two bars were skipped. ``gaps[0]`` is the first
survivor's own minute position — the distance from the session open, which is
where the path is anchored — so every gap is at least 1.

Examples vary in length, so the result is a list of ``SessionExample``
records rather than one stacked tensor. Batching and padding are deliberately
not designed here; that belongs with the SBBTS training code.

A pair with fewer than ``MIN_SURVIVING_BARS`` surviving bars yields no return
and is dropped; ``build_training_examples`` reports how many.

The evaluation window holds two early-close sessions of 210 bars (1:00pm ET).
They are detected by bar count, never by date, and:

- excluded entirely from the per-ticker scale pool and from training;
- reconstructed in direction 2 by generating a normal 390-bar path and
  keeping its first 210 bars — a plain truncation, no resampling.

``split_sessions`` raises unless exactly ``N_EARLY_CLOSE_SESSIONS`` short
sessions are found, each of exactly ``EARLY_CLOSE_BARS`` bars, so a change in
the curated window cannot pass through unnoticed.

Run from the repository root:

    uv run python -m scripts.sbbts_panel
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from fineval.benchmark.config import CURATED_DIR, ROOT_DIR
from fineval.config import TRADING_MINUTES

REAL_PATH = CURATED_DIR / "real_prices.parquet"

#: Training artifacts live outside data/curated/, which stays reserved for
#: CurationPipeline output. The whole directory is gitignored.
SBBTS_DIR = ROOT_DIR / "data" / "sbbts"
EXAMPLES_OUTPUT_PATH = SBBTS_DIR / "sbbts_training_examples.npz"
SCALES_OUTPUT_PATH = SBBTS_DIR / "sbbts_ticker_scales.npz"

SESSION_BARS = TRADING_MINUTES  # 390; bars in a regular NYSE session
WORKING_BARS = SESSION_BARS - 1  # 389; minute position 390 is never used
EARLY_CLOSE_BARS = 210  # bars in a 1:00pm ET early close
N_EARLY_CLOSE_SESSIONS = 2  # early closes in the curated window
MIN_SURVIVING_BARS = 2  # below this a session yields no return at all
PATH_FEATURES = 1  # d in SBBTS's (M_simu, N, d) output contract
VALUE_DTYPE = np.float32
GAP_DTYPE = np.int32


@dataclass(frozen=True)
class SessionExample:
    """One variable-length training example for a single (ticker, session).

    Attributes:
        ticker: Column of the curated panel this path came from.
        session: Session date it came from.
        values: (K,) float32 cumulative standardized log-return path over the
            K surviving bars, anchored so ``values[0] == 0.0``.
        gaps: (K,) int32 minutes since the previous surviving bar. Always at
            least 1; ``gaps[0]`` is the first survivor's minute position.
    """

    ticker: str
    session: pd.Timestamp
    values: np.ndarray
    gaps: np.ndarray

    def __len__(self) -> int:
        return int(self.values.shape[0])


def split_sessions(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Partition a market clock into full-length and early-close sessions.

    Sessions are identified by bar count, not by date. The curated window is
    expected to hold exactly ``N_EARLY_CLOSE_SESSIONS`` short sessions of
    exactly ``EARLY_CLOSE_BARS`` bars each; anything else raises, so a
    re-curated window with different holiday coverage cannot silently change
    what gets excluded.

    Args:
        index: Market-clock DatetimeIndex of the curated panel.

    Returns:
        ``(full_sessions, early_close_sessions)``, both DatetimeIndex of
        session dates.

    Raises:
        ValueError: If the number of short sessions is not
            ``N_EARLY_CLOSE_SESSIONS``, or a short session does not have
            exactly ``EARLY_CLOSE_BARS`` bars.
    """
    sessions = index.normalize()
    counts = pd.Series(1, index=sessions).groupby(level=0).size()

    full = pd.DatetimeIndex(counts.index[counts == SESSION_BARS])
    early = pd.DatetimeIndex(counts.index[counts != SESSION_BARS])

    if len(early) != N_EARLY_CLOSE_SESSIONS:
        found = {str(day.date()): int(counts.loc[day]) for day in early}
        raise ValueError(
            f"Expected exactly {N_EARLY_CLOSE_SESSIONS} early-close sessions "
            f"({EARLY_CLOSE_BARS} bars each), found {len(early)}: {found}. "
            "The curated window changed; re-check the session geometry before "
            "building an SBBTS tensor from it."
        )

    wrong_length = {
        str(day.date()): int(counts.loc[day])
        for day in early
        if int(counts.loc[day]) != EARLY_CLOSE_BARS
    }
    if wrong_length:
        raise ValueError(
            f"Early-close sessions must have exactly {EARLY_CLOSE_BARS} bars, got {wrong_length}."
        )

    return full, early


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down axis 0, leaving a leading NaN run untouched."""
    positions = np.where(np.isfinite(values), np.arange(values.shape[0])[:, None], 0)
    np.maximum.accumulate(positions, axis=0, out=positions)
    return np.take_along_axis(values, positions, axis=0)


def working_range_blocks(
    prices: pd.DataFrame,
    full_sessions: pd.DatetimeIndex,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per full-length session, the surviving-bar returns, gaps and presence.

    Minute position ``SESSION_BARS`` is dropped first, so every block covers
    minutes 1..``WORKING_BARS``. Within a block, a return at a surviving bar
    is the log difference against the *previous surviving bar*, however many
    minutes back that is; the first survivor of a session has no return and
    is NaN. Non-surviving rows are NaN throughout.

    Args:
        prices: Curated wide-format prices (T, N).
        full_sessions: Full-length session dates, from ``split_sessions``.

    Returns:
        One ``(returns, gaps, present)`` triple per session, in
        ``full_sessions`` order, each of shape ``(WORKING_BARS, N)``.

    Raises:
        ValueError: If the selected rows are not exactly one full-length
            session per entry of ``full_sessions``.
    """
    in_full = prices.index.normalize().isin(full_sessions)
    selected = prices.loc[in_full]

    expected = len(full_sessions) * SESSION_BARS
    if selected.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} full-session rows ({len(full_sessions)} x "
            f"{SESSION_BARS}), got {selected.shape[0]}."
        )

    cube = selected.to_numpy(dtype=float).reshape(len(full_sessions), SESSION_BARS, -1)
    # Position 390 goes before anything else looks at the data.
    cube = cube[:, :WORKING_BARS, :]

    bar_index = np.arange(WORKING_BARS)[:, None]
    blocks = []
    for block in cube:
        present = np.isfinite(block)

        filled = _forward_fill(np.log(block))
        returns = np.full_like(filled, np.nan)
        returns[1:] = filled[1:] - filled[:-1]
        # A non-surviving row carries no observation, and the session's first
        # survivor has no predecessor to difference against.
        returns[~present] = np.nan

        # Index of the most recent surviving bar at or before each row, -1
        # before the first one, so the leading gap counts from the open.
        last_seen = np.where(present, bar_index, -1)
        np.maximum.accumulate(last_seen, axis=0, out=last_seen)
        previous = np.empty_like(last_seen)
        previous[0] = -1
        previous[1:] = last_seen[:-1]
        gaps = (bar_index - previous).astype(GAP_DTYPE)

        blocks.append((returns, gaps, present))
    return blocks


def _scales_from_returns(returns: np.ndarray, columns: pd.Index) -> np.ndarray:
    """M1's per-column standard deviation, with degenerate columns named."""
    observed = np.where(np.isfinite(returns), returns, np.nan)
    scales = np.nanstd(observed, axis=0, ddof=1)

    degenerate = ~np.isfinite(scales) | (scales <= 0.0)
    if degenerate.any():
        raise ValueError(
            "Cannot standardize columns with zero or non-finite standard "
            f"deviation: {columns[degenerate].tolist()}"
        )
    return scales


def per_ticker_scales(prices: pd.DataFrame, full_sessions: pd.DatetimeIndex) -> np.ndarray:
    """One volatility scalar per ticker, pooled over its surviving returns.

    Reuses M1's standardization convention exactly: ``np.nanstd`` with
    ``ddof=1`` down each column, with a degenerate column named rather than
    dropped. The pool is every return the gap logic produces — one per
    surviving bar except each session's first — over full-length sessions
    only, on minutes 1..``WORKING_BARS``.

    Args:
        prices: Curated wide-format prices (T, N).
        full_sessions: Session dates to pool over, from ``split_sessions``.

    Returns:
        Array of shape (N,), one positive finite scale per ticker, ordered
        like ``prices.columns``.

    Raises:
        ValueError: If any ticker's scale is zero or non-finite.
    """
    blocks = working_range_blocks(prices, full_sessions)
    pooled = np.concatenate([returns for returns, _, _ in blocks], axis=0)
    return _scales_from_returns(pooled, prices.columns)


def standardize_returns(returns: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Divide returns by their per-ticker scale.

    Inverse of ``invert_standardization``. ``scales`` must broadcast against
    ``returns``: a scalar for one ticker's path, or one entry per column for
    a wide return block.
    """
    return np.asarray(returns, dtype=float) / np.asarray(scales, dtype=float)


def invert_standardization(standardized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Multiply standardized values back by their per-ticker scale.

    Exact inverse of ``standardize_returns`` up to floating-point round-off.
    """
    return np.asarray(standardized, dtype=float) * np.asarray(scales, dtype=float)


def build_training_examples(
    prices: pd.DataFrame,
) -> tuple[list[SessionExample], dict[str, float], int]:
    """Build SBBTS's variable-length training examples from the real panel.

    One example per (ticker, full-length session) that has at least
    ``MIN_SURVIVING_BARS`` surviving bars on minutes 1..``WORKING_BARS``.
    Missing bars are dropped, not filled; the elapsed minutes they represent
    are carried in the example's ``gaps`` instead. Early-close sessions never
    enter. Examples are ordered ticker-major.

    Args:
        prices: Curated wide-format real prices (T, N).

    Returns:
        ``(examples, scales, dropped)`` where ``examples`` is the list of
        ``SessionExample`` records, ``scales`` maps ticker to the scalar used
        to standardize it, and ``dropped`` counts the (ticker, session) pairs
        skipped for having fewer than ``MIN_SURVIVING_BARS`` surviving bars.
    """
    full_sessions, _ = split_sessions(prices.index)
    blocks = working_range_blocks(prices, full_sessions)

    pooled = np.concatenate([returns for returns, _, _ in blocks], axis=0)
    scale_values = _scales_from_returns(pooled, prices.columns)
    scales = dict(zip(prices.columns, scale_values.tolist(), strict=True))
    del pooled

    tickers = list(prices.columns)
    examples: list[SessionExample] = []
    dropped = 0

    for column, ticker in enumerate(tickers):
        scale = scale_values[column]
        for session, (returns, gaps, present) in zip(full_sessions, blocks, strict=True):
            surviving = np.flatnonzero(present[:, column])
            if surviving.size < MIN_SURVIVING_BARS:
                dropped += 1
                continue

            # The first survivor anchors the path, so its (absent) return is
            # not consumed; every later survivor contributes one increment.
            increments = standardize_returns(returns[surviving[1:], column], scale)
            values = np.empty(surviving.size, dtype=VALUE_DTYPE)
            values[0] = 0.0
            values[1:] = np.cumsum(increments)

            examples.append(
                SessionExample(
                    ticker=ticker,
                    session=session,
                    values=values,
                    gaps=gaps[surviving, column].astype(GAP_DTYPE),
                )
            )

    return examples, scales, dropped


def assemble_price_panel(
    generated: torch.Tensor | np.ndarray,
    path_tickers: list[str],
    path_sessions: list[pd.Timestamp],
    scales: dict[str, float],
    real_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Reassemble SBBTS output into a panel shaped like the curated real file.

    Each generated path is a standardized cumulative log-return path over one
    session, ``(M_simu, SESSION_BARS, PATH_FEATURES)``. It is rescaled by its
    ticker's own scalar, exponentiated onto that (ticker, session)'s first
    observed real price, and written into the real clock's rows.

    Early-close sessions take the **first ``EARLY_CLOSE_BARS`` bars** of the
    full 390-bar generated path. That is a plain truncation: no resampling,
    no interpolation, no separate short-session generation.

    Real's NaN mask is imposed on the result, so both sides carry identical
    missingness, exactly as the GBM and MSV baselines do.

    Args:
        generated: SBBTS output, shape (M_simu, SESSION_BARS, PATH_FEATURES).
        path_tickers: Ticker each generated path is assigned to, length M_simu.
        path_sessions: Session each generated path belongs to, length M_simu.
        scales: Per-ticker scalars from ``build_training_tensor``.
        real_prices: Curated real prices, supplying the clock, column order,
            per-session anchor prices and NaN mask.

    Returns:
        Wide-format DataFrame with real's index, columns and NaN structure.

    Raises:
        ValueError: On a shape mismatch, an inconsistent assignment length,
            an unknown ticker or session, or a session whose bar count is
            neither ``SESSION_BARS`` nor ``EARLY_CLOSE_BARS``.
    """
    values = (
        generated.detach().cpu().numpy() if torch.is_tensor(generated) else np.asarray(generated)
    )
    if values.ndim != 3 or values.shape[1:] != (SESSION_BARS, PATH_FEATURES):
        raise ValueError(
            f"generated must have shape (M_simu, {SESSION_BARS}, {PATH_FEATURES}), "
            f"got {tuple(values.shape)}."
        )
    if not len(path_tickers) == len(path_sessions) == values.shape[0]:
        raise ValueError(
            f"path_tickers ({len(path_tickers)}) and path_sessions "
            f"({len(path_sessions)}) must both match M_simu ({values.shape[0]})."
        )

    column_of = {ticker: position for position, ticker in enumerate(real_prices.columns)}
    codes, uniques = pd.factorize(real_prices.index.normalize())
    bounds = {}
    for position, session in enumerate(uniques):
        rows = np.flatnonzero(codes == position)
        bounds[session] = (int(rows[0]), int(rows[-1]) + 1)

    real_values = real_prices.to_numpy(dtype=float)
    panel = np.full(real_prices.shape, np.nan, dtype=float)

    for path, ticker, session in zip(values, path_tickers, path_sessions, strict=True):
        if ticker not in column_of:
            raise ValueError(f"Unknown ticker {ticker!r}; not a column of real_prices.")
        if ticker not in scales:
            raise ValueError(f"No scale recorded for ticker {ticker!r}.")
        session = pd.Timestamp(session)
        if session not in bounds:
            raise ValueError(f"Unknown session {session!r}; not on the real market clock.")

        start, stop = bounds[session]
        length = stop - start
        if length not in (SESSION_BARS, EARLY_CLOSE_BARS):
            raise ValueError(
                f"Session {session.date()} has {length} bars; expected "
                f"{SESSION_BARS} or {EARLY_CLOSE_BARS}."
            )

        column = column_of[ticker]
        observed = real_values[start:stop, column]
        anchor_positions = np.flatnonzero(np.isfinite(observed))
        if anchor_positions.size == 0:
            # No real print anywhere in this session: nothing to anchor on.
            continue

        # Plain truncation for an early close: the first `length` bars of the
        # full-length generated path, untouched otherwise.
        log_path = invert_standardization(path[:length, 0], scales[ticker])
        panel[start:stop, column] = observed[anchor_positions[0]] * np.exp(log_path)

    panel[~np.isfinite(real_values)] = np.nan
    return pd.DataFrame(panel, index=real_prices.index, columns=real_prices.columns)


def main() -> None:
    print(f"Loading curated real prices: {REAL_PATH}")
    prices = pd.read_parquet(REAL_PATH)
    print(f"Real shape: {prices.shape}")

    full_sessions, early_sessions = split_sessions(prices.index)
    print(
        f"Sessions: {len(full_sessions)} full ({SESSION_BARS} bars), "
        f"{len(early_sessions)} early close ({EARLY_CLOSE_BARS} bars): "
        f"{[str(day.date()) for day in early_sessions]}"
    )

    examples, scales, dropped = build_training_examples(prices)
    assert len(scales) == prices.shape[1]
    assert all(example.values.shape == example.gaps.shape for example in examples)
    assert all(np.isfinite(example.values).all() for example in examples)

    lengths = np.array([len(example) for example in examples])
    possible = prices.shape[1] * len(full_sessions)
    print(
        f"Examples: {len(examples):,} of {possible:,} possible (ticker, session) "
        f"cells; {dropped:,} dropped for fewer than {MIN_SURVIVING_BARS} "
        f"surviving bars."
    )
    print(
        f"Surviving bars per example: min={lengths.min()}, "
        f"median={int(np.median(lengths))}, max={lengths.max()}, "
        f"mean={lengths.mean():.1f}; total points={int(lengths.sum()):,}."
    )

    SBBTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ragged sequences are stored flat with offsets, so no pickle is needed.
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(
        EXAMPLES_OUTPUT_PATH,
        values=np.concatenate([example.values for example in examples]),
        gaps=np.concatenate([example.gaps for example in examples]),
        offsets=offsets,
        tickers=np.array([example.ticker for example in examples]),
        sessions=np.array([example.session.value for example in examples], dtype=np.int64),
    )
    print(f"Saved: {EXAMPLES_OUTPUT_PATH} ({len(examples):,} examples)")

    np.savez(
        SCALES_OUTPUT_PATH,
        tickers=np.array(list(scales)),
        scales=np.array(list(scales.values()), dtype=float),
    )
    print(f"Saved: {SCALES_OUTPUT_PATH} ({len(scales)} per-ticker scales)")


if __name__ == "__main__":
    main()
