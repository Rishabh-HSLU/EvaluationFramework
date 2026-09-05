"""
Bridge the curated real panel and the SBBTS generator, in both directions.

Direction 1 (``build_training_tensor``) turns the curated real prices into
the tensor SBBTS trains on. Direction 2 (``assemble_price_panel``) turns
SBBTS's raw output back into a wide price panel shaped exactly like the
curated real file, so it can be written to parquet and read back through
``SBBTSBaselineLoader`` (fineval/data/curate.py).

Nothing here is wired into the benchmark. ``load_default_datasets`` and its
call sites are untouched; this module is standalone.

Conventions reused, not reinvented
----------------------------------
- **Per-ticker scale.** One volatility scalar per ticker, computed as
  ``np.nanstd(returns, axis=0, ddof=1)`` over that ticker's own pooled
  returns. This is M1's convention (fineval/metrics/tail_weighted_marginal.py),
  including its refusal to silently drop a degenerate column: a zero or
  non-finite scale names the offending tickers and raises.
- **Returns.** ``overnight_masked_log_returns`` from the shared session
  clock, so the bar at each session's open — a diff against the prior
  session's close, not a genuine 1-minute return — is masked exactly as
  everywhere else in the framework.
- **The masked opening bar becomes a zero increment**, matching
  ``scripts/baseline_generation.py``, which does the same with
  ``increments[session_start, :] = 0.0``.
- **Output panel format.** Wide DataFrame, curated market-clock index,
  same column order, float64, real's NaN mask imposed — identical to what
  ``CurationPipeline._save_datasets`` writes.

Session geometry
----------------
A regular session is ``TRADING_MINUTES = 390`` bars but only 389 valid
close-to-close returns; the opening bar is a structural NaN (see
``fineval/preprocessing/fff.py`` and ``AggregationalGaussianity``). The
training path therefore carries ``N + 1 = 391`` points: a t=0 anchor at the
session open, then one point per bar. Because the opening increment is zero
by the convention above, positions 0 and 1 of every path are both 0.0. That
redundancy is a property of the data — 390 bars carrying 389 returns — not a
padding choice made here.

The evaluation window holds two early-close sessions of 210 bars (1:00pm ET).
They are detected by bar count, never by date, and:

- excluded entirely from the per-ticker scale calculation and from the
  training tensor;
- reconstructed in direction 2 by generating a normal 390-bar path and
  keeping its first 210 bars — a plain truncation, no resampling.

``split_sessions`` raises unless exactly ``N_EARLY_CLOSE_SESSIONS`` short
sessions are found, each of exactly ``EARLY_CLOSE_BARS`` bars, so a change in
the curated window cannot pass through unnoticed.

Run from the repository root:

    uv run python -m scripts.sbbts_panel
"""

import numpy as np
import pandas as pd
import torch

from fineval.benchmark.config import CURATED_DIR
from fineval.config import TRADING_MINUTES
from fineval.preprocessing.session_clock import overnight_masked_log_returns

REAL_PATH = CURATED_DIR / "real_prices.parquet"
TENSOR_OUTPUT_PATH = CURATED_DIR / "sbbts_training_tensor.pt"
SCALES_OUTPUT_PATH = CURATED_DIR / "sbbts_ticker_scales.npz"

SESSION_BARS = TRADING_MINUTES  # 390; bars in a regular NYSE session
EARLY_CLOSE_BARS = 210  # bars in a 1:00pm ET early close
N_EARLY_CLOSE_SESSIONS = 2  # early closes in the curated window
PATH_FEATURES = 1  # d in SBBTS's (M, N + 1, d) contract
PATH_DTYPE = torch.float32


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


def per_ticker_scales(prices: pd.DataFrame, full_sessions: pd.DatetimeIndex) -> np.ndarray:
    """One volatility scalar per ticker, pooled over its full-length sessions.

    Reuses M1's standardization convention exactly: ``np.nanstd`` with
    ``ddof=1`` down each column of overnight-masked log returns, with a
    degenerate column named rather than dropped. Early-close sessions are
    excluded from the pool entirely.

    Args:
        prices: Curated wide-format prices (T, N).
        full_sessions: Session dates to pool over, from ``split_sessions``.

    Returns:
        Array of shape (N,), one positive finite scale per ticker, ordered
        like ``prices.columns``.

    Raises:
        ValueError: If any ticker's scale is zero or non-finite.
    """
    returns = overnight_masked_log_returns(prices)
    in_full = returns.index.normalize().isin(full_sessions)

    observed = returns.loc[in_full].to_numpy(dtype=float)
    observed = np.where(np.isfinite(observed), observed, np.nan)
    scales = np.nanstd(observed, axis=0, ddof=1)

    degenerate = ~np.isfinite(scales) | (scales <= 0.0)
    if degenerate.any():
        raise ValueError(
            "Cannot standardize columns with zero or non-finite standard "
            f"deviation: {prices.columns[degenerate].tolist()}"
        )
    return scales


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


def build_training_tensor(
    prices: pd.DataFrame,
    drop_incomplete: bool = True,
) -> tuple[torch.Tensor, dict[str, float], list[tuple[str, pd.Timestamp]]]:
    """Build the SBBTS training tensor from the curated real panel.

    One example per (ticker, full-length session): the standardized
    cumulative log-return path within that session, anchored at zero at the
    session open. Early-close sessions are excluded.

    Each example has ``SESSION_BARS + 1`` points — the t=0 anchor plus one
    per bar — giving the (M, N + 1, d) shape SBBTS expects with N = 390 and
    d = 1. Examples are ordered ticker-major, matching the returned index.

    Args:
        prices: Curated wide-format real prices (T, N).
        drop_incomplete: When True (default), a (ticker, session) cell with
            any missing bar is skipped rather than emitted with NaNs. The
            framework does not impute, and a NaN in a training tensor is
            worse than a smaller M.

    Returns:
        ``(tensor, scales, index)`` where ``tensor`` is float32 of shape
        (M, SESSION_BARS + 1, PATH_FEATURES), ``scales`` maps ticker to the
        scalar used to standardize it, and ``index`` lists the
        ``(ticker, session)`` pair behind each example, in tensor order.
    """
    full_sessions, _ = split_sessions(prices.index)
    scale_values = per_ticker_scales(prices, full_sessions)
    scales = dict(zip(prices.columns, scale_values.tolist(), strict=True))

    returns = overnight_masked_log_returns(prices)
    session_of_row = returns.index.normalize()

    tickers = list(prices.columns)
    n_tickers = len(tickers)
    n_sessions = len(full_sessions)

    paths = np.empty((n_sessions, SESSION_BARS + 1, n_tickers), dtype=np.float32)
    complete = np.zeros((n_sessions, n_tickers), dtype=bool)

    for position, session in enumerate(full_sessions):
        block = returns.loc[session_of_row == session].to_numpy(dtype=float, copy=True)
        # The session's opening bar is a structural NaN, not a missing print;
        # it enters the path as a zero increment, as in baseline_generation.py.
        block[0, :] = 0.0
        complete[position] = np.isfinite(block).all(axis=0)

        standardized = standardize_returns(block, scale_values)
        paths[position, 0, :] = 0.0
        paths[position, 1:, :] = np.cumsum(standardized, axis=0)

    if not drop_incomplete:
        complete[:] = True

    # Ticker-major ordering: all of ticker 0's sessions, then ticker 1's, ...
    ordered = paths.transpose(2, 0, 1).reshape(n_tickers * n_sessions, SESSION_BARS + 1)
    keep = complete.T.reshape(-1)

    tensor = torch.from_numpy(np.ascontiguousarray(ordered[keep])).to(PATH_DTYPE)
    tensor = tensor.reshape(-1, SESSION_BARS + 1, PATH_FEATURES)

    index = [
        (tickers[t], full_sessions[s])
        for t in range(n_tickers)
        for s in range(n_sessions)
        if complete[s, t]
    ]
    return tensor, scales, index


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
    real_prices = pd.read_parquet(REAL_PATH)
    print(f"Real shape: {real_prices.shape}")

    full_sessions, early_sessions = split_sessions(real_prices.index)
    print(
        f"Sessions: {len(full_sessions)} full ({SESSION_BARS} bars), "
        f"{len(early_sessions)} early close ({EARLY_CLOSE_BARS} bars): "
        f"{[str(day.date()) for day in early_sessions]}"
    )

    tensor, scales, index = build_training_tensor(real_prices)
    assert tensor.shape[1:] == (SESSION_BARS + 1, PATH_FEATURES)
    assert tensor.dtype == PATH_DTYPE
    assert len(index) == tensor.shape[0]
    assert len(scales) == real_prices.shape[1]

    possible = real_prices.shape[1] * len(full_sessions)
    print(
        f"Training tensor: {tuple(tensor.shape)} {tensor.dtype}, "
        f"{tensor.shape[0]:,} of {possible:,} possible (ticker, session) cells "
        f"({possible - tensor.shape[0]:,} dropped for missing bars)."
    )

    torch.save(tensor, TENSOR_OUTPUT_PATH)
    print(f"Saved: {TENSOR_OUTPUT_PATH} ({tuple(tensor.shape)})")

    np.savez(
        SCALES_OUTPUT_PATH,
        tickers=np.array(list(scales), dtype=object),
        scales=np.array(list(scales.values()), dtype=float),
    )
    print(f"Saved: {SCALES_OUTPUT_PATH} ({len(scales)} per-ticker scales)")


if __name__ == "__main__":
    main()
