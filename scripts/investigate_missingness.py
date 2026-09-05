"""
Diagnose the missing-bar structure of the curated real panel.

`scripts/sbbts_panel.py` drops any (ticker, session) cell with a missing
bar, and a six-ticker probe suggested that discards most of the panel. This
script asks why: are the gaps isolated no-trade minutes or genuine data
outages, are they concentrated in a few chronically thin tickers or spread
across the universe, do they cluster at the open or the close, and are they
mostly coming from tickers that barely cleared the 70% inclusion floor?

This is an investigation, not a framework feature. It reads the curated real
parquet and writes a report; it modifies nothing and is wired into nothing.
Session geometry is borrowed from `scripts.sbbts_panel.split_sessions`, so
early closes are excluded by bar count rather than by hardcoded date, and
only full-length 390-bar sessions are analysed.

Coverage ratios reuse the curation pipeline's own definition — the
per-ticker fraction of non-NaN bars over the whole market clock, which is
what `CurationPipeline._compute_joint_coverage` thresholds at
`COVERAGE_FLOOR`.

Run from the repository root:

    uv run python -m scripts.investigate_missingness
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fineval.benchmark.config import CURATED_DIR
from fineval.config import COVERAGE_FLOOR
from scripts.sbbts_panel import SESSION_BARS, split_sessions

REAL_PATH = CURATED_DIR / "real_prices.parquet"
REPORT_PATH = Path(__file__).resolve().parent / "missingness_report.md"

#: Tickers that only just cleared the inclusion floor, versus ones with room
#: to spare. Used to test whether incomplete sessions are a marginal-ticker
#: problem or a universal one.
NEAR_FLOOR_BAND = (0.70, 0.80)
WELL_COVERED_FLOOR = 0.95

#: "Chronic" tickers are the worst decile by incomplete-session rate.
CHRONIC_QUANTILE = 0.90

PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)


def missing_cube(
    prices: pd.DataFrame,
    full_sessions: pd.DatetimeIndex,
) -> np.ndarray:
    """Missing-bar mask for full-length sessions, shaped (S, 390, N).

    Rows are already ordered session by session on the curated clock, and
    every retained session has exactly ``SESSION_BARS`` bars, so the filtered
    mask reshapes directly into a session cube.
    """
    missing = ~np.isfinite(prices.to_numpy(dtype=float))
    in_full = prices.index.normalize().isin(full_sessions)
    selected = missing[in_full]

    expected = len(full_sessions) * SESSION_BARS
    if selected.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} full-session rows ({len(full_sessions)} x "
            f"{SESSION_BARS}), got {selected.shape[0]}."
        )
    return selected.reshape(len(full_sessions), SESSION_BARS, prices.shape[1])


def gap_run_lengths(cube: np.ndarray) -> np.ndarray:
    """Lengths of every contiguous run of missing bars, pooled over cells.

    A run of 1 is an isolated missing minute; a run of 20 is a twenty-minute
    hole. Returned unsorted, one entry per run.
    """
    n_sessions, n_bars, n_tickers = cube.shape
    cells = cube.transpose(0, 2, 1).reshape(n_sessions * n_tickers, n_bars)

    padding = np.zeros((cells.shape[0], 1), dtype=np.int8)
    padded = np.concatenate([padding, cells.astype(np.int8), padding], axis=1)
    edges = np.diff(padded, axis=1)

    starts = np.argwhere(edges == 1)
    ends = np.argwhere(edges == -1)
    return ends[:, 1] - starts[:, 1]


def describe(values: np.ndarray, label: str) -> list[str]:
    """Render a percentile table for one distribution."""
    if values.size == 0:
        return [f"No {label} to describe."]
    quantiles = np.percentile(values, PERCENTILES)
    lines = [f"| statistic | {label} |", "|---|---|", f"| count | {values.size:,} |"]
    names = {0: "min", 50: "median", 100: "max"}
    for percentile, value in zip(PERCENTILES, quantiles, strict=True):
        name = names.get(percentile, f"p{percentile}")
        lines.append(f"| {name} | {value:,.0f} |")
    lines.append(f"| mean | {values.mean():,.2f} |")
    return lines


def build_report(prices: pd.DataFrame) -> list[str]:
    """Compute every section and return the report as markdown lines."""
    full_sessions, early_sessions = split_sessions(prices.index)
    tickers = list(prices.columns)
    cube = missing_cube(prices, full_sessions)

    n_sessions, _, n_tickers = cube.shape
    per_cell = cube.sum(axis=1)  # (S, N) missing bars per (session, ticker)
    total_cells = n_sessions * n_tickers
    incomplete = per_cell > 0
    n_incomplete = int(incomplete.sum())
    n_complete = total_cells - n_incomplete

    lines: list[str] = [
        "# Missing-bar structure of the curated real panel",
        "",
        f"Source: `{REAL_PATH}` — {prices.shape[0]:,} rows x {prices.shape[1]:,} tickers.",
        f"Restricted to {n_sessions} full-length {SESSION_BARS}-bar sessions; "
        f"{len(early_sessions)} early-close sessions excluded by bar count "
        f"({', '.join(str(day.date()) for day in early_sessions)}).",
        "",
        "## 1. Overall shape",
        "",
        f"- (ticker, session) pairs: **{total_cells:,}** "
        f"({n_tickers:,} tickers x {n_sessions} sessions)",
        f"- Fully complete (zero missing bars): **{n_complete:,}** "
        f"({n_complete / total_cells:.1%})",
        f"- At least one missing bar: **{n_incomplete:,}** ({n_incomplete / total_cells:.1%})",
        "",
        f"Missing-bar counts among the incomplete pairs (out of {SESSION_BARS} bars):",
        "",
    ]
    lines += describe(per_cell[incomplete].astype(float), "missing bars")

    # ---- 2. Concentration across tickers -------------------------------
    incomplete_per_ticker = incomplete.sum(axis=0)  # (N,)
    rate_per_ticker = incomplete_per_ticker / n_sessions
    any_incomplete = int((incomplete_per_ticker > 0).sum())
    never_complete = int((incomplete_per_ticker == n_sessions).sum())

    threshold = float(np.quantile(rate_per_ticker, CHRONIC_QUANTILE))
    chronic = rate_per_ticker >= threshold
    chronic_share = incomplete_per_ticker[chronic].sum() / max(n_incomplete, 1)
    worst_order = np.argsort(rate_per_ticker)[::-1][:5]
    worst_tickers = ", ".join(f"{tickers[i]} ({rate_per_ticker[i]:.0%})" for i in worst_order)

    lines += [
        "",
        "## 2. Concentration across tickers",
        "",
        f"- Tickers with at least one incomplete session: **{any_incomplete:,}** "
        f"of {n_tickers:,} (**{any_incomplete / n_tickers:.1%}**)",
        f"- Tickers with **no** complete session at all: **{never_complete:,}** "
        f"({never_complete / n_tickers:.1%})",
        f"- Worst decile by incomplete-session rate ({int(chronic.sum())} tickers, "
        f"rate >= {threshold:.1%}) accounts for **{chronic_share:.1%}** of all "
        "incomplete (ticker, session) pairs",
        f"- Worst five tickers: {worst_tickers}",
        "",
        "Per-ticker share of sessions that are incomplete:",
        "",
    ]
    lines += describe(rate_per_ticker * 100.0, "incomplete sessions (%)")

    # ---- 3. Contiguity of gaps -----------------------------------------
    runs = gap_run_lengths(cube)
    singletons = int((runs == 1).sum())
    long_runs = int((runs >= 10).sum())
    bars_in_singletons = singletons
    bars_total = int(runs.sum())

    lines += [
        "",
        "## 3. Contiguity of gaps within a session",
        "",
        f"- Contiguous missing-bar runs: **{runs.size:,}**, covering {bars_total:,} missing bars",
        f"- Isolated single minutes (run length 1): **{singletons:,}** "
        f"(**{singletons / max(runs.size, 1):.1%}** of runs, "
        f"{bars_in_singletons / max(bars_total, 1):.1%} of missing bars)",
        f"- Runs of 10+ consecutive minutes: **{long_runs:,}** "
        f"({long_runs / max(runs.size, 1):.2%} of runs)",
        "",
        "Run-length distribution:",
        "",
    ]
    lines += describe(runs.astype(float), "run length (minutes)")

    buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 60), (61, SESSION_BARS)]
    lines += [
        "",
        "| run length | runs | share of runs | missing bars | share of bars |",
        "|---|---|---|---|---|",
    ]
    for low, high in buckets:
        selected = runs[(runs >= low) & (runs <= high)]
        label = f"{low}" if low == high else f"{low}-{high}"
        lines.append(
            f"| {label} | {selected.size:,} | {selected.size / max(runs.size, 1):.1%} | "
            f"{int(selected.sum()):,} | {selected.sum() / max(bars_total, 1):.1%} |"
        )

    # ---- 4. Time-of-day location ---------------------------------------
    per_minute = cube.sum(axis=(0, 2)) / total_cells  # (390,)
    positions = np.arange(1, SESSION_BARS + 1)

    # The 16:00 close bar behaves nothing like its neighbours, so it is
    # summarised on its own rather than averaged into the last half hour.
    close_rate = float(per_minute[-1])
    prior_rate = float(per_minute[-2])
    interior = per_minute[:-1]  # positions 1..389
    first_30 = float(interior[:30].mean())
    last_30 = float(interior[-30:].mean())
    middle = float(interior[30:-30].mean())

    close_per_session = cube[:, -1, :].mean(axis=1)
    worst = np.argsort(interior)[::-1][:10]

    # Elevated / suppressed relative to the session's own middle baseline.
    open_elevated = first_30 > 1.25 * middle
    close_clean = last_30 < 0.75 * middle
    close_elevated = last_30 > 1.25 * middle

    if open_elevated and close_clean:
        shape = (
            "a **dirty open decaying into a clean close**: missingness peaks in "
            "the first minutes, drifts down through the session, and all but "
            "vanishes over the final half hour"
        )
    elif close_clean:
        shape = "**flat then clean**: a stable middle giving way to a much cleaner close"
    elif open_elevated and close_elevated:
        shape = "**U-shaped**, elevated at both ends relative to the middle"
    elif open_elevated:
        shape = "concentrated near the **open**"
    elif close_elevated:
        shape = "concentrated near the **close**"
    else:
        shape = "roughly **uniform** across the session"

    lines += [
        "",
        "## 4. Time-of-day location",
        "",
        f"- Missing rate per minute position, averaged over all {total_cells:,} pairs",
        f"- First 30 minutes: **{first_30:.2%}** | middle: **{middle:.2%}** | "
        f"last 30 excluding the close bar: **{last_30:.2%}**",
        f"- Ratio open/middle: **{first_30 / middle:.2f}x**, "
        f"close/middle: **{last_30 / middle:.2f}x**",
        f"- Excluding position {SESSION_BARS}, the pattern is {shape}.",
        "",
        f"**Position {SESSION_BARS} (16:00) is a structural outlier and is excluded "
        "from the figures above.**",
        "",
        f"- Missing rate at position {SESSION_BARS}: **{close_rate:.2%}**",
        f"- Missing rate at position {SESSION_BARS - 1} (15:59): **{prior_rate:.2%}** "
        f"— the close bar is **{close_rate / max(prior_rate, 1e-12):,.0f}x** its neighbour",
        f"- Per-session spread of the close-bar gap: min **{close_per_session.min():.1%}**, "
        f"median **{np.median(close_per_session):.1%}**, max **{close_per_session.max():.1%}** "
        f"across all {len(close_per_session)} sessions",
        "",
        "It affects roughly half of all tickers in *every* session rather than "
        "clustering on particular days, so it is a systematic property of the "
        "16:00 print — plausibly closing-auction timing — not an outage. Note "
        "`session_clock.py` already documents a separate close-bar issue "
        "(`SESSION_OFFSET` excluding minute position 390 from FFF slots).",
        "",
        f"Ten worst minute positions excluding {SESSION_BARS}:",
        "",
        "| minute position | missing rate |",
        "|---|---|",
    ]
    for position in worst:
        lines.append(f"| {positions[position]} | {interior[position]:.2%} |")

    # ---- 5. Relationship to the inclusion floor ------------------------
    coverage = prices.notna().to_numpy().mean(axis=0)  # pipeline's definition
    complete_rate = 1.0 - rate_per_ticker

    near_low, near_high = NEAR_FLOOR_BAND
    near = (coverage >= near_low) & (coverage < near_high)
    well = coverage >= WELL_COVERED_FLOOR

    lines += [
        "",
        "## 5. Relationship to the 70% inclusion floor",
        "",
        f"Coverage is the per-ticker non-NaN fraction over the whole market clock — "
        f"the quantity `CurationPipeline` thresholds at `COVERAGE_FLOOR = "
        f"{COVERAGE_FLOOR:.0%}`.",
        "",
        f"- Coverage range across retained tickers: "
        f"**{coverage.min():.1%}** to **{coverage.max():.1%}** "
        f"(median {np.median(coverage):.1%})",
        "",
        "| group | tickers | mean coverage | mean complete-session rate | "
        "share of all incomplete pairs |",
        "|---|---|---|---|---|",
    ]
    for label, mask in (
        (f"near floor ({near_low:.0%}-{near_high:.0%})", near),
        (f"well covered (>= {WELL_COVERED_FLOOR:.0%})", well),
        ("all retained tickers", np.ones(n_tickers, dtype=bool)),
    ):
        count = int(mask.sum())
        if count == 0:
            lines.append(f"| {label} | 0 | — | — | — |")
            continue
        share = incomplete_per_ticker[mask].sum() / max(n_incomplete, 1)
        lines.append(
            f"| {label} | {count:,} | {coverage[mask].mean():.1%} | "
            f"{complete_rate[mask].mean():.1%} | {share:.1%} |"
        )

    correlation = float(np.corrcoef(coverage, complete_rate)[0, 1])
    lines += [
        "",
        f"- Correlation between overall coverage and session-completeness rate: "
        f"**{correlation:+.3f}**",
    ]
    return lines


def main() -> None:
    print(f"Loading curated real prices: {REAL_PATH}")
    prices = pd.read_parquet(REAL_PATH)
    print(f"Real shape: {prices.shape}")

    lines = build_report(prices)

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report)

    print()
    print(report)
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
