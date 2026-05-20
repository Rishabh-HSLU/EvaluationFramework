"""
Canonical real-data pipeline for synthetic generator benchmarks.

Steps 1-6  : session filter, gap handling, 1-min log returns
Step 7     : liquidity tiers (280 train / 350 eval)
Step 8     : train/eval ticker split (seed 42)
Step 9     : pooled FFF deseasonalization (train-fitted)
Step 10    : per-ticker z-score on training tickers only

See README.md for full spec and output files.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .canonical import BenchmarkReference

SESSION_START_MIN = 570
SESSION_END_MIN = 959
MINUTES_PER_DAY = SESSION_END_MIN - SESSION_START_MIN + 1

TRAIN_BAR_THRESHOLD = 280
EVAL_BAR_THRESHOLD = 350
EVAL_HOLDOUT_FRAC = 0.20
RANDOM_SEED = 42
FFF_HARMONICS = 3

CRASH_CUTOFF = pd.Timestamp("2020-02-19").date()


def clean_ticker(csv_path: Path) -> pd.DataFrame | None:
    """Steps 1-6 for one ticker CSV. Columns: date_ny, minute_of_day_ny, log_return."""
    try:
        df = pd.read_csv(csv_path)
    except OSError as e:
        warnings.warn(f"Could not read {csv_path.name}: {e}")
        return None

    if "timestamp" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
    df["minute_of_day_ny"] = df["ts_ny"].dt.hour * 60 + df["ts_ny"].dt.minute
    df["date_ny"] = df["ts_ny"].dt.date

    regular = df[
        (df["minute_of_day_ny"] >= SESSION_START_MIN)
        & (df["minute_of_day_ny"] <= SESSION_END_MIN)
    ].copy()
    if regular.empty:
        return None

    regular = regular.sort_values("timestamp").reset_index(drop=True)
    regular["diff_min"] = regular["timestamp"].diff().dt.total_seconds() / 60
    regular["same_day"] = regular["date_ny"] == regular["date_ny"].shift(1)

    is_session_start = ~regular["same_day"]
    is_post_gap = regular["same_day"] & (regular["diff_min"] > 1)
    regular_clean = regular[~(is_session_start | is_post_gap)].copy()
    if len(regular_clean) < 2:
        return None

    regular_clean["log_return"] = np.log(
        regular_clean["close"] / regular_clean["close"].shift(1)
    )
    regular_clean = regular_clean.dropna(subset=["log_return"])
    return regular_clean[["date_ny", "minute_of_day_ny", "log_return"]]


def load_cleaned_tickers(csv_files: list[Path]) -> dict[str, pd.DataFrame]:
    """One pass over CSVs; returns {ticker: cleaned DataFrame}."""
    out: dict[str, pd.DataFrame] = {}
    for path in csv_files:
        df = clean_ticker(path)
        if df is not None and not df.empty:
            out[path.stem] = df
    return out


def liquidity_row(ticker: str, df: pd.DataFrame) -> dict:
    bars_per_day = df.groupby("date_ny").size()
    return {
        "ticker": ticker,
        "median_bars_per_day": float(bars_per_day.median()),
        "trading_days": len(bars_per_day),
        "total_returns": len(df),
    }


def liquidity_stats_table(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Step 7 liquidity classes from already-cleaned tickers."""
    df = pd.DataFrame([liquidity_row(t, d) for t, d in sorted(cleaned.items())])
    df["liquidity_class"] = "excluded"
    df.loc[df["median_bars_per_day"] >= TRAIN_BAR_THRESHOLD, "liquidity_class"] = (
        "train_only"
    )
    df.loc[df["median_bars_per_day"] >= EVAL_BAR_THRESHOLD, "liquidity_class"] = (
        "eval_eligible"
    )
    return df


def make_ticker_split(stats_df: pd.DataFrame) -> dict[str, list[str]]:
    rng = np.random.default_rng(RANDOM_SEED)
    eval_eligible = stats_df.loc[
        stats_df["liquidity_class"] == "eval_eligible", "ticker"
    ].tolist()
    train_only = stats_df.loc[
        stats_df["liquidity_class"] == "train_only", "ticker"
    ].tolist()

    arr = np.array(sorted(eval_eligible))
    rng.shuffle(arr)
    n_eval = max(1, round(len(arr) * EVAL_HOLDOUT_FRAC)) if len(arr) else 0
    eval_tickers = arr[:n_eval].tolist()
    train_liquid = arr[n_eval:].tolist()
    return {"train": sorted(train_liquid + train_only), "eval": sorted(eval_tickers)}


def _fff_func(tau: np.ndarray, *params) -> np.ndarray:
    c0 = params[0]
    result = np.full_like(tau, c0, dtype=float)
    for j in range(1, FFF_HARMONICS + 1):
        a_j, b_j = params[2 * j - 1], params[2 * j]
        result += a_j * np.cos(2 * np.pi * j * tau / MINUTES_PER_DAY) + b_j * np.sin(
            2 * np.pi * j * tau / MINUTES_PER_DAY
        )
    return result


def fit_fff(train_returns: list[pd.DataFrame]) -> np.ndarray:
    pooled = pd.concat(train_returns, ignore_index=True)
    pooled["tau"] = pooled["minute_of_day_ny"] - SESSION_START_MIN
    mean_abs = (
        pooled.groupby("tau")["log_return"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="mean_abs_return")
        .sort_values("tau")
    )
    tau_vals = mean_abs["tau"].values.astype(float)
    y_vals = mean_abs["mean_abs_return"].values
    n_params = 1 + 2 * FFF_HARMONICS
    p0 = np.zeros(n_params)
    p0[0] = y_vals.mean()
    popt, _ = curve_fit(_fff_func, tau_vals, y_vals, p0=p0, maxfev=10_000)
    s = _fff_func(np.arange(MINUTES_PER_DAY, dtype=float), *popt)
    return np.clip(s, a_min=1e-8, a_max=None)


def deseasonalize(df: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    tau = (df["minute_of_day_ny"] - SESSION_START_MIN).values.astype(int)
    out = df.copy()
    out["return_deseas"] = out["log_return"] / s[tau]
    return out


def normalize_ticker(
    df: pd.DataFrame,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    if mean is None:
        mean = float(df["return_deseas"].mean())
    if std is None:
        std = max(float(df["return_deseas"].std()), 1e-8)
    out = df.copy()
    out["return_normed"] = (out["return_deseas"] - mean) / std
    return out, mean, std


def make_windows(
    ticker_dfs: dict[str, pd.DataFrame],
    col: str,
    window_len: int = 2520,
    regime_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Non-overlapping windows in sorted ticker order.

    Returns (windows, ticker_labels, regime_labels).
    regime_labels is None when regime_labels=False.
    """
    windows: list[np.ndarray] = []
    tickers: list[str] = []
    regimes: list[int] = []

    for ticker in sorted(ticker_dfs):
        df = ticker_dfs[ticker]
        vals = df[col].values
        dates = df["date_ny"].values if regime_labels else None
        for i in range(len(vals) // window_len):
            start, end = i * window_len, (i + 1) * window_len
            windows.append(vals[start:end])
            tickers.append(ticker)
            if regime_labels:
                regimes.append(0 if dates[start] < CRASH_CUTOFF else 1)

    if not windows:
        empty_w = np.empty((0, window_len, 1))
        empty_l = np.array([], dtype=str)
        if regime_labels:
            return empty_w, empty_l, np.array([], dtype=np.int8)
        return empty_w, empty_l, None

    arr = np.stack(windows)[:, :, np.newaxis]
    ticker_arr = np.array(tickers, dtype=str)
    if regime_labels:
        return arr, ticker_arr, np.array(regimes, dtype=np.int8)
    return arr, ticker_arr, None


def split_train_val_indices(
    ticker_labels: np.ndarray,
    val_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ticker holdout: last val_frac of each ticker's windows go to validation."""
    n = len(ticker_labels)
    is_val = np.zeros(n, dtype=bool)
    for ticker in np.unique(ticker_labels):
        idx = np.flatnonzero(ticker_labels == ticker)
        if len(idx) <= 1:
            continue
        n_val = max(1, round(len(idx) * val_frac))
        is_val[idx[-n_val:]] = True
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def build_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    window_len: int = 2520,
) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        f for f in data_dir.glob("*.csv") if f.name != "download_log.csv"
    )
    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    print("Cleaning tickers (Steps 1-6, single pass)...")
    cleaned = load_cleaned_tickers(csv_files)
    print(f"  Cleaned: {len(cleaned)} tickers")

    print("Liquidity stats and split (Steps 7-8)...")
    stats_df = liquidity_stats_table(cleaned)
    stats_df.to_csv(output_dir / "stats_df.csv", index=False)
    split = make_ticker_split(stats_df)
    with open(output_dir / "ticker_split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"  Training tickers : {len(split['train'])}")
    print(f"  Evaluation tickers: {len(split['eval'])}")

    train_dfs = {t: cleaned[t] for t in split["train"] if t in cleaned}
    eval_dfs = {t: cleaned[t] for t in split["eval"] if t in cleaned}
    del cleaned

    print("FFF fit and deseasonalization (Step 9)...")
    s = fit_fff(list(train_dfs.values()))
    np.save(output_dir / "fff_pattern.npy", s)
    train_dfs = {t: deseasonalize(df, s) for t, df in train_dfs.items()}
    eval_dfs = {t: deseasonalize(df, s) for t, df in eval_dfs.items()}

    print("Normalization (Step 10)...")
    norm_stats: dict[str, dict] = {}
    train_normed: dict[str, pd.DataFrame] = {}
    for ticker, df in train_dfs.items():
        df_n, mean, std = normalize_ticker(df)
        train_normed[ticker] = df_n
        norm_stats[ticker] = {"mean": mean, "std": std}
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Windowing (window_len={window_len})...")
    train_arr, train_labels, _ = make_windows(
        train_normed, col="return_normed", window_len=window_len
    )
    eval_arr, eval_labels, eval_regimes = make_windows(
        eval_dfs,
        col="return_deseas",
        window_len=window_len,
        regime_labels=True,
    )

    np.save(output_dir / "train_normalized.npy", train_arr)
    np.save(output_dir / "train_ticker_labels.npy", train_labels)
    np.save(output_dir / "eval_deseasonalized.npy", eval_arr)
    np.save(output_dir / "eval_ticker_labels.npy", eval_labels)
    np.save(output_dir / "eval_regime_labels.npy", eval_regimes)

    ref = BenchmarkReference.from_eval(output_dir / "eval_deseasonalized.npy")
    ref.write_manifest(output_dir)

    print(f"\nDone -> {output_dir}")
    print(f"  train_normalized.npy     {train_arr.shape}")
    print(f"  eval_deseasonalized.npy  {eval_arr.shape}")
    if eval_regimes is not None and len(eval_regimes):
        print(
            f"  eval regimes  pre={int((eval_regimes == 0).sum())}  "
            f"crash={int((eval_regimes == 1).sum())}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build canonical train/eval datasets.")
    p.add_argument("data_dir")
    p.add_argument("output_dir")
    p.add_argument("--window_len", type=int, default=2520)
    args = p.parse_args()
    build_dataset(args.data_dir, args.output_dir, args.window_len)
