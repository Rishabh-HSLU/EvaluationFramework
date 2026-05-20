"""
Transform the AIL synthetic parquet into the benchmark .npy format.

AIL ships a single wide-format parquet of long synthetic sequences. The
benchmark expects non-overlapping windows of `WINDOW_LEN` minutes,
deseasonalized with the same FFF curve fitted on the real training corpus.

Pipeline (mirrors data_prep.clean_ticker but on grouped parquet rows):
    Steps 1-6  : NY session filter, sort, intra-day gap mask, log returns
    Step 9     : deseasonalize using the saved fff_pattern.npy (no refit)
    Windowing  : non-overlapping windows of WINDOW_LEN minutes

Run:
    uv run python -m data_pipeline.transform_ail \\
        --parquet path/to/dataset.parquet \\
        --output-dir data/output_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import save_benchmark_corpus

WINDOW_LEN        = 2520
SESSION_START_MIN = 570
SESSION_END_MIN   = 959


def load_fff(path: Path) -> np.ndarray:
    s = np.load(path)
    assert s.shape == (390,), f"Expected fff_pattern shape (390,), got {s.shape}"
    return s


def clean_group(df: pd.DataFrame) -> np.ndarray | None:
    """Apply Steps 1-6 to one ticker's rows. Returns (N, 2) of [tau, log_return] or None."""
    ts_utc = df["timestamp"].dt.tz_localize("UTC")
    ts_ny  = ts_utc.dt.tz_convert("America/New_York")
    minute_of_day = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    date_ny       = ts_ny.dt.date

    mask    = (minute_of_day >= SESSION_START_MIN) & (minute_of_day <= SESSION_END_MIN)
    close   = df["close"].values[mask]
    mod     = minute_of_day.values[mask]
    dates   = date_ny.values[mask]
    ts_vals = df["timestamp"].values[mask]
    if len(close) < 2:
        return None

    order   = np.argsort(ts_vals)
    close, mod, dates, ts_vals = close[order], mod[order], dates[order], ts_vals[order]

    diff_min  = np.diff(ts_vals).astype("timedelta64[m]").astype(float)
    same_day  = dates[1:] == dates[:-1]

    is_session_start      = np.zeros(len(close), dtype=bool)
    is_post_gap           = np.zeros(len(close), dtype=bool)
    is_session_start[0]   = True
    is_session_start[1:]  = ~same_day
    is_post_gap[1:]       = same_day & (diff_min > 1)
    bad = is_session_start | is_post_gap

    valid_return = ~bad[1:]
    log_returns  = np.log(close[1:] / close[:-1])
    tau_returns  = mod[1:] - SESSION_START_MIN

    log_returns = log_returns[valid_return]
    tau_returns = tau_returns[valid_return]
    if len(log_returns) == 0:
        return None
    return np.stack([tau_returns, log_returns], axis=1)


def process_ticker(tau_ret: np.ndarray, s: np.ndarray, window_len: int) -> list[np.ndarray]:
    tau         = tau_ret[:, 0].astype(int)
    log_returns = tau_ret[:, 1]
    deseas      = log_returns / s[tau]
    n_windows   = len(deseas) // window_len
    return [
        deseas[i * window_len:(i + 1) * window_len].astype(np.float32)
        for i in range(n_windows)
    ]


def transform(
    parquet_path: Path,
    output_dir:   Path,
    out_name:     str = "ail_synthetic.npy",
    window_len:   int = WINDOW_LEN,
) -> np.ndarray:
    fff_path = output_dir / "fff_pattern.npy"
    print(f"FFF pattern from   : {fff_path}")
    s = load_fff(fff_path)

    print(f"Loading parquet    : {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  shape            : {df.shape}")
    tickers = df["tr_ric"].unique()
    print(f"  tickers          : {len(tickers)}")

    all_windows: list[np.ndarray] = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            print(f"  processing {i+1}/{len(tickers)}  windows={len(all_windows)}")
        group   = df[df["tr_ric"] == ticker]
        tau_ret = clean_group(group)
        if tau_ret is None:
            skipped += 1
            continue
        all_windows.extend(process_ticker(tau_ret, s, window_len))

    print(f"Skipped {skipped} tickers (empty after cleaning); windows={len(all_windows)}")
    if not all_windows:
        raise RuntimeError("No windows produced — check parquet path.")
    arr = np.stack(all_windows)
    out = save_benchmark_corpus(arr, output_dir / out_name, output_dir)
    print(f"Saved {output_dir / out_name}  shape=({out.shape[0]}, {out.shape[1]}, 1)")
    print(f"  mean={out.mean():.6f}  std={out.std():.6f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="AIL parquet -> benchmark .npy.")
    p.add_argument("--parquet",    type=Path, required=True,
                   help="Path to the AIL synthetic parquet.")
    p.add_argument("--output-dir", type=Path, default=Path("data/output_data"),
                   help="Directory containing fff_pattern.npy + benchmark_manifest.json.")
    p.add_argument("--out-name",   type=str,  default="ail_synthetic.npy")
    p.add_argument("--window-len", type=int,  default=WINDOW_LEN)
    args = p.parse_args()
    transform(args.parquet, args.output_dir, args.out_name, args.window_len)


if __name__ == "__main__":
    main()
