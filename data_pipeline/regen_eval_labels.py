"""
Helper: regenerate eval_ticker_labels.npy + eval_regime_labels.npy companion
files for an existing eval_deseasonalized.npy.

Avoids re-running the full pipeline when only the label arrays need rebuilding
(e.g. after CRASH_CUTOFF changes). Cleans only the eval-side tickers,
deseasonalizes with the saved FFF curve, and rewindows with labels=True.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_prep import (
    CRASH_CUTOFF,
    clean_ticker,
    deseasonalize,
    make_windows,
)

DEFAULT_RAW_DIR    = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/output_data")
WINDOW_LEN         = 2520


def regenerate(raw_dir: Path, output_dir: Path, window_len: int = WINDOW_LEN) -> None:
    split             = json.loads((output_dir / "ticker_split.json").read_text())
    eval_tickers_set  = split["eval"]
    fff               = np.load(output_dir / "fff_pattern.npy")
    eval_arr_existing = np.load(output_dir / "eval_deseasonalized.npy")

    print(f"ticker_split eval count : {len(eval_tickers_set)}")
    print(f"existing eval array     : {eval_arr_existing.shape}")
    print(f"CRASH_CUTOFF            : {CRASH_CUTOFF}")

    eval_dfs: dict[str, object] = {}
    for ticker in eval_tickers_set:
        csv_path = raw_dir / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"  WARN: missing CSV for {ticker}")
            continue
        df = clean_ticker(csv_path)
        if df is None or df.empty:
            continue
        eval_dfs[ticker] = deseasonalize(df, fff)

    print(f"cleaned eval tickers    : {len(eval_dfs)}")

    eval_arr, eval_tickers, eval_regimes = make_windows(
        eval_dfs, col="return_deseas",
        window_len=window_len, regime_labels=True,
    )

    assert eval_arr.shape == eval_arr_existing.shape, (
        f"shape mismatch: regen={eval_arr.shape} existing={eval_arr_existing.shape}"
    )
    assert np.allclose(eval_arr, eval_arr_existing), (
        "regenerated eval array differs from existing eval_deseasonalized.npy"
    )
    assert eval_tickers.shape[0] == eval_arr.shape[0]
    assert eval_regimes.shape[0] == eval_arr.shape[0]
    assert set(np.unique(eval_regimes).tolist()) == {0, 1}
    assert set(eval_tickers.tolist()) <= set(eval_tickers_set)

    np.save(output_dir / "eval_ticker_labels.npy", eval_tickers)
    np.save(output_dir / "eval_regime_labels.npy", eval_regimes)

    n_total   = eval_arr.shape[0]
    n0        = int((eval_regimes == 0).sum())
    n1        = int((eval_regimes == 1).sum())
    n_tickers = len(np.unique(eval_tickers))
    print()
    print(f"eval windows total : {n_total}")
    print(f"pre-crash (0)      : {n0}  ({100 * n0 / n_total:.1f}%)")
    print(f"crash (1)          : {n1}  ({100 * n1 / n_total:.1f}%)")
    print(f"unique tickers     : {n_tickers}")


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate eval label companion arrays.")
    p.add_argument("--raw-dir",    type=Path, default=DEFAULT_RAW_DIR,
                   help="Directory of per-ticker raw CSVs (default: data/raw)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory containing eval_deseasonalized.npy + ticker_split.json + fff_pattern.npy")
    p.add_argument("--window-len", type=int,  default=WINDOW_LEN)
    args = p.parse_args()
    regenerate(args.raw_dir, args.output_dir, args.window_len)


if __name__ == "__main__":
    main()
