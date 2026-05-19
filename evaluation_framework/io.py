"""Load benchmark return corpora from disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .canonical import BenchmarkReference, validate_corpus
from .paths import real_corpus, real_regime_labels, real_ticker_labels


def load_corpus(
    path: Path,
    reference: BenchmarkReference | None = None,
    name: str | None = None,
) -> np.ndarray:
    """(N, T) float64 from .npy; optional validation against benchmark manifest."""
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.float64, copy=False)
    if reference is not None:
        validate_corpus(arr, reference, name or path.stem)
    return arr


def load_eval_corpus(
    reference: BenchmarkReference | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the real eval corpus alongside its companion ticker and regime labels.

    Returns
    -------
    (real, ticker_labels, regime_labels)
        real          : (N, T) float64
        ticker_labels : (N,)  str   — ticker for each window
        regime_labels : (N,)  int8  — 0 = pre-crash, 1 = crash
    """
    real    = load_corpus(real_corpus(), reference, "real")
    tickers = np.load(real_ticker_labels())
    regimes = np.load(real_regime_labels())
    if not (len(real) == len(tickers) == len(regimes)):
        raise ValueError(
            f"Length mismatch: real={len(real)}, tickers={len(tickers)}, "
            f"regimes={len(regimes)}"
        )
    return real, tickers, regimes
