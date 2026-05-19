"""
Variance decomposition of g_rr for high-CV buckets.

For each bucket in {B1, B4, B5, B6} and each sampler arm:
    1. PooledSampler         N=200,  200 draws, regime=all
    2. WithinTickerSampler   N=10,   200 draws, regime=all   (random ticker per draw)
    3. WithinRegimeSampler   N=50,   200 draws, regime=pre_crash
    4. WithinRegimeSampler   N=50,   200 draws, regime=crash

Emits runs/<ts>_variance_decomposition/results.csv with columns:
    bucket, sampler, regime, N_per_half, n_draws, n_valid,
    mean_g_rr, std_g_rr, cv_g_rr, seed
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from evaluation_framework.buckets import (        # noqa: E402
    BucketCFVC,
    BucketKurtosis,
    BucketMarginal,
    BucketTailRegime,
)
from evaluation_framework.io import load_eval_corpus  # noqa: E402
from evaluation_framework.samplers import (        # noqa: E402
    PooledSampler,
    Sampler,
    WithinRegimeSampler,
    WithinTickerSampler,
)
from scripts._runlog import (                       # noqa: E402
    create_run_dir,
    tee_stdout,
    write_config,
    write_env,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_SEED        = 42
N_DRAWS          = 200

POOLED_N         = 200
WITHIN_TICKER_N  = 10
WITHIN_REGIME_N  = 50

REGIME_NAMES     = {0: "pre_crash", 1: "crash"}

# ---------------------------------------------------------------------------
# Bucket gap functions — pooled distributional / cross-scale, per spec
# ---------------------------------------------------------------------------
_b1 = BucketMarginal(tail_q=0.05, n_quantile_grid=1000)
_b4 = BucketKurtosis()
_b5 = BucketCFVC()
_b6 = BucketTailRegime()

BUCKETS = {
    "B1": _b1.compute_gap,
    "B4": _b4.compute_gap,
    "B5": _b5.compute_gap,
    "B6": _b6.compute_gap,
}


def run_arm(
    bucket_key: str,
    gap_fn,
    real:       np.ndarray,
    sampler:    Sampler,
    n_per_half: int,
    n_draws:    int,
    seed:       int,
) -> tuple[np.ndarray, int]:
    """Return (valid_gaps, n_nan)."""
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_draws)
    for i in range(n_draws):
        idx_a, idx_b = sampler.draw_pair(n_per_half, rng)
        gaps[i] = gap_fn(real[idx_a], real[idx_b])
    nan_mask = ~np.isfinite(gaps)
    return gaps[~nan_mask], int(nan_mask.sum())


def summarize(name: str, gaps: np.ndarray, n_nan: int) -> dict:
    mean = float(gaps.mean()) if len(gaps) else float("nan")
    std  = float(gaps.std())  if len(gaps) else float("nan")
    cv   = (std / mean) if mean else float("nan")
    print(f"  [{name:<28}] mean={mean:.6g}  std={std:.6g}  "
          f"CV={cv:.4f}  n_valid={len(gaps)}/{len(gaps)+n_nan}")
    return {"mean": mean, "std": std, "cv": cv, "n_valid": len(gaps), "n_nan": n_nan}


def main() -> int:
    run_dir = create_run_dir("variance_decomposition")
    print(f"Run dir: {run_dir}")

    config = {
        "base_seed":          BASE_SEED,
        "n_draws":            N_DRAWS,
        "pooled_n":           POOLED_N,
        "within_ticker_n":    WITHIN_TICKER_N,
        "within_regime_n":    WITHIN_REGIME_N,
        "buckets":            list(BUCKETS.keys()),
    }
    write_config(run_dir, config)
    write_env(run_dir)

    with tee_stdout(run_dir / "log.txt"):
        print(f"Run dir: {run_dir}")
        print(f"Config:  {config}")

        print("\nLoading eval corpus + labels...")
        real, tickers, regimes = load_eval_corpus()
        print(f"  real    : {real.shape}")
        print(f"  tickers : {tickers.shape}  unique={len(set(tickers.tolist()))}")
        print(f"  regimes : pre_crash={int((regimes==0).sum())} "
              f"crash={int((regimes==1).sum())}")

        # B6 needs vol-regime thresholds fitted on the FULL corpus before any draw.
        print("\nFitting B6 vol-regime thresholds on full real corpus...")
        _b6.fit(real)

        # Deterministic per-arm seeds via SeedSequence.spawn — order of arms matters.
        arms: list[tuple[str, Sampler, int, str]] = []
        # Build samplers once per (sampler arm, regime); each is reused across buckets.
        pooled         = PooledSampler(len(real))
        within_ticker  = WithinTickerSampler(tickers, max_n_per_half=WITHIN_TICKER_N)
        within_reg_pre = WithinRegimeSampler(regimes, target_regime=0,
                                             max_n_per_half=WITHIN_REGIME_N)
        within_reg_cra = WithinRegimeSampler(regimes, target_regime=1,
                                             max_n_per_half=WITHIN_REGIME_N)
        arm_specs = [
            # (sampler_name, regime, sampler, N)
            ("pooled",        "all",       pooled,         POOLED_N),
            ("within_ticker", "all",       within_ticker,  WITHIN_TICKER_N),
            ("within_regime", "pre_crash", within_reg_pre, WITHIN_REGIME_N),
            ("within_regime", "crash",     within_reg_cra, WITHIN_REGIME_N),
        ]

        print(f"\nWithinTicker eligible tickers ({WITHIN_TICKER_N}×2 floor): "
              f"{len(within_ticker.eligible_tickers)}")
        print(f"WithinRegime pre_crash pool: {within_reg_pre.pool_size}")
        print(f"WithinRegime crash     pool: {within_reg_cra.pool_size}")

        # Spawn one independent rng-seed per (bucket, arm) cell so cells are
        # independent regardless of execution order.
        seq = np.random.SeedSequence(BASE_SEED)
        cell_seeds = seq.spawn(len(BUCKETS) * len(arm_specs))
        cell_iter = iter(cell_seeds)

        rows: list[dict] = []
        for bkey, gap_fn in BUCKETS.items():
            print(f"\n=== {bkey} ===")
            t_b = time.time()
            for sampler_name, regime, sampler, n in arm_specs:
                ss = next(cell_iter)
                seed = int(ss.generate_state(1, dtype=np.uint32)[0])
                t0 = time.time()
                gaps, n_nan = run_arm(bkey, gap_fn, real, sampler, n,
                                       N_DRAWS, seed)
                stats = summarize(f"{sampler_name}({regime}) N={n}", gaps, n_nan)
                rows.append({
                    "bucket":      bkey,
                    "sampler":     sampler_name,
                    "regime":      regime,
                    "N_per_half":  n,
                    "n_draws":     N_DRAWS,
                    "n_valid":     stats["n_valid"],
                    "mean_g_rr":   stats["mean"],
                    "std_g_rr":    stats["std"],
                    "cv_g_rr":     stats["cv"],
                    "seed":        seed,
                    "elapsed_s":   round(time.time() - t0, 2),
                })
            print(f"  (bucket {bkey} total {time.time() - t_b:.1f}s)")

        csv_path = run_dir / "results.csv"
        cols = ["bucket", "sampler", "regime", "N_per_half", "n_draws",
                "n_valid", "mean_g_rr", "std_g_rr", "cv_g_rr",
                "seed", "elapsed_s"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults written: {csv_path}")

        # Echo CSV
        print("\n--- results.csv ---")
        print(csv_path.read_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
