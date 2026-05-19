"""
B1 N-ablation: CV of g_rr as a function of N (paths per half).

PooledSampler, 200 draws per N. N ∈ {50, 100, 200, 400}.

Emits runs/<ts>_b1_n_ablation/results.csv with columns:
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

from evaluation_framework.buckets import BucketMarginal       # noqa: E402
from evaluation_framework.io import load_eval_corpus           # noqa: E402
from evaluation_framework.samplers import PooledSampler        # noqa: E402
from scripts._runlog import (                                   # noqa: E402
    create_run_dir,
    tee_stdout,
    write_config,
    write_env,
)

BASE_SEED = 42
N_DRAWS   = 200
N_VALUES  = [50, 100, 200, 400]

_b1 = BucketMarginal(tail_q=0.05, n_quantile_grid=1000)


def run_n(real: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampler = PooledSampler(len(real))
    gaps = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        a, b = sampler.draw_pair(n, rng)
        gaps[i] = _b1.compute_gap(real[a], real[b])
    return gaps


def main() -> int:
    run_dir = create_run_dir("b1_n_ablation")
    print(f"Run dir: {run_dir}")
    config = {
        "base_seed": BASE_SEED,
        "n_draws":   N_DRAWS,
        "n_values":  N_VALUES,
        "bucket":    "B1",
        "sampler":   "pooled",
    }
    write_config(run_dir, config)
    write_env(run_dir)

    with tee_stdout(run_dir / "log.txt"):
        print(f"Run dir: {run_dir}")
        print(f"Config:  {config}")

        real, _, _ = load_eval_corpus()
        print(f"\nReal corpus: {real.shape}")

        # Independent per-N rng streams
        seq = np.random.SeedSequence(BASE_SEED)
        seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                 for s in seq.spawn(len(N_VALUES))]

        rows = []
        for n, seed in zip(N_VALUES, seeds):
            t0 = time.time()
            gaps = run_n(real, n, seed)
            mean = float(gaps.mean())
            std  = float(gaps.std())
            cv   = std / mean if mean else float("nan")
            print(f"  N={n:>3}  mean={mean:.6f}  std={std:.6f}  "
                  f"CV={cv:.4f}  ({time.time()-t0:.1f}s)")
            rows.append({
                "bucket":     "B1",
                "sampler":    "pooled",
                "regime":     "all",
                "N_per_half": n,
                "n_draws":    N_DRAWS,
                "n_valid":    N_DRAWS,
                "mean_g_rr":  mean,
                "std_g_rr":   std,
                "cv_g_rr":    cv,
                "seed":       seed,
                "elapsed_s":  round(time.time() - t0, 2),
            })

        csv_path = run_dir / "results.csv"
        cols = ["bucket", "sampler", "regime", "N_per_half", "n_draws",
                "n_valid", "mean_g_rr", "std_g_rr", "cv_g_rr",
                "seed", "elapsed_s"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults: {csv_path}")
        print("\n--- results.csv ---")
        print(csv_path.read_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
