"""
Paired vs unpaired bootstrap CI comparison.

For each (bucket × generator) cell:
  1. Construct PAIRED (g_rr, g_sr) arrays of length n_resamples via
     rr_sr_gaps_paired — g_rr[i] and g_sr[i] share real-side indices idx_a.
  2. Unpaired CI: bootstrap independent indices into g_rr and g_sr separately
     (the current bench behavior, but fed the paired arrays so the point
     estimate is identical to the paired CI's).
  3. Paired CI: bootstrap the SAME index vector into both arrays, preserving
     the per-i correlation that shared idx_a induces.

Same point estimate for both methods. The width comparison isolates the CI
estimator (paired vs unpaired) from estimand differences.

Output: runs/<ts>_paired_vs_unpaired/results.csv with columns
    bucket, generator, sim_point,
    ci_unpaired_lo, ci_unpaired_hi, ci_unpaired_width,
    ci_paired_lo,   ci_paired_hi,   ci_paired_width,
    width_reduction_pct,
    corr_rr_sr,     # diagnostic: per-i Pearson correlation
    seed, elapsed_construct_s, elapsed_paired_boot_s, elapsed_unpaired_boot_s
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from evaluation_framework.buckets import (             # noqa: E402
    BucketCFVC,
    BucketKurtosis,
    BucketMarginal,
    BucketTailRegime,
)
from evaluation_framework.fast_gaps import b2_gap, b3_gap  # noqa: E402
from evaluation_framework.io import load_corpus           # noqa: E402
from evaluation_framework.paths import (                  # noqa: E402
    generator_paths,
    real_corpus,
)
from evaluation_framework.protocol import (               # noqa: E402
    rr_sr_gaps_paired,
    similarity_with_ci_paired,
    similarity_with_ci_unpaired,
)
from evaluation_framework.samplers import PooledSampler  # noqa: E402
from scripts._runlog import (                              # noqa: E402
    create_run_dir,
    tee_stdout,
    write_config,
    write_env,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N            = 200
N_RESAMPLES  = 200
N_BOOTSTRAP  = 2000
SEED         = 42
GENERATORS   = ["AIL", "GARCH", "SFAGan"]

_b1 = BucketMarginal(tail_q=0.05, n_quantile_grid=1000)
_b4 = BucketKurtosis()
_b5 = BucketCFVC()
_b6 = BucketTailRegime()

BUCKETS = {
    "B1": _b1.compute_gap,
    "B2": b2_gap,
    "B3": b3_gap,
    "B4": _b4.compute_gap,
    "B5": _b5.compute_gap,
    "B6": _b6.compute_gap,
}


def cell_run(
    gap_fn,
    real: np.ndarray,
    syn:  np.ndarray,
    seed: int,
) -> dict:
    sampler = PooledSampler(len(real))

    # 1. Paired construction of (g_rr, g_sr)
    t0 = time.time()
    g_rr, g_sr = rr_sr_gaps_paired(
        gap_fn, real, syn,
        n_per_half=N, n_resamples=N_RESAMPLES,
        rng=np.random.default_rng(seed),
        sampler=sampler,
    )
    t_construct = time.time() - t0

    # 2. Unpaired bootstrap on the paired arrays — independent index draws
    t0 = time.time()
    sim_u, lo_u, hi_u = similarity_with_ci_unpaired(
        g_rr, g_sr, N_BOOTSTRAP,
        np.random.default_rng(seed + 1),
    )
    t_unpaired = time.time() - t0

    # 3. Paired bootstrap on the same arrays — same index per resample
    t0 = time.time()
    sim_p, lo_p, hi_p = similarity_with_ci_paired(
        g_rr, g_sr, N_BOOTSTRAP,
        np.random.default_rng(seed + 2),
    )
    t_paired = time.time() - t0

    assert abs(sim_u - sim_p) < 1e-12, (
        f"point estimates must match: unpaired={sim_u} paired={sim_p}"
    )
    width_u = hi_u - lo_u
    width_p = hi_p - lo_p
    reduction = 100.0 * (1.0 - width_p / width_u) if width_u > 0 else float("nan")

    # Diagnostic — per-i Pearson correlation between g_rr and g_sr
    if g_rr.std() > 0 and g_sr.std() > 0:
        corr = float(np.corrcoef(g_rr, g_sr)[0, 1])
    else:
        corr = float("nan")

    return {
        "sim_point":           sim_u,
        "ci_unpaired_lo":      lo_u,
        "ci_unpaired_hi":      hi_u,
        "ci_unpaired_width":   width_u,
        "ci_paired_lo":        lo_p,
        "ci_paired_hi":        hi_p,
        "ci_paired_width":     width_p,
        "width_reduction_pct": reduction,
        "corr_rr_sr":          corr,
        "elapsed_construct_s": round(t_construct, 2),
        "elapsed_paired_boot_s":  round(t_paired, 2),
        "elapsed_unpaired_boot_s": round(t_unpaired, 2),
    }


def main() -> int:
    run_dir = create_run_dir("paired_vs_unpaired")
    print(f"Run dir: {run_dir}")
    config = {
        "n":            N,
        "n_resamples":  N_RESAMPLES,
        "n_bootstrap":  N_BOOTSTRAP,
        "seed":         SEED,
        "buckets":      list(BUCKETS.keys()),
        "generators":   GENERATORS,
    }
    write_config(run_dir, config)
    write_env(run_dir)

    with tee_stdout(run_dir / "log.txt"):
        print(f"Run dir: {run_dir}")
        print(f"Config:  {config}")

        print("\nLoading corpora...")
        real = load_corpus(real_corpus())
        gen_paths = generator_paths()
        gens = {name: load_corpus(gen_paths[name]) for name in GENERATORS}
        print(f"  real    : {real.shape}")
        for n, a in gens.items():
            print(f"  {n:<6}  : {a.shape}")

        print("\nFitting B6 vol-regime thresholds on full real corpus...")
        _b6.fit(real)

        seq = np.random.SeedSequence(SEED)
        cell_seeds = seq.spawn(len(BUCKETS) * len(GENERATORS))
        cell_iter = iter(cell_seeds)

        rows: list[dict] = []
        for bkey, gap_fn in BUCKETS.items():
            print(f"\n=== {bkey} ===")
            for gname in GENERATORS:
                ss = next(cell_iter)
                seed = int(ss.generate_state(1, dtype=np.uint32)[0])
                t0 = time.time()
                cell = cell_run(gap_fn, real, gens[gname], seed)
                cell.update({"bucket": bkey, "generator": gname, "seed": seed,
                              "wall_s": round(time.time() - t0, 1)})
                rows.append(cell)
                print(
                    f"  [{gname:<6}] sim={cell['sim_point']:.4f}  "
                    f"unpaired w={cell['ci_unpaired_width']:.4f}  "
                    f"paired w={cell['ci_paired_width']:.4f}  "
                    f"red={cell['width_reduction_pct']:+.1f}%  "
                    f"corr(rr,sr)={cell['corr_rr_sr']:+.3f}  "
                    f"({cell['wall_s']}s)"
                )

        cols = [
            "bucket", "generator", "sim_point",
            "ci_unpaired_lo", "ci_unpaired_hi", "ci_unpaired_width",
            "ci_paired_lo",   "ci_paired_hi",   "ci_paired_width",
            "width_reduction_pct", "corr_rr_sr",
            "seed", "elapsed_construct_s",
            "elapsed_paired_boot_s", "elapsed_unpaired_boot_s", "wall_s",
        ]
        csv_path = run_dir / "results.csv"
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
