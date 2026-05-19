#!/usr/bin/env python3
"""
Six-bucket benchmark: rank generators by similarity to the real eval corpus.

    uv run python bench.py

Uses paired-bootstrap CIs by default — g_rr[i] and g_sr[i] are constructed
to share the real-side index draw idx_a, and the bootstrap resamples a
single index vector into both arrays per iteration. See protocol.py.
"""

from __future__ import annotations

import time

import numpy as np

from evaluation_framework.buckets import (
    BucketCFVC,
    BucketKurtosis,
    BucketMarginal,
    BucketTailRegime,
)
from evaluation_framework.canonical import format_corpus_line, load_manifest
from evaluation_framework.fast_gaps import b2_gap, b3_gap, verify_fft_vs_bucket
from evaluation_framework.io import load_corpus
from evaluation_framework.paths import generator_paths, output_dir, real_corpus
from evaluation_framework.protocol import (
    rr_sr_gaps_paired,
    similarity_with_ci_paired,
)
from evaluation_framework.samplers import PooledSampler

N = 200
N_RESAMPLES = 200
N_BOOTSTRAP = 2000
SEED = 42

_b1 = BucketMarginal(tail_q=0.05, n_quantile_grid=1000)
_b4 = BucketKurtosis()
_b5 = BucketCFVC()
_b6 = BucketTailRegime()


def run_benchmark() -> dict[str, dict[str, tuple[float, float, float]]]:
    """Compute the full benchmark table; returns {bucket: {gen: (sim, lo, hi)}}.

    Uses paired-bootstrap CIs. Per-cell seeding via SeedSequence(SEED).spawn()
    matches scripts/run_paired_vs_unpaired.py for bit-level reproducibility.
    """
    ref = load_manifest(output_dir())
    print(f"Benchmark manifest: T={ref.window_len}, ref_std={ref.std:.4f}, N={ref.n_paths}")

    print("\nLoading corpora...")
    real = load_corpus(real_corpus(), ref, "real")
    paths = generator_paths()
    gens: dict[str, np.ndarray] = {}
    for name, p in paths.items():
        if not p.exists():
            print(f"  SKIP {name}: missing {p.name}")
            continue
        gens[name] = load_corpus(p, ref, name)

    print(format_corpus_line("real", {"n": len(real), "mean": real.mean(), "std": real.std()}))
    for name, arr in gens.items():
        print(format_corpus_line(name, {"n": len(arr), "mean": arr.mean(), "std": arr.std()}))

    if not gens:
        raise SystemExit("No synthetic corpora found under data/output_data/")

    first = next(iter(gens.values()))
    print("\nVerifying FFT B2/B3 vs bucket implementations (20 paths)...")
    verify_fft_vs_bucket(real, first)

    print("\nFitting B6 vol regime thresholds on full real corpus...")
    _b6.fit(real)

    buckets = {
        "B1": ("B1 tail-W1 (q=0.05)", _b1.compute_gap),
        "B2": ("B2 ACF |r| lags 60-390", b2_gap),
        "B3": ("B3 leverage lags 1-390", b3_gap),
        "B4": ("B4 scale-weighted L-kurtosis", _b4.compute_gap),
        "B5": ("B5 CFVC Frobenius", _b5.compute_gap),
        "B6": ("B6 conditional GPD tail regime", _b6.compute_gap),
    }

    sampler = PooledSampler(len(real))

    # Per-cell SeedSequence — same scheme as scripts/run_paired_vs_unpaired.py
    # so the published paired-bootstrap CIs are reproducible at the cell level.
    seq        = np.random.SeedSequence(SEED)
    cell_seeds = seq.spawn(len(buckets) * len(gens))
    cell_iter  = iter(cell_seeds)

    table: dict[str, dict[str, tuple[float, float, float]]] = {}
    for bkey, (bdesc, gap_fn) in buckets.items():
        print(f"\n=== {bkey}: {bdesc} ===")
        table[bkey] = {}
        for gname, syn in gens.items():
            ss   = next(cell_iter)
            seed = int(ss.generate_state(1, dtype=np.uint32)[0])
            t0   = time.time()
            g_rr, g_sr = rr_sr_gaps_paired(
                gap_fn, real, syn,
                n_per_half=N, n_resamples=N_RESAMPLES,
                rng=np.random.default_rng(seed),
                sampler=sampler,
            )
            sim, lo, hi = similarity_with_ci_paired(
                g_rr, g_sr, N_BOOTSTRAP,
                np.random.default_rng(seed + 2),
            )
            elapsed = time.time() - t0
            cv_rr   = g_rr.std() / g_rr.mean() if g_rr.mean() else float("nan")
            print(
                f"  [{gname:<6}] g_rr_mean={g_rr.mean():.6f} CV={cv_rr:.3f}  "
                f"g_sr_mean={g_sr.mean():.6f}  "
                f"sim={sim:.4f}  paired CI=[{lo:.4f}, {hi:.4f}]  "
                f"w={hi - lo:.4f}  ({elapsed:.1f}s)"
            )
            table[bkey][gname] = (sim, lo, hi)

    return table


def main() -> None:
    table = run_benchmark()
    names = sorted({g for d in table.values() for g in d.keys()},
                   key=["AIL", "GARCH", "SFAGan", "SBBTS"].index)
    _print_table(table, names)


def _print_table(
    table: dict[str, dict[str, tuple[float, float, float]]],
    names: list[str],
) -> None:
    w = 78
    print("\n\n" + "=" * w)
    print(f"BENCHMARK — N={N}, resamples={N_RESAMPLES}, "
          f"paired-bootstrap CIs (B={N_BOOTSTRAP})")
    print("=" * w)
    header = f"{'Bucket':<8} | " + " | ".join(f"{g:<22}" for g in names)
    print(header)
    print("-" * len(header))
    for bkey in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        row = f"{bkey:<8} |"
        for g in names:
            sim, lo, hi = table[bkey][g]
            row += f" {sim:.4f} [{lo:.4f},{hi:.4f}]{'':<6} |"
        print(row.rstrip("|").rstrip())
    print("-" * len(header))

    for label, fn in [
        ("geometric", lambda s: float(np.exp(np.mean(np.log(s))))),
        ("arithmetic", np.mean),
    ]:
        row = f"Composite ({label}) |"
        for g in names:
            sims = [max(table[b][g][0], 1e-6) for b in ["B1", "B2", "B3", "B4", "B5", "B6"]]
            row += f" {fn(sims):.4f}{' ' * 18}|"
        print(row.rstrip("|").rstrip())

    ranked = sorted(
        names,
        key=lambda g: float(np.mean([table[b][g][0] for b in ["B1", "B2", "B3", "B4", "B5", "B6"]])),
        reverse=True,
    )
    print("\nRank (arithmetic composite, higher = better):")
    for i, g in enumerate(ranked, 1):
        comp = np.mean([table[b][g][0] for b in ["B1", "B2", "B3", "B4", "B5", "B6"]])
        print(f"  {i}. {g:<8}  composite={comp:.4f}")

    print("\ns_b = mean(g_rr) / (mean(g_rr) + mean(g_sr)); 1 = indistinguishable from real–real")


if __name__ == "__main__":
    main()
