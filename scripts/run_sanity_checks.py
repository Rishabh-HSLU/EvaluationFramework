"""
Run every bucket's sanity_checks(real) on the bundled eval corpus and
record the underlying numerics (baseline g_rr, perturbed gap, ratio,
threshold) alongside the pass/fail bool returned by each check.

Strategy:
  - Monkey-patch each bucket's compute_gap with a logger before calling
    sanity_checks. The first call inside sanity_checks is always the
    baseline g_rr; subsequent calls are the named perturbations in the
    order they appear in the docstring. Zip the returned bool dict's
    keys with the logged calls (skipping the first) to map gap values
    to check names.
  - Each bucket's check thresholds are hardcoded below, mirroring the
    source. They are the comparison expressions inside sanity_checks
    (e.g. "g_x > 3 * g_rr").

Outputs (runs/<ts>_sanity_checks/):
  results.csv     bucket, check, gap_rr, gap_perturbed, ratio, threshold_kind,
                  threshold_factor, expected_relation, observed_passes
  log.txt         captured stdout
  config.yaml     params
  env.txt         uv pip list
  report.md       human-readable summary with failure highlights
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from evaluation_framework.buckets import (  # noqa: E402
    BucketCFVC,
    BucketKurtosis,
    BucketLeverageEffect,
    BucketMarginal,
    BucketNonlinearTemporal,
    BucketTailRegime,
)
from evaluation_framework.io import load_corpus  # noqa: E402
from evaluation_framework.paths import real_corpus  # noqa: E402
from scripts._runlog import (  # noqa: E402
    create_run_dir,
    tee_stdout,
    write_config,
    write_env,
)

# Hard-coded threshold definitions (mirror the comparisons in each
# bucket's sanity_checks source). `factor` is the multiplier of g_rr;
# `relation` is what the OBSERVED ratio must satisfy for the check to
# pass — '>' means we want the perturbation to be much larger than the
# noise floor, '<' means we want it close to / less than the floor.
THRESHOLDS: dict[str, dict[str, tuple[float, str]]] = {
    "B1_marginal": {
        "N1.1_tail_replacement": (3.0, ">"),
        # N1.2 is asymmetry-adaptive: > 2.0 if asym_ratio > 1.3, else >= 0
        "N1.2_skew_flip": (float("nan"), "adaptive (asymmetry-aware)"),
        "N1.3_temporal_shuffle": (2.0, "<"),
        "N1.4_bulk_perturbation": (2.0, "<"),
        "N1.5_scale_sensitivity": (3.0, ">"),
    },
    "B2_nonlinear_temporal": {
        "N2.1_shuffle": (3.0, ">"),
        "N2.2_iid_resample": (3.0, ">"),
        "N2.3_short_memory_garch": (2.0, ">"),
        # N2.4 compares to mean(g_shuffle, g_iid) — destructive baseline
        "N2.4_tail_replacement": (0.6, "<0.6·mean(g_shuf,g_iid)"),
        "N2.5_scale_invariance": (1.5, "<"),
    },
    "B3_leverage_effect": {
        "N3.1_time_reversal": (1.5, ">"),
        "N3.2_sign_symmetrization": (1.5, ">"),
        "N3.3_symmetric_garch": (1.5, ">"),
        "N3.4_shuffle": (1.5, ">"),
        "N3.5_scale_invariance": (1.5, "<"),
    },
    "B4_kurtosis": {
        # Temporal checks (N4.1, N4.2, N4.5) use a temp bucket dropping h=1
        # → comparison vs g_rr_temp (temporal-horizon baseline)
        "N4.1_iid_resample": (2.0, "> (temp baseline)"),
        "N4.2_shuffle": (2.0, "> (temp baseline)"),
        "N4.3_scale_invariance": (1.5, "< (full baseline)"),
        "N4.4_tail_replacement": (float("nan"), "<g_iid"),
        "N4.5_block_resample": (1.5, "> (temp baseline)"),
    },
    "B5_cfvc": {
        "S5.1_independent_scale": (3.0, ">"),
        "S5.2_baseline_calibration": (float("nan"), "s5 in (0.2, 0.8)"),
        "S5.3_within_day_shuffle": (8.0, "<"),
    },
    "B6_tail_regime": {
        "S6.1_homogeneous_tail": (2.0, ">"),
        "S6.2_baseline_calibration": (float("nan"), "s6 in (0.2, 0.8)"),
        "S6.3_reversed_tail_curve": (1.5, ">"),
    },
}


def instrument(bucket) -> list[dict]:
    """Wrap bucket.compute_gap with a logger. Returns the log list."""
    orig = bucket.compute_gap
    log: list[dict] = []

    def wrapper(real, syn):
        result = orig(real, syn)
        log.append(
            {
                "real_shape": tuple(real.shape),
                "syn_shape": tuple(syn.shape),
                "gap": float(result),
            }
        )
        return result

    bucket.compute_gap = wrapper
    return log


def run_bucket(bucket, real: np.ndarray) -> list[dict]:
    """Run one bucket's sanity_checks and return per-check records."""
    log = instrument(bucket)
    t0 = time.time()
    results = bucket.sanity_checks(real)
    elapsed = time.time() - t0

    if not log:
        raise RuntimeError(f"{bucket.name}: no compute_gap calls captured")
    baseline = log[0]["gap"]
    check_calls = log[1:]
    if len(check_calls) != len(results):
        # B6's S6.2 makes an extra duplicate compute_gap call for s6 ratio,
        # so it actually has one MORE call than checks. The fix already
        # works because we zip(results.keys(), check_calls) and the order
        # matches. But assert in case order changes.
        print(
            f"  WARN {bucket.name}: {len(check_calls)} perturbation calls "
            f"vs {len(results)} checks — taking first {len(results)}."
        )
        check_calls = check_calls[: len(results)]

    thresholds = THRESHOLDS[bucket.name]

    rows: list[dict] = []
    for (check_name, passed), call in zip(results.items(), check_calls, strict=True):
        gap = call["gap"]
        ratio = gap / baseline if baseline > 0 else float("nan")
        thr = thresholds.get(check_name, (float("nan"), "?"))
        factor, relation = thr
        rows.append(
            {
                "bucket": bucket.name,
                "check": check_name,
                "gap_rr": baseline,
                "gap_perturbed": gap,
                "ratio": ratio,
                "threshold_factor": factor,
                "expected_relation": relation,
                "observed_passes": bool(passed),
                "elapsed_bucket_s": round(elapsed, 2),
            }
        )
    return rows


def main() -> int:
    run_dir = create_run_dir("sanity_checks")
    print(f"Run dir: {run_dir}")
    write_config(
        run_dir,
        {
            "corpus": str(real_corpus()),
            "buckets": ["B1", "B2", "B3", "B4", "B5", "B6"],
            "rng_seed": 0,  # all sanity_checks methods use rng = default_rng(0)
        },
    )
    write_env(run_dir)

    with tee_stdout(run_dir / "log.txt"):
        print(f"Loading real corpus: {real_corpus()}")
        real = load_corpus(real_corpus())
        print(f"  shape: {real.shape}  mean={real.mean():.4f} std={real.std():.4f}")

        buckets = [
            BucketMarginal(tail_q=0.05, n_quantile_grid=1000),
            BucketNonlinearTemporal(k_min=60, k_max=390),
            BucketLeverageEffect(k_min=1, k_max=390),
            BucketKurtosis(),
            BucketCFVC(),
            BucketTailRegime(),
        ]

        # B6 needs fit() before any compute_gap call. Sanity_checks calls
        # it lazily if thresholds aren't set, but we fit explicitly here
        # to keep the noise-floor call consistent with the main benchmark.
        print("\nFitting B6 vol-regime thresholds on full real corpus...")
        buckets[-1].fit(real)

        all_rows: list[dict] = []
        for b in buckets:
            print(f"\n=== {b.name} ===")
            rows = run_bucket(b, real)
            for r in rows:
                rel = r["expected_relation"]
                if rel in ("<g_iid", "s5 in (0.2, 0.8)", "s6 in (0.2, 0.8)"):
                    rel_str = rel
                else:
                    rel_str = f"ratio {rel} {r['threshold_factor']:.1f}"
                mark = "PASS" if r["observed_passes"] else "FAIL"
                print(
                    f"  [{mark}] {r['check']:<32}  g_rr={r['gap_rr']:.4g}  "
                    f"gap={r['gap_perturbed']:.4g}  ratio={r['ratio']:.3f}  "
                    f"({rel_str})"
                )
            all_rows.extend(rows)

        # ----------------- CSV -----------------
        csv_path = run_dir / "results.csv"
        cols = [
            "bucket",
            "check",
            "gap_rr",
            "gap_perturbed",
            "ratio",
            "threshold_factor",
            "expected_relation",
            "observed_passes",
            "elapsed_bucket_s",
        ]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nResults: {csv_path}")

        # ----------------- Markdown report -----------------
        report_path = run_dir / "report.md"
        with report_path.open("w") as f:
            f.write("# Bucket sanity-check report\n\n")
            f.write(f"Corpus: `{real_corpus()}`  (shape `{real.shape}`)\n\n")
            n_pass = sum(1 for r in all_rows if r["observed_passes"])
            n_total = len(all_rows)
            f.write(f"**Overall: {n_pass}/{n_total} checks passed.**\n\n")

            for bname in [b.name for b in buckets]:
                bucket_rows = [r for r in all_rows if r["bucket"] == bname]
                bp = sum(1 for r in bucket_rows if r["observed_passes"])
                f.write(f"## {bname} ({bp}/{len(bucket_rows)} pass)\n\n")
                f.write("| check | gap_rr | gap_perturbed | ratio | expected | pass |\n")
                f.write("|---|---|---|---|---|---|\n")
                for r in bucket_rows:
                    rel = r["expected_relation"]
                    if rel in ("<g_iid", "s5 in (0.2, 0.8)", "s6 in (0.2, 0.8)"):
                        exp = rel
                    else:
                        exp = f"{rel} {r['threshold_factor']:.1f}"
                    mark = "✓" if r["observed_passes"] else "✗"
                    f.write(
                        f"| {r['check']} | {r['gap_rr']:.4g} | "
                        f"{r['gap_perturbed']:.4g} | {r['ratio']:.3f} | "
                        f"{exp} | {mark} |\n"
                    )
                f.write("\n")

            # Failure section
            failures = [r for r in all_rows if not r["observed_passes"]]
            if failures:
                f.write(f"## Failures ({len(failures)})\n\n")
                for r in failures:
                    f.write(
                        f"- **{r['bucket']} / {r['check']}** — "
                        f"g_rr={r['gap_rr']:.4g}, gap={r['gap_perturbed']:.4g}, "
                        f"ratio={r['ratio']:.3f}, expected "
                        f"{r['expected_relation']} {r['threshold_factor']}\n"
                    )
            else:
                f.write("## Failures\n\nNone.\n")

        print(f"Report : {report_path}")

        # ----------------- Console summary -----------------
        print("\n\n" + "=" * 60)
        print(f"SUMMARY: {n_pass}/{n_total} checks passed")
        print("=" * 60)
        if failures:
            print(f"\nFailures ({len(failures)}):")
            for r in failures:
                print(
                    f"  {r['bucket']:<22}  {r['check']:<32}  "
                    f"ratio={r['ratio']:.3f}  expected "
                    f"{r['expected_relation']} {r['threshold_factor']}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
