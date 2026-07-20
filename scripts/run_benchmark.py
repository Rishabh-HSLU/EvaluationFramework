"""Run the full fineval benchmark: Real vs {AIL, GBM, MSV}.

End-to-end flow:

1. Load and preprocess the curated real and synthetic panels.
2. Run the matched-N ticker estimator for the point estimates.
3. Optionally run an outer ticker-corpus bootstrap with
   ``--n-outer-resamples N``. This produces percentile 95% confidence
   intervals conditional on the preprocessed and generated panels.
4. Optionally run independent inner seeds with ``--n-mc-repeats N``.
   This reports Monte Carlo mean, standard deviation, quantiles and
   range as a numerical-stability analysis; these are not confidence
   intervals.
5. Save summary and replicate-level CSV files and append the run to the
   manifest. Canonical files are refreshed only when explicitly
   requested with ``--update-canonical``; this requires an outer
   bootstrap so canonical results cannot contain mislabeled intervals.

Examples from the repository root:

    uv run python -m scripts.run_benchmark --n-resamples 20
    uv run python -m scripts.run_benchmark --n-outer-resamples 200
    uv run python -m scripts.run_benchmark --n-mc-repeats 10
    uv run python -m scripts.run_benchmark \
        --n-outer-resamples 500 --n-mc-repeats 20 --update-canonical
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from fineval.bootstrap import MatchedTickerBootstrap
from fineval.config import (
    M1_N_GRID,
    M1_TAIL_ALPHA,
    M1_TAIL_LAMBDA,
    M2_LAG_MAX,
    M2_LAG_MIN,
    M4_MIN_OBS,
    M4_SCALES,
    N_REGIME_QUINTILES,
    REGIME_WEIGHTS,
    ROLLING_VOL_MIN_PERIODS,
    ROLLING_VOL_WINDOW,
    SEED,
    TAIL_QUANTILE,
)
from fineval.data import CuratedParquetLoader, GBMBaselineLoader, MSVBaselineLoader
from fineval.metrics import (
    AggregationalGaussianity,
    RegimeConditionalTails,
    UnconditionalHeavyTails,
    VolatilityClustering,
)
from fineval.preprocessing import PreprocessingPipeline

CURATED_DIR = Path(__file__).resolve().parent.parent / "data" / "curated"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METRIC_LABELS = {
    "M1": "Unconditional heavy tails",
    "M2": "Volatility clustering",
    "M4": "Aggregational Gaussianity",
    "M6": "Regime-conditional tails",
}


def build_metrics() -> list:
    """All implemented metrics, parametrized from fineval.config."""
    return [
        UnconditionalHeavyTails(
            name="M1",
            n_grid=M1_N_GRID,
            tail_alpha=M1_TAIL_ALPHA,
            tail_lambda=M1_TAIL_LAMBDA,
        ),
        VolatilityClustering(name="M2", lag_min=M2_LAG_MIN, lag_max=M2_LAG_MAX),
        AggregationalGaussianity(name="M4", scales=M4_SCALES, min_obs=M4_MIN_OBS),
        RegimeConditionalTails(
            name="M6",
            window=ROLLING_VOL_WINDOW,
            min_periods=ROLLING_VOL_MIN_PERIODS,
            n_regimes=N_REGIME_QUINTILES,
            tail_quantile=TAIL_QUANTILE,
            regime_weights=REGIME_WEIGHTS,
        ),
    ]


def format_table(results: pd.DataFrame) -> str:
    """Render point estimates, with outer CIs when available."""
    generators = list(dict.fromkeys(results["generator"]))
    header = "| Metric | Stylized fact | " + " | ".join(generators) + " |"
    rule = "|---|---|" + "---|" * len(generators)
    lines = [header, rule]
    has_ci = {"ci_low", "ci_high"}.issubset(results.columns)

    for metric in list(dict.fromkeys(results["metric"])):
        cells = []
        for gen in generators:
            row = results[(results["metric"] == metric) & (results["generator"] == gen)].iloc[0]
            if has_ci and pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
                cells.append(f"{row['score']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
            else:
                cells.append(f"{row['score']:.3f}")
        label = METRIC_LABELS.get(metric, metric)
        lines.append(f"| {metric} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_aggregate_table(aggregate: pd.DataFrame) -> str:
    """Render aggregate G point estimates, with outer CIs when available."""
    has_ci = {"ci_low", "ci_high"}.issubset(aggregate.columns)
    value_header = "G [95% outer-bootstrap CI]" if has_ci else "G"
    lines = [f"| Generator | {value_header} | metrics used |", "|---|---|---|"]
    for _, row in aggregate.iterrows():
        value = f"{row['G']:.3f}"
        if has_ci and pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            value += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        lines.append(
            f"| {row['generator']} | {value} | {int(row['k_used'])}/{int(row['k_total'])} |"
        )
    return "\n".join(lines)


def format_mc_table(summary: pd.DataFrame, value_column: str) -> str:
    """Render independent-seed Monte Carlo stability summaries."""
    if value_column == "score":
        lines = [
            "| Metric | Generator | mean ± SD | min–max |",
            "|---|---|---|---|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['metric']} | {row['generator']} | "
                f"{row['mc_mean']:.3f} ± {row['mc_sd']:.3f} | "
                f"{row['mc_min']:.3f}–{row['mc_max']:.3f} |"
            )
    else:
        lines = [
            "| Generator | mean G ± SD | min–max |",
            "|---|---|---|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['generator']} | {row['mc_mean']:.3f} ± {row['mc_sd']:.3f} | "
                f"{row['mc_min']:.3f}–{row['mc_max']:.3f} |"
            )
    return "\n".join(lines)


def _git_commit() -> str:
    """Short commit hash of the repository at run time, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _append_manifest(record: dict) -> int:
    """Append one run while safely extending an older manifest schema.

    The manifest is the append-only registry of every benchmark run,
    failed ones included. The assigned run number is one past the
    highest already recorded.

    Returns:
        The run number assigned to this record.
    """
    manifest_path = RESULTS_DIR / "runs_manifest.csv"
    has_rows = manifest_path.exists() and manifest_path.stat().st_size > 0
    if not has_rows:
        run_number = 1
        pd.DataFrame([{"run_number": run_number, **record}]).to_csv(manifest_path, index=False)
        return run_number

    previous = pd.read_csv(manifest_path)
    run_number = int(previous["run_number"].max()) + 1 if len(previous) else 1
    current = pd.DataFrame([{"run_number": run_number, **record}])
    columns = list(previous.columns) + [
        column for column in current.columns if column not in previous.columns
    ]
    pd.concat(
        [previous.reindex(columns=columns), current.reindex(columns=columns)],
        ignore_index=True,
    ).to_csv(manifest_path, index=False)
    return run_number


def _run_tag(args: argparse.Namespace, stamp: str) -> str:
    """Build a collision-resistant filename tag containing all run controls."""
    return (
        f"B{args.n_resamples}_m{args.tickers_per_draw}_seed{args.seed}"
        f"_O{args.n_outer_resamples}_MC{args.n_mc_repeats}_{stamp}"
    )


def _execute(args: argparse.Namespace, stamp: str, record: dict) -> None:
    """Load, preprocess, evaluate and save one benchmark run."""
    print("Loading curated datasets...")
    real = CuratedParquetLoader(
        parquet_path=str(CURATED_DIR / "real_prices.parquet"),
        name="Real",
        is_synthetic=False,
    ).load()
    ail = CuratedParquetLoader(
        parquet_path=str(CURATED_DIR / "ail_prices.parquet"),
        name="AIL",
        is_synthetic=True,
    ).load()
    gbm = GBMBaselineLoader(parquet_path=str(CURATED_DIR / "gbm_prices.parquet")).load()
    msv = MSVBaselineLoader(parquet_path=str(CURATED_DIR / "msv_prices.parquet")).load()

    print("Preprocessing (log returns, overnight mask, conditional FFF)...")
    deseas_real = None
    synthetics = {}
    for dataset in (ail, gbm, msv):
        pipeline = PreprocessingPipeline().run(real.prices, dataset.prices)
        synthetics[dataset.name] = pipeline.deseas_synthetic
        if deseas_real is None:
            deseas_real = pipeline.deseas_real

    engine = MatchedTickerBootstrap(
        metrics=build_metrics(),
        n_resamples=args.n_resamples,
        tickers_per_draw=args.tickers_per_draw,
        seed=args.seed,
    )

    print("Running matched ticker point estimator...")
    point_results = engine.run(deseas_real, synthetics)
    point_aggregate = engine.compute_aggregate()
    results = point_results.copy()
    aggregate = point_aggregate.copy()
    tag = _run_tag(args, stamp)

    if args.n_outer_resamples:
        outer = engine.run_outer_bootstrap(
            deseas_real,
            synthetics,
            n_outer_resamples=args.n_outer_resamples,
        )
        results = results.merge(
            outer.metric_summary,
            on=["metric", "generator"],
            how="left",
            validate="one_to_one",
        )
        aggregate = aggregate.merge(
            outer.aggregate_summary,
            on="generator",
            how="left",
            validate="one_to_one",
        )

        outer_metric_path = RESULTS_DIR / f"outer_metric_replicates_{tag}.csv"
        outer_aggregate_path = RESULTS_DIR / f"outer_aggregate_replicates_{tag}.csv"
        outer.metric_replicates.to_csv(outer_metric_path, index=False)
        outer.aggregate_replicates.to_csv(outer_aggregate_path, index=False)
        record["outer_metric_replicates_csv"] = outer_metric_path.name
        record["outer_aggregate_replicates_csv"] = outer_aggregate_path.name

    mc = None
    if args.n_mc_repeats:
        mc = engine.run_monte_carlo_stability(
            deseas_real,
            synthetics,
            n_repeats=args.n_mc_repeats,
            base_metric_result=point_results,
            base_aggregate_result=point_aggregate,
        )
        results = results.merge(
            mc.metric_summary,
            on=["metric", "generator"],
            how="left",
            validate="one_to_one",
        )
        aggregate = aggregate.merge(
            mc.aggregate_summary,
            on="generator",
            how="left",
            validate="one_to_one",
        )

        mc_metric_path = RESULTS_DIR / f"mc_metric_replicates_{tag}.csv"
        mc_aggregate_path = RESULTS_DIR / f"mc_aggregate_replicates_{tag}.csv"
        mc.metric_replicates.to_csv(mc_metric_path, index=False)
        mc.aggregate_replicates.to_csv(mc_aggregate_path, index=False)
        record["mc_metric_replicates_csv"] = mc_metric_path.name
        record["mc_aggregate_replicates_csv"] = mc_aggregate_path.name

    csv_path = RESULTS_DIR / f"benchmark_{tag}.csv"
    agg_path = RESULTS_DIR / f"aggregate_{tag}.csv"
    results.to_csv(csv_path, index=False)
    aggregate.to_csv(agg_path, index=False)
    record["results_csv"] = csv_path.name
    record["aggregate_csv"] = agg_path.name

    if args.update_canonical:
        results.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
        aggregate.to_csv(RESULTS_DIR / "aggregate_results.csv", index=False)
        record["canonical_updated"] = True

    record["status"] = "completed"

    interval_text = (
        f", {args.n_outer_resamples} outer corpus replicates"
        if args.n_outer_resamples
        else ", no corpus interval"
    )
    print(
        f"\nBenchmark (B={args.n_resamples}, {args.tickers_per_draw} tickers/draw, "
        f"seed={args.seed}{interval_text}):\n"
    )
    print(format_table(results))

    print(
        "\nAggregate fidelity G = exp(mean_k ln r_k), "
        "r_k = mean(g_sr)/mean(g_rr);\n"
        "G ≈ 1 is noise-floor parity, larger G = larger aggregate gap:\n"
    )
    print(format_aggregate_table(aggregate))

    if mc is not None:
        print(
            "\nMonte Carlo numerical stability across independent inner seeds "
            "(not a confidence interval):\n"
        )
        print(format_mc_table(mc.metric_summary, value_column="score"))
        print("\nAggregate Monte Carlo stability:\n")
        print(format_mc_table(mc.aggregate_summary, value_column="G"))

    print(f"\nTidy results saved to: {csv_path}")
    print(f"Aggregate results saved to: {agg_path}")
    if args.update_canonical:
        print(f"Canonical results updated: {RESULTS_DIR / 'benchmark_results.csv'}")
        print(f"Canonical aggregate updated: {RESULTS_DIR / 'aggregate_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fineval benchmark table.")
    parser.add_argument("--n-resamples", type=int, default=100, help="inner matched draws B")
    parser.add_argument("--tickers-per-draw", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-outer-resamples",
        type=int,
        default=0,
        help="complete corpus bootstrap replicates; 0 disables corpus CIs",
    )
    parser.add_argument(
        "--n-mc-repeats",
        type=int,
        default=0,
        help="independent inner-seed repeats; 0 disables stability analysis",
    )
    parser.add_argument(
        "--update-canonical",
        action="store_true",
        help="refresh canonical CSVs; requires an enabled outer bootstrap",
    )
    args = parser.parse_args()

    if args.n_resamples < 1:
        parser.error("--n-resamples must be at least 1")
    if args.tickers_per_draw < 1:
        parser.error("--tickers-per-draw must be at least 1")
    if args.n_outer_resamples == 1 or args.n_outer_resamples < 0:
        parser.error("--n-outer-resamples must be 0 or at least 2")
    if args.n_mc_repeats == 1 or args.n_mc_repeats < 0:
        parser.error("--n-mc-repeats must be 0 or at least 2")
    if args.update_canonical and args.n_outer_resamples < 2:
        parser.error("--update-canonical requires --n-outer-resamples >= 2")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    record = {
        "run_id": stamp,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "n_resamples": args.n_resamples,
        "tickers_per_draw": args.tickers_per_draw,
        "seed": args.seed,
        "n_outer_resamples": args.n_outer_resamples,
        "n_mc_repeats": args.n_mc_repeats,
        "git_commit": _git_commit(),
        "status": "failed",
        "duration_seconds": 0.0,
        "results_csv": "",
        "aggregate_csv": "",
        "outer_metric_replicates_csv": "",
        "outer_aggregate_replicates_csv": "",
        "mc_metric_replicates_csv": "",
        "mc_aggregate_replicates_csv": "",
        "canonical_updated": False,
        "error": "",
    }
    t0 = time.monotonic()
    try:
        _execute(args, stamp, record)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["duration_seconds"] = round(time.monotonic() - t0, 1)
        run_number = _append_manifest(record)
        print(
            f"Run #{run_number} ({record['status']}) recorded in "
            f"{RESULTS_DIR / 'runs_manifest.csv'}"
        )


if __name__ == "__main__":
    main()
