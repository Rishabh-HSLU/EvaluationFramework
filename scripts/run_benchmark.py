"""Run the full fineval benchmark: Real vs {AIL, GBM, MSV}.

End-to-end flow:

1. Load and preprocess the curated real and synthetic panels.
2. Run the matched-N ticker estimator for the point estimates.
3. Optionally run an outer ticker-corpus bootstrap with
   ``--n-outer-resamples N``. At least 40 replicates are required before
   the 2.5th and 97.5th percentiles are labelled a nominal 95% interval;
   substantially more (typically 500-1000+) are recommended for final
   reporting.
4. Optionally run independent inner seeds with ``--n-mc-repeats N``.
   This reports Monte Carlo mean, standard deviation, quantiles and
   range as a numerical-stability analysis; these are not confidence
   intervals.
5. Parallelize the point estimator, outer replicates and Monte Carlo
   repeats under one bounded worker budget. All random inputs are
   pre-generated, so worker counts and completion order do not change
   results.
6. Show one script-wide progress bar plus detailed bars for the active
   estimator, print stage-level logs and diagnostics, save summary and
   replicate-level CSV files, and append the run to the manifest.

Examples from the repository root:

    uv run python -m scripts.run_benchmark --n-resamples 20
    uv run python -m scripts.run_benchmark --n-outer-resamples 200
    uv run python -m scripts.run_benchmark --n-mc-repeats 10
    uv run python -m scripts.run_benchmark --n-jobs 12
    uv run python -m scripts.run_benchmark \
        --n-outer-resamples 1000 --n-mc-repeats 20 --update-canonical
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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

# Fewer than 40 observations cannot empirically resolve both 2.5% tails
# without interpolation. This is a minimum labelling threshold, not a
# recommendation for final inference; 500-1000+ is preferable.
MIN_OUTER_REPLICATES_FOR_CI = 40
RECOMMENDED_OUTER_REPLICATES = 1000

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
    """Render point estimates, with outer percentile limits when available."""
    generators = list(dict.fromkeys(results["generator"]))
    header = "| Metric | Stylized fact | " + " | ".join(generators) + " |"
    rule = "|---|---|" + "---|" * len(generators)
    lines = [header, rule]
    has_interval = {"ci_low", "ci_high"}.issubset(results.columns)

    for metric in list(dict.fromkeys(results["metric"])):
        cells = []
        for gen in generators:
            row = results[(results["metric"] == metric) & (results["generator"] == gen)].iloc[0]
            if has_interval and pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
                cells.append(f"{row['score']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
            else:
                cells.append(f"{row['score']:.3f}")
        label = METRIC_LABELS.get(metric, metric)
        lines.append(f"| {metric} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_aggregate_table(aggregate: pd.DataFrame, interval_label: str | None) -> str:
    """Render aggregate G point estimates and optional outer percentiles."""
    has_interval = {"ci_low", "ci_high"}.issubset(aggregate.columns)
    value_header = f"G [{interval_label}]" if has_interval and interval_label else "G"
    lines = [f"| Generator | {value_header} | metrics used |", "|---|---|---|"]
    for _, row in aggregate.iterrows():
        value = f"{row['G']:.3f}"
        if has_interval and pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            value += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        lines.append(
            f"| {row['generator']} | {value} | {int(row['k_used'])}/{int(row['k_total'])} |"
        )
    return "\n".join(lines)


def format_mc_table(summary: pd.DataFrame, value_column: str) -> str:
    """Render independent-seed Monte Carlo stability summaries."""
    if value_column == "score":
        lines = [
            "| Metric | Generator | mean ± SD | min-max |",
            "|---|---|---|---|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['metric']} | {row['generator']} | "
                f"{row['mc_mean']:.3f} ± {row['mc_sd']:.3f} | "
                f"{row['mc_min']:.3f}-{row['mc_max']:.3f} |"
            )
    else:
        lines = [
            "| Generator | mean G ± SD | min-max |",
            "|---|---|---|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['generator']} | {row['mc_mean']:.3f} ± "
                f"{row['mc_sd']:.3f} | {row['mc_min']:.3f}-{row['mc_max']:.3f} |"
            )
    return "\n".join(lines)


def _log(message: str) -> None:
    """Write a timestamped message without corrupting active progress bars."""
    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def _dataset_shape(dataset) -> str:
    """Format the loaded price-panel shape for logs."""
    rows, columns = dataset.prices.shape
    return f"{rows:,} timestamps × {columns:,} paths"


def _interval_label(n_outer_resamples: int) -> str | None:
    """Return the statistically honest label for outer percentiles."""
    if n_outer_resamples == 0:
        return None
    if n_outer_resamples < MIN_OUTER_REPLICATES_FOR_CI:
        return f"diagnostic outer 2.5%-97.5% quantiles; O={n_outer_resamples}, not a 95% CI"
    return f"95% outer-bootstrap percentile CI; O={n_outer_resamples}"


def _attach_outer_diagnostics(
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add bootstrap bias and point-inside-percentile diagnostics."""
    results = results.copy()
    aggregate = aggregate.copy()

    results["outer_bias"] = results["outer_mean"] - results["score"]
    results["point_in_outer_interval"] = results["score"].ge(results["ci_low"]) & results[
        "score"
    ].le(results["ci_high"])
    aggregate["outer_bias"] = aggregate["outer_mean"] - aggregate["G"]
    aggregate["point_in_outer_interval"] = aggregate["G"].ge(aggregate["ci_low"]) & aggregate[
        "G"
    ].le(aggregate["ci_high"])
    return results, aggregate


def _log_outer_diagnostics(
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    n_outer_resamples: int,
) -> None:
    """Explain interval resolution and point-estimate discrepancies."""
    if n_outer_resamples < MIN_OUTER_REPLICATES_FOR_CI:
        _log(
            "WARNING: only "
            f"{n_outer_resamples} outer replicates were requested. The reported "
            "percentiles are smoke-test diagnostics, not a usable 95% confidence "
            f"interval. Use at least {MIN_OUTER_REPLICATES_FOR_CI} to resolve the "
            "2.5% tails and preferably "
            f"{RECOMMENDED_OUTER_REPLICATES}+ for final reporting."
        )

    outside = results.loc[
        ~results["point_in_outer_interval"],
        ["metric", "generator", "score", "ci_low", "ci_high", "outer_bias"],
    ]
    aggregate_outside = aggregate.loc[
        ~aggregate["point_in_outer_interval"],
        ["generator", "G", "ci_low", "ci_high", "outer_bias"],
    ]

    if outside.empty and aggregate_outside.empty:
        _log("Outer-bootstrap diagnostic: every point estimate lies within its percentile limits.")
        return

    _log(
        "Outer-bootstrap diagnostic: "
        f"{len(outside)}/{len(results)} metric-generator point estimates and "
        f"{len(aggregate_outside)}/{len(aggregate)} aggregate estimates lie outside "
        "their bootstrap percentile limits. This is possible for a percentile "
        "bootstrap and is not by itself a code error."
    )
    if n_outer_resamples < MIN_OUTER_REPLICATES_FOR_CI:
        _log(
            "With so few outer replicates this behavior is expected: the two tail "
            "percentiles are interpolated almost entirely from the observed outer "
            "scores. Within each metric, all generators also share the same "
            "outer-resampled real noise floor g_rr, so one unusual real corpus can "
            "move every generator score in the same direction."
        )

    for row in outside.itertuples(index=False):
        _log(
            f"  {row.metric}/{row.generator}: point={row.score:.4f}, "
            f"outer=[{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"outer_mean-point={row.outer_bias:+.4f}"
        )
    for row in aggregate_outside.itertuples(index=False):
        _log(
            f"  aggregate/{row.generator}: point={row.G:.4f}, "
            f"outer=[{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"outer_mean-point={row.outer_bias:+.4f}"
        )


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
    """Append one run while safely extending an older manifest schema."""
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


def _run_tag(args: argparse.Namespace, stamp: str, resolved_jobs: int) -> str:
    """Build a collision-resistant filename tag containing all run controls."""
    return (
        f"B{args.n_resamples}_m{args.tickers_per_draw}_seed{args.seed}"
        f"_O{args.n_outer_resamples}_MC{args.n_mc_repeats}"
        f"_J{resolved_jobs}_{stamp}"
    )


def _overall_work_units(args: argparse.Namespace) -> int:
    """Approximate total work, dominated by completed inner draws."""
    fixed_units = 4 + 3 + 1 + 2 + 1  # load, preprocess, setup, save, report
    point_units = args.n_resamples
    outer_units = args.n_outer_resamples * args.n_resamples
    # Repeat zero reuses the point estimate and must not be counted twice.
    mc_units = max(0, args.n_mc_repeats - 1) * args.n_resamples
    return fixed_units + point_units + outer_units + mc_units


def _execute(args: argparse.Namespace, stamp: str, record: dict) -> None:
    """Load, preprocess, evaluate, report and save one benchmark run."""
    total_stages = 6 + int(bool(args.n_outer_resamples)) + int(bool(args.n_mc_repeats))
    stage_number = 0
    total_work = _overall_work_units(args)

    with tqdm(
        total=total_work,
        desc="Overall benchmark",
        unit="work",
        position=0,
        leave=True,
        dynamic_ncols=True,
    ) as overall:

        def stage(title: str, detail: str = "") -> float:
            nonlocal stage_number
            stage_number += 1
            overall.set_postfix_str(f"stage {stage_number}/{total_stages}: {title}")
            suffix = f" — {detail}" if detail else ""
            _log(f"[{stage_number}/{total_stages}] {title}{suffix}")
            return time.monotonic()

        _log(
            "Starting benchmark with "
            f"B={args.n_resamples}, m={args.tickers_per_draw}, seed={args.seed}, "
            f"outer={args.n_outer_resamples}, MC repeats={args.n_mc_repeats}, "
            f"requested workers={args.n_jobs}. Planned matched draws: "
            f"{args.n_resamples * (1 + args.n_outer_resamples + max(0, args.n_mc_repeats - 1)):,}."
        )

        t_stage = stage("Load curated datasets", str(CURATED_DIR))
        loaders = [
            CuratedParquetLoader(
                parquet_path=str(CURATED_DIR / "real_prices.parquet"),
                name="Real",
                is_synthetic=False,
            ),
            CuratedParquetLoader(
                parquet_path=str(CURATED_DIR / "ail_prices.parquet"),
                name="AIL",
                is_synthetic=True,
            ),
            GBMBaselineLoader(parquet_path=str(CURATED_DIR / "gbm_prices.parquet")),
            MSVBaselineLoader(parquet_path=str(CURATED_DIR / "msv_prices.parquet")),
        ]
        loaded = []
        for loader in loaders:
            dataset_t0 = time.monotonic()
            dataset = loader.load()
            loaded.append(dataset)
            overall.update(1)
            _log(
                f"Loaded {dataset.name}: {_dataset_shape(dataset)} "
                f"in {time.monotonic() - dataset_t0:.1f}s."
            )
        real, ail, gbm, msv = loaded
        _log(f"Dataset loading completed in {time.monotonic() - t_stage:.1f}s.")

        t_stage = stage(
            "Preprocess real-synthetic pairs",
            "log returns, overnight mask and conditional FFF",
        )
        deseas_real = None
        synthetics = {}
        for dataset in (ail, gbm, msv):
            pair_t0 = time.monotonic()
            _log(f"Preprocessing Real vs {dataset.name}...")
            pipeline = PreprocessingPipeline().run(real.prices, dataset.prices)
            synthetics[dataset.name] = pipeline.deseas_synthetic
            if deseas_real is None:
                deseas_real = pipeline.deseas_real
            overall.update(1)
            _log(
                f"Preprocessed {dataset.name}: real={pipeline.deseas_real.shape}, "
                f"synthetic={pipeline.deseas_synthetic.shape} "
                f"in {time.monotonic() - pair_t0:.1f}s."
            )
        _log(f"Preprocessing completed in {time.monotonic() - t_stage:.1f}s.")

        t_stage = stage("Configure metrics and parallel engine")
        metrics = build_metrics()
        engine = MatchedTickerBootstrap(
            metrics=metrics,
            n_resamples=args.n_resamples,
            tickers_per_draw=args.tickers_per_draw,
            seed=args.seed,
            n_jobs=args.n_jobs,
            inner_chunk_size=args.inner_chunk_size or None,
        )
        record["n_jobs_resolved"] = engine.n_jobs
        overall.update(1)
        _log(
            f"Configured metrics: {', '.join(metric.name for metric in metrics)}. "
            f"Resolved total worker budget: {engine.n_jobs}; "
            f"inner chunk size: {args.inner_chunk_size or 'automatic'}; "
            f"replicate chunk size: {args.replicate_chunk_size or 'automatic'}."
        )
        _log(f"Engine setup completed in {time.monotonic() - t_stage:.1f}s.")

        t_stage = stage(
            "Matched ticker point estimate",
            f"{args.n_resamples} inner draws using up to {engine.n_jobs} workers",
        )
        point_results = engine.run(
            deseas_real,
            synthetics,
            progress_callback=overall.update,
            progress_position=1,
            leave_progress=False,
        )
        point_aggregate = engine.compute_aggregate()
        results = point_results.copy()
        aggregate = point_aggregate.copy()
        tag = _run_tag(args, stamp, engine.n_jobs)
        _log(
            f"Point estimation completed in {time.monotonic() - t_stage:.1f}s; "
            f"{len(point_results)} metric-generator cells and "
            f"{len(point_aggregate)} aggregate scores computed."
        )

        outer = None
        interval_label = _interval_label(args.n_outer_resamples)
        if args.n_outer_resamples:
            t_stage = stage(
                "Outer ticker-corpus bootstrap",
                f"{args.n_outer_resamples} complete corpus replicates",
            )
            outer = engine.run_outer_bootstrap(
                deseas_real,
                synthetics,
                n_outer_resamples=args.n_outer_resamples,
                replicate_chunk_size=args.replicate_chunk_size or None,
                progress_callback=overall.update,
                progress_position=1,
                leave_progress=False,
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
            results, aggregate = _attach_outer_diagnostics(results, aggregate)
            record["n_point_outside_outer_interval"] = int(
                (~results["point_in_outer_interval"]).sum()
            )
            record["n_aggregate_outside_outer_interval"] = int(
                (~aggregate["point_in_outer_interval"]).sum()
            )
            record["outer_interval_label"] = interval_label
            _log(
                f"Outer bootstrap completed in {time.monotonic() - t_stage:.1f}s; "
                f"valid metric replicate counts range from "
                f"{int(outer.metric_summary['n_outer_valid'].min())} to "
                f"{int(outer.metric_summary['n_outer_valid'].max())}."
            )
            _log_outer_diagnostics(results, aggregate, args.n_outer_resamples)

        mc = None
        if args.n_mc_repeats:
            t_stage = stage(
                "Monte Carlo numerical stability",
                f"{args.n_mc_repeats} seeds; repeat zero reuses the point estimate",
            )
            mc = engine.run_monte_carlo_stability(
                deseas_real,
                synthetics,
                n_repeats=args.n_mc_repeats,
                base_metric_result=point_results,
                base_aggregate_result=point_aggregate,
                replicate_chunk_size=args.replicate_chunk_size or None,
                progress_callback=overall.update,
                progress_position=1,
                leave_progress=False,
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
            _log(
                f"Monte Carlo stability completed in {time.monotonic() - t_stage:.1f}s. "
                f"Metric seed SD range: {mc.metric_summary['mc_sd'].min():.4g}-"
                f"{mc.metric_summary['mc_sd'].max():.4g}."
            )

        t_stage = stage("Save result and replicate files", str(RESULTS_DIR))
        if outer is not None:
            outer_metric_path = RESULTS_DIR / f"outer_metric_replicates_{tag}.csv"
            outer_aggregate_path = RESULTS_DIR / f"outer_aggregate_replicates_{tag}.csv"
            outer.metric_replicates.to_csv(outer_metric_path, index=False)
            outer.aggregate_replicates.to_csv(outer_aggregate_path, index=False)
            record["outer_metric_replicates_csv"] = outer_metric_path.name
            record["outer_aggregate_replicates_csv"] = outer_aggregate_path.name
            _log(
                f"Saved outer replicate details: {outer_metric_path.name}, "
                f"{outer_aggregate_path.name}."
            )

        if mc is not None:
            mc_metric_path = RESULTS_DIR / f"mc_metric_replicates_{tag}.csv"
            mc_aggregate_path = RESULTS_DIR / f"mc_aggregate_replicates_{tag}.csv"
            mc.metric_replicates.to_csv(mc_metric_path, index=False)
            mc.aggregate_replicates.to_csv(mc_aggregate_path, index=False)
            record["mc_metric_replicates_csv"] = mc_metric_path.name
            record["mc_aggregate_replicates_csv"] = mc_aggregate_path.name
            _log(
                f"Saved Monte Carlo replicate details: {mc_metric_path.name}, "
                f"{mc_aggregate_path.name}."
            )
        overall.update(1)

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
            _log("Canonical benchmark_results.csv and aggregate_results.csv refreshed.")
        overall.update(1)
        _log(
            f"Saved summary files in {time.monotonic() - t_stage:.1f}s: "
            f"{csv_path.name}, {agg_path.name}."
        )

        stage("Print benchmark tables and run summary")
        if interval_label is None:
            _log(
                "Per-metric benchmark table — each cell is the matched ticker point "
                "estimate; no corpus interval was requested."
            )
        else:
            _log(
                "Per-metric benchmark table — each cell is point estimate "
                f"[lower, upper], where lower/upper are {interval_label}."
            )
        tqdm.write("\n" + format_table(results) + "\n")

        _log(
            "Aggregate fidelity table — G = exp(mean_k log(mean(g_sr)/mean(g_rr))). "
            "G=1 is real-real noise-floor parity; larger G means a larger aggregate gap."
        )
        tqdm.write("\n" + format_aggregate_table(aggregate, interval_label) + "\n")

        if mc is not None:
            _log(
                "Per-metric Monte Carlo stability table — variability across "
                "independent inner seeds on fixed corpora; not a confidence interval."
            )
            tqdm.write("\n" + format_mc_table(mc.metric_summary, value_column="score") + "\n")
            _log(
                "Aggregate Monte Carlo stability table — seed sensitivity of G on "
                "the fixed observed corpora."
            )
            tqdm.write("\n" + format_mc_table(mc.aggregate_summary, value_column="G") + "\n")

        _log(f"Tidy benchmark results: {csv_path}")
        _log(f"Aggregate results: {agg_path}")
        record["status"] = "completed"
        overall.update(1)
        overall.set_postfix_str("completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fineval benchmark table.")
    parser.add_argument("--n-resamples", type=int, default=100, help="inner matched draws B")
    parser.add_argument("--tickers-per-draw", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help=(
            "total worker budget; 0 uses all but one logical CPU capped at 8, "
            "-1 uses every logical CPU"
        ),
    )
    parser.add_argument(
        "--inner-chunk-size",
        type=int,
        default=0,
        help="inner draws per submitted task; 0 chooses automatically",
    )
    parser.add_argument(
        "--replicate-chunk-size",
        type=int,
        default=0,
        help="outer/Monte Carlo replicates per process task; 0 chooses automatically",
    )
    parser.add_argument(
        "--n-outer-resamples",
        type=int,
        default=0,
        help=(
            "complete corpus bootstrap replicates; 0 disables, 2-39 are "
            "diagnostic only, >=40 enables nominal 95 percent percentile-CI labelling"
        ),
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
        help=(
            "refresh canonical CSVs; requires at least "
            f"{MIN_OUTER_REPLICATES_FOR_CI} outer replicates"
        ),
    )
    args = parser.parse_args()

    if args.n_resamples < 1:
        parser.error("--n-resamples must be at least 1")
    if args.tickers_per_draw < 1:
        parser.error("--tickers-per-draw must be at least 1")
    if args.n_jobs < -1:
        parser.error("--n-jobs must be -1, 0, or a positive integer")
    if args.inner_chunk_size < 0:
        parser.error("--inner-chunk-size must be 0 or a positive integer")
    if args.replicate_chunk_size < 0:
        parser.error("--replicate-chunk-size must be 0 or a positive integer")
    if args.n_outer_resamples == 1 or args.n_outer_resamples < 0:
        parser.error("--n-outer-resamples must be 0 or at least 2")
    if args.n_mc_repeats == 1 or args.n_mc_repeats < 0:
        parser.error("--n-mc-repeats must be 0 or at least 2")
    if args.update_canonical and args.n_outer_resamples < MIN_OUTER_REPLICATES_FOR_CI:
        parser.error(
            f"--update-canonical requires --n-outer-resamples >= {MIN_OUTER_REPLICATES_FOR_CI}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    record = {
        "run_id": stamp,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "n_resamples": args.n_resamples,
        "tickers_per_draw": args.tickers_per_draw,
        "seed": args.seed,
        "n_jobs_requested": args.n_jobs,
        "n_jobs_resolved": "",
        "inner_chunk_size": args.inner_chunk_size,
        "replicate_chunk_size": args.replicate_chunk_size,
        "n_outer_resamples": args.n_outer_resamples,
        "n_mc_repeats": args.n_mc_repeats,
        "outer_interval_label": "",
        "n_point_outside_outer_interval": "",
        "n_aggregate_outside_outer_interval": "",
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
        _log(
            f"Run #{run_number} ({record['status']}) recorded in "
            f"{RESULTS_DIR / 'runs_manifest.csv'} after "
            f"{record['duration_seconds']:.1f}s."
        )


if __name__ == "__main__":
    main()
