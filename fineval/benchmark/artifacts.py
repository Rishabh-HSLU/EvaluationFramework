"""Run metadata, result persistence, and manifest management."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from fineval.bootstrap.models import ReplicateAnalysis

from .config import RESULTS_DIR
from .reporting import log


def git_commit() -> str:
    """Return the short repository commit hash, or ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_tag(args: argparse.Namespace, stamp: str, resolved_jobs: int) -> str:
    """Build a collision-resistant filename tag for one run."""
    spec = getattr(args, "spec", "primary").replace("=", "-")
    return (
        f"spec-{spec}_B{args.n_resamples}_m{args.tickers_per_draw}_seed{args.seed}"
        f"_O{args.n_outer_resamples}_MC{args.n_mc_repeats}"
        f"_J{resolved_jobs}_{stamp}"
    )


def new_run_record(args: argparse.Namespace, stamp: str) -> dict:
    """Create the manifest row populated throughout the run."""
    return {
        "run_id": stamp,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "spec": getattr(args, "spec", "primary"),
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
        "git_commit": git_commit(),
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


def append_manifest(record: dict) -> int:
    """Append a run while safely extending an older manifest schema."""
    manifest_path = RESULTS_DIR / "runs_manifest.csv"
    has_rows = manifest_path.exists() and manifest_path.stat().st_size > 0
    if not has_rows:
        run_number = 1
        pd.DataFrame([{"run_number": run_number, **record}]).to_csv(
            manifest_path,
            index=False,
        )
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


def save_outputs(
    *,
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    outer: ReplicateAnalysis | None,
    mc: ReplicateAnalysis | None,
    tag: str,
    update_canonical: bool,
    record: dict,
) -> tuple[Path, Path]:
    """Persist summary and optional replicate-level CSV files."""
    if outer is not None:
        outer_metric_path = RESULTS_DIR / f"outer_metric_replicates_{tag}.csv"
        outer_aggregate_path = RESULTS_DIR / f"outer_aggregate_replicates_{tag}.csv"
        outer.metric_replicates.to_csv(outer_metric_path, index=False)
        outer.aggregate_replicates.to_csv(outer_aggregate_path, index=False)
        record["outer_metric_replicates_csv"] = outer_metric_path.name
        record["outer_aggregate_replicates_csv"] = outer_aggregate_path.name
        log(
            f"Saved outer replicate details: {outer_metric_path.name}, {outer_aggregate_path.name}."
        )

    if mc is not None:
        mc_metric_path = RESULTS_DIR / f"mc_metric_replicates_{tag}.csv"
        mc_aggregate_path = RESULTS_DIR / f"mc_aggregate_replicates_{tag}.csv"
        mc.metric_replicates.to_csv(mc_metric_path, index=False)
        mc.aggregate_replicates.to_csv(mc_aggregate_path, index=False)
        record["mc_metric_replicates_csv"] = mc_metric_path.name
        record["mc_aggregate_replicates_csv"] = mc_aggregate_path.name
        log(
            f"Saved Monte Carlo replicate details: {mc_metric_path.name}, {mc_aggregate_path.name}."
        )

    results_path = RESULTS_DIR / f"benchmark_{tag}.csv"
    aggregate_path = RESULTS_DIR / f"aggregate_{tag}.csv"
    results.to_csv(results_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    record["results_csv"] = results_path.name
    record["aggregate_csv"] = aggregate_path.name

    if update_canonical:
        results.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
        aggregate.to_csv(RESULTS_DIR / "aggregate_results.csv", index=False)
        record["canonical_updated"] = True
        log("Canonical benchmark_results.csv and aggregate_results.csv refreshed.")

    return results_path, aggregate_path
