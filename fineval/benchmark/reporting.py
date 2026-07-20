"""Progress, logging, diagnostics, and table rendering for the benchmark."""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from .config import (
    METRIC_LABELS,
    MIN_OUTER_REPLICATES_FOR_CI,
    RECOMMENDED_OUTER_REPLICATES,
)


def log(message: str) -> None:
    """Write a timestamped message without corrupting progress bars."""
    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def interval_label(n_outer_resamples: int) -> str | None:
    """Return an honest label for the reported outer percentiles."""
    if n_outer_resamples == 0:
        return None
    if n_outer_resamples < MIN_OUTER_REPLICATES_FOR_CI:
        return f"diagnostic outer 2.5%-97.5% quantiles; O={n_outer_resamples}, not a 95% CI"
    return f"95% outer-bootstrap percentile CI; O={n_outer_resamples}"


def attach_outer_diagnostics(
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


def log_outer_diagnostics(
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    n_outer_resamples: int,
) -> None:
    """Explain interval resolution and point-estimate discrepancies."""
    if n_outer_resamples < MIN_OUTER_REPLICATES_FOR_CI:
        log(
            f"WARNING: {n_outer_resamples} outer replicates only provide smoke-test "
            "quantiles, not a usable 95% confidence interval. Use at least "
            f"{MIN_OUTER_REPLICATES_FOR_CI}, preferably "
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
        log("Every point estimate lies within its outer percentile limits.")
        return

    log(
        f"{len(outside)}/{len(results)} metric-generator points and "
        f"{len(aggregate_outside)}/{len(aggregate)} aggregate points lie outside "
        "their outer percentile limits. A percentile interval need not contain "
        "the original point estimate."
    )
    for row in outside.itertuples(index=False):
        log(
            f"  {row.metric}/{row.generator}: point={row.score:.4f}, "
            f"outer=[{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"bias={row.outer_bias:+.4f}"
        )
    for row in aggregate_outside.itertuples(index=False):
        log(
            f"  aggregate/{row.generator}: point={row.G:.4f}, "
            f"outer=[{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"bias={row.outer_bias:+.4f}"
        )


def format_metric_table(results: pd.DataFrame) -> str:
    """Render metric point estimates and optional outer intervals."""
    generators = list(dict.fromkeys(results["generator"]))
    lines = [
        "| Metric | Stylized fact | " + " | ".join(generators) + " |",
        "|---|---|" + "---|" * len(generators),
    ]
    has_interval = {"ci_low", "ci_high"}.issubset(results.columns)
    for metric in dict.fromkeys(results["metric"]):
        cells = []
        for generator in generators:
            row = results[(results["metric"] == metric) & (results["generator"] == generator)].iloc[
                0
            ]
            value = f"{row['score']:.3f}"
            if has_interval and pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
                value += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            cells.append(value)
        lines.append(
            f"| {metric} | {METRIC_LABELS.get(metric, metric)} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def format_aggregate_table(
    aggregate: pd.DataFrame,
    outer_label: str | None,
) -> str:
    """Render aggregate point estimates and optional outer intervals."""
    has_interval = {"ci_low", "ci_high"}.issubset(aggregate.columns)
    value_header = f"G [{outer_label}]" if has_interval and outer_label else "G"
    lines = [f"| Generator | {value_header} | metrics used |", "|---|---|---|"]
    for row in aggregate.itertuples(index=False):
        value = f"{row.G:.3f}"
        if has_interval and pd.notna(row.ci_low) and pd.notna(row.ci_high):
            value += f" [{row.ci_low:.3f}, {row.ci_high:.3f}]"
        lines.append(f"| {row.generator} | {value} | {int(row.k_used)}/{int(row.k_total)} |")
    return "\n".join(lines)


def format_mc_table(summary: pd.DataFrame, value_column: str) -> str:
    """Render independent-seed stability summaries."""
    if value_column == "score":
        lines = [
            "| Metric | Generator | mean ± SD | min-max |",
            "|---|---|---|---|",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {row.generator} | {row.mc_mean:.3f} ± "
                f"{row.mc_sd:.3f} | {row.mc_min:.3f}-{row.mc_max:.3f} |"
            )
    else:
        lines = ["| Generator | mean G ± SD | min-max |", "|---|---|---|"]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.generator} | {row.mc_mean:.3f} ± {row.mc_sd:.3f} | "
                f"{row.mc_min:.3f}-{row.mc_max:.3f} |"
            )
    return "\n".join(lines)


def overall_work_units(args: argparse.Namespace) -> int:
    """Approximate total work, dominated by completed inner draws."""
    fixed_units = 11
    point_units = args.n_resamples
    outer_units = args.n_outer_resamples * args.n_resamples
    mc_units = max(0, args.n_mc_repeats - 1) * args.n_resamples
    return fixed_units + point_units + outer_units + mc_units


class BenchmarkProgress:
    """One persistent overall bar plus stage-level timestamped logging."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.total_stages = 6 + int(bool(args.n_outer_resamples)) + int(bool(args.n_mc_repeats))
        self.stage_number = 0
        self.bar = tqdm(
            total=overall_work_units(args),
            desc="Overall benchmark",
            unit="work",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

    def __enter__(self) -> BenchmarkProgress:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.bar.close()

    def update(self, amount: int = 1) -> None:
        self.bar.update(amount)

    def stage(self, title: str, detail: str = "") -> float:
        self.stage_number += 1
        self.bar.set_postfix_str(f"stage {self.stage_number}/{self.total_stages}: {title}")
        suffix = f" — {detail}" if detail else ""
        log(f"[{self.stage_number}/{self.total_stages}] {title}{suffix}")
        return time.monotonic()

    def complete(self) -> None:
        self.bar.set_postfix_str("completed")
