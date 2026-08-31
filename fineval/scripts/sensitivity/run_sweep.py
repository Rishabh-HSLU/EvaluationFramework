"""Exploratory sensitivity sweep over the pre-specified OAT grid.

Wraps the existing benchmark internals — ``load_default_datasets``,
``preprocess_pairs``, ``build_metrics`` and ``MatchedTickerBootstrap`` — and
runs them once per spec in ``grid.GRID``. No engine or metric logic is
reimplemented; the only formula the harness owns is the fast-mode ``G_dev``
recombination, which is self-checked against ``engine.compute_aggregate()``
on the primary spec every run.

Outputs go to ``fineval/scripts/sensitivity/results/`` (gitignored), never
to the canonical ``RESULTS_DIR``, and every artifact is labeled
``run_kind = "exploratory"``.

Usage::

    python -m fineval.scripts.sensitivity.run_sweep --dry-run
    python -m fineval.scripts.sensitivity.run_sweep [--fast] [--n-resamples B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fineval.benchmark.artifacts import git_commit
from fineval.benchmark.config import CURATED_DIR, build_metrics
from fineval.benchmark.datasets import load_default_datasets, preprocess_pairs
from fineval.bootstrap import MatchedTickerBootstrap
from fineval.config import (
    M1_N_GRID,
    M1_TAIL_ALPHA,
    M1_TAIL_LAMBDA,
    N_REGIME_QUINTILES,
    REGIME_WEIGHTS,
    ROLLING_VOL_MIN_FRAC,
    ROLLING_VOL_MIN_PERIODS,
    ROLLING_VOL_WINDOW,
    SEED,
    TAIL_FRACTION,
)
from fineval.scripts.sensitivity.grid import (
    DERIVED_PARAMS,
    DIAL_TO_METRIC,
    GRID,
    PARAM_KEYS,
    PRIMARY_PARAMS,
    RUN_KIND,
    Spec,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_METRICS = ("M1", "M2", "M3", "M4")
CURATED_FILES = (
    "real_prices.parquet",
    "ail_prices.parquet",
    "gbm_prices.parquet",
    "msv_prices.parquet",
)


def log(message: str) -> None:
    print(f"[sweep {datetime.now():%H:%M:%S}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Startup guards
# ---------------------------------------------------------------------------


def assert_primary_matches_config() -> None:
    """Hard-fail unless the grid's primary spec equals fineval.config."""
    p = PRIMARY_PARAMS
    normalized = np.asarray(p["regime_weights"], dtype=float)
    normalized = normalized / normalized.sum()
    checks = [
        ("n_grid", p["n_grid"], M1_N_GRID),
        ("tail_alpha", p["tail_alpha"], M1_TAIL_ALPHA),
        ("tail_lambda", p["tail_lambda"], M1_TAIL_LAMBDA),
        ("window", p["window"], ROLLING_VOL_WINDOW),
        ("min_periods", p["min_periods"], ROLLING_VOL_MIN_PERIODS),
        ("n_regimes", p["n_regimes"], N_REGIME_QUINTILES),
        ("tail_fraction", p["tail_fraction"], TAIL_FRACTION),
        ("min_frac", p["min_frac"], ROLLING_VOL_MIN_FRAC),
    ]
    mismatches = [
        f"  {name}: grid={grid_value!r} != config={config_value!r}"
        for name, grid_value, config_value in checks
        if grid_value != config_value
    ]
    if not np.allclose(normalized, REGIME_WEIGHTS):
        mismatches.append(
            f"  regime_weights (normalized): grid={normalized.tolist()!r}"
            f" != config={np.asarray(REGIME_WEIGHTS).tolist()!r}"
        )
    if mismatches:
        raise SystemExit(
            "ABORT: grid.PRIMARY_PARAMS has drifted from fineval/config.py — "
            "the sensitivity grid no longer brackets the canonical run:\n" + "\n".join(mismatches)
        )


def assert_grid_integrity() -> None:
    """Hard-fail on structural defects in the grid itself."""
    problems: list[str] = []
    ids = [spec.spec_id for spec in GRID]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate spec_id values in {ids}")
    if len(GRID) != 16:
        problems.append(f"expected 16 specs, found {len(GRID)}")
    for spec in GRID:
        if set(spec.params) != set(PARAM_KEYS):
            problems.append(f"{spec.spec_id}: params keys {sorted(spec.params)} incomplete")
            continue
        if spec.dial == "none":
            if spec.params != dict(PRIMARY_PARAMS):
                problems.append(f"{spec.spec_id}: primary params differ from PRIMARY_PARAMS")
            continue
        unknown = [dial for dial in spec.dials if dial not in DIAL_TO_METRIC]
        if unknown:
            problems.append(f"{spec.spec_id}: unknown dial(s) {unknown!r}")
            continue
        if spec.dial == "corner":
            if len(spec.dials) < 2:
                problems.append(f"{spec.spec_id}: corner spec varies {len(spec.dials)} dial(s)")
        elif spec.dials != (spec.dial,):
            problems.append(
                f"{spec.spec_id}: dials {spec.dials!r} disagree with dial {spec.dial!r}"
            )
        for dial in spec.dials:
            if spec.params[dial] == PRIMARY_PARAMS[dial]:
                problems.append(f"{spec.spec_id}: {dial} equals primary; should have collapsed")
        allowed = set(spec.dials)
        for dial in spec.dials:
            allowed.update(DERIVED_PARAMS.get(dial, ()))
        for key in PARAM_KEYS:
            if key not in allowed and spec.params[key] != PRIMARY_PARAMS[key]:
                problems.append(
                    f"{spec.spec_id}: off-axis variation — {key} moves but is not a declared dial"
                )
        expected_mp = math.ceil(spec.params["window"] * spec.params["min_frac"])
        if spec.params["min_periods"] != expected_mp:
            problems.append(
                f"{spec.spec_id}: min_periods={spec.params['min_periods']} violates "
                f"the ceil(window * min_frac) = ceil({spec.params['window']} * "
                f"{spec.params['min_frac']}) = {expected_mp} rule"
            )
    if problems:
        raise SystemExit("ABORT: grid integrity check failed:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def metrics_for(spec: Spec, fast: bool) -> list:
    """Build the metric objects a spec's engine run needs.

    Full mode (and the primary spec in every mode) runs all four metrics.
    Fast mode runs only the metric the spec's dial affects; unaffected
    M2/M3 rows are copied from the primary run afterwards, which is exact
    because draw indices do not depend on the metric list.
    """
    metrics = build_metrics(**spec.metric_params)
    if not fast or spec.dial == "none":
        return metrics
    affected = set(spec.affected_metrics)
    return [metric for metric in metrics if metric.name in affected]


def identity_block(spec: Spec, args: argparse.Namespace, stamp: str) -> dict:
    """Columns identifying one spec run in every tidy output row."""
    weights = ":".join(f"{w:g}" for w in spec.params["regime_weights"])
    dial_value = (
        ":".join(f"{v:g}" for v in spec.dial_value)
        if isinstance(spec.dial_value, tuple)
        else spec.dial_value
    )
    return {
        "run_kind": RUN_KIND,
        "spec_id": spec.spec_id,
        "dial": spec.dial,
        "dial_value": dial_value,
        **{key: spec.params[key] for key in PARAM_KEYS if key != "regime_weights"},
        "regime_weights": weights,
        "B": args.n_resamples,
        "m": args.tickers_per_draw,
        "seed": args.seed,
        "mode": "fast" if args.fast else "full",
        "git_commit": git_commit(),
        "run_timestamp": stamp,
    }


# ---------------------------------------------------------------------------
# Fast-mode G_dev recombination (the one harness-owned formula; self-checked)
# ---------------------------------------------------------------------------


def aggregate_from_metric_rows(metric_rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute G_dev per generator from tidy per-metric rows.

    Mirrors ``MatchedTickerBootstrap.compute_aggregate``:
    ``G_dev = exp(mean_k |log(g_sr_mean / g_rr_mean)|)``, NaN unless every
    metric contributes a finite positive ratio. Validated each run against
    the engine's own output on the primary spec.
    """
    rows = []
    for generator, cells in metric_rows.groupby("generator", sort=False):
        log_deviations: list[float] = []
        valid_draw_counts: list[int] = []
        for cell in cells.itertuples():
            rr, sr = cell.g_rr_mean, cell.g_sr_mean
            if np.isfinite(rr) and np.isfinite(sr) and rr > 0.0 and sr > 0.0:
                log_deviations.append(abs(float(np.log(sr / rr))))
                valid_draw_counts.append(int(cell.n_valid_draws))
        k_used = len(log_deviations)
        k_total = len(cells)
        rows.append(
            {
                "generator": generator,
                "G_dev": float(np.exp(np.mean(log_deviations)))
                if k_used == k_total
                else float("nan"),
                "k_used": k_used,
                "k_total": k_total,
                "min_valid_draws": min(valid_draw_counts) if valid_draw_counts else 0,
            }
        )
    return pd.DataFrame(rows)


def check_gdev_reproduction(engine_aggregate: pd.DataFrame, metric_rows: pd.DataFrame) -> None:
    """Hard-fail unless the harness formula reproduces the engine's G_dev."""
    ours = aggregate_from_metric_rows(metric_rows).set_index("generator")
    theirs = engine_aggregate.set_index("generator")
    for generator in theirs.index:
        a, b = theirs.loc[generator, "G_dev"], ours.loc[generator, "G_dev"]
        same = (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-12, atol=0.0)
        if not same:
            raise SystemExit(
                f"ABORT: fast-mode G_dev recombination mismatch on primary spec "
                f"({generator}): engine={a!r}, harness={b!r}"
            )
    log("Self-check passed: harness G_dev formula reproduces engine G_dev on the primary spec.")


# ---------------------------------------------------------------------------
# M4 effective-sample-size diagnostic (reuses the metric's own methods)
# ---------------------------------------------------------------------------


def m4_panel_diagnostics(m4_metric, panel: pd.DataFrame) -> dict[str, float]:
    """Effective sample size of the M4 pipeline on one full panel.

    Uses the metric instance's own ``_rolling_vol`` and ``extract_features``
    so the counts describe exactly what the metric sees:

    - ``valid_pairs``: pooled (volatility, return) pairs that are finite —
      the observations that survive regime labeling.
    - ``valid_label_sessions``: (ticker, session) cells contributing at
      least one valid pair; thin cells (short half-day sessions under
      window=120/min_periods=60) drop out here.
    - ``regime_fits``: regimes (of n_regimes) whose full-panel GPD fit
      returned a finite xi.
    """
    vol = m4_metric._rolling_vol(panel).reindex(panel.index)
    valid = vol.notna() & panel.notna()
    sessions_with_labels = valid.groupby(panel.index.normalize()).any()
    xis = m4_metric.extract_features(panel)
    return {
        "valid_pairs": int(valid.to_numpy().sum()),
        "valid_label_sessions": int(sessions_with_labels.to_numpy().sum()),
        "regime_fits": int(np.isfinite(xis).sum()),
    }


# ---------------------------------------------------------------------------
# Snapshot / manifest
# ---------------------------------------------------------------------------


def dataset_hashes() -> dict[str, str]:
    hashes = {}
    for name in CURATED_FILES:
        path = CURATED_DIR / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def write_grid_snapshot(run_dir: Path, args: argparse.Namespace, stamp: str) -> None:
    snapshot = {
        "run_kind": RUN_KIND,
        "created_at": stamp,
        "git_commit": git_commit(),
        "n_resamples": args.n_resamples,
        "tickers_per_draw": args.tickers_per_draw,
        "seed": args.seed,
        "mode": "fast" if args.fast else "full",
        "dataset_sha256": dataset_hashes(),
        "specs": [asdict(spec) for spec in GRID],
    }
    (run_dir / "grid_spec.json").write_text(json.dumps(snapshot, indent=2, default=list))
    log(f"Wrote grid snapshot: {run_dir / 'grid_spec.json'}")


def append_manifest(row: dict) -> None:
    manifest_path = RESULTS_DIR / "sweep_manifest.csv"
    frame = pd.DataFrame([row])
    header = not (manifest_path.exists() and manifest_path.stat().st_size > 0)
    frame.to_csv(manifest_path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_sweep(args: argparse.Namespace) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"sweep_{stamp}_{RUN_KIND}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_grid_snapshot(run_dir, args, stamp)

    log("Loading curated datasets (shared across all specs)...")
    real, ail, gbm, msv = load_default_datasets(lambda _n: None)
    deseas_real, synthetics = preprocess_pairs(real, (ail, gbm, msv), lambda _n: None)
    log(f"Shared data ready: real={deseas_real.shape}, generators={list(synthetics)}.")

    metric_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    primary_metric_rows: pd.DataFrame | None = None
    diagnostics_cache: dict[tuple, dict[str, dict[str, float]]] = {}

    ordered = [spec for spec in GRID if spec.dial == "none"] + [
        spec for spec in GRID if spec.dial != "none"
    ]
    for position, spec in enumerate(ordered, start=1):
        started = time.monotonic()
        manifest_row = {
            "run_kind": RUN_KIND,
            "run_timestamp": stamp,
            "spec_id": spec.spec_id,
            "mode": "fast" if args.fast else "full",
            "B": args.n_resamples,
            "m": args.tickers_per_draw,
            "seed": args.seed,
            "git_commit": git_commit(),
            "status": "failed",
            "duration_seconds": 0.0,
            "error": "",
        }
        try:
            log(f"[{position}/{len(ordered)}] Running spec {spec.spec_id!r}...")
            metrics = metrics_for(spec, args.fast)
            engine = MatchedTickerBootstrap(
                metrics=metrics,
                n_resamples=args.n_resamples,
                tickers_per_draw=args.tickers_per_draw,
                seed=args.seed,
                n_jobs=args.n_jobs or None,
            )
            metric_rows = engine.run(deseas_real, synthetics, show_progress=False)

            if spec.dial == "none":
                primary_metric_rows = metric_rows.copy()
                engine_aggregate = engine.compute_aggregate()
                check_gdev_reproduction(engine_aggregate, metric_rows)
                aggregate_rows = engine_aggregate
            elif args.fast:
                if primary_metric_rows is None:
                    raise RuntimeError(
                        "fast mode needs the primary spec's rows as M2/M3 donors, "
                        "but the primary run did not complete"
                    )
                donors = primary_metric_rows.loc[
                    ~primary_metric_rows["metric"].isin({m.name for m in metrics})
                ]
                metric_rows = pd.concat([metric_rows, donors], ignore_index=True)
                metric_rows["metric"] = pd.Categorical(
                    metric_rows["metric"], categories=ALL_METRICS, ordered=True
                )
                metric_rows = metric_rows.sort_values(
                    ["metric", "generator"], kind="stable"
                ).reset_index(drop=True)
                metric_rows["metric"] = metric_rows["metric"].astype(str)
                aggregate_rows = aggregate_from_metric_rows(metric_rows)
            else:
                aggregate_rows = engine.compute_aggregate()

            diag = spec_diagnostics(spec, deseas_real, synthetics, diagnostics_cache)
            identity = identity_block(spec, args, stamp)
            metric_frames.append(attach_identity(metric_rows, identity, diag))
            aggregate_frames.append(attach_identity(aggregate_rows, identity, None))
            manifest_row["status"] = "completed"
            log(f"Spec {spec.spec_id!r} completed in {time.monotonic() - started:.1f}s.")
        except Exception as exc:
            manifest_row["error"] = f"{type(exc).__name__}: {exc}"
            log(f"Spec {spec.spec_id!r} FAILED: {manifest_row['error']}")
        finally:
            manifest_row["duration_seconds"] = round(time.monotonic() - started, 1)
            append_manifest(manifest_row)

    if metric_frames:
        metric_path = run_dir / "sweep_metric_results.csv"
        aggregate_path = run_dir / "sweep_aggregate_results.csv"
        pd.concat(metric_frames, ignore_index=True).to_csv(metric_path, index=False)
        pd.concat(aggregate_frames, ignore_index=True).to_csv(aggregate_path, index=False)
        log(f"Tidy metric results: {metric_path}")
        log(f"Tidy aggregate results: {aggregate_path}")
    log("Sweep finished.")


def spec_diagnostics(
    spec: Spec,
    deseas_real: pd.DataFrame,
    synthetics: dict[str, pd.DataFrame],
    cache: dict[tuple, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Full-panel M4 effective-sample-size diagnostics, cached by M4 params."""
    key = tuple(
        spec.params[name] for name in ("window", "min_periods", "n_regimes", "tail_fraction")
    )
    if key not in cache:
        m4_metric = next(m for m in build_metrics(**spec.metric_params) if m.name == "M4")
        per_dataset = {"Real": m4_panel_diagnostics(m4_metric, deseas_real)}
        for generator, panel in synthetics.items():
            per_dataset[generator] = m4_panel_diagnostics(m4_metric, panel)
        cache[key] = per_dataset
    return cache[key]


def attach_identity(
    frame: pd.DataFrame,
    identity: dict,
    diagnostics: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    """Prefix identity columns; append per-generator M4 diagnostics on M4 rows."""
    out = frame.copy()
    for column, value in reversed(identity.items()):
        out.insert(0, column, value)
    if diagnostics is not None:
        for stat in ("valid_pairs", "valid_label_sessions", "regime_fits"):
            is_m4 = out["metric"] == "M4"
            out[f"diag_m4_{stat}_real"] = np.where(is_m4, diagnostics["Real"][stat], np.nan)
            out[f"diag_m4_{stat}_syn"] = [
                diagnostics[row.generator][stat] if row.metric == "M4" else np.nan
                for row in out.itertuples()
            ]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory FinEval sensitivity sweep (OAT grid)."
    )
    parser.add_argument("--n-resamples", type=int, default=100, help="inner matched draws B")
    parser.add_argument("--tickers-per-draw", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED, help="same seed for every spec (paired)")
    parser.add_argument("--n-jobs", type=int, default=0, help="0 = engine automatic")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Option B: run only the dial-affected metric per spec and copy "
            "M2/M3 rows from the primary run (exact; draw indices are "
            "metric-independent). Default is full runs of all four metrics."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate guards and print all specs without loading data or running anything",
    )
    return parser


def dry_run(args: argparse.Namespace) -> None:
    log(f"DRY RUN — {len(GRID)} specs validated; nothing loaded, nothing executed.")
    mode = "fast" if args.fast else "full"
    header = (
        f"{'spec_id':<20} {'dial':<15} {'metrics run':<12} "
        f"{'alpha':>6} {'lambda':>7} {'n_grid':>7} {'win':>4} {'m_frac':>7} "
        f"{'min_p':>6} {'n_reg':>6} {'tail_f':>7}  weights"
    )
    print("\n" + header)
    print("-" * len(header))
    for spec in GRID:
        p = spec.params
        run_metrics = (
            ",".join(spec.affected_metrics) if args.fast and spec.dial != "none" else "M1,M2,M3,M4"
        )
        weights = ":".join(f"{w:g}" for w in p["regime_weights"])
        print(
            f"{spec.spec_id:<20} {spec.dial:<15} {run_metrics:<12} "
            f"{p['tail_alpha']:>6g} {p['tail_lambda']:>7g} {p['n_grid']:>7d} "
            f"{p['window']:>4d} {p['min_frac']:>7g} {p['min_periods']:>6d} "
            f"{p['n_regimes']:>6d} {p['tail_fraction']:>7g}  {weights}"
        )
    print()
    log(
        f"Planned execution: mode={mode}, B={args.n_resamples}, m={args.tickers_per_draw}, "
        f"seed={args.seed} (identical across specs — paired draws), "
        f"outer=0, MC=0, outputs under {RESULTS_DIR}/sweep_<stamp>_{RUN_KIND}/."
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    assert_grid_integrity()
    assert_primary_matches_config()
    log("Startup guards passed: grid integrity OK; primary spec matches fineval/config.py.")
    if args.dry_run:
        dry_run(args)
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_sweep(args)


if __name__ == "__main__":
    main()
