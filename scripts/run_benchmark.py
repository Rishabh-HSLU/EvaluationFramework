"""Command-line interface for the full fineval benchmark."""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from fineval.benchmark.artifacts import append_manifest, new_run_record
from fineval.benchmark.config import MIN_OUTER_REPLICATES_FOR_CI, RESULTS_DIR
from fineval.benchmark.reporting import log
from fineval.benchmark.runner import run_benchmark
from fineval.config import SEED
from fineval.scripts.sensitivity.grid import GRID

#: Selectable specifications, keyed by ``spec_id``. Every run resolves to
#: exactly one of these, so a run's metric parameters are always a member of
#: the pre-specified grid rather than free-form CLI values.
SPECS = {spec.spec_id: spec for spec in GRID}


def build_parser() -> argparse.ArgumentParser:
    """Create the benchmark argument parser."""
    parser = argparse.ArgumentParser(description="Run the fineval benchmark table.")
    parser.add_argument("--n-resamples", type=int, default=100, help="inner matched draws B")
    parser.add_argument("--tickers-per-draw", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--spec",
        type=str,
        default="primary",
        choices=sorted(SPECS),
        help=(
            "pre-specified sensitivity spec whose metric parameters to run; "
            "primary reproduces the canonical fineval.config parameters"
        ),
    )
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate benchmark CLI arguments.

    Also resolves ``--spec`` into ``args.spec_params``, the complete
    eight-key parameter set passed to ``build_metrics``. Resolution happens
    here rather than in ``main`` so that every consumer of a parsed
    namespace sees both ``spec`` and ``spec_params``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

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
    if args.update_canonical and args.spec != "primary":
        parser.error("--update-canonical is only allowed for --spec primary")
    args.spec_params = dict(SPECS[args.spec].metric_params)
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark and record completion or failure in the manifest."""
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    record = new_run_record(args, stamp)
    started = time.monotonic()
    try:
        run_benchmark(args, stamp, record)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["duration_seconds"] = round(time.monotonic() - started, 1)
        run_number = append_manifest(record)
        log(
            f"Run #{run_number} ({record['status']}) recorded in "
            f"{RESULTS_DIR / 'runs_manifest.csv'} after "
            f"{record['duration_seconds']:.1f}s."
        )


if __name__ == "__main__":
    main()
