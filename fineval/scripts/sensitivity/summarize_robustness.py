"""Report what a sensitivity sweep does and does not establish.

Reads the tidy CSVs written by ``run_sweep`` and answers the two questions
a robustness claim actually rests on:

1. **Ordering preservation.** Does every spec rank the generators the same
   way the primary spec does? A robustness claim is a claim about the
   ranking, not about the score values, so this is the load-bearing check.
2. **Interpretation relative to s = 0.5.** Per-metric scores are read
   against the real-real noise floor: ``s < 0.5`` means the synthetic-real
   gap exceeds that floor, ``s > 0.5`` means it is smaller (a memorization
   red flag, not a better score). A spec that moves a score across 0.5
   changes the sentence written about that cell, even when the ranking
   survives.

What this script deliberately does not claim
--------------------------------------------
``run_sweep`` runs with ``outer = 0`` and ``MC = 0``, so its CSVs carry
point estimates only -- there is no interval column anywhere in either
file. Every difference reported here is therefore a difference between
point estimates. This script reports the margin alongside each finding so
a narrow flip is visibly narrow, but it cannot say whether any flip
exceeds sampling noise. Establishing that needs outer-bootstrap
replicates per spec, which the sweep does not currently produce.

Usage::

    python -m fineval.scripts.sensitivity.summarize_robustness
    python -m fineval.scripts.sensitivity.summarize_robustness --run-dir DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRIC_CSV = "sweep_metric_results.csv"
AGGREGATE_CSV = "sweep_aggregate_results.csv"
REFERENCE_SPEC = "primary"
NOISE_FLOOR = 0.5

#: Columns each answer needs. Missing ones are named rather than guessed.
METRIC_REQUIRED = ("spec_id", "dial", "metric", "generator", "score")
AGGREGATE_REQUIRED = ("spec_id", "dial", "generator", "G_dev")


class SchemaError(RuntimeError):
    """A sweep CSV is missing a column this report depends on."""


def latest_run_dir() -> Path:
    """Most recent sweep_<stamp>_<kind> directory under RESULTS_DIR."""
    candidates = sorted(path for path in RESULTS_DIR.glob("sweep_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No sweep run directories under {RESULTS_DIR}. "
            "Run `python -m fineval.scripts.sensitivity.run_sweep` first, "
            "or pass --run-dir."
        )
    return candidates[-1]


def load(run_dir: Path, filename: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(
            f"{path.name} is missing {missing}; it has {sorted(frame.columns)}. "
            "This report needs those columns and will not infer them."
        )
    if REFERENCE_SPEC not in set(frame["spec_id"]):
        raise SchemaError(
            f"{path.name} has no {REFERENCE_SPEC!r} spec, so there is no "
            "reference ordering to compare the other specs against."
        )
    return frame


def ranking(frame: pd.DataFrame, value: str, ascending: bool) -> list[str]:
    """Generators best-first, with NaN scores excluded and named by caller."""
    usable = frame.dropna(subset=[value])
    ordered = usable.sort_values(value, ascending=ascending, kind="stable")
    return list(ordered["generator"])


def inversions(reference: list[str], other: list[str]) -> list[tuple[str, str]]:
    """Ordered pairs whose relative order differs between two rankings."""
    position = {name: index for index, name in enumerate(other)}
    flipped = []
    for i, first in enumerate(reference):
        for second in reference[i + 1 :]:
            in_both = first in position and second in position
            if in_both and position[first] > position[second]:
                flipped.append((first, second))
    return flipped


def side_of_floor(score: float) -> str:
    if pd.isna(score):
        return "undefined"
    if score < NOISE_FLOOR:
        return "below"
    if score > NOISE_FLOOR:
        return "above"
    return "at"


def report_ordering(
    frame: pd.DataFrame, value: str, ascending: bool, label: str, group: str | None
) -> list[str]:
    """Ordering preservation for one metric, or for the aggregate."""
    lines: list[str] = []
    subsets = [(None, frame)] if group is None else list(frame.groupby(group, sort=False))
    for key, block in subsets:
        heading = label if key is None else f"{label} {key}"
        reference_rows = block[block["spec_id"] == REFERENCE_SPEC]
        reference = ranking(reference_rows, value, ascending)
        if not reference:
            lines.append(f"  {heading:<12} no finite {value} under {REFERENCE_SPEC}; skipped")
            continue
        lines.append(f"  {heading:<12} reference order: {' > '.join(reference)}")
        broken = []
        for spec_id, spec_rows in block.groupby("spec_id", sort=False):
            if spec_id == REFERENCE_SPEC:
                continue
            order = ranking(spec_rows, value, ascending)
            flipped = inversions(reference, order)
            if flipped:
                pairs = ", ".join(f"{a}<->{b}" for a, b in flipped)
                margins = []
                for a, b in flipped:
                    va = spec_rows.loc[spec_rows["generator"] == a, value]
                    vb = spec_rows.loc[spec_rows["generator"] == b, value]
                    if not va.empty and not vb.empty:
                        margins.append(abs(float(va.iloc[0]) - float(vb.iloc[0])))
                margin = f", closest margin {min(margins):.4g}" if margins else ""
                broken.append(f"    {spec_id}: {pairs}{margin}")
        if broken:
            lines.append(f"    ORDERING CHANGES ({len(broken)} spec(s)):")
            lines.extend(broken)
        else:
            n = block["spec_id"].nunique() - 1
            lines.append(f"    ordering preserved across all {n} non-primary spec(s)")
    return lines


def report_noise_floor(frame: pd.DataFrame) -> list[str]:
    """Cells whose reading against s = 0.5 changes under some spec."""
    lines: list[str] = []
    reference = frame[frame["spec_id"] == REFERENCE_SPEC]
    keyed = {(row.metric, row.generator): row.score for row in reference.itertuples(index=False)}
    changed: list[str] = []
    closest: list[tuple[float, str]] = []
    for row in frame.itertuples(index=False):
        if row.spec_id == REFERENCE_SPEC:
            continue
        base = keyed.get((row.metric, row.generator))
        if base is None:
            continue
        if not pd.isna(row.score):
            closest.append((abs(float(row.score) - NOISE_FLOOR), f"{row.metric}/{row.generator}"))
        before, after = side_of_floor(base), side_of_floor(row.score)
        if before != after:
            changed.append(
                f"    {row.spec_id}: {row.metric}/{row.generator} "
                f"{before} -> {after} (s {base:.4f} -> {row.score:.4f})"
            )
    for (metric, generator), score in sorted(keyed.items()):
        lines.append(
            f"  {metric}/{generator:<6} s={score:.4f} ({side_of_floor(score)} 0.5), "
            f"margin {abs(score - NOISE_FLOOR):.4f}"
        )
    if changed:
        lines.append(f"  INTERPRETATION CHANGES ({len(changed)}):")
        lines.extend(changed)
    else:
        lines.append("  no spec moves any cell across s = 0.5")
    if closest:
        margin, cell = min(closest)
        lines.append(f"  narrowest margin to 0.5 across all specs: {cell} at {margin:.4f}")
    return lines


def summarize(run_dir: Path) -> str:
    metric_frame = load(run_dir, METRIC_CSV, METRIC_REQUIRED)
    aggregate_frame = load(run_dir, AGGREGATE_CSV, AGGREGATE_REQUIRED)

    out = [
        f"Sensitivity robustness summary — {run_dir.name}",
        "=" * 72,
        f"specs: {metric_frame['spec_id'].nunique()}   "
        f"generators: {metric_frame['generator'].nunique()}   "
        f"metrics: {metric_frame['metric'].nunique()}",
        "",
        "1. Ordering preservation (per metric, generators ranked by score)",
        "   Higher score = smaller synthetic-real gap relative to the noise floor.",
    ]
    out += report_ordering(metric_frame, "score", ascending=False, label="metric", group="metric")
    out += [
        "",
        "2. Ordering preservation (aggregate, ranked by G_dev)",
        "   Lower G_dev = closer to real; G_dev = 1 is a perfect match.",
    ]
    out += report_ordering(aggregate_frame, "G_dev", ascending=True, label="aggregate", group=None)
    out += ["", "3. Interpretation relative to s = 0.5 (primary values, then any changes)"]
    out += report_noise_floor(metric_frame)
    out += [
        "",
        "Scope: point estimates only. run_sweep runs outer=0 and MC=0, so neither",
        "CSV carries an interval column and no finding above is tested against",
        "sampling noise. Margins are printed so narrow results read as narrow.",
    ]
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="sweep run directory; defaults to the most recent one",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or latest_run_dir()
    print(summarize(run_dir))


if __name__ == "__main__":
    main()
