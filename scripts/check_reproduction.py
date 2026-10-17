"""Verify the paper's saved evidence and optionally compare a reproduced run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "results_csv": ("metric_summary.csv", ("metric", "generator")),
    "aggregate_csv": ("aggregate_summary.csv", ("generator",)),
    "outer_metric_replicates_csv": (
        "outer_metric_replicates.csv",
        ("outer_id", "metric", "generator"),
    ),
    "outer_aggregate_replicates_csv": ("outer_aggregate_replicates.csv", ("outer_id", "generator")),
    "mc_metric_replicates_csv": ("mc_metric_replicates.csv", ("repeat_id", "metric", "generator")),
    "mc_aggregate_replicates_csv": ("mc_aggregate_replicates.csv", ("repeat_id", "generator")),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict], columns: tuple[str, ...]) -> dict:
    result = {tuple(row[column] for column in columns): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate result identifiers: {columns}")
    return result


def close(left, right, rtol=2e-12, atol=2e-14) -> bool:
    return (
        math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
    )


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (position - low) * (values[high] - values[low])


def compare_tables(reference: list[dict], actual: list[dict], columns, rtol, atol) -> dict:
    """Compare by statistical identifiers, allowing stated numerical tolerances."""
    ref = keyed(reference, columns)
    other = keyed(actual, columns)
    if ref.keys() != other.keys():
        raise ValueError(f"Result identifier sets differ for {columns}")
    differences = []
    max_absolute = 0.0
    for key, row in ref.items():
        if not row.keys() <= other[key].keys():
            raise ValueError(f"Reproduced output is missing columns for {key}")
        for column, value in row.items():
            new = other[key][column]
            try:
                number, other_number = float(value), float(new)
            except ValueError:
                same = value == new
            else:
                same = close(number, other_number, rtol, atol)
                if math.isfinite(number) and math.isfinite(other_number):
                    max_absolute = max(max_absolute, abs(number - other_number))
                if column.startswith(("n_", "k_")) or column in ("seed", "inner_seed"):
                    same = number == other_number
            if not same:
                differences.append(
                    {"key": key, "column": column, "reference": value, "actual": new}
                )
    return {
        "passed": not differences,
        "rows": len(ref),
        "max_absolute_numeric_difference": max_absolute,
        "difference_count": len(differences),
        "examples": differences[:10],
    }


def verify_reference(directory: Path) -> dict:
    manifest = json.loads((directory / "checksums.json").read_text())
    for name, wanted in manifest["artifacts"].items():
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError(f"Reference file digest mismatch: {name}")
    tables = {name: read_csv(directory / name) for name, _ in FILES.values()}
    counts = {
        "metric_summary.csv": 16,
        "aggregate_summary.csv": 4,
        "outer_metric_replicates.csv": 3200,
        "outer_aggregate_replicates.csv": 800,
        "mc_metric_replicates.csv": 320,
        "mc_aggregate_replicates.csv": 80,
    }
    for name, columns in FILES.values():
        rows = tables[name]
        if len(rows) != counts[name]:
            raise ValueError(f"Wrong row count: {name}")
        keyed(rows, columns)
        for row in rows:
            if "n_valid_draws" in row:
                if row["n_valid_draws"] != "100" or row["n_total_draws"] != "100":
                    raise ValueError(f"Incomplete matched draws: {name}")
                rr, sr = float(row["g_rr_mean"]), float(row["g_sr_mean"])
                if rr <= 0 or sr <= 0 or not close(row["score"], rr / (rr + sr)):
                    raise ValueError(f"Invalid saved gap ratio: {name}")
            elif not all(
                row[k] == v
                for k, v in [("k_used", "4"), ("k_total", "4"), ("min_valid_draws", "100")]
            ):
                raise ValueError(f"Incomplete aggregate: {name}")
    # Independently recover intervals and sample standard deviations.
    for prefix, id_column, count in [("outer", "outer_id", 200), ("mc", "repeat_id", 20)]:
        for category, value in [("metric", "score"), ("aggregate", "G_dev")]:
            source = tables[f"{prefix}_{category}_replicates.csv"]
            summary = tables[f"{category}_summary.csv"]
            group = ("metric", "generator") if category == "metric" else ("generator",)
            if {int(row[id_column]) for row in source} != set(range(count)):
                raise ValueError(f"Incomplete {prefix} replicate IDs")
            for row in summary:
                selected = [x for x in source if all(x[k] == row[k] for k in group)]
                if len(selected) != count:
                    raise ValueError(f"Incomplete {prefix} cell")
                values = [float(x[value]) for x in selected]
                if prefix == "outer":
                    expected = {
                        "outer_mean": statistics.mean(values),
                        "outer_sd": statistics.stdev(values),
                        "ci_low": quantile(values, 0.025),
                        "ci_high": quantile(values, 0.975),
                        "n_outer_valid": count,
                    }
                else:
                    expected = {
                        "mc_mean": statistics.mean(values),
                        "mc_sd": statistics.stdev(values),
                        "mc_min": min(values),
                        "mc_max": max(values),
                        "mc_q025": quantile(values, 0.025),
                        "mc_q975": quantile(values, 0.975),
                        "n_mc_valid": count,
                    }
                if not all(close(row[k], v) for k, v in expected.items()):
                    raise ValueError(f"Incorrect {prefix} summary: {group}")
            # Check each aggregate from the four saved mean-gap ratios.
            if category == "metric":
                aggregate = tables[f"{prefix}_aggregate_replicates.csv"]
                for row in aggregate:
                    selected = [
                        x
                        for x in source
                        if x[id_column] == row[id_column] and x["generator"] == row["generator"]
                    ]
                    result = math.exp(
                        statistics.mean(
                            abs(math.log(float(x["g_sr_mean"]) / float(x["g_rr_mean"])))
                            for x in selected
                        )
                    )
                    if len(selected) != 4 or not close(row["G_dev"], result):
                        raise ValueError(f"Incorrect {prefix} aggregate")
    # Resolve every paper-number index entry to its source cell or JSON pointer.
    index = read_csv(directory / "paper_number_map.csv")
    for entry in index:
        path = directory / entry["source_file"]
        if path.suffix == ".csv":
            rows = tables.get(path.name, read_csv(path))
            keys = json.loads(entry["source_key"])
            selected = [row for row in rows if all(row[k] == str(v) for k, v in keys.items())]
            if len(selected) != 1:
                raise ValueError(f"Ambiguous paper number: {entry['quantity']}")
            value = float(selected[0][entry["source_column"]])
        else:
            value = json.loads(path.read_text())
            for part in entry["source_pointer"].split("."):
                value = value[int(part)] if isinstance(value, list) else value[part]
        if not close(value, json.loads(entry["exact_value"])):
            raise ValueError(f"Paper source value mismatch: {entry['quantity']}")
        places = entry["decimal_places"]
        if places and entry["display"] != f"{float(value):.{int(places)}f}":
            raise ValueError(f"Paper rounding mismatch: {entry['quantity']}")
    return {
        "reference_files_verified": len(manifest["artifacts"]),
        "paper_number_entries_verified": len(index),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "results/run4")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/runs_manifest.csv")
    parser.add_argument("--run-number", type=int, help="completed reproduced run to compare")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)
    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be nonnegative")
    try:
        report = verify_reference(args.reference_dir)
        if args.run_number is not None:
            selected = [
                row for row in read_csv(args.manifest) if int(row["run_number"]) == args.run_number
            ]
            if len(selected) != 1 or selected[0]["status"] != "completed":
                raise ValueError("Choose exactly one completed run from the manifest")
            row = selected[0]
            expected = json.loads((args.reference_dir / "provenance/run_manifest.json").read_text())
            for key in [
                "spec",
                "n_resamples",
                "tickers_per_draw",
                "seed",
                "n_outer_resamples",
                "n_mc_repeats",
            ]:
                if str(row[key]) != str(expected[key]):
                    raise ValueError(f"Run setting differs: {key}")
            comparisons = {}
            for field, (name, columns) in FILES.items():
                comparisons[name] = compare_tables(
                    read_csv(args.reference_dir / name),
                    read_csv(args.manifest.parent / row[field]),
                    columns,
                    args.rtol,
                    args.atol,
                )
            report.update(
                {
                    "run_number": args.run_number,
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "comparisons": comparisons,
                    "passed": all(x["passed"] for x in comparisons.values()),
                }
            )
        else:
            report["passed"] = True
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
