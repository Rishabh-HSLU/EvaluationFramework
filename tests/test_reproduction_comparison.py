"""Reference integrity and reproduction comparison must detect meaningful changes."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from scripts.check_reproduction import compare_tables, main, verify_reference

REFERENCE = Path(__file__).resolve().parents[1] / "results/run4"


def test_comparison_matches_identifiers_not_row_order():
    reference = [
        {"metric": "M1", "generator": "AIL", "score": "0.34"},
        {"metric": "M2", "generator": "AIL", "score": "0.21"},
    ]
    result = compare_tables(
        reference, list(reversed(reference)), ("metric", "generator"), 1e-10, 1e-12
    )
    assert result["passed"]


def test_small_roundoff_is_reported_but_material_difference_fails():
    reference = [{"generator": "AIL", "score": "0.34", "n_valid_draws": "100"}]
    actual = [{"generator": "AIL", "score": "0.34000000000001", "n_valid_draws": "100"}]
    result = compare_tables(reference, actual, ("generator",), 1e-10, 1e-12)
    assert result["passed"]
    assert result["max_absolute_numeric_difference"] > 0
    actual[0]["score"] = "0.35"
    assert not compare_tables(reference, actual, ("generator",), 1e-10, 1e-12)["passed"]
    actual[0]["score"] = "0.34"
    actual[0]["n_valid_draws"] = "99"
    assert not compare_tables(reference, actual, ("generator",), 1, 1)["passed"]


def test_missing_and_duplicate_cells_are_rejected():
    reference = [{"generator": "AIL", "score": "0.34"}]
    with pytest.raises(ValueError, match="identifier sets"):
        compare_tables(reference, [], ("generator",), 1e-10, 1e-12)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_tables(reference, reference * 2, ("generator",), 1e-10, 1e-12)


def test_reference_bundle_and_tampering(tmp_path):
    if not REFERENCE.exists():
        pytest.skip("run #4 reference bundle is not available")
    result = verify_reference(REFERENCE)
    assert result["paper_number_entries_verified"] == 201
    copied = tmp_path / "reference"
    shutil.copytree(REFERENCE, copied)
    with (copied / "metric_summary.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_reference(copied)


def test_cli_rejects_different_run_settings(tmp_path, capsys):
    if not REFERENCE.exists():
        pytest.skip("run #4 reference bundle is not available")
    record = json.loads((REFERENCE / "provenance/run_manifest.json").read_text())
    record["run_number"] = 99
    record["seed"] = 123
    manifest = tmp_path / "runs_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    assert main(["--manifest", str(manifest), "--run-number", "99"]) == 1
    assert "Run setting differs: seed" in capsys.readouterr().out


def test_point_only_accepts_o0_m0_and_detects_changed_score(tmp_path, capsys):
    reference = REFERENCE
    record = json.loads((reference / "provenance/run_manifest.json").read_text())
    record.update(run_number=99, n_outer_resamples=0, n_mc_repeats=0)
    for field, name in [
        ("results_csv", "metric_summary.csv"),
        ("aggregate_csv", "aggregate_summary.csv"),
    ]:
        record[field] = name
        shutil.copy2(reference / name, tmp_path / name)
    manifest = tmp_path / "runs_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    args = ["--manifest", str(manifest), "--run-number", "99", "--point-only"]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["comparison_scope"] == "point_estimates_only"
    assert len(report["comparisons"]) == 2
    p = tmp_path / "metric_summary.csv"
    text = p.read_text().replace("0.34359018272044994", "0.44", 1)
    p.write_text(text)
    assert main(args) == 1
    assert not json.loads(capsys.readouterr().out)["passed"]
