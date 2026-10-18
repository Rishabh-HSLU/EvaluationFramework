"""The published grid must remain read-only and detect material inconsistencies."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.check_reproduction import verify_sensitivity

REFERENCE = Path(__file__).resolve().parents[1] / "results/sensitivity"


@pytest.fixture
def reference(tmp_path):
    directory = tmp_path / "reference"
    directory.mkdir()
    manifest = json.loads((REFERENCE / "reference_checksums.json").read_text())
    for name in manifest["files"]:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REFERENCE / name, target)
    shutil.copy2(REFERENCE / "reference_checksums.json", directory / "reference_checksums.json")
    return directory


def change_cell(directory, filename, column, value):
    path = directory / filename
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0][column] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Refreshing a digest must not conceal an inconsistent numerical summary.
    checksums = directory / "reference_checksums.json"
    manifest = json.loads(checksums.read_text())
    manifest["files"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    checksums.write_text(json.dumps(manifest))


def test_grid_reference_is_verified_without_writing(reference):
    before = {p.relative_to(reference): p.read_bytes() for p in reference.rglob("*") if p.is_file()}
    report = verify_sensitivity(reference)
    assert report["metric_rows_verified"] == 1520
    assert report["aggregate_rows_verified"] == 380
    assert report["paper_number_entries_verified"] == 36
    assert all(x["changed_orderings"] == 0 for x in report["ordering_changes"].values())
    assert before == {
        p.relative_to(reference): p.read_bytes() for p in reference.rglob("*") if p.is_file()
    }


def test_grid_reference_rejects_file_tampering(reference):
    with (reference / "metric_results.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_sensitivity(reference)


@pytest.mark.parametrize(
    ("filename", "column", "value", "message"),
    [
        ("metric_results.csv", "score", "0.9", "Combined sensitivity table"),
        ("metric_seed_summary.csv", "seed_sd", "0.9", "seed summary"),
        ("effect_ranges.csv", "max_change", "0.9", "effect range"),
        ("seed_4/sweep_metric_results.csv", "n_valid_draws", "99", "draws are invalid"),
    ],
)
def test_numerical_inconsistency_fails_even_after_digest_refresh(
    reference, filename, column, value, message
):
    change_cell(reference, filename, column, value)
    with pytest.raises(ValueError, match=message):
        verify_sensitivity(reference)
