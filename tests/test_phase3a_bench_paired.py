"""
Phase 3a regression — bench.run_benchmark() must reproduce the paired-bootstrap
CIs published in runs/<...>_paired_vs_unpaired/results.csv (the Phase 3 table).

Spot-checks three cells covering the dynamic range of width_reduction:
    B1 / AIL    (negative reduction, near-zero corr)
    B3 / SFAGan (small positive reduction)
    B6 / GARCH  (largest positive reduction in the table)
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import run_benchmark  # noqa: E402

PHASE3_CSV = (
    ROOT / "runs" / "20260519T091149Z_paired_vs_unpaired" / "results.csv"
)

# (bucket, generator) — chosen to span the reduction-percent range
SPOT_CELLS = [
    ("B1", "AIL"),
    ("B3", "SFAGan"),
    ("B6", "GARCH"),
]


def _load_phase3() -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with PHASE3_CSV.open() as f:
        for row in csv.DictReader(f):
            out[(row["bucket"], row["generator"])] = {
                "sim":  float(row["sim_point"]),
                "lo":   float(row["ci_paired_lo"]),
                "hi":   float(row["ci_paired_hi"]),
            }
    return out


@unittest.skipUnless(PHASE3_CSV.exists(), f"Phase 3 CSV missing: {PHASE3_CSV}")
class TestBenchPairedMatchesPhase3(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.phase3 = _load_phase3()
        cls.bench  = run_benchmark()

    def _check_cell(self, bkey: str, gname: str) -> None:
        ref = self.phase3[(bkey, gname)]
        sim, lo, hi = self.bench[bkey][gname]
        with self.subTest(cell=f"{bkey}/{gname}"):
            np.testing.assert_allclose(sim, ref["sim"], atol=0.001,
                err_msg=f"{bkey}/{gname} sim mismatch")
            np.testing.assert_allclose(lo,  ref["lo"],  atol=0.001,
                err_msg=f"{bkey}/{gname} ci_lo mismatch")
            np.testing.assert_allclose(hi,  ref["hi"],  atol=0.001,
                err_msg=f"{bkey}/{gname} ci_hi mismatch")

    def test_b1_ail(self) -> None:
        self._check_cell("B1", "AIL")

    def test_b3_sfagan(self) -> None:
        self._check_cell("B3", "SFAGan")

    def test_b6_garch(self) -> None:
        self._check_cell("B6", "GARCH")


if __name__ == "__main__":
    unittest.main()
