"""SBBTS: Schrodinger-Bass Bridge for Time Series.

Standalone drift-network and training-loop implementation following
Alouadi et al., "SBBTS: A Unified Schrodinger-Bass Framework for Synthetic
Financial Time Series" (arXiv:2604.07159), Section 4 and Appendix B/C.

Not wired into `fineval` or the main benchmark pipeline. See `reasoning.md`
for the design decisions made where the paper leaves a choice open (beta
calibration, validation split, batching) or where our data (irregular
1-minute market-clock grids with missing bars) required an adaptation
beyond the paper's regular-grid experiments.
"""
