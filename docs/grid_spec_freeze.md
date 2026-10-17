# Frozen sensitivity grid

Status update, 17 October 2026: the five-seed analysis is running with the exact
run #4 panels. Completion and validation are pending. No sensitivity results
are claimed here.

Source of truth: [grid.py](../fineval/scripts/sensitivity/grid.py), as recorded
in execution commit `b9f404c`. The grid contains 19 specifications: the primary
configuration, 15 single-parameter variations and three joint variations.
The specifications remain unchanged during this run.

## Interpretation

This is an exploratory robustness analysis. Several settings were selected
while AIL output was visible; the grid does not establish that these choices
were independent of the evaluated generators. The
[historical parameter audit](history/parameter_provenance_audit.md) records the
selection evidence.

The analysis compares point estimates across specifications and seeds. It does
not calculate an O=200 outer-bootstrap interval for each alternative. The
primary paper intervals remain those from [run #4](../results/run4/README.md).

## Primary settings

| Parameter | Value | Selection evidence |
|---|---|---|
| M1 quantile-grid size | 5001 | Original selection rationale not recorded. |
| M1 tail exponent | 0.3 | Selected with AIL output visible. |
| M1 tail-weight scale | 1.0 | Compared with AIL output visible; final unit scale described as conventional. |
| M4 volatility window | 60 minutes | Selected with AIL output visible. |
| M4 minimum-observation fraction | 1/2 | Selected using gaps and AIL coverage. |
| M4 minimum observations | 30 | Derived as ceil(window × fraction). |
| M4 regimes | 5 | Original selection rationale not recorded. |
| M4 tail fraction | 0.05 | Original selection rationale not recorded. |
| M4 regime weights | (1, 1, 1, 2, 3), normalized | Greater weight on high-volatility regimes was intended; the specific magnitudes have no recorded rationale. |

M2 uses lags 60–388. M3 uses scales 1, 5, 15 and 30, a minimum of 100
observations per scale, and common support anchored at the session close.
These settings are fixed across the grid.

## Specifications

Each variation starts from the primary configuration. Only the listed
parameters and their derived values change.

| Specification ID | Change from primary |
|---|---|
| `primary` | None |
| `tail_alpha=0.00` | M1 exponent 0.00 |
| `tail_alpha=0.15` | M1 exponent 0.15 |
| `tail_alpha=0.50` | M1 exponent 0.50 |
| `tail_alpha=0.75` | M1 exponent 0.75 |
| `tail_alpha=0.90` | M1 exponent 0.90 |
| `tail_lambda=0.0` | M1 tail-weight scale 0.0 |
| `tail_lambda=0.5` | M1 tail-weight scale 0.5 |
| `tail_lambda=2.0` | M1 tail-weight scale 2.0 |
| `window=30` | M4 window 30; minimum observations 15 |
| `window=120` | M4 window 120; minimum observations 60 |
| `tail_fraction=0.025` | M4 tail fraction 0.025 |
| `tail_fraction=0.100` | M4 tail fraction 0.100 |
| `min_frac=0.333` | M4 fraction 1/3; minimum observations 20 |
| `min_frac=0.667` | M4 fraction 2/3; minimum observations 40 |
| `regime_weights=flat` | Equal weights across the five M4 regimes |
| `corner=alpha0.75-lambda2.0` | M1 exponent 0.75 and tail-weight scale 2.0 |
| `corner=win120-frac0.667` | M4 window 120 and fraction 2/3; minimum observations 80 |
| `corner=tailf0.10-flat` | M4 tail fraction 0.100 and equal regime weights |

The joint variations examine three selected parameter combinations; they do
not exhaust all interactions.

## Execution and comparison

All four generators are included: AIL, GBM, MSV and SBBTS. The five curated
panels are identical to those used by run #4, including its SBBTS panel.
Their identities are recorded in [panel_digests.txt](panel_digests.txt).

Each specification uses B=100 draws and 200 tickers per draw. The analysis uses
root seeds 42, 1, 2, 3 and 4. Specifications share the same draw indices within
each seed. The seed-42 primary metric scores must match the run #4 reference
before the remaining specifications proceed.

For one seed, run from the repository root:

```bash
uv run --frozen python -m scripts.reproduce sensitivity -- \
    --fast --n-resamples 100 --tickers-per-draw 200 \
    --seed 42 --n-jobs 4 --run-dir results/sensitivity/seed_42 \
    --primary-reference results/run4/metric_summary.csv
```

Repeat with seeds 1, 2, 3 and 4, using a separate output directory for each;
the primary-reference comparison applies only to seed 42. Fast mode reuses
metrics whose parameters are unchanged and recalculates the aggregate score.
It uses the same metric implementations as a full sweep.

Each seed directory records its resolved specifications, input and source
hashes, environment, status and result tables. Earlier September sweeps lack
a recorded SBBTS input hash and are not the current reporting reference.

## Scope of the analysis

The grid holds preprocessing, the M4 volatility-proxy family, M3 support and
scales, quantile-grid size, regime count, B and ticker sample size fixed.
It therefore assesses the listed parameter variations, not every metric or
preprocessing design choice. In particular, it does not compare rolling
volatility with EWMA or establish robustness to generator-informed structural
choices outside the grid.
