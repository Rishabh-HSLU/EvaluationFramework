# TODO

## FFF drops every session's close bar (16:00 ET)

**Observation (2026-07-08, not yet fixed):** Discovered while
deduplicating session-clock logic across `FFFDeseasonalizer`,
`PreprocessingPipeline`, and `scripts/baseline_generation.py` (now
consolidated into `fineval/preprocessing/session_clock.py`, behavior
otherwise unchanged). `SESSION_OFFSET = 570` maps the first bar of a
session (09:31 ET) to minute position 1, not 0 as the old
`FFFDeseasonalizer` docstring claimed — but `fit()` and `transform()`
both treat the valid slot range as `[0, trading_minutes - 1]` =
`[0, 389]`. Since actual bars occupy minute positions `[1, 390]`
(09:31 → 16:00), this excludes position 390 — the session close —
from the fitted volatility smile. `transform()` then produces NaN for
every 16:00 bar in any FFF-processed series.

Confirmed against the real curated data: 137/137 session-close (16:00)
bars are entirely NaN in `deseas_real` post-FFF, vs. 0/139 for a
mid-session control timestamp (12:00). Only affects series that go
through FFF (Real, AIL); GBM and MSV skip it (flat CV) and are
unaffected — an asymmetry between generators that itself wasn't
previously documented anywhere.

Magnitude: 137 of ~53,850 timestamps (~0.25%), dropped consistently at
the same point in every session, across all 600 tickers at once. Every
metric already tolerates NaN, so nothing crashes or silently
miscomputes — this is extra data loss, not data corruption — but it is
unintentional (contradicts the FFF docstring's own stated design) and
asymmetric across generators, so it isn't neutral either.

Options to consider:

1. Fix the slot range to `[1, trading_minutes]` (or shift
   `SESSION_OFFSET` by one) so the close bar is retained. Changes
   `deseas_real`/`deseas_synthetic` for every FFF-processed dataset —
   would require re-running the benchmark and updating every published
   score/CI, however slightly.
2. Leave as-is and document it as a known, small, generator-asymmetric
   data-loss quirk in the manuscript's preprocessing section.
3. Investigate whether it interacts with anything session-boundary
   sensitive downstream (M2's ACF, M6's rolling vol both restart at
   session boundaries and could be mildly affected near session end).

Decision affects: `deseas_real`/`deseas_synthetic` for Real and AIL
only (not GBM/MSV), every downstream metric and benchmark number if
fixed, and the FFF section of `reasoning.md`/the mathematical
foundations doc.

## M1: decide whether shape should be compared at a common scale

**Observation (2026-07-02, not yet changed):** M1's score for FFF-skipped
generators (GBM, MSV) is dominated by a *scale* gap rather than tail shape.
The conditional-FFF preprocessing divides seasonal series (Real, AIL) by their
per-minute seasonal volatility, leaving them approximately unit-variance —
while flat generators (GBM, MSV) skip FFF and stay at raw 1-minute scale
(~1e-3). M1's tail-weighted Wasserstein-1 then compares quantile functions
living on scales three orders of magnitude apart, so the distance collapses to
the weighted mean of |real quantiles| regardless of the synthetic tail shape.

Evidence: GBM and MSV get nearly identical M1 distances (2.7112 vs 2.7124)
despite very different marginals (Gaussian vs heavy-tailed lognormal vol
mixture). This matches the historically documented Real–GBM M1 distance
(2.69 in scripts/reasoning.md), so it is existing framework behavior, not a
regression.

Options to consider:

1. Standardize every panel to unit variance before M1 (compare shape only;
   scale errors would then need their own diagnostic axis).
2. Keep as-is and document that M1 conflates scale + shape for FFF-skipped
   generators.
3. Score scale and shape as separate terms inside M1.

Decision affects: M1 scores for GBM/MSV columns, reasoning.md M1 section,
and any generator whose intraday variance profile is flat (CV < 0.3).

## M2: confidence interval runs narrow under reseeding

**Observation (2026-07-08, not yet resolved):** An 8-seed sensitivity
sweep (`fineval/scripts/notebooks/seed_sensitivity.ipynb`, B=100,
200 tickers/draw) found M2's paired-bootstrap CI does not reliably
bound the score's actual spread across different seeds — the other
three implemented metrics do. Leave-one-out coverage (every seed used
as its own anchor, averaged): M1 0.88, M2 0.61, M4 0.80, M6 0.83.
Confirming diagnostic: point-estimate range / CI width is ≥ 0.94 for
all three of M2's generator cells (AIL 1.13, GBM 1.04, MSV 0.94) — the
true spread barely fits, or doesn't fit, inside the reported interval —
while every other metric/generator cell sits at 0.41–0.83.

Full write-up: `scripts/reasoning.md`, "Seed-Sensitivity Audit: M2's
Confidence Interval Runs Narrow (2026-07-08)".

Two leading candidate mechanisms, not yet distinguished (no per-draw
g_rr/g_sr arrays are persisted, so neither can be tested without new
instrumentation):

1. Cross-ticker heterogeneity in volatility-clustering persistence that
   a single seed's fixed 200-ticker draws can't see.
2. The 331 summed lags in `compute_distance` are strongly
   autocorrelated, not independent, so the sum converges to Gaussian
   more slowly than its term count suggests — percentile-bootstrap CIs
   are known to under-cover skewed statistics at modest B.

Confirmed and ruled out: increasing B alone will not fix this — CI
width and the true across-seed spread both scale as ~1/√B, so a larger
B likely preserves the same relative under-coverage at a smaller
absolute scale (follows from standard bootstrap theory, not
independently tested).

Options to consider:

1. Persist per-draw g_rr/g_sr arrays (at least for M2) to directly test
   the two mechanisms above — check skew/autocorrelation across the
   331 lags, or rerun a v1-style within-ticker/within-regime variance
   decomposition scoped to M2.
2. Switch M2's CI to a bias-corrected (BCa) bootstrap instead of plain
   percentile, which handles skewed statistics better regardless of
   which mechanism dominates.
3. Do neither, and instead state plainly in the manuscript that M2's
   reported CI is empirically anti-conservative by roughly this
   magnitude — cheapest option, honest, defensible given the sweep
   already exists as a versioned artifact.

Decision affects: M2's CI in every benchmark table, the "Bootstrap
foundations" section of the mathematical-foundations doc (validity/
stationarity assumptions), and whether the CI-honesty remark needs a
per-metric caveat rather than a blanket claim.
