# TODO

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
