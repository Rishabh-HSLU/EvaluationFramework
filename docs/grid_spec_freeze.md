# Sensitivity grid — pre-registration freeze

Status: committed on merged `main`; the grid has not been run and nothing is
tagged yet. Two rationales below are explicitly marked **PLACEHOLDER — Rish's
own words needed** and must be filled before this file is cited as a
pre-registration.

Source of truth for every value below: `fineval/scripts/sensitivity/grid.py`
at the merged trial state (tree identical to commit `0855491`). Provenance
classifications are taken verbatim from `docs/parameter_provenance_audit.md`
(audit date 2026-07-31); nothing is reclassified here.

---

## 1. Scope of the freeze

This file freezes **which specifications will be run**, not their results. It
covers the one-at-a-time (OAT) sensitivity grid only. The canonical benchmark
configuration (`spec = primary`) is frozen by `fineval/config.py` and is
reproduced in the grid as a self-contained table, so a divergence between the
two is a test failure (`tests/test_spec_flag.py`), not a silent change.

Run kind at the time of drafting: `RUN_KIND = "exploratory"` (grid.py line 39).
This string is written into the run directory name
(`results/sweep_<stamp>_exploratory/`), into `grid_spec.json`, and into every
manifest row. **PLACEHOLDER — Rish's own words needed.** Changing it to a confirmatory
label is a decision for Rish and the supervisor, not a code cleanup; `grid.py`
and `RUN_KIND` are deliberately untouched by this commit.

## 2. Primary (canonical) specification

| parameter | value | metric | provenance (audit) |
|---|---|---|---|
| `n_grid` | 5001 | M1 | UNDOCUMENTED |
| `tail_alpha` | 0.3 | M1 | **AIL-INFORMED** |
| `tail_lambda` | 1.0 | M1 | **AIL-INFORMED** (via the sweep; final pick conventional) |
| `window` | 60 | M4 | **AIL-INFORMED** |
| `min_frac` | 1/2 | M4 (derives `min_periods`) | **AIL-INFORMED** |
| `min_periods` | 30 | M4 | derived = `ceil(window × min_frac)`; inherits AIL-INFORMED |
| `n_regimes` | 5 | M4 | UNDOCUMENTED |
| `tail_fraction` | 0.05 | M4 | UNDOCUMENTED |
| `regime_weights` | (1, 1, 1, 2, 3) raw | M4 | AMBIGUOUS (direction a-priori; magnitudes undocumented) |

Not exposed as grid dials but fixed for every spec: `M2_LAG_MIN = 60`
(INDEPENDENT), `M2_LAG_MAX = 388` (INDEPENDENT, structural),
`M3_SCALES = [1, 5, 15, 30]` (UNDOCUMENTED origin), `M3_MIN_OBS = 100`
(UNDOCUMENTED), `M3_SUPPORT_ANCHOR = "close"` (UNDOCUMENTED).

## 3. The 19 specifications

| # | spec_id | dial(s) | changed value(s) | notes |
|---|---|---|---|---|
| 1 | `primary` | — | canonical | reference for every OAT comparison |
| 2 | `tail_alpha=0.00` | tail_alpha | 0.00 | recovers plain W1 weighting |
| 3 | `tail_alpha=0.15` | tail_alpha | 0.15 | |
| 4 | `tail_alpha=0.50` | tail_alpha | 0.50 | |
| 5 | `tail_alpha=0.75` | tail_alpha | 0.75 | |
| 6 | `tail_alpha=0.90` | tail_alpha | 0.90 | **not in the PDF's list** |
| 7 | `tail_lambda=0.0` | tail_lambda | 0.0 | **not in the PDF's list**; removes tail emphasis |
| 8 | `tail_lambda=0.5` | tail_lambda | 0.5 | |
| 9 | `tail_lambda=2.0` | tail_lambda | 2.0 | |
| 10 | `window=30` | window | 30 (min_periods 15) | |
| 11 | `window=120` | window | 120 (min_periods 60) | |
| 12 | `tail_fraction=0.025` | tail_fraction | 0.025 | |
| 13 | `tail_fraction=0.100` | tail_fraction | 0.100 | |
| 14 | `min_frac=0.333` | min_frac | 1/3 (min_periods 20) | **not in the PDF's list** |
| 15 | `min_frac=0.667` | min_frac | 2/3 (min_periods 40) | **not in the PDF's list** |
| 16 | `regime_weights=flat` | regime_weights | (1,1,1,1,1) | |
| 17 | `corner=alpha0.75-lambda2.0` | tail_alpha + tail_lambda | 0.75, 2.0 | **corner, not in the PDF** |
| 18 | `corner=win120-frac0.667` | window + min_frac | 120, 2/3 (min_periods 80) | **corner, not in the PDF** |
| 19 | `corner=tailf0.10-flat` | tail_fraction + regime_weights | 0.100, flat | **corner, not in the PDF** |

Swept ranges as implemented: `tail_alpha ∈ {0, 0.15, 0.3*, 0.5, 0.75, 0.90}`,
`tail_lambda ∈ {0, 0.5, 1.0*, 2.0}`, `window ∈ {30, 60*, 120}`,
`min_frac ∈ {1/3, 1/2*, 2/3}`, `tail_fraction ∈ {0.025, 0.05*, 0.10}`,
`regime_weights ∈ {(1,1,1,2,3)*, flat}` (`*` = primary value, collapses onto
spec 1 and is not re-run).

Corners exist so the OAT no-interaction assumption can be tested rather than
asserted; `Spec.dials` is the authoritative record of what each spec varies.

## 4. Parameters deliberately NOT varied, and why

| parameter | value | stated rationale |
|---|---|---|
| `n_grid` | 5001 | Quantile-grid resolution, not a modelling choice; no recorded basis for 5001 specifically (UNDOCUMENTED), and the metric is a weighted mean over the grid, so resolution affects numerical precision rather than the comparison. **PLACEHOLDER — Rish's own words needed** before this is defended in print. |
| `n_regimes` | 5 | Regime count fixes the meaning of `regime_weights`; varying both would confound the weighting question with the partition question. UNDOCUMENTED origin. |
| `B` (inner draws) | 100 | Controls Monte-Carlo error of the estimator, not the estimand; it is a precision dial reported separately, not a specification choice. |
| `m` (tickers/draw) | 200 | Same: a sampling-precision dial. Also fixed so the real-real noise floor is comparable across specs. |
| raw vs standardized returns (M4) | raw (Option B) | Recorded in `scripts/reasoning.md` as an AIL-informed structural decision; a sweep over it would be a different metric, not a sensitivity of this one. |
| rolling vs EWMA volatility proxy (M4) | rolling | Recorded as AIL-informed; the proxy *family* was selected, the window is what the grid varies. |
| M2 lag window | 60–388 | Structural: 388 is the longest lag with within-session support. |
| M3 scales / support anchor | [1,5,15,30] / close | Changing them changes the common support (360 returns/session at lcm = 30) and therefore the estimand. |

## 5. Seeds

The harness accepts a **single** `--seed` (default 42), applied identically to
every spec so that specs are paired on the same draw indices. Multi-seed
support does not exist in `run_sweep.py` and would have to be added (an outer
loop over seeds, or repeated invocations with distinct `--seed` and a seed
column in the manifest). **Decision needed** (handoff report, decision 3).

Proposed, if multi-seed is wanted: root seeds `{42, 1, 2, 3, 4}` (5 seeds,
matching the existing 8-seed audit's convention of 42 plus small integers), with
42 as the anchor reported in the main table.

## 6. What a freeze commit would contain

1. This file at `docs/grid_spec_freeze.md`.
2. The tagged commit hash of merged `main` (the grid file's content hash is
   implied by the tag).
3. No results. `grid_spec.json` is emitted per run by `run_sweep.py` and
   already records `git_commit`, `seed`, `mode`, the full spec list and the
   SHA-256 of every curated panel — that file, not this one, is the per-run
   provenance record.
