# Parameter-selection evidence

This record summarizes the July 2026 parameter audit. It retains the selection
evidence needed to interpret the sensitivity analysis, including choices made
while AIL output was visible. Source locations below refer to the files at the
audit date; the [research notes](../../scripts/reasoning.md) preserve the
underlying experiments.

The [current frozen grid](../grid_spec_freeze.md) contains 19 specifications.
It replaces the earlier 13-specification design discussed in the original
audit. This page records selection provenance; it is not a reproduction guide.

Classes: **AIL-INFORMED** means AIL scores, distances or coverage informed the
choice; **INDEPENDENT** records a stated basis without generator-output tuning;
**UNDOCUMENTED** means no selection rationale was found; **AMBIGUOUS** means
the evidence was insufficient to establish either classification. Qualifiers
in the table distinguish documented facts from unverified claims.

## Parameters and selection evidence

| Parameter | Value | Consumed by | Basis (as recorded) | Class | Evidence |
|---|---|---|---|---|---|
| `SEED` | 42 | `bootstrap/engine.py:43`, `run_benchmark.py:21`, `baseline_generation.py` | Convention; later audited with an 8-seed sweep | INDEPENDENT | `reasoning.md:1141` "Every benchmark run so far used `seed = 42`"; seed-audit section 1141–1223 |
| `TRADING_MINUTES` | 390 | `preprocessing/pipeline.py:42`, `metrics/aggregational_gaussianity.py:99`, `session_clock.py` | NYSE session structure (09:30–16:00), a market fact | INDEPENDENT | `config.py:18` comment; `session_clock.py:10-11` |
| `SEASONALITY_CV_THRESHOLD` | 0.3 | `preprocessing/pipeline.py:95` (FFF gate) | Set between empirical CV clusters of flat (~0.05) vs. seasonal (~0.6+) series; AIL's CV (0.974) was among the values consulted to validate the split | **AMBIGUOUS** | `reasoning.md:234-243`: "A threshold of 0.3 robustly separates the two regimes… Validated empirically: real CV = 0.944, AIL CV = 0.974, GBM CV ≈ 0." AIL's *CV* was consulted, but not AIL's scores/distances/ratios — doesn't cleanly fit either AIL-INFORMED or INDEPENDENT |
| `NUM_HARMONICS` | 4 | `preprocessing/pipeline.py:42` (FFF) | Config comment **claims** "BIC-selected; Andersen & Bollerslev (1997)"; no recorded BIC procedure found anywhere (reasoning.md has zero mentions of harmonics/BIC) | **AMBIGUOUS** (claim of INDEPENDENT, unverifiable) | `config.py:24`; grep of `reasoning.md` for "harmonic\|BIC" → no hits |
| `MARKET_CALENDAR` | "XNYS" | `data/curate.py:160` | Data definition (NYSE) | INDEPENDENT | `config.py:27` comment |
| `BAR_FREQUENCY` | "1min" | `data/curate.py:162` | Data definition (native bar frequency) | INDEPENDENT | `config.py:28` comment |
| `COVERAGE_FLOOR` | 0.70 | `data/curate.py:107` (ticker retention) | Claimed statistical-power reasoning (≥37,700 valid returns/ticker); applied to `min(real, AIL)` coverage but the value itself not tuned on generator scores | INDEPENDENT (claim) | `reasoning.md:152-160`: "The 70% floor is set by statistical-power considerations… 70% is where marginal ticker quality and marginal estimation precision both remain favorable." |
| `M1_N_GRID` | 5001 | `benchmark/config.py:46` → `TailWeightedMarginal` | No recorded basis for 5001 specifically. `mathematical_foundations.md:194` defines the midpoint grid with n = 5001 but never argues the resolution choice; introduced at this value in `c7d93ae`, never changed | UNDOCUMENTED | `git log -S"M1_N_GRID"` → single commit `c7d93ae` ("added metrics and updated reasoning.md"); config comment is descriptive only |
| `M1_TAIL_ALPHA` | 0.3 | `benchmark/config.py:47` → M1 | Sweep over {0, .15, .3, .5, .75, 1.0} scored by Real–AIL and Real–GBM distances and their ratio | **AIL-INFORMED** | `reasoning.md:350-373`: sweep table with Real–AIL column; "AIL is being penalized faster than GBM as α grows"; "**Decision:** `M1_TAIL_ALPHA = 0.3` — the most tail emphasis available before degradation begins" |
| `M1_TAIL_LAMBDA` | 1.0 | `benchmark/config.py:48` → M1 | Sweep over {0, .5, 1, 2} computed Real–AIL/Real–GBM at every value; found flat; final value then picked as "natural unit scale" | **AIL-INFORMED** (via the sweep; final pick itself conventional) | `reasoning.md:380-398`: λ table with Real–AIL column; "**Decision:** `M1_TAIL_LAMBDA = 1.0`. No data-driven reason to prefer 0.5 or 2.0" |
| `M2_LAG_MIN` | 60 | `benchmark/config.py:68` → M2 | A-priori design argument about generator *classes* (not AIL output): short lags are reproducible by naive short-memory generators | INDEPENDENT (documented a-priori claim) | `reasoning.md:416-420`: "short-lag autocorrelation is easily reproduced by naive short-memory generators; the long-lag regime is where real long-memory behavior is actually distinguishing" |
| `M2_LAG_MAX` | 388 | `benchmark/config.py:68` → M2 | Introduced as 390 ("1 trading day"); changed to 388 in the session-support normalization commit — 388 is the longest lag with within-session support (390 bars − masked first bar − 1) | INDEPENDENT (structural) | `git show 7999cd3`: `-M2_LAG_MAX = 390 … +M2_LAG_MAX = 388`; commit msg "preserve unsupported lags as NaN"; no AIL reference in diff |
| `M3_SCALES` | [1, 5, 15, 30] | `benchmark/config.py:70` → M3 | Origin of the planned set {1,5,15,30,390} unrecorded; removal of 390 documented as structural (scale returned NaN for **all** generators — validity, not scores) | UNDOCUMENTED (origin); the 390-drop is documented and not score-based | `reasoning.md:542-575`: "The originally planned scale set was {1, 5, 15, 30, 390}… Empirically, this scale returned NaN for all three generators… **Decision:** drop scale 390." No text on why 1/5/15/30 |
| `M3_MIN_OBS` | 100 | `benchmark/config.py:70` → M3 | No recorded basis; only mechanism description | UNDOCUMENTED | `reasoning.md:607` describes the NaN rule, not the value; introduced in `c7d93ae` with descriptive comment only |
| `M3_SUPPORT_ANCHOR` | "close" | `benchmark/config.py:70` → M3 | Common-support mechanism documented in commit; the specific open-vs-close choice has no recorded rationale | UNDOCUMENTED (the "close" choice) | Commit `77e0748` "Fix M3 to use common intraday support across scales"; docstring (`aggregational_gaussianity.py:51-54`) describes semantics only |
| `ROLLING_VOL_WINDOW` | 60 | `benchmark/config.py:49` → M4 | Rolling60 chosen over EWMA(0.94) in a comparison whose two explicit axes were real–GBM discrimination **and real–AIL tracking**; never re-swept after the session-boundary fix | **AIL-INFORMED** | `reasoning.md:660-663`: "Rolling60 gave tighter tracking of a good generator (real–AIL \|Δξ\| = 0.0048 vs. EWMA's 0.0080)"; `reasoning.md:809-812`: "`window = 60` was not re-swept" |
| `ROLLING_VOL_MIN_FRAC` | 1/2 | `config.py:51` only (derives `ROLLING_VOL_MIN_PERIODS` = 30) | Two min_periods sweeps (original and post-bugfix re-run), both selected on real–AIL gap, discrimination ratio, and **AIL coverage** | **AIL-INFORMED** | `reasoning.md:683-706` (first sweep: "tightest real–AIL gap (0.0022), the best discrimination ratio (68×), and 92% AIL coverage"); `reasoning.md:790-812` (re-sweep: "**Revised decision:** `min_periods = 30`… 30 retains meaningfully more data (69.7% vs. 51.7% AIL coverage)") |
| `ROLLING_VOL_MIN_PERIODS` | 30 (derived) | `benchmark/config.py:50` → M4 | Derived: `ceil(window × min_frac)` — inherits both parents' classifications | AIL-INFORMED (by derivation from two AIL-informed parents) | `config.py:51` |
| `N_REGIME_QUINTILES` | 5 | `benchmark/config.py:51` → M4 | No recorded basis for 5; "quintiles" used as given throughout; post-hoc exceedance-count adequacy check only | UNDOCUMENTED | `git log -S` → introduced in `c7d93ae`, never changed; `reasoning.md:715-722` is adequacy, not selection |
| `TAIL_FRACTION` | 0.05 | `benchmark/config.py:52` → M4 | No recorded basis; introduced as `TAIL_QUANTILE = 0.05` and never changed or swept; only post-hoc adequacy (~259k exceedances/quintile) | UNDOCUMENTED | `git show c7d93ae` (introduction); `reasoning.md:715-722, 843-847` (adequacy only); no selection text anywhere |
| `REGIME_WEIGHTS` | (1,1,1,2,3)/8 | `benchmark/config.py:53` → M4 | Direction (upweight turbulent quintiles) documented as a-priori design intent; the specific magnitudes 2 and 3 have no recorded basis | **AMBIGUOUS** (direction documented a-priori; magnitudes UNDOCUMENTED). No AIL evidence either way | `reasoning.md:838-839`: "weighted MAE… with extra weight on the top two quintiles (stress-testing focus)" |

## Coverage and remaining limits

The original 13-specification grid did not vary the minimum-observation
fraction independently of the volatility window. The current grid includes
fractions 1/3 and 2/3, an additional M1 exponent, and three joint variations.
These additions address the corresponding gaps in numeric parameter coverage.

The rolling-volatility proxy family was selected using both real–GBM
discrimination and real–AIL tracking (research notes, lines 660–664). The grid
varies its window and minimum-observation fraction, but does not compare it
with EWMA. It therefore cannot establish robustness to the proxy-family choice.
The notes also compare raw and standardized M4 inputs (lines 640–653), but
the recorded curves do not establish whether AIL informed that structural choice.

Preprocessing settings, including the seasonality gate and Fourier harmonics,
are fixed across the grid. The audit found no recorded BIC-selection procedure
supporting the historical four-harmonic comment. Sensitivity to the listed
parameters does not establish robustness to all metric and preprocessing designs.
