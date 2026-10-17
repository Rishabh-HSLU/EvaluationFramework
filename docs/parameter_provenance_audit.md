# Parameter Provenance Audit — FinEval

Date: 2026-07-31. Read-only audit of every parameter in `fineval/config.py`:
enumeration, provenance classification, consistency check against
`scripts/reasoning.md`, and coverage cross-check against the sensitivity grid
(`fineval/scripts/sensitivity/grid.py`).

Classes: **AIL-INFORMED** (chosen via a sweep/comparison that looked at AIL's
scores, distances, or discrimination ratios) / **INDEPENDENT** (statistical
theory, literature, or convention with no reference to generator output) /
**UNDOCUMENTED** (no recorded basis found) / **AMBIGUOUS** (evidence does not
cleanly fit one class).

## Steps 1–2 — Parameter enumeration and provenance

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

Note on scope: reasoning.md also documents two AIL-informed **structural**
(non-config) M4 decisions — no-standardization Option B
(`reasoning.md:640-653`; the A/B ξ-curves are unlabeled as to dataset, so AIL
involvement there is unproven) and the rolling-vs-EWMA proxy *family*
(AIL-informed, quoted above). These are not `config.py` parameters but matter
for the coverage check in Step 4.

## Step 3 — Consistency check (reasoning.md / config.py / code disagreements)

1. **Confirmed known example:** `reasoning.md:700-706` records
   `min_periods = 20 = ⌈window × 1/3⌉` as a "Decision", marked
   "(superseded — see below)" and left in place; `reasoning.md:806` revises to
   30 (= 1/2). Config: `ROLLING_VOL_MIN_FRAC = 1/2`. Additionally, commit
   `c7d93ae` shipped `FRAC = 1/2` alongside a stale `# = 20` comment
   (arithmetically wrong at the time: ceil(60·½)=30); fixed only in `006e894`.
2. **M2 lag window:** `reasoning.md` says 60–390 in at least four places
   (`:416, :440, :988, :1129` "[60, 390]"); config has been
   `M2_LAG_MAX = 388` since `7999cd3`. Never reconciled.
3. **Metric numbering drift:** reasoning.md uses the old scheme — regime tails
   is "M6" (`:612`), aggregational Gaussianity was "M4" (`c7d93ae` config),
   and "M3" refers to a signed-asymmetry/leverage metric (`:269, :865-866`)
   that no longer exists in the benchmark. Current code: M3 = aggregational
   Gaussianity, M4 = regime tails. `mathematical_foundations.md:197` also
   still says "M6" for regime tails.
4. **Stale identifier:** `regime_tails.py:51` docstring imports
   `TAIL_QUANTILE` — the pre-rename name; config has `TAIL_FRACTION`. Same
   docstring (`:12`) cites "preprocessing/reasoning.md", a path that doesn't
   exist (file is `scripts/reasoning.md`).
5. **NUM_HARMONICS:** config comment asserts a BIC selection that is recorded
   nowhere in the repo — a claim without an audit trail.
6. Minor: reasoning.md's M1 section title "Unconditional Heavy Tails" predates
   the rename to "Tail-weighted marginal distribution" (commit `e60fe80`);
   `mathematical_foundations.md:39-57` itself logs known doc/code drift items.

## Step 4 — Coverage cross-check against `fineval/scripts/sensitivity/grid.py`

**What the 13 specs actually vary:** `tail_alpha` ∈ {0, 0.15, 0.3, 0.5, 0.75};
`tail_lambda` ∈ {0, 0.5, 1.0, 2.0}; `window` ∈ {30, 60, 120} with
`min_periods` **locked to the derived rule** ceil(window·½) = {15, 30, 60};
`tail_fraction` ∈ {0.025, 0.05, 0.10}; `regime_weights` ∈ {(1,1,1,2,3), flat}.
Held constant in every spec: `n_grid`, `n_regimes`, the min-periods
**coefficient** (1/2), and everything in M2, M3, preprocessing, and curation.

**Key deliverable — AIL-INFORMED parameters NOT varied by the grid:**

1. **`ROLLING_VOL_MIN_FRAC` (= 1/2) — the one hard gap.** Treating the
   coefficient as the parameter: the grid *appears* to vary `min_periods`
   (15/30/60), but only as a passenger of `window`; the coefficient chosen by
   the two AIL-scored sweeps is pinned at 1/2 in all 13 specs. This is
   precisely the parameter whose selection evidence is most explicitly
   AIL-dependent (chosen partly on *AIL coverage %*), and the grid cannot
   exonerate it. **Specs needed:** two OAT specs at window=60 varying the
   coefficient to its historical sweep alternatives — `min_frac = 1/3` (mp=20,
   the superseded optimum) and `min_frac = 2/3` (mp=40, the runner-up the
   re-sweep rejected on AIL-coverage grounds). Both are M4-only dials, so they
   are cheap under fast mode.
2. **AIL-informed structural choices — outside the grid's vocabulary
   entirely.** The volatility-proxy *family* (rolling std vs. EWMA(λ=0.94))
   was selected with real–AIL tracking as an explicit criterion, and no spec
   varies the proxy family (the grid only varies the rolling window's length).
   Covering it would need an `EWMA` variant spec, which the current
   `build_metrics()` surface cannot express — it would require a metric-code
   change, so it is a scope decision, not a grid edit. (The related Option A/B
   standardization choice is *plausibly* but not provably AIL-informed — the
   recorded ξ-ranges aren't labeled by dataset.)
3. For completeness, the two AMBIGUOUS parameters are also uncovered:
   `SEASONALITY_CV_THRESHOLD` (would require per-spec re-preprocessing — the
   harness currently shares one preprocessing pass across specs by design, so
   this is a structural extension) and `NUM_HARMONICS` (same). Neither is
   established as AIL-informed; flagged so the pre-registration can state the
   boundary explicitly.

All AIL-INFORMED config parameters that are grid-expressible — `tail_alpha`,
`tail_lambda`, `window` — are covered except `ROLLING_VOL_MIN_FRAC`. Adding
the two min_frac specs to `grid.py` (13 → 15 specs) before any freeze closes
the only hard gap this audit found.

## Step 5 — Disclosure: M4 volatility-proxy selection (DRAFT, pending review)

M4's volatility proxy is a rolling 60-minute causal standard deviation,
chosen over EWMA(λ = 0.94) on two recorded axes (`reasoning.md:660-664`).
One of those axes is real–AIL tracking: *"Rolling60 gave tighter tracking
of a good generator (real–AIL |Δξ| = 0.0048 vs. EWMA's 0.0080)."* AIL is
the generator the benchmark exists to evaluate, so its output informed the
choice of the proxy **family**, not merely the value of a numeric constant.
Three consequences follow, and the pre-registration should state all three.

1. **The family is AIL-informed.** The second recorded axis is real–GBM
   discrimination (0.153 vs. 0.065), which is also generator-informed,
   though GBM is the negative control. The decision was not made blind to
   the evaluated generator on either axis.
2. **No spec covers it.** The grid varies the rolling window's *length*
   (30/60/120) and, since the min_frac specs, its `min_periods`
   coefficient — but every spec still uses the rolling-std family. An EWMA
   variant cannot be expressed through `build_metrics()`, so covering it is
   a metric-code change rather than a grid edit: a scope decision, not an
   oversight.
3. **The recorded margin is small and untested.** |Δξ| = 0.0048 against
   0.0080 is a difference of 0.0032 in a tail-index gap, reported without
   an interval and from a single configuration. Whether the two proxies are
   separable at all is not established by the recorded evidence.

A related wording correction: `reasoning.md:664` states Rolling60 "has no
hyperparameter beyond window length". It has two — `min_periods` is a free
parameter, and the coefficient behind it was itself selected on real–AIL
gap and AIL coverage (`reasoning.md:683-706`, `:790-812`).

**Proposed statement.** M4's proxy family was selected with the evaluated
generator's output in view; the sensitivity grid does not, and in its
current form cannot, bracket that choice; and the recorded margin between
the two candidates is small and carries no uncertainty estimate. Reporting
the generator ranking as robust to M4's *parameters* is supported by the
sweep. Reporting it as robust to M4's *design* is not.
