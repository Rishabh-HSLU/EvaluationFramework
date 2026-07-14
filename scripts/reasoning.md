# Data Loading

## Ticker Universe Curation

**Summary.** We started this project with 948 real-data tickers and an
AIL synthetic universe of 952. After curation, 600 tickers remain — the
subset where both real and AIL have at least 70% non-missing data on
the NYSE 1-minute market clock. This section explains how we got
there, including an earlier curation framing we discarded once a
deeper diagnostic showed it was solving the wrong problem.

### 1. Why curation matters here

The framework compares real intraday equity returns against returns
produced by a synthetic generator. Both datasets have missing minutes:
the real data because some stocks don't trade every minute, the
synthetic data because the generator has internal limits on how many
rows it produces per ticker. How we handle those missing minutes
directly shapes every stylized-fact measurement downstream — tails,
volatility clustering, kurtosis, leverage. Curation decides which
tickers carry enough data to participate in the evaluation at all, and
on what terms.

The choice is not cosmetic. Two tickers with the same overall missing
fraction can have very different gap structures, and those structures
interact with the preprocessing pipeline in ways that can quietly bias
the final scores. Curation has to be designed against a specific
preprocessing strategy — and when the preprocessing strategy changes,
the curation criterion has to change with it.

### 2. How we arrived at the current criterion

#### 2.1 First attempt: gap-length filtering

The original pipeline filled missing intraday prices using a Brownian
Bridge — a smooth Gaussian path drawn between the last observed price
and the next observed price. Brownian Bridge is a sensible
interpolation for short illiquidity gaps, but it would fabricate weeks
of fake price history if asked to bridge an IPO-sized absence. So
curation initially existed to keep the bridge in its valid regime:
filter out tickers whose longest missing run was too long.

On the real-data side, this produced a clean 248-ticker universe:
median missing fraction 1.7%, median longest gap 19 minutes. AIL
covered all 248. We thought the universe was locked.

#### 2.2 The problem the diagnostic revealed

Applying the same lens to AIL's gap structure broke the picture. AIL
had a mean missing fraction of 14.7% — roughly 7× higher than real —
and 74 of the 248 tickers had longest gaps exceeding a full trading
session, some up to 44 sessions long. "Apply the same Brownian Bridge
to both sides" therefore meant applying eight times as much Gaussian
fill to the synthetic side as to the real side.

That asymmetry matters because Brownian Bridge increments are
Gaussian by construction, and the framework's metrics are specifically
designed to detect departures from Gaussianity — fat tails, volatility
clustering, kurtosis decay. Every minute we bridge injects a return
drawn from exactly the distribution the metrics are meant to flag as
unrealistic. Bridging 2% of real data and 15% of synthetic data
silently pulls AIL's measured statistics toward the null hypothesis
the framework is trying to test against. Equal procedure turned out to
mean unequal contamination.

A deeper look explained where AIL's missingness comes from. The
generator hard-caps at 50,000 rows per ticker and spends roughly 10%
of those rows on timestamps outside NYSE regular session hours (mostly
the hour after the closing bell). That accounts almost exactly for the
baseline ~15% missingness observed: the generator doesn't produce
enough usable rows to fill the 53,850-minute market clock. It's a
structural property of the generator, not a fixable artifact.

#### 2.3 Consulting the literature

Two traditions converge on the same conclusion. The high-frequency
volatility literature (Hayashi and Yoshida, 2005, and the
asynchronous-data work that followed) explicitly avoids interpolating
intraday prices, because interpolation introduces *extrinsic bias* —
distortion caused by the preprocessing layer rather than the data
itself. Separately, the missing-data inference literature (Rubin and
standard treatments) shows that *single imputation* — replacing each
missing value with one drawn value and treating it as observed —
understates variance and narrows confidence intervals. Brownian Bridge
fill is exactly this: one path drawn, then treated as data.

Both traditions point the same direction: don't impute. Compute
returns only on observed adjacent prices, and let the metrics handle
absence directly — distribution-shape metrics pool whatever observed
returns exist, lag-based metrics use only adjacent valid pairs, and
any metric that needs a contiguous series is fit on contiguous runs
rather than a stitched-together imputed one.

This also resolved a quiet inconsistency in the old pipeline: it
already treated overnight gaps as structural NaN without imputing
them. Intraday gaps are the same kind of object. One rule — never
compute a return across absence — covers both.

#### 2.4 Checking that drop-NaN is unbiased (MNAR)

Dropping NaNs gives unbiased estimates only if missingness is
unrelated to the value that would have been observed. The natural
worry: maybe AIL fails to emit during turbulent minutes because
they're harder to generate. If so, dropping AIL's NaNs would
systematically remove its turbulent observations, making it look
better at tail and clustering behavior than it actually is.

We tested this by correlating daily missingness rates with a market
volatility proxy. The result was clean: both real and AIL show
*negative* correlation between missingness and volatility (−0.575 and
−0.573), and missingness drops during the March 2020 crash on both
sides. The hard minutes aren't being dropped — they're better
observed. The likely mechanism is mundane: calm periods have less
trading activity, so more no-print minutes; crisis periods see every
minute trade. Real and AIL show nearly identical patterns, so whatever
drives the gaps affects both sides symmetrically. The
missingness-not-at-random concern is resolved empirically.

#### 2.5 The criterion that emerged

Once Brownian Bridge was out, the gap-length filter lost its purpose —
a long internal gap isn't a problem if no return is ever computed
across it. What still mattered was that each ticker have enough valid
observations on *both* sides to support stable estimation. The natural
criterion is a coverage floor applied to the worse of the two sides,
since a ticker is only as useful as its weaker dataset.

### 3. Literature grounding

Sample curation by completeness is standard in high-frequency and
realized-volatility studies: requiring continuous trading history
across the sample window, starting from a fixed universe and retaining
only tickers spanning the full period, defining filters by minimum
participation rate (e.g. ≥95% of trading days), and layering filters
sequentially with each filter's effect reported explicitly. The
no-imputation tradition contributes a complementary principle:
estimators should be computed natively on observed data rather than on
an imputed grid. The criterion adopted here combines both —
coverage-based filtering, applied symmetrically across datasets, with
no imputation downstream.

### 4. The final criterion

A ticker is retained if and only if

> `min(real_coverage, synthetic_coverage) ≥ 0.70`

where coverage is the fraction of NYSE 1-minute market-clock slots
(53,850 in this sample window, September 2019 – March 2020) with a
non-NaN price.

The 70% floor is set by statistical-power considerations. At 70% per
ticker, each ticker contributes at least 37,700 valid returns —
comfortably above the thresholds needed for stable autocorrelation
estimation, kurtosis estimation, and covariance conditioning at the
panel level. Going lower would add tickers but admit names whose
missingness patterns are heavy enough to be untrustworthy in the same
way. Going higher would shed tickers without statistical
justification. 70% is where marginal ticker quality and marginal
estimation precision both remain favorable.

### 5. Result

| Step | Tickers |
|---|---|
| Raw real-data universe | 948 |
| Raw AIL synthetic universe | 952 |
| Intersection (present in both) | 948 |
| After 70% joint-coverage floor | 600 |

The 600 surviving tickers form a well-diversified cross-section across
sectors and market caps. The real and synthetic frames are written to
disk in matching wide-format shape `(53850, 600)`, with NaN preserved
where data is absent. These two files are the input to the
preprocessing layer.

### 6. Caveat

The curation introduces a mild survivorship/liquidity bias: the
evaluated universe tilts toward more continuously-traded names. This
is a deliberate trade-off. The alternative would either require
returning to imputation — rejected on literature-grounded and
empirical bias grounds — or admit much sparser tickers whose
gap-randomness has not been verified to the standard we've verified
for the current universe. The bias runs in the same direction as
standard universe-construction conventions in the cited literature and
should be stated explicitly in the paper's data section.

---

# Preprocessing

## Log Returns with Overnight Masking

Log-differencing prices is lossless for the information every metric
needs. However, `diff()` propagates NaN forward: if `P_t` is missing,
both returns at `t` and `t+1` become NaN. A decomposition analysis
confirmed this is not new information loss — the "extra" NaN at each
gap boundary reflects the honest fact that a 1-minute return requires
both `P_t` and `P_{t-1}` to exist. Real NaN fraction rises from 9.5%
(prices) to 16.5% (returns); AIL from 18% to 30%. The valid returns
that survive are genuinely adjacent 1-minute price pairs with
untouched temporal ordering and distributional shape.

Overnight returns (09:31 ET, the first bar of each session) are
masked because they span a 17.5-hour gap. Leaving them in would
distort every downstream metric: a GARCH-style filter would treat one
as a 1-minute shock, the ACF would see false persistence, and the
marginal distribution would absorb a structurally different return
than the intraday ones it's meant to characterize.

## Conditional Intraday Deseasonalization (FFF)

Real equity returns show a strong diurnal pattern: high variance at
the open, dropping to a trough at midday, and rising again into the
close. The Flexible Fourier Form procedure (Andersen & Bollerslev,
1997) fits a Fourier series to this log-variance profile and divides
it out, leaving approximately unit-variance returns. The risk is
applying that profile blindly.

**The danger.** Baseline generators like GBM or a plain GARCH(1,1) are
typically simulated with no intraday time-of-day effect — their
expected variance profile across the session is flat. Dividing a flat
series by a U-shaped real profile would synthetically suppress its
open/close returns and inflate its midday returns: an artificial,
inverted smile injected before the evaluation metrics ever see the
data. Empirical checks confirmed AIL, by contrast, learns the real
smile closely — open/mid variance ratio of 5.91 (AIL) vs. 5.98 (real),
close/mid of 1.32 vs. 1.52 — so applying FFF to AIL is exactly as
justified as applying it to real.

**The test.** For each series, the pipeline computes the pooled
per-minute variance profile across the session and takes its
coefficient of variation (CV = std / mean). A flat series like GBM has
a profile dominated by noise, giving a near-zero CV (empirically
~0.05). A genuinely seasonal series has a highly dispersed profile,
giving a large CV (empirically ~0.6+). A threshold of 0.3 robustly
separates the two regimes.

**The decision.** If CV > 0.3, the pipeline fits an FFF model native
to that specific series and deseasonalizes it. If not, the series
passes through unchanged. Validated empirically: real CV = 0.944, AIL
CV = 0.974 (both fitted and applied), GBM CV ≈ 0 (correctly skipped).

**Why this matters for the benchmark.** Each series' own seasonality
is removed on its own terms, so the six metrics see only non-seasonal
structure. A sophisticated model that correctly replicates the smile
is judged on its underlying signal after its own smile is removed; a
naive baseline is evaluated fairly on its raw output without a
spurious penalty from mismatched preprocessing. The evaluation
measures the generators' intrinsic properties, not preprocessing
artifacts. If seasonal fidelity itself is ever worth scoring, it
belongs as a separate diagnostic axis — not smuggled into the six
metrics through a mismatched deseasonalization.

---

# Metrics

## M1: Unconditional Heavy Tails (Marginal Distribution)

### What it measures

M1 targets the shape of the marginal return distribution — the
unconditional tail thickness — using a tail-weighted Wasserstein-1
(W1) distance between the empirical quantile functions of real and
synthetic returns. It deliberately ignores temporal ordering: this is
the "does the distribution of return sizes look right, on average"
question, orthogonal to the clustering (M2), leverage (M3), and
regime-conditional (M6) metrics.

### From hard truncation to soft-weight

The first implementation used a hard indicator weight,

> `w(u) = 1` for `u ≤ θ` or `u ≥ 1 − θ`, else `0`,

which is precisely the hard 5% truncation the framework's finalized
design had already moved away from in favor of

> `w(u) = 1 + λ[u⁻ᵅ + (1 − u)⁻ᵅ]`.

The hard indicator has two problems. First, it creates a discontinuity
at `u = θ`: a quantile point just inside the tail counts fully, one
just outside counts zero, and under bootstrap resampling points near
that boundary jitter across it, injecting resampling variance that has
nothing to do with distributional fidelity. Second, it discards the
entire bulk of the distribution (`θ < u < 1 − θ`), even though the
bulk carries real distributional-shape information — the metric's
score ends up depending entirely on an arbitrary cutoff rather than on
the full quantile function.

The soft-weight form fixes both: it is continuous everywhere (no
cliff, no boundary jitter), and it keeps the whole distribution at a
baseline weight of 1 while progressively up-weighting the tails as
`u → 0` or `u → 1`. It is a full-distribution W1 with smooth tail
emphasis, not a knife-edge tails-only integral — strictly more
information, strictly less discontinuity.

### Two implementation bugs fixed alongside the weight change

1. **Grid/weight desynchronization.** The original code rebuilt the
   quantile grid `u` twice — once in `extract_features` to compute
   quantiles, once again in `compute_distance` to compute weights.
   Two independent constructions of the same array is a latent
   coupling: if `n_grid` or the grid formula ever diverged between the
   two call sites, quantiles and weights would silently misalign with
   no error raised. Fixed by building the grid once in `__init__` and
   caching it on the instance, since it depends only on config, never
   on data.
2. **Missing NaN tolerance in `compute_distance`.** The locked
   `BaseMetric` contract requires `compute_distance` to tolerate NaN
   via masking. The original used `integrand.mean()`, which would
   propagate a single NaN quantile (possible under a thin bootstrap
   resample) into a NaN distance for the whole draw. Fixed by
   switching to `np.nanmean`.
`extract_features` itself needed no structural fix: M1 is
order-invariant, so pooling all tickers and dropping NaN before
computing quantiles is the correct, already-locked policy — there is
no lag structure for an overnight-style leak to hide in.

### Empirical validation: hard vs. soft weight

At the initial soft-weight defaults (α = 0.3, λ = 1.0):

| | Real–AIL | Real–GBM | Discrimination ratio |
|---|---:|---:|---:|
| Hard indicator (before) | 0.001764 | 0.236065 | 133.9× |
| Soft-weight (after) | 0.021027 | 2.693146 | 128.1× |

Both distances grew roughly 12×, as expected: the bulk, previously
zeroed out by the indicator, now contributes at weight 1 across the
whole grid instead of only the outer 10%. The discrimination ratio
held (133.9× → 128.1×, a ~4% shift) rather than collapsing — evidence
that the bulk is informative in the *same direction* as the tails.
GBM's Gaussian shape is wrong almost everywhere, not just in the
tails, while AIL's marginal shape tracks real closely almost
everywhere; had the ratio crashed toward 1, that would have meant the
bulk was diluting a signal that only lived in the tails. It doesn't.

### Hyperparameter sweep: α (tail sharpness)

With λ fixed at 1.0, the outermost grid point (`u ≈ 1×10⁻⁴` at
`n_grid = 5001`) makes `α` the higher-risk parameter: `u⁻ᵅ` grows from
~4.6 at α = 0.15 to ~10,000 at α = 1.0, so an aggressive α risks
letting a single noisy extreme-quantile estimate dominate the whole
metric — the same over-emphasis failure mode that had already ruled
out raw kurtosis for M4.

| α | Real–AIL | Real–GBM | ratio | ratio Δ vs. α=0 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0126 | 1.604 | 127.6× | — |
| 0.15 | 0.0153 | 1.966 | 128.9× | +1.0% |
| 0.30 | 0.0210 | 2.693 | 128.1× | −0.6% |
| 0.50 | 0.0442 | 5.298 | 119.8× | −6.5% |
| 0.75 | 0.1966 | 20.215 | 102.8× | −14.2% |
| 1.00 | 1.3054 | 119.450 | 91.5× | −11.0%\* |

\*cumulative degradation continues past α = 0.75; the ratio does not
recover.

The knee sits at α ≈ 0.3–0.5. From α = 0 to 0.3 the ratio is flat,
even marginally improving — tail emphasis adds clean signal in this
range. Past α = 0.5 the ratio erodes, and by α = 0.75–1.0 it
collapses. The mechanism is visible directly in the raw numbers: from
α = 0 to α = 1, the real–AIL distance inflates **103×** (0.0126 →
1.305) while real–GBM inflates only **74×** (1.604 → 119.4) — AIL is
being penalized faster than GBM as α grows, meaning the metric is
increasingly scoring sampling noise in the single most extreme
empirical quantile rather than genuine tail mismatch.

**Decision:** `M1_TAIL_ALPHA = 0.3` — the most tail emphasis available
before degradation begins, not an arbitrary round number.

### Hyperparameter sweep: λ (tail magnitude)

With α fixed at the now-locked 0.3, λ was swept over {0.0, 0.5, 1.0,
2.0}:

| λ | Real–AIL | Real–GBM | ratio |
|---:|---:|---:|---:|
| 0.0 | 0.0042 | 0.535 | 127.6× |
| 0.5 | 0.0126 | 1.614 | 128.0× |
| 1.0 | 0.0210 | 2.693 | 128.1× |
| 2.0 | 0.0379 | 4.852 | 128.1× |

The ratio is flat across the full range (127.6×–128.1×), and both
distances scale almost exactly linearly with λ. This is expected:
bounding α = 0.3 keeps `u⁻ᵅ` bounded (~15.8 at the grid edge, not the
~10⁴ blowup seen at α = 1), so λ acts as a well-behaved linear
multiplier on a bounded term rather than interacting with anything
degenerate. Unlike α, there is no knee — λ carries no discrimination
cost anywhere tested, and its role is purely how large the tail term's
absolute contribution is, not whether the metric stays sound.

**Decision:** `M1_TAIL_LAMBDA = 1.0`. No data-driven reason to prefer
0.5 or 2.0 over it; chosen as the natural unit scale for the power-law
term.

### Final specification

`extract_features` pools all tickers' returns, drops NaN, and
evaluates the empirical quantile function on a cached uniform grid of
`n_grid` points. `compute_distance` applies the cached soft-weight
array (`w(u) = 1 + λ[u⁻ᵅ + (1−u)⁻ᵅ]`, α = 0.3, λ = 1.0, both
empirically validated) and returns the NaN-tolerant mean of the
weighted absolute quantile gap. `normalize` applies the shared
ratio-of-means form.

## M2: Nonlinear Temporal Dependence (Volatility Clustering)

### What it measures

M2 targets long-memory volatility clustering: the tendency of large
absolute returns to cluster in time, measured via the ACF of `|r|`
over long lags (60–390 minutes). Long lags are the focus because
short-lag autocorrelation is easily reproduced by naive short-memory
generators; the long-lag regime is where real long-memory behavior is
actually distinguishing.

### A hidden bug: overnight leakage at long lags

The initial implementation computed the panel ACF using global
`nansum`/`nanmean` reductions with pairwise NaN masking, on the
assumption that masking the single overnight NaN (Section: Log Returns
with Overnight Masking) was sufficient to prevent lag-k pairs from
spanning session boundaries.

It isn't. Masking blocks only the *first* bar of each session — that
removes lag-1 cross-session pairs, since the previous day's close and
the masked overnight bar can't form a pair. But at lag `k ≥ 2`, the
pair (last bar of session *d*, bar `k−1` of session *d+1*) has both
endpoints valid. Nothing masks it. The estimator was silently treating
those as legitimate lag-k intraday autocorrelation, when in fact each
one spans a 17.5-hour gap.

This is the same class of bug — overnight gaps mistaken for 1-minute
intervals — that forced the earlier project reset, now recurring in a
new estimator. At the long lags M2 is specifically designed to probe
(60–390), the leaked pair count grows relative to genuine within-session
pairs: within-session pairs at lag k number `390 − k`, while
cross-session leaked pairs accumulate across all 138 session
boundaries in the sample. Near `k ≈ 389`, there is roughly one genuine
within-session pair per session against hundreds of leaked ones. Since
end-of-day and next-open volatility are both elevated, the leak
inflates the long-lag ACF — the metric was partly measuring overnight
persistence, not intraday clustering.

### The fix

The numerator must be restricted to within-session pairs. Session
boundaries are taken from the calendar clock (`returns.index`
normalized to date), not a fixed 390-row stride, because the sample
window includes variable-length half-day sessions (Thanksgiving,
Christmas Eve). The global mean and denominator are left unchanged, to
preserve the existing biased-estimator shrinkage; only the
lag-k cross-product sum is confined within each session block.

### Empirical validation

| | Real–AIL distance | Real–GBM distance | Discrimination ratio |
|---|---:|---:|---:|
| Before fix | 6.196 | 83.451 | 13.5× |
| After fix | 2.794 | 38.644 | 13.8× |

Both distances roughly halved, and the discrimination ratio was
essentially unchanged. This is the expected signature of a common-mode
leak: overnight contamination inflated the ACF for real, AIL, and GBM
alike, so it partly cancelled out of the real-vs-synthetic gap while
still being present in absolute terms. The ratio looking stable is why
the bug went unnoticed — but stability of the ratio doesn't mean the
metric was measuring the right thing.

Two consequences of the fix beyond the headline numbers:

1. The metric now measures what its docstring claims — intraday
   volatility clustering, not a blend of intraday and overnight
   persistence. A generator with correct intraday clustering but wrong
   overnight behavior (or vice versa) would previously have been
   scored on the wrong basis.
2. The leaked-pair count varies with how many session boundaries land
   inside a given bootstrap resample of tickers, so the leak was
   adding resample-dependent noise to the real-real baseline `g_rr`.
   Removing it should tighten the bootstrap's real-real distribution
   and make the normalized similarity score more stable draw-to-draw.

### Final specification

`extract_features` computes, per ticker, the |return| ACF at lags 1
through `lag_max`, with the lag-k numerator summed only over pairs
falling within the same session, using calendar-derived session
boundaries; the panel feature is the cross-sectional mean of the
per-ticker ACF curves. `compute_distance` sums `|ρ_real − ρ_synth|`
over `[lag_min, lag_max]`, using `nansum` so an all-NaN lag does not
propagate into a NaN total. `normalize` applies the shared ratio-of-means
form, `mean(g_rr) / (mean(g_rr) + mean(g_sr))`.

## M4: Aggregational Gaussianity

### What it measures

Real financial returns are leptokurtic (heavy-tailed) at fine
timescales but converge toward Gaussian as returns are aggregated to
coarser scales (Cont, 2001 §2.5). This metric tracks the *rate* of
that convergence via the normalized excess kurtosis ratio κ(k)/κ(1)
across a set of aggregation scales. GBM, being Gaussian at every
scale by construction, produces a flat ratio curve near 1.0 — no
convergence, because there is no excess kurtosis to converge from. A
generator that reproduces real's decay rate is preserving the
correct balance between fine-scale tail risk and coarse-scale
normality; a generator that converges too quickly is losing
persistence of heavy tails across scales.

### Session-aware aggregation

Aggregating k one-minute log returns into a k-minute return by direct
summation is only valid within a single session — summing across the
overnight boundary would blend a real intraday return with a
structurally different 17.5-hour gap return, the same contamination
pattern guarded against in M2 and M6. Aggregation is therefore done
per session (grouped by calendar date, not a fixed row stride, since
the sample includes variable-length half-day sessions), splitting
each session into non-overlapping blocks of exactly `scale` bars and
summing within each block. NaN propagates honestly: any block
containing a missing bar becomes NaN for that block and is dropped
rather than estimated from a partial sum.

### Kurtosis estimator: standard vs. Moors

The locked metric spec calls for a Moors (1988) robust fallback.
Standard (Fisher) excess kurtosis was adopted as the primary
estimator rather than Moors by default, because the stylized fact
under test is stated in terms of classical kurtosis, and the pooled
sample size at scale 1 (~600 tickers × hundreds of thousands of
valid returns) is far above where small-sample instability would
matter. Moors' quantile-based octile kurtosis remains available via
a `use_moors` flag on the class — a config-level switch, not a code
change — should the bootstrap later reveal instability in κ(1)
across resamples.

### Scale 390 dropped: full-session NaN propagation

The originally planned scale set was `{1, 5, 15, 30, 390}`, with 390
included as the "full trading day" anchor showing convergence all the
way to the daily level. Empirically, this scale returned NaN for all
three generators.

The cause is structural, not a bug: summing 390 one-minute returns
into a single daily return requires all 390 bars in that
session-ticker pair to be valid, since NaN propagates through the
sum. Under real's ~16.5% per-bar NaN rate, the probability that an
entire 390-bar session survives intact is approximately
`(1 − 0.165)^390 ≈ 0` — essentially no complete sessions exist
in the curated data, for any of the three generators (all inherit
real's NaN mask or a comparable one).

Two remedies were considered: (1) drop scale 390 from the set, or
(2) compute daily returns directly from each session's first and
last *observed* price rather than summing all intermediate bars,
sidestepping full-coverage propagation. Option 2 is algebraically
identical to option 1's summation when all bars are present, but
becomes a materially different estimator once bars are missing — it
would mix a sum-of-log-returns estimator (scales 1–30) with a
log-price-difference estimator (scale 390) within a single feature
vector, which is not a fatal flaw but adds a genuine methodological
inconsistency, and 390 minutes of possible slippage between first and
last trade widens what "the return" even means at that scale.

**Decision:** drop scale 390. The scale set is `{1, 5, 15, 30}`. The
decay curve at four points already shows a clean, monotonic,
interpretable convergence trend, and scale 30 remains the coarsest
scale with a consistent estimator across the whole curve. Losing the
daily-level anchor is an acceptable cost against keeping every scale
computed the same way.

### Empirical validation

At the final scale set `{1, 5, 15, 30}`:

| Scale | κ/κ(1) Real | κ/κ(1) AIL | κ/κ(1) GBM |
|------:|------------:|-----------:|-----------:|
|     1 |      1.0000 |     1.0000 |     1.0000 |
|     5 |      0.6447 |     0.5225 |     1.0400 |
|    15 |      0.4974 |     0.4016 |     1.0874 |
|    30 |      0.4104 |     0.3049 |     1.0998 |

- Real–AIL distance: 0.0809, Real–GBM distance: 0.4187
- **Discrimination ratio: 5.2×**
Real shows the expected decay toward Gaussianity as scale increases.
GBM is flat near 1.0 at every scale — slightly above 1.0 due to
sampling noise, consistent with no genuine tail-thinning process to
converge from. AIL tracks real's *direction* of decay but converges
noticeably faster (0.52 vs. 0.64 at scale 5; 0.30 vs. 0.41 at scale
30) — a real, interpretable finding: AIL does not fully preserve the
persistence of heavy tails across aggregation scales, thinning out
too quickly relative to real data. Discrimination is weaker than M1,
M2, and M6 (5.2× vs. 128×, 13.8×, 32.6× respectively), but the
direction is unambiguous and the finding is substantive rather than
noise.

### Final specification

`extract_features` aggregates deseasonalized returns to each scale in
`{1, 5, 15, 30}` via non-overlapping, session-confined summation,
pools across tickers, drops NaN, and computes excess kurtosis (Fisher,
or Moors if `use_moors=True`) at each scale; the feature vector is the
ratio κ(k)/κ(1). Scales with fewer than `min_obs` valid observations,
or a non-finite/zero κ(1), emit NaN throughout. `compute_distance` is
the masked mean absolute difference between two ratio curves.
`normalize` applies the shared ratio-of-means form.

## M6: Regime-Conditional Tail Design

### What it measures

After stripping predictable patterns (seasonality via FFF), do return
tails get heavier in turbulent market regimes? This is a conditional
property that an unconditional tail metric cannot capture: a generator
could produce the right average tail by being too thin during crises
and too heavy during calm periods, and an unconditional metric would
never notice.

### Original design: McNeil-Frey AR-GARCH → GPD (abandoned)

The initial plan followed McNeil & Frey (2000): fit AR-GARCH(1,1) per
ticker to extract standardized residuals, partition time into
volatility regimes, fit GPD on tail exceedances per regime, and
extract the shape parameter ξ.

This failed empirically on our data. Contiguous-segment analysis
showed a median non-NaN run length of 4 bars (real) and 3 bars (AIL).
AR-GARCH needs roughly 250+ contiguous observations for stable
parameter estimates. Requiring 250-bar segments discards 79% of real
data and 95% of AIL data — and the discarded data is biased toward
volatile periods, where gaps cluster, which is exactly what a
regime-conditional metric should be most sensitive to.

### Standardization suppresses the signal

Two approaches were compared directly:

- **Option A:** standardize returns by a volatility proxy (`r_t / σ_t`),
  then fit GPD on standardized residuals per regime.
- **Option B:** fit GPD on raw deseasonalized returns per regime,
  using the volatility proxy only to assign regime labels.
Option A produced a flat ξ curve across regimes (EWMA: −0.169 to
−0.162) — dividing by the same quantity used for regime partitioning
removes the very signal being measured. Option B showed a strong
regime effect (EWMA: −0.134 to +0.188), with tails clearly heavier in
the turbulent quintile.

**Decision:** no standardization. The volatility proxy assigns regime
labels only; it never transforms the returns that feed the GPD fit.

### Volatility proxy: rolling 60-minute std

Two candidates were tested — EWMA (λ = 0.94) and a rolling 60-minute
causal standard deviation:

- Rolling60 gave better discrimination against a bad generator
  (real–GBM `|Δξ|` = 0.153 vs. EWMA's 0.065).
- Rolling60 gave tighter tracking of a good generator
  (real–AIL `|Δξ|` = 0.0048 vs. EWMA's 0.0080).
Rolling60 wins on both axes, has no hyperparameter beyond window
length, and is fully gap-tolerant — it restarts cleanly after NaN
runs.

### Rolling std `min_periods`

The rolling std needs a `min_periods` value: the minimum count of
valid observations within the 60-bar window required to emit a value
rather than NaN. Too low and the variance estimate is noisy; too high
and valid data is discarded.

The MNAR finding above (missingness anti-correlated with volatility)
means the risk from a strict `min_periods` isn't losing the turbulent
regime — turbulent periods are already well covered. The risk is the
opposite: disproportionately excluding *calm*-regime windows and
biasing the lower quintiles.

An empirical sweep tested seven values from 10% to 83% of the window:

| min_p | frac | \|Δξ\| R–AIL | \|Δξ\| R–GBM | ratio | valid % AIL |
|------:|-----:|-------------:|-------------:|------:|------------:|
|     6 |  10% |        0.0033 |        0.1521 | 45.8× |       99.1% |
|    10 |  17% |        0.0035 |        0.1521 | 43.8× |       98.4% |
|    15 |  25% |        0.0034 |        0.1522 | 44.5× |       96.2% |
| **20** | **33%** | **0.0022** | **0.1520** | **68.1×** | **92.3%** |
|    30 |  50% |        0.0048 |        0.1521 | 31.9× |       79.5% |
|    40 |  67% |        0.0120 |        0.1544 | 12.8× |       60.9% |
|    50 |  83% |        0.0219 |        0.1573 |  7.2× |       37.6% |

GBM discrimination stays essentially flat (~0.152) across every
setting — the metric never loses its ability to reject a bad
generator. All the damage from an over-strict `min_periods` shows up
on the good-generator side: AIL's real–AIL gap balloons from 0.002 to
0.022 as coverage drops, because the metric starts penalizing AIL for
data sparsity rather than genuine tail differences.

**Decision (superseded — see below):** `min_periods = 20` (33% of
window). It has the tightest real–AIL gap (0.0022), the best
discrimination ratio (68×), and 92% AIL coverage — no meaningful data
loss. Below 20, noisier std estimates add jitter that inflates the
AIL distance; above 20, calm-period exclusion dominates. Expressed as
a fraction (`min_periods = ⌈window × 1/3⌉`) so it scales automatically
if the window length changes.

This sweep was later found to have been run against a rolling window
that bled across session boundaries — the same class of bug fixed in
M2. Once corrected, the sweep was re-run and the optimum shifted from
`min_periods = 20` to `min_periods = 30`. The analysis above is kept
for the record; the corrected sweep and final decision are documented
in "A second hidden bug" below.

### GPD estimation method

Monte Carlo validation of the WNLS estimator (Park & Kim, 2016) showed
reliable ξ recovery (relative bias < 1%, RMSE < 0.08) at n ≥ 500
exceedances. Regime-cell analysis found roughly 259,000 exceedances
per quintile after pooling across 600 tickers — at this sample size,
MLE and WNLS are essentially exact. MLE is retained for simplicity
(scipy-native, no custom optimization required).

### GBM baseline discrimination (validated)

GBM prices were generated on the same clock with real's NaN mask
imposed, passed through the preprocessing pipeline (FFF correctly
skipped due to flat CV), and evaluated under Option B / Rolling60:

- Real turbulent-regime ξ: +0.181
- AIL turbulent-regime ξ: +0.184
- GBM turbulent-regime ξ: −0.010
The real–GBM gap (0.153) is 32× larger than the real–AIL gap (0.0048).
The metric cleanly separates a generator that captures
regime-conditional tail structure from one that doesn't.

### A second hidden bug: cross-session bleed in the rolling window, and the `min_periods` revision it forced

The implementation review that caught M2's overnight-leakage bug
(session boundaries not respected by a plain positional rolling
operation) surfaced the same class of bug here. `df.rolling(window=60,
...).std()` operates on row position, not calendar time. Only the
single overnight bar is NaN-masked, so for the first ~59 rows of every
session the 60-row window still reached backward into the *previous*
session's returns. Unlike M2, this doesn't fabricate a return across
an unobserved gap — the borrowed values are real, valid returns from
the prior session — but it does mean the volatility proxy near every
session's open was partly informed by the prior session's closing
volatility rather than being purely intraday. This directly
contradicted the "fully gap-tolerant, restarts cleanly after NaN runs"
claim made when Rolling60 was selected over EWMA (Section: Volatility
proxy).

**Fix.** The rolling std is now computed per session, grouping by
calendar date rather than a fixed row stride (the sample includes
variable-length half-day sessions), so the window restarts at every
session boundary instead of reaching into the prior day.

**Immediate effect on the ξ vectors.** Before vs. after the fix, at
the original `min_periods = 20`:

| | Real ξ (Q0–Q4) | AIL ξ (Q0–Q4) | R–AIL dist | R–GBM dist | ratio |
|---|---|---|---:|---:|---:|
| Before (cross-session bleed) | [−0.012, 0.009, 0.035, 0.057, 0.183] | [−0.014, 0.009, 0.031, 0.057, 0.188] | 0.0027 | 0.1664 | 61.8× |
| After (session-aware, mp=20) | [−0.025, −0.008, 0.008, 0.029, 0.185] | [−0.036, −0.020, −0.005, 0.017, 0.192] | 0.0097 | 0.1528 | 15.7× |

GBM was essentially unaffected (no autocorrelation structure for
cross-session bleed to exploit). Real and AIL both shifted toward
lower (more negative) ξ in the calm regimes (Q0–Q2) — consistent with
removing prior-day closing volatility that had been inflating the
apparent turbulence of the "calm" label. But AIL moved roughly twice
as far as real (Q0: real −0.013, AIL −0.022), which is why the
real–AIL distance nearly quadrupled: the old cross-session bleed had
been masking a genuine discrepancy in how AIL handles the
session-open transition, not just adding noise.

**Re-sweeping `min_periods`.** Because the original sweep (Section:
Rolling std `min_periods` above) validated `min_periods = 20` against
the leaky window — where reach-back into the prior session provided
"free" coverage near the open — session-awareness meant that sweep no
longer applied. It was re-run identically (7 values, 10%–83% of
window) under the corrected window:

| min_p | frac | \|Δξ\| R–AIL | \|Δξ\| R–GBM | ratio | valid % R | valid % AIL |
|------:|-----:|-------------:|-------------:|------:|----------:|-------------:|
|     6 |  10% |        0.0110 |        0.1368 | 12.5× |     97.4% |       96.1% |
|    10 |  17% |        0.0120 |        0.1369 | 11.4× |     95.9% |       93.8% |
|    15 |  25% |        0.0121 |        0.1364 | 11.3× |     93.8% |       89.7% |
| 20 (old) | 33% | 0.0107 |     0.1357 | 12.7× |     91.0% |       84.4% |
| **30** | **50%** | **0.0045** | **0.1359** | **30.4×** | **82.9%** | **69.7%** |
|    40 |  67% |        0.0042 |        0.1375 | 33.0× |     71.1% |       51.7% |
|    50 |  83% |        0.0080 |        0.1396 | 17.6× |     54.8% |       31.6% |

The shape changed entirely from the original sweep. Under session-aware
rolling, `min_periods = 20` sits in a shallow trough (12.7×), not a
peak — the old spike at 20 was an artifact of the leaky window's
artificial coverage boost near session opens. The real–AIL gap falls
steadily from mp=6 to mp=40 (0.0110 → 0.0042) as stricter thresholds
filter out noisier near-open volatility estimates, up until mp=50
where AIL coverage collapses to 31.6% and the gap widens again from
data sparsity — the same trade-off shape seen in the original sweep,
just shifted to a higher `min_periods` because a genuinely intraday
window now needs more real observations to reach an equally stable
estimate.

**Revised decision:** `min_periods = 30` (50% of window), not 40.
Discrimination is essentially tied (30.4× vs. 33.0×), but 30 retains
meaningfully more data (69.7% vs. 51.7% AIL coverage) and avoids
resting the published number on a thin-data regime. `window = 60` was
not re-swept — the session-boundary fix changes how far `min_periods`
can safely reach, not the window length itself, so window is treated
as still-validated.

**Final validation at `min_periods = 30`:**

- Real ξ: [−0.025, −0.008, 0.009, 0.030, 0.184]
- AIL ξ: [−0.029, −0.010, 0.004, 0.024, 0.189]
- GBM ξ: [−0.115, −0.122, −0.118, −0.124, −0.011]
- Real–AIL distance: 0.0047, Real–GBM distance: 0.1530
- **Discrimination ratio: 32.6×**
This lands between the 30.4×/33.0× sweep values, confirming the swept
number transfers cleanly to the production class. It sits below the
pre-fix 61.8×, but that figure was inflated by cross-session leakage
masking a real AIL discrepancy at the session open; 32.6× is the
honest number for a metric now measuring what its docstring claims.

### Final specification

1. Compute rolling 60-minute causal std on deseasonalized returns,
   restarted at each session boundary via calendar-date grouping
   (`window = 60`, `min_periods = 30`).
2. Pool `(volatility, return)` pairs across all tickers, drop NaN.
3. Split into quintiles on volatility — each sample uses its own
   quintile boundaries (self-labeled).
4. Within each quintile, fit GPD (MLE, `floc = 0`) on the 5% upper-tail
   exceedances of `|return|`.
5. Extract ξ per quintile → 5-element feature curve.
6. Gap: weighted MAE between real and synthetic ξ curves, with extra
   weight on the top two quintiles (stress-testing focus).
**Why pooled across tickers, not per-ticker.** The rolling-60-minute
std already encodes temporal structure before pooling happens — each
`σ_t` was computed causally from the preceding 60 minutes. Pooling
stacks individual `(σ_t, r_t)` pairs into one list without averaging
or blending returns across tickers. Per-ticker GPD fitting would yield
only ~75 exceedances per regime cell (37,700 returns × 5% tail ÷ 5
quintiles) — well below the ~500 needed for a stable ξ. Pooling across
600 tickers gives ~259,000 exceedances per quintile.

**Why self-labeled, not anchored to real's boundaries.** A generator
with a compressed volatility scale would classify most of its data as
"calm" under real's cut points, leaving its turbulent regime nearly
empty. Self-labeling asks: "when *you* are in *your* most stressed
state, what do *your* tails look like?" — keeping this metric focused
on conditional tail structure without conflating it with absolute
volatility-scale errors, which the volatility clustering metric (M2)
already measures.

**Why absolute returns instead of signed tails.** Taking the absolute
value folds the left and right tails together — a return of −0.05 and
+0.05 both become 0.05 — so the upper tail of `|returns|` represents
extreme moves in either direction, and a single GPD fit captures
overall tail thickness regardless of sign. Fitting GPD separately on
positive and negative signed returns would instead capture tail
*asymmetry* (e.g. whether crashes are heavier-tailed than rallies
under stress) — but that question belongs to the signed-asymmetry
metric (M3). M6 asks a narrower, orthogonal question: do extreme
moves, in either direction, get more extreme during turbulent
regimes? Using `|returns|` answers exactly that while keeping the
feature vector concise (5 elements) and free of overlap with M3.

---

# Benchmark Results and Interpretation

## The first full benchmark run

The first complete run of the matched-N ticker bootstrap (B = 100
resamples, 200 tickers per draw, seed 42; `scripts/run_benchmark.py`)
produced:

| Metric | Stylized fact | AIL | GBM |
|---|---|---|---|
| M1 | Unconditional heavy tails | 0.484 [0.450, 0.517] | 0.028 [0.024, 0.031] |
| M2 | Volatility clustering | 0.220 [0.199, 0.242] | 0.020 [0.018, 0.023] |
| M4 | Aggregational Gaussianity | 0.380 [0.352, 0.410] | 0.125 [0.110, 0.141] |
| M6 | Regime-conditional tails | 0.461 [0.431, 0.490] | 0.045 [0.041, 0.049] |

GBM is rejected by every metric, as a Gaussian, memoryless baseline
should be. AIL sits essentially at real-sample parity on M1 (0.484)
and M6 (0.461) — its marginal distribution and its
regime-conditional tail response are statistically indistinguishable
from an independent draw of real data at this resolution. The two
scores that stand out are M2 (0.220) and, less severely, M4 (0.380).

## Why AIL's M2 score is low: a small bias caught by a tight noise floor

The headline number looks harsh for a generator that plainly *does*
cluster volatility, so the raw components matter:

| M2 | mean distance |
|---|---:|
| g_rr (real vs real, noise floor) | 0.794 |
| g_sr (real vs AIL) | 2.809 |

The real-vs-AIL gap is not the anomaly — 2.809 on the full panel
matches the 2.794 recorded when the session-leak fix was validated
(Section: M2, Empirical validation). What drives the score is the
*noise floor*: real data's cross-sectional |r| ACF curve is extremely
stable across independent 200-ticker subsamples, so mean(g_rr) is
small and the metric has high statistical power.

A full-panel diagnostic (600 tickers, single pass, no bootstrap) shows
what that power is detecting. AIL's ACF curve tracks real's *shape*
almost perfectly, but sits uniformly below it at nearly every lag:

| | Real | AIL |
|---|---:|---:|
| ACF of \|r\| at lag 1 | 0.389 | 0.377 |
| ACF at lag 60 | 0.246 | 0.228 |
| ACF at lag 120 | 0.192 | 0.178 |
| ACF half-life (lag where ACF falls below half its lag-1 value) | 119 min | 107 min |

The per-lag gap is only ~0.01–0.02, but it is a persistent bias, not
scattered noise — AIL's volatility clustering decays roughly 10%
faster than real's. Summed over the 330 lags of M2's scoring window,
that bias accumulates to 3.5× the real-vs-real noise floor, which the
ratio-of-means normalization converts to 0.22.

Two conclusions follow:

1. **The metric is behaving correctly, not harshly.** A tight noise
   floor is what allows a subtle, systematic discrepancy to be
   detected at all; a noisier metric would wave this through as
   sampling variation.
2. **The finding is corroborated independently by M4.** AIL's excess
   kurtosis also decays faster than real's under temporal aggregation
   (κ(5)/κ(1) = 0.52 vs 0.64; Section: M4, Empirical validation).
   Both metrics point at the same underlying property: AIL
   under-persists long-memory structure — it reproduces short-range
   volatility texture well but loses persistence faster than real
   markets do, a known tendency of generative sequence models with no
   explicit long-memory mechanism.

## Why a "resampled real" positive control would be tautological

An obvious sanity check — build a synthetic dataset that satisfies
volatility clustering *by construction* (e.g. block-bootstrap real
sessions) and confirm it scores well — turns out to be uninformative
for the current metric set. M1, M2, M4 and M6 are all invariant to
session order: each pools or averages statistics across sessions
without regard to which day came first. Any dataset built by
resampling, reordering or recombining real sessions is therefore just
another draw of real data as far as these metrics are concerned, and
its expected score is ≈ 0.5 by the same argument that defines the
real-vs-real baseline. It would confirm the top of the scale doesn't
break, but nothing more.

A meaningful positive control has to be an *independent model* that
generates fresh paths with the right long-memory structure — which is
what the FIGARCH baseline below provides.

---

# The MSV Positive-Control Baseline

## Why a positive control, and why it must be an independent model

GBM anchors the bottom of the score range, but nothing anchored the
top with an *independent generator* — the real-vs-real baseline is
internal to the scoring rule itself. Since resampled-real controls
are tautological for the current metric set (previous section), the
positive control had to be a parametric model *designed* to satisfy
one stylized fact — volatility clustering (M2) — and generated the
same way as GBM: same market clock, same ticker universe, per-ticker
scale calibrated to real, real's NaN mask imposed
(`scripts/baseline_generation.py`).

Getting a model to actually match real's measured ACF curve took
three attempts, and the failures were more informative than the
success.

## Attempt 1: FIGARCH — hyperbolic decay is far too steep

FIGARCH (Baillie, Bollerslev & Mikkelsen, 1996) is the canonical
long-memory GARCH: ARCH(∞) weights decaying as k^-(1+d). A
FIGARCH(0, d, 0) simulation (truncated ARCH(∞), per-ticker variance
targeting, 100-ticker sweep panel) produced, against real's
full-panel ACF curve (sum |Δρ| over lags 60–390; real–AIL = 2.79,
real–GBM = 38.6 on the same scale):

| d | ACF lag 1 | lag 60 | lag 120 | curve distance |
|---:|---:|---:|---:|---:|
| 0.30 | 0.319 | 0.052 | 0.031 | 32.9 |
| 0.45 | 0.517 | 0.141 | 0.089 | 21.6 |
| 0.60 | 0.605 | 0.132 | 0.074 | 24.8 |
| 0.75 | 0.643 | 0.068 | 0.031 | 33.1 |

Real's curve is 0.389 / 0.246 / 0.192 at those lags. Even the best d
misses by 8× the AIL distance: FIGARCH's hyperbolic decay collapses
by lag 60 no matter how d is tuned.

The diagnostic that explains why: undoing the ACF estimator's
within-session pair-count taper (at lag k a session contributes
~(390−k) pairs but the denominator is fixed, so measured ≈ true ×
(1 − k/390)) shows real's *true* within-session |r| ACF is nearly
flat — ≈ 0.28–0.29 from lag 60 all the way to lag 250. Real intraday
volatility clustering is not a decaying curve at the session scale;
it is dominated by a *day-level volatility factor*. A volatile day
stays volatile all session (March 2020), a calm day stays calm
(September 2019), so |r_t| and |r_{t+k}| are correlated at *every*
intraday lag roughly equally. No hyperbolically-decaying one-factor
process can reproduce a flat plateau.

## Attempt 2: single-factor LMSV — killed by per-path centering

Long-Memory Stochastic Volatility (Breidt, Crato & de Lima, 1998)
with log-vol driven by fractional Gaussian noise looked like the
right class: fGn's ACF decays as k^(2H−2), nearly flat for H → 1.
Exact fGn was sampled via Davies–Harte circulant embedding. The sweep
showed the level and flatness *worsening* as H approached 1
(H = 0.98: lag-60/lag-1 ratio 0.37 vs real's 0.63; H = 0.999: ACF
collapsed to ≈ 0.02 everywhere).

The mechanism is an estimator interaction, not a bug: M2 centers each
ticker *within its own path* (per-ticker mean and variance over the
sample window). As H → 1, fGn becomes almost perfectly correlated
across the window — each path's log-vol is nearly one constant random
level. Per-path centering removes exactly that level, leaving almost
no within-path vol variation for the ACF to detect. The ensemble
long memory is real, but invisible to a within-path estimator at
finite T. What the estimator *can* see is vol variation across days
within the window — which is again the day-level factor.

## Attempt 3 (adopted): two-factor multi-scale SV

Both failures point to the same structure, which is also the standard
multi-scale stochastic volatility setup (fast + slow factors; Fouque,
Papanicolaou & Sircar):

    r_t = c_i · exp(ν_slow · s_d(t) + ν_fast · f_t) · z_t

- **s_d** — slow factor: exact fGn (H = 0.9) across the 138 sessions,
  held constant within each session. Produces the flat within-session
  ACF plateau via day-level vol persistence.
- **f_t** — fast factor: stationary AR(1) in minutes with
  autocorrelation time τ, producing the short-lag decay from lag 1
  down to the plateau.
- **c_i** — per-ticker scale matched exactly to real's return std.

Sweep over (ν_slow, ν_fast, τ) on the 100-ticker panel:

| ν_slow | ν_fast | τ | lag 1 | lag 60 | lag 120 | lag 250 | curve distance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.3 | 60 | 0.327 | 0.208 | 0.155 | 0.081 | 6.85 |
| **0.9** | **0.3** | **20** | **0.391** | **0.244** | **0.195** | **0.106** | **1.85** |
| 0.9 | 0.3 | 60 | 0.392 | 0.263 | 0.201 | 0.107 | 2.74 |
| 0.9 | 0.5 | 60 | 0.431 | 0.238 | 0.166 | 0.084 | 4.85 |

(real: 0.389 / 0.246 / 0.192 / 0.098)

The selected parameters (ν_slow = 0.9, ν_fast = 0.3, τ = 20, H = 0.9)
match real's measured curve nearly point-for-point, with a curve
distance of 1.85 — *closer than AIL's 2.79*. MSV is intentionally an
M2 specialist: it has no heavy-tail mechanism beyond the lognormal
vol mixture and no regime-tail design, so it should beat AIL on M2
while losing to it elsewhere — the signature of a valid
single-purpose positive control.

## Engineering notes from the same pass

- **M2 vectorization.** The per-lag Python loop in
  `VolatilityClustering.extract_features` was replaced by per-session
  FFT autocorrelation (NaN→0 is exact under the pairwise-masking
  semantics; zero-padding to ≥ 2n makes the circular correlation
  linear). Verified bit-identical to the loop implementation
  (max |Δ| = 1.1e-16) and ~18× faster; the full benchmark drops from
  ~1.6 h to ~35 min.
- **M6 rolling-vol.** The per-column `groupby().transform(lambda ...)`
  was replaced with one vectorized rolling per session block —
  verified identical output (values and NaN mask).
- **Per-cell RNG streams.** The bootstrap engine originally drew all
  ticker subsamples from one shared RNG stream, so *adding* a
  generator changed the draw sequence and perturbed every other
  generator's scores. Streams are now independent: `[seed, 0]` for
  real draws, `[seed, 1, crc32(name)]` per generator,
  `[seed, 2, crc32(metric), crc32(name)]` for each cell's CI
  bootstrap. Every (metric, generator) cell is now reproducible in
  isolation and invariant to the generator set. This restructure
  changes the draw sequence once relative to the first benchmark run,
  so scores move within their CIs against that run; they are stable
  thereafter.

## The second full benchmark run

Same protocol (B = 100, 200 tickers per draw, seed 42), now on the
per-cell RNG streams and with MSV added:

| Metric | Stylized fact | AIL | GBM | MSV |
|---|---|---|---|---|
| M1 | Unconditional heavy tails | 0.478 [0.447, 0.509] | 0.025 [0.023, 0.028] | 0.025 [0.023, 0.028] |
| M2 | Volatility clustering | 0.203 [0.181, 0.226] | 0.019 [0.016, 0.021] | 0.265 [0.237, 0.291] |
| M4 | Aggregational Gaussianity | 0.378 [0.346, 0.410] | 0.121 [0.108, 0.136] | 0.170 [0.152, 0.189] |
| M6 | Regime-conditional tails | 0.485 [0.456, 0.514] | 0.047 [0.043, 0.051] | 0.133 [0.122, 0.145] |

Every AIL and GBM score sits inside its first-run confidence interval
(compare to the table in "The first full benchmark run" above) — the
RNG restructure changed which tickers get drawn, not the measurement.
MSV lands exactly where it was designed to: above AIL on M2 (0.265 vs.
0.203, the metric it was built to specialize in) and below AIL on
every other metric, since it has no heavy-tail or regime-conditional
mechanism beyond its lognormal volatility mixture. This is the
positive control behaving as a positive control — it validates that
the benchmark can discriminate a generator that is *better* than AIL
at the one property it targets, not only worse generators like GBM.

# M2 Lag Weighting: The Window Is the Weight (2026-07-05)

While reconciling docstrings with code ahead of the mathematical
foundations document, M2's docstrings described its distance as
"lag-weighted," but the implementation applies no per-lag weights —
it sums |Δρ(k)| uniformly over k ∈ [lag_min, lag_max]. Decision: the
window restriction [lag_min, lag_max] is the *only* lag weighting, a
deliberate 0/1 weight, and this is the authoritative definition.

Two reasons. First, results are in hand: both full benchmark runs
above were produced under this definition, and changing the distance
now would invalidate them for no measurement benefit. Second, the
window already encodes what a smooth weight would aim for — it
excludes the short-memory regime entirely (lags below lag_min, which
naive short-memory generators reproduce anyway) and treats every
long-memory lag as equally informative, the honest choice absent
evidence that any sub-band of [60, 390] matters more.

The docstrings were corrected to read "unweighted L1 sum of absolute
ACF gaps over the lag window"; the distance definition is unchanged.

# Seed-Sensitivity Audit: M2's Confidence Interval Runs Narrow (2026-07-08)

### Why: does the reported CI actually bound what it claims to?

Every benchmark run so far used `seed = 42`. The matched-N ticker
bootstrap's paired CI (Section: Confidence intervals) is a bootstrap
over the *B = 100 draws realized under one seed's RNG stream* — it was
never structurally guaranteed to bound variability across *different*
seeds, i.e. different realizations of which 200 tickers get drawn into
each of those 100 resamples. To check whether it does anyway, the full
benchmark (B = 100, 200 tickers/draw, all four metrics × AIL/GBM/MSV)
was rerun at seven additional seeds — `1, 2, ..., 7` — holding
everything else fixed
(`fineval/scripts/notebooks/seed_sensitivity.ipynb`, self-contained,
no external CSVs).

### The good news first: every qualitative conclusion held

Across all 8 seeds, the generator rank ordering by the aggregate score
G was identical every time — `AIL < MSV < GBM` (best to worst) — and
MSV beat AIL on M2 specifically in all 8 seeds, exactly as the
positive-control design intends. Per-metric point estimates moved by
the modest amount the sampling-noise story predicts (score ranges of
0.004–0.05 across the 8 seeds, depending on the cell) — nothing that
would change which generator wins, only enough that quoting a score to
three decimals overstates its precision.

### The finding: M2's CI does not bound its own across-seed spread

Checking, for each metric × generator, what fraction of the other 7
seeds' point estimates fell inside the `seed = 42` run's own CI:

| Metric | Anchor-42 coverage |
|---|---:|
| M1 | 0.81 |
| M2 | 0.33 |
| M4 | 0.81 |
| M6 | 0.71 |

M2 stood out badly. Re-checked with a fairer diagnostic — every seed
used as its own anchor in turn, coverage averaged across all 8 choices
(leave-one-out) rather than trusting one arbitrary reference point:

| Metric | LOO-averaged coverage |
|---|---:|
| M1 | 0.88 |
| M2 | 0.61 |
| M4 | 0.80 |
| M6 | 0.83 |

Roughly half of M2's apparent problem was that `seed = 42` happened to
be an atypical (low) draw for M2/AIL specifically — M6/AIL's coverage
recovered almost completely under LOO (0.43 → 0.82), confirming that
kind of anchor artifact is real and worth correcting for. But M2
remains the worst-covered metric even under the fair diagnostic, so an
anchor artifact is not the whole story.

### Confirming diagnostic: point-estimate range vs. reported CI width

`point_range / anchor_ci_width` per cell — a ratio ≥ 1 means the true
across-seed spread does not physically fit inside the reported
interval:

| | M1 | M2 | M4 | M6 |
|---|---:|---:|---:|---:|
| AIL | 0.41 | **1.13** | 0.77 | 0.81 |
| GBM | 0.81 | **1.04** | 0.78 | 0.65 |
| MSV | 0.83 | 0.94 | 0.73 | 0.75 |

M2 is the only metric where every one of its three generator cells
sits at or above 0.94; every other metric/generator cell sits at
0.41–0.83. This is metric-wide, not confined to one generator or one
choice of anchor.

### Candidate mechanisms (unresolved — see TODO)

1. **Cross-ticker heterogeneity in volatility-clustering persistence.**
   The v1 variance-decomposition analysis (`paper/snippets/
   methodology_snippet.tex`) found, for bucket B1 (marginal
   distribution), that the noise floor's pooled CV barely shrank when
   draws were restricted to a single ticker or regime (0.65 → 0.64 →
   0.62) — a real share of the dispersion there is structural
   cross-sectional heterogeneity, not estimation noise that more
   resampling washes out. That result was for B1, not M2, but the same
   mechanism plausibly applies more to M2 than to M1/M4/M6: volatility-
   clustering persistence plausibly varies more across megacap/
   small-cap microstructure than unconditional tail shape or kurtosis
   decay does. If so, between-seed variance is partly driven by *which*
   200 of the 600 tickers get drawn — which a bootstrap confined to one
   seed's fixed draws structurally cannot see. Not yet measured
   directly for M2 on v2.1.
2. **331 summed lags are not independent.** `compute_distance` sums
   `|Δρ(k)|` over k = 60..390; adjacent lags of an ACF curve are
   strongly autocorrelated, so the effective independent-component
   count behind that sum is far below 331. A sum of few, correlated,
   non-negative terms converges to Gaussian more slowly, and
   percentile-bootstrap CIs are known to under-cover skewed statistics
   at modest B.
3. **Lower priority:** the full-sample (not per-session) ACF
   denominator reused across the session-confined numerator (Section:
   M2, The fix) could introduce estimator behavior across draws not yet
   characterized.

Not resolved: raw per-draw `g_rr`/`g_sr` arrays are not persisted
anywhere (only summary statistics survive a run), so (1) and (2) cannot
be distinguished without new instrumentation. Confirmed and ruled out
as a fix: increasing B alone will not correct this — CI width and the
true across-seed spread both scale as ~1/√B, so doubling B narrows both
together and likely preserves the same relative under-coverage, just at
a smaller absolute scale (follows from standard bootstrap scaling, not
independently tested).
