# Data Loading
## Ticker Universe Curation

**Summary.** We started this project with 948 real-data tickers and an
AIL synthetic universe of 952. After curation, 600 tickers remain — the
subset where both real and AIL have at least 70% non-missing data on
the NYSE 1-minute market clock. This section explains how we got there.
The path involved discarding an earlier curation framing once a deeper
diagnostic revealed it was solving the wrong problem.

---

### 1. Why curation matters here

The framework compares real intraday equity returns against returns
produced by a synthetic generator. Both datasets have missing minutes:
the real data because some stocks don't trade every minute, the
synthetic data because the generator has internal limits on how many
rows it produces per ticker. How we handle those missing minutes
directly shapes every stylized-fact measurement downstream — tails,
volatility clustering, kurtosis, leverage. Curation is the step that
decides which tickers carry enough data to participate in the
evaluation at all, and on what terms.

The choice is not cosmetic. Two tickers with the same overall missing
fraction can have very different gap structures, and those structures
interact with the preprocessing pipeline in ways that can quietly bias
the final scores. So curation has to be designed against a specific
preprocessing strategy — and when the preprocessing strategy changes,
the curation criterion has to change with it.

---

### 2. How we arrived at the current criterion

#### 2.1 The first attempt: gap-length filtering

The original pipeline filled in missing intraday prices using a
Brownian Bridge — a smooth Gaussian path drawn between the last
observed price and the next observed price. Brownian Bridge is a
sensible interpolation for short illiquidity gaps (a few missing
minutes), but it would fabricate weeks of fake price history if asked
to bridge an IPO-sized absence. So curation initially existed to keep
the bridge in its valid regime: filter out tickers whose longest
missing run was too long.

On the real-data side, this produced a clean 248-ticker universe:
median missing fraction 1.7%, median longest gap 19 minutes. AIL
covered all 248. We thought the universe was locked.

#### 2.2 The problem the diagnostic revealed

When we looked at AIL's gap structure with the same lens, the picture
fell apart. AIL had a mean missing fraction of 14.7% — roughly 7×
higher than real — and 74 of the 248 tickers had longest gaps
exceeding a full trading session, some up to 44 sessions long. So
"apply the same Brownian Bridge to both sides" meant applying *eight
times as much Gaussian fill to the synthetic side as to the real side*.

Why does that matter? Because Brownian Bridge increments are Gaussian
by construction. The framework's metrics are specifically designed to
detect departures from Gaussianity — fat tails, volatility clustering,
kurtosis decay. Every minute we bridge, we inject a return drawn from
exactly the distribution the metrics are designed to flag as
"unrealistic." Bridging 2% of real data and 15% of synthetic data
silently pulls AIL's measured statistics toward the very null
hypothesis the framework is trying to test against. Equal procedure
turned out to mean unequal contamination.

A deeper look explained where AIL's missingness comes from. The
generator hard-caps at 50,000 rows per ticker and spends roughly 10%
of those rows on timestamps outside NYSE regular session hours (mostly
the hour after the closing bell). That accounts almost exactly for the
baseline ~15% missingness we observed: the generator simply doesn't
produce enough usable rows to fill the 53,850-minute market clock.
It's a structural property of the generator, not a fixable artifact.

#### 2.3 Consulting the literature

Two traditions in the literature converge on the same conclusion. The
high-frequency volatility literature (notably Hayashi and Yoshida,
2005, and the asynchronous-data work that followed) explicitly avoids
interpolation of intraday prices, because interpolation introduces
*extrinsic bias* — distortion caused by the preprocessing layer rather
than by the data itself. Separately, the missing-data inference
literature (Rubin and standard treatments) shows that *single
imputation* — replacing each missing value with one drawn value and
treating it as observed — understates variance and narrows confidence
intervals. Brownian Bridge fill is exactly this: one path drawn, then
treated as data.

Both traditions point the same direction: don't impute. Compute returns
only on observed adjacent prices. Where gaps exist, the metrics handle
absence directly — distribution-shape metrics pool whatever observed
returns exist, lag-based metrics use only adjacent valid pairs, and
the bucket that needs a contiguous series (B6, the AR-GARCH residual
filter) is fit on contiguous runs rather than on a stitched-together
imputed series.

This also resolved a quiet inconsistency in the old pipeline: it
already treated overnight gaps as structural NaN without imputing
them. Intraday gaps are the same kind of object. Treating both under
one rule — never compute a return across absence — is cleaner.

#### 2.4 Checking that drop-NaN is unbiased here (MNAR)

Dropping NaNs gives unbiased estimates only if the missingness is
unrelated to the value that would have been observed. The natural
worry: maybe AIL fails to emit during turbulent minutes because they're
harder to generate. If so, dropping AIL's NaNs would systematically
remove its turbulent observations, making it look better at tail and
clustering behavior than it actually is.

We tested this by correlating daily missingness rates with a market
volatility proxy. The result was clean: both real and AIL show
*negative* correlation between missingness and volatility (−0.575 and
−0.573), and missingness drops during the March 2020 crash for both
sides. The hard minutes aren't being dropped — they're better-observed,
not worse. The likely mechanism is mundane: calm periods have less
trading activity, so more no-print minutes; crisis periods see every
minute trade. Crucially, real and AIL show nearly identical patterns,
so whatever drives the gaps affects both sides symmetrically. The
missingness-not-at-random concern is resolved empirically.

#### 2.5 The criterion that emerged

Once Brownian Bridge was out, the gap-length filter lost its purpose:
a ticker with a long internal gap isn't a problem if no return is
ever computed across it. What still mattered was that each ticker has
enough valid observations on *both sides* — real and synthetic — to
support stable estimation. The natural criterion is a coverage floor
applied to the worse of the two sides, since a ticker is only as
useful as its weaker dataset.

---

### 3. Literature grounding

Sample curation by completeness is standard in high-frequency and
realized-volatility studies. Common practices: requiring continuous
trading history across the sample window; starting from a fixed
universe (e.g. index constituents) and retaining only tickers spanning
the full period; defining filters by minimum participation rate
(e.g. ≥95% of trading days); layering filters sequentially with each
filter's effect reported explicitly. The no-imputation tradition
contributes a complementary principle: estimators should be computed
natively on observed data rather than on imputed grids. The criterion
adopted here combines both — coverage-based filtering, applied
symmetrically across datasets, with no imputation downstream.

---

### 4. The final criterion

A ticker is retained if and only if:

> `min(real_coverage, synthetic_coverage) ≥ 0.70`

where coverage is the fraction of NYSE 1-minute market-clock slots
(53,850 in this sample window, September 2019 – March 2020) on which
a non-NaN price is observed.

The 70% floor is set by statistical-power considerations. At 70% per
ticker, each ticker contributes at least 37,700 valid returns —
comfortably above the thresholds needed for stable autocorrelation
estimation, kurtosis estimation, and covariance conditioning at the
panel level. Going lower would add tickers but admit names where
missingness patterns are heavy enough that they may not be trustworthy
in the same way. Going higher would shed tickers without statistical
justification. 70% is the level at which marginal ticker quality and
marginal estimation precision both remain favorable.

---

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

---

### 6. Caveat



The curation introduces a mild survivorship/liquidity bias: the
evaluated universe tilts toward more continuously-traded names. This
is a deliberate trade-off. The alternative would either require
returning to imputation — which we rejected on literature-grounded and
empirical bias grounds — or admit much sparser tickers whose
gap-randomness has not been verified at the level we've verified the
current universe. The bias is in the same direction as standard
universe-construction conventions in the cited literature and should
be stated explicitly in the paper's data section.

------------------------
# Preprocessing

## Conditional Intraday Deseasonalization (FFF)

**Summary.** The Flexible Fourier Form (FFF) procedure removes the deterministic U-shaped intraday volatility smile from returns. While real equities and advanced synthetic generators (like AIL) exhibit this genuine smile, standard baselines (like GBM or standard GARCH) are structurally flat. Blindly applying a deseasonalization profile to a flat series injects an artificial inverted smile. Therefore, the pipeline must conditionally detect if a genuine smile exists per series, and only fit/apply FFF if it does.

---

### 1. The Danger of Blind Deseasonalization

Real equity returns show a strong diurnal pattern: high variance at the open (09:31 ET), dropping to a trough at midday, and rising again into the close. The FFF procedure (Andersen & Bollerslev, 1997) fits a Fourier series to this log-variance profile and divides out the deterministic seasonality, leaving approximately unit-variance returns.

Our empirical checks confirmed that AIL successfully learns this intraday behavior, presenting a variance profile nearly identical to real equities (open/mid variance ratio of 5.91 for AIL vs. 5.98 for real). For both real data and AIL, FFF is statistically justified and necessary to expose the true return signal.

However, baseline generators like Geometric Brownian Motion (GBM) or GARCH(1,1) are typically simulated without intraday time-of-day effects—their expected variance profile across the session is flat. If we were to blindly divide a flat GBM series by a U-shaped volatility profile, we would synthetically suppress its open/close returns and inflate its midday returns. We would be actively injecting an artificial, inverted smile into the baseline, corrupting its intraday structure before the evaluation metrics even see it.

### 2. The Solution: Conditional FFF Application

To prevent this extrinsic contamination, the preprocessing pipeline applies FFF conditionally per-series. For each dataset (real and synthetic), the pipeline must evaluate whether a genuine smile exists before acting.

1. **Variance Profile Extraction:** We compute the pooled per-minute variance profile across the trading session (i.e., average squared return for minute $\tau$).
2. **Dispersion Measurement:** We use the coefficient of variation (CV = standard deviation / mean) of this minute-by-minute profile as our test statistic.
3. **Thresholding:** A flat series like GBM will have a variance profile dominated by noise, yielding a near-zero CV (empirically ~0.05). A seasonal series with a massive open spike will have a highly dispersed profile, yielding a large CV (empirically ~0.6+). A threshold of 0.3 robustly separates the two regimes.

If the CV > 0.3, the pipeline fits an FFF model native to that specific dataset and deseasonalizes it. If false, the pipeline recognizes the series as flat and passes the returns through unchanged.

### 3. Implications for the Benchmark

This conditional logic ensures the evaluation remains intellectually honest across the entire spectrum of generator complexity. A sophisticated model that correctly replicates the smile is judged on its underlying signal (after its own smile is removed), while a naive baseline is evaluated fairly on its raw output without having a spurious penalty injected by the preprocessing layer. This guarantees that the evaluation framework measures the generators' intrinsic properties rather than preprocessing artifacts.

---------------------------

## Log Returns with Overnight Masking

Log-differencing prices is lossless for the information every metric
needs. However, `diff()` propagates NaN forward: if P_t is missing,
both returns at t and t+1 become NaN. A decomposition analysis
confirmed this is not new information loss — the "extra" NaN at each
gap boundary reflects the honest fact that a 1-minute return requires
both P_t and P_{t-1} to exist. Real NaN fraction rises from 9.5%
(prices) to 16.5% (returns); AIL from 18% to 30%. The valid returns
that survive are genuinely adjacent 1-minute price pairs with
untouched temporal ordering and distributional shape.

Overnight returns (09:31 ET, the first bar of each session) are
masked because they span a 17.5-hour gap. Leaving them in would
distort every downstream metric: GARCH treats them as 1-minute
shocks, ACF sees false persistence, and the marginal distribution
absorbs a structurally different return.


## FFF Deseasonalisation — Conditional Per Series

The original design assumed FFF should be fit on real data only and
applied identically to both real and synthetic series. Empirical
investigation revealed this is wrong for two reasons:

1. **Applying real's smile to a flat generator injects an inverted
   smile.** GBM and GARCH baselines have no intraday seasonality by
   construction. Dividing their returns by real's U-shaped profile
   would create artificial variance at midday and suppress variance
   at the open/close — actively distorting the data rather than
   cleaning it.

2. **AIL learned real's smile almost exactly.** Intraday variance
   profile comparison showed open/mid ratios of 5.98 (real) vs 5.91
   (AIL), close/mid of 1.52 vs 1.32. The profiles track slot-by-slot.
   Fitting AIL's own FFF versus applying real's would produce nearly
   identical deseasonalised series.

**Resolution:** FFF is now conditional per series. The pipeline
computes the coefficient of variation (CV) of the per-minute variance
profile: flat series (CV ≈ 0) skip FFF entirely; seasonal series
(CV > 0.3) get their own FFF fitted and applied. Empirically
validated: real CV = 0.944, AIL CV = 0.974 (both applied), GBM
CV ≈ 0 (correctly skipped).

This means each series' own seasonality is removed cleanly, and
the metrics see only the non-seasonal structure. If seasonal fidelity
matters, it belongs as a separate diagnostic axis — not smuggled
into the six metrics through a mismatched deseasonalisation.


## Regime-Conditional Tail Design

### What it measures

After stripping predictable patterns (seasonality via FFF), do
return tails get heavier in turbulent market regimes? This is a
conditional property that the unconditional tail metric cannot
capture: a generator could produce the right average tail by being
too thin during crises and too heavy during calm periods, and the
marginal metric would never notice.

### Original design: McNeil-Frey AR-GARCH → GPD (abandoned)

The initial plan followed McNeil & Frey (2000): fit AR-GARCH(1,1)
per ticker to extract standardised residuals, partition time into
volatility regimes, fit GPD on tail exceedances per regime, extract
shape parameter ξ.

This failed empirically on our data. Contiguous-segment analysis
showed median non-NaN run length of 4 bars (real) and 3 bars (AIL).
AR-GARCH needs ≈250+ contiguous observations for stable parameter
estimates. Requiring 250-bar segments discards 79% of real data and
95% of AIL data — and the discarded data is biased toward volatile
periods (where gaps cluster), which is exactly what a regime-
conditional metric should be most sensitive to.

### Standardisation suppresses the signal

A direct comparison tested two approaches:
- **Option A:** standardise returns by a volatility proxy (r_t / σ_t),
  then fit GPD on standardised residuals per regime.
- **Option B:** fit GPD on raw deseasonalised returns per regime,
  using the volatility proxy only for regime labelling.

Option A produced a flat ξ curve across regimes (EWMA: −0.169 to
−0.162), because dividing by the same quantity used for regime
partitioning removes the very signal being measured. Option B showed
a strong regime effect (EWMA: −0.134 to +0.188), with tails clearly
heavier in the turbulent quintile.

**Decision:** no standardisation. The volatility proxy is used only
to assign regime labels, not to transform the returns.

### Volatility proxy: rolling 60-minute std

Two candidates tested — EWMA (λ = 0.94) and rolling 60-minute
causal standard deviation:

- Rolling60 produced better discrimination against a bad generator
  (real-GBM |Δξ| = 0.153 vs EWMA's 0.065).
- Rolling60 produced tighter tracking of a good generator
  (real-AIL |Δξ| = 0.0048 vs EWMA's 0.0080).

Rolling60 wins on both axes. It is also simpler, has no
hyperparameter beyond the window length, and is fully gap-tolerant
(restarts cleanly after NaN runs).

### GPD estimation method

Monte Carlo validation of the WNLS estimator (Park & Kim 2016)
showed reliable ξ recovery (rel_bias < 1%, rmse < 0.08) at n ≥ 500
exceedances. However, regime-cell analysis found ~259,000
exceedances per quintile after pooling across 600 tickers. At this
sample size, MLE and WNLS are both essentially exact. MLE is
retained for simplicity (scipy-native, no custom optimisation).

### GBM baseline discrimination (validated)

GBM prices were generated on the same clock with real's NaN mask
imposed, processed through PreprocessingPipeline (FFF correctly
skipped due to flat CV), and evaluated with Option B / Rolling60:

- Real turbulent-regime ξ: +0.181
- AIL turbulent-regime ξ:  +0.184
- GBM turbulent-regime ξ:  −0.010

Real-GBM gap (0.153) is 32× larger than real-AIL gap (0.0048).
The metric cleanly separates a generator that captures regime-
conditional tail structure from one that does not.

### Final specification

1. Compute rolling 60-min causal std on deseasonalised returns.
2. Pool (volatility, return) pairs across all tickers, drop NaN.
3. Split into quintiles on volatility (boundaries from real data).
4. Within each quintile, fit GPD (MLE, floc=0) on the 5% upper-tail
   exceedances of |return|.
5. Extract ξ per quintile → 5-element feature curve.
6. Gap: weighted MAE between real and synthetic ξ curves, with
   weight on the top two quintiles (stress-testing focus).
