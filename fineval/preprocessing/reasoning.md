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
