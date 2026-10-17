# Figure captions

## M2: Absolute-return autocorrelation

Cross-sectional mean autocorrelation of absolute preprocessed one-minute
returns for the real panel and the four fixed generator panels. Panel (a)
shows lags 1–388 minutes; panel (b) shows the 60–388-minute window used by
M2. Lagged products never cross trading-session boundaries. Normalization
uses each ticker's full-sample variance, as in the metric, so long-lag
shrinkage is part of the estimator. Curves use all 600 ticker series over
the evaluation window and are descriptive full-panel features, not
matched-bootstrap score estimates or confidence intervals.

## M3: Kurtosis under temporal aggregation

Pearson kurtosis (a) and normalized kurtosis K(h)/K(1) (b) of preprocessed
returns aggregated over h = 1, 5, 15 and 30 minutes. Each horizon uses the
same final 360 return positions of each eligible full-length session;
November 29 and December 24, 2019 are excluded because their early closes
cannot supply this support. Blocks containing missing returns are
excluded. Returns are pooled across all tickers and eligible sessions
before estimating bias-corrected Pearson kurtosis. The horizontal line in
(a) marks Gaussian kurtosis, 3; the normalized curves in (b) are the
features compared by M3. Lines connect the four evaluated horizons and do
not represent estimates at intermediate horizons.

## M4: Tail shape across relative volatility regimes

Generalized Pareto shape parameter xi fitted to excesses of absolute
preprocessed returns above the within-regime 95th percentile. Regimes Q1
through Q5 are panel-specific quintiles of strictly lagged within-session
rolling volatility, using a 60-minute window with at least 30 prior valid
observations. Q1 denotes the calmest relative regime and Q5 the most
turbulent. Equal quintile labels do not imply equal absolute volatility
thresholds across panels. The horizontal line marks xi = 0, the
exponential-tail case of the generalized Pareto family. Lines connect
discrete regime estimates; no uncertainty bands are implied.

All three figures assess in-sample empirical fidelity over the historical
window, including SBBTS early-stopping dates. The synthetic panels are the
fixed inputs evaluated in run #4. Full-panel profiles help interpret the
feature definitions but do not replace real–real calibration or the
matched-resampling results reported in the benchmark tables.
