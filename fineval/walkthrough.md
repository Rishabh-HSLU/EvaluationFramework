## 2. The Data Layer

### The starting point

Before the framework can do anything, it needs data. Two kinds:

- **Real data** — actual close prices from a stock exchange.
- **Synthetic data** — prices produced by whatever generator we want
  to evaluate.

These can arrive in all sorts of formats. Maybe the real data is
hundreds of separate CSV files, one per ticker. Maybe the synthetic
data is a single parquet file with every ticker stacked vertically,
or a wide Excel spreadsheet. The formats differ, but the initial
information is the same: a price, for a ticker, at a point in time.

### The contract

The framework doesn't care about the data's starting point. It cares
how the data arrives at preprocessing. That arrival format is the
**contract**, and it looks like this:

```
                        AAPL    MSFT    GOOG
2019-09-03 09:30 UTC    205.1   137.8   1205.3
2019-09-03 09:31 UTC    205.2   NaN     1205.5
2019-09-03 09:32 UTC    205.3   137.9   1205.4
...
```

Rows are timestamps. Columns are tickers. Values are close prices.

The rules:
- The index must be a `DatetimeIndex` in UTC.
- Column names must be strings (ticker symbols).
- Values must be floats. Where there was no trade, the value is `NaN`.

Why UTC? Because if you mix timezones, a "9:30" in New York and a
"9:30" in the file metadata might mean different things. UTC removes
the ambiguity. (NYSE's regular session, 09:30–16:00 Eastern, lands at
13:30–20:00 UTC during summer and 14:30–21:00 UTC after daylight
saving ends — the same wall-clock session, two different UTC windows.
Storing in UTC keeps this unambiguous; conversion for display happens
later if needed.)

Why `NaN` for missing trades? Because the framework needs to know
where data is genuinely missing before it decides what to do about
those gaps. If we pre-fill zeros or forward-fill prices before the
framework sees them, we've made a methodological decision that the
framework can't undo.

### The container: `MarketDataset`

Once data is in the right format, it gets wrapped in a `MarketDataset`
object. This is a simple container — it holds the price DataFrame, a
name (like "Real" or "GARCH"), and a flag indicating whether the data
is synthetic.

```python
MarketDataset(
    prices       = df,          # the wide DataFrame
    name         = "AIL",       # shows up in plots and reports
    is_synthetic = True,        # tells the evaluator this is synthetic
    metadata     = {"version": "2.1"}  # optional, framework ignores it
)
```

It doesn't do anything to the data. It's a labeled envelope.

### The loader: getting messy raw data into the contract

Raw data rarely shows up in the contract format. We need to read
files, pivot columns, rename tickers, parse timestamps. This is the
loader's job.

The framework provides a `BaseLoader` — an abstract class with one
rule: **implement `_load_raw()` and return a wide DataFrame** matching
the contract. The base class handles validation automatically.

Here's the pattern:

```python
class MyLoader(BaseLoader):
    def __init__(self, path):
        super().__init__(name="MyGenerator", is_synthetic=True)
        self.path = path

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        # ... whatever reshaping your format needs ...
        return df

# Usage
dataset = MyLoader("path/to/data.parquet").load()
```

When you call `.load()`, three things happen in order:
1. Your `_load_raw()` runs and returns a DataFrame.
2. The base class validates it against the contract.
3. If validation passes, it wraps the DataFrame in a `MarketDataset`
   and returns it.

If validation fails — wrong index type, missing timezone, non-string
columns — you get a clear error message telling you exactly what's
wrong.

The framework ships with four concrete loaders as reference
implementations, all in `data/curate.py` alongside the curation
pipeline that consumes them:

- `RealDataLoader` — a directory of per-ticker CSVs (raw real data).
- `AILSyntheticLoader` — a long-format parquet from the AIL generator.
- `CuratedParquetLoader` — a wide-format parquet that is already on
  the curated market clock; used to reload the curation output
  without re-running the raw loaders.
- `GBMBaselineLoader` — the GBM baseline parquet produced by
  `scripts/baseline_generation.py`. GBM is generated directly on the
  curated clock with real's NaN mask imposed, so it is curated by
  construction and skips the `CurationPipeline` entirely.

To benchmark a new generator with its own raw format, write a new
subclass following the pattern above.

### Curation: from raw loaded data to evaluation-ready files

Loading produces two `MarketDataset` objects — one real, one
synthetic — each carrying whatever ticker universe and timestamps its
source file happened to contain. The two are not yet comparable: they
may share most tickers but not all; their timestamps may not align
to the trading calendar; some tickers may be present but too sparse
to support reliable estimation.

The `CurationPipeline` resolves these issues in one place. It:

1. Intersects the ticker universes — only tickers present in both
   real and synthetic survive.
2. Reindexes both onto the NYSE 1-minute regular-session market
   clock — bars outside the regular session are dropped, missing
   minutes become NaN.
3. Computes per-ticker coverage on each side and retains only tickers
   whose worse side meets a 70% coverage floor.

The output is two parquet files of identical shape, sharing the same
timestamp index and the same column order, plus a metadata JSON that
records every parameter used and SHA-256 hashes of the source files
for reproducibility. Everything downstream of this point — preprocessing,
metrics, bootstrap — operates only on these curated files. The raw
sources are never touched again.

The reasoning behind the 70% coverage floor and the no-imputation
decision that motivates the curation criterion is documented
separately in `reasoning.md`.

### Why this design?

Because the alternative is chaos. Without a contract, every piece of
downstream code has to guess what the data looks like. Is the index a
DatetimeIndex or strings? Are the columns tickers or OHLCV fields?
Are missing values zeros or NaNs?

With a contract — and a curation step that enforces it — preprocessing
can assume the data is valid. Metrics can assume the data is valid.
The bootstrap can assume the data is valid. Nobody checks. Nobody
guesses. The work was done once, at the door.

---

## 3. Preprocessing

The preprocessing pipeline acts as the bridge between curated raw prices and the stationary, approximately unit-variance returns expected by the downstream evaluation metrics. It has two main responsibilities: computing clean log returns and conditionally handling intraday seasonality.

### Computing Log Returns and Masking Overnight Gaps

The pipeline first converts the `(T, N)` price matrix into log returns using standard differencing (`np.log(prices).diff()`).

Crucially, it must handle the overnight gap. The regular stock market session closes at 16:00 ET and reopens the next day at 09:30 ET. The return observed at 09:31 ET incorporates 17.5 hours of accumulated overnight news, making its variance fundamentally different from a standard 1-minute intraday return. If left in the dataset, these massive overnight jumps will distort any metric measuring tail behavior or volatility clustering.

The pipeline applies an **overnight mask**, explicitly setting the 09:31 ET return (the first minute of the session) to `NaN`. This ensures that all returns passed to the metrics are strictly 1-minute intraday returns, preserving the statistical homogeneity of the sample.

### Conditional Intraday Deseasonalization (FFF)

Real equity returns are not stationary across the trading day. They exhibit a U-shaped variance profile (the "volatility smile"): high variance at the open, dropping at midday, and rising into the close. If left untreated, this deterministic pattern inflates autocorrelations and dominates the marginal distribution. The standard solution is the Flexible Fourier Form (FFF) deseasonalization, which fits a Fourier curve to the log-variance profile and divides it out.

However, applying FFF blindly is dangerous when evaluating synthetic generators. Standard baseline generators like Geometric Brownian Motion (GBM) or standard GARCH are structurally flat—they have no intraday smile. If the pipeline divides a flat GBM series by a U-shaped variance profile, it artificially suppresses the open/close and inflates the midday, actively injecting an inverted smile and corrupting the baseline.

To ensure fairness across all generator types, the pipeline uses **conditional deseasonalization**:

1. **Detect the smile:** For each dataset, the pipeline calculates the pooled variance for each minute of the day and computes the Coefficient of Variation (CV) of this profile.
2. **Branching logic:** A flat series (like GBM) will have a CV near zero (~0.05). A seasonal series (like real data or advanced models like AIL) will have a high CV (~0.6+). We use a robust threshold of `0.3` to separate the regimes.
3. **Application:** If the CV > 0.3, the pipeline fits an FFF model native to that dataset and deseasonalizes the returns. If the CV < 0.3, it recognizes the series as flat and passes it through unchanged.

This guarantees that sophisticated generators are evaluated on their underlying signal (after removing their properly learned smile), while basic baselines are evaluated on their raw output without being penalized by preprocessing artifacts.

---

## 4. Stylized Facts and Metrics

### The idea

Real financial returns have well-documented statistical signatures —
*stylized facts* (Cont, 2001). A good synthetic generator should
reproduce them; a naive one won't. Each metric isolates one stylized
fact and turns "how well does the generator reproduce it?" into a
number.

### The contract: `BaseMetric`

Every metric subclasses `BaseMetric` (`metrics/base.py`) and
implements three methods:

1. `extract_features(sample)` — takes one panel of deseasonalized
   returns (timestamps × tickers) and returns a fixed-length feature
   vector. Stateless: no fit step, no reference distribution. Regime
   boundaries, quantile cuts, etc. are computed from the sample
   itself.
2. `compute_distance(fa, fb)` — a non-negative scalar distance between
   two feature vectors. Must tolerate NaN by masking: only dimensions
   where both vectors are finite are compared.
3. `normalize(g_rr, g_sr)` — turns raw distance arrays into a [0, 1]
   similarity score (see Section 5 for where those arrays come from).

The NaN policy mirrors the framework-wide no-imputation principle:
where the data is too thin for a stable estimate, the feature is NaN —
absence is preserved, never fabricated over.

### The four implemented metrics

**M1 — Unconditional heavy tails** (`unconditional_heavy_tails.py`).
The marginal distribution of returns. Features: the empirical quantile
function on a 5001-point grid, pooled across all tickers. Distance: a
tail-weighted Wasserstein-1 — the mean absolute quantile gap under a
smooth weight `w(u) = 1 + λ[u^-α + (1-u)^-α]` that keeps the whole
distribution at baseline weight 1 while progressively up-weighting the
extremes (α = 0.3, λ = 1.0, both validated by hyperparameter sweeps —
see `scripts/reasoning.md`).

**M2 — Volatility clustering** (`volatility_clustering.py`).
Long-memory temporal dependence. Features: the autocorrelation
function of |returns| per ticker, averaged cross-sectionally, with the
lag-k numerator confined within session boundaries so no pair ever
spans an overnight gap. Distance: summed absolute ACF gap over lags
60–390 — the long-lag regime that short-memory generators cannot fake.

**M4 — Aggregational Gaussianity** (`aggregational_gaussianity.py`).
Real returns are heavy-tailed at 1-minute resolution but converge
toward Gaussian as returns are aggregated to coarser scales. Features:
the normalized excess-kurtosis ratio κ(k)/κ(1) at scales {1, 5, 15,
30} minutes, with aggregation confined within sessions and NaN
propagating honestly through incomplete blocks. Distance: masked mean
absolute difference between decay curves. GBM is flat at 1.0
everywhere (Gaussian at every scale); real decays; a good generator
matches the decay *rate*.

**M6 — Regime-conditional tails** (`regime_tails.py`). Do tails get
heavier when the market is turbulent? Features: a causal rolling
60-minute volatility proxy (session-aware, restarted at every open)
assigns each observation to one of five self-labeled volatility
quintiles; within each quintile a Generalized Pareto Distribution is
fitted to the top 5% of |returns|, and the shape parameter ξ is the
feature. Distance: weighted mean absolute ξ gap, with extra weight on
the turbulent quintiles (stress-testing emphasis).

M3 (leverage effect) and M5 (multi-scale volatility structure) are
planned but not yet implemented.

Every hyperparameter lives in `config.py`, and every non-obvious
design decision — including two session-boundary bugs found and fixed
along the way — is documented with its empirical evidence in
`scripts/reasoning.md`.

---

## 5. The Bootstrap Engine

### Why a bootstrap at all

Suppose M2's distance between real and synthetic is 2.79. Is that
good? Bad? Raw distances have no scale of their own. The trick: ask
how far two *independent samples of real data* are from each other
under the same metric. That real-vs-real distance is the noise floor —
the discrepancy you'd measure even for a perfect generator, purely
from sampling variability.

### Matched-N ticker resampling

The engine (`bootstrap/engine.py`, `MatchedTickerBootstrap`) makes
this operational. For each of B resamples (default 100):

1. Draw two ticker subsamples A and B (200 tickers, with replacement)
   from the real panel.
2. `g_rr[b] = distance(features(A), features(B))` — the noise floor.
3. For each generator, draw a subsample S from its panel and compute
   `g_sr[b] = distance(features(A), features(S))` — the generator gap.

The same real draw A is shared between `g_rr` and every generator's
`g_sr` (that's the "matched" part), so per-draw sampling noise
partially cancels out of the final score.

### The score

Each metric's `normalize` maps the two arrays into a similarity score:

```
s = mean(g_rr) / (mean(g_rr) + mean(g_sr))
```

- `s = 1.0` — zero measured discrepancy.
- `s = 0.5` — real-sample parity: the generator is as far from real
  as real is from itself. You cannot ask for more.
- `s < 0.5` — the generator deviates more than sampling noise alone
  explains; the smaller, the worse.

### Confidence intervals

The 95% CI comes from a paired bootstrap over the resample index: the
same index array is drawn into both `g_rr` and `g_sr` per iteration,
preserving the per-draw correlation created by sharing A. This yields
narrower, honest CIs versus resampling the two arrays independently.

---

## 6. Putting It All Together

### The one-command benchmark

`scripts/run_benchmark.py` wires every layer into a single run:

```
prices (curated parquet)
   │  loaders: CuratedParquetLoader (Real, AIL), GBMBaselineLoader
   ▼
PreprocessingPipeline          log returns → overnight mask → conditional FFF
   │  per (real, synthetic) pair
   ▼
MatchedTickerBootstrap         B resamples × {M1, M2, M4, M6} × {AIL, GBM}
   │
   ▼
benchmark table                score [95% CI] per (metric, generator)
```

Run it from the repository root:

```bash
uv run python -m fineval.scripts.run_benchmark
# or, for a quick pass:
uv run python -m fineval.scripts.run_benchmark --n-resamples 20
```

It prints the markdown benchmark table and writes the tidy results to
`scripts/results/benchmark_results.csv`.

### The GBM baseline

`scripts/baseline_generation.py` produces the anchor at the low end of
the score range. It calibrates per-ticker drift and volatility from
the curated real data's overnight-masked 1-minute log returns,
simulates one i.i.d.-Gaussian GBM path per ticker on the same market
clock, and imposes real's exact NaN mask — so both sides carry
identical missingness and no metric can score coverage instead of
dynamics. Regenerate it (deterministic, seed 42) with:

```bash
uv run python -m fineval.scripts.baseline_generation
```

The two anchors make every score interpretable: the real-vs-real
baseline defines what "indistinguishable" looks like (0.5), and GBM
defines what "structurally wrong" looks like. Any generator worth
evaluating should land between them, and *where* it lands — per
metric — tells you which stylized facts it captures and which it
misses.
