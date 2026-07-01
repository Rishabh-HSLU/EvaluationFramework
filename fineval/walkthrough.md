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

The framework ships with two concrete loaders as reference
implementations: `RealDataLoader` (a directory of per-ticker CSVs) and
`AILSyntheticLoader` (a long-format parquet from the AIL generator).
Both live in `data/curate.py` alongside the curation pipeline that
consumes them. To benchmark a new generator with its own raw format,
write a new subclass following the pattern above.

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

*To be written as we build the metric classes.*

---

## 5. The Bootstrap Engine

*To be written as we build the bootstrap module.*

---

## 6. Putting It All Together

*To be written after all modules are complete.*
