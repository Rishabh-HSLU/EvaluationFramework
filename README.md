# fineval

A multi-metric fidelity benchmark for synthetic 1-minute financial return generators.
Each metric targets an empirical stylized fact, and every synthetic-real gap is
benchmarked against an empirical real-corpus resampling baseline so the scores
remain interpretable without selecting a reference generator.

```text
raw market data ──► loaders/curation ──► preprocessing ──► matched estimator ──► metric scores
synthetic data  ──►                  ──►               ──►                   ├─► outer bootstrap CI
                                                                                └─► seed-stability audit
```

For metric $b$, the similarity score is

$$
s_b = \frac{\text{mean}(g_{rr,b})}
           {\text{mean}(g_{rr,b}) + \text{mean}(g_{sr,b})},
$$

where $g_{rr,b}$ is the real-real distance and $g_{sr,b}$ is the
synthetic-real distance.

Interpretation:

- `s_b = 0.5`: the estimated synthetic-real and real-real resampling gaps
  are equal under the observed corpora and configured sampling protocol.
- `s_b < 0.5`: the estimated synthetic-real gap is larger than the empirical
  real-corpus resampling gap.
- `s_b > 0.5`: the estimated synthetic-real gap is smaller than that
  resampling gap; this can reflect close agreement, but also over-smoothing
  or reduced synthetic variability.
- The score is a relative gap comparison, not an equivalence test and not a
  grade to maximize.

The optional aggregate deviation score is

$$
G_{dev} = \exp\left(\frac{1}{K} \sum_{k=1}^{K} \left| \log \frac{\text{mean}(g_{sr,k})} {\text{mean}(g_{rr,k})} \right| \right).
$$

- `G_dev = 1` only when every included estimated gap ratio equals one.
- `G_dev > 1` indicates aggregate multiplicative departure from the empirical
  real-corpus reference.
- Ratios above and below one cannot cancel.
- `G_dev` loses the direction and identity of individual discrepancies and
  must therefore be reported alongside the per-metric results.

---

## Quick start

```bash
git clone https://github.com/Rishabh-HSLU/EvaluationFramework.git
cd EvaluationFramework
uv sync
```

### 1. Download the raw market data

The canonical experiment uses 948 tickers of 1-minute Alpaca bars from
September 2019 to March 2020.

```bash
pip install "alpaca-py>=0.13"

python -m scripts.fetch_alpaca \
    --api-key <YOUR_KEY> \
    --secret <YOUR_SECRET> \
    --out-dir data/raw_intraday
```

The downloader writes a checkpoint log to `download_log.csv`, so interrupted
runs can resume.

### 2. Curate the datasets

Loaders normalize each raw source into the common wide-format contract. The
curation pipeline aligns all datasets to the NYSE 1-minute market clock and
writes evaluation-ready parquet files.

```bash
uv run python -m scripts.build_curated_datasets
```

### 3. Generate the controls

GBM is the Gaussian, memoryless negative control. MSV is a multi-scale
stochastic-volatility positive control designed primarily for volatility
clustering. Both are generated on the real market clock with the real missing
value mask imposed.

```bash
uv run python -m scripts.baseline_generation
```

### 4. Run the benchmark

Point estimate only:

```bash
uv run python -m scripts.run_benchmark
```

Fast development run:

```bash
uv run python -m scripts.run_benchmark \
    --n-resamples 20 \
    --n-jobs 4
```

Final run with corpus-level outer bootstrap intervals and an independent-seed
numerical-stability audit:

```bash
uv run python -m scripts.run_benchmark \
    --n-resamples 100 \
    --n-outer-resamples 1000 \
    --n-mc-repeats 20 \
    --n-jobs 0 \
    --update-canonical
```

Worker settings:

- `--n-jobs 0`: automatic worker budget; leaves one logical CPU free and caps the default at 8.
- `--n-jobs -1`: use every logical CPU.
- `--inner-chunk-size`: matched draws per submitted inner task.
- `--replicate-chunk-size`: outer or Monte Carlo replicates per process task.

Random indices and replicate seeds are generated before parallel execution, so
results are independent of worker completion order, worker count, and chunk
size.

---

## Uncertainty and numerical stability

The benchmark separates three different quantities.

### Point estimate

The inner matched estimator repeatedly draws ticker subsets from the fixed
observed panels and estimates the mean real-real and synthetic-real gaps. The
number of inner draws is controlled by `--n-resamples`.

Increasing `--n-resamples` reduces Monte Carlo error in the score calculation.
It does not create a corpus-level confidence interval.

### Outer-bootstrap confidence interval

With `--n-outer-resamples O`, each outer replicate:

1. resamples the complete real ticker corpus with replacement;
2. resamples each complete synthetic corpus with replacement;
3. reruns the full inner matched estimator on those resampled corpora;
4. returns one metric score and one aggregate score.

The reported interval is the 2.5th to 97.5th percentile of the resulting outer
scores.
The CLI requires at least 40 outer replicates before canonical results can be
updated; approximately 1,000 are recommended for final reporting.

The current outer bootstrap starts from the already preprocessed real and
synthetic panels. Its interval therefore measures ticker-corpus resampling
uncertainty conditional on:

- the fitted preprocessing transformation;
- the fitted generator;
- the available generated synthetic panel.

It does not include generator retraining, synthetic regeneration, or
re-estimation of preprocessing inside each outer replicate.

### Monte Carlo stability

With `--n-mc-repeats R`, the benchmark reruns the inner estimator under
independent seeds while keeping the real and synthetic corpora fixed. It reports
mean, standard deviation, and range across seeds.

These values quantify numerical stability and are not confidence intervals.

---

## Outputs

Every run writes parameter- and timestamp-stamped files to `results/`:

```text
benchmark_B...csv
aggregate_B...csv
outer_metric_replicates_B...csv       # when outer bootstrap is enabled
outer_aggregate_replicates_B...csv
mc_metric_replicates_B...csv          # when MC stability is enabled
mc_aggregate_replicates_B...csv
runs_manifest.csv
```

`runs_manifest.csv` records completed, failed, and interrupted runs, including
parameters, git commit, duration, output files, and status.

Canonical files are refreshed only when `--update-canonical` is explicitly
passed together with a qualifying outer-bootstrap run:

```text
results/benchmark_results.csv
results/aggregate_results.csv
```

---

## Development setup

```bash
uv sync --dev
uv run pre-commit install
uv run pre-commit run --all-files
```

Manual checks:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest
```

---

## Repository layout

```text
EvaluationFramework/
├── fineval/                         # reusable package code
│   ├── config.py                    # global seeds, market clock and metric hyperparameters
│   ├── data/
│   │   ├── loader.py                # BaseLoader and MarketDataset contract
│   │   └── curate.py                # curation pipeline and concrete dataset loaders
│   ├── preprocessing/
│   │   ├── pipeline.py              # log returns, overnight mask and conditional FFF
│   │   └── fff.py                   # Flexible Fourier Form deseasonalizer
│   ├── metrics/
│   │   ├── base.py                  # feature, distance and normalization contract
│   │   ├── tail_weighted_marginal.py      # M1
│   │   ├── volatility_clustering.py       # M2
│   │   ├── aggregational_gaussianity.py   # M4
│   │   └── regime_tails.py                # M6
│   ├── bootstrap/
│   │   ├── engine.py                # public MatchedTickerBootstrap API
│   │   ├── execution.py             # deterministic parallel inner draws
│   │   ├── uncertainty.py           # outer bootstrap and seed-stability analyses
│   │   ├── replicate_execution.py   # parallel replicate workers and chunk execution
│   │   └── models.py                # repeated-analysis result container
│   └── benchmark/
│       ├── runner.py                # high-level benchmark orchestration
│       ├── config.py                # benchmark paths, labels and metric suite
│       ├── datasets.py              # curated loading and preprocessing
│       ├── reporting.py             # logs, progress bars, diagnostics and tables
│       └── artifacts.py             # CSV persistence and run manifest
├── scripts/                         # executable entry points only
│   ├── fetch_alpaca.py
│   ├── build_curated_datasets.py
│   ├── baseline_generation.py
│   └── run_benchmark.py
├── docs/
│   ├── walkthrough.md               # narrative framework tour
│   └── reasoning.md                 # design decisions and empirical evidence
├── tests/
├── data/
├── results/
├── pyproject.toml
└── README.md
```

The distinction is intentional:

- `fineval/` contains importable and testable application logic;
- `scripts/` contains thin command-line launchers;
- `docs/` contains project documentation;
- `results/` contains generated artifacts, not source code.

---

## Adding a generator

Subclass `BaseLoader` and implement `_load_raw()` to return a wide-format price
DataFrame:

```python
import pandas as pd

from fineval.data.loader import BaseLoader


class MyLoader(BaseLoader):
    def __init__(self, path: str):
        super().__init__(name="MyGenerator", is_synthetic=True)
        self.path = path

    def _load_raw(self) -> pd.DataFrame:
        dataframe = pd.read_parquet(self.path)
        # Required format:
        # index   = UTC DatetimeIndex
        # columns = unique string path/ticker names
        # values  = float64 close prices, with NaN where unavailable
        return dataframe
```

Then add the loader to the curation or benchmark workflow. The curation pipeline
handles market-clock alignment, coverage filtering, and the common dataset
contract.

---

## Metrics

| ID | Stylized fact | Statistic | Aggregation |
|----|---|---|---|
| M1 | Tail-weighted marginal distribution | Tail-weighted L1 distance between pooled empirical quantile functions of per-ticker standardized returns | Pooled marginal |
| M2 | Volatility clustering | Gap between absolute-return ACFs over configured intraday lags | Per path, then cross-sectional mean |
| M3 | Aggregational Gaussianity | Difference in excess-kurtosis decay across configured aggregation scales | Session-confined pooled blocks |
| M4 | Regime-conditional tails | Difference in GPD tail-shape estimates across volatility quintiles | Regime-stratified pooled tails |

Metric hyperparameters are defined in `fineval/config.py`. The metric suite used
by the full benchmark is assembled in `fineval/benchmark/config.py`.

---

## Methodology

### Data contract

All data enters the framework as a `MarketDataset`: a wide price DataFrame with
a UTC `DatetimeIndex`, unique string columns, and `float64` values. Missing
observations remain `NaN`. Curation produces evaluation-ready parquet files
before preprocessing or metric calculation begins.

### Matched-N ticker estimator

For each inner draw $b=1,\ldots,B$:

1. sample `idx_a` and `idx_b` independently from the real panel;
2. compute $g_{rr,b}=d(\text{real}[idx_a],\text{real}[idx_b])$;
3. sample `idx_s` independently from each synthetic panel;
4. compute $g_{sr,b}=d(\text{real}[idx_a],\text{synthetic}[idx_s])$.

The same real reference sample `idx_a` is used in the real-real and
synthetic-real comparisons. This matched design reduces irrelevant per-draw
noise while preserving independent real comparison and synthetic samples.

Synthetic indices are reused from the real draw only when a generator has an
explicit one-to-one ticker correspondence. Unconditional synthetic paths are
sampled independently.

### Parallel reproducibility

The engine pre-generates all random indices and assigns results by draw or
replicate identifier rather than completion order. Consequently, changing
`--n-jobs`, task chunking, or scheduling does not change the sampled data or the
reported values.

---

## Results

The README does not duplicate a static benchmark table because those values can
become stale when metrics, preprocessing, corpora, or uncertainty settings
change. The current canonical results are stored in:

```text
results/benchmark_results.csv
results/aggregate_results.csv
```

Regenerate them with a documented final configuration, for example:

```bash
uv run python -m scripts.run_benchmark \
    --n-resamples 100 \
    --n-outer-resamples 1000 \
    --n-mc-repeats 20 \
    --update-canonical
```

The command prints the same per-metric and aggregate tables in Markdown format.

---

## Citation

If you use this benchmark, please cite the accompanying paper (forthcoming).
