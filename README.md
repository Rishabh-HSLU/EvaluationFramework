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
uv sync --frozen --extra data --extra dev
uv run --frozen pytest
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before
running these commands. Run the remaining commands from the repository root.
The `data` extra installs the Alpaca client; the `dev` extra installs the test
dependencies. Keep `uv.lock` unchanged when reproducing a run. The current
cross-machine check uses Python 3.13.16; record the Python version and platform
alongside the Git commit and lockfile used for your run.

The commands below use `scripts.reproduce` to apply consistent NumPy CPU
settings before computation. This restored identical GBM and MSV panels on
the two tested x86_64 machines. Options after `--` go to the selected script.

### 1. Download the raw market data

The canonical experiment downloads the 948-ticker overlap with the supplied
AIL dataset, using Alpaca's SIP feed, 1-minute bars, and split adjustment. The
requested period is September 3, 2019 to March 20, 2020. Your Alpaca account
must have access to these historical SIP bars.

Use hidden prompts to keep credentials out of the source code and command
history:

```bash
uv run --frozen python - <<'PYTHON'
from getpass import getpass
from scripts.fetch_alpaca import download_corpus

download_corpus(
    api_key=getpass("Alpaca API key: "),
    secret_key=getpass("Alpaca secret key: "),
    out_dir="data/raw_intraday",
)
PYTHON
```

Each ticker is saved as `data/raw_intraday/<TICKER>.csv`. The downloader writes
`data/raw_intraday/download_log.csv`, so running the same command again resumes
by skipping successful tickers. It catches individual download errors; a
process that finishes is not by itself proof of a complete download. Before
curation, check that the log has all 948 canonical tickers, each with status
`ok` and a positive bar count, and that their CSV files exist.

A fresh provider download can differ from an earlier snapshot because of
market-data revisions. Preserve the raw inputs and compare their provenance
and curated outputs before claiming an exact reproduction.

### 2. Supply the AIL input and curate the datasets

Obtain the original AIL input parquet from the study authors and place it at:

```text
data/ail_synthetic_data/dataset_US_1-10B_2019-09-2020-03.parquet
```

This external input is not included in Git. It is the input to AIL curation,
not the smaller curated AIL panel. Reproducing AIL model training is a separate
workflow; the benchmark workflow starts from this supplied input. No public
download link is currently provided in this repository.

Loaders normalize each raw source into the common wide-format contract. The
curation pipeline aligns all datasets to the NYSE 1-minute market clock and
writes `data/curated/real_prices.parquet`, `ail_prices.parquet`, and
`curation_metadata.json`. The study snapshot retains 600 real tickers from
the 948-ticker overlap; check the actual retained universe after downloading
and curating fresh data. Both raw input paths are resolved from the repository
root by `scripts.build_curated_datasets`.

```bash
uv run --frozen python -m scripts.reproduce curate
```

### 3. Generate the controls

GBM is the Gaussian, memoryless negative control. MSV is a multi-scale
stochastic-volatility positive control designed primarily for volatility
clustering. Both are generated on the real market clock with the real missing
value mask imposed.

```bash
uv run --frozen python -m scripts.reproduce controls
```

The script writes `data/curated/gbm_prices.parquet` and `msv_prices.parquet`.
The configured seeds are 42 for GBM and 43 for MSV.

### 4. Train and generate SBBTS

The default benchmark also loads SBBTS. Generate its panel before evaluating:

```bash
uv run --frozen python -m scripts.reproduce sbbts-panel
uv run --frozen python -m scripts.reproduce sbbts-train -- --seed 0
uv run --frozen python -m scripts.reproduce sbbts-generate -- \
    --checkpoint data/sbbts/checkpoints/outer_k4.pt \
    --seed 0 \
    --batch-paths 2048 \
    --n-euler 50 \
    --context-on y \
    --invert-iters 0
```

Panel preparation reads the curated real prices and writes training examples
and ticker scales to `data/sbbts/`. Training writes checkpoints and logs there;
generation writes `data/curated/sbbts_prices.parquet`. Training automatically
resumes from `data/sbbts/checkpoints/latest.pt` when present. For a fresh
retraining check, start without checkpoints from another run.

These commands make the current script defaults explicit, including selection
of the fifth outer iteration (`outer_k4.pt`). They do not establish that the
historical panel used every current generation default. Retain the checkpoint
identity, training configuration, generation arguments, and resulting panel
digest for each new run. The choice between the fourth and fifth outer
iterations remains under review for the submission.

The training script requires a CUDA GPU and exits when CUDA is unavailable;
generation can use CUDA or CPU. Record the device and PyTorch/CUDA versions: matching seeds alone does not
guarantee identical retraining outputs across hardware. Retraining is a
separate, substantial computation from evaluating existing panels.

### 5. Run the benchmark

Point estimate only:

```bash
uv run --frozen python -m scripts.reproduce benchmark
```

Fast development run:

```bash
uv run --frozen python -m scripts.reproduce benchmark -- \
    --n-resamples 20 \
    --n-jobs 4
```

Final run with corpus-level outer bootstrap intervals and an independent-seed
numerical-stability audit:

```bash
uv run --frozen python -m scripts.reproduce benchmark -- \
    --n-resamples 100 \
    --tickers-per-draw 200 \
    --seed 42 \
    --n-outer-resamples 200 \
    --n-mc-repeats 20 \
    --n-jobs 4 \
    --replicate-chunk-size 1 \
    --update-canonical
```

This submission configuration uses 100 inner draws, 200 tickers per draw,
seed 42, 200 outer replicates (`O`), and 20 independent inner-seed repeats
(`M`). It evaluates existing panels; it does not retrain generators.

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
updated. The submission configuration above uses 200; the general framework
recommendation remains approximately 1,000 when computational resources permit.

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
benchmark_spec-...csv
aggregate_spec-...csv
outer_metric_replicates_spec-...csv       # when outer bootstrap is enabled
outer_aggregate_replicates_spec-...csv
mc_metric_replicates_spec-...csv          # when MC stability is enabled
mc_aggregate_replicates_spec-...csv
runs_manifest.csv
```

`runs_manifest.csv` records completed runs and caught failures, including
parameters, git commit, duration, output files, and status.

Canonical files are refreshed only when `--update-canonical` is explicitly
passed together with a qualifying outer-bootstrap run:

```text
results/benchmark_results.csv
results/aggregate_results.csv
```

---

## Checking reproduction on a new machine

Clone the same Git commit, install from `uv.lock`, and follow the steps above
without copying curated panels or benchmark results into the working clone.
Supply only the external AIL input for this workflow. Keep historical panels
and results in a separate reference directory for comparison.

Record the commit, Python and dependency versions, raw-input hashes, curation
metadata, training/generation settings, and SHA-256 digests of all five panels
(`real`, `ail`, `gbm`, `msv`, `sbbts`). Compare panel values, index, columns, and
missing-value masks as well as file digests: a serialization difference and a
data difference need separate explanations. Compare benchmark CSV values by
metric, generator, and replicate identifier using the same run settings.

The [run #4 reference bundle](results/run4/README.md) and
[panel digest manifest](docs/panel_digests.txt) are tracked in Git.
The original curated Parquet panels are supplied separately; no public panel
download is included yet. The independent-machine check reproduced the
real/AIL/GBM/MSV panels, but the full Lightning benchmark was stopped.
Do not interpret that partial check as a completed independent reproduction
of the final table.

---

## Development setup

```bash
uv sync --frozen --extra dev
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
│   │   ├── aggregational_gaussianity.py   # M3
│   │   └── regime_tails.py                # M4
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
│   ├── reproducibility/             # reviewer reference and numerical environment
│   ├── panel_digests.txt            # exact curated input identities
│   └── grid_spec_freeze.md          # sensitivity settings and provenance
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
| M3 | Aggregational Gaussianity | Difference in normalized Pearson-kurtosis decay across configured aggregation scales | Session-confined pooled blocks |
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

The implemented estimator samples each synthetic panel independently, including
when ticker labels happen to match.

### Parallel reproducibility

The engine pre-generates all random indices and assigns results by draw or
replicate identifier rather than completion order. Consequently, changing
`--n-jobs`, task chunking, or scheduling does not change the sampled data or the
reported values.

---

## Paper results and reproduction comparison

**[Run #4 is the current paper reference](results/run4/README.md):**
B=100, 200 tickers per draw, seed 42, O=200 and M=20, with all four generators
and the original SBBTS panel.

The bundle contains the final tables, every outer/seed replicate, the exact
configuration, input hashes and environment/source provenance.
The [paper-number index](results/run4/paper_number_map.csv) links 201
reported data/configuration/table values to their source cells and precision.
The [reproducibility index](docs/reproducibility/README.md) distinguishes this
current reference from historical run #3 and development records.

Verify the saved reference without rerunning the benchmark:

~~~bash
uv run --frozen python scripts/check_reproduction.py
~~~

After running the evaluation with the same inputs and settings, compare your
new run using the number printed at completion (5 is an example):

~~~bash
uv run --frozen python scripts/check_reproduction.py --run-number 5
~~~

This compares all six result files by metric, generator and replicate ID.
It reports numerical differences with explicit tolerances and exits nonzero
on a mismatch. See the [run #4 instructions](results/run4/README.md) for exact
panel replay and comparison details. New runs can update the root canonical
CSVs; results/run4/ remains the fixed paper reference.

---

## Diagnostic figures

Generate the M2 absolute-return ACF, M3 kurtosis aggregation curves, and
M4 tail-shape profiles from the five curated run #4 panels:

```bash
python -m scripts.reproduce profiles
```

[Figure outputs and reproduction details](results/figures/run4/README.md)
include PDF/PNG figures, plotted CSV data and provenance. To redraw from
saved CSVs without loading the panels, add `-- --plot-only`.

## Citation

If you use this benchmark, please cite the accompanying paper (forthcoming).
