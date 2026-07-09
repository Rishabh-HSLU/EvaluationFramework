# fineval

A multi-metric fidelity benchmark for synthetic 1-minute financial return generators.
Each metric measures empirical stylized fact; scores are normalized against a
real-vs-real baseline so results are interpretable without a reference model.

```
real market data  ──► loader ──► curate ──► preprocess ──► metrics ──► s_b ∈ [0, 1]
synthetic data    ──►        ──►        ──►            ──►
```

`s_b = mean(g_rr) / (mean(g_rr) + mean(g_sr))`

Interpretation:

- `s_b = 1.0` means zero measured discrepancy between synthetic and real data under bucket `b`.
- `s_b = 0.5` means the synthetic data are at real-sample parity: the synthetic-vs-real gap equals the natural real-vs-real baseline variability.
- `s_b > 0.5` means the synthetic-vs-real gap is smaller than the real-vs-real baseline gap. In other words, the synthetic data are closer to the reference real sample than an independent real sample would be under that bucket.
- `s_b < 0.5` means the synthetic data deviate more from real data than two independent real samples deviate from each other.
- `s_b → 0` indicates increasing structural divergence from the real data along that bucket.

---

## Quick start

```bash
git clone https://github.com/Rishabh-HSLU/EvaluationFramework.git
cd EvaluationFramework
uv sync
```

**1. Curate the raw data** (once). Loaders normalize each raw format into the
wide-format contract; the curation pipeline aligns everything onto the NYSE
1-minute market clock:

```python
from fineval.data import RealDataLoader, AILSyntheticLoader, CurationPipeline

real = RealDataLoader(directory="data/raw_intraday").load()
ail  = AILSyntheticLoader(parquet_path="data/ail.parquet").load()

pipeline = CurationPipeline(
    real_dataset=real,
    synthetic_datasets=[ail],
    start_date="2019-09-03",
    end_date="2020-03-20",
    output_dir="data/curated",
)
pipeline.run()
```

**2. Generate the baselines** (deterministic, seeds from `fineval/config.py`).
GBM is the negative control (Gaussian, memoryless, rejected by every metric);
MSV — a two-factor multi-scale stochastic volatility model — is the positive
control for volatility clustering (M2). Both are calibrated per ticker to the
curated real data and generated on the same market clock with real's NaN mask
imposed:

```bash
uv run python -m fineval.scripts.baseline_generation
```

**3. Run the benchmark.** Loads Real, AIL, GBM and MSV through their loaders,
preprocesses each pair, runs the matched-N ticker bootstrap over all metrics,
and prints the benchmark table:

```bash
uv run python -m fineval.scripts.run_benchmark              # B=100 resamples
uv run python -m fineval.scripts.run_benchmark --n-resamples 20   # quick pass
```

Results are printed as a markdown table and saved to a per-run CSV in
`fineval/scripts/results/`, stamped with the run parameters and a
timestamp (e.g. `benchmark_B100_m200_seed42_20260702-123500.csv`).
Only a run at the default parameters also refreshes the canonical
`fineval/scripts/results/benchmark_results.csv`; quick passes
(`--n-resamples 20`, reduced `--tickers-per-draw`, non-default seeds)
never overwrite it.

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

```
EvaluationFramework/
├── fineval/                        # core package
│   ├── config.py                   # global constants (seed, market clock, metric hyperparameters)
│   ├── data/
│   │   ├── loader.py               # BaseLoader ABC + MarketDataset contract
│   │   └── curate.py               # CurationPipeline + loaders (Real, AIL, curated parquet, GBM, MSV)
│   ├── preprocessing/
│   │   ├── pipeline.py             # log returns, overnight mask, conditional deseasonalization
│   │   └── fff.py                  # Flexible Fourier Form intraday volatility deseasonalizer
│   ├── metrics/
│   │   ├── base.py                 # BaseMetric ABC (features / distance / normalize contract)
│   │   ├── unconditional_heavy_tails.py    # M1
│   │   ├── volatility_clustering.py        # M2
│   │   ├── aggregational_gaussianity.py    # M3
│   │   └── regime_tails.py                 # M4
│   ├── bootstrap/
│   │   └── engine.py               # MatchedTickerBootstrap + paired-bootstrap CI
│   ├── scripts/
│   │   ├── build_curated_datasets.py       # raw sources → data/curated/*.parquet
│   │   ├── baseline_generation.py          # GBM + MSV baselines → data/curated/*.parquet
│   │   ├── run_benchmark.py                # full benchmark table (this is the entry point)
│   │   ├── reasoning.md                    # design decisions + empirical evidence
│   │   └── data/curated/                   # evaluation-ready parquet files + metadata
│   └── walkthrough.md              # narrative tour of the whole framework
│
├── tests/
├── pyproject.toml
└── README.md
```

---

## Adding your own generator

Subclass `BaseLoader` and implement `_load_raw()` to return a wide-format price DataFrame:

```python
from fineval.data.loader import BaseLoader
import pandas as pd

class MyLoader(BaseLoader):
    def __init__(self, path: str):
        super().__init__(name="MyGenerator", is_synthetic=True)
        self.path = path

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        # pivot to wide format: index=timestamps (UTC DatetimeIndex),
        # columns=ticker symbols, values=float64 close prices
        return df

dataset = MyLoader("path/to/data.parquet").load()
```

Then pass it to `CurationPipeline` alongside the real dataset. The pipeline handles ticker intersection, market clock alignment, and coverage filtering automatically.

---

## The Metrics

| ID | Stylized fact             | Statistic | Aggregation | Status |
|----|---------------------------|-----------|-------------|--------|
| M1 | Unconditional Heavy Tails | Tail-weighted Wasserstein-1 on the quantile function (α=0.3, λ=1.0) | Pooled | ✅ |
| M2 | Volatility clustering     | Summed ACF gap on \|r\|, lags 60–390, session-confined pairs | Per-path, cross-sectional mean | ✅ |
| M3 | Aggregational Gaussianity | Excess-kurtosis decay ratio κ(k)/κ(1) across scales {1, 5, 15, 30} min | Session-confined blocks, pooled | ✅ |
| M4 | Regime-conditional tails   | GPD shape parameter ξ gap across 5 self-labeled volatility quintiles | Regime-stratified, pooled | ✅ |

The rationale behind every statistic, hyperparameter and design revision is
documented with its empirical evidence in `fineval/scripts/reasoning.md`.

---

## Results

Similarity scores `s ∈ [0, 1]` (95% paired-bootstrap CI in brackets);
0.5 means real-sample parity. B=100 resamples, 200 tickers per draw, seed 42,
Sep 2019 – Mar 2020, 600-ticker universe.

*(regenerate with `uv run python -m fineval.scripts.run_benchmark`)*

| Metric | Stylized fact | AIL | GBM | MSV |
|--------|---|---|---|---|
| M1     | Unconditional heavy tails | 0.478 [0.447, 0.509] | 0.025 [0.023, 0.028] | 0.025 [0.023, 0.028] |
| M2     | Volatility clustering | 0.203 [0.181, 0.226] | 0.019 [0.016, 0.021] | 0.265 [0.237, 0.291] |
| M3     | Aggregational Gaussianity | 0.378 [0.346, 0.410] | 0.121 [0.108, 0.136] | 0.170 [0.152, 0.189] |
| M4     | Regime-conditional tails | 0.485 [0.456, 0.514] | 0.047 [0.043, 0.051] | 0.133 [0.122, 0.145] |

AIL tracks real-sample parity (0.5) closely on M1 and M6, is weaker on M2
(long-memory volatility clustering) and M4 (aggregational kurtosis decay).
GBM is rejected by every metric — as expected for a Gaussian, memoryless
baseline with no intraday structure — confirming all four metrics discriminate
correctly between a strong and a trivial generator. MSV, a positive control
purpose-built to exhibit volatility clustering and nothing else, beats AIL on
M2 (0.265 vs. 0.203) while losing to it everywhere else — the signature of a
working single-purpose control that validates the benchmark itself, not just
the generator.

Between this run and the first (AIL/GBM only, `scripts/reasoning.md` §
Benchmark Results), the bootstrap's RNG was restructured into independent
per-cell streams (see `bootstrap/engine.py`); every score moved within its
prior confidence interval, not outside it, confirming the restructure changed
sampling machinery, not the underlying measurement.

---

## Methodology

### Data contract

All data enters the framework as a `MarketDataset`: a wide-format price DataFrame with a UTC `DatetimeIndex`, string column names (ticker symbols), and `float64` values (`NaN` where no trade occurred). The `CurationPipeline` enforces this contract and produces evaluation-ready parquet files before any preprocessing or metric computation touches the data.

### Matched-N ticker resampling

For every bootstrap resample `b` (default B=100), `MatchedTickerBootstrap` draws
ticker subsamples of size N=200 with replacement:

1. Draw ticker index sets `idx_a`, `idx_b` from the real panel.
2. `g_rr[b] = gap(real[:, idx_a], real[:, idx_b])` — real vs real (noise floor).
3. Draw `idx_s` from each synthetic panel.
4. `g_sr[b] = gap(real[:, idx_a], syn[:, idx_s])` — real vs synthetic, reusing `idx_a`.

Features are extracted per metric on each subsample; the real draw `idx_a` is
shared between `g_rr` and every generator's `g_sr`, so per-draw sampling noise
partially cancels out of the score `mean(g_rr) / (mean(g_rr) + mean(g_sr))`.

### Paired-bootstrap CI

```python
for b in range(2000):
    idx  = rng.integers(0, n_resamples, n_resamples)  # same index into both arrays
    rr_b = g_rr[idx].mean()
    sr_b = g_sr[idx].mean()
    boot[b] = rr_b / (rr_b + sr_b)
lo, hi = np.percentile(boot, [2.5, 97.5])
```

Sharing the index per iteration preserves the per-i correlation from construction, producing CIs 5–12% narrower than the unpaired form.

---

## Citation

If you use this benchmark, please cite the accompanying paper (forthcoming).
