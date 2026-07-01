# fineval

A six-bucket fidelity benchmark for synthetic 1-minute financial return generators, built as an installable Python package. Each bucket measures one orthogonal stylized fact; scores are normalized against a real-vs-real baseline so results are interpretable without a reference model.

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

Load your data, curate it, and run evaluation:

```python
from fineval.data.curate import RealDataLoader, AILSyntheticLoader, CurationPipeline

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
│   ├── config.py                   # global constants (seed, market clock, coverage floor)
│   ├── data/
│   │   ├── loader.py               # BaseLoader ABC + MarketDataset contract
│   │   └── curate.py               # CurationPipeline, RealDataLoader, AILSyntheticLoader
│   ├── preprocessing/              # return computation, deseasonalization (in progress)
│   ├── metrics/                    # six-bucket metric classes (in progress)
│   │   └── base.py
│   ├── bootstrap/                  # matched-N resampling + paired-bootstrap CI (in progress)
│   └── scripts/
│       └── build_curated_datasets.py
│
├── data/
│   └── output_data/                # eval corpus + synthetic outputs + manifest
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

## The six buckets

| ID | Stylized fact | Statistic | Aggregation |
|----|--------------|-----------|-------------|
| B1 | Marginal distribution, 5% tails | Tail-weighted Wasserstein-1 | Pooled |
| B2 | Volatility clustering | Mean ACF gap on \|r\|, lags 60–390 | Per-path, FFT |
| B3 | Leverage effect | Cross-correlation gap Corr(r_t, \|r_{t+k}\|), lags 1–390 | Per-path, FFT |
| B4 | Aggregational kurtosis | Uniform-weighted L-kurtosis gap across horizons {1, 5, 30, 60, 390} min | Per-path then pooled |
| B5 | Multi-scale vol structure | Frobenius gap on cross-scale vol correlation matrix | Per-path |
| B6 | Tail index by vol regime | GPD shape parameter ξ gap across low/high vol regimes | Regime-stratified |

B4 uses **L-kurtosis** (τ₄ = λ₄/λ₂) rather than moment-based excess kurtosis. Financial returns during crisis regimes have tail indices ξ ≈ 0.3–0.5, meaning the eighth moment does not exist. L-kurtosis requires only E[|X|] < ∞ and is stable for all financial return distributions.

---

## Results

Benchmark on the canonical corpus: 60 eval tickers, NASDAQ 1-minute bars, Sep 2019 – Mar 2020 (1,187 windows of 2,520 returns). N = 200 paths, 200 resamples, paired-bootstrap CIs (B = 2,000).

| Bucket | AIL | GARCH | SFAGan |
|--------|-----|-------|--------|
| B1 — Marginal tails | 0.315 [0.291, 0.340] | 0.424 [0.403, 0.445] | 0.296 [0.275, 0.317] |
| B2 — Volatility clustering | 0.260 [0.252, 0.268] | 0.136 [0.131, 0.141] | 0.053 [0.051, 0.055] |
| B3 — Leverage effect | 0.422 [0.419, 0.424] | 0.378 [0.374, 0.381] | 0.401 [0.398, 0.403] |
| B4 — Aggregational kurtosis | 0.543 [0.518, 0.568] | 0.374 [0.347, 0.402] | 0.143 [0.133, 0.155] |
| B5 — Cross-scale vol structure | 0.132 [0.123, 0.143] | 0.044 [0.040, 0.047] | 0.038 [0.035, 0.041] |
| B6 — Tail index by vol regime | 0.130 [0.124, 0.137] | 0.259 [0.247, 0.271] | 0.101 [0.095, 0.107] |
| **Composite (arithmetic)** | **0.300** | **0.269** | **0.172** |
| **Composite (geometric)** | **0.262** | **0.212** | **0.123** |

**Rank: AIL > GARCH > SFAGan**

---

## Methodology

### Data contract

All data enters the framework as a `MarketDataset`: a wide-format price DataFrame with a UTC `DatetimeIndex`, string column names (ticker symbols), and `float64` values (`NaN` where no trade occurred). The `CurationPipeline` enforces this contract and produces evaluation-ready parquet files before any preprocessing or metric computation touches the data.

### Matched-N resampling

For every (bucket, generator) cell:

1. Draw `(idx_a, idx_b)` of size N=200 from the real corpus.
2. `g_rr = gap(real[idx_a], real[idx_b])` — real vs real (noise floor).
3. Draw `idx_s` of size N=200 from the synthetic corpus.
4. `g_sr = gap(real[idx_a], syn[idx_s])` — real vs synthetic, reusing `idx_a`.
5. Repeat 200 times; aggregate means.

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
