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

## The Metrics

| ID | Stylized fact             | Statistic | Aggregation |
|----|---------------------------|-----------|-------------|
| M1 | Unconditional Heavy Tails | Tail-weighted Wasserstein-1 | Pooled |
| M2 | Volatility clustering     | Mean ACF gap on \|r\|, lags 60–390 | Per-path, FFT |
| M3 | Leverage effect           | Cross-correlation gap Corr(r_t, \|r_{t+k}\|), lags 1–390 | Per-path, FFT |
| M4 | Aggregational kurtosis    | Uniform-weighted L-kurtosis gap across horizons {1, 5, 30, 60, 390} min | Per-path then pooled |
| M5 | Multi-scale vol structure | Frobenius gap on cross-scale vol correlation matrix | Per-path |
| M6 | Conditional Heavy Tails   | GPD shape parameter ξ gap across low/high vol regimes | Regime-stratified |

---

## Results


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
