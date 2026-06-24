# EvaluationFramework

A six-facets (termed facets in the paper, TODO harmonize) fidelity benchmark for synthetic 1-minute financial return generators. Each bucket measures one orthogonal stylized fact; scores are normalized against a real-vs-real baseline so results are interpretable without any reference model.

```
real eval corpus  ──┐
                    ├──►  six gap functions  ──►  matched-N resampling  ──►  s_b ∈ [0, 1]
synthetic corpus  ──┘                                                      (1 = indistinguishable)
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
uv run python bench.py
```

The repo ships with the built eval corpus (`data/output_data/`) and three example synthetic outputs. `bench.py` runs out of the box and prints the full table.

---

## Development setup

This project uses `uv` for dependency management and `ruff` for formatting, linting, and import sorting.

```bash
uv sync --dev
uv run pre-commit install
uv run pre-commit run --all-files
```

After installation, the pre-commit hooks run automatically before each commit. To run the same checks manually:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest
```

## Results

Benchmark run on the canonical corpus: 60 eval tickers, NASDAQ 1-minute bars, Sep 2019 – Mar 2020 (1,187 windows of 2,520 returns each). N = 200 paths, 200 resamples, paired-bootstrap CIs (B = 2,000).

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

## Repository layout

```
EvaluationFramework/
├── bench.py                        # entry point — runs all six buckets, prints table
│
├── evaluation_framework/           # scoring package
│   ├── bucket.py                   # Bucket ABC — input contract, validate, compute_gap
│   ├── buckets/                    # B1..B6 implementations
│   │   ├── b1_marginal.py
│   │   ├── b2_nonlinear_temporal.py
│   │   ├── b3_leverage_effect.py
│   │   ├── b4_kurtosis.py
│   │   ├── b5_cfvc.py
│   │   └── b6_tail_regime.py
│   ├── fast_gaps.py                # FFT-batched B2/B3 (bit-identical to bucket loops)
│   ├── protocol.py                 # matched-N resampling + paired-bootstrap CI
│   ├── samplers.py                 # PooledSampler / WithinTicker / WithinRegime
│   ├── canonical.py                # load + validate corpus against manifest
│   ├── io.py                       # corpus loaders
│   └── paths.py                    # local data paths + EVALFRAMEWORK_DATA_DIR override
│
├── data_pipeline/                  # data preparation package
│   ├── fetch_alpaca.py             # download the canonical 948-ticker raw corpus
│   ├── data_prep.py                # raw CSVs → canonical windowed corpus (Steps 1–10)
│   ├── canonical.py                # corpus I/O contract (manifest, subsample, std-align)
│   ├── transform_ail.py            # AIL parquet → benchmark .npy
│   └── regen_eval_labels.py        # rebuild ticker / regime label arrays
│
├── data/
│   └── output_data/                # eval corpus + synthetic outputs + manifest
│       ├── eval_deseasonalized.npy     (1187, 2520, 1) float32 — the benchmark reference
│       ├── eval_ticker_labels.npy      (1187,) str
│       ├── eval_regime_labels.npy      (1187,) int8  — 0 = pre-crash, 1 = crash
│       ├── fff_pattern.npy             (390,) fitted intraday seasonality curve
│       ├── benchmark_manifest.json     T=2520, ref mean/std, N=200, seed=42
│       ├── ail_synthetic.npy           (200, 2520, 1)
│       ├── garch_synthetic.npy         (200, 2520, 1)
│       └── sfagan_synthetic.npy        (200, 2520, 1)
│
├── examples/
│   └── data_preparation_example.ipynb
│
└── pyproject.toml
```

---
TODO: Yet to add the pipeline of all the synthetic generators results feeding into the
data pipeline.
## Adding your own generator

Drop a `(N, 2520)` or `(N, 2520, 1)` float array into `data/output_data/<name>_synthetic.npy`. Pass it through `save_benchmark_corpus` first to subsample to N=200 and align volatility to the real corpus:

```python
import numpy as np
from data_pipeline import save_benchmark_corpus
from evaluation_framework.paths import output_dir

raw = np.load("my_generator_output.npy")        # (N, T) or (N, T, 1)
save_benchmark_corpus(raw, output_dir() / "mygen_synthetic.npy", output_dir())
```

Then register it in `evaluation_framework/paths.py`:

```python
def generator_paths() -> dict[str, Path]:
    out = output_dir()
    return {
        "AIL":   out / "ail_synthetic.npy",
        "GARCH": out / "garch_synthetic.npy",
        ...
        "MyGen": out / "mygen_synthetic.npy",   # add this line
    }
```

`bench.py` picks it up automatically on the next run.

---

## The six buckets

| ID | Stylized fact | Statistic | Aggregation |
|----|--------------|-----------|-------------|
| B1 | Marginal distribution, 5 % tails | Tail-weighted Wasserstein-1 | Pooled |
| B2 | Volatility clustering | Mean ACF gap on \|r\|, lags 60–390 | Per-path, FFT |
| B3 | Leverage effect | Cross-correlation gap corr(r_t, \|r_{t+k}\|), lags 1–390 | Per-path, FFT |
| B4 | Aggregational kurtosis | Uniform-weighted L-kurtosis gap across horizons {1, 5, 30, 60, 390} min | Per-path then pooled |
| B5 | Multi-scale vol structure | Frobenius gap on cross-scale vol correlation matrix | Per-path |
| B6 | Tail index by vol regime | GPD shape parameter ξ gap across low/high vol regimes | Regime-stratified |

B2 and B3 use FFT-vectorised fast paths in `fast_gaps.py` that are verified bit-identical to the reference bucket loops at every `bench.py` startup.

B4 uses **L-kurtosis** (τ₄ = λ₄/λ₂) rather than moment-based excess kurtosis. Financial returns during crisis regimes have tail indices ξ ≈ 0.3–0.5, meaning the eighth moment does not exist. Standard kurtosis estimators produce CV > 0.8 at N=200 on such data, rendering the metric uninformative. L-kurtosis requires only E[|X|] < ∞ and is stable for all financial return distributions.

---

## Methodology

### Matched-N resampling

For every (bucket, generator) cell:

1. Draw `(idx_a, idx_b)` of size N=200 from the real corpus via `PooledSampler`.
2. `g_rr = gap(real[idx_a], real[idx_b])` — real vs real.
3. Draw `idx_s` of size N=200 from the synthetic corpus.
4. `g_sr = gap(real[idx_a], syn[idx_s])` — real vs synthetic, **reusing `idx_a`**.
5. Repeat 200 times; aggregate means.

Reusing `idx_a` in step 4 induces positive correlation between `g_rr[i]` and `g_sr[i]` that the paired bootstrap exploits.

### Paired-bootstrap CI

```python
for b in range(2000):
    idx  = rng.integers(0, n_resamples, n_resamples)   # same indices into both arrays
    rr_b = g_rr[idx].mean()
    sr_b = g_sr[idx].mean()
    boot[b] = rr_b / (rr_b + sr_b)
lo, hi = np.percentile(boot, [2.5, 97.5])
```

Sharing the index per iteration preserves the per-i correlation from construction, producing CIs that are 5–12 % narrower than the unpaired form on buckets where corr(g_rr, g_sr) > 0.

### Corpus construction (Steps 1–10)

The canonical pipeline in `data_pipeline.data_prep` turns raw 1-minute CSVs into the windowed eval corpus:

| Step | Action                                                                                                                          |
|------|---------------------------------------------------------------------------------------------------------------------------------|
| 1 | NY regular-session filter: minute-of-day 570–959 (9:30 am – 3:59 pm) (To remove any confusion caused by day light saving time.) |
| 2 | Sort by timestamp within ticker                                                                                                 |
| 3 | Flag within-day gaps where consecutive bars are > 1 minute apart                                                                |
| 4 | Drop bars immediately after a gap (return would span multiple minutes)                                                          |
| 5 | Drop the first bar of each session day (overnight jump)                                                                         |
| 6 | log_return = log(close_t / close_{t-1})                                                                                         |
| 7 | Liquidity tiers: `train_only` (≥ 280 bars/day median), `eval_eligible` (≥ 350)                                                  |
| 8 | Random 20 % of `eval_eligible` → eval set (seed 42); rest + `train_only` → train                                                |
| 9 | Fit pooled FFF intraday seasonality on training returns; r̃_t = r_t / s(τ) applied to both                                      |
| 10 | Per-ticker z-score on training tickers only; eval set stays deseasonalised                                                      |

Eval windows are tagged with a regime label: 0 = before 2020-02-19 (pre-crash), 1 = on or after (the S&P 500 peak).

---

## Reproducing the real data corpus

The repo ships with a pre-built eval corpus so `bench.py` runs immediately. To rebuild it from scratch:

### 1 — Download raw data from Alpaca

```bash
pip install "alpaca-py>=0.13"

python -m data_pipeline.fetch_alpaca \
    --api-key  <YOUR_KEY> \
    --secret   <YOUR_SECRET> \
    --out-dir  data/raw_intraday
```

This downloads 948 tickers of 1-minute bars (Sep 2019 – Mar 2020, split-adjusted, SIP feed) to `data/raw_intraday/`. A checkpoint log (`download_log.csv`) lets interrupted runs resume. Expect 30–60 minutes on a standard Alpaca connection. Sign up for free credentials at [alpaca.markets](https://alpaca.markets).

### 2 — Build the eval corpus

```bash
python -m data_pipeline.data_prep data/raw_intraday data/output_data
```

Applies Steps 1–10 and writes `eval_deseasonalized.npy`, `benchmark_manifest.json`, and companion label arrays into `data/output_data/`.

---

## Citation

If you use this benchmark, please cite the accompanying paper (forthcoming).
