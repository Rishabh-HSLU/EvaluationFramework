# EvaluationFramework

Six-bucket fidelity benchmark for synthetic 1-minute financial return generators. Scores each generator against a fixed real corpus of NASDAQ minute bars (Sep 2019 – Mar 2020, 59 tickers) under a matched-N resampling protocol with paired-bootstrap confidence intervals.

```
real eval corpus  ──┐
                    ├──►  six gap functions  ──►  matched-N resampling  ──►  s_b ∈ [0, 1]
synthetic corpus  ──┘                                                      (1.0 = indistinguishable)
```

## Install

```bash
git clone https://github.com/Rishabh-HSLU/EvaluationFramework.git
cd EvaluationFramework
uv sync
```

The repo ships with a built eval corpus (`data/output_data/eval_deseasonalized.npy`, ~24 MB) and three example synthetic outputs (`ail_synthetic.npy`, `garch_synthetic.npy`, `sfagan_synthetic.npy`). `uv run python bench.py` reproduces the published table out of the box.

## Repository layout

```
EvaluationFramework/
├── bench.py                  # entry point — runs the six buckets on every generator
├── evaluation_framework/     # scoring package
│   ├── bucket.py             # Bucket ABC
│   ├── buckets/              # B1..B6 implementations
│   ├── fast_gaps.py          # FFT-batched B2/B3
│   ├── protocol.py           # matched-N resampling + paired bootstrap
│   ├── samplers.py           # Pooled / WithinTicker / WithinRegime samplers
│   ├── canonical.py          # load + validate manifest
│   ├── io.py                 # corpus loaders
│   └── paths.py              # local data paths
├── data_pipeline/            # data preparation package
│   ├── data_prep.py          # raw CSV → canonical windowed corpus
│   ├── canonical.py          # I/O contract (manifest, subsample, std-align)
│   ├── transform_ail.py      # AIL parquet → benchmark .npy
│   └── regen_eval_labels.py  # rebuild ticker / regime labels
├── data/
│   └── output_data/          # eval corpus + synthetic outputs + manifest
├── examples/
│   └── data_preparation_example.ipynb
└── pyproject.toml
```

## Running the benchmark

```bash
uv run python bench.py
```

prints a six-bucket × N-generator table of similarity scores

`s_b = mean(g_rr) / (mean(g_rr) + mean(g_sr))`

with bootstrap-95% CIs and a composite ranking. `s_b = 1` means the synthetic is indistinguishable from real, `s_b = 0.5` is the real-vs-real noise floor, `s_b → 0` means the synthetic is far from real.

## The six buckets

| ID | Measures | Stat |
|----|----------|------|
| B1 | Marginal distribution, 5 % tails | Tail-weighted Wasserstein-1 |
| B2 | Volatility clustering | ACF gap on \|r\|, lags 60–390 |
| B3 | Leverage effect | Corr(r_t, \|r_{t+k}\|) gap, lags 1–390 |
| B4 | Heavy tails by horizon | Scale-weighted excess kurtosis |
| B5 | Multi-scale vol structure | Frobenius gap on cross-scale vol correlation |
| B6 | Tail index conditional on vol | GPD ξ gap across vol regimes |

B2 and B3 ship FFT-vectorised fast paths in `fast_gaps.py`; both are unit-verified bit-identical to the reference loop implementations on a 20-path sample at every `bench.py` startup.

## Methodology

### Matched-N resampling

For every (bucket, generator) cell:

1. Draw `(idx_a, idx_b)` of size `N=200` from the real corpus via `PooledSampler.draw_pair`.
2. `g_rr = gap(real[idx_a], real[idx_b])`.
3. Draw `idx_s` of size `N=200` from the synthetic corpus.
4. `g_sr = gap(real[idx_a], syn[idx_s])`.
5. Repeat 200 times; aggregate means.

Step 4 reuses `idx_a` from step 1 by design — this induces positive correlation between `g_rr[i]` and `g_sr[i]` that the paired bootstrap exploits.

### Paired-bootstrap CI

```python
for b in range(2000):
    idx = rng.integers(0, n_resamples, n_resamples)   # SAME indices into both arrays
    rr_b = g_rr[idx].mean()
    sr_b = g_sr[idx].mean()
    boot[b] = rr_b / (rr_b + sr_b)
lo, hi = np.percentile(boot, [2.5, 97.5])
```

Sharing the index per iteration preserves the per-`i` correlation induced by shared `idx_a` at construction. CIs are 5–12 % narrower than the unpaired form on buckets where `corr(g_rr, g_sr) > 0` and identical in width where the correlation is near zero. Same point estimate either way.

## Bring your own data

Two entry points depending on what you have.

### 1. You have a directory of per-ticker 1-minute CSVs

The full pipeline turns raw CSVs into the canonical windowed corpus.

```bash
uv run python -m data_pipeline.data_prep <raw_csv_dir> <output_dir>
```

Each CSV must have at minimum `timestamp` (UTC) and `close` columns. Outputs:

```
<output_dir>/
├── eval_deseasonalized.npy       (N, 2520, 1) float32
├── eval_ticker_labels.npy        (N,) str
├── eval_regime_labels.npy        (N,) int8     0 = pre-crash, 1 = crash
├── train_normalized.npy          training-corpus z-scored returns
├── train_ticker_labels.npy
├── fff_pattern.npy               (390,)  fitted intraday seasonality
├── benchmark_manifest.json       T, ref mean/std, N=200, seed=42
├── ticker_split.json
├── stats_df.csv                  liquidity stats per ticker
└── norm_stats.json               per-ticker train mean/std
```

Point `bench.py` at this directory by setting `EVALFRAMEWORK_DATA_DIR=<output_dir>` or by moving the files into `data/output_data/`. See `examples/data_preparation_example.ipynb` for a step-by-step walkthrough.

### 2. You have synthetic outputs from your own generator

Drop a `(N_paths, 2520)` or `(N_paths, 2520, 1)` float array into `data/output_data/<name>_synthetic.npy`. For volatility alignment to the real corpus (recommended), pass it through `save_benchmark_corpus`:

```python
import numpy as np
from data_pipeline import save_benchmark_corpus
from evaluation_framework.paths import output_dir

raw  = np.load("my_generator_output.npy")           # (N, T) or (N, T, 1)
save_benchmark_corpus(raw, output_dir() / "mygen_synthetic.npy", output_dir())
```

`save_benchmark_corpus` subsamples to `BENCHMARK_N_PATHS = 200`, zero-means, and rescales the pooled std to match `benchmark_manifest.json`. Add `"mygen": out / "mygen_synthetic.npy"` to `generator_paths()` in `evaluation_framework/paths.py` and `bench.py` will pick it up.

## Data preparation pipeline (Steps 1–10)

The canonical pipeline in `data_pipeline.data_prep` applies ten deterministic steps:

| Step | Action |
|------|--------|
| 1 | NY regular-session filter: minute-of-day 570–959 (9:30–15:59) |
| 2 | Sort by timestamp within ticker |
| 3 | Flag within-day gaps where consecutive bars are >1 minute apart |
| 4 | Drop bars immediately after a gap (the return would span multiple minutes) |
| 5 | Drop the first bar of each session day (overnight jump) |
| 6 | `log_return = log(close_t / close_{t-1})` |
| 7 | Liquidity tiers: `train_only` (≥280 bars/day), `eval_eligible` (≥350) |
| 8 | Random 20 % of `eval_eligible` → eval set (seed 42); rest + `train_only` → train |
| 9 | Fit pooled FFF intraday seasonality on training returns; `r̃_t = r_t / s(τ)` applied to both sets |
| 10 | Per-ticker z-score on training tickers only; eval stays deseasonalised |

Eval windows are tagged with a regime label (0 = before 2020-02-19, 1 = on or after; the S&P 500 peak). See `data_pipeline.CRASH_CUTOFF`.

## Variance decomposition of `g_rr`

`evaluation_framework.samplers` exposes three samplers that decompose the variance in the real-real baseline into corpus-level vs ticker-level vs regime-level components:

```python
from evaluation_framework.samplers import (
    PooledSampler, WithinTickerSampler, WithinRegimeSampler,
)

PooledSampler(n_real)                                          # full corpus
WithinTickerSampler(ticker_labels, max_n_per_half=10)          # one ticker per draw
WithinRegimeSampler(regime_labels, target_regime=0, max_n_per_half=50)
```

Plug any of them into `rr_gaps(...)` or `rr_sr_gaps_paired(...)` to measure CV under that restriction. The decomposition is the basis for honest reporting of the wide marginal CIs on `g_rr` — they reflect cross-ticker / cross-regime heterogeneity in the real corpus rather than estimator instability.

## Citation

If you use this benchmark, cite the accompanying paper (TBD).
