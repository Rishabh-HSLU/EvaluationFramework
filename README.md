# EvaluationFramework

Scores synthetic 1-minute return generators against a fixed real eval corpus. Generator-agnostic: same units, same N paths, same gaps — higher similarity wins.

## Contract

All corpora under `SyntheticGenerators/data/output_data/`:

| File | Role |
|------|------|
| `eval_deseasonalized.npy` | Real reference |
| `benchmark_manifest.json` | T=2520, ref mean/std, N=200 |
| `garch_synthetic.npy`, `sfagan_synthetic.npy`, `sbbts_synthetic.npy`, `ail_synthetic.npy` | Generator outputs |

Built by `SyntheticGenerators/data/canonical.save_benchmark_corpus` (deseasonalized, subsampled to 200, volatility-aligned to real).

Set `SYNTHGEN_ROOT` if the clone is not at `~/PycharmProjects/SyntheticGenerators`.

## Scoring

Matched-N protocol (N=200 paths, 200 resamples, seed 42):

`s_b = mean(g_rr) / (mean(g_rr) + mean(g_sr))`

Bootstrap CIs on `s_b`. `bench.py` prints per-bucket table and **rank by composite** (higher = closer to real).

## Buckets

| ID | Measures |
|----|----------|
| B1 | Tail Wasserstein |
| B2 | ACF \|r\| |
| B3 | Leverage |
| B4 | L-kurtosis |
| B5 | Cross-scale vol correlation |
| B6 | Tail index vs vol regime |

## Run

```bash
uv sync
uv run python bench.py
uv run pytest
```
