# Manuscript results

The current paper uses [run #4](run4/README.md) for the primary evaluation
(O=200, M=20) and the [five-seed sensitivity grid](sensitivity/README.md).
All numerical sources and artwork linked below are committed in this repository.

| Manuscript item | Saved numerical results | Artwork / settings |
|---|---|---|
| Table 1: evaluation settings | [Configuration](run4/provenance/configuration.json) | [Run settings](run4/provenance/run_manifest.json) |
| Table 2: scores and 95% intervals | [Metric scores](run4/metric_summary.csv), [aggregate scores](run4/aggregate_summary.csv) | [Underlying outer metric replicates](run4/outer_metric_replicates.csv), [aggregate replicates](run4/outer_aggregate_replicates.csv) |
| Table A.3: metric seed stability | [Means, SDs and ranges](run4/metric_summary.csv) | [All 20 seed repeats](run4/mc_metric_replicates.csv) |
| Table A.4: aggregate seed stability | [Means, SDs and ranges](run4/aggregate_summary.csv) | [All 20 seed repeats](run4/mc_aggregate_replicates.csv) |
| Table B.5: sensitivity ranges | [Paired-change ranges](sensitivity/effect_ranges.csv) | [Complete metric grid](sensitivity/metric_results.csv), [aggregate grid](sensitivity/aggregate_results.csv) |
| Figure 1: absolute-return ACF (M2) | [Plotted values](figures/run4/m2_acf.csv) | [PDF](figures/run4/m2_absolute_return_acf.pdf), [PNG](figures/run4/m2_absolute_return_acf.png) |
| Figure 2: kurtosis aggregation (M3) | [Plotted values](figures/run4/m3_kurtosis.csv) | [PDF](figures/run4/m3_kurtosis_aggregation.pdf), [PNG](figures/run4/m3_kurtosis_aggregation.png) |
| Figure 3: tail shape by regime (M4) | [Plotted values](figures/run4/m4_regime_xi.csv) | [PDF](figures/run4/m4_xi_by_regime.pdf), [PNG](figures/run4/m4_xi_by_regime.png) |
| Figure B.4: M1/M4 sensitivity | [Five-seed summaries](sensitivity/metric_seed_summary.csv) | [PDF](sensitivity/score_sensitivity.pdf), [PNG](sensitivity/score_sensitivity.png) |
| Figure B.5: aggregate sensitivity | [Five-seed summaries](sensitivity/aggregate_seed_summary.csv) | [PDF](sensitivity/aggregate_sensitivity.pdf), [PNG](sensitivity/aggregate_sensitivity.png) |

The [primary number index](run4/paper_number_map.csv) and
[sensitivity number index](sensitivity/paper_number_map.csv) identify source
cells and rounding for the reported values. Dataset counts and training evidence
are in [dataset facts](run4/provenance/dataset_facts.json) and
[SBBTS validation minima](run4/provenance/sbbts_validation_minima.csv).

## Table 2: primary results

Each cell gives the point estimate followed by its 95% outer-bootstrap interval.
Similarity-score parity is 0.5; aggregate deviation parity is 1.

| Metric | InDiGO | GBM | MSV | SBBTS |
|---|---|---|---|---|
| M1 | 0.3436 [0.2892, 0.4292] | 0.0184 [0.0178, 0.0225] | 0.0790 [0.0753, 0.0971] | 0.0245 [0.0237, 0.0300] |
| M2 | 0.2136 [0.1726, 0.3482] | 0.0186 [0.0183, 0.0237] | 0.2925 [0.2400, 0.3664] | 0.0203 [0.0200, 0.0259] |
| M3 | 0.3622 [0.2445, 0.4912] | 0.1025 [0.0702, 0.1433] | 0.1294 [0.0885, 0.1865] | 0.2872 [0.1904, 0.4074] |
| M4 | 0.4818 [0.3922, 0.5053] | 0.0384 [0.0349, 0.0438] | 0.0929 [0.0833, 0.1042] | 0.1245 [0.1114, 0.1396] |
| G_dev | 1.911 [1.522, 2.325] | 28.031 [23.750, 30.219] | 6.561 [5.541, 7.277] | 13.532 [11.220, 14.722] |

## Verify or reproduce

Check the saved results without loading market panels:

```bash
uv run --frozen python scripts/check_reproduction.py --sensitivity
```

Use the [run #4 instructions](run4/README.md) for full or point-only evaluation,
the [frozen-grid instructions](../docs/grid_spec_freeze.md) for sensitivity, and
[figure instructions](figures/run4/README.md) to regenerate the diagnostic plots.

[Run #3](run3/README.md) is historical. The root canonical CSVs can be replaced
by new computations; the fixed run #4 and sensitivity directories identify
the paper's results. Large Parquet inputs are supplied separately, with their
[identities recorded here](../docs/panel_digests.txt).
