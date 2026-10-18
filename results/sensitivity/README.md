# Sensitivity reference

Completed 17 October 2026: 19 specifications × five paired seeds (42, 1, 2,
3, 4), B=100 and 200 tickers per draw, using all four generators and the exact
run #4 panels. The specifications are in [the frozen grid](../../docs/grid_spec_freeze.md).

The complete M1, M4 and aggregate orderings are unchanged across the tested
alternatives. Scores still change; [effect_ranges.csv](effect_ranges.csv)
reports their minimum and maximum paired changes. This is exploratory
point-estimate sensitivity, not an outer-bootstrap interval for each alternative.

| Files | Contents |
|---|---|
| `metric_results.csv`, `aggregate_results.csv` | All 1,520 metric and 380 aggregate results, with changes from each same-seed primary. |
| `metric_seed_summary.csv`, `aggregate_seed_summary.csv` | Means, standard deviations and ranges across five seeds for each specification. |
| `effect_ranges.csv`, `paper_number_map.csv` | Paired-change ranges and source cells for the new manuscript numbers. |
| `seed_*/` | Original per-seed result tables, resolved settings, source/input hashes, environment and completion status. |
| `protocol.json`, `validation.json` | Execution settings and validated completeness, gap formulas, coverage and ordering comparisons. |
| `*_sensitivity.pdf`, `*_sensitivity.png` | Vector artwork and PNG previews, rendered from the summary CSVs. |
| `reference_checksums.json` | File identities for this published reference. |

Verify the paper and sensitivity records without loading market panels:

```bash
uv run --frozen python scripts/check_reproduction.py --sensitivity
```

To rerun the computation, follow the commands in
[the frozen-grid instructions](../../docs/grid_spec_freeze.md), using separate
output directories. [Run #4](../run4/README.md) remains the primary benchmark
reference with O=200 and M=20. Its seed-repeat design is distinct from the
five paired root seeds used here.

To redraw the figures from saved CSVs:

```bash
uv run --frozen python -m scripts.finalize_sensitivity \
    --run-dir results/sensitivity --figures-only \
    --figure-output-dir results/sensitivity_reproduction/figures
```

M2 and M3 settings, preprocessing, the volatility-proxy family and generator
fits are fixed. Stable orderings here do not establish robustness to changes
in those choices. The matching five September sweeps have identical numeric
scores; this reference additionally records the previously omitted SBBTS input
identity and a complete validation record.
