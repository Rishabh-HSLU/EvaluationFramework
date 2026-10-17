# Run #4 diagnostic feature figures

These three figures describe the full 600-ticker panels over the evaluation
window. They use the exact run #4 inputs and conditional preprocessing.
They are in-sample empirical fidelity diagnostics, not matched-bootstrap
scores, confidence intervals, or out-of-sample tests.

From the repository root, after installing the README environment and
placing the five curated Parquet panels in `data/curated/`:

```bash
python -m scripts.reproduce profiles
```

To redraw the figures using only the saved numerical source data:

```bash
python -m scripts.reproduce profiles -- --plot-only
```

- `m2_absolute_return_acf.pdf` / `.png`: cross-sectional mean absolute-return
  ACF; full lags and a separate view of the M2 comparison window, 60–388 minutes.
  Pairs never cross session boundaries. The full-sample variance denominator
  and long-lag shrinkage follow the benchmark estimator.
- `m3_kurtosis_aggregation.pdf` / `.png`: pooled Pearson kurtosis and its
  normalized curve K(h)/K(1), the feature actually compared by M3. Horizons
  are 1, 5, 15, 30 minutes on identical closing 360-return support; the two
  early-close sessions are excluded. Missing blocks are not imputed.
- `m4_xi_by_regime.pdf` / `.png`: GPD shape xi for panel-specific lagged
  volatility quintiles, calmest to most turbulent. Regimes and 95th-percentile
  absolute-return thresholds are fitted separately within each panel.
  Equal regime labels therefore mean relative volatility ranks, not equal
  absolute volatility thresholds. Lines join discrete quintiles only.

`m2_acf.csv`, `m3_kurtosis.csv`, and `m4_regime_xi.csv` contain every plotted
value. M3 also records valid aggregate counts; M4 records volatility bin
edges, tail thresholds, observations and exceedance counts. Missing or
unsupported feature estimates cause generation to fail rather than being
silently dropped. No uncertainty bands are inferred from a single curve.

Extraction runs one panel at a time, with a 6-GiB process-address-space
limit on Linux. M4 uses the benchmark rolling-volatility and GPD-fit
routines with array pairing to avoid constructing a full-panel MultiIndex;
its numerical equivalence to the canonical feature extractor is tested.
No panels are sampled, no models are refitted, and no random draws are used.

`provenance.json` records panel/source digests, versions, parameters and
preprocessing decisions. `checksums.json` identifies the saved outputs.
PDFs are the submission artwork: vector graphics with embedded fonts.
PNGs are 300-dpi review previews, not the journal submission line artwork. Replotting from CSV does not
revalidate the original Parquet files. Full regeneration checks all five
panel identities against run #4.

Separate publication captions are in [captions.md](captions.md).

## Literature and implementation precedents

Sandve et al. (2013), [Ten Simple Rules for Reproducible Computational
Research](https://doi.org/10.1371/journal.pcbi.1003285), recommend executable
workflows, version-controlled scripts, recorded software/input identities,
and preserved intermediate results. This motivates separating curve
extraction, saved CSVs and rendering here.

Li et al., [MarS (ICLR 2025)](https://arxiv.org/abs/2409.07486), provide a
[public stylized-facts report script](https://github.com/microsoft/MarS/blob/main/market_simulation/examples/report_stylized_facts.py)
that saves numerical CSVs and plots absolute-return dependence and kurtosis
across horizons. It is a presentation/repository precedent; its estimators,
data cleaning and uncertainty conventions are not substituted for ours.

The [Journal of Finance and Data Science author guide](https://www.keaipublishing.com/en/journals/the-journal-of-finance-and-data-science/guide-for-authors/)
is the journal-specific source for artwork and research-data requirements.
The xi-by-regime figure displays this paper's M4 feature definition; we do
not claim that the exact five-quintile design is a universal literature standard.

The journal requests separate artwork files, embedded fonts, consistent lettering,
and colour-vision-accessible graphics; its guide accepts vector PDF/EPS.
The PDF exports follow this route rather than treating 300-dpi preview PNGs
as submission-quality line drawings.

Cont (2001), [Empirical properties of asset returns](https://doi.org/10.1080/713665670),
provides the stylized-fact background for volatility persistence, aggregational
Gaussianity and conditional heavy tails. It does not specify our exact M4
regime partition, which remains the paper’s own operational definition.
