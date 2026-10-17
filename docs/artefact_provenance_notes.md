# Artefact provenance notes

Short notes on repository artefacts whose provenance is not obvious from the
file itself. Each entry says what the artefact is, what code state produced it,
and what it may and may not be used for.

## `scripts/notebooks/seed_sensitivity.ipynb` — a June-definition artefact

**Do not cite its numbers as current results.**

The notebook records an eight-seed sensitivity audit (seeds 42, 1, …, 7) of the
matched-ticker bootstrap at B = 100 and 200 tickers per draw. It was produced in
July 2026 and its stored outputs come from the metric definitions of that time.
Specifically:

- It uses the **old metric numbering and labels** — `M1`, `M2`, `M4`, `M6`.
  Since `e60fe80` (2026-07-21) the code's metrics are `M1`–`M4`, with old `M4`
  (aggregational Gaussianity) renumbered to **M3** and old `M6`
  (regime-conditional tails) renumbered to **M4**. There is no `M5` or `M6` in
  the current code.
- Its outputs **predate the 2026-07-21 metric revisions**: M1 tail-weight
  normalization (`c318e24`), M2 ACF normalization (`7999cd3`), M3 common
  intraday support (`77e0748`) and normalization (`f025055`), M4 causal regime
  assignment (`734a818`), the FFF least-squares refit (`8afc4a4`) and the
  exclusion of the structurally unavailable 09:31 return (`065fc00`).
- They also predate `ff78592` (2026-08-31), which made M1 standardize each
  ticker by its own sample standard deviation before pooling. That change alone
  moves M1 substantially and inverts the GBM/MSV ordering on that metric.
- It covers **three generators** (AIL, GBM, MSV). SBBTS did not exist.

What it is still good for: the qualitative finding it was written to establish —
that across all eight seeds the generator ordering by the aggregate score was
stable, and that M2's paired-bootstrap interval did not reliably bound its own
across-seed spread. That finding is about the *estimator*, not about the metric
values, and it motivated the move to the corpus-level outer bootstrap.

The notebook is deliberately left unmodified. Re-running it under current
definitions would mean an eight-seed sweep (~7.7 core-hours) and would produce a
different artefact; if that is done, it should be added alongside rather than
overwriting this one.

## `results/` — tracked benchmark artefacts

The current paper reference is [run #4](../results/run4/README.md), completed
on 15 October 2026 with O=200 and M=20. It includes summary tables, all outer
and seed replicates, source/input/environment provenance and a paper-number
index. It uses the original SBBTS panel.

[Run #3](../results/run3/README.md) is historical O=40/MC=0 evidence produced
on 16 September. Its four original filenames are preserved inside results/run3/
rather than the root results/ directory. The historical manifest retains the
original CLI output names. These intervals must not replace the run #4 intervals.

results/runs_manifest.csv records the run history. Earlier exploratory outputs
are not all included. The curated Parquet panels remain outside Git; their
exact SHA-256 identities are in [panel_digests.txt](panel_digests.txt).
