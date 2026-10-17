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

Only run #3 is tracked: `*_spec-primary_B100_m200_seed42_O40_MC0_J4_20260916-000258-077905.csv`
(benchmark, aggregate, and both outer-replicate files), produced at commit
`0855491` on 2026-09-16 over four generators with O = 40 outer bootstrap
replicates. `results/runs_manifest.csv` is committed exactly as the CLI wrote
it and begins at run #1 (2026-09-13); the earlier manifest was deleted in
`f72d97b` and the recovered June rows are held outside the repository.

The curated panels those numbers were computed on are gitignored; their
SHA-256 digests are recorded in `docs/panel_digests.txt`.
