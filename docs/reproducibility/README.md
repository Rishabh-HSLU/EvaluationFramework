# Reproducibility records

Start with the [run #4 paper reference](../../results/run4/README.md).
It contains the comparison tables, replicate records, settings, input hashes
and [paper-number source index](../../results/run4/paper_number_map.csv).

- [Numerical environment](numerical_environment.md): CPU-dispatch launcher and
  evidence from the two tested x86_64 machines.
- [Panel digests](../panel_digests.txt): exact original inputs used by runs #3/#4.
- [Frozen sensitivity settings](../grid_spec_freeze.md): grid and provenance;
  historical sweeps require an input-provenance audit before reporting.
- [Parameter provenance audit](../parameter_provenance_audit.md) and
  [artifact provenance notes](../artefact_provenance_notes.md): historical
  development evidence, not proof that every outstanding issue is resolved.
- [Missingness report](../missingness_report.md): historical clock/support audit.

The run #3 records, seed-sensitivity notebook and research notes were restored
from before d85c2d4 to preserve the experiment history. They remain historical;
run #4 is the current numerical reference. Large market and synthetic Parquet
panels are supplied separately, not committed.
