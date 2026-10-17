# Numerical environment for reproduction

Use the commands in the root README, which launch computation through
[scripts/reproduce.py](../../scripts/reproduce.py).

The launcher sets NumPy CPU-dispatch options before scientific libraries are
imported. On x86_64 it disables X86_V4, AVX512_ICL and AVX512_SPR, preserving
any previously disabled features. It also sets OpenBLAS, OpenMP and MKL to one
thread per worker, then forwards the arguments and process exit status.

## Evidence

A clean independent Lightning clone regenerated the real and AIL panels with
the original file digests. Initially the regenerated GBM and MSV panels had
identical shapes, indices, columns, missing masks and dtypes, but different
finite prices: maximum absolute differences were 6.82e-13 and 8.49e-9.

The two environments used NumPy 2.4.5 and pandas 3.0.3 with the same reported
NumPy/OpenBLAS build. The random-normal probe and input-real-value fingerprints
matched. NumPy log/exp and derived-return fingerprints differed.

Disabling the three CPU-feature groups together restored all tested
intermediate fingerprints. Full GBM and MSV regeneration then reproduced
the exact original panel SHA-256 values in [panel_digests.txt](../panel_digests.txt).
The responsible individual CPU kernel was not isolated. Values were not
rounded and reference panels were not substituted for generated controls.

## Scope

This setting was demonstrated on the two tested x86_64 environments; it is not
a guarantee for every architecture, library build or Parquet writer version.
CPU settings cannot reconcile panels built from different input snapshots.
An older real-panel curation also changes controls generated from that panel.
Check the curated input identities before comparing benchmark values.
Fresh provider downloads can contain revisions, and neural-network retraining
can differ across devices even with matching seeds.

The cross-machine check reproduced the real, AIL, GBM and MSV panel hashes.
The full independent O=200/M=20 benchmark was stopped before completion, so
the final table has not been independently reproduced by that check.
