# Supervisor's independent panel and point-estimate check

Status, 17 October 2026: two run-labelled panel bundles have been prepared
and verified locally. Delivery and the supervisor's independent rerun are
pending. Local archive checks do not count as an independent reproduction.

The user confirmed that the supervisor's local real panel came from an older
14 July curation build. GBM and MSV generated from that input consequently
differed from the reference panels. This is separate from the CPU-dispatch
roundoff issue observed during our own two-machine reproduction exercise.

Runs #3 and #4 used exactly the same five panel files, including the original
SBBTS panel. All 16 metric point scores, mean gaps, coverage counts and four
aggregate point estimates match between those runs. Run #3 has O=40/M=0
uncertainty results; run #4 has O=200/M=20. Neither is run #5.

The separately delivered ZIPs each contain the five Parquet files, a panel
SHA-256 manifest, curation metadata, run-specific reference CSVs and a compact
reference checksum manifest, instructions, and the standalone comparison tool.
They are excluded from Git because they contain large market-data panels.
The ZIP names are `Run_3_Curated_Panels.zip` and `Run_4_Curated_Panels.zip`.
Archive checksums and recipient access must accompany the actual delivery.

After checking the supplied panel hashes, use the repository environment and
run the independent point calculation with the established numerical launcher:

```bash
uv run --frozen python -m scripts.reproduce benchmark -- \
    --n-resamples 100 --tickers-per-draw 200 --seed 42 \
    --n-outer-resamples 0 --n-mc-repeats 0 --n-jobs 4
```

Compare the NEW local run number N against the reference bundled with each ZIP:

```bash
uv run --frozen python /ABS/BUNDLE/tools/check_reproduction.py \
    --reference-dir /ABS/BUNDLE/reference \
    --manifest results/runs_manifest.csv --run-number N --point-only
```

Alternatively use the repository checker against its run #4 reference:

```bash
uv run --frozen python scripts/check_reproduction.py --run-number N --point-only
```

Point-only mode checks primary specification, B, ticker sample size and seed,
then compares 16 metric scores, mean gaps, valid-draw counts and four aggregate
estimates/coverage counts. It does not compare outer-bootstrap or seed-repeat
summaries. Floating comparison defaults to rtol=1e-10, atol=1e-12; counts must
match exactly. Use `--rtol 0 --atol 0` for exact numeric equality. Panel file
identities always require exact SHA-256 agreement.

One independent point run can be compared against both references because the
inputs and point results are identical. Preserve its output CSVs, manifest row,
comparison JSON, environment and actual source revision. Report independent
fixed-panel numerical reproduction only after that evidence is returned; it
is separate from independent curation, generator retraining and generalisation
to unseen periods.
