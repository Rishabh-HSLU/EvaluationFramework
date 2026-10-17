# Run #4 — current paper reference

Completed 15 October 2026 at 22:08 UTC; runtime 53 hours 30 minutes.
Primary specification, B=100 matched draws, 200 tickers per draw, root seed 42,
O=200 outer replicates, M=20 seeds, four workers and four generators.
The original SBBTS panel is used, exactly as in run #3.

## What to compare

| File | Content |
|---|---|
| metric_summary.csv | All 16 point scores, outer intervals and seed summaries |
| aggregate_summary.csv | Four G_dev scores, intervals and seed summaries |
| outer_metric_replicates.csv | 3,200 rows: 200 replicates x 4 metrics x 4 generators |
| outer_aggregate_replicates.csv | 800 rows: 200 replicates x 4 generators |
| mc_metric_replicates.csv | 320 rows: 20 seeds x 4 metrics x 4 generators |
| mc_aggregate_replicates.csv | 80 rows: 20 seeds x 4 generators |
| paper_number_map.csv | Source cells and display precision for 201 paper/data/configuration values |
| paper_claims.json | Derived interval-containment and aggregate-order claims |
| provenance/ | Configuration, input hashes, dataset facts, environment, source fingerprints and run records |
| checksums.json | SHA-256 identities of the reference evidence files |

Seed repeat zero reuses the seed-42 point estimate. The other 19 repeats use
distinct derived seeds. Outer intervals describe ticker-corpus composition;
seed summaries are numerical stability diagnostics, not confidence intervals.

## Verify the reference without rerunning the benchmark

From the repository root:

~~~bash
uv run --frozen python scripts/check_reproduction.py
~~~

This checks file identities, replicate completeness, recorded score/aggregate
calculations, interval/seed summaries and the paper-number index. It does not
recompute the expensive feature extraction or require the Parquet inputs.

## Reproduce and compare a new run

For an exact table replay, obtain the original five curated panels from the
authors and put them in data/curated/. Verify their identities with:

~~~bash
sha256sum -c docs/panel_digests.txt
~~~

The root README also describes independently rebuilding the inputs. Fresh
provider data or neural-network retraining can produce different panels;
matching seeds alone does not establish that they reproduce this input snapshot.
No public panel download is included in this repository yet.

Run the evaluation:

~~~bash
uv run --frozen python -m scripts.reproduce benchmark -- \
    --n-resamples 100 \
    --tickers-per-draw 200 \
    --seed 42 \
    --n-outer-resamples 200 \
    --n-mc-repeats 20 \
    --n-jobs 4 \
    --replicate-chunk-size 1
~~~

Use the new run number printed at completion in this command (5 is an example):

~~~bash
uv run --frozen python scripts/check_reproduction.py --run-number 5
~~~

The comparison matches identifiers rather than row order and checks all six
result files. It reports differences and exits nonzero on a mismatch.
Default numeric tolerances are rtol=1e-10 and atol=1e-12; counts and seed IDs
must match exactly. Use --rtol 0 --atol 0 for an exact numeric comparison.
CSV file-byte hashes need not match for numeric agreement.

## Provenance and limits

provenance/original_artifacts.json maps the organised filenames back to the
unaltered original output names and hashes. The recorded startup source commit
is e8bc552d18a6d86e4dffeda21e0dd1fe0cf10d9e. Source/environment evidence was
captured eight minutes after launch. Operational checkpoint code was later
added to the working tree during the uninterrupted run; the startup evidence
must not be confused with that later working tree.

The historical original panel-to-SBBTS-checkpoint generation manifest is not
available. This bundle identifies the panel bytes, not a proven historical
training/generation recipe. The newer Lightning-trained panel is different.
The independent full Lightning run #5 was stopped and is not a completed
reproduction of this table.

The original validation report records 59 passed checks. Raw per-draw feature
and gap arrays were not saved; checks recover summaries from recorded mean
gaps and complete replicate outputs. They are not a second execution of
every feature extraction.
