"""
Build the curated dataset files from raw sources.

Run this once before any preprocessing or evaluation. Output goes to
data/curated/.
"""

from fineval.data import AILSyntheticLoader, CurationPipeline, RealDataLoader

REAL_DIR = "../data/raw_intraday"
AIL_PATH = "../data/ail_synthetic_data/dataset_US_1-10B_2019-09-2020-03.parquet"
OUTPUT = "../data/curated"

real = RealDataLoader(directory=REAL_DIR).load()
ail = AILSyntheticLoader(parquet_path=AIL_PATH).load()

pipeline = CurationPipeline(
    real_dataset=real,
    synthetic_datasets=[ail],
    start_date="2019-09-03",
    end_date="2020-03-20",
    output_dir=OUTPUT,
    source_paths={"Real": REAL_DIR, "AIL": AIL_PATH},
)
pipeline.run()
