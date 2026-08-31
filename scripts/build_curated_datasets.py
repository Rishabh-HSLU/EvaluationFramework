"""
Build the curated dataset files from raw sources.

Run this once before any preprocessing or evaluation. Output goes to
data/curated/.
"""

from fineval.benchmark.config import CURATED_DIR, ROOT_DIR
from fineval.data import AILSyntheticLoader, CurationPipeline, RealDataLoader

REAL_DIR = ROOT_DIR / "data" / "raw_intraday"
AIL_PATH = ROOT_DIR / "data" / "ail_synthetic_data" / "dataset_US_1-10B_2019-09-2020-03.parquet"
OUTPUT_DIR = CURATED_DIR

real = RealDataLoader(directory=str(REAL_DIR)).load()
ail = AILSyntheticLoader(parquet_path=str(AIL_PATH)).load()

pipeline = CurationPipeline(
    real_dataset=real,
    synthetic_datasets=[ail],
    start_date="2019-09-03",
    end_date="2020-03-20",
    output_dir=str(OUTPUT_DIR),
    source_paths={"Real": str(REAL_DIR), "AIL": str(AIL_PATH)},
)
pipeline.run()
