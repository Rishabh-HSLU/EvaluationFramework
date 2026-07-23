"""Static benchmark configuration and metric construction."""

from __future__ import annotations

from pathlib import Path

from fineval.config import (
    M1_N_GRID,
    M1_TAIL_ALPHA,
    M1_TAIL_LAMBDA,
    M2_LAG_MAX,
    M2_LAG_MIN,
    M4_MIN_OBS,
    M4_SCALES,
    N_REGIME_QUINTILES,
    REGIME_WEIGHTS,
    ROLLING_VOL_MIN_PERIODS,
    ROLLING_VOL_WINDOW,
    TAIL_QUANTILE,
)
from fineval.metrics import (
    AggregationalGaussianity,
    RegimeConditionalTails,
    UnconditionalHeavyTails,
    VolatilityClustering,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CURATED_DIR = ROOT_DIR / "data" / "curated"
RESULTS_DIR = ROOT_DIR / "results"

MIN_OUTER_REPLICATES_FOR_CI = 40
RECOMMENDED_OUTER_REPLICATES = 1000

METRIC_LABELS = {
    "M1": "Unconditional heavy tails",
    "M2": "Volatility clustering",
    "M4": "Aggregational Gaussianity",
    "M6": "Regime-conditional tails",
}


def build_metrics() -> list:
    """Construct all configured benchmark metrics."""
    return [
        UnconditionalHeavyTails(
            name="M1",
            n_grid=M1_N_GRID,
            tail_alpha=M1_TAIL_ALPHA,
            tail_lambda=M1_TAIL_LAMBDA,
        ),
        VolatilityClustering(name="M2", lag_min=M2_LAG_MIN, lag_max=M2_LAG_MAX),
        AggregationalGaussianity(name="M4", scales=M4_SCALES, min_obs=M4_MIN_OBS),
        RegimeConditionalTails(
            name="M6",
            window=ROLLING_VOL_WINDOW,
            min_periods=ROLLING_VOL_MIN_PERIODS,
            n_regimes=N_REGIME_QUINTILES,
            tail_quantile=TAIL_QUANTILE,
            regime_weights=REGIME_WEIGHTS,
        ),
    ]
