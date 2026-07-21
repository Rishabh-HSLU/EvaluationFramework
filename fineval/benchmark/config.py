"""Static benchmark configuration and metric construction."""

from __future__ import annotations

from pathlib import Path

from fineval.config import (
    M1_N_GRID,
    M1_TAIL_ALPHA,
    M1_TAIL_LAMBDA,
    M2_LAG_MAX,
    M2_LAG_MIN,
    M3_MIN_OBS,
    M3_SCALES,
    M3_SUPPORT_ANCHOR,
    N_REGIME_QUINTILES,
    REGIME_WEIGHTS,
    ROLLING_VOL_MIN_PERIODS,
    ROLLING_VOL_WINDOW,
    TAIL_FRACTION,
)
from fineval.metrics import (
    AggregationalGaussianity,
    RegimeConditionalTails,
    TailWeightedMarginal,
    VolatilityClustering,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CURATED_DIR = ROOT_DIR / "data" / "curated"
RESULTS_DIR = ROOT_DIR / "results"

MIN_OUTER_REPLICATES_FOR_CI = 40
RECOMMENDED_OUTER_REPLICATES = 1000

METRIC_LABELS = {
    "M1": "Tail-weighted marginal distribution",
    "M2": "Volatility clustering",
    "M3": "Aggregational Gaussianity",
    "M4": "Regime-conditional tails",
}


def build_metrics() -> list:
    """Construct all configured benchmark metrics."""
    return [
        TailWeightedMarginal(
            name="M1",
            n_grid=M1_N_GRID,
            tail_alpha=M1_TAIL_ALPHA,
            tail_lambda=M1_TAIL_LAMBDA,
        ),
        VolatilityClustering(name="M2", lag_min=M2_LAG_MIN, lag_max=M2_LAG_MAX),
        AggregationalGaussianity(
            name="M3", scales=M3_SCALES, min_obs=M3_MIN_OBS, support_anchor=M3_SUPPORT_ANCHOR
        ),
        RegimeConditionalTails(
            name="M4",
            window=ROLLING_VOL_WINDOW,
            min_periods=ROLLING_VOL_MIN_PERIODS,
            n_regimes=N_REGIME_QUINTILES,
            tail_fraction=TAIL_FRACTION,
            regime_weights=REGIME_WEIGHTS,
        ),
    ]
