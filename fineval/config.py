"""
The seed for reproducibility and other constants necessary
for the framework are defined here
"""

import math

import numpy as np

# Reproducibility
SEED = 42

# Bootstrap hyper-parameters (B, tickers per draw) live as defaults on
# MatchedTickerBootstrap and the run_benchmark CLI; the implemented
# scheme is a cross-sectional ticker bootstrap with no temporal blocks.

# Market clock
TRADING_MINUTES = 390  # minutes per NYSE session (09:30–16:00)

# FFF deseasonalization
NUM_HARMONICS = 4  # BIC-selected; Andersen & Bollerslev (1997)

# Data curation
MARKET_CALENDAR = "XNYS"  # NYSE; pandas_market_calendars identifier
BAR_FREQUENCY = "1min"  # intraday bar frequency
COVERAGE_FLOOR = 0.70  # minimum joint coverage required to retain a ticker


# Stylized fact scales

## M1: Marginal distribution (Unconditional heavy tails)
M1_N_GRID = 5001  # resolution of the empirical quantile grid
M1_TAIL_ALPHA = 0.3  # tail-emphasis sharpness; controls how fast w(u) rises near u→0,1
M1_TAIL_LAMBDA = 1.0  # tail-emphasis magnitude; 0 recovers plain W1

## M2: Nonlinear temporal dependence (Volatility clustering)
M2_LAG_MIN = 60  # shortest lag (in minutes) included in the ACF gap
M2_LAG_MAX = 390  # longest lag (in minutes) included in the ACF gap (1 trading day)

## M4: Aggregational Gaussianity
M4_SCALES = [1, 5, 15, 30]
M4_MIN_OBS = 100  # minimum pooled observations per scale for reliable kurtosis

## M6: Regime-conditional tail
ROLLING_VOL_WINDOW = 60  # causal rolling-std window (minutes)
ROLLING_VOL_MIN_FRAC = 1 / 2  # min valid obs as fraction of window; empirically validated
ROLLING_VOL_MIN_PERIODS = math.ceil(ROLLING_VOL_WINDOW * ROLLING_VOL_MIN_FRAC)  # = 30
N_REGIME_QUINTILES = 5  # number of volatility regime bins
TAIL_QUANTILE = 0.05  # proportion of |returns| treated as extreme within each regime cell
_RAW_REGIME_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 3.0])  # Q0(calm)→Q4(turbulent)
REGIME_WEIGHTS = _RAW_REGIME_WEIGHTS / _RAW_REGIME_WEIGHTS.sum()  # normalized to sum to 1
