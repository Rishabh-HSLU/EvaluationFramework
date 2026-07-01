"""
The seed for reproducibility and other constants necessary
for the framework are defined here
"""

import numpy as np

# Reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)


# Bootstrap hyper-parameters
N_RESAMPLES = 500
BLOCK_SIZE_MINUTES = 3900  # i.e. 10 trading days

# Market clock
TRADING_MINUTES = 390  # minutes per NYSE session (09:30–16:00)

# FFF deseasonalization
NUM_HARMONICS = 4  # BIC-selected; Andersen & Bollerslev (1997)

# Data curation
MARKET_CALENDAR = "XNYS"  # NYSE; pandas_market_calendars identifier
BAR_FREQUENCY = "1min"  # intraday bar frequency
COVERAGE_FLOOR = 0.70  # minimum joint coverage required to retain a ticker

# Stylized fact scales (all must divide TRADING_MINUTES = 390)
## yet to be defined
