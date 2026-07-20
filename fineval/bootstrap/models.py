"""Small value objects returned by bootstrap analyses."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReplicateAnalysis:
    """Summary and replicate-level outputs from a repeated analysis."""

    metric_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    metric_replicates: pd.DataFrame
    aggregate_replicates: pd.DataFrame
