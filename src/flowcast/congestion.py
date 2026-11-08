"""Shared traffic definitions: free-flow speed and congestion.

Used by the EDA, the feature pipeline, evaluation slices and (later) the API's
congestion indicator, so the definition lives in one place. See docs/eda.md §2.
"""

from __future__ import annotations

import pandas as pd

FREE_FLOW_PERCENTILE = 85  # traffic-engineering convention for a sensor's free-flow speed
CONGESTION_RATIO = 0.6  # congested when speed < 60% of free-flow speed


def free_flow_speed(speed: pd.DataFrame) -> pd.Series:
    """Per-sensor free-flow speed: the 85th percentile of observed speed.

    Always estimate it on the training period only.
    """
    return speed.quantile(FREE_FLOW_PERCENTILE / 100)
