"""Access to the processed METR-LA artifacts written by ``scripts/prepare_metr_la.py``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flowcast import paths
from flowcast.data.metr_la import MISSING_VALUE, SPLITS


def load_speed(split: str | None = None) -> pd.DataFrame:
    """Speed matrix (index = timestamp, columns = sensor_id) with missing readings as NaN.

    ``split`` restricts rows to that split's *owned period* (each timestamp
    belongs to exactly one split); ``None`` returns the full timeline.
    """
    df = pd.read_parquet(paths.SPEED_PATH).set_index("timestamp")
    df = df.mask(df == MISSING_VALUE)
    if split is None:
        return df
    start, end = owned_period(split)
    return df.loc[start:end]


def owned_period(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    splits = json.loads(paths.SPLITS_PATH.read_text(encoding="utf-8"))
    start, end = splits[split]["owned_period"]
    return pd.Timestamp(start), pd.Timestamp(end)


def load_sensors() -> pd.DataFrame:
    """node_id, sensor_id, latitude, longitude, missing_pct (row order = node_id)."""
    return pd.read_parquet(paths.SENSORS_PATH)


def load_adjacency() -> np.ndarray:
    from flowcast.data.metr_la import load_sensor_graph

    return load_sensor_graph().adjacency
