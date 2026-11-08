"""METR-LA acquisition and reconstruction.

The Hugging Face release ``witgaw/METR-LA`` stores *pre-windowed* samples: one
row per (sensor ``node_id``, reference time ``t0_timestamp``) carrying 12 input
steps (``x_t-11`` .. ``x_t+0``) and 12 target steps (``y_t+1`` .. ``y_t+12``).
Every speed reading therefore appears up to 24 times across rows.

FlowCast works from a single canonical ``(timestamp, sensor)`` speed matrix
reconstructed from those windows. Reconstruction doubles as an integrity check:
every redundant copy of a reading must agree exactly, or we refuse the data.

Conventions inherited from the original DCRNN data (verified in Phase 1):

* ``d0`` is speed in **mph**; ``0.0`` encodes a missing reading.
* ``d1`` is time of day as a fraction of 24h (``minutes / 1440``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

HF_REPO = "witgaw/METR-LA"
SPLITS = ("train", "validation", "test")
FREQ = pd.Timedelta(minutes=5)
INPUT_OFFSETS = tuple(range(-11, 1))  # x_t-11 .. x_t+0
TARGET_OFFSETS = tuple(range(1, 13))  # y_t+1 .. y_t+12
MISSING_VALUE = 0.0


def window_column(offset: int, dim: int) -> str:
    """Column name holding feature ``dim`` at ``offset`` steps from t0."""
    prefix = "x" if offset <= 0 else "y"
    return f"{prefix}_t{offset:+d}_d{dim}"


EXPECTED_COLUMNS = (
    "node_id",
    "t0_timestamp",
    *(window_column(k, d) for k in INPUT_OFFSETS + TARGET_OFFSETS for d in (0, 1)),
)


def load_windows():
    """Load the windowed dataset (reuses the local HF cache when present)."""
    from datasets import load_dataset

    return load_dataset(HF_REPO)


@dataclass(frozen=True)
class SplitLayout:
    """Shape facts about one windowed split, established by :func:`check_layout`."""

    t0: pd.DatetimeIndex
    n_sensors: int


def check_layout(table: pa.Table) -> SplitLayout:
    """Verify the row layout that reconstruction relies on.

    Rows must be ordered by (t0, node_id) with node ids ``0..N-1`` repeated in
    every block, and t0 must advance in exact 5-minute steps (which also rules
    out duplicate rows and gaps).
    """
    missing = set(EXPECTED_COLUMNS) - set(table.column_names)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    node = table.column("node_id").to_numpy()
    n_sensors = int(node.max()) + 1
    if len(node) % n_sensors:
        raise ValueError(f"{len(node)} rows is not a multiple of {n_sensors} sensors")
    if not (node.reshape(-1, n_sensors) == np.arange(n_sensors)).all():
        raise ValueError("rows are not ordered by (t0, node_id) with every sensor present")

    ts = _parse_timestamps(table.column("t0_timestamp"))
    blocks = ts.reshape(-1, n_sensors)
    if not (blocks == blocks[:, :1]).all():
        raise ValueError("t0 differs within a sensor block")
    t0 = pd.DatetimeIndex(blocks[:, 0])
    if len(t0) > 1 and not (np.diff(t0.asi8) == FREQ.value).all():
        raise ValueError("t0 is not a gap-free, duplicate-free 5-minute sequence")
    return SplitLayout(t0=t0, n_sensors=n_sensors)


def _parse_timestamps(column: pa.ChunkedArray) -> np.ndarray:
    parsed = pc.strptime(column, format="%Y-%m-%dT%H:%M:%S", unit="ns")
    return parsed.to_numpy().astype("datetime64[ns]")


@dataclass(frozen=True)
class SpeedSeries:
    """Canonical speed matrix reconstructed from all splits."""

    timestamps: pd.DatetimeIndex
    speed: np.ndarray  # (n_timestamps, n_sensors), mph, 0.0 = missing
    anchors: dict[str, tuple[pd.Timestamp, pd.Timestamp]]  # split -> first/last t0


def reconstruct_series(tables: dict[str, pa.Table]) -> SpeedSeries:
    """Rebuild the ``(timestamp, sensor)`` speed matrix from windowed splits.

    Raises ``ValueError`` if splits are not chronologically contiguous, if any
    two copies of the same reading disagree, if any time-of-day feature does
    not match its timestamp, or if any timestamp is left uncovered.
    """
    layouts = {name: check_layout(table) for name, table in tables.items()}
    ordered = sorted(layouts, key=lambda n: layouts[n].t0[0])
    n_sensors = {lay.n_sensors for lay in layouts.values()}
    if len(n_sensors) != 1:
        raise ValueError(f"splits disagree on sensor count: {n_sensors}")
    (n_sensors,) = n_sensors
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        if layouts[nxt].t0[0] - layouts[prev].t0[-1] != FREQ:
            raise ValueError(f"splits {prev!r} and {nxt!r} are not contiguous")

    start = layouts[ordered[0]].t0[0] + INPUT_OFFSETS[0] * FREQ
    end = layouts[ordered[-1]].t0[-1] + TARGET_OFFSETS[-1] * FREQ
    timestamps = pd.date_range(start, end, freq=FREQ)
    speed = np.full((len(timestamps), n_sensors), np.nan)

    for name in ordered:
        table, t0 = tables[name], layouts[name].t0
        base = timestamps.get_loc(t0[0])
        rows = base + np.arange(len(t0))
        for k in INPUT_OFFSETS + TARGET_OFFSETS:
            _check_time_of_day(table, k, t0, name)
            values = table.column(window_column(k, 0)).to_numpy().reshape(-1, n_sensors)
            target = speed[rows + k]
            seen = ~np.isnan(target)
            if (target[seen] != values[seen]).any():
                raise ValueError(f"{name}: conflicting copies of readings at offset {k:+d}")
            speed[rows + k] = values

    if np.isnan(speed).any():
        raise ValueError("some timestamps were not covered by any window")
    anchors = {name: (layouts[name].t0[0], layouts[name].t0[-1]) for name in ordered}
    return SpeedSeries(timestamps=timestamps, speed=speed, anchors=anchors)


def _check_time_of_day(table: pa.Table, k: int, t0: pd.DatetimeIndex, name: str) -> None:
    tod = table.column(window_column(k, 1)).to_numpy().reshape(len(t0), -1)
    ts = t0 + k * FREQ
    expected = ((ts.hour * 60 + ts.minute) / 1440).to_numpy()
    if not np.allclose(tod, expected[:, None], atol=1e-9):
        raise ValueError(f"{name}: time-of-day feature mismatch at offset {k:+d}")


@dataclass(frozen=True)
class SensorGraph:
    """Sensor metadata and the published adjacency matrix (row order = node_id)."""

    sensors: pd.DataFrame  # node_id, sensor_id, latitude, longitude
    adjacency: np.ndarray  # (n, n) float32, directed
    adjacency_meta: dict
    distances: pd.DataFrame  # from, to, cost (metres); covers more than the 207 sensors


def load_sensor_graph() -> SensorGraph:
    """Download (or reuse cached) sensor graph files.

    ``node_id`` i is mapped to ``adj_mx_mapping.json['sensor_ids'][i]``, the
    DCRNN graph ordering. That mapping is not stated in the windowed files
    themselves; Phase 1 validates it empirically (see ``validation.py``).
    """
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> str:
        return hf_hub_download(HF_REPO, f"sensor_graph/{name}", repo_type="dataset")

    with open(fetch("adj_mx_mapping.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    locations = pd.read_csv(fetch("sensor_locations.csv"), dtype={"sensor_id": str})
    if locations["sensor_id"].tolist() != meta["sensor_ids"]:
        raise ValueError("sensor_locations.csv order differs from adjacency sensor order")
    sensors = locations.reset_index(names="node_id")
    adjacency = np.load(fetch("adj_mx.npy"))
    distances = pd.read_csv(fetch("distances.csv"))
    return SensorGraph(sensors, adjacency, meta, distances)
