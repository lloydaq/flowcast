"""Data-quality checks on the reconstructed METR-LA series.

All statistics are computed on the canonical ``(timestamp, sensor)`` matrix,
not on the windowed rows, so each reading is counted once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.data.metr_la import FREQ, INPUT_OFFSETS, MISSING_VALUE, TARGET_OFFSETS, SensorGraph

SPEED_LIMIT_MPH = 70.0  # observed sensor ceiling; values above are treated as invalid
LOW_SPEED_MPH = 5.0  # positive but implausibly slow for highway loops; flagged, not removed


def owned_periods(
    timestamps: pd.DatetimeIndex, anchors: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Assign every timestamp to exactly one split.

    A split owns the timestamps of its own t0 anchors. The first split also owns
    the 11-step warm-up before its first t0, and the last split owns the 12
    target steps after its last t0.
    """
    names = list(anchors)
    periods = {}
    for i, name in enumerate(names):
        first, last = anchors[name]
        start = timestamps[0] if i == 0 else first
        end = timestamps[-1] if i == len(names) - 1 else last
        periods[name] = (start, end)
    return periods


def boundary_overlap(
    anchors: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, int]:
    """Count samples per split whose targets reach past that split's owned period.

    The published splits cut on sample index (DCRNN convention), so the last 12
    samples of train/validation predict readings owned by the next split.
    """
    out = {}
    for name, (first, last) in anchors.items():
        t0 = pd.date_range(first, last, freq=FREQ)
        out[name] = int((t0 + TARGET_OFFSETS[-1] * FREQ > periods[name][1]).sum())
    return out


def longest_run(mask: np.ndarray) -> np.ndarray:
    """Length of the longest run of ``True`` in each column of a 2-D mask."""
    best = np.zeros(mask.shape[1], dtype=int)
    current = np.zeros(mask.shape[1], dtype=int)
    for row in mask:
        current = np.where(row, current + 1, 0)
        np.maximum(best, current, out=best)
    return best


def missing_summary(speed: np.ndarray) -> dict:
    """Missing-value statistics; missing means exactly ``MISSING_VALUE``."""
    missing = speed == MISSING_VALUE
    per_sensor = missing.mean(axis=0) * 100
    per_time = missing.mean(axis=1)
    runs = longest_run(missing)
    return {
        "missing_pct": round(float(missing.mean() * 100), 3),
        "per_sensor_pct": {
            "min": round(float(per_sensor.min()), 3),
            "median": round(float(np.median(per_sensor)), 3),
            "max": round(float(per_sensor.max()), 3),
            "sensors_over_20pct": int((per_sensor > 20).sum()),
        },
        "timestamps_all_sensors_missing": int((per_time == 1).sum()),
        "timestamps_over_half_missing": int((per_time > 0.5).sum()),
        "longest_missing_run_steps": {
            "median_sensor": int(np.median(runs)),
            "max_sensor": int(runs.max()),
            "max_sensor_node_id": int(runs.argmax()),
        },
    }


def value_checks(speed: np.ndarray) -> dict:
    """Counts of invalid or suspicious readings."""
    observed = speed[speed != MISSING_VALUE]
    return {
        "nan": int(np.isnan(speed).sum()),
        "negative": int((speed < 0).sum()),
        "above_limit": int((speed > SPEED_LIMIT_MPH).sum()),
        "positive_below_low_speed": int(((speed > 0) & (speed < LOW_SPEED_MPH)).sum()),
        "observed_min": float(observed.min()),
        "observed_max": float(observed.max()),
        "observed_mean": round(float(observed.mean()), 3),
        "observed_std": round(float(observed.std()), 3),
    }


def neighbour_correlation(speed: np.ndarray, adjacency: np.ndarray) -> dict:
    """Mean speed correlation of graph-connected vs. all distinct sensor pairs.

    If node_id -> sensor_id were mis-ordered, connected pairs would not be more
    correlated than arbitrary pairs. Call with training-period data only.
    """
    frame = pd.DataFrame(np.where(speed == MISSING_VALUE, np.nan, speed))
    corr = frame.corr(min_periods=1000).to_numpy()
    off_diag = ~np.eye(len(corr), dtype=bool)
    connected = (adjacency > 0) & off_diag
    return {
        "connected_pairs": int(connected.sum()),
        "mean_corr_connected": round(float(np.nanmean(corr[connected])), 4),
        "mean_corr_all_pairs": round(float(np.nanmean(corr[off_diag])), 4),
    }


def graph_checks(graph: SensorGraph, n_sensors: int) -> dict:
    """Consistency of sensor metadata and adjacency with the speed data."""
    s, adj = graph.sensors, graph.adjacency
    ids = set(s["sensor_id"].astype(int))
    in_distances = ids & set(graph.distances["from"]) & set(graph.distances["to"])
    return {
        "sensor_count": len(s),
        "matches_speed_sensor_count": len(s) == n_sensors,
        "unique_sensor_ids": bool(s["sensor_id"].is_unique),
        "latitude_range": [float(s["latitude"].min()), float(s["latitude"].max())],
        "longitude_range": [float(s["longitude"].min()), float(s["longitude"].max())],
        "adjacency_shape": list(adj.shape),
        "adjacency_value_range": [float(adj.min()), float(adj.max())],
        "adjacency_self_loops": bool((np.diag(adj) == 1).all()),
        "adjacency_symmetric": bool(np.allclose(adj, adj.T)),
        "adjacency_nonzero": int((adj > 0).sum()),
        "adjacency_parameters": graph.adjacency_meta.get("parameters"),
        "sensors_present_in_distances": len(in_distances),
        "distances_unique_sensors": int(
            pd.concat([graph.distances["from"], graph.distances["to"]]).nunique()
        ),
    }


def split_summary(
    timestamps: pd.DatetimeIndex,
    anchors: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    speed: np.ndarray,
) -> dict:
    """Per-split sample counts, time ranges and missingness on owned periods."""
    periods = owned_periods(timestamps, anchors)
    overlap = boundary_overlap(anchors, periods)
    out = {}
    for name, (first, last) in anchors.items():
        p_start, p_end = periods[name]
        mask = (timestamps >= p_start) & (timestamps <= p_end)
        out[name] = {
            "samples": int((last - first) / FREQ) + 1,
            "t0_first": str(first),
            "t0_last": str(last),
            "input_coverage": [str(first + INPUT_OFFSETS[0] * FREQ), str(last)],
            "target_coverage": [str(first + FREQ), str(last + TARGET_OFFSETS[-1] * FREQ)],
            "owned_period": [str(p_start), str(p_end)],
            "owned_timestamps": int(mask.sum()),
            "owned_missing_pct": round(float((speed[mask] == MISSING_VALUE).mean() * 100), 3),
            "samples_with_targets_beyond_owned_period": overlap[name],
        }
    return out
