"""Point-in-time feature generation for multi-horizon speed forecasting.

A row is (sensor, as-of time t). Its targets are the speeds at t + h for each
horizon h. **Every feature for row t is a function of readings at timestamps
<= t only.** That is guaranteed by construction (backward shifts and trailing
windows, never centred ones) and verified by ``tests/test_features.py``, which
overwrites the future with noise and checks that no feature changes.

Features are computed on the whole canonical matrix at once, as one
``(n_timestamps, n_sensors)`` float32 array per feature. Rows are then selected
per split by their timestamp. Validation and test rows may use history from
earlier splits, because the past is known at prediction time.

Missing readings are NaN. They stay NaN (LightGBM handles them natively), with
explicit missing-indicator features. Forward-filling is limited to 3 steps (15 min).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

HORIZONS = (3, 6, 9, 12)  # steps ahead: 15, 30, 45, 60 minutes
LAGS = (1, 2, 3, 6, 12)  # steps back, in addition to the current reading
WINDOWS = (3, 6, 12)  # trailing windows: 15, 30, 60 minutes
TRENDS = (3, 6)
STEPS_PER_DAY = 288
STEPS_PER_WEEK = 2016
WEEKS_IN_PROFILE = 4
FFILL_LIMIT = 3
MIN_HISTORY = 11  # first usable row has 11 earlier steps, matching the official first t0 (00:55)

CATEGORICAL = ("node_id", "day_of_week")


def target_column(h: int) -> str:
    return f"target_h{h}"


SEASONAL_PREFIXES = ("day_ago_h", "week_ago_h", "week_profile_h")


def feature_columns(all_columns: list[str], h: int) -> list[str]:
    """Model inputs for horizon ``h``: every shared feature plus the seasonal ones aimed at t + h.

    Seasonal features for other horizons are legal (past data) but redundant, so they are left out.
    """
    if h not in HORIZONS:
        raise ValueError(f"horizon {h} not in {HORIZONS}")
    keep = []
    for c in all_columns:
        if c == "timestamp" or c.startswith("target_h"):
            continue
        prefix = next((p for p in SEASONAL_PREFIXES if c.startswith(p)), None)
        if prefix is None or c == f"{prefix}{h}":
            keep.append(c)
    return keep


def compute_features(
    speed: pd.DataFrame, adjacency: np.ndarray, free_flow: pd.Series
) -> dict[str, np.ndarray]:
    """All features as ``(n_timestamps, n_sensors)`` float32 arrays.

    ``speed``: canonical matrix (index = timestamp, columns = sensors in node_id
    order, missing = NaN). ``adjacency``: directed weights, ``A[i, j]`` = kernel of
    the road distance from sensor i to j. ``free_flow``: per-sensor free-flow
    speed estimated on the training period only.
    """
    if speed.shape[1] != adjacency.shape[0]:
        raise ValueError("speed columns and adjacency size differ")
    x = speed.to_numpy(dtype=np.float64)
    f = _Float32Dict()  # cast on insert: keeps peak memory at half of float64

    # Recent readings
    f["speed"] = x
    for k in LAGS:
        f[f"speed_lag{k}"] = _shift(x, k)
    f["speed_last_valid"] = speed.ffill(limit=FFILL_LIMIT).to_numpy()

    # Trailing windows (include the current reading, never anything later)
    for w in WINDOWS:
        roll = speed.rolling(w, min_periods=1)
        f[f"roll{w}_mean"] = roll.mean().to_numpy()
        f[f"roll{w}_min"] = roll.min().to_numpy()
        f[f"roll{w}_max"] = roll.max().to_numpy()
        f[f"roll{w}_std"] = speed.rolling(w, min_periods=2).std().to_numpy()

    for k in TRENDS:
        f[f"trend{k}"] = x - _shift(x, k)

    # Seasonal: the same clock time as the *target*, yesterday and in earlier weeks.
    # Offsets are (period - h) > 0 because h <= 12 << 288, so they look back only.
    for h in HORIZONS:
        f[f"day_ago_h{h}"] = _shift(x, STEPS_PER_DAY - h)
        f[f"week_ago_h{h}"] = _shift(x, STEPS_PER_WEEK - h)
        weeks = np.stack([_shift(x, k * STEPS_PER_WEEK - h) for k in range(1, WEEKS_IN_PROFILE + 1)])
        f[f"week_profile_h{h}"] = _nanmean(weeks)

    # Calendar (of the as-of time)
    idx = speed.index
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    angle = 2 * np.pi * minutes / 1440
    n = speed.shape[1]
    f["tod_sin"] = np.repeat(np.sin(angle)[:, None], n, axis=1)
    f["tod_cos"] = np.repeat(np.cos(angle)[:, None], n, axis=1)
    f["day_of_week"] = np.repeat(idx.day_of_week.to_numpy()[:, None], n, axis=1)
    f["is_weekend"] = (f["day_of_week"] >= 5).astype(np.float64)

    # Sensor
    ff = free_flow.to_numpy(dtype=np.float64)
    f["node_id"] = np.repeat(np.arange(n)[None, :], len(idx), axis=0)
    f["free_flow"] = np.repeat(ff[None, :], len(idx), axis=0)
    f["relative_speed"] = x / ff

    # Spatial, from the directed graph (self-loops removed)
    a = np.where(np.eye(n, dtype=bool), 0.0, adjacency.astype(np.float64))
    roll3 = f["roll3_mean"].astype(np.float64)
    f["down_mean"] = _weighted_neighbour_mean(x, a)  # sensors reachable from this one
    f["up_mean"] = _weighted_neighbour_mean(x, a.T)  # sensors that reach this one
    f["down_roll3_mean"] = _weighted_neighbour_mean(roll3, a)
    f["up_roll3_mean"] = _weighted_neighbour_mean(roll3, a.T)
    f["neighbour_min"] = _neighbour_min(x, (a > 0) | (a.T > 0))

    # Missingness
    missing = np.isnan(x)
    f["is_missing"] = missing.astype(np.float64)
    f["is_missing_lag1"] = _shift(missing.astype(np.float64), 1)
    f["missing_count_1h"] = speed.isna().rolling(12, min_periods=1).sum().to_numpy()

    return dict(f)


class _Float32Dict(dict):
    def __setitem__(self, key: str, value: np.ndarray) -> None:
        super().__setitem__(key, np.asarray(value, dtype=np.float32))


def compute_targets(speed: pd.DataFrame) -> dict[str, np.ndarray]:
    """Future speeds: ``target_h{h}[t] = speed[t + h]`` (NaN where missing or past the end)."""
    x = speed.to_numpy(dtype=np.float64)
    return {target_column(h): _shift(x, -h).astype(np.float32) for h in HORIZONS}


def row_times(timestamps: pd.DatetimeIndex, period: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    """Time indices of rows owned by a split.

    A row at t is kept when t and every target t + h lie in the split's owned
    period, so no target crosses into another split, and t has MIN_HISTORY
    earlier steps.
    """
    start, end = timestamps.get_loc(period[0]), timestamps.get_loc(period[1])
    first = max(start, MIN_HISTORY)
    last = end - max(HORIZONS)
    return np.arange(first, last + 1)


def to_table(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    times: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pa.Table:
    """Long table, one row per (time, sensor), ordered by time then node_id.

    Rows whose targets are all missing are dropped: they cannot train or score anything.
    """
    n = next(iter(features.values())).shape[1]
    tgt = np.stack([targets[c][times].reshape(-1) for c in targets], axis=1)
    keep = ~np.isnan(tgt).all(axis=1)
    columns = {"timestamp": pa.array(np.repeat(timestamps[times].to_numpy(), n)[keep])}
    for name, arr in features.items():
        values = arr[times].reshape(-1)[keep]
        columns[name] = pa.array(values.astype(np.int16) if name in CATEGORICAL else values)
    for i, name in enumerate(targets):
        columns[name] = pa.array(tgt[keep, i])
    return pa.table(columns)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """``out[t] = x[t - k]`` (k > 0 looks back, k < 0 looks ahead), NaN-padded."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def _nanmean(stack: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _weighted_neighbour_mean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``out[t, i] = sum_j w[i, j] x[t, j] / sum_j w[i, j]`` over j with a reading."""
    valid = ~np.isnan(x)
    num = np.where(valid, x, 0.0) @ weights.T
    den = valid.astype(np.float64) @ weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _neighbour_min(x: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(x.shape[1]):
        idx = np.flatnonzero(neighbours[i])
        if len(idx):
            block = x[:, idx]
            has = ~np.isnan(block).all(axis=1)
            out[has, i] = np.nanmin(block[has], axis=1)
    return out
