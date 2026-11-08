"""Exploratory statistics for traffic speed matrices.

Every function takes a speed frame indexed by timestamp with one column per
sensor and missing readings as NaN (see ``flowcast.data.processed.load_speed``).
Callers decide which period to pass; FlowCast's EDA uses the training period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.congestion import CONGESTION_RATIO, FREE_FLOW_PERCENTILE, free_flow_speed  # noqa: F401

STEPS_PER_HOUR = 12
STEPS_PER_DAY = 288


def congestion_mask(speed: pd.DataFrame, free_flow: pd.Series | None = None) -> pd.DataFrame:
    """Boolean-like frame: 1.0 congested, 0.0 free, NaN where the reading is missing."""
    ff = free_flow_speed(speed) if free_flow is None else free_flow
    congested = (speed.div(ff, axis=1) < CONGESTION_RATIO).astype(float)
    return congested.mask(speed.isna())


def run_lengths(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of contiguous ``True`` runs in a 1-D boolean array."""
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    starts, ends = edges[::2], edges[1::2]
    return starts, ends - starts


def congestion_episodes(congested: pd.DataFrame) -> pd.DataFrame:
    """One row per congestion episode: sensor, start timestamp, duration in steps.

    Missing readings end an episode (we do not assume congestion through gaps).
    """
    rows = []
    for sensor in congested.columns:
        flags = congested[sensor].to_numpy() == 1.0
        starts, lengths = run_lengths(flags)
        rows.append(pd.DataFrame({"sensor_id": sensor, "start": congested.index[starts], "steps": lengths}))
    return pd.concat(rows, ignore_index=True)


def time_of_day_profile(speed: pd.DataFrame) -> pd.DataFrame:
    """Network-mean speed per 5-min time slot, separately for weekdays and weekends."""
    network = speed.mean(axis=1)
    idx = network.index
    frame = pd.DataFrame(
        {
            "speed": network.to_numpy(),
            "slot": idx.hour * STEPS_PER_HOUR + idx.minute // 5,
            "day_type": np.where(idx.day_of_week >= 5, "weekend", "weekday"),
        }
    )
    return frame.pivot_table(index="slot", columns="day_type", values="speed", aggfunc="mean")


def weekly_grid(values: pd.Series) -> pd.DataFrame:
    """Mean of ``values`` per (day of week, hour); rows Mon..Sun, columns 0..23."""
    idx = values.index
    grid = values.groupby([idx.day_of_week, idx.hour]).mean().unstack()
    grid.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(grid)]
    return grid


def autocorrelation(series: pd.Series, lags: list[int]) -> pd.Series:
    """Pearson autocorrelation at each lag, ignoring pairs with a missing value."""
    return pd.Series({lag: series.autocorr(lag) for lag in lags})


def horizon_autocorrelation(speed: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Per-sensor corr(speed_t, speed_{t+h}): an upper bound on what persistence can exploit."""
    return pd.DataFrame({h: speed.apply(lambda s, h=h: s.autocorr(h)) for h in horizons})


def edge_lag_correlation(speed: pd.DataFrame, adjacency: np.ndarray, max_lag: int) -> pd.DataFrame:
    """For each directed graph edge i->j, corr(x_i(t), x_j(t+lag)) for lag = 0..max_lag.

    Uses first differences so shared daily seasonality does not dominate. A peak
    at lag > 0 means changes at i tend to appear at j later.
    """
    diff = speed.diff().to_numpy()
    src, dst = np.nonzero((adjacency > 0) & ~np.eye(len(adjacency), dtype=bool))
    out = {}
    for lag in range(max_lag + 1):
        a = diff[: len(diff) - lag, src]
        b = diff[lag:, dst]
        out[lag] = _columnwise_corr(a, b)
    return pd.DataFrame(out, index=pd.MultiIndex.from_arrays([src, dst], names=["src", "dst"]))


def _columnwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = np.where(valid, a, 0.0), np.where(valid, b, 0.0)
    n = valid.sum(axis=0)
    mean_a, mean_b = a.sum(0) / n, b.sum(0) / n
    cov = (a * b).sum(0) / n - mean_a * mean_b
    var_a = (a * a).sum(0) / n - mean_a**2
    var_b = (b * b).sum(0) / n - mean_b**2
    return cov / np.sqrt(var_a * var_b)


def correlation_by_distance(
    speed: pd.DataFrame, distances: pd.DataFrame, sensor_ids: list[str], bins_km: list[float]
) -> pd.DataFrame:
    """Mean pairwise speed correlation binned by directed road-network distance."""
    corr = speed.corr(min_periods=1000)
    ids = set(sensor_ids)
    d = distances[distances["from"].astype(str).isin(ids) & distances["to"].astype(str).isin(ids)]
    d = d[d["from"] != d["to"]]
    values = corr.to_numpy()
    pos = {s: i for i, s in enumerate(corr.columns)}
    rows = d["from"].astype(str).map(pos).to_numpy()
    cols = d["to"].astype(str).map(pos).to_numpy()
    frame = pd.DataFrame({"km": d["cost"].to_numpy() / 1000, "corr": values[rows, cols]})
    frame["bin"] = pd.cut(frame["km"], bins_km)
    return frame.groupby("bin", observed=True)["corr"].agg(["mean", "count"])


def isolated_low_readings(speed: pd.DataFrame, low: float, neighbour_floor: float) -> dict:
    """Classify positive readings below ``low`` mph by their temporal neighbours.

    *Isolated* means both the previous and next readings exceed ``neighbour_floor``:
    a single-step dip that is physically implausible on a highway and likely a
    sensor glitch. The rest sit inside sustained slow traffic.
    """
    prev, nxt = speed.shift(1), speed.shift(-1)
    is_low = (speed > 0) & (speed < low)
    isolated = is_low & (prev > neighbour_floor) & (nxt > neighbour_floor)
    return {"low_readings": int(is_low.sum().sum()), "isolated": int(isolated.sum().sum())}


def morning_trough_slot(speed: pd.DataFrame) -> int:
    """Weekday 5-min slot (0..287) with the lowest network-mean speed before noon."""
    profile = time_of_day_profile(speed)["weekday"]
    return int(profile.loc[: 12 * STEPS_PER_HOUR - 1].idxmin())
