"""Masked error metrics and evaluation slices, shared by every model.

Missing targets (NaN) are excluded, never scored as zero. A model must predict
every row whose target exists: a NaN prediction there raises, so no model can
look better by abstaining on hard rows.

Slices are defined on the **target time** (when the predicted speed happens),
not the as-of time, because that is the traffic condition being forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.congestion import CONGESTION_RATIO

MAPE_MIN_SPEED = 5.0  # mph; MAPE excludes targets below this (division by ~0 explodes)
# Weekday peaks: the hours where the training-period weekday mean dips (docs/experiments.md §1)
PEAK_HOURS = ((7, 10), (15, 19))  # [start, end) in hours


def masked_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE (mph) and MAPE (%) over rows with a target."""
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    mask = ~np.isnan(y_true)
    if np.isnan(y_pred[mask]).any():
        raise ValueError("prediction is NaN where a target exists; every scored row needs a prediction")
    err = y_pred[mask] - y_true[mask]
    n = int(mask.sum())
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0, "n_mape": 0}
    t = y_true[mask]
    pct = t >= MAPE_MIN_SPEED
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float((np.abs(err[pct]) / t[pct]).mean() * 100) if pct.any() else float("nan"),
        "n": n,
        "n_mape": int(pct.sum()),
    }


def is_peak(times: pd.DatetimeIndex) -> np.ndarray:
    hour = times.hour.to_numpy()
    in_window = np.zeros(len(times), dtype=bool)
    for start, end in PEAK_HOURS:
        in_window |= (hour >= start) & (hour < end)
    return in_window & (times.day_of_week.to_numpy() < 5)


def slice_masks(
    target_time: pd.DatetimeIndex, y_true: np.ndarray, free_flow: np.ndarray
) -> dict[str, np.ndarray]:
    """Boolean row masks for the reported slices (rows without a target are dropped by the metric)."""
    peak = is_peak(target_time)
    with np.errstate(invalid="ignore"):
        congested = y_true < CONGESTION_RATIO * free_flow
    return {
        "all": np.ones(len(y_true), dtype=bool),
        "peak": peak,
        "off_peak": ~peak,
        "congested": congested,
        "free_flowing": ~congested & ~np.isnan(y_true),
    }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_time: pd.DatetimeIndex,
    node_id: np.ndarray,
    free_flow: np.ndarray,
) -> dict:
    """Metrics overall and per slice, MAE by target hour, and MAE per sensor."""
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    masks = slice_masks(target_time, y_true, free_flow)
    out: dict = {name: masked_metrics(y_true[m], y_pred[m]) for name, m in masks.items()}
    err = pd.Series(np.abs(y_pred - y_true))  # NaN where the target is missing; groupby mean skips it
    by_hour = err.groupby(target_time.hour.to_numpy()).mean().dropna()
    by_sensor = err.groupby(np.asarray(node_id)).mean().dropna()
    out["mae_by_hour"] = {int(k): round(float(v), 4) for k, v in by_hour.items()}
    out["mae_by_sensor"] = {int(k): round(float(v), 4) for k, v in by_sensor.items()}
    return out
