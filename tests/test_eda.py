"""Unit tests for EDA statistics on small hand-built frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.analysis import eda


def frame(values, start="2012-03-05 00:00", columns=("a", "b")) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="5min")
    return pd.DataFrame(values, index=idx, columns=list(columns), dtype=float)


def test_run_lengths():
    starts, lengths = eda.run_lengths(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool))
    assert starts.tolist() == [0, 3, 6]
    assert lengths.tolist() == [2, 1, 3]


def test_run_lengths_empty():
    starts, lengths = eda.run_lengths(np.zeros(4, dtype=bool))
    assert len(starts) == len(lengths) == 0


def test_congestion_mask_uses_ratio_and_keeps_missing():
    speed = frame([[60, 60], [30, 59], [np.nan, 20]])
    mask = eda.congestion_mask(speed, free_flow=pd.Series({"a": 60.0, "b": 60.0}))
    assert mask["a"].tolist()[:2] == [0.0, 1.0] and np.isnan(mask["a"].iloc[2])
    assert mask["b"].tolist() == [0.0, 0.0, 1.0]


def test_congestion_episodes_split_on_missing():
    cong = frame([[1, 0], [1, 0], [np.nan, 1], [1, 1], [0, 1]])
    ep = eda.congestion_episodes(cong)
    a = ep[ep.sensor_id == "a"]
    assert a["steps"].tolist() == [2, 1]  # the NaN breaks the episode
    assert ep[ep.sensor_id == "b"]["steps"].tolist() == [3]
    assert a["start"].iloc[1] == cong.index[3]


def test_free_flow_is_85th_percentile():
    speed = frame(np.column_stack([np.arange(1, 101), np.full(100, 50)]))
    ff = eda.free_flow_speed(speed)
    assert ff["a"] == pytest.approx(np.percentile(np.arange(1, 101), 85))
    assert ff["b"] == 50


def test_time_of_day_profile_separates_weekends():
    idx = pd.date_range("2012-03-09 00:00", periods=288 * 2, freq="5min")  # Fri, Sat
    speed = pd.DataFrame({"a": np.where(idx.day_of_week >= 5, 40.0, 60.0)}, index=idx)
    prof = eda.time_of_day_profile(speed)
    assert prof.shape == (288, 2)
    assert (prof["weekday"] == 60).all() and (prof["weekend"] == 40).all()


def test_weekly_grid_shape():
    idx = pd.date_range("2012-03-05", periods=288 * 7, freq="5min")
    grid = eda.weekly_grid(pd.Series(idx.hour, index=idx, dtype=float))
    assert grid.shape == (7, 24)
    assert grid.index[0] == "Mon" and grid.loc["Sun", 23] == 23


def test_edge_lag_correlation_finds_known_delay():
    rng = np.random.default_rng(0)
    src = np.cumsum(rng.normal(size=2000)) + 60
    dst = np.roll(src, 2)  # dst repeats src two steps later
    speed = frame(np.column_stack([src, dst])[5:])
    adj = np.array([[1, 1], [0, 1]], dtype=float)  # single edge a -> b
    lagc = eda.edge_lag_correlation(speed, adj, max_lag=4)
    assert len(lagc) == 1
    assert lagc.iloc[0].idxmax() == 2
    assert lagc.iloc[0, 2] == pytest.approx(1.0)


def test_isolated_low_readings():
    speed = frame([[60, 60], [2, 3], [60, 4], [60, 4], [60, 60]])
    out = eda.isolated_low_readings(speed, low=5, neighbour_floor=30)
    assert out == {"low_readings": 4, "isolated": 1}  # only a's single-step dip is isolated
