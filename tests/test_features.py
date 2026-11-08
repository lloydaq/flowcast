"""Tests for point-in-time features: leakage, alignment and row selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.features.build import (
    FFILL_LIMIT,
    HORIZONS,
    MIN_HISTORY,
    STEPS_PER_WEEK,
    WEEKS_IN_PROFILE,
    compute_features,
    compute_targets,
    row_times,
    to_table,
)

START = pd.Timestamp("2012-03-01 00:00")


def speed_frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range(START, periods=len(values), freq="5min")
    return pd.DataFrame(values, index=idx, columns=[str(i) for i in range(values.shape[1])])


def random_case(n_steps: int, n_sensors: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(5, 70, size=(n_steps, n_sensors))
    x[rng.random(x.shape) < 0.1] = np.nan
    adj = np.where(rng.random((n_sensors, n_sensors)) < 0.5, rng.uniform(0.1, 1, (n_sensors, n_sensors)), 0.0)
    np.fill_diagonal(adj, 1.0)
    return speed_frame(x), adj, pd.Series(rng.uniform(55, 70, n_sensors))


def assert_same(a: np.ndarray, b: np.ndarray, name: str) -> None:
    np.testing.assert_array_equal(a, b, err_msg=f"feature {name!r} changed when only the future changed")


# --- leakage ---------------------------------------------------------------


def test_features_ignore_the_future():
    """Overwrite every reading after t_cut with noise: no feature at or before t_cut may change."""
    n_steps = WEEKS_IN_PROFILE * STEPS_PER_WEEK + 200  # long enough to exercise every seasonal lag
    speed, adj, ff = random_case(n_steps, 6, seed=1)
    cut = n_steps - 60

    future = speed.copy()
    rng = np.random.default_rng(99)
    noise = rng.uniform(0, 70, size=future.iloc[cut + 1 :].shape)
    noise[rng.random(noise.shape) < 0.3] = np.nan  # different missing pattern too
    future.iloc[cut + 1 :] = noise

    before, after = compute_features(speed, adj, ff), compute_features(future, adj, ff)
    assert before.keys() == after.keys()
    for name in before:
        assert_same(before[name][: cut + 1], after[name][: cut + 1], name)

    # Sanity: the perturbation did reach the targets, so the test is not vacuous.
    t_before, t_after = compute_targets(speed), compute_targets(future)
    assert not np.array_equal(t_before["target_h3"][cut], t_after["target_h3"][cut], equal_nan=True)


def test_seasonal_features_only_look_back():
    speed, adj, ff = random_case(STEPS_PER_WEEK * 2, 3, seed=2)
    feats = compute_features(speed, adj, ff)
    x = speed.to_numpy()
    t = STEPS_PER_WEEK + 500
    for h in HORIZONS:
        assert feats[f"day_ago_h{h}"][t, 0] == pytest.approx(x[t + h - 288, 0], nan_ok=True)
        assert feats[f"week_ago_h{h}"][t, 0] == pytest.approx(x[t + h - STEPS_PER_WEEK, 0], nan_ok=True)
        assert t + h - 288 < t  # offset strictly in the past


# --- alignment against hand-computed values --------------------------------


def test_lag_rolling_and_trend_values():
    x = np.array([[10.0], [20.0], [np.nan], [40.0], [50.0], [60.0], [70.0]])
    adj = np.ones((1, 1))
    f = compute_features(speed_frame(x), adj, pd.Series([60.0]))
    t = 4  # value 50
    assert f["speed"][t, 0] == 50
    assert f["speed_lag1"][t, 0] == 40
    assert np.isnan(f["speed_lag2"][t, 0])  # the missing reading stays missing
    assert f["speed_lag3"][t, 0] == 20
    assert f["roll3_mean"][t, 0] == pytest.approx(45)  # mean of 40, 50 (NaN skipped)
    assert f["roll3_min"][t, 0] == 40 and f["roll3_max"][t, 0] == 50
    assert f["trend3"][t, 0] == pytest.approx(50 - 20)
    assert f["relative_speed"][t, 0] == pytest.approx(50 / 60)
    assert f["is_missing"][2, 0] == 1 and f["is_missing_lag1"][3, 0] == 1
    assert f["missing_count_1h"][t, 0] == 1
    assert np.isnan(f["speed_lag1"][0, 0])  # nothing before the first reading


def test_forward_fill_is_limited():
    x = np.array([[50.0]] + [[np.nan]] * (FFILL_LIMIT + 2) + [[60.0]])
    f = compute_features(speed_frame(x), np.ones((1, 1)), pd.Series([60.0]))
    last = f["speed_last_valid"][:, 0]
    assert (last[1 : FFILL_LIMIT + 1] == 50).all()
    assert np.isnan(last[FFILL_LIMIT + 1])  # gap longer than 15 min is not filled


def test_week_profile_is_nan_aware_mean():
    n = WEEKS_IN_PROFILE * STEPS_PER_WEEK + 20
    x = np.full((n, 1), np.nan)
    t, h = n - 13, 3
    for k, v in zip(range(1, WEEKS_IN_PROFILE + 1), (10.0, np.nan, 30.0, 50.0), strict=True):
        x[t + h - k * STEPS_PER_WEEK, 0] = v
    f = compute_features(speed_frame(x), np.ones((1, 1)), pd.Series([60.0]))
    assert f["week_profile_h3"][t, 0] == pytest.approx(30.0)  # mean of 10, 30, 50


def test_neighbour_features_follow_direction_and_skip_missing():
    # sensor 0 -> 1 (w 1.0), 0 -> 2 (w 0.5); sensor 3 is isolated
    adj = np.eye(4)
    adj[0, 1], adj[0, 2] = 1.0, 0.5
    x = np.array([[60.0, 30.0, 60.0, 40.0], [60.0, np.nan, 60.0, 40.0]])
    f = compute_features(speed_frame(x), adj, pd.Series([65.0] * 4))
    assert f["down_mean"][0, 0] == pytest.approx((30 * 1.0 + 60 * 0.5) / 1.5)
    assert f["down_mean"][1, 0] == pytest.approx(60.0)  # missing neighbour dropped, weights renormalised
    assert f["up_mean"][0, 1] == pytest.approx(60.0)  # sensor 1 is reached from sensor 0
    assert np.isnan(f["down_mean"][0, 1])  # sensor 1 reaches nobody
    assert f["neighbour_min"][0, 0] == 30.0
    assert np.isnan(f["down_mean"][0, 3]) and np.isnan(f["neighbour_min"][0, 3])


def test_calendar_features():
    speed, adj, ff = random_case(300, 2, seed=3)  # starts Thu 2012-03-01 00:00
    f = compute_features(speed, adj, ff)
    assert f["day_of_week"][0, 0] == 3 and f["is_weekend"][0, 0] == 0
    assert f["tod_sin"][0, 0] == pytest.approx(0) and f["tod_cos"][0, 0] == pytest.approx(1)
    six_am = 72
    assert f["tod_sin"][six_am, 0] == pytest.approx(1, abs=1e-6)


def test_targets_are_future_values():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    t = compute_targets(speed_frame(x))
    assert t["target_h3"][5, 0] == 8 and t["target_h12"][5, 0] == 17
    assert np.isnan(t["target_h12"][10, 0])  # past the end


# --- row selection ---------------------------------------------------------


def test_row_times_keep_targets_inside_period():
    ts = pd.date_range(START, periods=200, freq="5min")
    rows = row_times(ts, (ts[50], ts[120]))
    assert rows[0] == 50 and rows[-1] == 120 - max(HORIZONS)
    first = row_times(ts, (ts[0], ts[120]))
    assert first[0] == MIN_HISTORY


def test_to_table_drops_rows_without_targets_and_types_categoricals():
    x = np.full((40, 2), 50.0)
    x[20:, 1] = np.nan  # sensor 1 has no future readings after step 20
    speed = speed_frame(x)
    feats, tgts = compute_features(speed, np.eye(2), pd.Series([60.0, 60.0])), compute_targets(speed)
    times = np.arange(15, 25)
    table = to_table(feats, tgts, times, speed.index)
    df = table.to_pandas()
    assert str(table.schema.field("node_id").type) == "int16"
    assert df["timestamp"].is_monotonic_increasing
    # sensor 1 rows at t >= 17 have all targets (t+3 .. t+12) missing -> dropped
    assert df[df.node_id == 1]["timestamp"].max() == speed.index[16]
    assert (df[df.node_id == 0]["timestamp"] == speed.index[times]).all()


def test_feature_columns_select_matching_seasonal_features():
    from flowcast.features.build import feature_columns

    speed, adj, ff = random_case(50, 2, seed=4)
    names = ["timestamp", *compute_features(speed, adj, ff), *compute_targets(speed)]
    cols = feature_columns(names, 6)
    assert {"day_ago_h6", "week_ago_h6", "week_profile_h6"} <= set(cols)
    assert not any(c.endswith(("_h3", "_h9", "_h12")) for c in cols)
    assert "timestamp" not in cols and not any(c.startswith("target") for c in cols)
    assert len(cols) == 39
    with pytest.raises(ValueError):
        feature_columns(names, 4)
