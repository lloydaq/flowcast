"""Tests for masked metrics, slices and the baselines (including a leakage test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.evaluation.metrics import MAPE_MIN_SPEED, evaluate, is_peak, masked_metrics, slice_masks
from flowcast.features.build import HORIZONS, STEPS_PER_WEEK
from flowcast.models.baselines import ARResidual, HistoricalAverage, Persistence, SeasonalNaive

START = pd.Timestamp("2012-03-05 00:00")  # a Monday


def frame(x: np.ndarray, start: pd.Timestamp = START) -> pd.DataFrame:
    return pd.DataFrame(x, index=pd.date_range(start, periods=len(x), freq="5min"))


# --- metrics ---------------------------------------------------------------


def test_masked_metrics_skip_missing_targets_and_low_speeds_for_mape():
    y = np.array([10.0, np.nan, 2.0, 50.0])
    p = np.array([12.0, 99.0, 3.0, 45.0])  # the 99 sits on a missing target and must not count
    m = masked_metrics(y, p)
    assert m["n"] == 3 and m["mae"] == pytest.approx((2 + 1 + 5) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 1 + 25) / 3))
    assert MAPE_MIN_SPEED > 2.0 and m["n_mape"] == 2  # the 2 mph target is excluded from MAPE
    assert m["mape"] == pytest.approx((2 / 10 + 5 / 50) / 2 * 100)


def test_masked_metrics_reject_missing_predictions():
    with pytest.raises(ValueError):
        masked_metrics(np.array([10.0, 20.0]), np.array([10.0, np.nan]))


def test_peak_is_weekday_rush_hours_of_the_target_time():
    t = pd.DatetimeIndex(["2012-03-05 08:00", "2012-03-05 12:00", "2012-03-05 17:55", "2012-03-10 08:00"])
    assert is_peak(t).tolist() == [True, False, True, False]  # 03-10 is a Saturday


def test_slices_and_breakdowns():
    t = pd.DatetimeIndex(["2012-03-05 08:00", "2012-03-05 12:00", "2012-03-05 13:00"])
    y, p, ff = np.array([20.0, 60.0, np.nan]), np.array([25.0, 58.0, 1.0]), np.array([60.0, 60.0, 60.0])
    masks = slice_masks(t, y, ff)
    assert masks["congested"].tolist() == [True, False, False]
    assert masks["free_flowing"].tolist() == [False, True, False]  # a missing target is in neither
    r = evaluate(y, p, t, np.array([0, 1, 1]), ff)
    assert r["congested"]["mae"] == 5 and r["peak"]["n"] == 1
    assert r["mae_by_hour"] == {8: 5.0, 12: 2.0} and r["mae_by_sensor"] == {0: 5.0, 1: 2.0}


# --- baselines -------------------------------------------------------------


def two_weeks(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(20, 70, size=(STEPS_PER_WEEK * 3, 3))
    x[rng.random(x.shape) < 0.1] = np.nan
    return frame(x)


def test_historical_average_and_its_fallbacks():
    x = np.full((STEPS_PER_WEEK * 2, 2), 50.0)
    x[:STEPS_PER_WEEK, 0] = 40.0  # week 1 at 40, week 2 at 50 -> mean 45
    x[:, 1] = np.nan
    x[5, 1] = 30.0  # sensor 1: a single reading on Monday 00:25
    ha = HistoricalAverage().fit(frame(x))
    assert ha.profile_[0, 0, 0] == 45
    assert ha.profile_[0, 5, 1] == 30  # its own cell
    assert ha.profile_[3, 5, 1] == 30  # same slot, other weekday -> (sensor, slot) fallback
    assert ha.profile_[3, 100, 1] == 30  # other slot -> sensor mean fallback
    pred = ha.predict(frame(x), 3)
    assert pred[2, 1] == 30  # forecast issued at 00:10 for 00:25


def test_persistence_holds_last_value_and_falls_back_to_profile():
    speed = two_weeks()
    ha = HistoricalAverage().fit(speed)
    speed.iloc[100:110, 0] = np.nan  # a 50-minute outage on sensor 0
    p = Persistence(ha).predict(speed, 6)
    assert p[99, 0] == speed.iloc[99, 0]
    assert p[102, 0] == speed.iloc[99, 0]  # filled within 15 min
    assert p[105, 0] == pytest.approx(ha.at(speed.index[[105 + 6]])[0, 0])  # then the profile


def test_seasonal_naive_uses_one_week_before_the_target():
    speed = two_weeks()
    p = SeasonalNaive(HistoricalAverage().fit(speed)).predict(speed, 3)
    t = STEPS_PER_WEEK + 40
    if not np.isnan(speed.iloc[t + 3 - STEPS_PER_WEEK, 1]):
        assert p[t, 1] == speed.iloc[t + 3 - STEPS_PER_WEEK, 1]
    assert not np.isnan(p).any()


def test_ar_residual_recovers_a_known_process():
    """Deviations follow d[t+3] = 0.5 d[t]; the fitted coefficient should find it."""
    rng = np.random.default_rng(5)
    n = STEPS_PER_WEEK * 3
    d = np.zeros(n)
    for t in range(3, n):
        d[t] = 0.5 * d[t - 3] + rng.normal(0, 1)
    speed = frame((50 + d)[:, None])
    ha = HistoricalAverage().fit(speed)
    ar = ARResidual(ha, lags=4).fit(speed, np.arange(12, n - 12), (3,))
    assert ar.coef_[3][0, 1] == pytest.approx(0.5, abs=0.05)  # phi_0, the deviation at t


@pytest.mark.parametrize("make", ["persistence", "seasonal", "ar", "ha"])
def test_baselines_ignore_the_future(make):
    speed = two_weeks(seed=7)
    ha = HistoricalAverage().fit(speed)  # fitted once, so only the inputs change below
    model = {
        "persistence": Persistence(ha),
        "seasonal": SeasonalNaive(ha),
        "ar": ARResidual(ha).fit(speed, np.arange(12, len(speed) - 12), HORIZONS),
        "ha": ha,
    }[make]
    cut = len(speed) - 100
    speed.iloc[cut - 1 : cut + 1, 0] = (
        np.nan
    )  # a gap at the cut: a rule that fills it from later readings leaks
    future = speed.copy()
    future.iloc[cut + 1 :] = np.random.default_rng(1).uniform(0, 70, size=future.iloc[cut + 1 :].shape)
    for h in HORIZONS:
        np.testing.assert_array_equal(model.predict(speed, h)[: cut + 1], model.predict(future, h)[: cut + 1])
