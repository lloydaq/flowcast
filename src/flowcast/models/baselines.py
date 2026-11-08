"""Baseline forecasters: the floor every learned model has to beat.

Every baseline exposes ``predict(speed, h) -> (T, N)`` on the canonical matrix:
row ``t`` is the forecast of ``speed[t + h]`` issued at as-of time ``t``, using
readings at timestamps <= t only (tested like the features). Anything fitted
(the historical profile, the AR coefficients) is fitted on the training period.

Each baseline predicts every cell: when its own rule has no input (a sensor
outage), it falls back to the historical average, and the fallback share is
reported so the comparison stays honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.data.metr_la import FREQ
from flowcast.features.build import FFILL_LIMIT, STEPS_PER_DAY, STEPS_PER_WEEK, _shift

SLOTS_PER_DAY = STEPS_PER_DAY


def _slot(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    return times.day_of_week.to_numpy(), ((times.hour * 60 + times.minute) // 5).to_numpy()


class HistoricalAverage:
    """Mean speed per (sensor, day of week, 5-min slot) on the training period.

    Cells with no training reading fall back to the (sensor, slot) mean, then the sensor mean,
    so the profile is defined everywhere.
    """

    name = "historical_average"

    def fit(self, train: pd.DataFrame) -> HistoricalAverage:
        x = train.to_numpy(np.float64)
        dow, slot = _slot(train.index)
        n = x.shape[1]
        week = np.full((7, SLOTS_PER_DAY, n), np.nan)
        day = np.full((SLOTS_PER_DAY, n), np.nan)
        df = pd.DataFrame(x)
        week_mean = df.groupby([dow, slot]).mean()
        day_mean = df.groupby(slot).mean()
        idx = week_mean.index.to_frame().to_numpy()
        week[idx[:, 0], idx[:, 1]] = week_mean.to_numpy()
        day[day_mean.index.to_numpy()] = day_mean.to_numpy()
        sensor = np.nanmean(x, axis=0)
        day = np.where(np.isnan(day), sensor[None, :], day)
        self.profile_ = np.where(np.isnan(week), day[None, :, :], week)
        if np.isnan(self.profile_).any():
            raise ValueError("a sensor has no training readings at all")
        return self

    def at(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Profile value for each timestamp, ``(len(times), N)``."""
        dow, slot = _slot(times)
        return self.profile_[dow, slot]

    def predict(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        return self.at(speed.index + h * FREQ)


class _WithFallback:
    def __init__(self, ha: HistoricalAverage):
        self.ha = ha

    def _rule(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        raise NotImplementedError

    def predict(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        own = self._rule(speed, h)
        return np.where(np.isnan(own), self.ha.predict(speed, h), own)

    def fallback_mask(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        return np.isnan(self._rule(speed, h))


class Persistence(_WithFallback):
    """The last observed speed (bridging gaps up to 15 min, like the features), held constant."""

    name = "persistence"

    def _rule(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        return speed.ffill(limit=FFILL_LIMIT).to_numpy(np.float64)


class SeasonalNaive(_WithFallback):
    """The speed at the target's clock time exactly one week earlier."""

    name = "seasonal_naive_week"

    def _rule(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        return _shift(speed.to_numpy(np.float64), STEPS_PER_WEEK - h)


class ARResidual(_WithFallback):
    """Per-sensor linear autoregression on deviations from the historical average.

    ``speed[t+h] = HA(t+h) + c + sum_k phi_k * (speed[t-k] - HA(t-k))`` for k = 0..lags-1,
    one least-squares fit per (sensor, horizon) on training rows. A missing lag counts as a
    zero deviation (that reading is assumed to equal its usual value), so it always predicts.
    This is the classic "remove the seasonal profile, model what is left" statistical baseline.
    """

    name = "ar_residual"

    def __init__(self, ha: HistoricalAverage, lags: int = 12):
        super().__init__(ha)
        self.lags = lags

    def _design(self, dev: np.ndarray) -> np.ndarray:
        """``(T, lags + 1)``: an intercept and deviations at t, t-1, ... with NaN -> 0."""
        cols = [np.ones(len(dev))] + [_shift(dev, k) for k in range(self.lags)]
        return np.nan_to_num(np.stack(cols, axis=1), nan=0.0)

    def fit(self, speed: pd.DataFrame, train_times: np.ndarray, horizons: tuple[int, ...]) -> ARResidual:
        """``train_times``: as-of indices whose t and t + h all lie in the training period."""
        dev = speed.to_numpy(np.float64) - self.ha.at(speed.index)
        self.coef_: dict[int, np.ndarray] = {}
        for h in horizons:
            coef = np.zeros((dev.shape[1], self.lags + 1))
            for n in range(dev.shape[1]):
                x = self._design(dev[:, n])[train_times]
                y = dev[train_times + h, n]
                ok = ~np.isnan(y)
                coef[n] = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
            self.coef_[h] = coef
        return self

    def fallback_mask(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        """Rows where every lag is missing, so the forecast is the historical average plus c."""
        missing = speed.isna().rolling(self.lags, min_periods=1).sum().to_numpy()
        return missing >= self.lags

    def _rule(self, speed: pd.DataFrame, h: int) -> np.ndarray:
        dev = speed.to_numpy(np.float64) - self.ha.at(speed.index)
        out = np.empty_like(dev)
        for n in range(dev.shape[1]):
            out[:, n] = self._design(dev[:, n]) @ self.coef_[h][n]
        return np.clip(self.ha.predict(speed, h) + out, 0.0, None)
