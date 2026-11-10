"""Everything the API serves, loaded once into memory.

Forecasts are the ones each model made during evaluation (``data/processed/predictions/``): for every
5-minute as-of time in the test period, every sensor and every horizon, each computed from readings at
or before that time only (leakage-tested in Phases 3, 5 and 6). Serving them is the replay clock: the
app "stands" at a past moment and shows what the model forecast then, and what actually happened.

All data is immutable, so lookups are cached without invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd

from flowcast import paths
from flowcast.congestion import CONGESTION_RATIO, free_flow_speed
from flowcast.data.metr_la import FREQ
from flowcast.data.processed import load_sensors, load_speed
from flowcast.evaluation.results import load_comparison, load_results
from flowcast.features.build import HORIZONS

STEP_MIN = int(FREQ / pd.Timedelta(minutes=1))  # 5
HORIZON_MIN = tuple(h * STEP_MIN for h in HORIZONS)  # (15, 30, 45, 60)

# Registry name -> (display name used in the metrics JSON, description). The first one is the default.
MODELS = {
    "dcrnn": ("DCRNN", "Graph recurrent network over the road graph; lowest MAE up to 45 min"),
    "lightgbm": ("LightGBM", "Gradient-boosted trees, one model per horizon"),
}


class SensorNotFoundError(LookupError):
    """A sensor that does not exist (the API maps it to 404)."""


class InvalidQueryError(ValueError):
    """An invalid query value: as-of time, horizon or model (the API maps it to 422)."""


def _num(x: float) -> float | None:
    """JSON-safe value: NaN (missing reading / no forecast) becomes null; 2 decimals is 0.01 mph."""
    return None if np.isnan(x) else round(float(x), 2)


@dataclass
class ForecastStore:
    speed: pd.DataFrame  # full canonical timeline (timestamp x node), NaN = missing
    train_speed: pd.DataFrame  # training period only: free flow and profiles come from here
    sensors: pd.DataFrame  # node_id, sensor_id, latitude, longitude, missing_pct (row order = node)
    predictions: dict[str, pd.DataFrame]  # registry name -> timestamp, node_id, pred_h{3,6,9,12}
    results: dict[str, dict] = field(default_factory=dict)  # display name -> split -> horizon -> slices
    comparison: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.predictions) - set(MODELS)
        if unknown or not self.predictions:
            raise ValueError(
                f"predictions must be for registered models {list(MODELS)}, got {unknown or 'none'}"
            )
        self.times = self.speed.index
        if not (self.times.to_series().diff().iloc[1:] == FREQ).all():
            raise ValueError("speed matrix must be a gap-free 5-minute grid (rows are indexed by position)")
        self.values = self.speed.to_numpy(np.float32)
        self.n_nodes = self.values.shape[1]
        self.free_flow = free_flow_speed(self.train_speed).to_numpy(np.float64)
        self.node_of = {
            s: int(n) for s, n in zip(self.sensors["sensor_id"], self.sensors["node_id"], strict=True)
        }

        # replay range: the as-of times with forecasts. Times inside it with no saved rows (network-wide
        # outages: every target missing) stay NaN on a dense grid, so a slider never hits a hole.
        first = min(p["timestamp"].min() for p in self.predictions.values())
        last = max(p["timestamp"].max() for p in self.predictions.values())
        self.replay = pd.date_range(first, last, freq=FREQ)
        self.forecast: dict[str, np.ndarray] = {}  # name -> (len(replay), N, 4) mph
        for name, p in self.predictions.items():
            grid = np.full((len(self.replay), self.n_nodes, len(HORIZONS)), np.nan, np.float32)
            r = self.replay.get_indexer(pd.DatetimeIndex(p["timestamp"]))
            if (r < 0).any():
                raise ValueError(f"{name}: prediction times off the 5-minute grid")
            for i, h in enumerate(HORIZONS):
                grid[r, p["node_id"].to_numpy(), i] = p[f"pred_h{h}"].to_numpy(np.float32)
            self.forecast[name] = grid
        self.replay_pos = self.times.get_indexer(self.replay)  # replay row -> canonical row
        if (self.replay_pos < 0).any() or self.replay_pos[-1] + max(HORIZONS) >= len(self.times):
            raise ValueError("speed matrix does not cover the replay range plus the longest horizon")
        self.predictions = {}  # the dense grids replace the frames; free the memory
        self.network = lru_cache(maxsize=4096)(self._network)
        self.sensor_detail = lru_cache(maxsize=None)(self._sensor_detail)

    # ---- loading --------------------------------------------------------------------------------

    @classmethod
    def load(cls) -> ForecastStore:
        """Read the processed data, test-period predictions and metrics JSON from disk."""
        missing = [m for m in MODELS if not (paths.PREDICTIONS_DIR / f"{m}_test.parquet").exists()]
        if missing:
            raise FileNotFoundError(
                f"no saved test predictions for {missing} in {paths.PREDICTIONS_DIR}; "
                "run scripts/train_lightgbm.py and scripts/train_dcrnn.py"
            )
        return cls(
            speed=load_speed(),
            train_speed=load_speed("train"),
            sensors=load_sensors(),
            predictions={m: pd.read_parquet(paths.PREDICTIONS_DIR / f"{m}_test.parquet") for m in MODELS},
            results=load_results(),
            comparison=load_comparison(),
        )

    # ---- validation -----------------------------------------------------------------------------

    def model(self, name: str | None) -> str:
        if name is None:
            return next(m for m in MODELS if m in self.forecast)
        if name not in self.forecast:
            raise InvalidQueryError(f"unknown model {name!r}; expected one of {sorted(self.forecast)}")
        return name

    def node(self, sensor_id: str) -> int:
        try:
            return self.node_of[sensor_id]
        except KeyError:
            raise SensorNotFoundError(f"unknown sensor {sensor_id!r}") from None

    def replay_row(self, as_of: pd.Timestamp) -> int:
        if as_of.tzinfo is not None:
            raise InvalidQueryError(
                "as_of must be a local Los Angeles time without a timezone, e.g. 2012-06-04T08:00"
            )
        if as_of != as_of.floor(FREQ):
            raise InvalidQueryError(f"as_of must fall on the {STEP_MIN}-minute grid, e.g. 08:00 or 08:05")
        if not self.replay[0] <= as_of <= self.replay[-1]:
            raise InvalidQueryError(
                f"as_of must be between {self.replay[0]} and {self.replay[-1]} (the test period)"
            )
        return int(self.replay.get_loc(as_of))

    @staticmethod
    def horizon_index(minutes: int) -> int:
        if minutes not in HORIZON_MIN:
            raise InvalidQueryError(f"horizon must be one of {list(HORIZON_MIN)} minutes")
        return HORIZON_MIN.index(minutes)

    # ---- queries --------------------------------------------------------------------------------

    def _network(self, model: str, row: int, hi: int) -> dict:
        """Every sensor's current speed, forecast and the actual speed that followed, at one horizon."""
        t = self.replay_pos[row]
        steps = HORIZONS[hi]
        fc = self.forecast[model][row, :, hi]
        now, actual = self.values[t], self.values[t + steps]
        sensors = [
            {
                "sensor_id": sid,
                "speed_now": _num(now[n]),
                "forecast": _num(fc[n]),
                "actual": _num(actual[n]),
                "free_flow": _num(self.free_flow[n]),
                "congested": None if np.isnan(fc[n]) else bool(fc[n] < CONGESTION_RATIO * self.free_flow[n]),
            }
            for sid, n in self.node_of.items()
        ]
        return {
            "model": model,
            "as_of": self.replay[row],
            "horizon_min": HORIZON_MIN[hi],
            "target_time": self.times[t + steps],
            "sensors": sensors,
        }

    def sensor_forecast(self, model: str, row: int, node: int, history_steps: int) -> dict:
        t = self.replay_pos[row]
        lo = max(0, t - history_steps)
        series = self.values[:, node]
        return {
            "model": model,
            "sensor_id": self.sensors["sensor_id"].iat[node],
            "as_of": self.replay[row],
            "free_flow": _num(self.free_flow[node]),
            "history": [{"time": self.times[i], "speed": _num(series[i])} for i in range(lo, t + 1)],
            "forecasts": [
                {
                    "horizon_min": HORIZON_MIN[i],
                    "target_time": self.times[t + h],
                    "forecast": _num(self.forecast[model][row, node, i]),
                    "actual": _num(series[t + h]),
                }
                for i, h in enumerate(HORIZONS)
            ],
            "actual_after": [
                {"time": self.times[i], "speed": _num(series[i])} for i in range(t + 1, t + max(HORIZONS) + 1)
            ],
        }

    @cached_property
    def sensor_list(self) -> list[dict]:
        s = self.sensors
        return [
            {
                "sensor_id": sid,
                "latitude": float(lat),
                "longitude": float(lon),
                "free_flow": _num(self.free_flow[n]),
                "missing_pct": float(miss),
                "train_congested_pct": _num(self.train_congested_pct[n]),
            }
            for n, sid, lat, lon, miss in zip(
                s["node_id"], s["sensor_id"], s["latitude"], s["longitude"], s["missing_pct"], strict=True
            )
        ]

    @cached_property
    def train_congested_pct(self) -> np.ndarray:
        """Per sensor: share of its training readings below 0.6 x its free-flow speed."""
        v = self.train_speed.to_numpy(np.float64)
        ok = ~np.isnan(v)
        with np.errstate(invalid="ignore"):
            return ((v < CONGESTION_RATIO * self.free_flow) & ok).sum(0) / np.maximum(ok.sum(0), 1) * 100

    def sensor_errors(self, model: str, split: str, minutes: int) -> list[dict]:
        """Every sensor's masked MAE for one evaluated model, split and horizon; worst first."""
        if model not in self.results:
            raise InvalidQueryError(
                f"unknown evaluated model {model!r}; expected one of {list(self.results)}"
            )
        if split not in ("validation", "test"):
            raise InvalidQueryError("split must be 'validation' or 'test'")
        by_node = self.results[model][split][f"h{HORIZONS[self.horizon_index(minutes)]}"]["mae_by_sensor"]
        rows = [
            {"sensor_id": self.sensors["sensor_id"].iat[int(n)], "mae": round(float(v), 3)}
            for n, v in by_node.items()
        ]
        return sorted(rows, key=lambda r: -r["mae"])

    def _sensor_detail(self, node: int) -> dict:
        col = self.train_speed.iloc[:, node]
        weekday = col[col.index.day_of_week < 5]
        hourly = weekday.groupby(weekday.index.hour).mean()
        return {
            **self.sensor_list[node],
            "train_mean_speed": _num(col.mean()),
            "weekday_hourly_speed": [_num(hourly.get(h, np.nan)) for h in range(24)],
        }

    @cached_property
    def daily_profile(self) -> dict:
        """Network mean speed per 5-minute slot of the day, weekdays vs weekends (training period)."""
        net = self.train_speed.mean(axis=1)
        slot = net.index.hour * 60 + net.index.minute
        out = {}
        for name, mask in (("weekday", net.index.day_of_week < 5), ("weekend", net.index.day_of_week >= 5)):
            prof = net[mask].groupby(slot[mask]).mean()
            out[name] = [_num(prof.get(m, np.nan)) for m in range(0, 24 * 60, STEP_MIN)]
        return {"slot_minutes": STEP_MIN, **out}

    @cached_property
    def weekly_congestion(self) -> dict:
        """Share of readings that are congested, by day of week (Mon = 0) and hour (training period)."""
        v = self.train_speed.to_numpy(np.float64)
        ok = ~np.isnan(v)
        cong = (v < CONGESTION_RATIO * self.free_flow) & ok
        idx = self.train_speed.index
        key = idx.day_of_week * 24 + idx.hour
        num = np.bincount(key, cong.sum(axis=1), minlength=168)
        den = np.bincount(key, ok.sum(axis=1), minlength=168)
        pct = np.where(den > 0, num / np.maximum(den, 1) * 100, np.nan).reshape(7, 24)
        return {"rate_pct": [[_num(x) for x in day] for day in pct]}
