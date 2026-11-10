"""Response models. Speeds are mph; ``None`` means a missing reading or no forecast at that time."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    models: list[str]
    default_model: str
    replay_start: datetime = Field(description="First as-of time the replay clock serves")
    replay_end: datetime = Field(description="Last as-of time the replay clock serves")
    step_min: int
    horizons_min: list[int]
    sensors: int


class Sensor(BaseModel):
    sensor_id: str
    latitude: float
    longitude: float
    free_flow: float | None = Field(description="85th-percentile training speed, mph")
    missing_pct: float


class SensorDetail(Sensor):
    train_mean_speed: float | None
    train_congested_pct: float | None
    weekday_hourly_speed: list[float | None] = Field(description="Training weekday mean speed, hours 0–23")


class NetworkPoint(BaseModel):
    sensor_id: str
    speed_now: float | None
    forecast: float | None
    actual: float | None = Field(description="Speed observed at target_time (replay only)")
    free_flow: float | None
    congested: bool | None = Field(description="forecast < 0.6 × free flow")


class NetworkForecast(BaseModel):
    model: str
    as_of: datetime
    horizon_min: int
    target_time: datetime
    sensors: list[NetworkPoint]


class Reading(BaseModel):
    time: datetime
    speed: float | None


class HorizonForecast(BaseModel):
    horizon_min: int
    target_time: datetime
    forecast: float | None
    actual: float | None


class SensorForecast(BaseModel):
    model: str
    sensor_id: str
    as_of: datetime
    free_flow: float | None
    history: list[Reading]
    forecasts: list[HorizonForecast]
    actual_after: list[Reading] = Field(description="Readings in the hour after as_of (replay only)")


class ModelInfo(BaseModel):
    name: str
    label: str
    description: str
    default: bool
    test_mae: dict[int, float] = Field(description="Test-period masked MAE by horizon (minutes)")


class Scores(BaseModel):
    mae: float
    rmse: float
    mape: float
    n: int


class HorizonScores(BaseModel):
    all: Scores
    peak: Scores
    off_peak: Scores
    congested: Scores
    free_flowing: Scores
    mae_by_hour: dict[int, float]


class ModelScores(BaseModel):
    model: str
    validation: dict[int, HorizonScores]
    test: dict[int, HorizonScores]


class Gap(BaseModel):
    mae_gap: float
    ci95: list[float]
    days: int
    days_dcrnn_better: int


class Metrics(BaseModel):
    models: list[ModelScores] = Field(description="Every model, weakest baseline first")
    dcrnn_minus_lightgbm: dict[str, dict[int, Gap]] = Field(
        description="MAE gap by split and horizon with a 95% day-block bootstrap interval"
    )


class DailyProfile(BaseModel):
    slot_minutes: int
    weekday: list[float | None]
    weekend: list[float | None]


class WeeklyCongestion(BaseModel):
    rate_pct: list[list[float | None]] = Field(description="7 days (Mon first) × 24 hours, training period")


class Problem(BaseModel):
    detail: str
