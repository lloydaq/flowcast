"""FlowCast HTTP API.

Run:   uvicorn flowcast.api.app:app            (docs at http://127.0.0.1:8000/docs)

Read-only. Every input is validated: as-of times must be on the 5-minute grid inside the test period,
sensors and models must exist in the registry. No endpoint accepts a path. CORS is limited to the
frontend origins in FLOWCAST_CORS_ORIGINS (comma-separated; default the Vite dev server).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from flowcast.api import schemas
from flowcast.api.store import (
    HORIZON_MIN,
    MODELS,
    STEP_MIN,
    ForecastStore,
    InvalidQueryError,
    SensorNotFoundError,
)
from flowcast.features.build import HORIZONS

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
ERRORS = {404: {"model": schemas.Problem}, 422: {"model": schemas.Problem}}

AsOf = Annotated[
    datetime, Query(description="Replay time, local LA, on the 5-minute grid", examples=["2012-06-05T08:00"])
]
ModelName = Annotated[str | None, Query(description=f"One of {list(MODELS)}; default {next(iter(MODELS))}")]


def create_app(store: ForecastStore | None = None) -> FastAPI:
    """``store`` is injected by tests; otherwise it is loaded from disk once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store if store is not None else ForecastStore.load()
        yield

    app = FastAPI(
        title="FlowCast API",
        version="0.1.0",
        description="Replay-clock traffic speed forecasts for 207 METR-LA sensors (test period, June 2012).",
        lifespan=lifespan,
    )
    origins = [
        o.strip() for o in os.environ.get("FLOWCAST_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()
    ]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])

    @app.exception_handler(SensorNotFoundError)
    async def sensor_not_found(request: Request, exc: SensorNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(InvalidQueryError)
    async def invalid_query(request: Request, exc: InvalidQueryError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def st(request: Request) -> ForecastStore:
        return request.app.state.store

    @app.get("/api/health", response_model=schemas.Health, tags=["meta"])
    def health(request: Request) -> dict:
        s = st(request)
        return {
            "status": "ok",
            "models": list(s.forecast),
            "default_model": s.model(None),
            "replay_start": s.replay[0],
            "replay_end": s.replay[-1],
            "step_min": STEP_MIN,
            "horizons_min": list(HORIZON_MIN),
            "sensors": s.n_nodes,
        }

    @app.get("/api/sensors", response_model=list[schemas.Sensor], tags=["sensors"])
    def sensors(request: Request) -> list[dict]:
        return st(request).sensor_list

    @app.get(
        "/api/sensors/{sensor_id}", response_model=schemas.SensorDetail, responses=ERRORS, tags=["sensors"]
    )
    def sensor(request: Request, sensor_id: str) -> dict:
        s = st(request)
        return s.sensor_detail(s.node(sensor_id))

    @app.get("/api/forecast", response_model=schemas.NetworkForecast, responses=ERRORS, tags=["forecast"])
    def network_forecast(
        request: Request,
        as_of: AsOf,
        horizon: Annotated[int, Query(description=f"Minutes ahead: one of {list(HORIZON_MIN)}")],
        model: ModelName = None,
    ) -> dict:
        s = st(request)
        return s.network(s.model(model), s.replay_row(pd.Timestamp(as_of)), s.horizon_index(horizon))

    @app.get(
        "/api/forecast/{sensor_id}",
        response_model=schemas.SensorForecast,
        responses=ERRORS,
        tags=["forecast"],
    )
    def sensor_forecast(
        request: Request,
        sensor_id: str,
        as_of: AsOf,
        history_hours: Annotated[int, Query(ge=1, le=24, description="Hours of readings before as_of")] = 3,
        model: ModelName = None,
    ) -> dict:
        s = st(request)
        node, name = s.node(sensor_id), s.model(model)
        steps = history_hours * 60 // STEP_MIN
        return s.sensor_forecast(name, s.replay_row(pd.Timestamp(as_of)), node, steps)

    @app.get("/api/models", response_model=list[schemas.ModelInfo], tags=["models"])
    def models(request: Request) -> list[dict]:
        s = st(request)
        default = s.model(None)
        out = []
        for name in s.forecast:
            label, description = MODELS[name]
            test = s.results.get(label, {}).get("test", {})
            out.append(
                {
                    "name": name,
                    "label": label,
                    "description": description,
                    "default": name == default,
                    "test_mae": {
                        m: round(test[f"h{h}"]["all"]["mae"], 3)
                        for m, h in zip(HORIZON_MIN, HORIZONS, strict=True)
                        if f"h{h}" in test
                    },
                }
            )
        return out

    @app.get("/api/metrics", response_model=schemas.Metrics, tags=["models"])
    def metrics(request: Request) -> dict:
        s = st(request)

        def by_minutes(split: dict) -> dict:
            return {m: split[f"h{h}"] for m, h in zip(HORIZON_MIN, HORIZONS, strict=True)}

        gap = s.comparison.get("mae_gap_dcrnn_minus_lightgbm", {})
        return {
            "models": [
                {"model": name, "validation": by_minutes(r["validation"]), "test": by_minutes(r["test"])}
                for name, r in s.results.items()
            ],
            "dcrnn_minus_lightgbm": {split: by_minutes(g) for split, g in gap.items()},
        }

    @app.get(
        "/api/metrics/sensors", response_model=list[schemas.SensorError], responses=ERRORS, tags=["models"]
    )
    def sensor_errors(
        request: Request,
        model: Annotated[str, Query(description="An evaluated model's name as listed by /api/metrics")],
        horizon: Annotated[int, Query(description=f"Minutes ahead: one of {list(HORIZON_MIN)}")],
        split: Annotated[str, Query(description="validation or test")] = "test",
    ) -> list[dict]:
        """Masked MAE per sensor, worst first."""
        return st(request).sensor_errors(model, split, horizon)

    @app.get("/api/analytics/daily-profile", response_model=schemas.DailyProfile, tags=["analytics"])
    def daily_profile(request: Request) -> dict:
        return st(request).daily_profile

    @app.get("/api/analytics/weekly-congestion", response_model=schemas.WeeklyCongestion, tags=["analytics"])
    def weekly_congestion(request: Request) -> dict:
        return st(request).weekly_congestion

    return app


app = create_app()
