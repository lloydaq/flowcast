"""Export everything the dashboard reads into static files, for the GitHub Pages demo.

Usage:  python scripts/export_static.py        (after the training scripts have saved test predictions)

The replay is a fixed past, so every answer the API can give is known in advance:

- Small answers (health, sensors, models, metrics, analytics) are written as JSON by calling the real
  API in-process, so they are byte-for-byte what the API returns.
- Forecasts and speeds would be about 55,000 network answers. They ship instead as compact per-day
  binary grids (uint16 hundredths of a mph, 65535 = missing), and frontend/src/lib/staticApi.ts
  rebuilds each answer from them in the browser.
- frontend/.static-samples.json holds real API answers for sampled requests;
  frontend/src/lib/staticApi.test.ts checks that the static rebuild reproduces them exactly.

Writes frontend/.static/ (git-ignored): the public folder of the static build.
"""

from __future__ import annotations

import json
import shutil
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from flowcast import paths
from flowcast.api.app import create_app
from flowcast.api.store import HORIZON_MIN, MODELS, ForecastStore

OUT = paths.PROJECT_ROOT / "frontend" / ".static"
DATA = OUT / "data"
PUBLIC = paths.PROJECT_ROOT / "frontend" / "public"
SAMPLES = paths.PROJECT_ROOT / "frontend" / ".static-samples.json"  # verification only, not deployed
MISSING = 65535
SLOTS = 288  # 5-minute slots per day
SPLITS = ("validation", "test")
SAMPLE_SEED, N_SAMPLES = 0, 40


def encode(a: np.ndarray) -> np.ndarray:
    """mph -> uint16 hundredths, exactly as the API rounds (round(float(x), 2)); NaN -> 65535."""
    out = np.full(a.shape, MISSING, np.uint16)
    ok = ~np.isnan(a)
    out[ok] = np.round(np.round(a[ok].astype(np.float64), 2) * 100).astype(np.uint16)
    return out


def write_json(rel: str, obj) -> None:
    path = DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    store = ForecastStore.load()
    if (
        float(np.nanmax(store.values)) > 600
        or max(float(np.nanmax(g)) for g in store.forecast.values()) > 600
    ):
        raise ValueError("speeds above 600 mph do not fit the uint16 encoding")
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(PUBLIC, OUT)

    with TestClient(create_app(store)) as api:
        get = lambda url: api.get(url).raise_for_status().json()  # noqa: E731

        health = get("/api/health")
        write_json("health.json", health)
        write_json("sensors.json", get("/api/sensors"))
        for sid in store.node_of:
            write_json(f"sensor/{sid}.json", get(f"/api/sensors/{sid}"))
        write_json("models.json", get("/api/models"))
        metrics = get("/api/metrics")
        write_json("metrics.json", metrics)
        for m in metrics["models"]:
            for split in SPLITS:
                for h in HORIZON_MIN:
                    rows = get(f"/api/metrics/sensors?model={m['model']}&horizon={h}&split={split}")
                    write_json(f"sensor-errors/{m['model']}/{split}/{h}.json", rows)
        write_json("daily-profile.json", get("/api/analytics/daily-profile"))
        write_json("weekly-congestion.json", get("/api/analytics/weekly-congestion"))

        # per-day binary grids: speeds from 24 h before the first replay time to the last target time
        first_day = (store.replay[0] - timedelta(hours=24)).normalize()
        last_day = (store.replay[-1] + timedelta(minutes=max(HORIZON_MIN))).normalize()
        days = pd.date_range(first_day, last_day, freq="D")
        speed = pd.DataFrame(store.values, index=store.times)
        for day in days:
            grid = np.full((SLOTS, store.n_nodes), np.nan, np.float32)
            block = speed.loc[day : day + timedelta(minutes=5 * SLOTS - 5)]
            grid[((block.index - day) / timedelta(minutes=5)).astype(int)] = block.to_numpy(np.float32)
            (DATA / "speed").mkdir(parents=True, exist_ok=True)
            encode(grid).tofile(DATA / "speed" / f"{day:%Y-%m-%d}.bin")
        replay = pd.Series(np.arange(len(store.replay)), index=store.replay)
        fc_days = pd.date_range(store.replay[0].normalize(), store.replay[-1].normalize(), freq="D")
        for model, fc in store.forecast.items():
            (DATA / "forecast" / model).mkdir(parents=True, exist_ok=True)
            for day in fc_days:
                grid = np.full((SLOTS, store.n_nodes, len(HORIZON_MIN)), np.nan, np.float32)
                rows = replay.loc[day : day + timedelta(minutes=5 * SLOTS - 5)]
                grid[((rows.index - day) / timedelta(minutes=5)).astype(int)] = fc[rows.to_numpy()]
                encode(grid).tofile(DATA / "forecast" / model / f"{day:%Y-%m-%d}.bin")

        write_json(
            "grid.json",
            {
                "slots_per_day": SLOTS,
                "missing": MISSING,
                "scale": 100,
                "sensor_ids": list(store.node_of),  # row order of every grid = node order
                "free_flow": [float(x) for x in store.free_flow],  # unrounded, as the API compares
                "horizons_min": list(HORIZON_MIN),
                "models": list(MODELS),
                "speed_days": [f"{d:%Y-%m-%d}" for d in days],
                "forecast_days": [f"{d:%Y-%m-%d}" for d in fc_days],
            },
        )

        # real API answers for sampled requests, to prove the static rebuild matches
        rng = np.random.default_rng(SAMPLE_SEED)
        samples = []
        for _ in range(N_SAMPLES):
            t = f"{store.replay[rng.integers(len(store.replay))]:%Y-%m-%dT%H:%M}"
            model = str(rng.choice(list(MODELS)))
            h = int(rng.choice(HORIZON_MIN))
            sid = str(rng.choice(list(store.node_of)))
            hours = int(rng.choice([1, 3, 24]))
            for url in (
                f"/api/forecast?as_of={t}&horizon={h}&model={model}",
                f"/api/forecast/{sid}?as_of={t}&model={model}&history_hours={hours}",
            ):
                samples.append({"url": url, "response": get(url)})
        SAMPLES.write_text(json.dumps(samples), encoding="utf-8")

    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 2**20:.1f} MB of data, {len(samples)} verification samples)")


if __name__ == "__main__":
    main()
