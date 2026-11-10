"""Measure API latency over real HTTP (uvicorn on 127.0.0.1, one worker, this machine).

Usage:  python scripts/benchmark_api.py     (after the training scripts have saved test predictions)

Requests go through the full stack (HTTP, validation, JSON) over loopback, so network time to a
remote client is excluded. "Cold" requests use as-of times not asked before (cache misses); "warm"
repeats them (cache hits). Writes models/metrics/api_latency.json.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import numpy as np
import uvicorn

from flowcast import paths
from flowcast.api.app import create_app
from flowcast.api.store import HORIZON_MIN, ForecastStore
from flowcast.reporting import git_commit, peak_memory_mb

N, SEED, PORT = 300, 0, 8765
OUT = paths.METRICS_DIR / "api_latency.json"


def timed(client: httpx.Client, urls: list[str]) -> tuple[list[float], int]:
    ms, size = [], 0
    for u in urls:
        t0 = time.perf_counter()
        r = client.get(u)
        ms.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        size = len(r.content)
    return ms, size


def summary(ms: list[float], size: int) -> dict:
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {
        "requests": len(ms),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "response_kb": round(size / 1024, 1),
    }


def main() -> None:
    t0 = time.perf_counter()
    store = ForecastStore.load()
    startup_s = time.perf_counter() - t0
    server = uvicorn.Server(
        uvicorn.Config(create_app(store), host="127.0.0.1", port=PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(store.replay), N, replace=False)
    times = [f"{store.replay[i]:%Y-%m-%dT%H:%M}" for i in picks]
    sensors = rng.choice(store.sensors["sensor_id"].to_numpy(), N)
    horizons = rng.choice(HORIZON_MIN, N)
    network = [f"/api/forecast?as_of={t}&horizon={h}" for t, h in zip(times, horizons, strict=True)]
    sensor = [f"/api/forecast/{s}?as_of={t}" for s, t in zip(sensors, times, strict=True)]

    with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=10) as client:
        timed(client, ["/api/health"] * 20)  # warm up connections and code paths
        results = {
            "network_forecast_cold": summary(*timed(client, network)),
            "network_forecast_warm": summary(*timed(client, network)),
            "sensor_forecast": summary(*timed(client, sensor)),
        }
    server.should_exit = True
    thread.join()

    payload = {
        "git_commit": git_commit(),
        "setup": "uvicorn, 1 worker, HTTP over loopback on the development laptop (Ryzen 7 7840HS); "
        "network time to a remote client excluded",
        "store_load_s": round(startup_s, 1),
        "peak_rss_mb": peak_memory_mb(),
        **results,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
