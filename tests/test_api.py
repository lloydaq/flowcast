"""API tests on a small synthetic store (always run), plus one check against the real saved predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flowcast import paths
from flowcast.api.app import create_app
from flowcast.api.store import ForecastStore
from flowcast.data.metr_la import FREQ

SENSORS = ["100", "200", "300"]
TRAIN = pd.date_range(
    "2012-05-28 04:00", periods=7 * 288, freq=FREQ
)  # one week up to the test start; Mon first
TEST = pd.date_range("2012-06-04 04:00", periods=60, freq=FREQ)  # replay = first 40 as-of times
AS_OF = TEST[10]  # 2012-06-04 04:50


def make_store() -> ForecastStore:
    rng = np.random.default_rng(0)
    times = TRAIN.append(TEST)
    speed = pd.DataFrame(60 + rng.normal(0, 3, (len(times), 3)), index=times, columns=SENSORS)
    speed.iloc[len(TRAIN) // 2 : len(TRAIN), 2] = 20.0  # sensor 300 congested in the second half of training
    speed.loc[TEST[13], "100"] = np.nan  # missing reading at AS_OF + 15 min
    speed.loc[TEST[3:7], "200"] = 10.0  # a slow spell in sensor 200's history
    sensors = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "sensor_id": SENSORS,
            "latitude": [34.1, 34.2, 34.3],
            "longitude": [-118.1, -118.2, -118.3],
            "missing_pct": [1.0, 2.0, 3.0],
        }
    )
    preds = {}
    for name, offset in (("dcrnn", 0.0), ("lightgbm", 100.0)):
        t = np.repeat(TEST[:40], 3)
        node = np.tile([0, 1, 2], 40)
        frame = pd.DataFrame({"timestamp": t, "node_id": node.astype(np.int16)})
        for h in (3, 6, 9, 12):
            # forecast encodes (model, as-of position, node, horizon) so lookups can be checked exactly
            frame[f"pred_h{h}"] = (offset + np.repeat(np.arange(40), 3) + node * 0.1 + h * 0.001).astype(
                np.float32
            )
        frame = frame[~((frame["timestamp"] == TEST[20]) & (frame["node_id"] == 1))]  # dropped row (outage)
        frame.loc[frame["node_id"] == 2, "pred_h3"] = 5.0  # sensor 300 forecast to jam at 15 min
        preds[name] = frame.reset_index(drop=True)
    by_h = {
        f"h{h}": {
            k: {"mae": 1.0 + h, "rmse": 2.0, "mape": 3.0, "n": 10}
            for k in ("all", "peak", "off_peak", "congested", "free_flowing")
        }
        | {"mae_by_hour": {"0": 1.5}, "mae_by_sensor": {"0": 1.0}}
        for h in (3, 6, 9, 12)
    }
    results = {name: {"validation": by_h, "test": by_h} for name in ("persistence", "LightGBM", "DCRNN")}
    gap = {"mae_gap": -0.1, "ci95": [-0.2, 0.0], "days": 2, "days_dcrnn_better": 1}
    comparison = {"mae_gap_dcrnn_minus_lightgbm": {"test": {f"h{h}": gap for h in (3, 6, 9, 12)}}}
    return ForecastStore(speed, speed.loc[TRAIN], sensors, preds, results, comparison)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(make_store())) as c:
        yield c


def get(client, url, status=200):
    r = client.get(url)
    assert r.status_code == status, r.text
    return r.json()


def test_health(client):
    h = get(client, "/api/health")
    assert h["models"] == ["dcrnn", "lightgbm"] and h["default_model"] == "dcrnn"
    assert h["replay_start"] == "2012-06-04T04:00:00" and h["replay_end"] == "2012-06-04T07:15:00"
    assert h["horizons_min"] == [15, 30, 45, 60] and h["sensors"] == 3


def test_sensors(client):
    s = get(client, "/api/sensors")
    assert [x["sensor_id"] for x in s] == SENSORS
    assert s[0]["latitude"] == 34.1 and s[2]["missing_pct"] == 3.0
    assert s[2]["train_congested_pct"] == pytest.approx(50.0, abs=0.1) and s[0]["train_congested_pct"] == 0.0
    d = get(client, "/api/sensors/300")
    assert d["train_congested_pct"] == pytest.approx(50.0, abs=0.1)
    assert len(d["weekday_hourly_speed"]) == 24
    assert get(client, "/api/sensors/999", 404)["detail"] == "unknown sensor '999'"


def test_network_forecast_values(client):
    f = get(client, f"/api/forecast?as_of={AS_OF:%Y-%m-%dT%H:%M}&horizon=60")
    assert f["model"] == "dcrnn" and f["target_time"] == "2012-06-04T05:50:00"
    by = {x["sensor_id"]: x for x in f["sensors"]}
    assert by["200"]["forecast"] == pytest.approx(10 + 0.1 + 0.012, abs=0.01)
    store = client.app.state.store
    assert by["100"]["actual"] == pytest.approx(store.speed.loc[TEST[22], "100"], abs=0.01)
    assert by["100"]["speed_now"] == pytest.approx(store.speed.loc[AS_OF, "100"], abs=0.01)
    assert by["100"]["free_flow"] > 60 and by["100"]["congested"] is True  # 10 mph < 0.6 x free flow


def test_network_forecast_nulls_and_congestion(client):
    f = get(client, f"/api/forecast?as_of={AS_OF:%Y-%m-%dT%H:%M}&horizon=15")
    by = {x["sensor_id"]: x for x in f["sensors"]}
    assert by["100"]["actual"] is None  # missing reading at the target time
    assert by["300"]["forecast"] == 5.0 and by["300"]["congested"] is True
    outage = get(client, f"/api/forecast?as_of={TEST[20]:%Y-%m-%dT%H:%M}&horizon=15")
    point = next(x for x in outage["sensors"] if x["sensor_id"] == "200")
    assert point["forecast"] is None and point["congested"] is None


def test_model_choice(client):
    f = get(client, f"/api/forecast?as_of={AS_OF:%Y-%m-%dT%H:%M}&horizon=30&model=lightgbm")
    assert f["model"] == "lightgbm" and f["sensors"][0]["forecast"] == pytest.approx(110.006, abs=0.01)
    assert f["sensors"][0]["congested"] is False


def test_sensor_forecast(client):
    f = get(client, f"/api/forecast/200?as_of={AS_OF:%Y-%m-%dT%H:%M}&history_hours=1")
    assert len(f["history"]) == 13  # 12 steps before as_of, plus as_of
    assert f["history"][-1]["time"] == "2012-06-04T04:50:00"
    assert [x["horizon_min"] for x in f["forecasts"]] == [15, 30, 45, 60]
    assert f["forecasts"][3]["target_time"] == "2012-06-04T05:50:00"
    assert len(f["actual_after"]) == 12 and f["actual_after"][0]["time"] == "2012-06-04T04:55:00"
    assert {x["speed"] for x in f["history"] if "04:15" <= x["time"][11:16] <= "04:30"} == {10.0}


@pytest.mark.parametrize(
    "query, message",
    [
        ("as_of=2012-06-04T04:52&horizon=15", "5-minute grid"),
        ("as_of=2012-06-05T04:50&horizon=15", "between 2012-06-04 04:00:00 and 2012-06-04 07:15:00"),
        ("as_of=2012-06-04T04:50Z&horizon=15", "without a timezone"),
        ("as_of=2012-06-04T04:50&horizon=20", "horizon must be one of [15, 30, 45, 60]"),
        ("as_of=2012-06-04T04:50&horizon=15&model=arima", "unknown model 'arima'"),
    ],
)
def test_invalid_queries_are_422_with_a_reason(client, query, message):
    assert message in get(client, f"/api/forecast?{query}", 422)["detail"]


@pytest.mark.parametrize(
    "url",
    [
        "/api/forecast?horizon=15",  # as_of missing
        "/api/forecast?as_of=yesterday&horizon=15",
        "/api/forecast?as_of=2012-06-04T04:50&horizon=soon",
        "/api/forecast/100?as_of=2012-06-04T04:50&history_hours=0",
        "/api/forecast/100?as_of=2012-06-04T04:50&history_hours=25",
    ],
)
def test_malformed_parameters_are_422(client, url):
    get(client, url, 422)


def test_unknown_sensor_forecast_is_404(client):
    get(client, f"/api/forecast/999?as_of={AS_OF:%Y-%m-%dT%H:%M}", 404)


def test_models_and_metrics(client):
    m = get(client, "/api/models")
    assert [(x["name"], x["label"], x["default"]) for x in m] == [
        ("dcrnn", "DCRNN", True),
        ("lightgbm", "LightGBM", False),
    ]
    assert m[0]["test_mae"] == {"15": 4.0, "30": 7.0, "45": 10.0, "60": 13.0}
    r = get(client, "/api/metrics")
    assert [x["model"] for x in r["models"]] == ["persistence", "LightGBM", "DCRNN"]
    assert r["models"][0]["test"]["60"]["congested"]["mae"] == 13.0
    assert "mae_by_sensor" not in r["models"][0]["test"]["60"]
    assert r["dcrnn_minus_lightgbm"]["test"]["15"]["ci95"] == [-0.2, 0.0]


def test_sensor_errors(client):
    rows = get(client, "/api/metrics/sensors?model=DCRNN&horizon=15")
    assert [r["sensor_id"] for r in rows] == ["100"]  # the synthetic results only score node 0
    assert get(client, "/api/metrics/sensors?model=GPT&horizon=15", 422)["detail"].startswith(
        "unknown evaluated"
    )
    assert "split" in get(client, "/api/metrics/sensors?model=DCRNN&horizon=15&split=train", 422)["detail"]
    assert "horizon" in get(client, "/api/metrics/sensors?model=DCRNN&horizon=10", 422)["detail"]


def test_analytics(client):
    p = get(client, "/api/analytics/daily-profile")
    assert p["slot_minutes"] == 5 and len(p["weekday"]) == len(p["weekend"]) == 288
    w = get(client, "/api/analytics/weekly-congestion")["rate_pct"]
    assert len(w) == 7 and all(len(day) == 24 for day in w)
    assert w[0][0] == pytest.approx(100 / 3, abs=0.1)  # Monday 00:00 (the 2nd Monday): 300 congested, 2 not


def test_network_forecast_is_cached(client):
    store = client.app.state.store
    url = f"/api/forecast?as_of={TEST[30]:%Y-%m-%dT%H:%M}&horizon=45"
    get(client, url)
    hits = store.network.cache_info().hits
    get(client, url)
    assert store.network.cache_info().hits == hits + 1


def test_cors_allows_only_the_frontend(client):
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_openapi_lists_every_endpoint(client):
    spec = get(client, "/openapi.json")
    assert set(spec["paths"]) == {
        "/api/health",
        "/api/sensors",
        "/api/sensors/{sensor_id}",
        "/api/forecast",
        "/api/forecast/{sensor_id}",
        "/api/models",
        "/api/metrics",
        "/api/metrics/sensors",
        "/api/analytics/daily-profile",
        "/api/analytics/weekly-congestion",
    }


def test_store_rejects_unregistered_models_and_gappy_timelines():
    store = make_store()
    with pytest.raises(ValueError, match="registered models"):
        ForecastStore(store.speed, store.train_speed, store.sensors, {"arima": pd.DataFrame()})
    gappy = store.speed.drop(TEST[5])
    preds = {
        "dcrnn": pd.DataFrame(
            {"timestamp": [AS_OF], "node_id": [0], **{f"pred_h{h}": [1.0] for h in (3, 6, 9, 12)}}
        )
    }
    with pytest.raises(ValueError, match="gap-free"):
        ForecastStore(gappy, store.train_speed, store.sensors, preds)


@pytest.mark.data
@pytest.mark.skipif(
    not (paths.PREDICTIONS_DIR / "dcrnn_test.parquet").exists(), reason="run the training scripts first"
)
def test_real_store_serves_the_saved_predictions():
    store = ForecastStore.load()
    with TestClient(create_app(store)) as c:
        for model in ("dcrnn", "lightgbm"):
            saved = pd.read_parquet(paths.PREDICTIONS_DIR / f"{model}_test.parquet")
            row = saved.iloc[123_456]
            sid = store.sensors["sensor_id"].iat[int(row["node_id"])]
            f = get(c, f"/api/forecast/{sid}?as_of={row['timestamp']:%Y-%m-%dT%H:%M}&model={model}")
            got = [x["forecast"] for x in f["forecasts"]]
            assert got == pytest.approx([row[f"pred_h{h}"] for h in (3, 6, 9, 12)], abs=0.006)
    assert store.replay[0] == pd.Timestamp("2012-06-04 04:10")
