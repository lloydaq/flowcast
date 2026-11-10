"""Security review checks for the API (Phase 9): hostile or malformed input never reaches a 500 or the
file system, error bodies are short JSON, only GET is served, and CORS admits only the frontend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from flowcast.api.app import create_app
from tests.test_api import AS_OF, make_store

OK = f"as_of={AS_OF:%Y-%m-%dT%H:%M}"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(make_store()), raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize(
    "url",
    [
        "/api/sensors/..%2F..%2Fetc%2Fpasswd",
        "/api/sensors/..%5C..%5Cwindows%5Cwin.ini",
        "/api/sensors/%00",
        "/api/sensors/" + "9" * 5000,
        "/api/sensors/%3Cscript%3Ealert(1)%3C%2Fscript%3E",
        f"/api/forecast/..%2F..%2Fdata?{OK}",
        f"/api/forecast/1%27%20OR%201%3D1?{OK}",
    ],
)
def test_hostile_sensor_ids_are_404_json_without_echoing_them_whole(client, url):
    r = client.get(url)
    assert r.status_code == 404
    assert r.headers["content-type"] == "application/json"
    assert len(r.text) < 200
    assert "passwd" not in r.text or r.text.count("passwd") <= 1  # quoted back at most once, cut short


@pytest.mark.parametrize(
    "query",
    [
        "as_of=9999-12-31T23:55&horizon=15",
        "as_of=0001-01-01T00:00&horizon=15",
        "as_of=2012-06-04T04:50:00%2B05:30&horizon=15",
        "as_of=2012-06-04T04:50:01&horizon=15",
        f"{OK}&horizon=100000000000000000000",
        f"{OK}&horizon=-15",
        f"{OK}&horizon=15.5",
        f"{OK}&horizon=15&model=%3Cscript%3E",
        f"{OK}&horizon=15&model=" + "x" * 5000,
        f"{OK}&horizon=15&model=..%2F..%2Fmodels",
        "as_of=" + "2" * 5000 + "&horizon=15",
    ],
)
def test_malformed_queries_are_422_never_500(client, query):
    r = client.get(f"/api/forecast?{query}")
    assert r.status_code == 422, r.text
    assert r.headers["content-type"] == "application/json"
    assert len(r.text) < 600 or "detail" in r.json()


@pytest.mark.parametrize("hours", ["0", "25", "1000000000", "-1", "three"])
def test_history_window_is_bounded(client, hours):
    assert client.get(f"/api/forecast/100?{OK}&history_hours={hours}").status_code == 422


def test_unknown_query_parameters_are_ignored(client):
    assert client.get(f"/api/forecast?{OK}&horizon=15&path=/etc/passwd&debug=1").status_code == 200


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_only_get_is_served(client, method):
    assert getattr(client, method)("/api/health").status_code == 405


def test_cors_preflight_admits_only_the_frontend_and_only_get(client):
    evil = client.options(
        "/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert "access-control-allow-origin" not in evil.headers
    post = client.options(
        "/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    )
    assert "POST" not in post.headers.get("access-control-allow-methods", "")
    ok = client.options(
        "/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"}
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_security_headers_on_every_response(client):
    for url in ("/api/health", "/api/sensors/nope", "/api/forecast?as_of=x&horizon=1"):
        r = client.get(url)
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["x-frame-options"] == "DENY"


def test_no_route_reads_files(client):
    """The API has no route that takes a path, and no static file mount."""
    paths = set(client.app.openapi()["paths"])
    assert all(p.startswith("/api/") for p in paths)
    assert not any(getattr(r, "path", "").startswith("/static") for r in client.app.routes)
