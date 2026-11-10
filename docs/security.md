# Security review (Phase 9)

Scope: the FastAPI backend (`src/flowcast/api/`), the React frontend (`frontend/`) and their dependencies, as of 2026-11-10. FlowCast is read-only: no accounts, no user data, no writes, no secrets. The review asks what an attacker could still do with it: read files, crash the service, use it to attack a viewer, or exploit a dependency.

Every check below is automated and runs with the test suites, except the two dependency audits, which are run by hand (commands at the end).

## Findings

| Area | Result | Evidence |
|---|---|---|
| File-system exposure | None. No route takes a path, there is no static-file mount, and the store reads only fixed project paths at startup. Path-traversal sensor IDs (`../../etc/passwd`, `..\..\windows\win.ini`, `%00`) return 404. | `tests/test_api_security.py`: `test_hostile_sensor_ids…`, `test_no_route_reads_files` |
| Input validation | Every parameter is checked: `as_of` (5-minute grid, replay range, no time zone), `horizon` (15/30/45/60), `model` and `split` (registry), `history_hours` (1–24), sensor ID (registry). Malformed, huge, negative, fractional, far-future and injection-style values all return 422, never 500. Unknown parameters are ignored. | `test_malformed_queries_are_422_never_500` (11 cases), `test_history_window_is_bounded`, plus the 25 tests in `tests/test_api.py` |
| Reflected input | Error messages quote the bad value back so the user can see what was wrong. **Fixed in this review:** the quote is now cut to 40 characters, so a 5,000-character input is not echoed whole. Responses are JSON with `nosniff`, so a quoted `<script>` is never rendered as HTML. | `_echo` in `store.py`; `test_hostile_sensor_ids…` (body under 200 bytes) |
| HTTP methods | Only GET. POST, PUT, PATCH and DELETE return 405. | `test_only_get_is_served` |
| CORS | Only the configured frontend origins (`FLOWCAST_CORS_ORIGINS`, default the Vite dev server) and only GET. A preflight from any other origin gets no `Access-Control-Allow-Origin`. | `test_cors_preflight…`, `test_cors_allows_only_the_frontend` |
| Response headers | **Added in this review:** `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY` on every API response, errors included. | `test_security_headers_on_every_response` |
| Error detail | Unexpected errors return Starlette's plain "Internal Server Error", with no stack trace in the response. Expected errors carry one sentence. | FastAPI default; no debug mode is enabled anywhere |
| Frontend injection | No `dangerouslySetInnerHTML`, `innerHTML`, `eval` or `new Function` in the source. All API text is rendered by React, which escapes it; map tooltips and popups are React children too. The only HTML string is the map's fixed attribution. | grep of `frontend/src` |
| Third-party requests | The browser contacts only the app's own origin and `tile.openstreetmap.org` (map tiles). Fonts are self-hosted. No analytics or trackers. | grep of `frontend/src` and `index.html` |
| Browser storage | One `localStorage` key, the theme choice. Nothing personal is stored. | `frontend/src/lib/theme.ts` |
| Denial of service | Every response is bounded: the largest is the network forecast (about 22 KB, 207 sensors) and the longest history is 24 hours of one sensor. Results are cached in memory. Measured network-forecast latency is 4.7 ms at p95. There is no rate limiting; that belongs to the host (Phase 10). | `models/metrics/api_latency.json` |
| JavaScript dependencies | `npm audit`: 0 vulnerabilities (production and development). | run on 2026-11-10 |
| Python dependencies | `pip-audit`: no known vulnerabilities, after upgrading the virtual environment's `setuptools` from 78.1 (three advisories) to 84.1 or later. `setuptools` is a build tool, not used by the running API. `torch` (a CUDA build) cannot be audited from PyPI; it is used only to train DCRNN, never by the API. | run on 2026-11-10 |

## Not covered, for Phase 10 (deployment)

- **Headers for the frontend host:** a Content-Security-Policy (`default-src 'self'; img-src 'self' https://tile.openstreetmap.org data:; style-src 'self' 'unsafe-inline'`), HSTS, and the same `nosniff` and framing headers. These are set by whichever static host serves the build.
- **Rate limiting and HTTPS** for the public API, at the host or a proxy.
- **OpenAPI docs** (`/docs`) load Swagger UI from a CDN. Fine for a portfolio demo; disable `docs_url` if the API should not advertise itself.
- **OpenStreetMap tile policy:** a busy public demo should use a keyed tile provider or self-hosted tiles.

## Re-running the checks

```powershell
pytest tests/test_api_security.py tests/test_api.py
cd frontend; npm audit; cd ..
pip install pip-audit; pip-audit --skip-editable
```
