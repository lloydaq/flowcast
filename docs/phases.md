# FlowCast — Phases

The roadmap, one phase at a time. Each phase ends with a report: files changed, verified results, tests actually run, and open issues. Then it stops for review. A phase is **done** only when its exit criteria are met and checked.

Related: [architect.md](architect.md) · [memory.md](memory.md)

| # | Phase | Status | Commit |
|---|---|---|---|
| 0 | Discovery and planning | ✅ done | – |
| 1 | Dataset acquisition and foundation | ✅ done | `736fa89` |
| 2 | Exploratory data analysis | ✅ done | `8665a3b` |
| 3 | Feature engineering | ⏭ next | |
| 4 | Baselines and evaluation framework | planned | |
| 5 | LightGBM | planned | |
| 6 | Graph deep learning (DCRNN) | planned | |
| 7 | Backend (FastAPI) | planned | |
| 8 | Frontend (React) | planned | |
| 9 | Integration and testing | planned | |
| 10 | Deployment and open-source release | planned | |

---

## Phase 0 — Discovery and planning ✅

- Repository inspected: it was empty. Python 3.12.4 was chosen alongside the system's 3.14. Hardware and the ₹0 budget were recorded as constraints.
- Dataset source verified (`witgaw/METR-LA`, ~223 MB Parquet + sensor graph) before downloading.

## Phase 1 — Dataset acquisition and foundation ✅

**Delivered:** `src/flowcast/data/metr_la.py`, `validation.py`, `scripts/prepare_metr_la.py`, 19 tests, [dataset.md](dataset.md), MIT license.

**Verified results:**
- 207 sensors; 34,272 timestamps at exactly 5-minute spacing (2012-03-01 → 06-27); speeds in mph, with 0 meaning missing.
- Splits: 23,974 / 3,425 / 6,850 samples, in chronological order. Every redundant copy of every reading was checked and agrees exactly.
- Missing: 8.11% overall, mostly from 97 network-wide outage blocks. The test period is worse (12.1%).
- `node_id ↔ sensor_id` mapping is supported by the data: graph neighbours correlate at 0.436, against 0.204 for all pairs.
- Peak RAM 3.3 GB, runtime ~33 s.

## Phase 2 — Exploratory data analysis ✅

**Delivered:** `src/flowcast/data/processed.py`, `src/flowcast/analysis/eda.py`, `scripts/run_eda.py`, 9 tests, [eda.md](eda.md) with 11 figures. Training period only, with validation used once for a drift check. The test set has not been looked at.

**Findings that drive Phase 3:**
- Same time last week correlates more strongly than same time yesterday: 0.88 vs 0.67.
- Predictability decays with horizon: median correlation 0.82 at 15 min, 0.57 at 60 min.
- Spatial correlation falls fast with distance: 0.76 under 1 km, 0.23 at 8–12 km.
- Speed changes mostly reach neighbouring sensors within one 5-minute step.
- Missingness depends on time of day.

---

## Phase 3 — Feature engineering ⏭

**Goal:** a reusable, leakage-proof generator of (sensor, as-of time) feature rows with targets at h = 3, 6, 9, 12 steps.

**Planned features** (every one computed from readings ≤ the as-of time *t*):

| Group | Features | Why (EDA evidence) |
|---|---|---|
| Recent lags | speed at t, t−1, t−2, t−3, t−6, t−12 | 5-min autocorrelation 0.995; 1 h 0.83 |
| Rolling | mean, std, min, max over the last 15 / 30 / 60 min (NaN-aware) | Smooths sensor noise |
| Trend | speed(t) − speed(t−3), speed(t) − speed(t−6) | Is speed dropping into a congestion onset? |
| Seasonal | speed at t+h−288 (same time yesterday) and t+h−2016 (same time last week); also the mean of the last 4 same-weekday values | Weekly correlation 0.88 > daily 0.67. These offsets point to the past only because h ≤ 12 is far smaller than 288. |
| Calendar | time of day (sin/cos), day of week, weekend flag | Two weekday troughs; Friday is different |
| Sensor | node id (categorical), free-flow speed and congestion rate from the training period | Sensors behave very differently (congested 0–49.5% of the time) |
| Spatial | adjacency-weighted mean of neighbours' current and 15-min-mean speeds; the slowest neighbour's speed | Strong correlation within 1–3 km; propagation within one step |
| Missingness | is-missing flags for the current and lag-1 readings; count missing in the last hour | Missingness is informative |

**Rules:** no centred windows; no forward-fill across gaps over 15 min; targets that are missing are masked. A row is kept only if all of its targets fall in one split's owned period.

**Exit criteria:**
1. A feature generator in `src/flowcast/features/` that is deterministic and documented.
2. A **leakage test**: overwrite every reading after *t* with random values, then assert every feature for *t* is unchanged.
3. Alignment tests: feature rows at known timestamps equal hand-computed values.
4. Feature tables written to Parquet as float32, with peak RAM measured and under 8 GB.
5. `docs/features.md`: the feature list, the reason for each feature, and table sizes.

## Phase 4 — Baselines and evaluation framework

- Masked MAE / RMSE / MAPE per horizon. MAPE excludes targets under 5 mph, and the threshold is stated.
- Slices: peak, off-peak, congested, per sensor, by hour.
- Baselines: persistence; historical average by (sensor, day of week, slot) from training data; one statistical model.
- Reported on validation, plus one test-period score per baseline, recorded as the floor to beat.
- **Exit:** a results table built from committed JSON, plus `docs/experiments.md` started.

## Phase 5 — LightGBM

- One global model per horizon, on CPU. Compared against a multi-output alternative on validation.
- Hyperparameter search on validation only, with early stopping on validation. The test period is scored once.
- Feature importance (gain plus permutation on validation), error analysis by slice, and the residual distribution.
- Optional: calibrated intervals (quantile objectives or conformal). They are drawn in the UI only if calibration holds on validation.
- Model serialisation and a model card (data period, features, metrics, git commit).
- **Exit:** it beats every baseline on validation at all horizons, or the doc says honestly where it doesn't.

## Phase 6 — Graph deep learning

- DCRNN on the published directed adjacency (no invented connectivity). Batch size and hidden size sized for 6 GB VRAM, with mixed precision.
- Same splits, same masked metrics, same test period as LightGBM.
- **Exit:** a side-by-side comparison table. Any improvement is claimed only if the results show it.

## Phase 7 — Backend

- FastAPI with typed schemas, the replay-clock forecast service, sensors / forecast / models / metrics / analytics endpoints, and in-memory caching.
- **Decision needed before starting:** PostgreSQL or Parquet-only ([architect.md](architect.md) §6).
- **Exit:** every endpoint tested, including validation errors; OpenAPI docs; p95 latency of the network forecast measured.

## Phase 8 — Frontend

- Built to the design system in [architect.md](architect.md) §10: framed sheet, pill controls, no neon, motion spec, accessibility rules.
- Screens: Network, Sensor, Models, Patterns. Stretch goal: the corridor time-space diagram.
- **Exit:** the taste-skill inspector in app mode at 1440 / 768 / 390 with no FAILs; reduced-motion checked; Lighthouse accessibility ≥ 95.

## Phase 9 — Integration and testing

- End-to-end flows (Playwright), error and empty states against a real backend, a performance check on the 207-marker map, and a security review (CORS, input validation, no filesystem exposure).

## Phase 10 — Deployment and release

- Reproducible setup docs, a GitHub Actions CI, and a free hosting option (static precomputed demo vs free-tier backend: decide here).
- README with real results, screenshots, a short demo GIF, and a v1.0 GitHub release.
