# FlowCast — Project Memory

Persistent context for anyone continuing this project: the developer, another engineer, Claude Code or ChatGPT. Read this first. It records **verified facts, decisions and conventions** that are not obvious from the code. Update it at the end of every phase.

Last updated: 2026-11-08 (Phase 5 complete). Current state: Phase 6 (graph deep learning) is next. See [phases.md](phases.md).

## Hard constraints

- **₹0 budget.** Free and open-source only. Never suggest paid APIs, cloud GPUs, datasets or hardware upgrades.
- **Hardware:** Ryzen 7 7840HS, RTX 3050 Laptop 6 GB VRAM, 16 GB RAM, Windows 11, PowerShell, VS Code.
- **Python 3.12** in `.venv`. The system default is 3.14, so don't use it.
- Work **phase by phase**. Stop at the end of each phase with a report. Ask before major architecture changes, destructive operations or scope changes.
- Never fabricate results. Never train or tune on the test set. Never shuffle time-series data.

## Verified data facts (Phase 1–2)

| Fact | Value |
|---|---|
| Source | `load_dataset("witgaw/METR-LA")`; sensor graph in `sensor_graph/` of the same repo |
| Raw format | **Pre-windowed**: one row per (node_id, t0), 12 input + 12 target steps; each reading is repeated up to 24× |
| Canonical series | 34,272 × 207, 2012-03-01 00:00 → 2012-06-27 23:55, regular 5-min grid |
| Units | **mph** (max 70.0). The values are not km/h. |
| Missing | **0.0 = missing** (8.11%). Mostly network-wide outages; the test period has 12.1%. |
| Splits (samples) | train 23,974 · validation 3,425 · test 6,850, chronological |
| Split boundary | The last 12 train and last 12 validation samples have targets in the next split. Use the **owned periods** in `data/processed/metr_la_splits.json`. |
| Sensor mapping | `node_id i` → `adj_mx_mapping.json["sensor_ids"][i]`. This is an assumption the data supports (neighbour correlation 0.436 vs 0.204 for all pairs). |
| Adjacency | 207×207, **directed**, 1,722 non-zeros. Verified: `A[i,j] = exp(-(d(i→j)/2584.5 m)²)`, threshold 0.1. Row i = downstream of i, column i = upstream. Median 7 neighbours each way; 1 sensor isolated. |
| `distances.csv` | Only pairs ≤ 11.9 km apart (11,546 of 42,642 directed pairs). Farther pairs are absent, not zero. |
| Time zone | Naive local clock. No DST shift was found at 2012-03-11. |
| d1 feature | time of day = minutes / 1440, exact |

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Rebuild a canonical (timestamp × sensor) matrix from the windows; build all features from it | The windows are 24× redundant, and building from the matrix gives full control over point-in-time features |
| 2026-11-08 | Owned periods: each timestamp belongs to exactly one split | Prevents targets from crossing split boundaries |
| 2026-11-08 | Keep 0 → NaN at load time (`load_speed`); keep the raw 0 in the stored Parquet | The stored file stays faithful to the source; analysis code works with NaN |
| 2026-11-08 | EDA uses the training period only; validation once for drift; test untouched | Prevents decisions from being tuned on the test set |
| 2026-11-08 | Congestion = speed < 0.6 × per-sensor p85 speed (training data) | Sensors' free-flow speeds range from 36 to 70 mph, so a fixed mph threshold would be wrong |
| 2026-11-08 | Keep the 2,390 low-speed (< 5 mph) readings | Only 7 are isolated one-step dips; the rest are real stop-and-go traffic |
| 2026-11-08 | MIT license, © 2026 Lloyd D'Souza | Developer's choice |
| 2026-11-08 | **Replay clock**: the app forecasts "as of" a time in the test period | The dataset is historical; this also allows forecast-vs-actual views |
| 2026-11-08 | Frontend style: taste-skill `framed-sheet` (light), with **pill controls** | Developer asked for round buttons and no neon. Pills chosen once instead of the generic 8–12 px radius. |
| 2026-11-08 | Data hues `#2a78d6` (forecast) and `#a86f0c` (comparison); congestion = single red ramp `#e2836b → #6e1510`; free flow = neutral grey | Passed the palette validator for colour-vision deficiency, chroma and contrast. Avoids red-green traffic-light colours. |
| 2026-11-08 | Features: 48 columns from the canonical matrix; NaN kept (no imputation), ffill ≤ 3 steps; one row per (t, sensor); rows with targets in another split dropped | LightGBM handles NaN natively; prevents leakage across splits |
| 2026-11-08 | Seasonal features are horizon-specific; each horizon's model uses 39 columns via `feature_columns(cols, h)` | Seasonal lags point at the target's clock time |
| 2026-11-08 | Free-flow and congestion definitions moved to `flowcast/congestion.py` | One definition shared by EDA, features, evaluation and the API |
| 2026-11-08 | Metrics are masked (missing targets excluded); a NaN prediction on a scored row raises; MAPE excludes targets < 5 mph | No model can look better by skipping hard rows or by dividing by near-zero speeds |
| 2026-11-08 | Slices use the **target** time; peak = weekdays 07:00–09:59 and 15:00–18:59; congested = target < 0.6 × free-flow | Peak hours are where the training weekday mean speed dips below ~53 mph |
| 2026-11-08 | Baselines fall back to the historical average when their own input is missing, and the fallback share is reported | Every model scores the same rows |
| 2026-11-08 | Floor to beat = the best baseline per horizon (AR on residuals for MAE/RMSE). Test scores were computed once in Phase 4 | Baselines have nothing to tune, so a test score is a fair floor |
| 2026-11-08 | LightGBM: one direct global model per horizon; L1 objective, 127 leaves, min leaf 500, lr 0.05 | L1 beat every L2 config by ~7% MAE (RMSE ~3–4% worse, still below floor); stacked model was within ~1.6% and needs 4× memory |
| 2026-11-08 | LightGBM checkpoints in `models/checkpoints/lightgbm/` (git-ignored, ~140 MB); predictions in `data/processed/predictions/` | Reproducible from `scripts/train_lightgbm.py` (~60 min) |
| 2026-11-08 | Prediction intervals deferred | Optional in Phase 5; need their own calibration check |
| 2026-11-10 | DCRNN with published settings (hidden 64, 2 layers, 2 diffusion steps, lr 0.01 stepped, scheduled sampling), fp16 autocast, batch 64; one model for 12 steps | Fits 6 GB VRAM (1.8 GB peak); no tuning budget at ~3.7 h per run |
| 2026-11-10 | All models scored through `flowcast/evaluation/rows.py` (same rows, same masked metrics) | A comparison is only fair on identical rows |
| 2026-11-10 | Model differences judged by a 95% day-block bootstrap of the MAE gap; claimed only if the interval excludes 0 | Errors within a day are correlated; one seed cannot show significance by itself |
| 2026-11-10 | DCRNN checkpoint in `models/checkpoints/dcrnn/` (`model.pt` best weights, `last.pt` resumable state; git-ignored) | Training saves every epoch, so a kill loses at most one epoch |
| 2026-11-10 | Backend: Parquet-only, no database | Data is static and ~10 MB; PostgreSQL only when mutable state appears |
| 2026-11-10 | API serves the saved test-period predictions (dense arrays), not live inference | Identical numbers to live inference and to the metrics; no torch/LightGBM in the API; cheap to host |
| 2026-11-10 | API in `src/flowcast/api/`; default model DCRNN; in-range outage times return null, not 422 | One package, shared code; lowest MAE; a replay slider never hits an error |
| 2026-11-10 | Frontend: plain CSS on the §10 tokens, own SVG charts, greyscale OSM tiles; light theme by default | Every shadcn default would be replaced; §10.7 rules fight Recharts; CARTO now needs a key; the dealt style is light |
| 2026-11-10 | Default replay time Tue 12 Jun 2012 16:30 | The evening build-up: 63 sensors congested now, 82 forecast within 30 min |
| 2026-11-10 | Redesign: navy + ivory palette, Inter, dark / light / system themes; `tokens.css` is the source of truth (architect.md §10 colours superseded) | Developer's new visual direction |
| 2026-11-10 | Sidebar lists only screens with real data (Network, Sensors, Models, Patterns, Settings) | No Dashboard / Analytics / Experiments / Reports pages exist; nothing is faked |
| 2026-11-10 | README images are theme-aware `<picture>` pairs; screenshots come from `frontend/scripts/screenshots.mjs` | Real captures in both themes; the script also fails on console errors or overflow |

## Conventions

- **Code:** `src/flowcast/<layer>/`; scripts in `scripts/` orchestrate and contain no logic; type hints throughout; Ruff (line length 110).
- **Tests:** synthetic unit tests always run. Real-data tests are marked `@pytest.mark.data` and skip when the processed files are absent.
- **Numbers in docs** come from committed JSON written by scripts (`docs/eda_summary.json`, later `models/metrics/`). Never type a result by hand.
- **Git:** conventional commits (`feat:`, `test:`, `docs:`, `chore:`). Never commit `data/`, `.venv/`, model checkpoints or `.env`.
- **Figures:** the dataviz reference palette in light mode; static PNGs in `docs/figures/`.

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/prepare_metr_la.py   # rebuild data/processed/ (~33 s with a warm HF cache)
python scripts/run_eda.py           # rebuild EDA figures and docs/eda_summary.json
python scripts/build_features.py    # data/processed/features/{split}.parquet (~34 s, 2.3 GB peak)
python scripts/run_baselines.py     # models/metrics/baselines.json + docs/baseline_results.md (~23 s)
python scripts/tune_lightgbm.py     # validation-only search -> models/metrics/lightgbm_search.json (~35 min)
python scripts/train_lightgbm.py    # final models, test scored once, model card + figures (~60 min)
python scripts/train_dcrnn.py       # DCRNN on GPU, ~134 s/epoch, 100 epochs (~3.7 h); --resume continues
python scripts/compare_models.py    # docs/model_comparison.md + models/metrics/comparison.json
uvicorn flowcast.api.app:app        # API (store loads in ~1 s); docs at /docs
python scripts/benchmark_api.py     # latency over HTTP -> models/metrics/api_latency.json
pytest                              # 91 tests at end of Phase 8
cd frontend; npm run dev            # dashboard on :5173 (needs the API on :8000); npm test, npm run typecheck
ruff check . ; ruff format --check .
```

## Gotchas

- The HF symlink warning on Windows is harmless; turning on Developer Mode silences it.
- pandas 3: use `DatetimeIndex.day_of_week`, not `.dayofweek` (deprecated).
- Peak RSS during preparation (~3.3 GB) is mostly memory-mapped Arrow pages, which the OS can reclaim. Never convert the windowed splits to Pandas.
- The test period has many more missing readings, so always use masked metrics.
- The official windows only reach back 11 steps (`x_t-11`); `speed_lag12` has no official counterpart.
- Feature tables are float32: compare with a tolerance (~1e-4 mph), not `==`.
- The seasonal-naive baseline falls back on 14% of test rows (week-ago readings lost to outages), against 3% in validation.
- Always pass `encoding="utf-8"` to `read_text`/`write_text`: Windows defaults to cp1252 and fails on `→`/`✅` (it once emptied README.md and once crashed the model-card write after a 60-min run).
- Use `.venv/Scripts/python`, not bare `python` (the system Python is 3.14 / a Store stub).
- psutil `rss` is current memory, not peak; on Windows use `memory_info().peak_wset`.
- Print generated reports with `reporting.print_report`: a plain `print` of `→`/`−` crashes on a cp1252 console after the run's files are written.
- DCRNN training MAE rises after epoch ~25 because of scheduled sampling; judge it on validation only.
- Run long GPU training from your own PowerShell window, not through Claude Code (its background reaper killed one run at epoch 45).
- The API store indexes time by row position, so the speed matrix must be a gap-free 5-minute grid (checked at load).
- The API needs `data/processed/` and the saved test predictions; they are git-ignored, so deployment (Phase 10) must ship or rebuild them.
- `vite preview` binds to `localhost` (IPv6), not 127.0.0.1: point tools at http://localhost:4173.
- CARTO basemap tiles now answer "API key required"; the map uses greyscale OSM tiles.
- After any API schema change run `python scripts/export_openapi.py`, then `npm run api:types` in frontend/.
- Validation contains Memorial Day (Mon 2012-05-28). Training has no US federal holidays.
