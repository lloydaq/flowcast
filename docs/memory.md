# FlowCast — Project Memory

Persistent context for anyone continuing this project: the developer, another engineer, Claude Code or ChatGPT. Read this first. It records **verified facts, decisions and conventions** that are not obvious from the code. Update it at the end of every phase.

Last updated: 2026-11-08 (Phase 4 complete). Current state: Phase 5 (LightGBM) is next. See [phases.md](phases.md).

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
| *open* | PostgreSQL vs Parquet-only backend | Decide before Phase 7 ([architect.md](architect.md) §6) |

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
pytest                              # 54 tests at end of Phase 4
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
- Validation contains Memorial Day (Mon 2012-05-28). Training has no US federal holidays.
