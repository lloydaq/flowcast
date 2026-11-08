# Experiments

The experiment log. Every number here is copied from committed JSON written by a script (`models/metrics/*.json`); the full generated tables are in [baseline_results.md](baseline_results.md). Newest entries at the bottom.

## 1. Evaluation protocol (fixed for every model)

| Rule | Value | Why |
|---|---|---|
| Rows scored | The rows of `data/processed/features/{validation,test}.parquet`: one per (sensor, as-of time), targets inside the split | LightGBM, DCRNN and the baselines all score exactly the same rows |
| Missing targets | Excluded (masked), never treated as 0 mph | A missing reading is not a traffic jam |
| Missing predictions | Not allowed: a NaN prediction on a scored row raises an error | A model cannot look better by abstaining on hard rows |
| MAE, RMSE | mph, per horizon | MAE = typical error; RMSE punishes big misses, such as a missed jam |
| MAPE | %, excluding targets under 5 mph | Dividing by a near-zero speed makes one row dominate the average |
| Peak slice | Weekday target times 07:00–09:59 and 15:00–18:59 | The hours where the training weekday mean speed dips (49.5 mph at 08:00, 46.9 mph at 17:00; [eda.md](eda.md)) |
| Congested slice | Target speed < 0.6 × the sensor's free-flow speed (training p85) | The shared definition in `flowcast/congestion.py` |
| Breakdowns | MAE by hour of the target time and per sensor (in the JSON) | Finds where a model fails, not just how much |
| Fitting | Anything fitted uses the training period only | No validation or test information leaks into a model |
| Test set | Scored once per model family, after its choices are frozen on validation | The test score is the honest estimate; tuning on it would make it optimistic |

Slices use the **target** time, the moment whose speed is predicted, because that is the traffic being forecast.

## 2. Baselines (Phase 4)

`src/flowcast/models/baselines.py`. Each forecaster outputs, for every as-of time *t*, the speed expected at *t + h*, using readings at or before *t* only. That is checked by the same future-perturbation test as the features, and the test was shown to fail on two deliberately injected lookahead bugs. When a baseline's own input is missing (a sensor outage), it falls back to the historical average, and the share of rows that needed this is reported.

| Baseline | Rule | ML idea it represents |
|---|---|---|
| `persistence` | The last reading (gaps up to 15 min bridged), held flat | "Nothing changes": strong at short horizons because traffic is smooth over 5 minutes |
| `historical_average` | Training mean for that (sensor, weekday, 5-min slot) | "It's like a usual Tuesday at 08:00": ignores today entirely, so its error does not grow with the horizon |
| `seasonal_naive_week` | The reading at the target's clock time one week earlier | A single past week instead of an average of 12 |
| `ar_residual` | Historical average + a per-sensor linear autoregression on today's deviation from it (12 lags, one least-squares fit per sensor and horizon) | Classic statistics: remove the seasonal pattern, then model what is left. It learns how fast today's deviation fades back to normal |

### Results

MAE in mph (lower is better). Full MAE / RMSE / MAPE tables: [baseline_results.md](baseline_results.md).

| Baseline | Val 15 min | Val 30 min | Val 45 min | Val 60 min | Test 15 min | Test 60 min |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 3.21 | 3.81 | 4.32 | 4.78 | 3.48 | 5.30 |
| historical_average | 4.34 | 4.33 | 4.33 | 4.33 | 4.19 | 4.19 |
| seasonal_naive_week | 5.04 | 5.04 | 5.03 | 5.03 | 5.09 | 5.09 |
| **ar_residual** | **3.15** | **3.59** | **3.83** | **3.98** | **3.27** | **4.02** |

At 60 min on validation, MAE is 3.31 off-peak but 6.79 at peak and 12.22 on congested targets for `ar_residual`. Every baseline is above 12 mph on congested targets.

### What this shows

- **Persistence vs history cross over at about 45 min.** Persistence is good when the near future looks like now and loses as the horizon grows (3.21 → 4.78). The historical average is flat (4.33) because it never looks at today. At 45 min they are level.
- **The AR model gets the best of both.** It starts near persistence and decays towards the profile, and it is the best baseline at every horizon on MAE and RMSE. Persistence still has a slightly better 15-min MAPE (7.6% vs 8.0%), since it is very accurate on free-flowing rows where a percentage error is cheap.
- **Averaging weeks beats copying one week.** The seasonal naive is worse than the historical average everywhere: a single week carries that week's accidents and outages. On test it also has to fall back on 14% of rows, because week-ago readings were lost to outages.
- **The hard part is congestion.** Peak-hour error is about twice the overall error, and congested-target error is about three times it. RMSE is roughly double MAE: errors are heavy-tailed, dominated by jams starting or clearing. This is where Phase 5 has to win, not on free-flowing traffic.
- **Test is harder than validation for anything that reads today's data** (persistence 60-min MAE 4.78 → 5.30), consistent with the test period's higher missing rate (12.1%). The historical average is slightly better on test (4.19 vs 4.33). Validation contains Memorial Day, a plausible but unverified cause.

### The floor to beat

A later model must beat `ar_residual` at every horizon on validation MAE and RMSE to count as an improvement:

| Horizon | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| 15 min | 3.15 | 5.50 | 3.27 | 5.71 |
| 30 min | 3.59 | 6.45 | 3.68 | 6.60 |
| 45 min | 3.83 | 6.98 | 3.89 | 7.06 |
| 60 min | 3.98 | 7.30 | 4.02 | 7.33 |

### Caveats

- These scores are not directly comparable with published METR-LA tables. Papers score the official windows with their own masking, and some count missing targets differently. Ours scores the feature-table rows, which excludes the 12 boundary-crossing as-of times per split and rows whose four targets are all missing.
- The historical average and AR coefficients are fitted on the whole training period, so their in-sample fit on training rows is optimistic. That does not affect validation or test.
- `ar_residual` has one fixed choice (12 lags) and was not tuned. It is a reference, not an optimised model.
