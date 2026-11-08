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

## 3. LightGBM (Phase 5)

One **direct, global** model per horizon: each predicts `speed[t + h]` straight from the 39 features at `t` (direct), and one model serves all 207 sensors with `node_id` as a categorical feature (global). Rows whose target is missing are dropped from training; missing feature values stay NaN and LightGBM learns which branch they take. Code: `src/flowcast/models/lgbm.py`. Full numbers: `models/metrics/lightgbm_search.json`, `models/metrics/lightgbm.json`, [model card](model_card_lightgbm.md).

### Search (validation only; `scripts/tune_lightgbm.py`)

Six configurations at learning rate 0.1, on the 15- and 60-minute horizons, trained on every 2nd as-of time to halve the cost. Early stopping on validation MAE (patience 50). The test period was not loaded.

| Config | 15 min MAE / RMSE | 60 min MAE / RMSE |
|---|---|---|
| L2, 63 leaves, min leaf 200 | 2.681 / 4.837 | 3.646 / 6.775 |
| L2, 255 leaves, min leaf 200 | 2.668 / 4.829 | 3.639 / 6.789 |
| L2, 255 leaves, min leaf 1000 | 2.684 / 4.846 | 3.643 / 6.784 |
| L2, 127 leaves, min leaf 500 | 2.678 / 4.836 | 3.638 / 6.776 |
| L2, 127 leaves, min leaf 500, 60% features | 2.681 / 4.833 | 3.640 / 6.756 |
| **L1, 127 leaves, min leaf 500** | **2.559** / 5.024 | **3.388** / 7.005 |

Tree shape barely matters: every L2 configuration lands within 0.02 mph. The **loss** is what moves the result. The L1 objective (absolute error) predicts the conditional *median*; L2 (squared error) predicts the *mean*. Traffic errors are skewed (a sudden jam is a large one-sided surprise), so the mean is pulled towards rare jams and is worse on typical rows. L1 wins on MAE by about 7% but loses on RMSE by about 3–4%, because RMSE rewards hedging towards those rare large errors. MAE was the declared selection metric, so L1 was chosen; its RMSE still beats the baseline floor at every horizon (below).

### Strategy: one model per horizon vs one stacked model

With the L1 configuration on every 4th as-of time, a single model taking the horizon as an input feature was compared with four separate models:

| Horizon | Per-horizon MAE | Stacked MAE |
|---|---:|---:|
| 15 min | 2.610 | 2.652 |
| 30 min | 2.979 | 2.975 |
| 45 min | 3.224 | 3.195 |
| 60 min | 3.408 | 3.365 |

Mixed, and within about 1.6%. Per-horizon models were kept: they were the plan, a stacked model on full data needs four times the rows (about 18.7 M, a memory risk on 16 GB), and the difference comes from one seed on a subsample.

### Final models (`scripts/train_lightgbm.py`)

Full training period, learning rate lowered to 0.05 (smaller steps, more rounds, usually slightly better), patience 100, up to 6,000 rounds. Rounds used: 3,540 / 2,906 / 2,291 / 2,192 for 15 / 30 / 45 / 60 min. About 60 minutes on 8 CPU threads.

| Split | Horizon | MAE | RMSE | MAPE | Floor MAE | Floor RMSE |
|---|---|---:|---:|---:|---:|---:|
| validation | 15 min | 2.53 | 4.96 | 6.4% | 3.15 | 5.50 |
| validation | 30 min | 2.92 | 5.93 | 8.0% | 3.59 | 6.45 |
| validation | 45 min | 3.18 | 6.56 | 9.1% | 3.83 | 6.98 |
| validation | 60 min | 3.36 | 6.96 | 9.9% | 3.98 | 7.30 |
| test | 15 min | 2.71 | 5.23 | 7.1% | 3.27 | 5.71 |
| test | 30 min | 3.10 | 6.19 | 8.7% | 3.68 | 6.60 |
| test | 45 min | 3.34 | 6.75 | 9.7% | 3.89 | 7.06 |
| test | 60 min | 3.49 | 7.10 | 10.4% | 4.02 | 7.33 |

**Exit criterion met:** validation MAE and RMSE beat `ar_residual` at every horizon (MAE 16–20% lower, RMSE 5–10% lower). On test, scored once after everything was fixed, MAE is 13–17% lower. Test is worse than validation for both models alike (it has more outages), and the gap to the floor holds.

### Where it fails

- **Congestion.** Validation MAE on congested targets is 7.07 / 9.50 / 11.12 / 12.18 mph against the floor's 8.20 / 10.28 / 11.48 / 12.22. At 60 minutes the two are effectively tied: neither can see a jam coming an hour ahead from these features.
- **Rush hours.** Error peaks at 07:00–09:00 and 15:00–19:00 ([figure](figures/lightgbm/mae_by_hour.png)); LightGBM cuts the afternoon peak most.
- **Residuals** at 60 minutes ([figure](figures/lightgbm/residuals_h12.png)): median −0.07 mph, but the 99th percentile is +30.8 mph against −18.8 at the 1st. The heavy right tail (forecast far above actual) is jam onsets the model did not foresee; the mean residual is +0.67 mph.

### What the model relies on

Group permutation importance (validation MAE increase when a whole feature group is shuffled; 100,000 rows):

- At **15 min**: rolling means (+3.26), current speed and lags (+1.47), spatial neighbours (+0.78). The recent past dominates.
- At **60 min**: seasonal (+1.62), rolling (+1.39), sensor identity (+1.06), calendar (+0.78). The further ahead, the more the model leans on "what this sensor usually does at this time".
- Trend and missingness flags add almost nothing (≤ 0.03); candidates for removal later.

Groups are shuffled together because correlated features (e.g. `roll3_mean` and `roll6_mean`) cover for each other, so shuffling one alone understates their value.

### Caveats

- Validation was used for both early stopping and model selection, so validation scores are slightly optimistic. That is why test was scored separately, once.
- Only six configurations and one seed were tried; the L1-vs-L2 gap (~7%) is far larger than the spread between L2 configurations (~0.5%), so the choice is robust, but finer tuning was not attempted.
- Peak memory of the final run was not recorded correctly (it was sampled after each fit); a separate full-data timing run peaked at about 2.8 GB. The script now records the true peak.
- **Prediction intervals are deferred.** The plan allowed quantile or conformal intervals as optional; they need their own calibration check and are left for a later phase.
