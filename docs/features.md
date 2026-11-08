# Feature Engineering

Built by `scripts/build_features.py` using `src/flowcast/features/build.py`. All numbers come from `docs/features_summary.json`.

## What a row is

One row = **one sensor at one as-of time *t***. The model sees only what was known at *t* and predicts the speed at *t + h*:

| Horizon | Steps *h* | Target column |
|---|---:|---|
| 15 min | 3 | `target_h3` |
| 30 min | 6 | `target_h6` |
| 45 min | 9 | `target_h9` |
| 60 min | 12 | `target_h12` |

Features are computed once on the canonical 34,272 × 207 matrix, as one array per feature. Rows are then cut out per split. A validation or test row may use readings from earlier splits, because the past is known at prediction time.

## The leakage guarantee

**Every feature at time *t* depends only on readings at timestamps ≤ *t*.** This is enforced three ways:

1. **By construction.** Only backward shifts and trailing windows are used, never centred ones. Seasonal lags point to the target's clock time a day or a week earlier, and since *h* ≤ 12 ≪ 288, those offsets are always in the past.
2. **By test.** `test_features_ignore_the_future` builds features on a synthetic series long enough for every seasonal lag (4 weeks + 200 steps). It then overwrites every reading after a cut point with random values and a different missing pattern, and requires every feature up to the cut to be **bit-identical**. A real-data version runs on the first 3,000 steps of METR-LA.
3. **The test was shown to catch a leak.** Centring the rolling windows (`center=True`) makes it fail immediately, naming `roll3_mean`. With the original code it passes.

**Split boundaries.** A row is kept only if *t* and all its targets lie in the same split's owned period (see [dataset.md](dataset.md)). That drops exactly the 12 boundary-crossing as-of times per boundary found in Phase 1.

**Fitted statistics.** The only one is `free_flow`, the per-sensor 85th-percentile speed, estimated on the training period only. Validation and test rows therefore see nothing from their own periods. Training rows do see a statistic computed over the whole training period, which is a small in-sample effect, like a sensor ID. It cannot inflate validation or test scores.

## Features (48 columns)

Each horizon's model uses **39** of them: the 36 shared features plus its own three seasonal ones. `feature_columns(columns, h)` returns that list.

| Group | Columns | Definition | Why (evidence) |
|---|---|---|---|
| Current and lags | `speed`, `speed_lag1/2/3/6/12` | Reading at *t*, *t−1* … *t−12* | 5-min autocorrelation 0.995; 1 h 0.83 ([eda.md](eda.md) §4) |
| Short fill | `speed_last_valid` | Last reading, forward-filled at most 3 steps (15 min) | Bridges brief dropouts without inventing values across outages |
| Rolling | `roll{3,6,12}_{mean,min,max,std}` | Trailing 15 / 30 / 60-min windows ending at *t*, NaN-aware | Smooth noisy single readings; min/max/std capture stop-and-go |
| Trend | `trend3`, `trend6` | speed(*t*) − speed(*t−3*), speed(*t*) − speed(*t−6*) | Is speed falling into a congestion onset? |
| Seasonal (per *h*) | `day_ago_h{h}`, `week_ago_h{h}`, `week_profile_h{h}` | Speed at *t+h* one day / one week earlier; NaN-aware mean of the last 4 weeks at that clock time | Weekly correlation 0.88 > daily 0.67 |
| Calendar | `tod_sin`, `tod_cos`, `day_of_week`, `is_weekend` | Time of day of *t*, encoded on a circle; day of week 0 = Monday | Two weekday troughs; Friday differs; weekends have no commute peaks |
| Sensor | `node_id`, `free_flow`, `relative_speed` | Sensor id (categorical); training p85 speed; speed ÷ free-flow | Sensors are congested 0–49.5% of the time; their free-flow speed ranges 36–70 mph |
| Spatial | `down_mean`, `up_mean`, `down_roll3_mean`, `up_roll3_mean`, `neighbour_min` | Adjacency-weighted mean of neighbours' current (and 15-min mean) speed, and the slowest neighbour | Correlation 0.76 within 1 km; propagation within one step |
| Missingness | `is_missing`, `is_missing_lag1`, `missing_count_1h` | Flags and a count of missing readings in the last hour | Missingness depends on time of day and is not random |

**Why "circular" time encoding.** Time of day as a plain number puts 23:55 and 00:00 at opposite ends of the scale. As (sin, cos) on a circle they are neighbours, so a model can learn smooth daily patterns.

**Spatial direction (verified).** The published adjacency is exactly `A[i,j] = exp(−(d(i→j)/σ)²)` with σ = 2,584.5 m, thresholded at 0.1. That was checked against `distances.csv` to within 1e-7. Row *i* therefore holds sensors reachable **from** *i* along the road network (`down_*`), and column *i* holds sensors that reach *i* (`up_*`). The direction matters physically: congestion waves travel *backwards*, against traffic, so slowdowns downstream are an early warning upstream. The "downstream" label assumes `distances.csv` measures driving distance in the direction of travel, which is the DCRNN convention. The median sensor has 7 neighbours each way, and 1 sensor has none (its spatial features are NaN).

## Missing values

Missing readings stay **NaN** and are not imputed. LightGBM learns which branch to send them down. Missing-indicator features let the model tell "the road is empty" apart from "the sensor is down". Rows whose four targets are all missing are dropped. Rows with some missing targets are kept, and each horizon's model ignores rows where its own target is missing (masked training).

Missing rate of selected features (%):

| Feature | Train | Validation | Test |
|---|---:|---:|---:|
| `speed` | 7.13 | 6.98 | 12.11 |
| `speed_last_valid` | 6.36 | 6.27 | 11.15 |
| `roll12_mean` | 5.48 | 5.19 | 9.64 |
| `down_mean` | 7.91 | 7.65 | 12.13 |
| `week_ago_h12` | 15.11 | 3.19 | 13.88 |
| `week_profile_h12` | 8.80 | **0.01** | **0.04** |

`week_ago` is missing for the whole first week of training, because there is no earlier week. The 4-week profile is almost never missing in validation and test, so it should carry predictions through network-wide outages, when every recent feature is NaN.

## Tables

| Split | As-of times | First → last as-of | Rows | Rows dropped (all targets missing) | Target missing % | Parquet |
|---|---:|---|---:|---:|---:|---:|
| train | 23,962 | 2012-03-01 00:55 → 05-23 05:40 | 4,677,385 | 282,749 | 1.52 | 357 MB |
| validation | 3,413 | 2012-05-23 06:45 → 06-04 03:05 | 668,200 | 38,291 | 1.65 | 54 MB |
| test | 6,850 | 2012-06-04 04:10 → 06-27 22:55 | 1,276,937 | 141,013 | 2.40 | 100 MB |

- Files: `data/processed/features/{split}.parquet` (zstd, git-ignored, rebuild in ~34 s). Rows are ordered by time, then `node_id`.
- Types: float32 features and targets (about 1e-7 relative rounding, checked against the official windows to 1e-4 mph); `node_id` and `day_of_week` stored as int16 for categorical use.
- Memory: the 48 feature arrays take 1.3 GB, and the peak working set is **2.27 GB**, well within 16 GB. The training table is 946 MB in memory if loaded whole. Phase 5 can load only one horizon's 39 columns.

## Verification

| Test | What it proves |
|---|---|
| `test_features_ignore_the_future` (+ real-data slice) | No feature uses data after *t* |
| `test_seasonal_features_only_look_back` | Seasonal offsets land on *t+h−288* and *t+h−2016* |
| `test_lag_rolling_and_trend_values`, `test_week_profile_is_nan_aware_mean`, `test_forward_fill_is_limited` | Hand-computed values, NaN handling, the 15-min fill limit |
| `test_neighbour_features_follow_direction_and_skip_missing` | Direction of the graph, renormalised weights, isolated sensors |
| `test_row_times_keep_targets_inside_period` | No target crosses a split boundary |
| `test_lags_and_targets_match_official_windows` | 300 random validation rows: lags and all four targets equal the official HF windows (`x_t-k`, `y_t+h`) |
| `test_row_counts_match_official_samples_minus_boundary` | Row counts reconcile with the official sample counts |

## Not done here (by design)

- **Feature selection and importance:** Phase 5, measured on validation.
- **Scaling or normalisation:** not needed for trees. Phase 6 (DCRNN) will standardise using training statistics only.
- **Holiday features:** training contains no US federal holidays, so there is nothing to learn from.
