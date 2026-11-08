"""Tests for the LightGBM helpers (synthetic data; no dataset needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcast.features.build import compute_features, compute_targets, feature_columns
from flowcast.models.lgbm import (
    BASE_PARAMS,
    FEATURE_GROUPS,
    as_model_input,
    feature_group,
    generic_seasonal,
    group_permutation_importance,
    train,
)


def all_feature_names() -> list[str]:
    rng = np.random.default_rng(0)
    speed = pd.DataFrame(
        rng.uniform(20, 70, (40, 2)), index=pd.date_range("2012-03-01", periods=40, freq="5min")
    )
    return list(compute_features(speed, np.eye(2), pd.Series([60.0, 60.0]))) + list(compute_targets(speed))


def test_every_feature_has_exactly_one_group():
    names = [n for n in all_feature_names() if not n.startswith("target")]
    groups = {n: feature_group(n) for n in names}
    assert set(groups.values()) == set(FEATURE_GROUPS)
    assert groups["speed"] == "current_and_lags" and groups["relative_speed"] == "sensor"
    assert groups["speed_lag12"] == "current_and_lags" and groups["week_profile_h3"] == "seasonal"


def test_generic_seasonal_names_match_across_horizons():
    names = all_feature_names()
    cols = {
        h: list(generic_seasonal(pd.DataFrame(columns=feature_columns(names, h)), h).columns) for h in (3, 12)
    }
    assert cols[3] == cols[12] and "week_profile" in cols[3]


def test_categoricals_keep_the_full_category_set():
    x = as_model_input(pd.DataFrame({"node_id": [5, 7], "day_of_week": [0, 6], "speed": [1.0, 2.0]}))
    assert list(x["node_id"].cat.categories) == list(range(207))
    assert x["speed"].dtype == float


def test_group_permutation_finds_the_feature_the_model_uses():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"speed": rng.uniform(0, 70, 5000), "tod_sin": rng.uniform(-1, 1, 5000)})
    y = x["speed"].to_numpy() * 1.0
    imp = group_permutation_importance(lambda d: d["speed"].to_numpy(), x, y)
    assert imp["current_and_lags"] > 10 and imp["calendar"] == pytest.approx(0.0)


def test_training_ignores_rows_with_missing_targets():
    rng = np.random.default_rng(2)
    x = pd.DataFrame({"speed": rng.uniform(0, 70, 4000)})
    y = pd.Series(x["speed"] + 1.0)
    y[::3] = np.nan  # masked targets must not break or bias training
    params = {**BASE_PARAMS, "num_leaves": 15, "min_data_in_leaf": 20, "num_threads": 1}
    booster = train(params, x, y, x.iloc[:500], y.iloc[:500], max_rounds=200, patience=20)
    pred = booster.predict(x.iloc[:500], num_iteration=booster.best_iteration)
    ok = y.iloc[:500].notna().to_numpy()
    assert np.abs(pred[ok] - y.iloc[:500].to_numpy()[ok]).mean() < 1.0
