"""LightGBM forecasters: one global model per horizon over all 207 sensors.

"Global" means one model learns from every sensor at once (``node_id`` is a
categorical feature), so a quiet sensor borrows strength from busy ones.
"Direct" means each horizon has its own model predicting ``speed[t + h]``
straight from the features at ``t``, rather than feeding predictions back in.

Training rows whose own target is missing are dropped (masked training). Missing
feature values stay NaN: at every split LightGBM learns which side NaN goes to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from flowcast import paths
from flowcast.features.build import CATEGORICAL, SEASONAL_PREFIXES, feature_columns, target_column

FEATURES_DIR = paths.PROCESSED_DIR / "features"

# Fixed settings; the searched ones live in models/metrics/lightgbm_search.json.
BASE_PARAMS = {
    "objective": "l2",
    "metric": ["l1", "l2"],  # early stopping uses the first: validation MAE
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_bin": 255,
    "num_threads": 8,  # physical cores of the Ryzen 7 7840HS
    "seed": 0,
    "deterministic": True,
    "verbose": -1,
}

# Correlated features share information, so permuting one alone understates what the group is worth.
FEATURE_GROUPS = {
    "current_and_lags": ("speed", "speed_lag", "speed_last_valid"),
    "rolling": ("roll",),
    "trend": ("trend",),
    "seasonal": SEASONAL_PREFIXES,
    "calendar": ("tod_", "day_of_week", "is_weekend"),
    "sensor": ("node_id", "free_flow", "relative_speed"),
    "spatial": ("down_", "up_", "neighbour_min"),
    "missingness": ("is_missing", "missing_count"),
}


def feature_group(name: str) -> str:
    if name in ("speed", "speed_last_valid") or name.startswith("speed_lag"):
        return "current_and_lags"
    for group, prefixes in FEATURE_GROUPS.items():
        if group != "current_and_lags" and name.startswith(prefixes):
            return group
    raise ValueError(f"feature {name!r} has no group")


def load_split(split: str, h: int, every: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Features ``X`` (39 columns for horizon h), target ``y`` (NaN where missing) and row metadata.

    ``every`` > 1 keeps only every n-th as-of time: a time-respecting subsample for quick experiments.
    """
    path = FEATURES_DIR / f"{split}.parquet"
    cols = feature_columns(pq.read_schema(path).names, h)
    df = pq.read_table(path, columns=["timestamp", *cols, target_column(h)]).to_pandas()
    if every > 1:
        slot = (df["timestamp"] - df["timestamp"].min()) // pd.Timedelta(minutes=5)
        df = df[slot % every == 0]
    meta = pd.DataFrame(
        {"timestamp": df["timestamp"], "node_id": df["node_id"], "free_flow": df["free_flow"]}
    )
    return as_model_input(df[cols]), df[target_column(h)], meta


def as_model_input(x: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as pandas ``category`` so LightGBM splits on them as groups, not as numbers."""
    x = x.copy()
    for c in CATEGORICAL:
        if c in x:
            x[c] = pd.Categorical(x[c].astype(np.int16), categories=range(_n_categories(c)))
    return x


def _n_categories(name: str) -> int:
    return {"node_id": 207, "day_of_week": 7}[name]


def generic_seasonal(x: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rename ``day_ago_h{h}`` etc. to ``day_ago`` so one model can serve every horizon."""
    return x.rename(columns={f"{p}{h}": p.removesuffix("_h") for p in SEASONAL_PREFIXES})


def train(
    params: dict,
    x: pd.DataFrame,
    y: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    max_rounds: int,
    patience: int,
):
    """Fit on rows with a target, early-stopping on validation MAE. Returns the booster."""
    import lightgbm as lgb

    keep, keep_val = y.notna().to_numpy(), y_val.notna().to_numpy()
    dtrain = lgb.Dataset(x[keep], y[keep], free_raw_data=True)
    dvalid = lgb.Dataset(x_val[keep_val], y_val[keep_val], reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
    )


def group_permutation_importance(predict, x: pd.DataFrame, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Increase in MAE (mph) when every feature of a group is shuffled across rows together.

    Shuffling breaks the link between that information and the target while keeping its distribution;
    the bigger the error increase, the more the model relies on it. ``predict``: X -> predictions.
    """
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(y)
    base = np.abs(predict(x)[mask] - y[mask]).mean()
    out = {}
    for group in FEATURE_GROUPS:
        cols = [c for c in x.columns if feature_group(c) == group]
        shuffled = x.copy()
        order = rng.permutation(len(x))
        for c in cols:
            shuffled[c] = x[c].to_numpy()[order]
            if isinstance(x[c].dtype, pd.CategoricalDtype):
                shuffled[c] = pd.Categorical(shuffled[c], categories=x[c].cat.categories)
        out[group] = float(np.abs(predict(shuffled)[mask] - y[mask]).mean() - base)
    return out
