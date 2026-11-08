"""Choose LightGBM settings on validation only. The test set is not loaded.

Usage:  python scripts/tune_lightgbm.py   (after scripts/build_features.py; ~30-40 min on CPU)

1. A small grid at learning rate 0.1 on the 15- and 60-minute horizons, trained on every 2nd
   as-of time (a time-respecting subsample, to halve the cost; rankings transfer to full data).
   The config with the lowest mean validation MAE over both horizons wins.
2. Strategy check with the winning config on every 4th as-of time: one model per horizon
   vs one stacked model that takes the horizon as an input feature.

Writes models/metrics/lightgbm_search.json, which scripts/train_lightgbm.py reads.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from flowcast import paths
from flowcast.evaluation.metrics import masked_metrics
from flowcast.features.build import HORIZONS
from flowcast.models.lgbm import BASE_PARAMS, generic_seasonal, load_split, train

OUT = paths.PROJECT_ROOT / "models" / "metrics" / "lightgbm_search.json"
SEARCH_HORIZONS = (3, 12)
GRID = [
    {"num_leaves": 63, "min_data_in_leaf": 200, "objective": "l2"},
    {"num_leaves": 255, "min_data_in_leaf": 200, "objective": "l2"},
    {"num_leaves": 255, "min_data_in_leaf": 1000, "objective": "l2"},
    {"num_leaves": 127, "min_data_in_leaf": 500, "objective": "l2"},
    {"num_leaves": 127, "min_data_in_leaf": 500, "objective": "l2", "feature_fraction": 0.6},
    {"num_leaves": 127, "min_data_in_leaf": 500, "objective": "l1"},
]
MAX_ROUNDS, PATIENCE = 3000, 50


def fit_and_score(params: dict, x, y, xv, yv) -> dict:
    t = time.perf_counter()
    booster = train(params, x, y, xv, yv, MAX_ROUNDS, PATIENCE)
    m = masked_metrics(yv.to_numpy(), booster.predict(xv, num_iteration=booster.best_iteration))
    return {
        "mae": m["mae"],
        "rmse": m["rmse"],
        "best_iteration": booster.best_iteration,
        "seconds": round(time.perf_counter() - t, 1),
    }


def main() -> None:
    search = []
    data = {h: (load_split("train", h, every=2), load_split("validation", h)) for h in SEARCH_HORIZONS}
    for i, cfg in enumerate(GRID):
        params = {**BASE_PARAMS, **cfg}
        run = {"config": cfg, "horizons": {}}
        for h, ((x, y, _), (xv, yv, _)) in data.items():
            run["horizons"][f"h{h}"] = fit_and_score(params, x, y, xv, yv)
        run["mean_mae"] = float(np.mean([r["mae"] for r in run["horizons"].values()]))
        search.append(run)
        print(i, cfg, {k: round(v["mae"], 4) for k, v in run["horizons"].items()}, flush=True)
    del data
    best = min(search, key=lambda r: r["mean_mae"])
    params = {**BASE_PARAMS, **best["config"]}

    # Strategy: direct per-horizon models vs one stacked model with the horizon as a feature.
    per_h, stacked_parts, stacked_val = {}, [], {}
    for h in HORIZONS:
        (x, y, _), (xv, yv, _) = load_split("train", h, every=4), load_split("validation", h, every=4)
        per_h[f"h{h}"] = fit_and_score(params, x, y, xv, yv)
        stacked_parts.append((generic_seasonal(x, h).assign(horizon=h), y))
        stacked_val[h] = (generic_seasonal(xv, h).assign(horizon=h), yv)
    xs = pd.concat([p[0] for p in stacked_parts], ignore_index=True)
    ys = pd.concat([p[1] for p in stacked_parts], ignore_index=True)
    xsv = pd.concat([v[0] for v in stacked_val.values()], ignore_index=True)
    ysv = pd.concat([v[1] for v in stacked_val.values()], ignore_index=True)
    del stacked_parts
    t = time.perf_counter()
    booster = train(params, xs, ys, xsv, ysv, MAX_ROUNDS, PATIENCE)
    stacked = {"best_iteration": booster.best_iteration, "seconds": round(time.perf_counter() - t, 1)}
    for h, (xv, yv) in stacked_val.items():
        m = masked_metrics(yv.to_numpy(), booster.predict(xv, num_iteration=booster.best_iteration))
        stacked[f"h{h}"] = {"mae": m["mae"], "rmse": m["rmse"]}

    payload = {
        "protocol": {
            "selection": "lowest mean validation MAE over horizons " + str(list(SEARCH_HORIZONS)),
            "search_train_subsample": "every 2nd as-of time",
            "strategy_subsample": "every 4th as-of time (train and validation)",
            "early_stopping": f"validation MAE, patience {PATIENCE}, max {MAX_ROUNDS} rounds",
            "test_set": "not loaded",
        },
        "base_params": BASE_PARAMS,
        "search": search,
        "chosen": best["config"],
        "strategy": {"per_horizon": per_h, "stacked": stacked},
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({"chosen": best["config"], "strategy": payload["strategy"]}, indent=1))


if __name__ == "__main__":
    main()
