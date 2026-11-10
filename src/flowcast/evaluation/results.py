"""Evaluation results of every model, read from the committed metrics JSON (``models/metrics/``)."""

from __future__ import annotations

import json

from flowcast import paths

SPLITS = ("validation", "test")
BASELINES = {
    "persistence": "persistence",
    "historical average": "historical_average",
    "seasonal naive (week)": "seasonal_naive_week",
    "AR on residuals": "ar_residual",
}
LEARNED = {"LightGBM": "lightgbm.json", "DCRNN": "dcrnn.json"}


def _read(name: str) -> dict:
    return json.loads((paths.METRICS_DIR / name).read_text(encoding="utf-8"))


def load_results() -> dict[str, dict]:
    """{model display name: {split: {"h3"...: {"all", "peak", ..., "mae_by_hour", ...}}}}, weakest first."""
    base = _read("baselines.json")["results"]
    models = {name: {s: base[s][key] for s in SPLITS} for name, key in BASELINES.items()}
    for name, file in LEARNED.items():
        models[name] = _read(file)["results"]
    return models


def load_comparison() -> dict:
    """The DCRNN - LightGBM MAE gap with bootstrap intervals (written by scripts/compare_models.py)."""
    return _read("comparison.json")
