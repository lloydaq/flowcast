"""Integration tests: real feature tables against the official windowed dataset.

Skipped unless scripts/build_features.py has been run. Uses the validation table
(no test-period inspection beyond row counts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from flowcast import paths
from flowcast.data.metr_la import FREQ, load_windows, window_column
from flowcast.data.processed import load_adjacency, load_speed
from flowcast.features.build import HORIZONS, LAGS, compute_features, target_column

FEATURES_DIR = paths.PROCESSED_DIR / "features"
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (FEATURES_DIR / "validation.parquet").exists(), reason="run build_features.py"),
]


def nan_to_zero(v: float) -> float:
    return 0.0 if np.isnan(v) else float(v)


def test_row_counts_match_official_samples_minus_boundary():
    counts = {
        s: pq.read_metadata(FEATURES_DIR / f"{s}.parquet").num_rows for s in ("train", "validation", "test")
    }
    as_of = {"train": 23974 - 12, "validation": 3425 - 12, "test": 6850}  # boundary-crossing rows purged
    for split, n in counts.items():
        assert 0.9 * as_of[split] * 207 < n <= as_of[split] * 207  # minus rows whose targets are all missing


def test_lags_and_targets_match_official_windows():
    table = pq.read_table(
        FEATURES_DIR / "validation.parquet",
        columns=[
            "timestamp",
            "node_id",
            "speed",
            *(f"speed_lag{k}" for k in LAGS),
            *(target_column(h) for h in HORIZONS),
        ],
    ).to_pandas()
    rows = table.sample(300, random_state=0)
    val = load_windows()["validation"]
    first_t0 = pd.Timestamp(val[0]["t0_timestamp"])
    for _, r in rows.iterrows():
        official = val[int((r.timestamp - first_t0) / FREQ) * 207 + int(r.node_id)]
        assert pd.Timestamp(official["t0_timestamp"]) == r.timestamp and official["node_id"] == r.node_id
        assert official[window_column(0, 0)] == pytest.approx(nan_to_zero(r.speed), abs=1e-4)  # float32
        for k in (k for k in LAGS if k <= 11):  # official windows reach back 11 steps
            assert official[window_column(-k, 0)] == pytest.approx(nan_to_zero(r[f"speed_lag{k}"]), abs=1e-4)
        for h in HORIZONS:
            assert official[window_column(h, 0)] == pytest.approx(nan_to_zero(r[target_column(h)]), abs=1e-4)


def test_no_leakage_on_real_data_slice():
    speed = load_speed().iloc[:3000]
    adj = load_adjacency()
    ff = pd.Series(np.full(speed.shape[1], 65.0))
    cut = 2500
    future = speed.copy()
    future.iloc[cut + 1 :] = np.random.default_rng(0).uniform(0, 70, size=future.iloc[cut + 1 :].shape)
    a, b = compute_features(speed, adj, ff), compute_features(future, adj, ff)
    for name in a:
        np.testing.assert_array_equal(a[name][: cut + 1], b[name][: cut + 1], err_msg=name)
