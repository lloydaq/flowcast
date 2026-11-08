"""Integration tests on the real METR-LA data.

Skipped unless ``scripts/prepare_metr_la.py`` has produced the processed artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from flowcast import paths
from flowcast.data.metr_la import FREQ, INPUT_OFFSETS, TARGET_OFFSETS, load_windows, window_column

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not paths.REPORT_PATH.exists(), reason="run scripts/prepare_metr_la.py first"),
]


@pytest.fixture(scope="module")
def speed() -> pd.DataFrame:
    return pd.read_parquet(paths.SPEED_PATH)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(paths.REPORT_PATH.read_text())


def test_series_shape_and_grid(speed):
    assert speed.shape == (34272, 208)  # timestamp + 207 sensors
    ts = speed["timestamp"]
    assert ts.iloc[0] == pd.Timestamp("2012-03-01 00:00") and ts.iloc[-1] == pd.Timestamp("2012-06-27 23:55")
    assert (ts.diff().dropna() == FREQ).all()


def test_values_in_valid_range(speed):
    values = speed.drop(columns="timestamp").to_numpy()
    assert not np.isnan(values).any()
    assert values.min() >= 0 and values.max() <= 70


def test_sensor_metadata_aligns_with_speed_columns(speed):
    sensors = pd.read_parquet(paths.SENSORS_PATH)
    assert len(sensors) == 207 and sensors["sensor_id"].is_unique
    assert sensors["node_id"].tolist() == list(range(207))
    assert speed.columns[1:].tolist() == sensors["sensor_id"].tolist()
    assert sensors["latitude"].between(33.5, 34.5).all() and sensors["longitude"].between(-119, -117.5).all()


def test_split_sizes_and_chronology(report):
    splits = report["splits"]
    assert [splits[s]["samples"] for s in ("train", "validation", "test")] == [23974, 3425, 6850]
    assert splits["train"]["t0_last"] < splits["validation"]["t0_first"]
    assert splits["validation"]["t0_last"] < splits["test"]["t0_first"]
    assert sum(s["owned_timestamps"] for s in splits.values()) == report["series"]["timestamps"]


def test_graph_neighbours_more_correlated_than_random_pairs(report):
    check = report["node_mapping_check"]
    assert check["mean_corr_connected"] > check["mean_corr_all_pairs"] + 0.1


def test_windows_match_reconstructed_series(speed):
    """Spot-check random raw windows from every split against the canonical matrix."""
    matrix = speed.drop(columns="timestamp").to_numpy()
    index = pd.Index(speed["timestamp"])
    ds = load_windows()
    rng = np.random.default_rng(0)
    for split in ds:
        rows = ds[split].select(rng.choice(ds[split].num_rows, 200, replace=False).tolist())
        for row in rows:
            t0 = index.get_loc(pd.Timestamp(row["t0_timestamp"]))
            for k in INPUT_OFFSETS + TARGET_OFFSETS:
                assert row[window_column(k, 0)] == matrix[t0 + k, row["node_id"]]
