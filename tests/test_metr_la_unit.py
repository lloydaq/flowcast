"""Unit tests on synthetic windowed tables built from a known series (no download needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from flowcast.data.metr_la import (
    EXPECTED_COLUMNS,
    FREQ,
    INPUT_OFFSETS,
    TARGET_OFFSETS,
    check_layout,
    reconstruct_series,
    window_column,
)
from flowcast.data.validation import boundary_overlap, longest_run, missing_summary, owned_periods

N_SENSORS = 3
START = pd.Timestamp("2012-03-01 00:00")


def make_series(n_steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 70, size=(n_steps, N_SENSORS)).round(3)


def make_split(series: np.ndarray, t0_idx: range) -> pa.Table:
    """Window ``series`` exactly like the HF release: rows ordered by (t0, node_id)."""
    ts = pd.date_range(START, periods=len(series), freq=FREQ)
    t0 = ts[list(t0_idx)]
    cols = {
        "node_id": np.tile(np.arange(N_SENSORS), len(t0)),
        "t0_timestamp": np.repeat(t0.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(), N_SENSORS),
    }
    for k in INPUT_OFFSETS + TARGET_OFFSETS:
        rows = np.array(t0_idx) + k
        cols[window_column(k, 0)] = series[rows].reshape(-1)
        tk = ts[rows]
        cols[window_column(k, 1)] = np.repeat(((tk.hour * 60 + tk.minute) / 1440).to_numpy(), N_SENSORS)
    return pa.table({c: cols[c] for c in EXPECTED_COLUMNS})


def make_splits(series: np.ndarray) -> dict[str, pa.Table]:
    """70/10/20 split on sample index, as in DCRNN."""
    n_samples = len(series) - 11 - 12
    a, b = int(n_samples * 0.7), int(n_samples * 0.8)
    idx = range(11, 11 + n_samples)
    return {
        "train": make_split(series, idx[:a]),
        "validation": make_split(series, idx[a:b]),
        "test": make_split(series, idx[b:]),
    }


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    return table.set_column(table.column_names.index(name), name, pa.array(values))


def test_window_column_names():
    assert window_column(-11, 0) == "x_t-11_d0"
    assert window_column(0, 1) == "x_t+0_d1"
    assert window_column(12, 0) == "y_t+12_d0"
    assert len(EXPECTED_COLUMNS) == 50


def test_reconstruction_round_trips_exactly():
    series = make_series(120)
    result = reconstruct_series(make_splits(series))
    np.testing.assert_array_equal(result.speed, series)
    assert result.timestamps[0] == START
    assert len(result.timestamps) == len(series)
    assert list(result.anchors) == ["train", "validation", "test"]


def test_reconstruction_is_independent_of_dict_order():
    series = make_series(120)
    splits = make_splits(series)
    shuffled = {k: splits[k] for k in ("test", "train", "validation")}
    np.testing.assert_array_equal(reconstruct_series(shuffled).speed, series)


def test_conflicting_copies_are_rejected():
    splits = make_splits(make_series(120))
    col = window_column(3, 0)
    values = splits["train"].column(col).to_numpy().copy()
    values[5] += 1.0  # one copy of a reading now disagrees with the others
    splits["train"] = replace_column(splits["train"], col, values)
    with pytest.raises(ValueError, match="conflicting"):
        reconstruct_series(splits)


def test_time_of_day_mismatch_is_rejected():
    splits = make_splits(make_series(120))
    col = window_column(-2, 1)
    splits["test"] = replace_column(splits["test"], col, splits["test"].column(col).to_numpy() + 0.01)
    with pytest.raises(ValueError, match="time-of-day"):
        reconstruct_series(splits)


def test_non_contiguous_splits_are_rejected():
    series = make_series(200)
    splits = {"train": make_split(series, range(11, 60)), "test": make_split(series, range(61, 100))}
    with pytest.raises(ValueError, match="contiguous"):
        reconstruct_series(splits)


def test_shuffled_rows_are_rejected():
    table = make_split(make_series(80), range(11, 50))
    perm = np.random.default_rng(1).permutation(table.num_rows)
    with pytest.raises(ValueError, match="ordered"):
        check_layout(table.take(perm))


def test_duplicate_timestamps_are_rejected():
    table = make_split(make_series(80), range(11, 50))
    doubled = pa.concat_tables([table.slice(0, N_SENSORS), table])
    with pytest.raises(ValueError, match="5-minute"):
        check_layout(doubled)


def test_missing_column_is_rejected():
    table = make_split(make_series(80), range(11, 50)).drop_columns(["y_t+7_d0"])
    with pytest.raises(ValueError, match="missing columns"):
        check_layout(table)


def test_owned_periods_partition_timeline_without_overlap():
    result = reconstruct_series(make_splits(make_series(150)))
    periods = owned_periods(result.timestamps, result.anchors)
    owned = np.zeros(len(result.timestamps), dtype=int)
    for start, end in periods.values():
        owned += (result.timestamps >= start) & (result.timestamps <= end)
    assert (owned == 1).all()


def test_boundary_overlap_counts_last_twelve_samples():
    result = reconstruct_series(make_splits(make_series(150)))
    overlap = boundary_overlap(result.anchors, owned_periods(result.timestamps, result.anchors))
    assert overlap == {"train": 12, "validation": 12, "test": 0}


def test_longest_run():
    mask = np.array([[1, 0], [1, 1], [0, 1], [1, 1], [1, 1], [1, 0]], dtype=bool)
    np.testing.assert_array_equal(longest_run(mask), [3, 4])


def test_missing_summary_counts_zeros_only():
    speed = np.array([[0.0, 50.0], [0.0, 0.0], [60.0, 0.0], [61.0, 62.0]])
    out = missing_summary(speed)
    assert out["missing_pct"] == 50.0
    assert out["timestamps_all_sensors_missing"] == 1
    assert out["longest_missing_run_steps"]["max_sensor"] == 2
