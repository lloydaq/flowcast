"""Acquire, validate and reconstruct METR-LA into FlowCast's processed format.

Usage:  python scripts/prepare_metr_la.py

Outputs (data/processed/):
  metr_la_speed.parquet            timestamp + one float64 column per sensor_id (mph, 0 = missing)
  metr_la_sensors.parquet          node_id, sensor_id, latitude, longitude, missing_pct
  metr_la_splits.json              per-split t0 anchors and owned periods
  metr_la_validation_report.json   full validation report

The download (~230 MB) and Arrow cache (~2.8 GB) are reused on later runs.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import psutil

from flowcast import paths
from flowcast.data import validation as v
from flowcast.data.metr_la import (
    EXPECTED_COLUMNS,
    HF_REPO,
    MISSING_VALUE,
    SPLITS,
    load_sensor_graph,
    load_windows,
    reconstruct_series,
)

MB = 2**20


def rss_mb() -> int:
    return psutil.Process().memory_info().rss // MB


def main() -> None:
    started = time.perf_counter()
    memory = {"start_rss_mb": rss_mb()}

    ds = load_windows()
    if tuple(ds.keys()) != SPLITS:
        raise ValueError(f"unexpected splits {list(ds.keys())}")
    tables = {name: ds[name].data.table for name in SPLITS}
    schema = {
        name: {
            "rows": t.num_rows,
            "columns": t.num_columns,
            "column_order_matches_expected": tuple(t.column_names) == EXPECTED_COLUMNS,
            "dtypes": sorted({str(f.type) for f in t.schema}),
            "arrow_mb": t.nbytes // MB,
        }
        for name, t in tables.items()
    }
    memory["after_load_rss_mb"] = rss_mb()

    series = reconstruct_series(tables)
    memory["after_reconstruct_rss_mb"] = rss_mb()
    ts, speed = series.timestamps, series.speed

    graph = load_sensor_graph()
    periods = v.owned_periods(ts, series.anchors)
    train_mask = ts <= periods["train"][1]

    report = {
        "source": {"hf_repo": HF_REPO, "cache_files": [f["filename"] for f in ds["train"].cache_files]},
        "schema": schema,
        "series": {
            "timestamps": len(ts),
            "sensors": speed.shape[1],
            "start": str(ts[0]),
            "end": str(ts[-1]),
            "frequency": "5min",
            "regular_grid": True,  # guaranteed by reconstruct_series
            "redundant_copies_consistent": True,  # guaranteed by reconstruct_series
            "time_of_day_feature_consistent": True,  # guaranteed by reconstruct_series
            "units": "mph",
            "missing_encoding": MISSING_VALUE,
        },
        "splits": v.split_summary(ts, series.anchors, speed),
        "missing": v.missing_summary(speed),
        "values": v.value_checks(speed),
        "graph": v.graph_checks(graph, speed.shape[1]),
        "node_mapping_check": v.neighbour_correlation(speed[train_mask], graph.adjacency),
    }

    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    speed_df = pd.DataFrame(speed, columns=graph.sensors["sensor_id"].tolist())
    speed_df.insert(0, "timestamp", ts)
    speed_df.to_parquet(paths.SPEED_PATH, index=False)

    sensors = graph.sensors.assign(missing_pct=((speed == MISSING_VALUE).mean(axis=0) * 100).round(3))
    sensors.to_parquet(paths.SENSORS_PATH, index=False)

    splits = {
        name: {"t0_first": str(a[0]), "t0_last": str(a[1]), "owned_period": [str(p) for p in periods[name]]}
        for name, a in series.anchors.items()
    }
    paths.SPLITS_PATH.write_text(json.dumps(splits, indent=2))

    memory["speed_matrix_mb"] = round(speed.nbytes / MB, 1)
    memory["end_rss_mb"] = rss_mb()
    peak = getattr(psutil.Process().memory_info(), "peak_wset", None)  # Windows only
    memory["peak_rss_mb"] = peak // MB if peak else None
    report["memory"] = memory
    report["runtime_s"] = round(time.perf_counter() - started, 1)
    paths.REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {paths.PROCESSED_DIR}")


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
