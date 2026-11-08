"""Build point-in-time feature tables for train / validation / test.

Usage:  python scripts/build_features.py   (after scripts/prepare_metr_la.py)

Writes data/processed/features/{split}.parquet and docs/features_summary.json.
Free-flow speed (the only fitted statistic) is estimated on the training period only.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from flowcast import paths
from flowcast.congestion import free_flow_speed
from flowcast.data.metr_la import SPLITS
from flowcast.data.processed import load_adjacency, load_speed, owned_period
from flowcast.features.build import (
    HORIZONS,
    compute_features,
    compute_targets,
    row_times,
    target_column,
    to_table,
)

OUT_DIR = paths.PROCESSED_DIR / "features"
SUMMARY_PATH = paths.PROJECT_ROOT / "docs" / "features_summary.json"
MB = 2**20


def main() -> None:
    started = time.perf_counter()
    proc = psutil.Process()
    speed = load_speed()  # full timeline; rows choose their own periods below
    free_flow = free_flow_speed(load_speed("train"))
    adjacency = load_adjacency()

    features = compute_features(speed, adjacency, free_flow)
    targets = compute_targets(speed)
    rss_after_compute = proc.memory_info().rss // MB

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "feature_count": len(features),
        "features": list(features),
        "targets": list(targets),
        "horizons_steps": list(HORIZONS),
        "free_flow_source": "train owned period, p85",
        "splits": {},
    }
    for split in SPLITS:
        times = row_times(speed.index, owned_period(split))
        table = to_table(features, targets, times, speed.index)
        path = OUT_DIR / f"{split}.parquet"
        pq.write_table(table, path, compression="zstd")
        df_nan = {name: float(np.isnan(arr[times]).mean()) for name, arr in features.items()}
        summary["splits"][split] = {
            "as_of_first": str(speed.index[times[0]]),
            "as_of_last": str(speed.index[times[-1]]),
            "as_of_times": len(times),
            "rows": table.num_rows,
            "rows_dropped_all_targets_missing": len(times) * speed.shape[1] - table.num_rows,
            "target_missing_pct": {
                c: round(float(pd.isna(table.column(c).to_numpy(zero_copy_only=False)).mean() * 100), 2)
                for c in (target_column(h) for h in HORIZONS)
            },
            "in_memory_mb": round(table.nbytes / MB, 1),
            "parquet_mb": round(path.stat().st_size / MB, 1),
            "feature_missing_pct": {k: round(v * 100, 2) for k, v in df_nan.items()},
        }
        del table

    peak = getattr(proc.memory_info(), "peak_wset", None)
    summary["memory"] = {
        "feature_arrays_mb": round(sum(a.nbytes for a in features.values()) / MB, 1),
        "rss_after_compute_mb": rss_after_compute,
        "peak_rss_mb": peak // MB if peak else None,
    }
    summary["runtime_s"] = round(time.perf_counter() - started, 1)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k not in ("features", "splits")}
    brief["splits"] = {
        s: {k: v for k, v in d.items() if k != "feature_missing_pct"} for s, d in summary["splits"].items()
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
