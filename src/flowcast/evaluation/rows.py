"""The rows every model is scored on: the feature table's (as-of time, sensor) rows per split.

Baselines, LightGBM and DCRNN all predict a full (time x sensor) grid or a table; scoring
them on these same rows, with these same targets, is what makes their numbers comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from flowcast import paths
from flowcast.data.metr_la import FREQ
from flowcast.evaluation.metrics import evaluate
from flowcast.features.build import HORIZONS, target_column

FEATURES_DIR = paths.PROCESSED_DIR / "features"


@dataclass
class EvalRows:
    frame: pd.DataFrame  # timestamp, node_id, free_flow, target_h*
    t_idx: np.ndarray  # row position of each as-of time on the canonical grid
    node: np.ndarray

    def target(self, h: int) -> np.ndarray:
        return self.frame[target_column(h)].to_numpy(np.float64)

    def score(self, grid_pred: np.ndarray, h: int) -> dict:
        """Score a (T, N) grid of forecasts made at each as-of time for horizon h."""
        return evaluate(
            self.target(h),
            grid_pred[self.t_idx, self.node],
            pd.DatetimeIndex(self.frame["timestamp"]) + h * FREQ,
            self.node,
            self.frame["free_flow"].to_numpy(np.float64),
        )


def load_eval_rows(split: str, speed: pd.DataFrame) -> EvalRows:
    """Rows of ``split``, checked against the canonical speed matrix (targets must be speed[t + h])."""
    cols = ["timestamp", "node_id", "free_flow", *(target_column(h) for h in HORIZONS)]
    df = pq.read_table(FEATURES_DIR / f"{split}.parquet", columns=cols).to_pandas()
    t_idx = speed.index.get_indexer(pd.DatetimeIndex(df["timestamp"]))
    if (t_idx < 0).any():
        raise ValueError(f"{split}: timestamps outside the canonical grid")
    node = df["node_id"].to_numpy()
    values = speed.to_numpy()
    for h in HORIZONS:
        np.testing.assert_allclose(
            df[target_column(h)].to_numpy(np.float64), values[t_idx + h, node], atol=1e-4, equal_nan=True
        )
    return EvalRows(df, t_idx, node)
