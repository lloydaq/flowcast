"""Filesystem locations used across FlowCast."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SPEED_PATH = PROCESSED_DIR / "metr_la_speed.parquet"
SENSORS_PATH = PROCESSED_DIR / "metr_la_sensors.parquet"
SPLITS_PATH = PROCESSED_DIR / "metr_la_splits.json"
REPORT_PATH = PROCESSED_DIR / "metr_la_validation_report.json"
