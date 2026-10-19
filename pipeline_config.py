"""整个项目共用的路径和处理约定。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data/raw/题2_车辆物联网遥测.csv"
PROCESSED = ROOT / "data/processed"
REVIEW = ROOT / "data/review"
QUALITY = ROOT / "data/quality"
DOCS = ROOT / "docs"
KEYS = ["deviceId", "timestamp"]
WINDOW_SIZE = "10s"
POSITION_COLUMNS = ["position_1", "position_2", "position_3"]
CORE_FEATURES = ["Vehicle speed", "ENGINE RPM", "ACCELERATION X", "ACCELERATION Y", "ACCELERATION Z"]
UNKNOWN_VARIABLE = "PIDNameNotAvailable"
SOURCE_SHA256 = "e0f5571f74c6922ab7b5e65a34c52b2939b79bb33768eef30b124d3467333779"


def read_timed_csv(path):
    import pandas as pd
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", errors="raise")
    return frame


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
