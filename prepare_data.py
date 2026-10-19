"""合并后的数据处理脚本：原始 CSV -> 主宽表、完整长表、统一核查表。"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

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
CODE_FEATURES = {
    "TOWING", "IGNITION_STATUS", "IDLING", "FUEL SYSTEM STATUS", "FUEL TYPE", "HARDWARE STATUS",
    "OBD STANDARDS THIS VEHICLE CONFORMS TO", "OXYGEN SENSORS PRESENT IN 2 BANKS",
    "MONITOR STATUS THIS DRIVE CYCLE", "MONITOR STATUS SINCE DTCS CLEARED",
}
META_COLUMNS = {
    *KEYS, "window_end", "record_count", "code_count", "code_values", "single_alarmClass",
    "multi_code", "core_feature_count", "has_all_core_features",
}


def read_timed_csv(path):
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", errors="raise")
    return frame


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sensor_columns(frame):
    return [
        column for column in frame
        if column not in META_COLUMNS
        and not (column.startswith("code_") and column.endswith("_count"))
    ]


def build_exact(raw):
    """拆分位置并展开精确时间宽表；中间表只在内存中保留。"""
    if raw[[*KEYS, "variable"]].isna().any().any():
        raise ValueError("设备、时间或变量名存在缺失。")
    if raw.duplicated([*KEYS, "variable"]).any():
        raise ValueError("同设备、时间、变量存在重复，请先核查。")
    numeric = raw.loc[~raw.variable.isin(["POSITION", UNKNOWN_VARIABLE])].copy()
    numeric["numeric_value"] = pd.to_numeric(numeric.value, errors="coerce")
    numeric = numeric.loc[numeric.numeric_value.notna()]
    exact = numeric.pivot(index=KEYS, columns="variable", values="numeric_value").reset_index()
    exact.columns.name = None
    positions = raw.loc[raw.variable.eq("POSITION"), KEYS + ["value"]].copy()
    if not positions.value.str.split(",").str.len().eq(3).all():
        raise ValueError("POSITION 存在不能拆成三段的记录。")
    parts = positions.value.str.split(",", expand=True)
    parts.columns = POSITION_COLUMNS
    for column in POSITION_COLUMNS:
        positions[column] = pd.to_numeric(parts[column], errors="raise")
    return exact.merge(positions[KEYS + POSITION_COLUMNS], on=KEYS, how="outer", validate="one_to_one").sort_values(KEYS)


def align_exact(exact):
    features = sensor_columns(exact)
    parts = []
    for device, group in exact.groupby("deviceId"):
        window = (
            group.set_index("timestamp")[features].sort_index()
            .resample(WINDOW_SIZE, closed="left", label="left").last().dropna(how="all")
        )
        window.insert(0, "deviceId", device)
        parts.append(window.reset_index())
    return pd.concat(parts, ignore_index=True)[KEYS + features]


def summarize_codes(raw):
    tagged = raw.assign(timestamp=raw.timestamp.dt.floor(WINDOW_SIZE))
    counts = tagged.groupby([*KEYS, "alarmClass"]).size().unstack(fill_value=0)
    codes = sorted(int(code) for code in counts.columns)
    result = counts.rename(columns=lambda code: f"code_{int(code)}_count").reset_index()
    columns = [f"code_{code}_count" for code in codes]
    result["window_end"] = result.timestamp + pd.Timedelta(WINDOW_SIZE)
    result["record_count"] = result[columns].sum(axis=1)
    present = result[columns].gt(0)
    result["code_count"] = present.sum(axis=1)
    result["code_values"] = [
        "|".join(str(code) for code, included in zip(codes, flags) if included)
        for flags in present.to_numpy()
    ]
    result["single_alarmClass"] = pd.array(
        [int(value) if count == 1 else pd.NA for value, count in zip(result.code_values, result.code_count)],
        dtype="Int64",
    )
    result["multi_code"] = result.code_count.gt(1).astype("uint8")
    return result


def collect_reviews(raw):
    """一条原始记录只保留一行，多个核查原因用 | 连接。"""
    reasons = {}
    def mark(indices, reason):
        for index in indices:
            reasons.setdefault(index, []).append(reason)
    unknown = raw.variable.eq(UNKNOWN_VARIABLE)
    mark(raw.index[unknown], "unknown_variable")
    numeric = raw.loc[raw.variable.ne("POSITION")].copy()
    numeric["numeric_value"] = pd.to_numeric(numeric.value, errors="coerce")
    mark(numeric.index[numeric.numeric_value.isna()], "numeric_parse_failed")
    known = numeric.loc[numeric.numeric_value.notna() & ~numeric.variable.eq(UNKNOWN_VARIABLE)]
    for (_, feature), group in known.groupby(["deviceId", "variable"]):
        values = group.numeric_value
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        if feature in CODE_FEATURES or len(group) < 20 or iqr <= 0:
            continue
        flag = values.lt(q1 - 1.5 * iqr) | values.gt(q3 + 1.5 * iqr)
        mark(group.index[flag], "iqr_candidate_retained")
    review = raw.loc[sorted(reasons)].copy()
    review.insert(0, "source_row_id", review.index + 1)
    review["review_reason"] = ["|".join(reasons[index]) for index in review.index]
    return review


def main():
    if hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("原始文件与本次交付快照不一致，请先确认输入数据。")
    raw = read_timed_csv(RAW_PATH)
    measurements = align_exact(build_exact(raw))
    # 告警与测量按相同粒度合在主表；模型输入只使用 sensor_columns 返回的列。
    clean = measurements.merge(summarize_codes(raw), on=KEYS, how="outer", validate="one_to_one")
    clean = clean.sort_values(KEYS).reset_index(drop=True)
    clean["core_feature_count"] = clean[CORE_FEATURES].notna().sum(axis=1)
    clean["has_all_core_features"] = clean.core_feature_count.eq(len(CORE_FEATURES)).astype("uint8")
    save_csv(clean, PROCESSED / "clean_wide.csv")
    long = raw.copy()
    long.insert(0, "source_row_id", range(1, len(long) + 1))
    long["window_start"] = long.timestamp.dt.floor(WINDOW_SIZE)
    save_csv(long, PROCESSED / "telemetry_long.csv")
    review = collect_reviews(raw)
    save_csv(review, REVIEW / "review_records.csv")
    print(f"处理完成：主表 {len(clean)} 行、{len(clean.columns)} 列；长表 {len(long)} 行；核查表 {len(review)} 行。")


if __name__ == "__main__":
    main()
