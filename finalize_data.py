"""生成正式宽表、缺失标记、告警关联表和逐单元格来源明细。"""
import pandas as pd

from pipeline_config import (
    RAW_PATH, PROCESSED, KEYS, WINDOW_SIZE, POSITION_COLUMNS,
    CORE_FEATURES, UNKNOWN_VARIABLE, read_timed_csv, save_csv,
)


def make_feature_lineage(raw):
    """每个有效宽表单元格保留最后一次原始观测及其数据行号。"""
    numeric = raw.loc[~raw["variable"].isin(["POSITION", UNKNOWN_VARIABLE])].copy()
    numeric["value_numeric"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.loc[numeric["value_numeric"].notna()]
    columns = ["source_row_id", "deviceId", "timestamp", "variable", "value_numeric"]
    parts = [numeric[columns]]
    position = raw.loc[raw["variable"].eq("POSITION")].copy()
    if not position.empty:
        coordinates = position["value"].str.split(",", expand=True)
        for index, name in enumerate(POSITION_COLUMNS):
            component = position[["source_row_id", "deviceId", "timestamp"]].copy()
            component["variable"] = name
            component["value_numeric"] = pd.to_numeric(coordinates[index], errors="raise")
            parts.append(component[columns])
    measurements = pd.concat(parts, ignore_index=True)
    measurements = measurements.rename(columns={"timestamp": "observed_at", "variable": "feature"})
    measurements["timestamp"] = measurements["observed_at"].dt.floor(WINDOW_SIZE)
    measurements = measurements.sort_values(["deviceId", "observed_at", "source_row_id"])
    lineage = measurements.groupby([*KEYS, "feature"], sort=False).tail(1)
    return lineage[[*KEYS, "feature", "value_numeric", "observed_at", "source_row_id"]].sort_values([*KEYS, "feature"])


def main():
    clean = read_timed_csv(PROCESSED / "wide_10s.csv").sort_values(KEYS).reset_index(drop=True)
    features = [column for column in clean if column not in KEYS]
    # 正式表不插值、不补零、不跨窗口沿用；空值代表该窗口内未观测到。
    save_csv(clean, PROCESSED / "clean_wide.csv")
    missing = clean[KEYS].copy()
    missing[features] = clean[features].isna().astype("uint8")
    save_csv(missing, PROCESSED / "missing_mask.csv")

    core_mask = clean[CORE_FEATURES].notna()
    core = clean.loc[core_mask.any(axis=1)].copy()
    core["core_feature_count"] = core_mask.loc[core.index].sum(axis=1)
    core["has_all_core_features"] = core["core_feature_count"].eq(len(CORE_FEATURES)).astype("uint8")
    save_csv(core, PROCESSED / "analysis_core_windows.csv")

    raw = read_timed_csv(RAW_PATH)
    raw.insert(0, "source_row_id", range(1, len(raw) + 1))
    # 保留包括 0 在内的全部编码；不假设任何编码等于“正常”。
    alarm_records = raw.copy()
    alarm_records["window_start"] = raw["timestamp"].dt.floor(WINDOW_SIZE)
    save_csv(alarm_records, PROCESSED / "alarm_records.csv")

    tagged = raw.assign(timestamp=raw["timestamp"].dt.floor(WINDOW_SIZE))
    counts = tagged.groupby([*KEYS, "alarmClass"]).size().unstack(fill_value=0)
    code_values = sorted(int(code) for code in counts.columns)
    alarm_windows = counts.rename(columns=lambda code: f"code_{int(code)}_count").reset_index()
    count_columns = [f"code_{code}_count" for code in code_values]
    alarm_windows.insert(2, "window_end", alarm_windows["timestamp"] + pd.Timedelta(WINDOW_SIZE))
    alarm_windows["record_count"] = alarm_windows[count_columns].sum(axis=1)
    present = alarm_windows[count_columns].gt(0)
    alarm_windows["code_count"] = present.sum(axis=1)
    alarm_windows["code_values"] = [
        "|".join(str(code) for code, included in zip(code_values, flags) if included)
        for flags in present.to_numpy()
    ]
    alarm_windows["single_alarmClass"] = pd.array(
        [int(values) if count == 1 else pd.NA
         for values, count in zip(alarm_windows["code_values"], alarm_windows["code_count"])],
        dtype="Int64",
    )
    alarm_windows["multi_code"] = alarm_windows["code_count"].gt(1).astype("uint8")
    save_csv(alarm_windows, PROCESSED / "alarm_windows.csv")

    lineage = make_feature_lineage(raw)
    save_csv(lineage, PROCESSED / "feature_lineage.csv")
    print(f"正式数据完成：{len(clean)} 个窗口；核心分析子集 {len(core)} 行；来源明细 {len(lineage)} 条。")


if __name__ == "__main__":
    main()
