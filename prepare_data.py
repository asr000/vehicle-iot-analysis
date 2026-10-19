"""预处理原始数据，生成宽表、完整长表和待核查记录。"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "题2_车辆物联网遥测.csv"
POSITION_COLUMNS = ["position_1", "position_2", "position_3"]


def main():
    df = pd.read_csv(DATA_PATH)

    # timestamp 完整可用，保留 timeMili 原样，不补写未知时区。
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="raise",
    )
    if df[["deviceId", "timestamp", "variable"]].isna().any().any():
        raise ValueError("设备、时间或变量名存在缺失，请先运行 inspect_data.py 检查。")
    if df.duplicated(subset=["deviceId", "timestamp", "variable"]).any():
        raise ValueError("设备、时间、变量存在重复，请先核查后再转宽表。")

    # 位置顺序尚未得到字段说明确认，暂用中性列名。
    position_df = df.loc[
        df["variable"].eq("POSITION"),
        ["deviceId", "timestamp", "value", "alarmClass"],
    ].copy()
    if position_df.empty:
        position_df[POSITION_COLUMNS] = float("nan")
    else:
        if not position_df["value"].str.split(",").str.len().eq(3).all():
            raise ValueError("POSITION 存在不能拆成三段的记录，请先检查原始值。")
        parts = position_df["value"].str.split(",", expand=True)
        parts.columns = POSITION_COLUMNS
        for column in POSITION_COLUMNS:
            position_df[column] = pd.to_numeric(parts[column], errors="raise")
        if position_df[POSITION_COLUMNS].isna().any().any():
            raise ValueError("POSITION 存在缺失坐标，请先检查原始值。")

    # 单独保留转换失败记录和全部未知变量记录。
    numeric_df = df.loc[df["variable"].ne("POSITION")].copy()
    numeric_df["numeric_value"] = pd.to_numeric(numeric_df["value"], errors="coerce")
    failed_mask = numeric_df["numeric_value"].isna()
    invalid_numeric_df = numeric_df.loc[failed_mask].copy()
    unknown_variable_df = df.loc[df["variable"].eq("PIDNameNotAvailable")].copy()
    valid_numeric_df = numeric_df.loc[
        (~failed_mask) & numeric_df["variable"].ne("PIDNameNotAvailable")
    ].copy()

    # 同一设备、同一时间的各变量展开为列。
    numeric_wide = valid_numeric_df.pivot(
        index=["deviceId", "timestamp"],
        columns="variable",
        values="numeric_value",
    ).reset_index()
    numeric_wide.columns.name = None

    position_wide = position_df[["deviceId", "timestamp", *POSITION_COLUMNS]]
    wide_df = numeric_wide.merge(
        position_wide,
        on=["deviceId", "timestamp"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["deviceId", "timestamp"])

    processed_dir = PROJECT_DIR / "data" / "processed"
    review_dir = PROJECT_DIR / "data" / "review"
    processed_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    wide_df.to_csv(processed_dir / "wide_unaligned.csv", index=False, encoding="utf-8-sig")
    df.to_csv(processed_dir / "telemetry_long.csv", index=False, encoding="utf-8-sig")
    invalid_numeric_df.to_csv(
        review_dir / "invalid_numeric_records.csv", index=False, encoding="utf-8-sig"
    )
    unknown_variable_df.to_csv(
        review_dir / "unknown_variable_records.csv", index=False, encoding="utf-8-sig"
    )

    print(f"预处理完成：宽表 {len(wide_df)} 行、{len(wide_df.columns)} 列；已保存到 {processed_dir}")


if __name__ == "__main__":
    main()
