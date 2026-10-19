"""检查原始数据，只输出检查摘要，不生成或修改数据文件。"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "题2_车辆物联网遥测.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    timestamps = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )
    print(
        f"数据：{len(df)} 行，{len(df.columns)} 列；"
        f"{df['deviceId'].nunique()} 个设备，{df['variable'].nunique()} 种变量"
    )
    print(f"字段缺失数：{df.isna().sum().to_dict()}")
    print(
        f"时间范围：{timestamps.min()} 至 {timestamps.max()}；"
        f"解析失败 {timestamps.isna().sum()} 条"
    )

    # 比较钟面时间差；此处不确定 timestamp 的时区。
    time_mask = df["timeMili"].notna() & timestamps.notna()
    utc_times = pd.to_datetime(df.loc[time_mask, "timeMili"], unit="ms", utc=True)
    time_diffs = timestamps.loc[time_mask] - utc_times.dt.tz_localize(None)
    print(f"与 UTC 的钟面时间差及条数：{time_diffs.astype(str).value_counts().to_dict()}")

    key_df = df[["deviceId", "variable"]].assign(timestamp=timestamps)
    print(
        f"完全重复 {df.duplicated().sum()} 条；"
        f"设备、时间、变量重复涉及 {key_df.duplicated(keep=False).sum()} 条"
    )

    position_values = df.loc[df["variable"].eq("POSITION"), "value"]
    bad_parts = position_values.str.split(",").str.len().ne(3)
    position_parts = position_values.loc[~bad_parts].str.split(",", expand=True)
    bad_coordinates = position_parts.apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    print(
        f"POSITION：{len(position_values)} 条；"
        f"分段异常 {bad_parts.sum()} 条，数值转换失败 {bad_coordinates.sum()} 条"
    )

    numeric_df = df.loc[df["variable"].ne("POSITION")]
    invalid_mask = pd.to_numeric(numeric_df["value"], errors="coerce").isna()
    unknown_count = df["variable"].eq("PIDNameNotAvailable").sum()
    print(
        f"非位置记录：{len(numeric_df)} 条；数值转换失败 {invalid_mask.sum()} 条；"
        f"未知变量 {unknown_count} 条（可能与转换失败记录重叠）"
    )
    if invalid_mask.any():
        invalid_counts = numeric_df.loc[invalid_mask, "variable"].value_counts().to_dict()
        print(f"转换失败的变量及条数：{invalid_counts}")


if __name__ == "__main__":
    main()
