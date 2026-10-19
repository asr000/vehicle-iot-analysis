"""读取预处理宽表，按设备整理为 10 秒窗口，不跨窗口填补。"""

from pathlib import Path

import pandas as pd
from pipeline_config import WINDOW_SIZE


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def main():
    input_path = PROCESSED_DIR / "wide_unaligned.csv"
    if not input_path.is_file():
        raise FileNotFoundError("找不到预处理宽表，请先运行 python prepare_data.py。")

    wide_df = pd.read_csv(input_path)
    # 导出后的时间可能混合整秒与小数秒，显式解析混合格式。
    wide_df["timestamp"] = pd.to_datetime(
        wide_df["timestamp"], format="mixed", errors="raise"
    )
    feature_columns = [
        column for column in wide_df.columns if column not in ["deviceId", "timestamp"]
    ]
    aligned_parts = []

    for device_id, device_data in wide_df.groupby("deviceId"):
        # 每列取 [窗口起点, 窗口终点) 内最后一个非空值。
        window_df = (
            device_data.set_index("timestamp")[feature_columns]
            .sort_index()
            .resample(WINDOW_SIZE, closed="left", label="left")
            .last()
            .dropna(how="all")
        )
        window_df.insert(0, "deviceId", device_id)
        aligned_parts.append(window_df.reset_index())

    if not aligned_parts:
        raise ValueError("宽表没有可对齐的设备记录，请先检查输入文件。")
    aligned_df = pd.concat(aligned_parts, ignore_index=True)
    aligned_df = aligned_df[["deviceId", "timestamp", *feature_columns]]
    output_path = PROCESSED_DIR / "wide_10s.csv"
    aligned_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"对齐完成：{len(aligned_df)} 行；已保存到 {output_path}")


if __name__ == "__main__":
    main()
