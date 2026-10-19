"""验收通过后生成含输入、代码、文档、结果和逐文件校验清单的 ZIP。"""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from pipeline_config import ROOT, QUALITY, SOURCE_SHA256

ROLES = {
    "clean_wide.csv": "主交付：全量 10 秒窗口宽表，保留观测缺失",
    "analysis_core_windows.csv": "B 使用的核心变量观测子集，包含完整度标记",
    "missing_mask.csv": "主表逐单元格缺失标记，1=缺失",
    "alarm_records.csv": "C 使用的逐条原始告警编码记录及来源行号，包含编码 0",
    "alarm_windows.csv": "按设备和窗口关联的编码集合、记录数及歧义标记",
    "feature_lineage.csv": "每个非空测量单元格的观测时间、值及原始行号",
    "telemetry_long.csv": "保留全部原始列和编码的长表",
    "wide_unaligned.csv": "精确时间戳展开的中间宽表",
    "wide_10s.csv": "10 秒窗口对齐的中间宽表",
    "data_dictionary.csv": "主表 55 列的字段含义、来源、类型、聚合与缺失规则",
    "device_summary.csv": "设备对应关系、行数、变量数、时间跨度和核心观测数量",
    "missingness.csv": "精确时间及窗口时间的全局、分设备逐字段缺失率",
    "sampling_gaps.csv": "分设备分原始变量的采样间隔统计",
    "window_comparison.csv": "5、10、30、60 秒窗口的覆盖比较",
    "numeric_statistics.csv": "分设备及全局的范围、分位数、常量和极值核查",
    "alarm_class_summary.csv": "原始各编码记录数与占比，未赋予未确认的业务含义",
    "summary.json": "预处理统计和运行环境摘要",
    "validation_results.json": "程序验收的逐项条件、结果及数据指纹",
    "invalid_numeric_records.csv": "29 条数值转换失败记录",
    "unknown_variable_records.csv": "全部 35 条未知变量记录，与转换失败记录存在重叠",
    "numeric_review_candidates.csv": "统计极端值候选，仅核查，原数据保留",
    "preprocessing_report.html": "可浏览和打印的数据介绍、清洗规则、结果、限制与验收说明",
    "题2_车辆物联网遥测.csv": "用户提供的原始输入快照，不改写",
}


def main():
    validation = json.loads((QUALITY / "validation_results.json").read_text(encoding="utf-8"))
    current_clean = ROOT / "data/processed/clean_wide.csv"
    current_hash = hashlib.sha256(current_clean.read_bytes()).hexdigest()
    if not validation.get("passed") or validation.get("clean_sha256") != current_hash:
        raise SystemExit("请先运行 python run_pipeline.py，确认当前数据验收通过后再打包。")
    if hashlib.sha256((ROOT / "data/raw/题2_车辆物联网遥测.csv").read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit("原始文件指纹与本次交付数据不一致。")

    files = sorted(ROOT.glob("*.py")) + [ROOT / "README.md", ROOT / "requirements.txt", ROOT / ".gitignore"]
    files += sorted(path for path in (ROOT / "docs").rglob("*") if path.is_file())
    files += sorted(path for path in (ROOT / "data").rglob("*") if path.is_file())
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    archive_path = destination / "vehicle_iot_A_delivery.zip"
    with tempfile.TemporaryDirectory(prefix="a-delivery-", dir=destination) as temporary:
        stage = Path(temporary) / "vehicle_iot_A_delivery"
        stage.mkdir()
        manifest_rows = []
        for path in files:
            relative = path.relative_to(ROOT)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            row_count = None
            if path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    next(reader, None)
                    row_count = sum(1 for _ in reader)
            manifest_rows.append({
                "path": relative.as_posix(), "bytes": path.stat().st_size,
                "rows": row_count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "purpose": ROLES.get(path.name, "运行代码或项目说明"),
            })
        manifest = {
            "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "technical_checks_passed": validation["passed"], "check_count": validation["check_count"],
            "note": "清单不包含自身及验收清单；生成数据保留缺失，不保证官方语义已确认。",
            "pending_provider_information": ["POSITION 三项含义与顺序", "字段单位与状态编码", "告警 0–5 含义和标签产生规则", "原始 timestamp 时区"],
            "files": manifest_rows,
        }
        (stage / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with (stage / "验收清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "purpose", "rows", "bytes", "sha256", "用户验收结果", "备注"])
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow({**row, "用户验收结果": "", "备注": ""})
        temporary_zip = Path(temporary) / archive_path.name
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent).as_posix())
        shutil.copy2(temporary_zip, archive_path)
    print(f"交付包完成：{len(manifest_rows)} 个业务文件及两份验收清单；{archive_path}")


if __name__ == "__main__":
    main()
