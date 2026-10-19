"""把合并后的12份业务文件和1份验收清单打成ZIP。"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from prepare_data import ROOT, RAW_PATH, PROCESSED, QUALITY, SOURCE_SHA256

FILES = {
    "README.md": "项目说明与运行方式",
    "requirements.txt": "固定运行依赖",
    ".gitignore": "Git排除规则",
    "prepare_data.py": "统一数据处理",
    "inspect_data.py": "统一统计、检查与报告",
    "build_delivery.py": "打包",
    "docs/preprocessing_report.html": "合并了字段说明、统计表和验收结果的可读报告",
    "data/raw/题2_车辆物联网遥测.csv": "原始输入，保持不变",
    "data/processed/clean_wide.csv": "主宽表，合并测量、告警汇总和核心完整度",
    "data/processed/telemetry_long.csv": "完整长表，合并原始告警明细和来源行号",
    "data/review/review_records.csv": "统一核查表，一行原始记录可带多个原因",
    "data/quality/quality_report.json": "统一质量数据，包含原先分散的统计、字典和验收结果",
}


def main():
    report = json.loads((QUALITY / "quality_report.json").read_text(encoding="utf-8"))
    checks = report["validation"]
    clean_hash = hashlib.sha256((PROCESSED / "clean_wide.csv").read_bytes()).hexdigest()
    if not checks["passed"] or checks["clean_sha256"] != clean_hash:
        raise SystemExit("请先运行prepare_data.py和inspect_data.py，确认当前数据验收通过。")
    if hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit("原始输入指纹不一致。")
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    output = destination / "vehicle_iot_A_delivery.zip"
    with tempfile.TemporaryDirectory(prefix="merged-delivery-", dir=destination) as temporary:
        stage = Path(temporary) / "vehicle_iot_A_delivery"
        stage.mkdir()
        rows = []
        for relative, purpose in FILES.items():
            source = ROOT / relative
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            count = None
            if source.suffix == ".csv":
                with source.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    next(reader, None)
                    count = sum(1 for _ in reader)
            rows.append({
                "path": relative, "purpose": purpose, "rows": count, "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "用户验收结果": "", "备注": "",
            })
        with (stage / "验收清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_zip = Path(temporary) / output.name
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent).as_posix())
        shutil.copy2(temporary_zip, output)
    print(f"打包完成：共13个文件；{output}")


if __name__ == "__main__":
    main()
