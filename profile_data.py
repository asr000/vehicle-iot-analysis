"""生成数据质量统计、字段字典和可直接浏览的预处理报告。"""
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import platform

import numpy as np
import pandas as pd

from pipeline_config import (
    RAW_PATH, PROCESSED, REVIEW, QUALITY, DOCS, KEYS, CORE_FEATURES,
    POSITION_COLUMNS, WINDOW_SIZE, UNKNOWN_VARIABLE, read_timed_csv, save_csv,
)

MEANINGS = {
    "Vehicle speed": "车速", "ENGINE RPM": "发动机转速",
    "ACCELERATION X": "X 轴加速度", "ACCELERATION Y": "Y 轴加速度", "ACCELERATION Z": "Z 轴加速度",
    "INTERNAL BATTERY": "内部电池测量值，电量或电压含义未确认",
    "EXTERNAL BATTERY": "外部电池测量值，电量或电压含义未确认",
    "TOWING": "拖车相关状态", "IGNITION_STATUS": "点火相关状态", "IDLING": "怠速相关状态",
    "THROTTLE POSITION": "节气门位置", "ABSOLUTE THROTTLE POSITION B": "绝对节气门位置 B",
    "ABSOLUTE THROTTLE POSITION D": "绝对节气门位置 D", "ABSOLUTE THROTTLE POSITION E": "绝对节气门位置 E",
    "RELATIVE THROTTLE POSITION": "相对节气门位置", "COMMANDED THROTTLE ACTUATOR": "节气门执行器指令",
    "DISTANCE TRAVELED WITH MIL ON": "故障指示灯点亮期间行驶距离",
    "DISTANCE TRAVELED SINCE CODES CLEARED": "清除故障码后行驶距离",
    "AIR INTAKE TEMPERATURE": "进气温度", "ENGINE COOLANT TEMPERATURE": "发动机冷却液温度",
    "AMBIENT AIR TEMPERATURE": "环境空气温度", "FUEL SYSTEM STATUS": "燃油系统状态编码",
    "OBD STANDARDS THIS VEHICLE CONFORMS TO": "车辆符合的 OBD 标准编码",
    "INTAKE MANIFOLD ABSOLUTE PRESSURE": "进气歧管绝对压力",
    "LONG TERM FUEL TRIM BANK 1": "第一列气缸长期燃油修正",
    "SHORT TERM FUEL TRIM BANK 1": "第一列气缸短期燃油修正",
    "FUEL AIR COMMANDED EQUIVALENCE RATE": "燃油空气指令当量比",
    "ABSOLUTE LOAD VALUE": "绝对负荷值", "CALCULATED ENGINE LOAD": "计算发动机负荷",
    "CATALYST TEMPERATURE BANK 1 SENSOR 1": "第一列气缸第一传感器催化器温度",
    "WARM UPS SINCE CODES CLEARED": "清除故障码后暖机次数",
    "OXYGEN SENSOR 2 - B: SHORT TERM FUEL TRIM": "氧传感器 2 B 短期燃油修正",
    "OXYGEN SENSOR 1 - B: SHORT TERM FUEL TRIM": "氧传感器 1 B 短期燃油修正",
    "RUN TIME SINCE ENGINE START": "发动机启动后运行时间",
    "OXYGEN SENSORS PRESENT IN 2 BANKS": "两列气缸氧传感器配置编码",
    "TIMING ADVANCE": "点火提前相关值", "HARDWARE STATUS": "硬件状态编码",
    "COMMANDED EVAPORATIVE PURGE": "蒸发排放清除指令",
    "MONITOR STATUS THIS DRIVE CYCLE": "本驾驶周期监测状态编码",
    "MONITOR STATUS SINCE DTCS CLEARED": "清除故障码后监测状态编码",
    "CONTROL MODULE VOLTAGE": "控制模块电压相关值",
    "ABSOLULTE BAROMETRIC PRESSURE": "大气绝对压力，保留原字段拼写",
    "OXYGEN SENSOR 1 - AB: FUEL AIR EQUIVALENCE RATIO": "氧传感器 1 AB 燃油空气当量比",
    "FUEL TANK LEVEL INPUT": "燃油箱液位输入", "MAF AIR FLOW RATE": "空气质量流量",
    "FUEL RAIL GAUGE PRESSURE": "燃油轨表压",
    "DRIVER'S DEMAND ENGIN - PERCENT TORQUE": "驾驶员需求发动机扭矩相关值，保留原拼写",
    "ACTUAL ENGIN - PERCENT TORQUE": "实际发动机扭矩相关值，保留原拼写",
    "ENGINE REFERENCE TORQUE": "发动机参考扭矩", "FUEL TYPE": "燃料类型编码",
}
CODE_FEATURES = {
    "TOWING", "IGNITION_STATUS", "IDLING", "FUEL SYSTEM STATUS", "FUEL TYPE", "HARDWARE STATUS",
    "OBD STANDARDS THIS VEHICLE CONFORMS TO", "OXYGEN SENSORS PRESENT IN 2 BANKS",
    "MONITOR STATUS THIS DRIVE CYCLE", "MONITOR STATUS SINCE DTCS CLEARED",
}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality_tables(raw, exact, clean):
    features = [column for column in clean if column not in KEYS]
    devices = sorted(raw["deviceId"].unique())
    device_rows, coverage_rows, gap_rows = [], [], []
    for index, device in enumerate(devices, 1):
        source = raw.loc[raw["deviceId"].eq(device)]
        windows = clean.loc[clean["deviceId"].eq(device)]
        device_rows.append({
            "device_alias": f"车辆{index}", "deviceId": device, "raw_rows": len(source),
            "variables": source["variable"].nunique(), "first_timestamp": source["timestamp"].min(),
            "last_timestamp": source["timestamp"].max(), "observed_windows": len(windows),
            "any_core_windows": int(windows[CORE_FEATURES].notna().any(axis=1).sum()),
            "all_core_windows": int(windows[CORE_FEATURES].notna().all(axis=1).sum()),
        })
    for stage, frame in [("exact_timestamp", exact), ("10s_window", clean)]:
        for scope, group in [("ALL", frame), *list(frame.groupby("deviceId"))]:
            for feature in features:
                valid = int(group[feature].notna().sum())
                coverage_rows.append({
                    "stage": stage, "deviceId": scope, "feature": feature, "window_count": len(group),
                    "observed_count": valid, "missing_count": len(group) - valid,
                    "missing_pct": 100 * (1 - valid / len(group)),
                    "coverage_status": "never_observed" if valid == 0 else "observed",
                })
    for (device, variable), group in raw.groupby(["deviceId", "variable"]):
        times = group["timestamp"].sort_values()
        gaps = times.diff().dt.total_seconds().dropna()
        gap_rows.append({
            "deviceId": device, "variable": variable, "record_count": len(group),
            "median_gap_s": gaps.median(), "p90_gap_s": gaps.quantile(.9), "max_gap_s": gaps.max(),
            "gaps_over_10s": int(gaps.gt(10).sum()), "gaps_over_60s": int(gaps.gt(60).sum()),
            "gaps_over_300s": int(gaps.gt(300).sum()),
        })
    save_csv(pd.DataFrame(device_rows), QUALITY / "device_summary.csv")
    save_csv(pd.DataFrame(coverage_rows), QUALITY / "missingness.csv")
    save_csv(pd.DataFrame(gap_rows), QUALITY / "sampling_gaps.csv")

    comparison = []
    for interval in ["5s", "10s", "30s", "60s"]:
        windows = exact.assign(timestamp=exact["timestamp"].dt.floor(interval)).groupby(KEYS)[features].last().reset_index()
        for scope, group in [("ALL", windows), *list(windows.groupby("deviceId"))]:
            row = {"window_size": interval, "deviceId": scope, "observed_windows": len(group)}
            for feature in CORE_FEATURES:
                row[f"{feature}_valid_pct"] = 100 * group[feature].notna().mean()
            row["all_core_windows"] = int(group[CORE_FEATURES].notna().all(axis=1).sum())
            comparison.append(row)
    save_csv(pd.DataFrame(comparison), QUALITY / "window_comparison.csv")

    numeric = raw.loc[~raw["variable"].isin(["POSITION", UNKNOWN_VARIABLE])].copy()
    numeric["numeric_value"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.loc[numeric["numeric_value"].notna()]
    stats, candidates = [], []
    groups = [("ALL", feature, group) for feature, group in numeric.groupby("variable")]
    groups += [(device, feature, group) for (device, feature), group in numeric.groupby(["deviceId", "variable"])]
    for device, feature, group in groups:
        values = group["numeric_value"]
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        eligible = feature not in CODE_FEATURES and len(group) >= 20 and iqr > 0
        unusual = (values.lt(q1 - 1.5 * iqr) | values.gt(q3 + 1.5 * iqr)) if eligible else pd.Series(False, index=group.index)
        stats.append({
            "deviceId": device, "feature": feature, "count": len(values), "unique_values": values.nunique(),
            "min": values.min(), "p01": values.quantile(.01), "median": values.median(),
            "p99": values.quantile(.99), "max": values.max(), "negative_count": int(values.lt(0).sum()),
            "iqr_review_applicable": eligible, "iqr_review_count": int(unusual.sum()),
        })
        if device != "ALL" and unusual.any():
            marked = group.loc[unusual, ["deviceId", "timestamp", "variable", "value", "alarmClass"]].copy()
            marked.insert(0, "source_row_id", marked.index + 1)
            marked["review_reason"] = "device-variable 1.5 IQR; candidate only; retained unchanged"
            candidates.append(marked)
    save_csv(pd.DataFrame(stats), QUALITY / "numeric_statistics.csv")
    columns = ["source_row_id", "deviceId", "timestamp", "variable", "value", "alarmClass", "review_reason"]
    save_csv(pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(columns=columns), REVIEW / "numeric_review_candidates.csv")

    dictionary = []
    for feature in clean.columns:
        if feature in KEYS:
            meaning = "原始设备编号" if feature == "deviceId" else "10 秒窗口起点；原始时区未提供"
            kind, unit, source = "key", "不适用", feature
        elif feature in POSITION_COLUMNS:
            meaning = f"POSITION 按逗号拆分的第 {POSITION_COLUMNS.index(feature) + 1} 项；坐标顺序和物理含义待确认"
            kind, unit, source = "position_component", "未提供", "POSITION"
        else:
            meaning = MEANINGS.get(feature, "按原变量名保留") + "；中文解释仅依据字段名"
            kind = "code_or_state" if feature in CODE_FEATURES else "numeric_measurement"
            unit, source = "未提供，未转换单位", feature
        dictionary.append({
            "column": feature, "role": kind, "meaning": meaning, "unit": unit,
            "source": source, "dtype": str(clean[feature].dtype),
            "aggregation": "key" if feature in KEYS else "last non-null observation inside left-closed 10s window",
            "missing_rule": "键不允许缺失" if feature in KEYS else "该窗口无观测则为空；不补零、不插值、不跨窗口沿用",
            "semantic_status": "source field" if feature in KEYS else "official dictionary not supplied",
        })
    save_csv(pd.DataFrame(dictionary), QUALITY / "data_dictionary.csv")


def render_report():
    summary = json.loads((QUALITY / "summary.json").read_text(encoding="utf-8"))
    devices = pd.read_csv(QUALITY / "device_summary.csv")
    coverage = pd.read_csv(QUALITY / "missingness.csv")
    core = coverage.loc[coverage.stage.eq("10s_window") & coverage.feature.isin(CORE_FEATURES + ["position_1"])]
    core = core.pivot(index="deviceId", columns="feature", values="missing_pct").round(2).reset_index()
    core["deviceId"] = core["deviceId"].replace(dict(zip(devices.deviceId, devices.device_alias)))
    core["deviceId"] = core["deviceId"].replace({"ALL": "全量"})
    core.columns.name = None
    core = core.rename(columns={"deviceId": "设备", "Vehicle speed": "车速缺失%", "ENGINE RPM": "转速缺失%", "ACCELERATION X": "X加速度缺失%", "ACCELERATION Y": "Y加速度缺失%", "ACCELERATION Z": "Z加速度缺失%", "position_1": "位置分量1缺失%"})
    mapping = devices[["device_alias", "deviceId"]].rename(columns={"device_alias": "设备别名", "deviceId": "原始设备编号"})
    devices = devices.drop(columns="deviceId").rename(columns={
        "device_alias": "设备", "raw_rows": "原始记录数", "variables": "变量数", "first_timestamp": "起始时间",
        "last_timestamp": "结束时间", "observed_windows": "观测窗口数", "any_core_windows": "核心子集窗口", "all_core_windows": "核心五项齐全",
    })
    comparison = pd.read_csv(QUALITY / "window_comparison.csv")
    comparison = comparison.loc[comparison.deviceId.eq("ALL")].drop(columns="deviceId").round(2)
    comparison = comparison.rename(columns={"window_size": "窗口", "observed_windows": "观测窗口数", "Vehicle speed_valid_pct": "车速有效%", "ENGINE RPM_valid_pct": "转速有效%", "ACCELERATION X_valid_pct": "X加速度有效%", "ACCELERATION Y_valid_pct": "Y加速度有效%", "ACCELERATION Z_valid_pct": "Z加速度有效%", "all_core_windows": "核心五项齐全"})
    alarm = pd.read_csv(QUALITY / "alarm_class_summary.csv")
    alarm = alarm.round(2).rename(columns={"alarmClass": "原始编码", "record_count": "标签记录数", "record_pct": "记录占比%", "meaning": "业务含义"})
    stats = pd.read_csv(QUALITY / "numeric_statistics.csv")
    stats = stats.loc[stats.deviceId.eq("ALL"), ["feature", "count", "unique_values", "min", "median", "max", "iqr_review_count"]]
    stats = stats.rename(columns={"feature": "原始变量名", "count": "数值记录数", "unique_values": "不同值数量", "min": "最小值", "median": "中位数", "max": "最大值", "iqr_review_count": "全局IQR候选数"})
    verification_path = QUALITY / "validation_results.json"
    verified = False
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verified = verification.get("passed", False) and verification.get("clean_sha256") == fingerprint(PROCESSED / "clean_wide.csv")
    def table(frame):
        return frame.to_html(index=False, border=0, escape=True, classes="data-table", na_rep="—")
    status = "程序验收已通过" if verified else "待运行 validate_data.py 完成程序验收"
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>车辆物联网 A 数据工程交付报告</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.75 'Microsoft YaHei',sans-serif;color:#17253a;background:#f3f6fb;margin:0}}
main{{width:calc(100% - 32px);max-width:1248px;margin:28px auto;background:white;padding:30px 32px;border-radius:12px}}
h1{{font-size:29px;margin:0 0 8px}}h2{{font-size:23px;margin-top:34px;border-bottom:2px solid #dce7f5;padding-bottom:7px}}
p,li{{overflow-wrap:anywhere}}.meta{{color:#54657a}}.callout{{padding:16px 20px;background:#eaf3ff;border-left:4px solid #2668ad}}
.notice{{background:#fff4dc;border-left:4px solid #bd7a13;padding:14px 20px}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;table-layout:fixed;width:100%;font-size:13px;margin:15px 0}}th,td{{border:1px solid #d8e0ea;padding:9px 10px;text-align:left;overflow-wrap:anywhere}}
th{{background:#edf3fa}}tr:nth-child(even){{background:#fafcff}}code{{background:#edf1f7;padding:2px 5px}}pre{{background:#17253a;color:#f1f6ff;padding:18px;overflow:auto}}
summary{{cursor:pointer;font-weight:bold;padding:10px;background:#f3f6fb}}a{{color:#1c62aa}}@media print{{body{{background:white}}main{{margin:0;padding:0;max-width:none}}.scroll{{overflow:visible}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}
</style></head><body><main><h1>车辆物联网 A 数据工程交付报告</h1>
<p class="meta">生成时间：{html.escape(summary['generated_at'])}　环境：Python {summary['python']} / pandas {summary['pandas']} / NumPy {summary['numpy']}</p>
<div class="callout"><strong>{status}。</strong>正式宽表包含 {summary['clean_rows']:,} 个设备时间窗口、{summary['clean_columns']} 列；保留可解释的缺失值和原始告警编码，并提供逐单元格来源与程序验收结果。</div>
<h2>数据介绍</h2><p>数据由用户提供，文件名为“题2_车辆物联网遥测.csv”，未提供公开下载地址或官方字段字典。共 {summary['raw_rows']:,} 条记录、6 列、3 个设备、52 种变量。时间范围为 {summary['first_timestamp']} 至 {summary['last_timestamp']}，时区未标注。原始文件 SHA256：<code>{summary['source_sha256']}</code>。</p>
{table(devices)}<details><summary>原始设备编号对应表</summary>{table(mapping)}</details>
<h2>清洗规则与处理结果</h2><ol>
<li>时间：所有 timestamp 均可解析。timeMili 缺失 {summary['timeMili_missing']:,} 条，保留原值；{summary['time_comparison_count']:,} 条可比较记录的钟面差均为 5 小时 30 分钟。未据此擅自指定地理时区。</li>
<li>重复：完全重复和“设备、时间、变量”重复均为 0，没有执行重复删除。</li>
<li>位置：90,355 条 POSITION 均可拆成三个数值，暂命名 position_1、position_2、position_3。经纬度顺序及第三项含义等待字段说明，不将候选解释写成已知事实。</li>
<li>数值：83,498 条非位置记录中有 29 条转换失败，全部属于 PIDNameNotAvailable。该变量共 35 条，全部退出数值宽表并单独保存；其余 83,463 条数值记录保留。异常文件之间存在重叠，不可相加计算删除数量。</li>
<li>时间窗口：同一设备独立处理，窗口为 [起点, 起点+10秒)，每个字段取窗口内最后一个非空值，删除全部测量字段为空的窗口；不跨设备、不跨窗口传递数值。</li>
<li>缺失：无观测保留为空，不补零、不做无限前向填充或插值。missing_mask.csv 中 1 表示缺失、0 表示有观测。未报告的时间段不生成人造完整轨迹。</li>
<li>数值核查：提供分设备分变量的范围、分位数、负值和常量统计。对适用的连续数值使用 1.5 IQR 标记待核查候选，全部保留原值；统计极端值不等于传感器错误，未按未经确认的单位阈值裁剪。</li></ol>
<h2>覆盖率与窗口选择</h2><p>下表是各字段的<strong>缺失百分比</strong>，分母为对应设备实际存在遥测记录的 10 秒窗口。长表的 value 无缺失，不代表每个变量在每个窗口都有观测。没有任何遥测记录的完整空档未计入该分母。</p>
<div class="scroll">{table(core)}</div>
<p>10 秒是本次交付的工程基线，依据主要数值变量常见约 5 秒、位置常见约 3 秒的采样间隔；不是经下游预测效果证明的最优窗口。不同窗口下的覆盖情况如下，百分比表示有效观测覆盖：</p>
<div class="scroll">{table(comparison)}</div>
<p>另提供 analysis_core_windows.csv，仅筛选至少具有一项车速、转速、三轴加速度观测的 {summary['core_rows']:,} 个窗口，其中五项齐全的有 {summary['core_complete_rows']:,} 个。该子集与主表的筛选差异明确记录，不能把它的缺失率当成全量数据的缺失率。</p>
<h2>告警编码与关联规则</h2><div class="notice">发现 {summary['same_time_code_conflicts']:,} 个同设备同时间记录组存在不同编码，{summary['multi_code_windows']:,} 个 10 秒窗口包含多个编码。未定义编码 0 为正常，未取最大值或多数票制造单一类别。</div>
{table(alarm)}
<p>alarm_records.csv 保留全部 {summary['raw_rows']:,} 条原始标签记录及原始数据行号，包含编码 0；它们不是已经去重识别的独立告警事件。alarm_windows.csv 按 deviceId、timestamp 与 clean_wide.csv 关联，保留每个编码的记录数、编码集合和 multi_code。single_alarmClass 仅在窗口内只有一个编码时填写，多编码窗口保持为空。编码记录数不等于独立告警发生次数。</p>
<p>C 应结合完整告警明细定义事件或分类样本；做告警前状态预测时只能使用事件发生前的原始观测。整窗汇总可能包含窗口后部观测，不能直接作为窗口起点或较早事件的预测输入。编码汇总列也不应作为输入特征泄漏标签。</p>
<h2>字段说明与数值核查</h2><p>正式宽表共 2 个键和 53 个测量字段。完整字典见 <a href="../data/quality/data_dictionary.csv">data_dictionary.csv</a>；中文解释依据原变量名，不代表官方单位或编码释义。</p>
<details><summary>展开 50 种数值变量的统计范围与核查数量</summary><div class="scroll">{table(stats)}</div></details>
<p>位置字段含义、所有未提供的单位、告警 0–5 的业务解释、部分状态编码含义、OBD 哨兵值及设备与真实车辆的对应关系仍需提供方确认。现有程序保留这些字段和原始值，因此取得说明后可更新字典，不需要恢复已被破坏的数据。</p>
<h2>复现与验收</h2><p>推荐 Python 3.11，实际验证版本为 {summary['python']}。终端已进入 course 环境时：</p>
<pre>python -m pip install -r requirements.txt
python run_pipeline.py
python build_delivery.py</pre>
<p>run_pipeline.py 顺序执行预处理、对齐、正式数据、质量报告和验收；每个处理阶段只输出一行摘要。inspect_data.py 是单独的只读检查脚本，不由流水线自动执行。</p>
<ol><li>确认 clean_wide.csv 的设备和时间键唯一、窗口起点对齐至 10 秒。</li>
<li>随机选择宽表非空单元格，在 feature_lineage.csv 找到同设备同窗口同字段记录，再按 source_row_id 定位原始 CSV 数据行；原文件含表头时物理行号为 source_row_id+1。</li>
<li>确认来源时间在窗口内，值等于该窗口最后一个有效原始观测；空单元格对应缺失标记 1。</li>
<li>检查告警各编码记录数合计与原始各编码数量一致，歧义窗口的单一标签为空。</li>
<li>查看 <a href="../data/quality/validation_results.json">validation_results.json</a> 中的实际检查项和结果；用交付清单核对文件大小与 SHA256。</li></ol>
<p>GitHub 保存代码、文档和依赖；原始及逐条测量数据不纳入 Git 跟踪，完整输入与生成结果随离线交付包提供。</p>
<h2>A 部分交付边界</h2><p>已完成可复现的数据整理、异常隔离、缺失规则、窗口关联、字段字典、预处理报告和程序验收。官方语义仍未提供的字段保持待确认。运行特征结论、独立告警事件识别、模型训练与效果评估由 B、C 后续完成，不将本次数据处理验收等同于建模效果验收。</p>
</main></body></html>"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "preprocessing_report.html").write_text(body, encoding="utf-8")


def main():
    QUALITY.mkdir(parents=True, exist_ok=True)
    raw = read_timed_csv(RAW_PATH)
    exact = read_timed_csv(PROCESSED / "wide_unaligned.csv")
    clean = read_timed_csv(PROCESSED / "clean_wide.csv")
    quality_tables(raw, exact, clean)
    code_summary = raw.groupby("alarmClass").size().rename("record_count").reset_index()
    code_summary["record_pct"] = 100 * code_summary["record_count"] / len(raw)
    code_summary["meaning"] = "未提供；不推断为正常或某种故障"
    save_csv(code_summary, QUALITY / "alarm_class_summary.csv")
    same_time = raw.groupby(KEYS)["alarmClass"].nunique()
    alarms = read_timed_csv(PROCESSED / "alarm_windows.csv")
    core_mask = clean[CORE_FEATURES].notna()
    summary = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "source_sha256": fingerprint(RAW_PATH), "raw_rows": len(raw),
        "first_timestamp": str(raw["timestamp"].min()), "last_timestamp": str(raw["timestamp"].max()),
        "timeMili_missing": int(raw["timeMili"].isna().sum()),
        "time_comparison_count": int(raw["timeMili"].notna().sum()),
        "clean_rows": len(clean), "clean_columns": len(clean.columns),
        "core_rows": int(core_mask.any(axis=1).sum()), "core_complete_rows": int(core_mask.all(axis=1).sum()),
        "same_time_code_conflicts": int(same_time.gt(1).sum()),
        "multi_code_windows": int(alarms["multi_code"].sum()),
        "window_size": WINDOW_SIZE, "missing_policy": "retain; no zero-fill, interpolation or cross-window carry",
    }
    (QUALITY / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report()
    print("质量报告完成：已生成字段字典、覆盖率、采样间隔、窗口对比、数值核查和 HTML 报告。")


if __name__ == "__main__":
    main()
