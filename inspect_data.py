"""合并后的检查与报告脚本：统一统计、校验和字段说明，生成一份质量数据和一份报告。"""
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import platform

import numpy as np
import pandas as pd

from prepare_data import (
    ROOT, RAW_PATH, PROCESSED, REVIEW, QUALITY, DOCS, KEYS, CORE_FEATURES,
    POSITION_COLUMNS, WINDOW_SIZE, UNKNOWN_VARIABLE, SOURCE_SHA256, CODE_FEATURES,
    read_timed_csv, sensor_columns, build_exact,
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

def collect_tables(raw, exact, clean):
    tables = {}
    def store_table(frame, path):
        if path.stem != "numeric_review_candidates":
            tables[path.stem] = frame
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
    store_table(pd.DataFrame(device_rows), QUALITY / "device_summary.csv")
    store_table(pd.DataFrame(coverage_rows), QUALITY / "missingness.csv")
    store_table(pd.DataFrame(gap_rows), QUALITY / "sampling_gaps.csv")

    comparison = []
    for interval in ["5s", "10s", "30s", "60s"]:
        windows = exact.assign(timestamp=exact["timestamp"].dt.floor(interval)).groupby(KEYS)[features].last().reset_index()
        for scope, group in [("ALL", windows), *list(windows.groupby("deviceId"))]:
            row = {"window_size": interval, "deviceId": scope, "observed_windows": len(group)}
            for feature in CORE_FEATURES:
                row[f"{feature}_valid_pct"] = 100 * group[feature].notna().mean()
            row["all_core_windows"] = int(group[CORE_FEATURES].notna().all(axis=1).sum())
            comparison.append(row)
    store_table(pd.DataFrame(comparison), QUALITY / "window_comparison.csv")

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
    store_table(pd.DataFrame(stats), QUALITY / "numeric_statistics.csv")
    columns = ["source_row_id", "deviceId", "timestamp", "variable", "value", "alarmClass", "review_reason"]
    store_table(pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(columns=columns), REVIEW / "numeric_review_candidates.csv")

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
    store_table(pd.DataFrame(dictionary), QUALITY / "data_dictionary.csv")
    return tables


def validate(raw, full, long, review):
    results = []
    def check(name, condition, detail):
        results.append({"check": name, "passed": bool(condition), "detail": str(detail)})
    features = sensor_columns(full)
    source_hash = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    check("source_fingerprint", source_hash == SOURCE_SHA256, source_hash)
    check("source_snapshot", (len(raw), raw.deviceId.nunique(), raw.variable.nunique()) == (173853, 3, 52), "173853 rows / 3 devices / 52 variables")
    check("valid_source_keys", not raw[[*KEYS, "variable"]].isna().any().any(), "keys non-null")
    check("unique_source_keys", not raw.duplicated([*KEYS, "variable"]).any(), "source keys unique")
    comparable = raw.loc[raw.timeMili.notna()]
    utc = pd.to_datetime(comparable.timeMili, unit="ms", utc=True).dt.tz_localize(None)
    check("time_offset", len(comparable) == 52202 and (comparable.timestamp - utc).dt.total_seconds().eq(19800).all(), "52202 comparable rows; 19800 clock seconds")
    check("unique_final_keys", not full.duplicated(KEYS).any(), f"{len(full)} unique windows")
    check("window_boundaries", full.timestamp.eq(full.timestamp.dt.floor(WINDOW_SIZE)).all(), WINDOW_SIZE)
    check("merged_schema", len(features) == 53 and len(full.columns) == 69 and UNKNOWN_VARIABLE not in full, "53 measurements + 2 keys + 14 summary fields")
    cells = full[features].to_numpy(dtype=float)
    check("finite_observed_values", np.isfinite(cells[~np.isnan(cells)]).all(), "all observed measurements finite")
    check("nonempty_measurement_windows", not full[features].isna().all(axis=1).any(), "each final window has measurements")

    # 从原始记录独立建立“设备/窗口/字段 -> 最后有效观测”的索引。
    source = raw.assign(source_row_id=np.arange(1, len(raw) + 1), window_start=raw.timestamp.dt.floor(WINDOW_SIZE))
    expected = {}
    for row in source.itertuples(index=False):
        if row.variable == UNKNOWN_VARIABLE:
            continue
        if row.variable == "POSITION":
            values = zip(POSITION_COLUMNS, map(float, row.value.split(",")))
        else:
            try:
                values = [(row.variable, float(row.value))]
            except (ValueError, TypeError):
                continue
        for feature, value in values:
            if not np.isfinite(value):
                continue
            key = (row.deviceId, row.window_start, feature)
            candidate = (row.timestamp, row.source_row_id, value)
            if key not in expected or candidate[:2] > expected[key][:2]:
                expected[key] = candidate
    actual = full.melt(id_vars=KEYS, value_vars=features, var_name="feature", value_name="value").dropna(subset=["value"])
    matched = len(actual) == len(expected)
    for row in actual.itertuples(index=False):
        item = expected.get((row.deviceId, row.timestamp, row.feature))
        if item is None or not np.isclose(row.value, item[2], rtol=1e-12, atol=1e-12):
            matched = False
            break
    check("all_cells_match_last_original_observation", matched, f"{len(actual)} non-null cells; no imputation or cross-window carry")
    check("all_sources_inside_window", all(start <= item[0] < start + pd.Timedelta(WINDOW_SIZE) for (_, start, _), item in expected.items()), "all selected observation times inside their own window")
    check("core_count", full.core_feature_count.eq(full[CORE_FEATURES].notna().sum(axis=1)).all(), "core subset membership can be reconstructed from main table")
    check("core_complete_flag", full.has_all_core_features.eq(full[CORE_FEATURES].notna().all(axis=1).astype(int)).all(), "five-core completeness flag matches observations")
    check("full_long_preserved", long[raw.columns].equals(raw), "all source columns and rows preserved")
    check("long_row_ids", long.source_row_id.tolist() == list(range(1, len(raw) + 1)), "data row id + 1 equals original CSV physical line")
    check("long_window_start", pd.to_datetime(long.window_start, format="mixed").eq(raw.timestamp.dt.floor(WINDOW_SIZE)).all(), "long table associates with main window keys")
    count_columns = [c for c in full if c.startswith("code_") and c.endswith("_count") and c.split("_")[1].isdigit()]
    totals = {int(c.split("_")[1]): int(full[c].sum()) for c in count_columns}
    expected_totals = {int(code): int(count) for code, count in raw.alarmClass.value_counts().items()}
    check("global_label_counts", totals == expected_totals and int(full.record_count.sum()) == len(raw), totals)
    expected_counts = source.groupby(["deviceId", "window_start", "alarmClass"]).size()
    indexed = full.set_index(KEYS)
    local_ok = all(indexed.loc[(device, start), f"code_{int(code)}_count"] == count for (device, start, code), count in expected_counts.items())
    check("per_window_label_counts", local_ok, "every original device/window/code count is preserved")
    original_keys = pd.MultiIndex.from_frame(source.rename(columns={"timestamp": "observed_at", "window_start": "timestamp"})[KEYS]).unique().sort_values()
    final_keys = pd.MultiIndex.from_frame(full[KEYS]).sort_values()
    check("label_window_coverage", original_keys.equals(final_keys), "main table covers every original labeled window")
    multi = full.code_count.gt(1)
    check("multicode_not_collapsed", full.loc[multi, "single_alarmClass"].isna().all() and full.multi_code.eq(multi.astype(int)).all(), f"{int(multi.sum())} multicode windows")
    check("single_code_consistent", full.loc[~multi, "single_alarmClass"].astype(int).astype(str).eq(full.loc[~multi, "code_values"].astype(str)).all(), "single code agrees with the retained set")
    expected_sets = ["|".join(c.split("_")[1] for c in count_columns if row[c] > 0) for _, row in full[count_columns].iterrows()]
    check("all_code_sets_consistent", full.code_values.tolist() == expected_sets, "multicode sets are conserved, including 0")
    check("label_count_metadata", full.code_count.eq(full[count_columns].gt(0).sum(axis=1)).all() and full.record_count.eq(full[count_columns].sum(axis=1)).all(), "count metadata reconciles")
    unknown_ids = set(source.loc[source.variable.eq(UNKNOWN_VARIABLE), "source_row_id"])
    review_unknown = set(review.loc[review.review_reason.str.contains("unknown_variable"), "source_row_id"])
    check("unknown_records_preserved", unknown_ids == review_unknown and len(review_unknown) == 35, "all 35 unknown-variable records in unified review table")
    check("review_reasons_and_ids", review.source_row_id.is_unique and review.source_row_id.between(1, len(raw)).all() and review.review_reason.str.contains("numeric_parse_failed").sum() == 29, "review rows are unique; 29 parse failures remain explicitly marked")
    return {
        "passed": all(item["passed"] for item in results), "check_count": len(results), "checks": results,
        "source_sha256": source_hash,
        "clean_sha256": hashlib.sha256((PROCESSED / "clean_wide.csv").read_bytes()).hexdigest(),
        "checked_measurement_cells": len(actual),
    }


def render_report(report):
    summary, tables = report["summary"], report["tables"]
    def table(name):
        frame = pd.DataFrame(tables[name])
        return frame.to_html(index=False, border=0, escape=True, na_rep="—", classes="data-table")
    devices = pd.DataFrame(tables["device_summary"])
    cover = pd.DataFrame(tables["missingness"])
    cover = cover.loc[cover.stage.eq("10s_window") & cover.feature.isin(CORE_FEATURES + ["position_1"])]
    cover = cover.pivot(index="deviceId", columns="feature", values="missing_pct").round(2).reset_index()
    cover.columns.name = None
    cover["deviceId"] = cover["deviceId"].replace(dict(zip(devices.deviceId, devices.device_alias))).replace({"ALL": "全量"})
    status = "程序验收通过" if report["validation"]["passed"] else "程序验收未通过"
    detail_tables = {
        "data_dictionary": "主表字段字典（69列）",
        "sampling_gaps": "采样间隔统计",
        "window_comparison": "不同窗口的覆盖比较",
        "numeric_statistics": "数值范围与极值核查",
        "alarm_class_summary": "原始编码分布",
        "review_summary": "统一核查表中的原因统计",
    }
    details = "".join(f"<details><summary>{title}</summary><div class='scroll'>{table(name)}</div></details>" for name, title in detail_tables.items())
    validation_frame = pd.DataFrame(report["validation"]["checks"])
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>车辆物联网数据预处理报告</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.8 'Microsoft YaHei',sans-serif;background:#f3f6fb;color:#17253a;margin:0}}
main{{max-width:1240px;width:calc(100% - 32px);margin:24px auto;background:white;padding:30px;border-radius:12px}}
h1{{font-size:28px}}h2{{font-size:22px;margin-top:30px;border-bottom:1px solid #d6e0eb}}
p,li,td,th{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px;margin:12px 0}}
td,th{{border:1px solid #d8e0ea;padding:8px;text-align:left}}th{{background:#edf3fa}}summary{{padding:12px;background:#edf3fa;cursor:pointer;font-weight:bold;margin-top:12px}}
.scroll{{overflow:auto}}.note{{background:#eaf3ff;padding:14px;border-left:4px solid #2668ad}}.pending{{background:#fff4dc;padding:14px}}
code{{background:#edf1f7;padding:2px 4px}}pre{{background:#17253a;color:white;padding:16px;overflow:auto}}
@media print{{main{{width:100%;margin:0;padding:0}}body{{background:white}}tr{{break-inside:avoid}}}}
</style></head><body><main><h1>车辆物联网数据预处理报告</h1>
<p>{html.escape(summary['generated_at'])}　Python {summary['python']} / pandas {summary['pandas']} / NumPy {summary['numpy']}</p>
<div class="note"><b>{status}：{report['validation']['check_count']}项。</b>主表 {summary['clean_rows']:,} 行、69列；原始测量值未变。主表已合并告警汇总与核心完整度，长表已合并原始告警明细与来源行号。</div>
<h2>数据介绍与处理结果</h2>
<p>原始文件由用户提供，共173,853条、3个设备、52种变量。未附公开下载地址或官方字段字典。数据时间为 {summary['first_timestamp']} 至 {summary['last_timestamp']}，时区未标注。</p>
{table('device_summary')}
<ol><li>timestamp全部可解析。timeMili缺失121,651条；52,202条可比较记录与其UTC时间的钟面差均为5小时30分钟，不据此指定原始地理时区。</li>
<li>完全重复和设备、时间、变量重复均为0。</li><li>90,355条POSITION均拆成三个数值，暂用position_1/2/3；原始顺序保留。</li>
<li>非位置记录中29条数值转换失败，均属于PIDNameNotAvailable；该变量35条全部不进入数值宽表，原始记录保留在长表和统一核查表。</li>
<li>按设备独立整理为左闭右开的10秒窗口，每列取窗口内最后一个非空值；不补零、不插值、不跨窗口或跨设备沿用。</li>
<li>缺失标记可用主表测量字段的isna()直接得到。核心变量子集可按core_feature_count &gt; 0筛选；五项齐全可按has_all_core_features == 1筛选。</li>
<li>统计极值只作核查候选，不据此删除原始观测。多个核查原因合在review_reason一列，用竖线连接；同一原始记录只保留一行。</li></ol>
<h2>合并后的两份数据</h2>
<p><b>clean_wide.csv</b>：设备和窗口两列键、53个测量字段、12个告警汇总字段及2个核心完整度字段。只有53个测量字段适合作为候选输入；告警汇总属于目标相关信息，建模时不能混入输入特征。</p>
<p><b>telemetry_long.csv</b>：全部173,853条原始记录、原始列、告警编码及source_row_id和window_start。source_row_id从1开始；带表头的原CSV物理行号为该值+1。</p>
<p>主表每行表示窗口内汇总，并不表示所有变量在窗口起点同时采到。告警预测应使用事件之前的原始观测；整窗汇总可能包含较晚观测。</p>
<h2>覆盖情况</h2><p>以下为缺失百分比，分母是有遥测记录的设备窗口，不包括完全没有上报的日历空档。10秒是工程基线，未宣称经模型验证为最优。</p>
<div class="scroll">{cover.to_html(index=False,border=0)}</div>
<p>核心变量至少一项有值的窗口为 {summary['core_rows']:,} 个，五项齐全为 {summary['core_complete_rows']:,} 个。两类筛选都可以从同一主表重建，因此不再重复保存子集文件。</p>
<h2>告警与核查</h2><p>发现12,134个同设备同时间组带有不同编码，8,799个窗口包含多个编码。code_0_count至code_5_count是标签记录数，不是独立告警事件次数。code_values保留全部编码；multi_code为1时single_alarmClass为空，不强制选最大值或多数票，也不将0直接解释为正常。</p>
<p>核查表共 {summary['review_rows']:,} 条不同原始记录，包含35条未知变量记录和3,905条统计极值候选；其中29条未知变量记录同时带有数值转换失败标记。完整原因统计见下方。</p>
{details}
<details><summary>程序验收结果（{report['validation']['check_count']}项）</summary>{validation_frame.to_html(index=False,border=0)}</details>
<h2>复现</h2><pre>python prepare_data.py
python inspect_data.py
python build_delivery.py</pre>
<p>固定依赖记录在requirements.txt，推荐Python3.11；实际验证为Python {summary['python']}。打包前检查当前主表与验收指纹一致。</p>
<div class="pending"><b>待提供方确认：</b>POSITION顺序与物理含义、各字段单位和状态编码、告警0–5含义及标注规则、原始timestamp时区。这些信息未用猜测补齐；程序验收验证数据一致性，不替代官方语义确认。</div>
</main></body></html>"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "preprocessing_report.html").write_text(body, encoding="utf-8")


def main():
    raw = read_timed_csv(RAW_PATH)
    full = read_timed_csv(PROCESSED / "clean_wide.csv")
    long = read_timed_csv(PROCESSED / "telemetry_long.csv")
    review = read_timed_csv(REVIEW / "review_records.csv")
    features = sensor_columns(full)
    measurements = full[KEYS + features]
    tables = collect_tables(raw, build_exact(raw), measurements)
    meta_meanings = {
        "window_end": "窗口右边界，不包含该时刻", "record_count": "窗口原始标签记录总数，不是独立事件次数",
        "code_count": "窗口不同编码数量", "code_values": "窗口编码集合，竖线分隔，包含0",
        "single_alarmClass": "只有一个编码时填写，多编码时为空，编码业务含义未确认",
        "multi_code": "1表示多编码，0表示编码唯一", "core_feature_count": "五项核心变量中有观测的数量",
        "has_all_core_features": "五项核心变量是否全部有观测，1是0否",
    }
    metadata = []
    for column in full:
        if column in features or column in KEYS:
            continue
        metadata.append({
            "column": column, "role": "window_metadata" if column == "window_end" else ("coverage_metadata" if column in ["core_feature_count", "has_all_core_features"] else "label_summary"),
            "meaning": meta_meanings.get(column, "对应原始编码在窗口内的标签记录数，不是独立事件次数"),
            "unit": "日期时间，时区未提供" if column == "window_end" else "记录数或编码，详见meaning",
            "source": "窗口起点加10秒" if column == "window_end" else ("五项核心字段有效性" if column in ["core_feature_count", "has_all_core_features"] else "原始alarmClass"),
            "dtype": str(full[column].dtype), "aggregation": "window summary",
            "missing_rule": "single_alarmClass在多编码窗口为空；计数0表示该编码无记录",
            "semantic_status": "derived metadata; alarm business meanings not supplied",
        })
    tables["data_dictionary"] = pd.concat([tables["data_dictionary"], pd.DataFrame(metadata)], ignore_index=True)
    code_summary = raw.groupby("alarmClass").size().rename("record_count").reset_index()
    code_summary["record_pct"] = 100 * code_summary.record_count / len(raw)
    code_summary["meaning"] = "未提供，不推断正常或故障"
    tables["alarm_class_summary"] = code_summary
    tables["review_summary"] = review.review_reason.str.split("|",regex=False).explode().value_counts().rename_axis("reason").reset_index(name="record_count")
    validation = validate(raw, full, long, review)
    summary = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "first_timestamp": str(raw.timestamp.min()), "last_timestamp": str(raw.timestamp.max()),
        "clean_rows": len(full), "clean_columns": len(full.columns), "source_rows": len(raw),
        "core_rows": int(full.core_feature_count.gt(0).sum()), "core_complete_rows": int(full.has_all_core_features.sum()),
        "review_rows": len(review), "source_sha256": SOURCE_SHA256, "window_size": WINDOW_SIZE,
    }
    report = {"summary": summary, "validation": validation, "tables": {
        name: json.loads(frame.to_json(orient="records",date_format="iso",double_precision=15))
        for name,frame in tables.items()
    }}
    QUALITY.mkdir(parents=True, exist_ok=True)
    (QUALITY / "quality_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    render_report(report)
    if not validation["passed"]:
        raise SystemExit("验收失败，详见质量报告中的validation。")
    print(f"检查通过：{validation['check_count']}项；全部统计合并到quality_report.json，字段说明已合并进HTML报告。")


if __name__ == "__main__":
    main()
