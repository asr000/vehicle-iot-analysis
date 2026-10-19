"""对最终数据执行可重复的来源、缺失、标签关联和结构验收。"""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from pipeline_config import (
    RAW_PATH, PROCESSED, QUALITY, KEYS, CORE_FEATURES, POSITION_COLUMNS,
    WINDOW_SIZE, UNKNOWN_VARIABLE, SOURCE_SHA256, read_timed_csv,
)


def main():
    QUALITY.mkdir(parents=True, exist_ok=True)
    results = []

    def check(name, condition, detail):
        results.append({"check": name, "passed": bool(condition), "detail": str(detail)})

    raw = read_timed_csv(RAW_PATH)
    clean = read_timed_csv(PROCESSED / "clean_wide.csv")
    exact = read_timed_csv(PROCESSED / "wide_unaligned.csv")
    window = read_timed_csv(PROCESSED / "wide_10s.csv")
    missing = read_timed_csv(PROCESSED / "missing_mask.csv")
    core = read_timed_csv(PROCESSED / "analysis_core_windows.csv")
    lineage = read_timed_csv(PROCESSED / "feature_lineage.csv")
    lineage["observed_at"] = pd.to_datetime(lineage["observed_at"], format="mixed")
    alarms = read_timed_csv(PROCESSED / "alarm_windows.csv")
    records = read_timed_csv(PROCESSED / "alarm_records.csv")
    features = [column for column in clean if column not in KEYS]

    source_hash = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    check("source_fingerprint", source_hash == SOURCE_SHA256, source_hash)
    check("source_snapshot", (len(raw), raw.deviceId.nunique(), raw.variable.nunique()) == (173853, 3, 52), "173853 rows / 3 devices / 52 variables")
    check("valid_keys", not raw[[*KEYS, "variable"]].isna().any().any(), "device, time and variable keys are non-null")
    check("no_duplicate_source_keys", not raw.duplicated([*KEYS, "variable"]).any(), "source measurement keys are unique")
    comparison = raw.loc[raw.timeMili.notna()]
    utc = pd.to_datetime(comparison.timeMili, unit="ms", utc=True).dt.tz_localize(None)
    delta = (comparison.timestamp - utc).dt.total_seconds()
    check("time_fields_observed_offset", len(delta) == 52202 and delta.eq(19800).all(), "all 52202 comparable records differ by 19800 clock seconds")
    check("clean_unique_keys", not clean.duplicated(KEYS).any(), f"{len(clean)} unique windows")
    check("window_boundaries", clean.timestamp.eq(clean.timestamp.dt.floor(WINDOW_SIZE)).all(), WINDOW_SIZE)
    check("expected_schema", len(features) == 53 and UNKNOWN_VARIABLE not in features and "alarmClass" not in features, "53 measurement features; unknown variable and labels excluded")
    check("finite_observed_values", np.isfinite(clean[features].to_numpy(dtype=float)[clean[features].notna().to_numpy()]).all(), "all observed numeric cells are finite")
    check("no_empty_measurement_windows", not clean[features].isna().all(axis=1).any(), "all-empty measurement windows omitted")

    # 用原始记录逐条建立独立索引，不依赖预处理的 pivot/resample 或来源表。
    raw = raw.assign(source_row_id=np.arange(1, len(raw) + 1), window_start=raw.timestamp.dt.floor(WINDOW_SIZE))
    expected = {}
    for row in raw.itertuples(index=False):
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
    actual_cells = clean.melt(id_vars=KEYS, value_vars=features, var_name="feature", value_name="value").dropna(subset=["value"])
    values_match = len(actual_cells) == len(expected)
    for row in actual_cells.itertuples(index=False):
        entry = expected.get((row.deviceId, row.timestamp, row.feature))
        if entry is None or not np.isclose(row.value, entry[2], rtol=1e-12, atol=1e-12):
            values_match = False
            break
    check("every_cell_matches_last_raw_observation", values_match, f"{len(actual_cells)} non-null cells checked against raw data")
    lineage_match = len(lineage) == len(expected) and not lineage.duplicated([*KEYS, "feature"]).any()
    for row in lineage.itertuples(index=False):
        entry = expected.get((row.deviceId, row.timestamp, row.feature))
        if entry is None or entry[:2] != (row.observed_at, row.source_row_id) or not np.isclose(entry[2], row.value_numeric, rtol=1e-12, atol=1e-12):
            lineage_match = False
            break
    check("cell_lineage_and_original_row_ids", lineage_match, "observation timestamps, raw row ids and values match independent raw index")
    check("no_cross_window_observations", ((lineage.observed_at >= lineage.timestamp) & (lineage.observed_at < lineage.timestamp + pd.Timedelta(WINDOW_SIZE))).all(), "all feature timestamps stay inside their own half-open window")
    check("final_equals_window_output", clean.equals(window), "final file contains no extra imputation or value deletion")
    check("missing_mask_exact", clean[KEYS].equals(missing[KEYS]) and np.array_equal(clean[features].isna().to_numpy(dtype=np.uint8), missing[features].to_numpy()), "1=missing and 0=observed for every measurement cell")
    expected_core = clean.loc[clean[CORE_FEATURES].notna().any(axis=1)]
    check("core_subset_preserves_values", core[clean.columns].reset_index(drop=True).equals(expected_core.reset_index(drop=True)), f"{len(core)} windows with at least one core feature")
    check("core_completeness_flags", core.core_feature_count.eq(core[CORE_FEATURES].notna().sum(axis=1)).all() and core.has_all_core_features.eq(core[CORE_FEATURES].notna().all(axis=1).astype(int)).all(), "subset metadata matches actual missingness")
    telemetry = read_timed_csv(PROCESSED / "telemetry_long.csv")
    check("full_long_table_preserved", telemetry.equals(raw[telemetry.columns]), "all raw columns, rows, values and alarm codes preserved after parsing time")
    check("alarm_record_traceability", len(records) == len(raw) and records.source_row_id.tolist() == list(range(1, len(raw) + 1)) and records[telemetry.columns].equals(telemetry), "all raw tagged records retained, including code 0")
    code_columns = [
        column for column in alarms
        if column.startswith("code_") and column.endswith("_count") and column.split("_")[1].isdigit()
    ]
    actual_code_totals = {int(column.split("_")[1]): int(alarms[column].sum()) for column in code_columns}
    expected_code_totals = {int(code): int(count) for code, count in raw.alarmClass.value_counts().items()}
    check("alarm_counts_conserved", actual_code_totals == expected_code_totals and int(alarms.record_count.sum()) == len(raw), actual_code_totals)
    independent_counts = raw.groupby(["deviceId", "window_start", "alarmClass"]).size()
    independent_ok = True
    indexed_alarms = alarms.set_index(KEYS)
    for (device, start, code), count in independent_counts.items():
        if indexed_alarms.loc[(device, start), f"code_{int(code)}_count"] != count:
            independent_ok = False
            break
    check("alarm_per_window_counts", independent_ok, "every device-window-code count matches original tagged records")
    key_match = pd.MultiIndex.from_frame(clean[KEYS]).equals(pd.MultiIndex.from_frame(alarms[KEYS]))
    check("alarm_join_is_one_to_one", key_match and not alarms.duplicated(KEYS).any(), "every clean window has exactly one alarm summary row")
    multi = alarms.code_count.gt(1)
    check("ambiguous_labels_not_collapsed", alarms.loc[multi, "single_alarmClass"].isna().all() and alarms.multi_code.eq(multi.astype(int)).all(), f"{int(multi.sum())} multi-code windows retain ambiguity")
    single_ok = alarms.loc[~multi, "single_alarmClass"].astype(int).astype(str).eq(alarms.loc[~multi, "code_values"].astype(str)).all()
    check("single_code_labels_consistent", single_ok, "single codes agree with retained code sets; code 0 has no assumed business meaning")
    check("metadata_counts_consistent", alarms.code_count.eq(alarms[code_columns].gt(0).sum(axis=1)).all() and alarms.record_count.eq(alarms[code_columns].sum(axis=1)).all(), "window count metadata reconciles")
    check("required_quality_reports_present", all((QUALITY / name).is_file() for name in ["device_summary.csv", "missingness.csv", "sampling_gaps.csv", "numeric_statistics.csv", "window_comparison.csv", "data_dictionary.csv", "alarm_class_summary.csv", "summary.json"]), "all eight quality artifacts present")

    passed = all(result["passed"] for result in results)
    report = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "passed": passed, "check_count": len(results), "checks": results,
        "source_sha256": source_hash,
        "clean_sha256": hashlib.sha256((PROCESSED / "clean_wide.csv").read_bytes()).hexdigest(),
        "semantic_confirmation": "official coordinate order, units and alarm meanings were not supplied; not asserted by these checks",
    }
    (QUALITY / "validation_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        failed = [result["check"] for result in results if not result["passed"]]
        raise SystemExit(f"验收未通过：{', '.join(failed)}。详见 data/quality/validation_results.json。")
    print(f"程序验收通过：{len(results)} 项；已核对全部非空单元格、原始行号和告警关联。")


if __name__ == "__main__":
    main()
