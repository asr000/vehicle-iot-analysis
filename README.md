# vehicle-iot-analysis

车辆 IoT 数据清洗、运行特征分析与告警建模课程项目。本仓库当前交付 A 部分的数据工程流程，B、C 可使用生成的公共数据开展后续分析。

## 快速运行

已在 Windows 的 `course` 环境验证：Python 3.11.16、pandas 3.0.5、NumPy 2.4.6。推荐使用 Python 3.11。

把用户提供的原始文件放到 `data/raw/题2_车辆物联网遥测.csv`。本次输入快照为 173,853 行、3 个设备、52 种变量，SHA256 为 `e0f5571f74c6922ab7b5e65a34c52b2939b79bb33768eef30b124d3467333779`。输入来源未附公开下载地址。

```powershell
python -m pip install -r requirements.txt
python .\run_pipeline.py
python .\build_delivery.py
```

流水线只有五行阶段摘要，验收失败时明确报错。打包程序生成 `dist/vehicle_iot_A_delivery.zip`，内含原始输入、代码、数据、报告和文件清单；验收未通过时拒绝打包。原始及逐条生成数据由 `.gitignore` 排除，不提交到 GitHub；组员可从离线交付包取得数据。

## 脚本分工

| 文件 | 职责 |
| --- | --- |
| `inspect_data.py` | 只读检查原始数据，打印 8 行摘要，不自动执行其他阶段 |
| `prepare_data.py` | 解析时间、拆分位置、隔离未知变量和转换失败记录，展开精确时间宽表 |
| `align_data.py` | 每辆车独立进行 10 秒窗口对齐，每列取窗口内最后一个非空值 |
| `finalize_data.py` | 输出正式宽表、缺失标记、核心子集、告警关联和单元格来源 |
| `profile_data.py` | 生成缺失、采样、窗口比较、数值核查、字段字典及 HTML 报告 |
| `validate_data.py` | 从原始行独立核对所有非空单元格、来源、缺失和告警关联，输出 JSON 验收结果 |
| `run_pipeline.py` | 按顺序执行处理与验收，可重复运行 |
| `build_delivery.py` | 验收通过后打包并生成逐文件大小、行数、SHA256 和人工验收栏 |
| `pipeline_config.py` | 共用路径、窗口、核心变量及原始数据指纹；导入不会执行处理 |

每个阶段也可单独运行。原始数据不变且只需重新对齐时，从 `align_data.py` 开始继续执行后续阶段即可；正式打包前建议完整运行流水线。

## B、C 的主要数据入口

| 文件 | 作用及约定 |
| --- | --- |
| `data/processed/clean_wide.csv` | 主表，35,306 行、55 列；2 个键、50 个数值变量和3个位置分量；保留缺失 |
| `data/processed/missing_mask.csv` | 与主表同行同列的测量缺失标记；1=缺失、0=有观测；两个键保留原值 |
| `data/processed/analysis_core_windows.csv` | 至少有一项车速、转速或三轴加速度的3,136个窗口；`core_feature_count` 为五项中有效数量，`has_all_core_features` 为是否五项齐全，其中2,084行齐全 |
| `data/processed/alarm_records.csv` | 全部173,853条原始标签记录，包含编码0；`source_row_id` 从1开始，原CSV物理行号为该值+1；`timestamp` 是原始观测时刻，`window_start` 是关联窗口起点 |
| `data/processed/alarm_windows.csv` | 与主表按 `deviceId + timestamp` 一对一关联；`timestamp` 是窗口起点 |
| `data/processed/feature_lineage.csv` | 176,299个非空单元格的原始观测时间、值、字段名与原始数据行号 |

`clean_wide.csv` 中一行表示 `[timestamp, timestamp + 10秒)` 的窗口内汇总，不表示全部变量在起点同时采集。各列独立取窗口内最后一个有效观测，窗口内未观测的字段保持为空。未对缺失值补零、插值或跨窗口沿用。

### 告警表字段

- `code_0_count` 至 `code_5_count`：该窗口各编码的原始记录数，**不等于独立告警事件次数**。
- `record_count`：窗口原始记录总数；`code_count`：不同编码数量。
- `code_values`：用 `|` 连接的全部编码，包含0；`multi_code=1` 表示多编码。
- `single_alarmClass`：仅窗口内只有一种编码时填写，否则为空。它表示编码唯一性，未证明业务标签有效性。
- `window_end`：窗口右边界，不包含该时刻。

编码0–5的业务含义及标签产生机制未提供，**不把0自动解释为正常**。12,134个同设备同时间的记录组带有不同编码，8,799个10秒窗口包含多个编码。没有用最大值、最后值或多数票强制生成唯一故障类别。

C 应先明确独立事件或分类样本的定义。预测告警发生前状态时使用事件之前的原始观测；整窗汇总可能包含较晚观测，不能直接用于窗口起点或较早事件的预测。告警计数、编码集合、单一编码等目标相关列不作为输入特征。

## 中间结果与待核查记录

- `telemetry_long.csv`：保留全部原始列、原始值和告警编码，timestamp 已解析。
- `wide_unaligned.csv`：100,838行、55列的精确时间宽表。
- `wide_10s.csv`：窗口对齐中间结果，与本次正式主表数值一致。
- `data/review/invalid_numeric_records.csv`：29条数值转换失败记录。
- `data/review/unknown_variable_records.csv`：全部35条未知变量记录，其中包含上述29条；两份文件不能相加当成删除数量。
- `data/review/numeric_review_candidates.csv`：分设备分变量的1.5 IQR候选极值，仅供核查，原值均保留；不对状态编码、零IQR或少于20个样本的组套用该规则。

## 质量报告与字段说明

打开 `docs/preprocessing_report.html` 查看可浏览、可打印的预处理报告。

`data/quality/` 提供：

| 文件 | 内容 |
| --- | --- |
| `data_dictionary.csv` | 主表55列的来源、类型、中文解释、单位状态、聚合与缺失规则 |
| `device_summary.csv` | 三个设备的别名、原编号、数据量、时间跨度与核心子集数量 |
| `missingness.csv` | 精确时间和10秒窗口的全局及分设备逐字段缺失率 |
| `sampling_gaps.csv` | 每台设备每个原始变量的采样间隔中位数、90分位、最大值和长间隔数量 |
| `window_comparison.csv` | 5/10/30/60秒窗口的覆盖与核心完整窗口数比较 |
| `numeric_statistics.csv` | 分设备及全局的范围、分位数、负值、常量和候选极值统计 |
| `alarm_class_summary.csv` | 六种原始编码的记录数与占比 |
| `summary.json` | 本次处理规模、输入指纹和运行环境 |
| `validation_results.json` | 实际程序验收项、结果及正式主表指纹 |

缺失率的分母是存在测量记录的设备窗口；完全没有遥测记录的日历空档未计入。10秒是本次工程基线，不声称是经下游建模验证的最优窗口。

## 尚待提供方确认的语义

1. POSITION三个分量的顺序和含义。暂用 `position_1/2/3`，尚未正式命名经度、纬度、海拔。
2. 原始各变量的单位、状态编码含义与OBD哨兵值。字典中的中文解释依据英文变量名，未做单位换算，也未按未经确认的阈值裁剪。
3. 告警0–5的业务解释与标注规则。
4. 原始timestamp的时区。52,202条可比较记录与timeMili转换所得UTC时间的钟面差均为5小时30分钟，但不据此声明地理时区。

这些语义待确认项不通过猜测补齐；本次交付保留原始信息，程序验收检查数据一致性与来源，不宣称已验证业务标签或物理单位。
