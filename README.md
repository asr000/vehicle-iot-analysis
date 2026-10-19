# vehicle-iot-analysis

车辆 IoT 课程项目，当前交付 A 部分的数据预处理。代码已从9个Python文件合并为3个，交付包从37个文件合并为13个。

## 看什么

| 要做的事 | 文件 |
| --- | --- |
| 了解数据、清洗规则、字段含义和检查结果 | `docs/preprocessing_report.html`，各类统计和字段字典都在同一报告中 |
| B 做运行特征分析 | `data/processed/clean_wide.csv` |
| C 做告警分析 | 主表的告警汇总列，以及 `data/processed/telemetry_long.csv` 的逐条记录 |
| 核查异常记录 | `data/review/review_records.csv`，一行原始记录可带多个原因 |
| 程序读取全部质量统计与校验结果 | `data/quality/quality_report.json` |
| 逐文件验收 | ZIP内的 `验收清单.csv` |

## 怎么运行

实际验证环境：Python3.11.16、pandas3.0.5、NumPy2.4.6。将原始文件放在 `data/raw/题2_车辆物联网遥测.csv`。

```powershell
python -m pip install -r requirements.txt
python .\prepare_data.py
python .\inspect_data.py
python .\build_delivery.py
```

三个脚本分别负责处理、检查与报告、打包。每个脚本正常输出一行摘要。打包位置为 `dist/vehicle_iot_A_delivery.zip`。原始和生成数据不提交Git，随离线包提供。

## 两份主要数据

**主表 clean_wide.csv：35,306行、69列。**

- 设备和窗口起点两列键。
- 53个测量字段，与合并前逐值一致。
- 12个告警汇总字段，原alarm_windows表已按相同键合入。
- `core_feature_count` 和 `has_all_core_features` 两个完整度字段，原核心子集不再重复保存。前者大于0的有3,136行，后者等于1的有2,084行。

每行对应左闭右开的10秒窗口，每个测量字段取窗口内最后一个非空观测。空值表示没有观测，不补零、不插值、不跨窗口沿用。缺失标记可以直接用测量字段的 `isna()` 得到。

**长表 telemetry_long.csv：173,853行、8列。**

保留原始6列，加上 `source_row_id` 和 `window_start`。原alarm_records表已合入，不再重复保存。原CSV物理行号等于source_row_id+1。

告警编码0–5业务含义未提供，不自动将0解释为正常。各编码计数是原始标签记录数，不是独立事件次数。8,799个窗口有多个编码，`single_alarmClass` 在这些窗口为空，完整集合保存在 `code_values`。

## 分析时选择正确的列

主表已合并告警汇总，训练模型时不能将告警相关列混入输入特征。以下函数只返回53个测量字段：

```python
import pandas as pd
from prepare_data import sensor_columns

df = pd.read_csv("data/processed/clean_wide.csv")
measurements = df[sensor_columns(df)]
core_windows = df.loc[df["core_feature_count"] > 0]
complete_core_windows = df.loc[df["has_all_core_features"] == 1]
missing_mask = measurements.isna()
```

预测告警前状态时需使用事件之前的原始观测；整窗汇总可能包含较晚观测。C仍需明确独立事件与分类样本的定义。

## 统一核查与质量报告

`review_records.csv` 共3,940条不同原始记录。原因用 `|` 连接：35条未知变量记录中有29条同时数值转换失败，另有3,905条IQR候选极值。候选不等于错误，原值均保留在长表。

`quality_report.json` 合并保存原先分散的设备概况、缺失率、采样间隔、窗口对比、数值范围、编码分布、字段字典和程序验收。HTML报告直接展示这些内容。中间宽表、重复子集、单独缺失标记和逐单元格来源文件不再交付；检查程序仍从原始记录逐个核对全部176,299个非空测量单元格。

## 待数据提供方确认

POSITION三项顺序与含义、变量单位和状态编码、告警含义与标注规则、原始timestamp时区仍未提供。位置保留position_1/2/3；中文字段解释依据原名，未进行未经确认的单位转换。程序验收验证数据一致性，不替代业务语义确认。
