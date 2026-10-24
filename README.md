# vehicle-iot-analysis

车辆 IoT 课程项目。当前已交付 **A 部分数据预处理**与 **B 部分运行特征分析**。代码已从9个Python文件合并为3个，交付包从37个文件合并为13个。

## 看什么

| 要做的事 | 文件 |
| --- | --- |
| 了解数据、清洗规则、字段含义和检查结果 | `docs/preprocessing_report.html`，各类统计和字段字典都在同一报告中 |
| 看 B 部分的运行特征分析结果 | `docs/B_运行特征分析.md`（报告章节），图表在 `figures/` |
| 复现或修改 B 部分的分析 | `notebooks/B_running_characteristics.ipynb` |
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

三个脚本分别负责处理、检查与报告、打包。每个脚本正常输出一行摘要。打包位置为 `dist/vehicle_iot_A_delivery.zip`。`data/` 随仓库提交，包含原始输入、两份处理数据、统一核查表和质量数据；克隆仓库后即可运行。ZIP同时作为离线交付。

B 部分单独运行（不依赖 A 的脚本，只读取已提交的 `data/processed/clean_wide.csv`）：

```powershell
jupyter lab notebooks/B_running_characteristics.ipynb
```

或直接执行并重新生成全部图表：

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/B_running_characteristics.ipynb
```

Notebook 会自动定位仓库根目录，从仓库根目录或 `notebooks/` 下启动都可以；图表统一写入 `figures/`。整份分析 10 秒内跑完。

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

## B 部分：运行特征分析

报告章节见 `docs/B_运行特征分析.md`，分析与图表由 `notebooks/B_running_characteristics.ipynb` 生成，共 15 张图（`figures/B01`–`B15`）。

内容覆盖分工要求的三项：车速/转速/加速度时序、三台车辆的运行特征对比（工况、怠速与行驶比例、加减速习惯）、变量相关性热力图，外加车辆轨迹/位置可视化。

分析中有四个口径选择，直接决定了结论是否成立，改动前请先看 Notebook 第 0 节：

1. **观测稀疏，不插值。** 53 个测量字段里车速只有 6.4% 的窗口有值，时序图一律在缺口处断线，统计量只基于有观测的窗口。
2. **纵向加速度用 dv/dt，不用原始三轴。** `ACCELERATION X/Y/Z` 的合幅值稳定在 9.4–9.9 m/s²（含重力），且与 dv/dt 几乎不相关，不能当作纵向加速度。
3. **`position_3` 的海拔不可信。** `(0,0,0)` 是「无定位」占位值（占 18.5%），画轨迹前已剔除；海拔存在 −1,677 m 这类值。
4. **相关矩阵做了共同观测数过滤。** 样本数不足 50 的字段对直接剔除，不参与解读，避免稀疏观测下的伪相关。

坐标口径由统计判定：用相邻定位点反推车速与 OBD 车速比对，`position_1`=纬度、`position_2`=经度（中位绝对误差 4.0 km/h，反向假设为 9.6 km/h）。这仍是推断，A 部分「待数据提供方确认」的结论不变。

`IDLING` 标志与「车速为 0」不等价（`IDLING=1` 的窗口约八成车速大于 0），B 部分以车速+转速定义怠速，`IDLING` 仅作交叉参考。

## 统一核查与质量报告

`review_records.csv` 共3,940条不同原始记录。原因用 `|` 连接：35条未知变量记录中有29条同时数值转换失败，另有3,905条IQR候选极值。候选不等于错误，原值均保留在长表。

`quality_report.json` 合并保存原先分散的设备概况、缺失率、采样间隔、窗口对比、数值范围、编码分布、字段字典和程序验收。HTML报告直接展示这些内容。中间宽表、重复子集、单独缺失标记和逐单元格来源文件不再交付；检查程序仍从原始记录逐个核对全部176,299个非空测量单元格。

## 待数据提供方确认

POSITION三项顺序与含义、变量单位和状态编码、告警含义与标注规则、原始timestamp时区仍未提供。位置保留position_1/2/3；中文字段解释依据原名，未进行未经确认的单位转换。程序验收验证数据一致性，不替代业务语义确认。
