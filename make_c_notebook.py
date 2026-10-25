"""生成 C 部分基线 notebook（C_alarm_analysis.ipynb）。"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []
cells.append(md("""# C 部分：告警分析与建模（基线版）

> 任务定义（本 notebook 的口径声明）：
> - **多标签任务**：每个 10 秒窗口对 1–5 号告警各是一个"是否出现"的二分类，标签 = `code_k_count > 0`。
>   这样天然兼容 8,799 个多编码窗口，不需要人为归并标签。
> - **样本单位**：窗口级（事件级口径留作后续对比）。
> - **特征**：只用 53 个测量列（`prepare_data.sensor_columns()`），告警汇总列仅作标签，绝不进特征。
> - **基线模型**：多数类基线 + 逻辑回归（含类别权重对比）；梯度提升树与不平衡处理在基线诊断后决定。"""))

cells.append(code("""from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "clean_wide.csv").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_data import sensor_columns

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "clean_wide.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
devices = sorted(df["deviceId"].unique())

SENSORS = sensor_columns(df)  # 必须在添加派生列之前取，避免非测量列混入
df["车辆"] = df["deviceId"].map({d: f"车辆{i+1}" for i, d in enumerate(devices)})
CODES = [1, 2, 3, 4, 5]
Y = pd.DataFrame({k: df[f"code_{k}_count"].fillna(0).gt(0).astype(int) for k in CODES})

print(f"宽表 {df.shape[0]} 行 × {df.shape[1]} 列；测量列 {len(SENSORS)} 个")
print(f"窗口按车辆：{df['车辆'].value_counts().to_dict()}")"""))

cells.append(md("""## 1 告警摸底：频次、按车分布、共现结构

先看标签本身长什么样，再决定怎么建模。"""))

cells.append(code("""# 1.1 各类告警的窗口频次与占比
freq = pd.DataFrame({
    "正样本窗口": Y.sum(),
    "占比": Y.mean().round(3),
})
per_veh = df.groupby("车辆").apply(
    lambda g: pd.Series({f"code{k}": round(Y.loc[g.index, k].mean(), 3) for k in CODES}),
    include_groups=False,
)
print("各类告警窗口频次：")
print(freq.T)
print("\\n各类告警在每台车的窗口占比：")
print(per_veh)"""))

cells.append(code("""# 1.2 共现矩阵：哪几类告警总爱一起出现（“重叠影响”的结构）
cooc = Y.T.dot(Y)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
ax.set_title("图 C-1　告警共现矩阵（同窗口出现次数）")
fig.tight_layout()
fig.savefig(FIG / "C01_cooccurrence.png", dpi=150)
plt.show()

off_diag = cooc.where(~np.eye(len(CODES), dtype=bool))
print("1–4 号互相共现率（共现数/该类总数）：")
for k in CODES[:4]:
    others = [c for c in CODES[:4] if c != k]
    rate = cooc.loc[k, others].min() / Y[k].sum()
    print(f"  code{k}: 至少与 1–4 中另一类同现的窗口占 {rate:.1%}")
print(f"code5: 与 1–4 号同现的窗口占 {(cooc.loc[5, [1,2,3,4]].sum() / Y[5].sum()):.1%}")"""))

cells.append(code("""# 1.3 时段分布：告警在一天 24 小时的分布
df["_hour"] = df["timestamp"].dt.hour
df["_any_alarm"] = Y.any(axis=1).astype(int)
hourly = df.groupby("_hour")["_any_alarm"].agg(count="count", mean="mean")
hourly.columns = ["窗口数", "告警占比"]

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(hourly.index, hourly["告警占比"], color="steelblue")
ax.set_xlabel("小时"); ax.set_ylabel("有告警窗口占比")
ax.set_title("图 C-2　各小时的告警占比")
fig.tight_layout()
fig.savefig(FIG / "C02_hourly_alarm.png", dpi=150)
plt.show()
df.drop(columns=["_any_alarm", "_hour"], inplace=True)
print("全天告警占比波动范围：", round(hourly["告警占比"].min(), 3), "~", round(hourly["告警占比"].max(), 3))"""))

cells.append(md("""## 2 特征集：按覆盖率分级

测量列的覆盖率差异极大（位置 ≈100%，车速/转速 ≈6%，过半列 <1%）。
基线取**覆盖率 ≥ 1%** 的列作为特征集，既保留语义最重要的车速/转速，又不被近零覆盖的列拖累。

⚠️ 注意一个泄漏风险：`position_1/2`（经纬度）几乎等同于"车辆身份证"——车辆2 在 600 km 外，
模型可能靠经纬度认出"这是车辆1"而不是学到真正的状态规律。第 5 节做消融对比。"""))

cells.append(code("""coverage = df[SENSORS].notna().mean().sort_values(ascending=False)
tier = coverage[coverage >= 0.01]
print(f"覆盖率 ≥1% 的测量列共 {len(tier)} 个：")
for name, c in tier.items():
    print(f"  {c:.1%}  {name}")
zero_cols = coverage[coverage == 0].index.tolist()
print(f"\\n覆盖率为 0 的列 {len(zero_cols)} 个，直接排除（如 {zero_cols[:3]}）")

FEATURES = tier.index.tolist()
X = df[FEATURES]"""))

cells.append(md("""## 3 数据切分：按时间切（主）与随机切（对照）

- **按时间切**：前 80% 时间训练、后 20% 测试，模拟"用历史预测未来"，不存在相邻窗口同时出现在两侧的泄漏；
- **随机切**：作为对照，展示泄漏会虚高多少指标。"""))

cells.append(code("""from sklearn.model_selection import train_test_split

cut = df["timestamp"].quantile(0.8)
time_train = df["timestamp"] < cut
tr_t = df.index[time_train]
te_t = df.index[~time_train]
tr_r, te_r = train_test_split(df.index, test_size=0.2, random_state=42)

print(f"时间切分：训练 {len(tr_t)} / 测试 {len(te_t)}（切点 {cut}）")
for k in CODES:
    a = Y.loc[te_t, k].mean()
    b = Y.loc[te_r, k].mean()
    print(f"  code{k} 测试集正样本占比：时间切 {a:.1%} / 随机切 {b:.1%}")"""))

cells.append(md("""## 4 基线模型

**基线 1：多数类基线**——对每类都预测"无告警"。它是下限标尺：
accuracy 会很高（≈85–93%），但召回率为 0，说明 accuracy 在不平衡数据上没有意义。
**基线 2：逻辑回归**——中位数填补缺失 + 标准化；对比 `class_weight` 开/关。"""))

cells.append(code("""from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate(y_true, y_pred, name):
    rows = {}
    for k in CODES:
        p, r, f1, _ = precision_recall_fscore_support(y_true[k], y_pred[k], zero_division=0)
        rows[f"code{k}"] = (round(p[1], 3), round(r[1], 3), round(f1[1], 3))
    code_keys = [f"code{k}" for k in CODES]
    rows["宏平均"] = tuple(
        round(float(np.mean([rows[ck][i] for ck in code_keys])), 3) for i in range(3)
    )
    out = pd.DataFrame(rows, index=["精确率", "召回率", "F1"]).T
    out.insert(0, "accuracy", round(np.mean([accuracy_score(y_true[k], y_pred[k]) for k in CODES]), 3))
    out.columns = pd.MultiIndex.from_product([[name], out.columns])
    return out

def fit_predict_lr(X_tr, y_tr, X_te, balanced=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None),
    )
    preds = {}
    for k in CODES:
        pipe.fit(X_tr, y_tr[k])
        preds[k] = pipe.predict(X_te)
    return preds

# 多数类基线（时间切分）
majority = {k: np.zeros(len(te_t), dtype=int) for k in CODES}
res_majority = evaluate(Y.loc[te_t], majority, "多数类基线")

# 逻辑回归（时间切分）
pred_lr = fit_predict_lr(X.loc[tr_t], Y.loc[tr_t], X.loc[te_t], balanced=False)
pred_lrb = fit_predict_lr(X.loc[tr_t], Y.loc[tr_t], X.loc[te_t], balanced=True)

res = pd.concat([res_majority, evaluate(Y.loc[te_t], pred_lr, "LR默认"),
                 evaluate(Y.loc[te_t], pred_lrb, "LRbalanced")], axis=1)
pd.set_option("display.width", 160)
print(res)"""))

cells.append(code("""# 4.1 可视化：三类基线的宏平均 F1
macro_f1 = {"多数类": 0.0,
            "LR默认": float(res[("LR默认", "F1")]["宏平均"]),
            "LRbalanced": float(res[("LRbalanced", "F1")]["宏平均"])}
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(macro_f1.keys(), macro_f1.values(), color=["gray", "steelblue", "darkorange"])
ax.set_ylabel("宏平均 F1"); ax.set_ylim(0, max(macro_f1.values()) * 1.4 + 1e-9)
for i, v in enumerate(macro_f1.values()):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
ax.set_title("图 C-3　基线宏平均 F1 对比（时间切分）")
fig.tight_layout()
fig.savefig(FIG / "C03_baseline_f1.png", dpi=150)
plt.show()"""))

cells.append(md("""## 5 消融：经纬度是不是在"作弊"？

位置列覆盖率近 100%，但可能让模型靠"认出车"拿分。剔除 position_1/2/3 再训一遍 LRbalanced，
对比指标变化——如果掉很多，说明当前指标含"车辆身份"的水分。"""))

cells.append(code("""pos = [c for c in FEATURES if c.startswith("position_")]
keep = [c for c in FEATURES if c not in pos]
pred_wo_pos = fit_predict_lr(X.loc[tr_t, keep], Y.loc[tr_t], X.loc[te_t, keep], balanced=True)
res_ab = pd.concat([res, evaluate(Y.loc[te_t], pred_wo_pos, "LR去位置")], axis=1)
pd.set_option("display.width", 160)
print(res_ab)"""))

cells.append(code("""# 5.1 多随机种子重复：确认“去位置”的收益不是单次切分的运气
from sklearn.model_selection import train_test_split

def macro_f1_of(preds, y_true):
    f1s = []
    for k in CODES:
        _, _, f1, _ = precision_recall_fscore_support(y_true[k], preds[k], zero_division=0)
        f1s.append(f1[1])
    return float(np.mean(f1s))

seeds = [0, 1, 2, 3, 4]
records = []
for seed in seeds:
    tr, te = train_test_split(df.index, test_size=0.2, random_state=seed)
    p_with = fit_predict_lr(X.loc[tr], Y.loc[tr], X.loc[te], balanced=True)
    p_without = fit_predict_lr(X.loc[tr, keep], Y.loc[tr], X.loc[te, keep], balanced=True)
    records.append({
        "seed": seed,
        "含经纬度": round(macro_f1_of(p_with, Y.loc[te]), 3),
        "去经纬度": round(macro_f1_of(p_without, Y.loc[te]), 3),
    })
seed_res = pd.DataFrame(records)
seed_res["差值"] = seed_res["去经纬度"] - seed_res["含经纬度"]
print(seed_res)
print()
print(seed_res[["含经纬度", "去经纬度"]].agg(["mean", "std"]).round(3))
print(f"差值：均值 {seed_res['差值'].mean():+.3f}，5 次全部为正：{(seed_res['差值'] > 0).all()}")"""))

cells.append(md("""## 7 梯度提升树（XGBoost）与不平衡处理对照

XGBoost 原生支持缺失值（无需填补），适合稀疏特征。与 LR（去位置）在时间切分上对比；
不平衡处理对照三组：默认 / `scale_pos_weight` / 在训练段内部调阈值（按验证段 F1 最大化选阈值）。"""))

cells.append(code("""# 7.1 XGBoost（去位置特征，时间切分，原生处理缺失）
import xgboost as xgb
from sklearn.metrics import f1_score

xgb_params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  random_state=42, n_jobs=4)

def fit_predict_xgb(X_tr, y_tr, X_te, scale_pos_weight=None):
    preds = {}
    for k in CODES:
        w = None
        if scale_pos_weight is not None:
            neg, pos = (y_tr[k] == 0).sum(), (y_tr[k] == 1).sum()
            w = scale_pos_weight if scale_pos_weight != "auto" else neg / pos
        clf = xgb.XGBClassifier(**xgb_params, scale_pos_weight=w)
        clf.fit(X_tr, y_tr[k])
        preds[k] = clf.predict(X_te)
    return preds

# 从训练段再切后 20% 作验证段（用于调阈值，测试段始终不碰）
val_cut = df.loc[tr_t, "timestamp"].quantile(0.8)
va = tr_t[df.loc[tr_t, "timestamp"] >= val_cut]
sub_tr = tr_t.difference(va)

def fit_predict_xgb_threshold(X_tr, y_tr, X_va, y_va, X_te):
    preds = {}
    for k in CODES:
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr[k])
        proba_va = clf.predict_proba(X_va)[:, 1]
        ths = np.linspace(0.05, 0.95, 37)
        f1s = [f1_score(y_va[k], (proba_va >= t).astype(int), zero_division=0) for t in ths]
        th = ths[int(np.argmax(f1s))]
        preds[k] = (clf.predict_proba(X_te)[:, 1] >= th).astype(int)
    return preds

Xtr, Xte = X.loc[sub_tr, keep], X.loc[te_t, keep]
Ytr, Yva = Y.loc[sub_tr], Y.loc[va]
pred_xgb = fit_predict_xgb(Xtr, Ytr, Xte)
pred_xgb_w = fit_predict_xgb(Xtr, Ytr, Xte, scale_pos_weight="auto")
pred_xgb_th = fit_predict_xgb_threshold(Xtr, Ytr, X.loc[va, keep], Yva, Xte)

res2 = pd.concat([
    evaluate(Y.loc[te_t], pred_lr, "LR默认"),
    evaluate(Y.loc[te_t], pred_lrb, "LRbalanced"),
    evaluate(Y.loc[te_t], pred_xgb, "XGB默认"),
    evaluate(Y.loc[te_t], pred_xgb_w, "XGB权重"),
    evaluate(Y.loc[te_t], pred_xgb_th, "XGB调阈值"),
], axis=1)
print(res2.xs("F1", axis=1, level=1))
print()
print("宏平均 F1：", {m: float(res2[(m, "F1")]["宏平均"]) for m in ["LR默认", "LRbalanced", "XGB默认", "XGB权重", "XGB调阈值"]})"""))

cells.append(md("""## 8 事件级口径的敏感性分析

窗口级把"一次持续 3 分钟的告警"拆成 18 个正样本，且相邻样本高度相关。
本节把同一车辆内、某类告警连续出现的窗口合并为**事件**（首窗即代表样本），重训模型对比。"""))

cells.append(code("""# 8.1 事件合并：按设备、按类，连续窗口（间隔一个 10s 窗口）合并
df_sorted = df.sort_values(["deviceId", "timestamp"]).reset_index(drop=True)
Y_sorted = Y.loc[df_sorted.index]
gap_ok = df_sorted.groupby("deviceId")["timestamp"].diff().dt.total_seconds().fillna(10**9).le(12)

def event_samples(code_k):
    col = Y_sorted[code_k].to_numpy()
    prev = np.roll(col, 1); prev[0] = 0
    is_start = (col == 1) & ((prev == 0) | (~gap_ok.to_numpy()))
    return np.flatnonzero(is_start)  # 行位置（positional）

event_pos = {k: event_samples(k) for k in CODES}
for k in CODES:
    print(f"code{k}: 窗口级正样本 {Y[k].sum()} → 事件数 {len(event_pos[k])}")"""))

cells.append(code("""# 8.2 事件级重训：每类用“事件首窗”为正样本、全部无该类窗口为负样本
def event_level_f1(feat_cols):
    f1s = {}
    ts = df_sorted["timestamp"].to_numpy()
    tr_mask_all = (df_sorted["timestamp"] < cut).to_numpy()
    for k in CODES:
        pos = event_pos[k]
        neg = np.flatnonzero(Y_sorted[k].to_numpy() == 0)
        tr = np.concatenate([pos[tr_mask_all[pos]], neg[tr_mask_all[neg]]])
        te = np.concatenate([pos[~tr_mask_all[pos]], neg[~tr_mask_all[neg]]])
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(df_sorted.iloc[tr][feat_cols], Y_sorted.iloc[tr][k])
        p = clf.predict(df_sorted.iloc[te][feat_cols])
        f1s[f"code{k}"] = round(f1_score(Y_sorted.iloc[te][k], p, zero_division=0), 3)
    f1s["宏平均"] = round(float(np.mean(list(f1s.values()))), 3)
    return f1s

print("XGB 事件级 F1：", event_level_f1(keep))"""))

cells.append(md("""## 9 告警前后状态对比（成因判断）

两组对比：
- **连续特征**（转速、车速、水温、电压等）：比较"有该类告警的窗口"与"无告警窗口"的中位数差异，
  在同一台车内部比较（Cliff's delta 效应量），避免车辆差异混入；
- **二值状态标志**（TOWING、IGNITION_STATUS 等）：覆盖率 18% 上下，样本充足，
  比较两组中该标志=1 的出现率差异。

注意：差异是相关而非因果；连续特征受传感器稀疏限制，只有外接电池电压有足够样本。"""))

cells.append(code("""# 9.1 同车内部对比：中位数差与 Cliff's delta（效应量）
COMPARE = ["ENGINE RPM", "Vehicle speed", "ENGINE COOLANT TEMPERATURE",
           "CONTROL MODULE VOLTAGE", "EXTERNAL BATTERY", "CALCULATED ENGINE LOAD"]

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))

alarm_free = df.index[(Y[[1,2,3,4,5]].sum(axis=1) == 0)]
MIN_SAMPLES = 10  # 传感器稀疏：每台车内可用样本有限，放宽到小样本下限并如实标注
rows = []
for k in CODES:
    pos_idx = df.index[Y[k] == 1]
    for f in COMPARE:
        deltas = []
        for dev in devices:
            p = df.loc[pos_idx.intersection(df.index[df["deviceId"] == dev]), f].dropna()
            n = df.loc[alarm_free.intersection(df.index[df["deviceId"] == dev]), f].dropna()
            if len(p) >= MIN_SAMPLES and len(n) >= MIN_SAMPLES:
                deltas.append(cliffs_delta(p, n))
        if deltas:
            rows.append({"告警类": f"code{k}", "特征": f,
                         "效应量(Cliff's delta)": round(float(np.nanmean(deltas)), 3),
                         "样本车辆数": len(deltas)})
eff = pd.DataFrame(rows)
pivot = eff.pivot(index="特征", columns="告警类", values="效应量(Cliff's delta)")
pd.set_option("display.width", 160)
print(pivot)
print(f"\\n判定说明：每格是该车内在 ≥{MIN_SAMPLES} 个样本下可算的车辆均值；")
print("空白 = 没有任何一辆车在该特征上有足够样本，本身反映了传感器稀疏的局限。")
print("|效应量| > 0.147 视为有实际意义的小效应（经验阈值）")"""))

cells.append(code("""# 9.2 二值状态标志：告警组与正常组的“出现率”差异（覆盖率高，样本足）
FLAGS = ["TOWING", "IGNITION_STATUS", "IDLING", "HARDWARE STATUS", "FUEL SYSTEM STATUS"]
MIN_N = 50
rate_rows = []
for k in CODES:
    pos_idx = df.index[Y[k] == 1]
    for f in FLAGS:
        rates = []
        for dev in devices:
            d_idx = df.index[df["deviceId"] == dev]
            p = df.loc[pos_idx.intersection(d_idx), f]
            n = df.loc[alarm_free.intersection(d_idx), f]
            p, n = p.dropna(), n.dropna()
            if len(p) >= MIN_N and len(n) >= MIN_N:
                rates.append((float((p == 1).mean()), float((n == 1).mean())))
        if rates:
            dp = round(float(np.mean([a - b for a, b in rates])), 3)
            rate_rows.append({"告警类": f"code{k}", "标志": f,
                              "告警组出现率-正常组": dp, "样本车辆数": len(rates)})
rate_diff = pd.DataFrame(rate_rows).pivot(index="标志", columns="告警类", values="告警组出现率-正常组")
print(rate_diff)
print(f"\\n每格：同车内 ≥{MIN_N} 窗口下，告警组该标志=1 的比例 减 正常组比例；")
print("正值 = 该标志在告警时更容易出现。")"""))

cells.append(code("""# 9.3 可视化：效应量与标志率差异合并展示
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5, ax=axes[0])
axes[0].set_title("左：连续特征的效应量（Cliff's delta）")
sns.heatmap(rate_diff, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5, ax=axes[1])
axes[1].set_title("右：二值标志的出现率差")
fig.suptitle("图 C-4　告警组 vs 正常组的特征差异（同车内比较，相关而非因果）")
fig.tight_layout()
fig.savefig(FIG / "C04_effect_size.png", dpi=150)
plt.show()"""))

cells.append(md("""## 10 梯度提升树超参精调（轻量版）

小网格搜索 + 训练段内的时间顺序交叉验证（ expanding window，3 折）。
目的：展示完整的调参过程；对提升幅度保持诚实预期。"""))

cells.append(code("""# 10.1 网格搜索（训练段子段，时间序 CV）
from sklearn.model_selection import GridSearchCV

grid = {"max_depth": [3, 5], "learning_rate": [0.03, 0.08],
        "n_estimators": [300, 600]}
ts = df.loc[sub_tr, "timestamp"].sort_values()
order = ts.index.to_numpy()
pos_map = {lab: i for i, lab in enumerate(order)}
N = len(order)
folds = []
for i in range(3):
    lo, hi = int(N * (i + 1) / 4), int(N * (i + 2) / 4)
    folds.append((np.array([pos_map[x] for x in order[:lo]]),
                  np.array([pos_map[x] for x in order[lo:hi]])))

best_params, tuned_f1 = {}, {}
for k in CODES:
    gs = GridSearchCV(xgb.XGBClassifier(subsample=0.8, tree_method="hist",
                                        random_state=42, n_jobs=4),
                      param_grid=grid, scoring="f1", cv=folds, n_jobs=2)
    gs.fit(Xtr, Ytr[k])
    best_params[k] = gs.best_params_
    tuned_f1[f"code{k}"] = round(gs.best_score_, 3)
print("各验证折最优 F1：", tuned_f1)
print("最优参数：")
for k in CODES:
    print(f"  code{k}: {best_params[k]}")"""))

cells.append(code("""# 10.2 精调后在测试段评估，与默认 XGB 对比
pred_xgb_tuned = {}
for k in CODES:
    p = dict(xgb_params); p.update(best_params[k])
    clf = xgb.XGBClassifier(**p)
    clf.fit(Xtr, Ytr[k])
    pred_xgb_tuned[k] = clf.predict(Xte)

res3 = pd.concat([
    evaluate(Y.loc[te_t], pred_xgb, "XGB默认"),
    evaluate(Y.loc[te_t], pred_xgb_tuned, "XGB精调"),
], axis=1)
print(res3.xs("F1", axis=1, level=1))
print("宏平均 F1：默认", float(res3[("XGB默认", "F1")]["宏平均"]),
      "→ 精调", float(res3[("XGB精调", "F1")]["宏平均"]))"""))

cells.append(md("""## 11 难点类深挖：4 号与 5 号为什么难

诊断两个问题：
1. **不可观测性**：4/5 号告警窗口里，有多大比例的特征完全缺失（全空）？这类样本理论上不可预测；
2. **模型在靠什么判断**：4/5 号模型的特征重要性排序，以及漏报（FN）样本里"全空"占多少。"""))

cells.append(code("""# 11.1 特征缺失对可预测性的限制
miss_stats = {}
for k in [4, 5]:
    pos_all = df.index[Y[k] == 1]
    all_nan = X.loc[pos_all, keep].isna().all(axis=1)
    some_obs = (~all_nan).sum()
    miss_stats[f"code{k}"] = (int(all_nan.sum()), int(some_obs),
                              round(float(all_nan.mean()), 3))
print("正样本窗口中特征全空 vs 有部分观测：")
for k, (nan_n, obs_n, nan_rate) in miss_stats.items():
    print(f"  {k}: 全空 {nan_n}（{nan_rate:.0%}），有观测 {obs_n}")

# 测试集漏报分析
fn_rows = []
for k in [4, 5]:
    yte = Y.loc[te_t, k]
    pos_mask = (yte.to_numpy() == 1)
    fn_mask = pos_mask & (pred_xgb_tuned[k] == 0)
    pos_te = yte.index[pos_mask]
    fn = yte.index[fn_mask]
    fn_all_nan = X.loc[fn, keep].isna().all(axis=1)
    fn_rows.append({"告警类": f"code{k}", "测试集正样本": len(pos_te), "漏报数": len(fn),
                    "漏报率": round(len(fn) / len(pos_te), 3),
                    "漏报中特征全空占比": round(float(fn_all_nan.mean()), 3)})
print()
print(pd.DataFrame(fn_rows))"""))

cells.append(code("""# 11.2 特征重要性（增益法，精调模型）
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, k in zip(axes, [4, 5]):
    p = dict(xgb_params); p.update(best_params[k])
    clf = xgb.XGBClassifier(**p).fit(Xtr, Ytr[k])
    imp = pd.Series(clf.feature_importances_, index=keep).sort_values().tail(10)
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"code{k} 特征重要性（gain 归一化）")
fig.suptitle("图 C-5　难点类的特征重要性")
fig.tight_layout()
fig.savefig(FIG / "C05_importance.png", dpi=150)
plt.show()"""))

cells.append(md("""## 12 总结

**建模结论**：
- 基线链：多数类（宏F1=0）→ LR默认（0.604）→ XGBoost 默认（0.779）→ 精调（0.779）；
- **超参精调零提升**——默认参数已接近该数据的上限；同时验证折 F1（0.93–0.97）远高于测试折（0.44–0.96），
  说明**性能瓶颈是时间分布漂移（尤其 code5：验证 0.97 vs 测试 0.44），不是模型或超参**；
- 不平衡处理（LR 权重 / XGB 权重 / 阈值调整）差异 <0.01：本数据的不平衡程度在树模型可处理范围内，无需强干预；
- 经纬度是"车辆身份"捷径，时间切分下失效并造成误报，已从特征中剔除；
- 难点类失败模式不同：code4 是**漏报型**（漏报率 50%，且漏报样本特征完整——告警窗口与正常窗口本身难以区分）；
  code5 是**误报型**（召回 97% 但精确率极低）——前者是可分性问题，后者是分布漂移问题。

**成因线索**（相关而非因果，编码含义待数据提供方确认）：
- 1–4 号高度共现、几乎只发生在点火状态（IGNITION_STATUS=1 出现率 +0.68~+0.86）、外接电池电压系统性偏高（Cliff's delta +0.11~0.23）——一组"行驶中、电气状态异常"的连锁告警；
- 5 号与点火状态无关（-0.07）、呈零散点状（事件合并几乎不缩样本）——独立的非行驶类告警。

**局限**：3 辆车、7 天数据；传感器稀疏（车速/转速覆盖率 6–7%），连续特征仅有外接电池电压支撑同车对比；
告警编码无业务语义，成因判断基于统计关联。"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, "C_alarm_analysis.ipynb")
print("notebook 已生成")
