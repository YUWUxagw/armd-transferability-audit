# -*- coding: utf-8 -*-
"""Step 59 — 四个 pre-specified suspect 特征的**联合**消融（及共线性诊断）。

背景
----
step14 的消融是 leave-one-out：`for s in SUSPECTS: feats_abl = feats - {s}`。
四个 suspect 特征彼此相关（ICU 是 ward 的一个类别 → 与 ward_h 近共线；
mucoid 与 specimen=respiratory 相关），因此"单独删除任一个 ≤0.012"**无法排除
冗余补偿**——即四个特征作为一组携带捷径信息，删掉一个后其余特征把信息补了回来。

本轮补充（审阅意见 §6.11 M-93）：
  1. 四个一起删除（joint-4）
  2. 两个近共线对分别删除（ward_h+icu；mucoid+specimen）
  3. 四个 suspect 特征之间的相关性诊断
并重算 baseline 与逐一消融，保证与 step14 的口径完全一致、可同表比较。

用法
----
    python step59_joint_ablation.py            # 全跑（5 任务 × 7 次 CV）
    python step59_joint_ablation.py --minimal  # 只跑 baseline + 逐一 + 联合4（最快）

前置
----
需要清洗后的任务文件：<BASE>/data/clean/MGB/task_<task>.csv
（即 step14 的输入；本机无原始数据，故需在有数据的机器上运行）

输出
----
  <BASE>/05_源数据/phase2/joint_ablation.csv    机器可读，本文件供后续引用
  <BASE>/05_源数据/phase2/report_joint_ablation.txt
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- 路径：若在别的机器上跑，只改这一行 ----
BASE = r"F:\E\Machine Learning\ARMD"
# -------------------------------------------

CLEAN = os.path.join(BASE, "data", "clean", "MGB")
CODE = os.path.join(BASE, "04_代码")
OUT = os.path.join(BASE, "05_源数据", "phase2")
os.makedirs(OUT, exist_ok=True)

if not os.path.isdir(CLEAN):
    sys.exit(f"[中止] 找不到清洗后的数据目录：{CLEAN}\n"
             f"       请把它指向含 task_meropenem.csv 等文件的位置。")

# 复用 step14 的模型与特征构造，保证与已发表口径逐位一致
sys.path.insert(0, CODE)
from step14_model_phase2 import (prep, feature_cols, cv_internal,  # noqa: E402
                                 PRIMARY, SUSPECTS)

SUSPECT4 = ["mucoid", "ward_h", "specimen", "icu"]      # adi 已在 EXCLUDE_FEATURES 中
PAIRS = [("ward_h", "icu"), ("mucoid", "specimen")]     # 两个近共线对
MINIMAL = "--minimal" in sys.argv

# 已发表口径的内部 CV 基线（未舍入；唯一权威来源 transfer_matrix.csv 的 internal_auroc 列）。
# 用来在跑之前自检环境：若 baseline 对不上，说明 lightgbm/sklearn 版本与冻结环境不一致，
# 此时联合消融的 Δ 仍可参考，但绝对值不可与正文直接比较。
KNOWN_INTERNAL = {
    "meropenem": 0.7706173197175368, "ciprofloxacin": 0.7432910457446811,
    "levofloxacin": 0.7500579746900284, "ceftazidime": 0.769395700876451,
    "cefepime": 0.7567914463071654,
}
TOL = 2e-4

rep = open(os.path.join(OUT, "report_joint_ablation.txt"), "w", encoding="utf-8")


def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    rep.write(line + "\n")
    rep.flush()


p("=" * 92)
p("Step 59 — suspect 特征的联合消融与共线性诊断")
p(f"数据：{CLEAN}")
p(f"模式：{'minimal（baseline + 逐一 + 联合4）' if MINIMAL else 'full（+ 两个近共线对）'}")
p("=" * 92)

rows = []
env_warn = False
for task in PRIMARY:
    fp = os.path.join(CLEAN, f"task_{task}.csv")
    if not os.path.exists(fp):
        p(f"[{task}] 文件缺失，跳过：{fp}")
        continue
    t0 = time.time()
    df = prep(pd.read_csv(fp, low_memory=False, encoding="utf-8-sig"))
    feats = feature_cols(df)
    present = [s for s in SUSPECT4 if s in feats]
    p(f"\n===== {task} | n={len(df):,} | R 率 {df['label'].mean():.3f} | 特征 {len(feats)} =====")
    p(f"  可消融的 suspect：{present}"
      + (f"（缺失：{[s for s in SUSPECT4 if s not in feats]}）"
         if len(present) < len(SUSPECT4) else ""))

    base = cv_internal(df, feats)["auroc"].mean()
    known = KNOWN_INTERNAL.get(task)
    flag = ""
    if known is not None:
        ok = abs(base - known) < TOL
        flag = "  [OK] 与已发表口径一致" if ok else f"  [MISMATCH] 与已发表值 {known:.6f} 不符（差 {base - known:+.6f}）"
        if not ok:
            env_warn = True
    p(f"  [baseline] 内部 CV AUROC = {base:.6f}{flag}")

    # ① 逐一（复现 step14，验证 ≤0.012）
    for s in present:
        a = cv_internal(df, [f for f in feats if f != s])["auroc"].mean()
        rows.append(dict(task=task, ablation=f"drop_{s}", n_removed=1,
                         auroc=a, baseline=base, delta=a - base))
        p(f"  [drop {s:<9s}] AUROC = {a:.4f}   Δ = {a - base:+.4f}")

    # ② 联合删除四个
    a4 = cv_internal(df, [f for f in feats if f not in present])["auroc"].mean()
    rows.append(dict(task=task, ablation="drop_joint4", n_removed=len(present),
                     auroc=a4, baseline=base, delta=a4 - base))
    p(f"  [drop 四个联合 ] AUROC = {a4:.4f}   Δ = {a4 - base:+.4f}   ← 关键")

    # ③ 近共线对
    if not MINIMAL:
        for x, y in PAIRS:
            if x in feats and y in feats:
                a = cv_internal(df, [f for f in feats if f not in (x, y)])["auroc"].mean()
                rows.append(dict(task=task, ablation=f"drop_{x}+{y}", n_removed=2,
                                 auroc=a, baseline=base, delta=a - base))
                p(f"  [drop {x}+{y:<9s}] AUROC = {a:.4f}   Δ = {a - base:+.4f}")

    # ④ 共线性诊断
    cols = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) >= 2:
        corr = df[cols].corr()
        p("  [共线性] 数值型 suspect 之间的相关：")
        for i, ci in enumerate(cols):
            for cj in cols[i + 1:]:
                p(f"      {ci:<9s} × {cj:<9s} r = {corr.loc[ci, cj]:+.3f}")
    p(f"  耗时 {time.time() - t0:.0f}s")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "joint_ablation.csv"), index=False)

p("\n" + "=" * 92)
p("汇总（Δ = 消融后 AUROC − baseline；负值 = 消融后变差）")
p("=" * 92)
if len(res):
    piv = res.pivot_table(index="ablation", columns="task", values="delta")
    p(piv.round(4).to_string())
    p("")
    j = res[res["ablation"] == "drop_joint4"]["delta"]
    ind = res[res["n_removed"] == 1].groupby("task")["delta"]
    p(f"联合删除四个：Δ 范围 {j.min():+.4f} 至 {j.max():+.4f}，"
      f"最大 |Δ| = {j.abs().max():.4f}")
    p(f"逐一删除    ：每个任务的最大 |Δ| "
      f"{[round(float(v), 4) for v in ind.apply(lambda x: x.abs().max())]}")
    p("")
    p("判读：")
    p("  · 若联合 |Δ| 与逐一相当（≤ ~0.012）→ 结论可升级为")
    p("    'individually and jointly invisible to internal performance'；")
    p("  · 若联合 |Δ| 明显更大 → 存在冗余补偿，§4.6/SM6 的措辞需相应限定，")
    p("    且这本身是一个发现（相关特征组内的补偿效应）。")
p(f"\n输出：{os.path.join(OUT, 'joint_ablation.csv')}")
if env_warn:
    p("")
    p("[WARN] 环境警告：至少一个任务的 baseline 与已发表值不符（容差 2e-4）。")
    p("   说明 lightgbm / scikit-learn 版本与冻结环境不一致（见 requirements.txt）。")
    p("   此时 Δ（消融前后之差）仍可用，但绝对值不可与正文直接比较。")
rep.close()
print("完成", flush=True)
