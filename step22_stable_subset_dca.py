# -*- coding: utf-8 -*-
"""
ARMD Step 22 — 假设三: 稳定特征子集模型 + DCA + bootstrap CI
- 稳定子集: step15 候选稳定特征 top-K (按 mean_shap, 任务特异)
- 比较: 全特征 vs 稳定子集, 内部CV + 跨系统(MGB→S/U), 迁移缺口
- DCA: 跨系统预测净获益 (目标侧阈值) vs 全治疗/全不治
- Bootstrap: 患者级重抽样 95% CI (主方向 AUROC)
输出: E:\\ARMD\\results\\phase3\\stable_subset.csv + dca.csv + report
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
os.makedirs(OUT, exist_ok=True)
rep = open(os.path.join(OUT, "report_stable_subset.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, metrics, cv_internal, fit_cat_maps, prep_cross
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
TARGETS = ["Stanford","UTSW"]
K = 10  # 稳定子集大小

stable = pd.read_csv(os.path.join(BASE, "05_源数据", "phase2", "stable_features.csv"))
stable_set = {}
for t in TASKS:
    s = stable[(stable["task"] == t) & (stable["rank_stability"] >= 0.7) & (stable["sign_consistency"] >= 0.75)]
    stable_set[t] = s.sort_values("mean_shap", ascending=False).head(K)["feature"].tolist()

def patient_boot_ci(y, p, n_boot=200, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        try: aucs.append(roc_auc_score(y[idx], p[idx]))
        except ValueError: pass
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def dca(y, p, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for pt in thresholds:
        tp = ((p >= pt) & (y == 1)).sum() / len(y)
        fp = ((p >= pt) & (y == 0)).sum() / len(y)
        nb = tp - fp * pt / (1 - pt)          # 净获益 (获益=1 代价=pt/(1-pt))
        rows.append(dict(threshold=round(pt, 2), net_benefit=round(nb, 4)))
    return pd.DataFrame(rows)

rows, dca_rows = [], []
for task in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    feats = feature_cols(mgb)
    sset = [f for f in stable_set[task] if f in feats]
    p(f"\n===== {task} | 全特征 {len(feats)} vs 稳定子集 {len(sset)} =====")
    # 内部 CV (全特征 vs 子集)
    cv_full = cv_internal(mgb, feats)["auroc"].mean()
    cv_sub = cv_internal(mgb, sset)["auroc"].mean()
    # 跨系统 (MGB 全量训练)
    m_full = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
    m_sub = train_lgb(mgb[sset], mgb["label"], mgb[sset], mgb["label"])
    maps = fit_cat_maps(mgb_raw)
    for tgt in TARGETS:
        te = prep_cross(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, mgb["adi"].median())
        y = te["label"].values
        Xf = te[[f for f in feats if f in te.columns]]
        Xs = te[[f for f in sset if f in te.columns]]
        pf = m_full.predict_proba(Xf)[:, 1]
        ps = m_sub.predict_proba(Xs)[:, 1]
        auc_f = roc_auc_score(y, pf); auc_s = roc_auc_score(y, ps)
        lo, hi = patient_boot_ci(y, pf)
        rows.append(dict(task=task, tgt=tgt, n_test=len(te),
                         full_internal=round(cv_full, 3), subset_internal=round(cv_sub, 3),
                         full_external=round(auc_f, 3), subset_external=round(auc_s, 3),
                         full_gap=round(auc_f - cv_full, 3), subset_gap=round(auc_s - cv_sub, 3),
                         gap_reduction=round((auc_f - cv_full) - (auc_s - cv_sub), 3),
                         boot_ci=f"{lo:.3f}-{hi:.3f}"))
        p(f"  [{tgt}] 全特征 {auc_f:.3f}(缺口 {auc_f-cv_full:+.3f}) vs 子集 {auc_s:.3f}(缺口 {auc_s-cv_sub:+.3f}) | "
          f"缺口变化 {rows[-1]['gap_reduction']:+.3f} | 95%CI {lo:.3f}-{hi:.3f}")
        d = dca(y, pf)
        d["task"] = task; d["tgt"] = tgt; d["model"] = "full"; dca_rows.append(d)
        d2 = dca(y, ps)
        d2["task"] = task; d2["tgt"] = tgt; d2["model"] = "subset"; dca_rows.append(d2)
    p(f"  内部CV: 全特征 {cv_full:.3f} vs 子集 {cv_sub:.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "stable_subset.csv"), index=False)
pd.concat(dca_rows, ignore_index=True).to_csv(os.path.join(OUT, "dca.csv"), index=False)
p("\n已存 stable_subset.csv + dca.csv")
rep.close()
print("完成", flush=True)
