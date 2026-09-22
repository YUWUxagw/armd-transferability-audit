# -*- coding: utf-8 -*-
"""
Step 28 — 折点年代敏感性 (评论人1 第一优先)
限制 S/U 至 ≥2020 年 (折点相对稳定年代), MGB 全量训练模型重评估
比较: 全量 vs ≥2020 子集的跨系统 AUROC 缺口
若缺口显著缩小 → 折点漂移贡献大; 若相似 → 影响有限
输出: results/phase3/era_sensitivity.csv
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_era_sensitivity.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross
from sklearn.metrics import roc_auc_score, brier_score_loss

def calib_slope(y, p):
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=500).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0])

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
TARGETS = ["Stanford","UTSW"]
CUT = 2020

rows = []
for task in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    feats = feature_cols(mgb)
    m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
    maps = fit_cat_maps(mgb_raw)
    for tgt in TARGETS:
        te = prep_cross(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, mgb["adi"].median())
        # 基座有 time → 取任务文件的时间列不存在, 需从基座取
        base = pd.read_csv(os.path.join(CLEAN, tgt, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
        base["time"] = pd.to_datetime(base["time"], errors="coerce")
        base["year"] = base["time"].dt.year
        yr = base[["order_proc_id_coded","year"]].dropna()
        te = te.merge(yr, on="order_proc_id_coded", how="left")
        y, X = te["label"].values, te[[f for f in feats if f in te.columns]]
        pred = m.predict_proba(X)[:, 1]
        auc_all = roc_auc_score(y, pred)
        b_all = brier_score_loss(y, pred)
        s_all = calib_slope(y, pred)
        for cut in [CUT, 2015]:
            sub = te[te["year"] >= cut]
            if len(sub) >= 200 and sub["label"].nunique() > 1:
                ps = m.predict_proba(sub[[f for f in feats if f in sub.columns]])[:, 1]
                auc_cut = roc_auc_score(sub["label"], ps)
                b_cut = brier_score_loss(sub["label"], ps)
                s_cut = calib_slope(sub["label"], ps)
                rows.append(dict(task=task, tgt=tgt, cut=cut, n_all=len(te), n_cut=len(sub),
                                 pct=round(100*len(sub)/len(te),1),
                                 auroc_all=round(auc_all,3), auroc_cut=round(auc_cut,3),
                                 delta=round(auc_cut-auc_all,3),
                                 brier_all=round(b_all,3), brier_cut=round(b_cut,3),
                                 slope_all=round(s_all,3), slope_cut=round(s_cut,3)))
                p(f"  {task}→{tgt}: 全量 {auc_all:.3f}(n={len(te)}) | ≥{cut} {auc_cut:.3f}(n={len(sub)}, "
                  f"{100*len(sub)/len(te):.0f}%) | Δ {auc_cut-auc_all:+.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "era_sensitivity.csv"), index=False)
p("\n已存 era_sensitivity.csv")
rep.close()
print("完成", flush=True)
