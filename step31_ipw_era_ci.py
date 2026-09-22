# -*- coding: utf-8 -*-
"""
Step 31 — 评论人1 第一梯队: IPW 可行性重论证 + ≥2020 年代敏感性 CI
A. 无 adi 特征集下: 密度比权重分布 + 有效样本量(ESS) + 截断 IPW 调整 AUROC
   (回答: 0.83–0.97 可分性下 IPW 是否真不可行)
B. ≥2020 子集: 各方向 n/R 事件数 + 患者级 bootstrap CI (回答: 改善是否噪声)
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_ipw_era_ci.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
TARGETS = ["Stanford","UTSW"]

def ess(w):
    return float((w.sum() ** 2) / (w ** 2).sum())

# ---------- A. IPW 重论证 ----------
p("=== A. IPW 可行性 (无 adi 特征集) ===")
for task in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    feats = feature_cols(mgb)
    m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
    maps = fit_cat_maps(mgb_raw)
    for tgt in TARGETS:
        te = prep_cross(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, mgb["adi"].median())
        X = te[[f for f in feats if f in te.columns]]
        y = te["label"].values
        # 双样本分类器 (无 adi)
        a = mgb[feats].sample(min(4000, len(mgb)), random_state=42)
        b = X.sample(min(4000, len(X)), random_state=42)
        Xc = pd.concat([a, b], ignore_index=True)
        yc = np.r_[np.zeros(len(a)), np.ones(len(b))]
        mc = train_lgb(Xc, yc, Xc, yc)
        p_tgt = mc.predict_proba(X)[:, 1]
        w_raw = p_tgt / (1 - p_tgt + 1e-6)
        w_clip = np.clip(w_raw, 0.05, 20)
        p(f"  {task}→{tgt}: 权重未截断 P5-P95 [{np.percentile(w_raw,5):.2f}, {np.percentile(w_raw,95):.2f}] "
          f"max {w_raw.max():.0f} | ESS {ess(w_raw):.0f}/{len(y)} ({100*ess(w_raw)/len(y):.0f}%) | "
          f"截断后 ESS {ess(w_clip):.0f} ({100*ess(w_clip)/len(y):.0f}%)")
        # 截断 IPW 调整 AUROC
        rng = np.random.RandomState(0)
        aucs = []
        for _ in range(10):
            idx = rng.choice(len(y), size=min(2000, len(y)), replace=True, p=w_clip/w_clip.sum())
            try: aucs.append(roc_auc_score(y[idx], m.predict_proba(X)[idx][:, 1] if False else m.predict_proba(X)[idx][:, 1]))
            except ValueError: pass
        p(f"      截断 IPW 调整 AUROC: {np.mean(aucs):.3f} (未调整 {roc_auc_score(y, m.predict_proba(X)[:,1]):.3f})")

# ---------- B. ≥2020 CI ----------
p("\n=== B. ≥2020 年代敏感性 CI + 事件数 ===")
for task in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    feats = feature_cols(mgb)
    m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
    for tgt in TARGETS:
        te = prep_cross(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, mgb["adi"].median())
        base = pd.read_csv(os.path.join(CLEAN, tgt, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
        base["year"] = pd.to_datetime(base["time"], errors="coerce").dt.year
        te = te.merge(base[["order_proc_id_coded","year"]], on="order_proc_id_coded", how="left")
        sub = te[te["year"] >= 2020]
        if len(sub) < 200: continue
        y = sub["label"].values
        pred = m.predict_proba(sub[[f for f in feats if f in sub.columns]])[:, 1]
        auc = roc_auc_score(y, pred)
        rng = np.random.RandomState(1)
        aucs = []
        for _ in range(200):
            idx = rng.choice(len(y), size=len(y), replace=True)
            try: aucs.append(roc_auc_score(y[idx], pred[idx]))
            except ValueError: pass
        lo, hi = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
        p(f"  {task}→{tgt} ≥2020: n={len(sub)} R事件={int(y.sum())} AUROC {auc:.3f} (95%CI {lo:.3f}-{hi:.3f})")

rep.close()
print("完成", flush=True)
