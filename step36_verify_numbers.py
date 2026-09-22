# -*- coding: utf-8 -*-
"""
Step 36 — 数字一致性裁决: CIP→S / MEM→S 外部 AUROC 的四种计算路径
(a) step16 式: prep_cross(MGB 拟合编码) + adi 剔除
(b) step30 式: prep(独立编码) + adi 剔除
(c) step16 式 + adi 保留
(d) step30 式 + adi 保留
对照历史值: 0.637(adi-in 链) / 0.654+0.689(step30) / 0.673(final step16)
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import step14_model_phase2 as m2
from sklearn.metrics import roc_auc_score

def prep_cross_local(df, maps, adi_med):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years", "", regex=False)
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
    df["adi"] = df["adi"].fillna(adi_med)
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df

for task in ["meropenem", "ciprofloxacin"]:
    mgb = m2.prep(pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"))
    te_raw = pd.read_csv(os.path.join(CLEAN, "Stanford", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    y = te_raw["label"].values
    # 编码一致性检查
    for c in ["age_bin", "ward_h", "specimen"]:
        a = sorted(set(mgb[c].astype(str)))
        b = sorted(set(te_raw[c].astype(str)))
        print(f"  [{task}] {c}: MGB 值集 {a} vs S 值集 {b} -> {'一致' if a == b else '不一致!'}")
    # 路径 (a): prep_cross 式 (MGB 拟合)
    maps = {}
    for c in ["age_bin", "ward_h", "specimen"]:
        vals = sorted(mgb[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    adi_med = mgb["adi"].median()
    feats_a = [f for f in m2.feature_cols(mgb)]
    feats_c = [f for f in m2.feature_cols(mgb) if f != "adi"]
    m_a = m2.train_lgb(mgb[feats_a], mgb["label"], mgb[feats_a], mgb["label"])
    m_c = m2.train_lgb(mgb[feats_c], mgb["label"], mgb[feats_c], mgb["label"])
    te_cross = prep_cross_local(te_raw, maps, adi_med)
    auc_a = roc_auc_score(y, m_a.predict_proba(te_cross[[f for f in feats_a if f in te_cross.columns]])[:, 1])
    auc_c = roc_auc_score(y, m_c.predict_proba(te_cross[[f for f in feats_c if f in te_cross.columns]])[:, 1])
    # 路径 (b): prep 式 (独立编码)
    te_ind = m2.prep(te_raw)
    auc_b = roc_auc_score(y, m_c.predict_proba(te_ind[[f for f in feats_c if f in te_ind.columns]])[:, 1])
    print(f"  [{task}→S] (a)cross+adi-out: {auc_c:.3f} | (c)cross+adi-in: {auc_a:.3f} | (b)独立编码+adi-out: {auc_b:.3f}")
