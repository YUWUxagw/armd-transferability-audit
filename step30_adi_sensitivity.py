# -*- coding: utf-8 -*-
"""
Step 30 — adi 剔除敏感性对照 (模型层, 不动冻结数据)
对照: 含 adi vs 剔除 adi 的内部 CV + 跨系统缺口
决策依据: 核心结论是否随 adi 剔除而实质变化
2026-09-03 修复: 旧版 feats=feature_cols() 已排除 adi, feats_no 过滤无效 →
  两臂恒等空转 (Δ=0.000)。修复: with-adi 臂显式 feats+["adi"];
  跨站侧改用规范 fit_cat_maps/prep_cross (与 step16 一致)。
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
rep = open(os.path.join(BASE, "05_源数据", "phase3", "report_adi_sensitivity.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, cv_internal, fit_cat_maps, prep_cross
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
TARGETS = ["Stanford","UTSW"]

for task in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    maps = fit_cat_maps(mgb_raw)
    adi_med = mgb["adi"].median()
    feats_no = feature_cols(mgb)                 # 63 特征, adi 已排除 (EXCLUDE_FEATURES)
    feats_w = feats_no + ["adi"]                 # with-adi 臂显式加回
    cv_w = cv_internal(mgb, feats_w)["auroc"].mean()
    cv_no = cv_internal(mgb, feats_no)["auroc"].mean()
    m_w = train_lgb(mgb[feats_w], mgb["label"], mgb[feats_w], mgb["label"])
    m_no = train_lgb(mgb[feats_no], mgb["label"], mgb[feats_no], mgb["label"])
    p(f"\n{task}: 内部CV 含adi {cv_w:.3f} vs 剔除 {cv_no:.3f} (Δ {cv_no-cv_w:+.3f})")
    for tgt in TARGETS:
        te = prep_cross(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, adi_med)
        y = te["label"].values
        aw = roc_auc_score(y, m_w.predict_proba(te[[f for f in feats_w if f in te.columns]])[:, 1])
        an = roc_auc_score(y, m_no.predict_proba(te[[f for f in feats_no if f in te.columns]])[:, 1])
        p(f"  →{tgt}: 含adi {aw:.3f}(缺口 {aw-cv_w:+.3f}) vs 剔除 {an:.3f}(缺口 {an-cv_no:+.3f}) | "
          f"缺口变化 {(an-cv_no)-(aw-cv_w):+.3f}")

rep.close()
print("完成", flush=True)
