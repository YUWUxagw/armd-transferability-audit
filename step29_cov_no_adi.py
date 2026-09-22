# -*- coding: utf-8 -*-
"""
Step 29 — 去除 adi 后: 协变量可分性 + SMD 复核
(MGB adi=原始分数 vs S/U=百分位, 标尺不一致 → 检查可分性是否依赖 adi)
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
rep = open(os.path.join(BASE, "05_源数据", "phase3", "report_cov_no_adi.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]
TARGETS = ["Stanford", "UTSW"]
CONT = ["comorb_count", "prior_pseudo_days", "age_bin", "proc_cvc", "nh_30d",
        "prior_pa_res_meropenem", "prior_pa_res_ciprofloxacin", "prior_pa_res_levofloxacin"]

def cov_auc(tr, te, cols):
    a = tr[cols].sample(min(4000, len(tr)), random_state=42)
    b = te[cols].sample(min(4000, len(te)), random_state=42)
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = train_lgb(X.iloc[tri], y[tri], X.iloc[vai], y[vai])
        aucs.append(roc_auc_score(y[vai], m.predict_proba(X.iloc[vai])[:, 1]))
    return float(np.mean(aucs))

def smd(tr, te, cols):
    out = {}
    for c in cols:
        a, b = tr[c].astype(float), te[c].astype(float)
        out[c] = abs(float((a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-9)))
    return sorted(out.items(), key=lambda x: -x[1])[:4]

for task in TASKS:
    mgb = prep(pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"))
    cols = [c for c in CONT if c in mgb.columns]
    for tgt in TARGETS:
        te = prep(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"))
        te = te[[c for c in cols if c in te.columns]]
        a = cov_auc(mgb, te, cols)
        s = smd(mgb, te, cols)
        p(f"  {task}→{tgt}: 无 adi 协变量AUC {a:.3f} | SMDtop: " + " | ".join(f"{f}:{v:.2f}" for f, v in s))

rep.close()
print("完成", flush=True)
