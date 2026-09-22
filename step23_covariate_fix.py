# -*- coding: utf-8 -*-
"""
Step 23 — 协变量偏移次级指标 (受限连续特征, 避免分类器退化)
替换: 全特征双样本分类器(0.999退化) → 连续特征子集 + SMD 主报告
输出: results/phase3/shift_covariate_restricted.csv
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_covariate_restricted.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
TARGETS = ["Stanford","UTSW"]
# 受限连续特征 (避免编码类别/罕见特征造成的完美可分)
CONT = ["adi","comorb_count","prior_pseudo_days","age_bin","prior_pa_res_meropenem",
        "prior_pa_res_ciprofloxacin","prior_pa_res_levofloxacin","proc_cvc","nh_30d"]

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

rows = []
for task in TASKS:
    mgb = prep(pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"))
    cols = [c for c in CONT if c in mgb.columns]
    for tgt in TARGETS:
        te = prep(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"))
        te = te[[c for c in cols if c in te.columns]]
        a = cov_auc(mgb, te, cols)
        rows.append(dict(task=task, tgt=tgt, covariate_auc_restricted=round(a, 3)))
        p(f"[{task}→{tgt}] 受限特征协变量AUC: {a:.3f} (连续特征 {len(cols)} 个)")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "shift_covariate_restricted.csv"), index=False)
rep.close()
print("完成", flush=True)
