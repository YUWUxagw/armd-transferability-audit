# -*- coding: utf-8 -*-
"""
Step 38 — mucoid 根因案例敏感性 (评论人1 第三轮)
A. S/U 剔除 mucoid 后主方向缺口变化 (MGB 训练队列 mucoid 仅 13 培养, 训练侧无需剔)
B. S/U mucoid 分层性能 (mucoid-only vs 非 mucoid AUROC — 域外机制检验)
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_mucoid_sensitivity.txt"), "w", encoding="utf-8")
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

rows = []
for task in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    feats = feature_cols(mgb)
    m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
    maps = fit_cat_maps(mgb_raw)
    for tgt in TARGETS:
        te_raw = pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
        te = prep_cross(te_raw, maps, mgb["adi"].median())
        y = te["label"].values
        pred = m.predict_proba(te[[f for f in feats if f in te.columns]])[:, 1]
        auc_all = roc_auc_score(y, pred)
        # A. 剔除 mucoid
        nm = te_raw["mucoid"].astype(str).str.strip().isin(["1","1.0"])
        te_nm = te[~nm.values]
        auc_nm = roc_auc_score(te_nm["label"], m.predict_proba(te_nm[[f for f in feats if f in te_nm.columns]])[:, 1])
        # B. mucoid-only 分层
        te_m = te[nm.values]
        if len(te_m) >= 100 and te_m["label"].nunique() > 1:
            auc_m = roc_auc_score(te_m["label"], m.predict_proba(te_m[[f for f in feats if f in te_m.columns]])[:, 1])
        else:
            auc_m = None
        m_str = f"{auc_m:.3f}" if auc_m is not None else "NA(样本不足)"
        p(f"  {task}→{tgt}: 全量 {auc_all:.3f}(n={len(te)}) | 剔除mucoid {auc_nm:.3f}(n={len(te_nm)}, "
          f"mucoid {int(nm.sum())} 行) | mucoid-only AUROC {m_str} (n={len(te_m)})")
        rows.append(dict(task=task, tgt=tgt, auroc_all=round(auc_all,3),
                         auroc_no_mucoid=round(auc_nm,3), n_no_mucoid=len(te_nm),
                         n_mucoid=int(nm.sum()), auroc_mucoid_only=round(auc_m,3) if auc_m else None))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "mucoid_sensitivity.csv"), index=False)
rep.close()
print("完成", flush=True)
