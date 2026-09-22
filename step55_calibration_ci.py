# -*- coding: utf-8 -*-
"""
Step 55 — calibration slope/intercept patient-level bootstrap CIs (终审 R2).
10 primary directions (MGB -> S/U), 200 patient-level bootstrap iterations.
Per iteration: refit logit(y ~ logit(p)) on the resampled patients;
percentile 2.5/97.5 -> 95% CI for slope and intercept.
"""
import os, sys, ast
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_calibration_ci.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(BASE, "04_代码"))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def _load_internal():
    """内参运行时从权威结果文件读取，禁止字面量（2026-09-18 修复）。

    原字面量 {0.795,0.765,0.764,0.804,0.790} 是 early-stopping 泄漏修复前的
    膨胀值，与 transfer_matrix.csv 的 internal_auroc 不符；按归档脚本重跑会
    系统性偏离已发表数字（例如 MEM→S 的 −0.112 会被算成 −0.136）。
    """
    _tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))
    _out = {}
    for _t in ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]:
        _v = _tm[_tm["task"] == _t]["internal_auroc"].unique()
        assert len(_v) == 1, f"internal_auroc not unique for {_t}"
        _out[_t] = float(_v[0])
    return _out


INTERNAL = _load_internal()TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]

src = open(os.path.join(BASE, "04_代码", "step51_pairwise_fixes.py"), encoding="utf-8").read()
tree = ast.parse(src)
ns = {"os": os, "np": np, "pd": pd, "sys": sys}
CONST_NAMES = ("BASE","CLEAN","OUT","TASKS","COHORT","PHENO","POSCOL","NEGCOL","PRELIM","INTERNAL")
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("build","encode"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<step51>", "exec"), ns)
    elif isinstance(node, ast.Assign):
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in CONST_NAMES:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<step51>", "exec"), ns)
build, encode = ns["build"], ns["encode"]

def calib_slope_int(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    lr = LogisticRegression(max_iter=2000).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def patient_boot_calib(y, p, pid, n_boot=200, seed=0):
    rng = np.random.RandomState(seed)
    pids = np.unique(pid)
    sl, it = [], []
    for _ in range(n_boot):
        sel = rng.choice(pids, size=len(pids), replace=True)
        cnt = pd.Series(sel).value_counts()
        idx = []
        for pid_i, c in cnt.items():
            idx.extend(np.where(pid == pid_i)[0].tolist() * c)
        idx = np.array(idx)
        if len(idx) < 50 or len(np.unique(y[idx])) < 2:
            continue
        try:
            s, i = calib_slope_int(y[idx], p[idx])
            sl.append(s); it.append(i)
        except ValueError:
            continue
    if len(sl) < 50:
        return None, None, None, None
    return (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5)),
            float(np.percentile(it, 2.5)), float(np.percentile(it, 97.5)))

p("Calibration slope/intercept patient-level CIs (primary directions, 200 boot)")
p("=" * 92)
rows = []
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw); adi = mgb_raw["adi"].median()
    tr = encode(build("MGB", t, False, False, None), maps, adi)
    feats = feature_cols(tr)
    m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
    for site in ["Stanford", "UTSW"]:
        te = encode(build(site, t, False, False, None), maps, adi)
        X = te[[f for f in feats if f in te.columns]]
        y = te["label"].values
        pred = m.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)
        s0, i0 = calib_slope_int(y, pred)
        slo, shi, ilo, ihi = patient_boot_calib(y, pred, te["anon_id"].values)
        rows.append(dict(task=t, tgt=site, n=len(te), auroc=round(auc,3),
                         slope=round(s0,3), slope_ci_lo=round(slo,3) if slo else None,
                         slope_ci_hi=round(shi,3) if shi else None,
                         intercept=round(i0,3), int_ci_lo=round(ilo,3) if ilo else None,
                         int_ci_hi=round(ihi,3) if ihi else None))
        p(f"  {t} {site[:1]} | slope {s0:.2f} [{slo:.2f}-{shi:.2f}] | "
          f"intercept {i0:.2f} [{ilo:.2f}-{ihi:.2f}] (n={len(te):,}, AUROC {auc:.3f})")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "calibration_ci.csv"), index=False)
p("\nsaved: calibration_ci.csv")
rep.close()
print("Done", flush=True)
