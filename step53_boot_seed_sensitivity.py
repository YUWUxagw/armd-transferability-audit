# -*- coding: utf-8 -*-
"""
Step 53 — bootstrap seed sensitivity (应检尽检 #6).
Joint-fix patient-level CIs (step50) used 200 iterations with seed=0.
This checks whether the CI conclusions are stable across bootstrap seeds:
  - 5 seeds (0-4) x 200 iterations x 10 joint-fix directions
  - Report per-seed CI; verdict: does the significant-residual set
    (internal outside CI) change across seeds?
"""
import os, sys, ast
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_boot_seed.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(BASE, "04_代码"))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols
from sklearn.metrics import roc_auc_score

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

# --- AST extract build/encode from step51 (zero drift, incl. constants) ---
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

def patient_boot(y, p, pid, n_boot=200, seed=0):
    rng = np.random.RandomState(seed)
    pids = np.unique(pid)
    aucs = []
    for _ in range(n_boot):
        sel = rng.choice(pids, size=len(pids), replace=True)
        cnt = pd.Series(sel).value_counts()
        idx = []
        for pid_i, c in cnt.items():
            idx.extend(np.where(pid == pid_i)[0].tolist() * c)
        idx = np.array(idx)
        if len(idx) < 2 or len(np.unique(y[idx])) < 2: continue
        try: aucs.append(roc_auc_score(y[idx], p[idx]))
        except ValueError: continue
    if len(aucs) < 50: return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

SEEDS = [0, 1, 2, 3, 4]
p("Bootstrap seed sensitivity: joint-fix patient CIs, 5 seeds x 200 iterations")
p("=" * 95)
rows = []
n_flip = 0
n_sig_all = 0
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw); adi = mgb_raw["adi"].median()
    # joint fix: drop-I + mucoid removal both sides + test-side >=2020 (MGB train never era-filtered)
    tr = encode(build("MGB", t, True, True, None), maps, adi)
    feats = feature_cols(tr)
    m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
    for site in ["Stanford", "UTSW"]:
        te = encode(build(site, t, True, True, 2020), maps, adi)
        X = te[[f for f in feats if f in te.columns]]
        y = te["label"].values
        pred = m.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)
        cis = [patient_boot(y, pred, te["anon_id"].values, seed=s) for s in SEEDS]
        los = [c[0] for c in cis]; his = [c[1] for c in cis]
        intern = INTERNAL[t]
        # verdict per seed: significant residual = internal above CI upper
        sig = [intern > hi for hi in his]
        stable = (all(sig) or not any(sig))
        n_sig_all += int(all(sig))
        if not stable: n_flip += 1
        rows.append(dict(task=t, tgt=site, auroc=round(auc,3), intern=intern,
                         lo_min=round(min(los),3), lo_max=round(max(los),3),
                         hi_min=round(min(his),3), hi_max=round(max(his),3),
                         sig_all_seeds=all(sig)))
        p(f"  {t} MGB\u2192{site[:1]}: AUC {auc:.3f} (intern {intern:.3f}) | "
          f"CI width swing lo {min(los):.3f}-{max(los):.3f} hi {min(his):.3f}-{max(his):.3f} | "
          f"significant-residual across seeds: {all(sig)} {'(stable)' if stable else '(FLIPS)'}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "boot_seed_sensitivity.csv"), index=False)
p("=" * 95)
p(f"Directions with significant residual under ALL seeds: {n_sig_all}/10")
p(f"Directions whose verdict FLIPS across seeds: {n_flip}/10")
p(f"SEED-SENSITIVITY VERDICT: {'STABLE' if n_flip == 0 else 'UNSTABLE (see rows)'}")
rep.close()
print("Done", flush=True)
