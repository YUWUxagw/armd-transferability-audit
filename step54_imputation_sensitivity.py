# -*- coding: utf-8 -*-
"""
Step 54 — missing-data imputation sensitivity (应检尽检 #7).
Primary pipeline imputes prior_pseudo_days=-1; alternative: 0 (no prior exposure).
2026-09-03 (汇总意见 L34): adi median-vs-mean arms removed — adi is excluded from
the primary feature set (EXCLUDE_FEATURES), so those arms were vacuous no-ops.
Sensitivity now covers the prior_pseudo_days fill rule only (2 rules x 10 directions).
For each rule, recompute all 10 MGB->S/U primary directions; check whether
gap ordering and the residual set are preserved vs the primary rule.
"""
import os, sys, ast
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_imputation_sensitivity.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(BASE, "04_代码"))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]

def _load_internal():
    tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))
    out = {}
    for t in TASKS:
        vals = tm[tm["task"]==t]["internal_auroc"].unique()
        assert len(vals) == 1, f"internal_auroc not unique for {t}"
        out[t] = float(vals[0])
    return out
INTERNAL = _load_internal()

src = open(os.path.join(BASE, "04_代码", "step51_pairwise_fixes.py"), encoding="utf-8").read()
tree = ast.parse(src)
ns = {"os": os, "np": np, "pd": pd, "sys": sys}
CONST_NAMES = ("BASE","CLEAN","OUT","TASKS","COHORT","PHENO","POSCOL","NEGCOL","PRELIM")
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("build","encode"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<step51>", "exec"), ns)
    elif isinstance(node, ast.Assign):
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in CONST_NAMES:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<step51>", "exec"), ns)
build, encode = ns["build"], ns["encode"]

def encode_alt(df, maps, adi_med, pdays_fill, adi_fill):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years", "", regex=False)
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
    df["adi"] = df["adi"].fillna(adi_fill)
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(pdays_fill)
    return df

RULES = {"primary(pdays=-1)": -1,
         "altA pdays=0":      0}

tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))
p("Imputation sensitivity: 10 primary directions x 2 rules (gap vs canonical)")
p("=" * 100)
rows = []
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw); adi_med = mgb_raw["adi"].median(); adi_mean = mgb_raw["adi"].mean()
    feats_ref = None
    for site in ["Stanford", "UTSW"]:
        for rname, pdays_fill in RULES.items():
            adi_fill = adi_med
            tr = encode_alt(build("MGB", t, False, False, None), maps, adi_fill, pdays_fill, adi_fill)
            feats = feature_cols(tr)
            if feats_ref is None: feats_ref = feats
            m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
            te = encode_alt(build(site, t, False, False, None), maps, adi_fill, pdays_fill, adi_fill)
            X = te[[f for f in feats if f in te.columns]]
            auc = roc_auc_score(te["label"], m.predict_proba(X)[:, 1])
            gap = round(auc - INTERNAL[t], 3)
            rows.append(dict(task=t, tgt=site, rule=rname, auroc=round(auc,3), gap=gap))
            p(f"  {t} {site[:1]} | {rname:22s} gap {gap:+.3f} (AUROC {auc:.3f})")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "imputation_sensitivity.csv"), index=False)
# ---- verdict: per direction, max gap swing across rules & rank stability ----
p("=" * 100)
dfr = pd.DataFrame(rows)
canon = []
for t in TASKS:
    for site in ["Stanford", "UTSW"]:
        rr = tm[(tm["task"]==t)&(tm["src"]=="MGB")&(tm["tgt"]==site)].iloc[0]
        canon.append(dict(task=t, tgt=site, gap=round(rr["auroc"]-INTERNAL[t],3)))
canon = pd.DataFrame(canon)
m = dfr.merge(canon, on=["task","tgt"], suffixes=("_rule","_canon"))
m["dev"] = (m["gap_rule"] - m["gap_canon"]).abs()
p("Per-direction max |deviation| from canonical gap across the 4 rules:")
for _, r in m.groupby(["task","tgt"])["dev"].max().reset_index().iterrows():
    p(f"  {r['task']} {r['tgt'][:1]}: max |dev| = {r['dev']:.3f}")
maxdev = m.groupby(["task","tgt"])["dev"].max().max()
# rank stability: order 10 directions by gap under each rule vs canonical rank
canon_rank = canon.sort_values("gap", ascending=False).reset_index(drop=True)
for rname in RULES:
    r_sub = dfr[dfr["rule"]==rname].sort_values("gap", ascending=False).reset_index(drop=True)
    same = (canon_rank["task"] == r_sub["task"]).all() and (canon_rank["tgt"] == r_sub["tgt"]).all()
    p(f"  rank order identical to canonical under '{rname}': {same}")
p(f"MAX DEVIATION across all rules and directions: {maxdev:.3f} "
  f"(VERDICT: {'ROBUST' if maxdev < 0.02 else 'CHECK BELOW'})")
rep.close()
print("Done", flush=True)
