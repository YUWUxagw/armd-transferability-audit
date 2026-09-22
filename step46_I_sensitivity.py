# -*- coding: utf-8 -*-
"""
Step 46 — Intermediate-exclusion sensitivity analysis (never run before).

Spec promised "I merged into R (primary); sensitivity analysis drops it".
The frozen task CSVs merge I into R and do not retain I identity, so this
analysis must rebuild cohorts from the RAW cohort tables, which also serves
as an end-to-end reproducibility check of the cleaning pipeline.

Protocol:
  - Labels: rows with Intermediate are EXCLUDED (only S/R used).
  - Applied to ALL sites (train and test sides) for comparability.
  - Culture-level dedup over S/R rows (any R -> resistant).
  - Features: reuse the frozen site_base_v3 features (F1-F9). F5 (prior
    resistance history) retains its R/I definition in the main analysis;
    difference declared as a secondary deviation (conservative direction:
    I-inclusive history matches the primary analysis).
  - Compare internal CV (MGB) and cross-system AUROC vs primary results.
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
RAW   = os.path.join(BASE, "ARMD-MGB")
CLEAN = os.path.join(BASE, "data", "clean")
OUT   = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_I_sensitivity.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import train_lgb, fit_cat_maps, prep_cross, feature_cols, prep
from sklearn.metrics import roc_auc_score

TASKS  = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SITES  = ["MGB","Stanford","UTSW"]
COHORT = {"MGB": os.path.join(BASE,"ARMD-MGB","microbiology_cohort_deid_tj_updated.csv"),
          "Stanford": os.path.join(BASE,"ARMD-Stanford","microbiology_cultures_cohort.csv"),
          "UTSW": os.path.join(BASE,"ARMD-UTSW","microbiology_cultures_cohort.csv")}
PHENO  = {"MGB":"CLSI_2022_pheno","Stanford":"susceptibility","UTSW":"susceptibility"}
POSCOL = {"MGB":None,"Stanford":"was_positive","UTSW":"was_positive"}
NEGCOL = {"MGB":"neg_cx","Stanford":None,"UTSW":None}
PRELIM = {"MGB":"prelim_AST","Stanford":None,"UTSW":None}

def build_dropI(site, task):
    cols = ["anon_id","order_proc_id_coded","organism","antibiotic",PHENO[site]]
    if POSCOL[site]: cols.append(POSCOL[site])
    if NEGCOL[site]: cols.append(NEGCOL[site])
    if PRELIM[site]: cols.append(PRELIM[site])
    df = pd.read_csv(COHORT[site], usecols=cols, low_memory=False, encoding="utf-8-sig")
    org = df["organism"].astype(str).str.upper()
    pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
    drug = df["antibiotic"].astype(str).str.lower().str.replace("/","_",regex=False).str.replace("-","_",regex=False).str.strip()
    m = pa & drug.eq(task)
    if NEGCOL[site]:
        m &= ~df[NEGCOL[site]].astype(str).str.strip().eq("X")
    if PRELIM[site]:
        m &= ~df[PRELIM[site]].astype(str).str.strip().eq("X")
    if POSCOL[site]:
        m &= df[POSCOL[site]].astype(str).str.strip().isin(["1","1.0"])
    sub = df[m].copy()
    ph = sub[PHENO[site]].astype(str).str.upper().str.strip()
    S = ph.eq("SUSCEPTIBLE"); R = ph.eq("RESISTANT")
    sub = sub[S | R]                      # I EXCLUDED (drop-I sensitivity)
    sub["label"] = R[S | R].astype(int)
    sub = sub.sort_values("label", ascending=False).drop_duplicates("order_proc_id_coded")
    # attach frozen features (drop duplicate anon_id from base to avoid _x/_y)
    base = pd.read_csv(os.path.join(CLEAN, site, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
    out = sub[["order_proc_id_coded","anon_id","label"]].merge(
        base.drop(columns=["anon_id"]), on="order_proc_id_coded", how="left")
    # 规范特征集对齐: 同药物既往耐药特征 prior_pa_res_{task} 在规范管线
    # (task 文件) 中不存在 (同药物既往耐药为泄漏控制排除); base 含此列,
    # 必须剔除, 否则 feature_cols() 数据驱动会把该强预测特征带入模型
    out = out.drop(columns=[f"prior_pa_res_{task}"], errors="ignore")
    # encode categoricals with MGB-fitted maps (consistent with main pipeline)
    raw_ref = pd.read_csv(os.path.join(CLEAN, site, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(raw_ref)
    out = prep_cross(out, maps, out["adi"].median())
    return out

def run():
    # Internal CV on MGB (drop-I cohort)
    p("I-exclusion sensitivity (labels: Intermediate excluded on all sites)")
    p("=" * 90)
    intern = {}
    for t in TASKS:
        mgb = build_dropI("MGB", t)
        nR = int(mgb["label"].sum()); prev = mgb["label"].mean()
        # patient-level 5-fold CV using fold_id from base
        feats = feature_cols(mgb)
        aucs = []
        for f in sorted(mgb["fold_id"].unique()):
            tr, va = mgb[mgb["fold_id"]!=f], mgb[mgb["fold_id"]==f]
            m = train_lgb(tr[feats], tr["label"], va[feats], va["label"])
            aucs.append(roc_auc_score(va["label"], m.predict_proba(va[feats])[:,1]))
        intern[t] = float(np.mean(aucs))
        p(f"[{t}] drop-I cohort n={len(mgb):,} nR={nR:,} prev={prev:.3f} | internal CV AUROC {intern[t]:.3f} "
          f"(primary { {'meropenem':0.795,'ciprofloxacin':0.765,'levofloxacin':0.764,'ceftazidime':0.804,'cefepime':0.790}[t] })")
    p("-" * 90)
    # Cross-system: MGB train -> S/U test (drop-I on both sides)
    for t in TASKS:
        mgb = build_dropI("MGB", t)
        feats = feature_cols(mgb)
        maps = fit_cat_maps(pd.read_csv(os.path.join(CLEAN,"MGB",f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig"))
        adi_med = mgb["adi"].median()
        m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
        for site in ["Stanford","UTSW"]:
            te = build_dropI(site, t)
            X = te[[f for f in feats if f in te.columns]]
            auc = roc_auc_score(te["label"], m.predict_proba(X)[:,1])
            gap = auc - intern[t]
            tm = pd.read_csv(os.path.join(OUT,"transfer_matrix.csv"))
            rr = tm[(tm["task"]==t)&(tm["src"]=="MGB")&(tm["tgt"]==site)]
            prim = f"{rr['auroc'].iloc[0]:.3f}" if len(rr) else "NA"
            p(f"  {t} MGB\u2192{site[:1]}: drop-I AUROC {auc:.3f} (gap {gap:+.3f}) | primary AUROC {prim}")
    rep.close()

if __name__ == "__main__":
    run()
    print("Done", flush=True)
