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
CLEAN = os.path.join(BASE, "clean", "clean")
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

def build_dropI(site, task, maps, adi_med):
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
    # encode categoricals with MGB-fitted maps (consistent with main pipeline).
    # 2026-10-08 修复：原先在此用 **目标站点自己** 的 task 文件拟合 maps 并用自己的
    # adi 中位数填补，与注释和 CHECKLIST.md 的契约（同源编码 source-maps）相反，
    # 会让外部站点的 ward_h/specimen/age_bin 按各自字母序编号，被 MGB 训练的模型
    # 读到错位编码。现改为由调用方传入 MGB 拟合的 maps/adi，并在此显式校验。
    for _c, _m in maps.items():
        _vals = set(out[_c].astype(str).unique())
        _miss = sorted(v for v in _vals if v not in _m)
        if _miss:
            raise ValueError(
                f"编码错位: {site} 的 {_c} 有 {len(_miss)} 个类别不在 MGB 码本中: "
                f"{_miss[:5]} —— 跨站编码不可用，需先对齐码本")
    out = prep_cross(out, maps, adi_med)
    return out

def _load_internal():
    """内参运行时从权威结果文件读取，禁止字面量（2026-10-08 修复，与 step52 同口径）。

    原字面量 {0.795,0.765,0.764,0.804,0.790} 是 early-stopping 泄漏修复前的膨胀值，
    与 transfer_matrix.csv 的 internal_auroc 不符，会让本脚本打印的对照值误导读者。
    """
    _tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))
    _out = {}
    for _t in TASKS:
        _v = _tm[_tm["task"] == _t]["internal_auroc"].unique()
        assert len(_v) == 1, f"internal_auroc not unique for {_t}"
        _out[_t] = float(_v[0])
    return _out


def run():
    # Internal CV on MGB (drop-I cohort)
    p("I-exclusion sensitivity (labels: Intermediate excluded on all sites)")
    p("=" * 90)
    INTERNAL = _load_internal()
    intern = {}
    for t in TASKS:
        _ref = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"),
                           low_memory=False, encoding="utf-8-sig")
        _maps = fit_cat_maps(_ref); _adi = _ref["adi"].median()
        mgb = build_dropI("MGB", t, _maps, _adi)
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
          f"(primary {INTERNAL[t]:.3f})")
    p("-" * 90)
    # Cross-system: MGB train -> S/U test (drop-I on both sides)
    for t in TASKS:
        _ref = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"),
                           low_memory=False, encoding="utf-8-sig")
        maps = fit_cat_maps(_ref); adi_med = _ref["adi"].median()   # 两侧同源，MGB 拟合
        mgb = build_dropI("MGB", t, maps, adi_med)
        feats = feature_cols(mgb)
        m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
        for site in ["Stanford","UTSW"]:
            te = build_dropI(site, t, maps, adi_med)
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
