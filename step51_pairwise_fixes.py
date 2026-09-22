# -*- coding: utf-8 -*-
"""
Step 51 — pairwise cross-validation of the three fixes.
Checks overlap structure: does the combined effect exceed/shortfall the sum
of isolated effects, and how do pairwise combinations compare?
  R1 = drop-I (labels)
  R2 = mucoid removal (both sides)
  R3 = era >=2020 (test side)
Combinations run: R1R2, R1R3, R2R3, R1R2R3 (joint), and none (primary).
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_pairwise_fixes.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols
from sklearn.metrics import roc_auc_score

TASKS  = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
COHORT = {"MGB": os.path.join(BASE,"ARMD-MGB","microbiology_cohort_deid_tj_updated.csv"),
          "Stanford": os.path.join(BASE,"ARMD-Stanford","microbiology_cultures_cohort.csv"),
          "UTSW": os.path.join(BASE,"ARMD-UTSW","microbiology_cultures_cohort.csv")}
PHENO  = {"MGB":"CLSI_2022_pheno","Stanford":"susceptibility","UTSW":"susceptibility"}
POSCOL = {"MGB":None,"Stanford":"was_positive","UTSW":"was_positive"}
NEGCOL = {"MGB":"neg_cx","Stanford":None,"UTSW":None}
PRELIM = {"MGB":"prelim_AST","Stanford":None,"UTSW":None}
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


INTERNAL = _load_internal()def build(site, task, drop_I=False, drop_mucoid=False, min_year=None):
    cols = ["anon_id","order_proc_id_coded","organism","antibiotic",PHENO[site]]
    if POSCOL[site]: cols.append(POSCOL[site])
    if NEGCOL[site]: cols.append(NEGCOL[site])
    if PRELIM[site]: cols.append(PRELIM[site])
    df = pd.read_csv(COHORT[site], usecols=cols, low_memory=False, encoding="utf-8-sig")
    org = df["organism"].astype(str).str.upper()
    pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
    drug = (df["antibiotic"].astype(str).str.lower().str.replace("/","_",regex=False)
            .str.replace("-","_",regex=False).str.strip())
    m = pa & drug.eq(task)
    if NEGCOL[site]: m &= ~df[NEGCOL[site]].astype(str).str.strip().eq("X")
    if PRELIM[site]: m &= ~df[PRELIM[site]].astype(str).str.strip().eq("X")
    if POSCOL[site]: m &= df[POSCOL[site]].astype(str).str.strip().isin(["1","1.0"])
    sub = df[m].copy()
    ph = sub[PHENO[site]].astype(str).str.upper().str.strip()
    S = ph.eq("SUSCEPTIBLE"); R = ph.eq("RESISTANT"); I = ph.eq("INTERMEDIATE")
    keep = (S | R) if drop_I else (S | R | I)
    sub = sub[keep].copy()
    lab = R.astype(int)
    if not drop_I: lab = (R | I).astype(int)
    sub["label"] = lab.astype(int)
    sub = sub.sort_values("label", ascending=False).drop_duplicates("order_proc_id_coded")
    if drop_mucoid:
        sub = sub[~sub["organism"].astype(str).str.contains("MUCOID")]
    base = pd.read_csv(os.path.join(CLEAN, site, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
    out = sub[["order_proc_id_coded","anon_id","label"]].merge(
        base.drop(columns=["anon_id"]), on="order_proc_id_coded", how="left")
    # 规范特征集对齐: 同药物既往耐药特征 prior_pa_res_{task} 在规范管线
    # (task 文件) 中不存在 (同药物既往耐药为泄漏控制排除); base 含此列,
    # 必须剔除, 否则 feature_cols() 数据驱动会把该强预测特征带入模型
    out = out.drop(columns=[f"prior_pa_res_{task}"], errors="ignore")
    if min_year and site != "MGB":
        # MGB 无日历年份 (日期移位, 不可恢复), 年代过滤只对 S/U 生效
        byr = pd.to_datetime(base["time"], errors="coerce").dt.year
        yr = base[["order_proc_id_coded"]].assign(year=byr).dropna()
        out = out.merge(yr, on="order_proc_id_coded", how="left")
        out = out[out["year"] >= min_year]
    return out

def encode(df, maps, adi_med):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years","",regex=False)
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
    df["adi"] = df["adi"].fillna(adi_med)
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df

COMBOS = {
    "primary":      dict(drop_I=False, drop_mucoid=False, min_year=None),
    "R1_only":      dict(drop_I=True,  drop_mucoid=False, min_year=None),
    "R2_only":      dict(drop_I=False, drop_mucoid=True,  min_year=None),
    "R3_only":      dict(drop_I=False, drop_mucoid=False, min_year=2020),
    "R1R2":         dict(drop_I=True,  drop_mucoid=True,  min_year=None),
    "R1R3":         dict(drop_I=True,  drop_mucoid=False, min_year=2020),
    "R2R3":         dict(drop_I=False, drop_mucoid=True,  min_year=2020),
    "R1R2R3":       dict(drop_I=True,  drop_mucoid=True,  min_year=2020),
}

p("Pairwise cross-validation of the three fixes (gap relative to primary internal)")
p("=" * 100)
ledger = []
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw)
    adi_med = mgb_raw["adi"].median()
    for site in ["Stanford", "UTSW"]:
        row = dict(task=t, tgt=site)
        for cname, kw in COMBOS.items():
            tr = encode(build("MGB", t, **kw), maps, adi_med)
            feats = feature_cols(tr)
            m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
            te = encode(build(site, t, **kw), maps, adi_med)
            X = te[[f for f in feats if f in te.columns]]
            auc = roc_auc_score(te["label"], m.predict_proba(X)[:, 1])
            row[cname + "_gap"] = round(auc - INTERNAL[t], 3)
        ledger.append(row)
        p(f"  {t} MGB\u2192{site[:1]} | primary {row['primary_gap']:+.3f} | R1 {row['R1_only_gap']:+.3f} "
          f"R2 {row['R2_only_gap']:+.3f} R3 {row['R3_only_gap']:+.3f} | "
          f"R1R2 {row['R1R2_gap']:+.3f} R1R3 {row['R1R3_gap']:+.3f} R2R3 {row['R2R3_gap']:+.3f} "
          f"| joint {row['R1R2R3_gap']:+.3f}")

pd.DataFrame(ledger).to_csv(os.path.join(OUT, "pairwise_fixes.csv"), index=False)
p("\nsaved: pairwise_fixes.csv")
rep.close()
print("Done", flush=True)
