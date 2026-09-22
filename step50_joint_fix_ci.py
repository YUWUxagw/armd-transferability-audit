# -*- coding: utf-8 -*-
"""
Step 50 — patient-level bootstrap CI for the fully-fixed (joint) cohorts,
all 10 primary directions. Same encoding semantics as step48 (source-fitted
maps, fillna(-1) for out-of-source categories). 200 iterations.
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_joint_fix_ci.txt"), "w", encoding="utf-8")
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


INTERNAL = _load_internal()def build(site, task, drop_I=True, drop_mucoid=True, min_year=2020):
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

p("Joint-fix patient-level CIs (drop-I + mucoid removal both sides + test >=2020)")
p("=" * 90)
rows = []
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw)
    adi_med = mgb_raw["adi"].median()
    tr = encode(build("MGB", t), maps, adi_med)
    feats = feature_cols(tr)
    m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
    for site in ["Stanford", "UTSW"]:
        te = encode(build(site, t), maps, adi_med)
        X = te[[f for f in feats if f in te.columns]]
        y = te["label"].values
        pred = m.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)
        lo, hi = patient_boot(y, pred, te["anon_id"].values)
        gap = auc - INTERNAL[t]
        rows.append(dict(task=t, tgt=site, n=len(te), auroc=round(auc,3), gap=round(gap,3),
                         ci_lo=round(lo,3) if lo else None, ci_hi=round(hi,3) if hi else None))
        p(f"  {t} MGB\u2192{site[:1]}: AUROC {auc:.3f} gap {gap:+.3f} | patient-CI [{lo:.3f}-{hi:.3f}] (n={len(te):,})")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "joint_fix_ci.csv"), index=False)
p("\nsaved: joint_fix_ci.csv")
rep.close()
print("Done", flush=True)
