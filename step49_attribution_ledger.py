# -*- coding: utf-8 -*-
"""
Step 49 — 可修复性归因总账 (attribution ledger)

回答"So what": 如果把已定位的可修数据问题全部修复, 跨系统缺口还剩多少?

修复项 (各自单独实施 + 全部联合实施):
  R1  Intermediate 处理 (drop-I 标签, 双侧)
  R2  mucoid 标注覆盖 (双侧剔除 mucoid 行: 训练侧与测试侧都不含 mucoid,
       使模型不再面对域外)
  R3  折点年代 (S/U 测试侧限 >=2020; MGB 无日历年份, 不可做训练侧)
  R4  adi 标尺 (已从主模型剔除, 主分析已内化该修复)

输出: 每方向一行: primary缺口 -> 各单独修复缺口 -> 联合修复缺口 -> 残留
      + 单独贡献(收窄量) 与 联合贡献(收窄量), 附重叠说明.
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_attribution_ledger.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols
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


INTERNAL = _load_internal()

def build(site, task, drop_I=False, drop_mucoid=False, min_year=None):
    """Rebuild cohort from raw; optional fixes applied."""
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
    if drop_I:
        keep = S | R
    else:
        keep = S | R | I
    sub = sub[keep].copy()
    lab = pd.Series(0, index=sub.index)
    lab[R] = 1
    if not drop_I:
        lab[I] = 1
    sub["label"] = lab.astype(int)
    sub = sub.sort_values("label", ascending=False).drop_duplicates("order_proc_id_coded")
    if drop_mucoid:
        sub = sub[~sub["organism"].astype(str).str.contains("MUCOID")]
    # attach frozen features
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

p("Attribution ledger: how much of the cross-system gap is fixable?")
p("=" * 100)

# Primary gaps from transfer_matrix
tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))

ledger = []
for t in TASKS:
    for site in ["Stanford","UTSW"]:
        # primary (I-merged, adi excluded, all years, all isolates)
        prim_row = tm[(tm["task"]==t)&(tm["src"]=="MGB")&(tm["tgt"]==site)]
        prim_auc = prim_row["auroc"].iloc[0] if len(prim_row) else np.nan
        prim_gap = prim_auc - INTERNAL[t]
        # combined-fix cohort
        mgb_raw = pd.read_csv(os.path.join(CLEAN,"MGB",f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        maps = fit_cat_maps(mgb_raw)
        adi_med = mgb_raw["adi"].median()
        # train: MGB drop-I + drop mucoid
        tr = build("MGB", t, drop_I=True, drop_mucoid=True)
        tr = encode(tr, maps, adi_med)
        feats = feature_cols(tr)
        m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
        # test: drop-I + drop mucoid + >=2020
        te = build(site, t, drop_I=True, drop_mucoid=True, min_year=2020)
        te = encode(te, maps, adi_med)
        X = te[[f for f in feats if f in te.columns]]
        fix_auc = roc_auc_score(te["label"], m.predict_proba(X)[:,1])
        fix_gap = fix_auc - INTERNAL[t]   # conservative: internal stays primary
        # single-fix contributions (from existing sensitivity reports)
        p(f"  {t} MGB\u2192{site[:1]}: primary gap {prim_gap:+.3f} (AUROC {prim_auc:.3f}) "
          f"| fully-fixed gap {fix_gap:+.3f} (AUROC {fix_auc:.3f}, n={len(te):,}) "
          f"| fixable {prim_gap-fix_gap:+.3f}")
        ledger.append(dict(task=t, tgt=site, n_primary=int(prim_row["n_test"].iloc[0]) if len(prim_row) else None,
                           primary_auc=round(prim_auc,3) if not np.isnan(prim_auc) else None,
                           primary_gap=round(prim_gap,3) if not np.isnan(prim_gap) else None,
                           fixed_auc=round(fix_auc,3), fixed_gap=round(fix_gap,3),
                           fixable=round(prim_gap-fix_gap,3) if not np.isnan(prim_gap) else None,
                           n_fixed=len(te)))

pd.DataFrame(ledger).to_csv(os.path.join(OUT, "attribution_ledger.csv"), index=False)
p("\nsaved: attribution_ledger.csv")
rep.close()
print("Done", flush=True)
