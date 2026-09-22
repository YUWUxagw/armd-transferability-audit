# -*- coding: utf-8 -*-
"""复核: 剔除行 vs 保留行的 mucoid 与标本分布 (step32 发现稳健性)"""
import pandas as pd

M = r"E:\ARMD\ARMD-MGB"
import sys
sys.path.insert(0, r"E:\ARMD\scripts")
from step9_clean import norm

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
         "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]
mc = pd.read_csv(M + r"\microbiology_cohort_deid_tj_updated.csv",
                 usecols=["organism","antibiotic","CLSI_2022_pheno","culture_description",
                          "prelim_AST","neg_cx","order_proc_id_coded"], low_memory=False, encoding="utf-8-sig")
org = mc["organism"].astype(str).str.upper()
pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
task = norm(mc["antibiotic"]).isin(TASKS)
neg = mc["neg_cx"].astype(str).str.strip().eq("X")
prel = mc["prelim_AST"].astype(str).str.strip().eq("X")
base = pa & task & ~neg & ~prel
ph = mc["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
valid = ph.isin(["RESISTANT","SUSCEPTIBLE","INTERMEDIATE"])
keep, drop = base & valid, base & ~valid

print("mucoid 复核 (培养级去重后):")
for label, m in [("保留", keep), ("剔除", drop)]:
    # 培养级: 每培养的菌种名 (取该培养任一 PA 行)
    sub = mc[m]
    cx_mucoid = sub[sub["organism"].astype(str).str.contains("MUCOID")]["order_proc_id_coded"]
    n_cx = sub["order_proc_id_coded"].nunique()
    print(f"  [{label}] 行 {int(m.sum()):,} | 培养 {n_cx:,} | mucoid 行 {len(cx_mucoid):,} "
          f"({100*len(cx_mucoid)/max(int(m.sum()),1):.1f}%) | mucoid 培养 {cx_mucoid.nunique():,}")

print("\n标本分布复核 (行级, 含 nan):")
for label, m in [("保留", keep), ("剔除", drop)]:
    vc = mc.loc[m, "culture_description"].value_counts(dropna=False)
    print(f"  [{label}] " + " | ".join(f"{v}:{100*c/len(m):.0f}%" for v, c in vc.head(5).items()))
