# -*- coding: utf-8 -*-
"""查证: PA 行 culture_description 缺失率 vs 基座 specimen 分布"""
import pandas as pd

M = r"E:\ARMD\ARMD-MGB"
import sys
sys.path.insert(0, r"E:\ARMD\scripts")
from step9_clean import norm

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
         "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]
mc = pd.read_csv(M + r"\microbiology_cohort_deid_tj_updated.csv",
                 usecols=["organism","antibiotic","CLSI_2022_pheno","culture_description",
                          "prelim_AST","neg_cx","order_proc_id_coded"],
                 low_memory=False, encoding="utf-8-sig")
org = mc["organism"].astype(str).str.upper()
pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
task = norm(mc["antibiotic"]).isin(TASKS)
neg = mc["neg_cx"].astype(str).str.strip().eq("X")
prel = mc["prelim_AST"].astype(str).str.strip().eq("X")
base = pa & task & ~neg & ~prel
ph = mc["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
valid = ph.isin(["RESISTANT","SUSCEPTIBLE","INTERMEDIATE"])

print("culture_description 原始类型与缺失:")
print("  dtype:", mc["culture_description"].dtype)
print("  全表非缺失:", mc["culture_description"].notna().mean())
print("  PA 基线行非缺失:", mc.loc[base, "culture_description"].notna().mean())
print("  保留行非缺失:", mc.loc[base & valid, "culture_description"].notna().mean())
print("  value_counts 前10 (PA 基线行):")
print(mc.loc[base, "culture_description"].value_counts(dropna=False).head(10).to_string())
print("\n对比: 全表 value_counts 前6:")
print(mc["culture_description"].value_counts(dropna=False).head(6).to_string())
