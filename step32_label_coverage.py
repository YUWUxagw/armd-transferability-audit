# -*- coding: utf-8 -*-
"""
Step 32 — 标注覆盖度不对称刻画 (评论人1 第三点, 标签口径第二轴)
MGB 被剔除(无有效 CLSI 表型) vs 保留行: 年份/标本/panel/变体 系统差异
回答: MGB 标注人群(可重推子集)与保留行是否代表性不同
"""
import os
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
M = os.path.join(BASE, "ARMD-MGB")
rep = open(os.path.join(BASE, "audit_out", "report_step32_label_coverage.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import sys
sys.path.insert(0, os.path.dirname(__file__))
from step9_clean import norm

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
         "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]

mc = pd.read_csv(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"),
                 usecols=["organism","antibiotic","CLSI_2022_pheno","culture_description",
                          "prelim_AST","neg_cx","order_time_jittered_utc_shifted"],
                 low_memory=False, encoding="utf-8-sig")
org = mc["organism"].astype(str).str.upper()
pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
task = norm(mc["antibiotic"]).isin(TASKS)
neg = mc["neg_cx"].astype(str).str.strip().eq("X")
prel = mc["prelim_AST"].astype(str).str.strip().eq("X")
base_rows = pa & task & ~neg & ~prel
ph = mc["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
valid = ph.isin(["RESISTANT","SUSCEPTIBLE","INTERMEDIATE"])
keep = base_rows & valid
drop = base_rows & ~valid
p(f"保留行: {int(keep.sum()):,} | 剔除行(无有效表型): {int(drop.sum()):,} "
  f"({100*drop.sum()/base_rows.sum():.1f}% of 基线行)")

# 年份 (剔除行无表型, 但时间可用)
mc["year"] = pd.to_datetime(mc["order_time_jittered_utc_shifted"], errors="coerce").dt.year
for label, m in [("保留", keep), ("剔除", drop)]:
    yr = mc.loc[m, "year"].dropna()
    p(f"\n[{label}] n={int(m.sum()):,}")
    p(f"  年份: 中位 {yr.median():.0f} (P25 {yr.quantile(.25):.0f}, P75 {yr.quantile(.75):.0f})")
    spec = mc.loc[m, "culture_description"].astype(str)
    p(f"  标本: " + " | ".join(f"{v}:{100*c/len(m):.0f}%" for v, c in spec.value_counts().head(4).items()))
    abx = mc.loc[m, "antibiotic"].astype(str)
    p(f"  药 top8: " + " | ".join(f"{v}:{c}" for v, c in abx.value_counts().head(8).items()))
    muc = org[m].str.contains("MUCOID")
    p(f"  mucoid 占比: {100*muc.mean():.1f}%")

# 药物×保留率: 哪些药更可能无表型
p("\n各任务药保留率 (有表型/基线行):")
for t in TASKS:
    tm = task & norm(mc["antibiotic"]).eq(t)
    base_t = pa & tm & ~neg & ~prel
    keep_t = base_t & valid
    p(f"  {t}: 保留 {100*keep_t.sum()/max(base_t.sum(),1):.0f}% (n={int(base_t.sum()):,})")
rep.close()
print("完成", flush=True)
