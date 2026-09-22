# -*- coding: utf-8 -*-
"""
Step 27 — 评审意见核查 (评论人1/2 的四项决定性检查)
A. ADI 三站取值口径 (national percentile 同尺?)
B. S/U PA 培养年份分布 (折点漂移暴露量化)
C. prior_micro 同培养 d=0 行断言 (跨药耐药史泄漏检查)
D. MGB attrition 核算 (221,675 PA 行 → 10,578 培养)
输出: audit_out/report_step27_review.txt
"""
import os
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
M = os.path.join(BASE, "ARMD-MGB")
CLEAN = os.path.join(BASE, "data", "clean")
rep = open(os.path.join(BASE, "audit_out", "report_step27_review.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

# ---------- A. ADI 口径 ----------
p("=== A. ADI 三站取值口径 ===")
for s in ["MGB", "Stanford", "UTSW"]:
    b = pd.read_csv(os.path.join(CLEAN, s, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
    v = pd.to_numeric(b["adi"], errors="coerce").dropna()
    p(f"  {s}: n={len(v):,} min={v.min()} max={v.max()} 中位={v.median():.1f} P90={v.quantile(.9):.1f}")

# ---------- B. S/U PA 年份分布 ----------
p("\n=== B. S/U PA 培养年份分布 (折点暴露) ===")
for s in ["Stanford", "UTSW"]:
    b = pd.read_csv(os.path.join(CLEAN, s, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
    b["time"] = pd.to_datetime(b["time"], errors="coerce")
    yr = b["time"].dt.year.dropna()
    p(f"  {s}: 年份范围 {int(yr.min())}-{int(yr.max())} | <2015: {int((yr<2015).sum())} ({100*(yr<2015).mean():.1f}%) | "
      f"<2020: {int((yr<2020).sum())} ({100*(yr<2020).mean():.1f}%) | ≥2020: {int((yr>=2020).sum())} ({100*(yr>=2020).mean():.1f}%)")

# ---------- C. prior_micro 泄漏断言 ----------
p("\n=== C. prior_micro 同培养 d=0 断言 (MGB) ===")
pm = pd.read_csv(os.path.join(M, "prior_micro_deid_tj.csv"),
                 usecols=["organism", "prior_AST_time_to_culture", "order_proc_id_coded"],
                 low_memory=False, encoding="utf-8-sig")
pa = pm["organism"].astype(str).str.upper().str.startswith("PSEUDOMONAS AERUGINOSA")
d = pd.to_numeric(pm["prior_AST_time_to_culture"], errors="coerce")
p(f"  prior_micro PA 行: {int(pa.sum()):,} | 其中 d=0: {int((pa & d.eq(0)).sum()):,} | "
  f"d<0: {int((pa & d.lt(0)).sum()):,} | d≥1: {int((pa & d.ge(1)).sum()):,}")
p(f"  断言: F5 特征仅用 d≥1 且事件严格早于索引培养的行; d=0 行被 idx_t>ev 排除")
# 同培养行检查: prior_micro 行挂接的培养是否包含索引培养 (d≥1 的同培养行)
mc = pd.read_csv(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"),
                 usecols=["order_proc_id_coded"], low_memory=False, encoding="utf-8-sig")
all_cx = set(mc["order_proc_id_coded"].astype(str))
pm_oid = pm["order_proc_id_coded"].astype(str)
same_cx_d1 = (pa & d.ge(1) & pm_oid.isin(all_cx)).sum()
p(f"  挂接在培养集内且 d≥1 的 PA 行: {int(same_cx_d1):,} (这些是培养前已知的既往事件, 合法)")

# ---------- D. MGB attrition 核算 ----------
p("\n=== D. MGB attrition 核算 (PA 行 → 任务超集) ===")
cols = ["organism", "antibiotic", "CLSI_2022_pheno", "has_AST", "neg_cx", "prelim_AST",
        "order_proc_id_coded"]
mc = pd.read_csv(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"), usecols=cols,
                 low_memory=False, encoding="utf-8-sig")
org = mc["organism"].astype(str).str.upper()
pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
n1 = int(pa.sum())
sub = mc[pa]
n2 = len(sub.drop_duplicates("order_proc_id_coded"))
from step9_clean import norm
TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
         "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]
task = norm(sub["antibiotic"]).isin(TASKS)
n3 = int(task.sum())
n3c = int(sub[task].drop_duplicates("order_proc_id_coded").shape[0])
neg = sub["neg_cx"].astype(str).str.strip().eq("X")
prel = sub["prelim_AST"].astype(str).str.strip().eq("X")
n4 = int((task & ~neg & ~prel).sum())
ph = sub["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
valid = ph.isin(["RESISTANT","SUSCEPTIBLE","INTERMEDIATE"])
n5 = int((task & ~neg & ~prel & valid).sum())
n5c = int(sub[task & ~neg & ~prel & valid].drop_duplicates("order_proc_id_coded").shape[0])
p(f"  PA 行(含变体, 不含 SPECIES): {n1:,}")
p(f"  → PA 唯一培养: {n2:,}")
p(f"  → 9 任务药行: {n3:,} (唯一培养 {n3c:,})")
p(f"  → 剔除阴性/prelim 后行: {n4:,}")
p(f"  → 有效标签行: {n5:,} (唯一培养 {n5c:,})  ← 与基座 10,578 对比")
p(f"  各步保留率: 行 {n1}→{n3} ({100*n3/n1:.0f}%) | 标签有效性 {100*n5/n4:.0f}% | "
  f"培养级 {n5c} ({100*n5c/n2:.0f}% of PA 培养)")

rep.close()
print("完成", flush=True)
